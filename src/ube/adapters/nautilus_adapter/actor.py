"""UbeActor — the Nautilus-specific trading piece (plan §4.5, requirements §4.5/§4.6/§6).

The Actor translates canonical 4-column signals plus :class:`RiskConfig.exit` levels into
native Nautilus order flow. It is a :class:`~nautilus_trader.trading.strategy.Strategy`
(in this version ``Strategy`` *is* ``Actor`` and is the only class exposing the trading
surface). All exit levels are **computed with the ``core`` pure functions** and injected
into Nautilus's native mechanism — no strategy re-implementation (plan §4.5).

Behavior summary (each item maps to a plan §4.5 bullet):

* **Entries** — on a bar whose signals demand a position, size via
  :func:`~ube.core.risk.sizing.size_position` against the venue account balance and the
  bar's close, submit a market order (fills at the same-bar close — the entry is
  evaluated at bar close, plan §4.4/§5.2).
* **Signal exits / flips** — ``long_exit`` / ``short_exit`` close the open position with a
  reduce-only market order; a same-bar flip closes + opens (two orders, the netting venue
  reconciles at the bar close). ``short_entry`` while long (and the mirror) is treated as
  an implicit flip; exits with no matching position are ignored (accumulate semantics).
* **Exit book** — while a position is open, per-bar levels are precomputed at entry with
  :func:`~ube.core.risk.exits.scale_out_plan` / :func:`~ube.core.risk.exits.exit_level`
  anchored at the entry bar, then injected:
    - ``trigger="touched"`` → native orders (``stop_market`` / ``limit``), so Nautilus
      performs the intra-bar high/low contact matching. Missing: ``trailing_stop_market``
      (a native trailing stop with a *price* offset does not reproduce the canonical
      percentage / ATR trailing levels — a plain stop whose trigger is ratcheted each bar
      does).
    - ``trigger="close"`` (and :class:`~ube.core.risk.exits.TimeExit`) → evaluated at bar
      close by the Actor, which submits a reduce-only market close.
* **Stale-risk-order lifecycle** — whenever a position closes or a new one opens, all
  working risk orders are cancelled (reference ``_cancel_live_risk_orders``), so a stale
  TP cannot fill against a later position. On every bar while open, working exit
  orders are re-synchronised to the remaining position quantity (scale-outs) and native
  stop triggers are moved to the next bar's precomputed level.
* **exit_reason stamping** — the reason (``signal``, ``take_profit``, ``atr_stop``,
  ``trailing_stop``, ``chandelier``, ``time_exit``) is recorded per closing fill in
  :attr:`exit_reasons` (keyed by client-order id) so the adapter can stamp the ledger's
  closing fill (§4.6). :attr:`position_reasons` mirrors the reference netting pattern.

Trigger timing notes (probe-verified against Nautilus 1.221.0 backtest matching):

* Orders submitted during ``on_bar(bar_i)`` become active for bar ``i+1`` matching; so a
  touched stop is submitted at ``level[entry_bar + 1]`` and re-targeted to ``level[i+1]``
  during each subsequent ``on_bar(bar_i)``. Because the canonical trailing/chandelier
  level for bar ``j`` is defined on the completed-bar running extreme through ``j``, the
  precomputed ``level[i+1]`` reproduces :func:`~ube.core.risk.exits.exit_triggered`
  exactly (no net lookahead — the *value* is the one the reference per-bar rule uses for
  bar ``i+1``).
* Touched exits trigger before the bar's close-time work (§8: intra-bar precedes
  close-time). On the entry bar the position opens at the close, so the entry-bar trigger
  row is ignored (a stop is live from the next bar).

Documented divergences (parity note): Nautilus's intra-bar matching is *target-first* —
on a bar that crosses both a stop and a target, the LIMIT fills before the STOP, whereas
§8 prescribes stop-before-target. No BacktestVenueConfig flag flips this (probe-proven:
``bar_adaptive_high_low_ordering`` and ``reject_stop_orders`` in all four combinations
still filled the target first). The Actor therefore implements §8 precedence
deterministically only where it *can*: close-time risk exits are evaluated *before*
close-time signal actions, and the same-bar stop/target *quantity* collision that would
over-fill a scale-out is bounded by re-syncing exit quantities each bar. The residual
same-bar stop-vs-target collision with native orders fills target-first; this is recorded
as a known divergence for the parity report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.events import OrderFilled, OrderRejected
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from ube.core.data import MarketData
from ube.core.errors import EngineError
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    TakeProfit,
    TimeExit,
    TrailingStop,
    exit_level,
    scale_out_plan,
)
from ube.core.risk.sizing import SizeModel, size_position

__all__ = [
    "UbeActor",
    "UbeActorConfig",
]

#: The 4-column signal row for "no signal on this bar".
_NO_SIGNAL: tuple[bool, bool, bool, bool] = (False, False, False, False)

#: `RiskConfig.exit` type -> §4.6 exit_reason string.
_EXIT_REASONS: dict[type, str] = {
    TakeProfit: "take_profit",
    ATRStop: "atr_stop",
    TrailingStop: "trailing_stop",
    ChandelierExit: "chandelier",
    TimeExit: "time_exit",
}


class UbeActorConfig(StrategyConfig):  # type: ignore[misc]
    """Actor configuration — only msgspec-serializable scalars (plan §4.5.3).

    The heavyweight runtime inputs (``market_data``, ``atr_series``, ``sizing``,
    ``exits``) are passed as constructor kwargs so Nautilus never serializes them.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    signal_map: dict[int, tuple[bool, bool, bool, bool]]

    @property
    def signals(self) -> dict[int, tuple[bool, bool, bool, bool]]:
        """The per-bar 4-column signal lookup (convenience alias)."""
        return self.signal_map


@dataclass
class _LiveExit:
    """One working native exit order (a ``touched`` stop or TP limit)."""

    order: Any
    reason: str
    fraction: float
    levels: np.ndarray | None = None


@dataclass
class _Book:
    """The exit book anchored at one position's entry (plan §4.5 item 3)."""

    side: int  # +1 long, -1 short
    entry_bar: int
    entry_price: float
    created_bar: int
    live: dict[str, _LiveExit]  # client_order_id -> exit order
    close_exits: list[tuple[str, float, np.ndarray]]  # (reason, fraction, triggered mask)


class UbeActor(Strategy):  # type: ignore[misc]
    """Translates canonical signals + risk exits into native Nautilus orders."""

    def __init__(
        self,
        config: UbeActorConfig,
        *,
        market_data: MarketData,
        atr_series: np.ndarray | None = None,
        sizing: SizeModel | None = None,
        exits: tuple[Exit, ...] = (),
        vol: np.ndarray | None = None,
    ) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self._market_data = market_data
        self._atr_series = atr_series
        self._sizing = sizing if sizing is not None else SizeModel()
        self._exits = exits
        self._vol = vol

        ts_ns = market_data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
        self._ts_to_index: dict[int, int] = {
            int(ts_ns[i]): i for i in range(market_data.n_bars)
        }
        self._bar_count = 0
        self._instrument: Any = None
        self._book: _Book | None = None
        self._fill_reasons: dict[str, str] = {}

        #: Closing-fill client_order_id -> exit_reason (§4.6); read by the adapter.
        self.exit_reasons: dict[str, str] = {}
        #: position_id -> last closing exit_reason (reference netting pattern).
        self.position_reasons: dict[str, str] = {}
        #: bar index -> action string (dropdown for the ledger's signal_evaluated rows).
        self.signal_evaluated: dict[int, str] = {}
        #: Set to a message if a market order was rejected (insufficient funds, §4.5).
        self.market_rejection: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.instrument_id)
        if self._instrument is None:
            raise EngineError(f"instrument {self.instrument_id} is not registered")
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self.market_rejection is not None:
            raise EngineError(f"market order rejected by the venue: {self.market_rejection}")
        idx = self._index_for(bar)
        self._bar_count += 1

        # A touched exit may have closed the position intra-bar; clean stale siblings
        # so they cannot fill against a later position (§4.5 item 4).
        if self._book is not None and self._open_position() is None:
            self._cancel_live_exits()
            self._book = None

        if self._book is not None and self._open_position() is not None:
            fired = self._close_exit_fired(idx)
            if fired is not None:
                reason, fraction = fired
                self._submit_close(reason, fraction)
                self.signal_evaluated[idx] = f"exit_{reason}"
            else:
                self._apply_signal(self._lookup_signal(bar), bar, idx)
        else:
            self._apply_signal(self._lookup_signal(bar), bar, idx)

        self._maintain_exits(bar, idx)

    def on_order_filled(self, event: OrderFilled) -> None:
        reason = self._fill_reasons.get(str(event.client_order_id))
        if reason is None:
            return
        # Every exit-ordered fill is stamped (§4.6): a scale-out partial and the
        # final closing fill both carry the reason. The fold's last-reason-wins rule
        # keeps the round-trip attribution on the closing fill (ledger._process_fill).
        self.exit_reasons[str(event.client_order_id)] = reason
        if event.position_id is not None:
            self.position_reasons[str(event.position_id)] = reason

    def on_order_rejected(self, event: OrderRejected) -> None:
        order = self.cache.order(client_order_id=event.client_order_id)
        if order is not None and order.order_type == OrderType.MARKET:
            detail = getattr(event, "reason", None)
            self.market_rejection = (
                f"{str(event.client_order_id)}: {detail if detail else 'insufficient funds'}"
                f" — market order rejected"
            )
            self.log.error(f"UbeActor market-order rejection: {self.market_rejection}")
        else:
            self.log.warning(
                f"UbeActor non-market order rejected (tolerated): {event.client_order_id}"
            )

    # -- entries + signal exits (plan §4.5 items 1-2) ------------------------

    def _apply_signal(self, signals: tuple[bool, bool, bool, bool], bar: Bar, idx: int) -> None:
        le, lx, se, sx = signals
        position = self._open_position()
        action = "hold"

        if position is not None and position.is_long:
            if lx or se:
                self._submit_close("signal")
                action = "close_long"
                if se:
                    self._open(-1, bar, idx)
                    action = "flip_short"
        elif position is not None and position.is_short:
            if sx or le:
                self._submit_close("signal")
                action = "close_short"
                if le:
                    self._open(1, bar, idx)
                    action = "flip_long"
        else:
            if le:
                self._open(1, bar, idx)
                action = "buy"
            elif se:
                self._open(-1, bar, idx)
                action = "sell"

        self.signal_evaluated[idx] = action

    def _open(self, side: int, bar: Bar, idx: int) -> None:
        price = float(bar.close)
        qty = self._entry_quantity(price)
        if qty is None:
            self.log.error(
                f"UbeActor entry skipped at bar {idx}: sizing produced no tradable lot "
                f"(price={price})"
            )
            return
        order = self.order_factory.market(
            self.instrument_id,
            OrderSide.BUY if side > 0 else OrderSide.SELL,
            qty,
        )
        self.submit_order(order)
        if self._exits:
            self._setup_book(side=side, entry_bar=idx, entry_price=price, size=qty)

    def _entry_quantity(self, price: float) -> Quantity | None:
        if price <= 0.0 or self._instrument is None:
            return None
        account = self.cache.account_for_venue(self.instrument_id.venue)
        balance = float(account.balance_total().as_double())
        units = np.asarray(
            size_position(
                self._sizing,
                capital=balance,
                price=price,
                n=1,
                vol=self._vol,
            ),
            dtype=np.float64,
        )
        try:
            return self._instrument.make_qty(float(units.ravel()[0]))
        except ValueError:
            return None

    def _submit_close(self, reason: str, fraction: float = 1.0) -> None:
        """Submit a reduce-only market close of ``fraction`` of the remaining position.

        A full close (``fraction >= 1.0`` — stops and time exits, §6.4) cancels the
        exit book and closes the position outright. A scale-out partial
        (``fraction < 1.0`` — a take-profit slice) closes only that slice with a
        reduce-only market order and leaves the book alive so the sibling stops
        keep protecting the remainder (§6.4).
        """
        position = self._open_position()
        if position is None:
            return
        remaining = float(position.quantity)
        qty = self._scaled_quantity_from(remaining, fraction)
        if qty is None or float(qty.as_double()) <= 0.0:
            return
        if fraction >= 1.0:
            self._cancel_live_exits()
        order = self.order_factory.market(
            self.instrument_id,
            OrderSide.SELL if position.is_long else OrderSide.BUY,
            qty,
            reduce_only=True,
        )
        self.submit_order(order)
        self._fill_reasons[str(order.client_order_id)] = reason
        if fraction >= 1.0:
            self._book = None

    # -- exit book (plan §4.5 item 3) ----------------------------------------

    def _setup_book(self, *, side: int, entry_bar: int, entry_price: float, size: Quantity) -> None:
        n = self._market_data.n_bars
        plan = scale_out_plan(
            self._exits,
            market_data=self._market_data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=self._atr_series,
        )
        book = _Book(
            side=side,
            entry_bar=entry_bar,
            entry_price=entry_price,
            created_bar=entry_bar,
            live={},
            close_exits=[],
        )
        for j, cfg in enumerate(self._exits):
            reason = _EXIT_REASONS[type(cfg)]
            fraction = float(plan.fractions[j])
            if isinstance(cfg, TimeExit) or getattr(cfg, "trigger", None) == "close":
                book.close_exits.append((reason, fraction, plan.triggered[j]))
                continue
            levels = exit_level(
                cfg,
                market_data=self._market_data,
                side=side,
                entry_price=entry_price,
                entry_bar=entry_bar,
                atr_series=self._atr_series,
            )
            exit_side = OrderSide.SELL if side > 0 else OrderSide.BUY
            if isinstance(cfg, TakeProfit):
                target = float(levels[entry_bar])
                oqty = self._scaled_quantity(size, fraction)
                if oqty is None:
                    continue
                order = self.order_factory.limit(
                    self.instrument_id,
                    exit_side,
                    oqty,
                    price=self._fmt_price(target),
                    reduce_only=True,
                )
                self.submit_order(order)
                self._fill_reasons[str(order.client_order_id)] = reason
                book.live[str(order.client_order_id)] = _LiveExit(
                    order=order, reason=reason, fraction=fraction
                )
            else:
                # ATRStop / TrailingStop / Chandelier: the trigger is the precomputed
                # level of the NEXT bar to be matched (active from entry_bar+1).
                target = (
                    float(levels[entry_bar + 1])
                    if entry_bar + 1 < n
                    else float(levels[entry_bar])
                )
                order = self.order_factory.stop_market(
                    self.instrument_id,
                    exit_side,
                    size,
                    trigger_price=self._fmt_price(target),
                    reduce_only=True,
                )
                self.submit_order(order)
                self._fill_reasons[str(order.client_order_id)] = reason
                book.live[str(order.client_order_id)] = _LiveExit(
                    order=order, reason=reason, fraction=1.0, levels=levels
                )
        self._book = book

    def _close_exit_fired(self, idx: int) -> tuple[str, float] | None:
        """§8 close-time risk exits (evaluated at the bar close, in configured order).

        Returns the first exit whose triggered mask is set on this bar as its
        ``(reason, fraction)`` and **pops** it — a fired close exit is consumed exactly
        once. The pop matters because a scale-out partial closes only a fraction of the
        position (the book stays alive) and must not re-fire on every later bar.

        Returns:
            The ``(reason, fraction)`` of the first fired close exit, or ``None``.
        """
        if self._book is None:
            return None
        n = self._market_data.n_bars
        if idx <= self._book.entry_bar:
            return None
        for i in range(len(self._book.close_exits)):
            reason, fraction, triggered = self._book.close_exits[i]
            if idx < n and bool(triggered[idx]):
                self._book.close_exits.pop(i)
                return reason, fraction
        return None

    def _maintain_exits(self, bar: Bar, idx: int) -> None:
        """Re-sync quantities to the remaining position and ratchet stop triggers.

        Skipped on the entry bar itself (the position is not in the cache until the
        same-bar close fills) — exit quantities are already correct at submission.
        """
        if self._book is None or self._book.created_bar == idx:
            return
        position = self._open_position()
        if position is None:
            return
        remaining = float(position.quantity)
        n = self._market_data.n_bars
        working = {
            str(o.client_order_id): o
            for o in self.cache.orders_open(instrument_id=self.instrument_id)
        }
        for cid, live in list(self._book.live.items()):
            order = working.get(cid)
            if order is None:
                continue
            new_qty = self._scaled_quantity_from(remaining, live.fraction)
            if new_qty is None or float(new_qty.as_double()) <= 0.0:
                self.cancel_order(order)
                self._book.live.pop(cid, None)
                continue
            trigger = None
            if live.levels is not None and idx + 1 < n:
                trigger = self._fmt_price(float(live.levels[idx + 1]))
            qty_same = float(order.quantity.as_double()) == float(new_qty.as_double())
            try:
                if trigger is not None and not qty_same:
                    self.modify_order(order, quantity=new_qty, trigger_price=trigger)
                elif trigger is not None:
                    self.modify_order(order, trigger_price=trigger)
                elif not qty_same:
                    self.modify_order(order, quantity=new_qty)
            except Exception:  # noqa: BLE001 — a rejected modify is non-fatal here.
                self.log.warning(
                    f"UbeActor failed to maintain exit {cid} (order no longer modifiable)"
                )

    # -- shared helpers -------------------------------------------------------

    def _lookup_signal(self, bar: Bar) -> tuple[bool, bool, bool, bool]:
        cfg = cast(UbeActorConfig, self.config)
        return cfg.signal_map.get(int(bar.ts_event), _NO_SIGNAL)

    def _index_for(self, bar: Bar) -> int:
        return self._ts_to_index.get(int(bar.ts_event), self._bar_count)

    def _open_position(self) -> Any | None:
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        return positions[0] if positions else None

    def _cancel_live_exits(self) -> None:
        """Cancel every still-working risk order in the book (stale-sibling rule)."""
        if self._book is None:
            return
        working = {
            str(o.client_order_id): o
            for o in self.cache.orders_open(instrument_id=self.instrument_id)
        }
        for cid, _live in list(self._book.live.items()):
            order = working.get(cid)
            if order is not None:
                try:
                    self.cancel_order(order)
                except Exception:  # noqa: BLE001 — already-completed orders just warn.
                    continue
        self._book.live.clear()

    def _scaled_quantity(self, size: Quantity, fraction: float) -> Quantity | None:
        return self._scaled_quantity_from(float(size.as_double()), fraction)

    def _scaled_quantity_from(self, remaining: float, fraction: float) -> Quantity | None:
        try:
            return self._instrument.make_qty(remaining * fraction)
        except ValueError:
            return None

    def _fmt_price(self, value: float) -> Price:
        precision = int(self._instrument.price_precision)
        return Price.from_str(f"{value:.{precision}f}")
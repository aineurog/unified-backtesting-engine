"""Nautilus paper-trading strategy (plan T3 / §4.3).

Ports ``sim_nautilus.strategy.SignalBracketStrategy`` but, for the T3 vertical slice, submits
plain MARKET orders driven by the *same* ``decide_action`` logic the recording backend uses
(§9.4 comparability) — no bespoke accounting. Sizing comes from
``core.risk.sizing.size_position`` + ``floor_to_step``; fills are bridged to ube
``LedgerEvent``s (the ``EventLedger`` is the single source of truth, §4.6).

TP/SL bracket exits and funding arrive in later tasks (T4/T5); the slice exits on the
signal (``exit_reason="signal"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

from ube.core.ledger import EventType, LedgerEvent
from ube.core.risk.sizing import floor_to_step, size_position
from ube.papertrading.core import decide_action

from .bridge import (
    cash_movement_event,
    commission_event,
    fill_event,
    position_change_event,
)
from .runtime import get_ready_event
from .signals import SIGNAL_REGISTRY


class UbePaperConfig(StrategyConfig):  # type: ignore[misc]
    """Configuration for :class:`UbePaperStrategy`."""

    instrument_id: str = ""
    bar_type: str = ""
    sizing: Any = None  # ube SizeModel (frozen)
    cost_model: Any = None  # ube CostModel | None
    no_short: bool = False
    on_opposite_signal: str = "reverse"
    balance: float = 0.0


class UbePaperStrategy(Strategy):  # type: ignore[misc]
    """Submits MARKET orders from the per-bar ``decide_action`` and bridges fills."""

    def __init__(self, config: UbePaperConfig) -> None:
        super().__init__(config)
        self._instrument: Instrument | None = None
        # The Nautilus ``InstrumentId`` is venue-qualified (e.g. ``BTC-USDT.SIM``); it is used
        # only for cache/venue lookups. The ube ``EventLedger`` is keyed by the **canonical
        # bare symbol** (``BTC-USDT``) — the same id every other ube path uses — so we tag
        # every ``LedgerEvent`` with ``_iid`` (bare). Mismatching the two was the bug that
        # made ``trades``/``trade_table`` raise "instrument_id has no Instrument entry".
        self._venue_iid = config.instrument_id
        self._iid = InstrumentId.from_str(config.instrument_id).symbol.value
        self._bar_type = BarType.from_str(config.bar_type) if config.bar_type else None
        # Bars are published to the sandbox with live-clock timestamps (so the sandbox's
        # own execution clock advances past the read-only live order timestamps and matches
        # each order to its own bar — we cannot override the order clock in nautilus 1.231).
        # The ube ledger must stay on the *historical* (test-clock) timeline, so every
        # ``LedgerEvent`` is tagged with the historical ts looked up from ``SIGNAL_REGISTRY``
        # (keyed by the bar's live ts). ``_hist_ts`` is that historical ts for the bar
        # currently being processed.
        self._hist_ts = 0
        self._sim_side = 0
        self._sim_qty = 0.0
        self._last_close: float | None = None
        self._entry_order_ids: set[Any] = set()
        self._close_order_ids: set[Any] = set()
        self._events: list[LedgerEvent] = []
        self._quote = "USDT"

    # -- lifecycle --------------------------------------------------------- #
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(InstrumentId.from_str(self._venue_iid))
        if self._instrument is None:
            self.log.error(f"Instrument {self._venue_iid} not found in cache")
            self.stop()
            return
        self._quote = str(getattr(self._instrument, "settlement_currency", "USDT"))
        if self._bar_type is not None:
            self.subscribe_bars(self._bar_type)
        get_ready_event().set()
        self.log.info(f"UbePaperStrategy started for {self._venue_iid}")

    @property
    def events(self) -> list[LedgerEvent]:
        return self._events

    # -- signals ----------------------------------------------------------- #
    def on_bar(self, bar: Bar) -> None:
        if self._instrument is None:
            return
        SIGNAL_REGISTRY.bars_processed += 1
        self._last_close = float(bar.close.as_double())
        # Bars carry live-clock timestamps for sandbox matching; recover the historical
        # (test-clock) ts from the registry so every emitted ``LedgerEvent`` stays on the
        # ube timeline (§9.4 comparability).
        le, lx, se, sx, hist = SIGNAL_REGISTRY.at(bar.ts_event)
        self._hist_ts = int(hist)
        if not (le or lx or se or sx):
            return
        action = decide_action(
            self._sim_side,
            long_entry=le,
            long_exit=lx,
            short_entry=se,
            short_exit=sx,
            allow_short=not self.config.no_short,
            policy=self.config.on_opposite_signal,
        )
        if action == "hold":
            return

        close_first = action in ("close", "reverse")
        desired = 0
        if action == "open_long":
            desired = 1
        elif action == "open_short":
            desired = -1
        elif action == "reverse":
            desired = -self._sim_side

        t = self._hist_ts
        self._events.append(
            LedgerEvent(EventType.SIGNAL_EVALUATED, t, self._iid, action=action)
        )
        # Predictive position tracking: update ``_sim_side`` synchronously here, on the
        # same bar the orders are submitted — not in ``on_order_filled`` (which lags a
        # fill). This is what prevents ``decide_action`` from double-submitting on the
        # next bar. The fill handler only uses the order-id sets to tag open vs close
        # fills (and to set ``_sim_qty`` to the *actual* filled quantity).
        if close_first:
            self._submit_close(self._last_close)
            self._sim_side = 0
        if desired != 0:
            self._submit_open(desired, self._last_close)

    # -- fills ------------------------------------------------------------- #
    def on_order_filled(self, event: Any) -> None:
        coid = event.client_order_id
        is_open = coid in self._entry_order_ids
        is_close = coid in self._close_order_ids
        self._entry_order_ids.discard(coid)
        self._close_order_ids.discard(coid)

        if is_open:
            exit_reason: str | None = None
            self._sim_side = 1 if event.is_buy else -1
            self._sim_qty = float(event.last_qty.as_double())
        elif is_close:
            exit_reason = "signal"
            # For a reverse, the opening order is already in flight; let its fill set the
            # new side/qty, and only flatten here when the close is a true exit.
            if not self._entry_order_ids:
                self._sim_side = 0
                self._sim_qty = 0.0
        else:
            # Unknown fill (e.g. partial / rejected) — ignore rather than corrupt state.
            return

        # The fill's ``ts_init`` is the bar's *live* ts; map it back to the historical
        # (test-clock) ts so the ledger stays on the ube timeline.
        hist = self._hist_ts
        reg = SIGNAL_REGISTRY.at(int(event.ts_init))
        if len(reg) == 5 and reg[4]:
            hist = int(reg[4])
        side = 1 if event.is_buy else -1
        qty = float(event.last_qty.as_double())
        price = float(event.last_px.as_double())
        notional = qty * price
        # Cash leg of the fill (§4.6): a buy pays the notional out of the account, a sell
        # pays it back in. Together with the starting-balance cash_movement and the
        # mark-to-market of open positions this keeps the equity curve self-financing.
        self._events.append(
            cash_movement_event(
                self._iid,
                amount=-side * notional,
                currency=self._quote,
                ts_override=hist,
            )
        )
        self._events.append(
            fill_event(event, self._iid, exit_reason=exit_reason, ts_override=hist)
        )
        comm = commission_event(
            event, self._iid, self.config.cost_model, ts_override=hist
        )
        if comm is not None:
            self._events.append(comm)
        self._events.append(
            position_change_event(
                event,
                self._iid,
                side=self._sim_side,
                position_after=self._sim_side * self._sim_qty,
                ts_override=hist,
            )
        )

    # -- order submission -------------------------------------------------- #
    def _lot_step(self) -> float:
        instr = self._instrument
        if instr is None:
            return 1.0
        prec = getattr(instr, "size_precision", None)
        if prec is not None:
            return 10.0 ** (-int(prec))
        for attr in ("size_increment", "lot_size"):
            val = getattr(instr, attr, None)
            if val is not None:
                try:
                    return float(val.as_double())
                except Exception:  # pragma: no cover - defensive
                    continue
        return 1.0

    def _size_qty(self, price: float) -> Any:
        instr = self._instrument
        if instr is None or self.config.sizing is None or price is None or price <= 0:
            return instr.make_qty(0.0) if instr is not None else None
        raw = size_position(
            self.config.sizing,
            capital=self.config.balance,
            price=price,
            cost_model=self.config.cost_model,
        )
        # ``size_position`` returns a 0-d ``ndarray`` (plan blocker #13); reduce to a plain
        # float via ``.item()`` so ``make_qty`` / ``floor_to_step`` get a scalar.
        scalar = float(np.asarray(raw).item())
        step = self._lot_step()
        stepped = floor_to_step(scalar, step)
        return instr.make_qty(float(stepped))

    def _submit_open(self, desired: int, price: float | None) -> None:
        if self._instrument is None or price is None:
            return
        side = OrderSide.BUY if desired == 1 else OrderSide.SELL
        qty = self._size_qty(price)
        if qty is None or float(qty.as_double()) <= 0.0:
            self.log.warning(f"Skipped open {side}: non-positive quantity")
            return
        order = self.order_factory.market(
            instrument_id=self._instrument.id,
            order_side=side,
            quantity=qty,
            reduce_only=False,
        )
        # Predictive: the position will be ``desired`` as soon as this order fills.
        self._sim_side = desired
        self._entry_order_ids.add(order.client_order_id)
        self._events.append(
            LedgerEvent(
                EventType.ORDER_SUBMITTED,
                self._hist_ts,
                self._iid,
                order_id=str(order.client_order_id),
                side=desired,
                quantity=float(qty.as_double()),
            )
        )
        self.submit_order(order)
        self.log.info(f"Submitted open {side} qty={qty}")

    def _submit_close(self, price: float | None) -> None:
        if self._instrument is None or self._sim_qty <= 0:
            return
        side = OrderSide.SELL if self._sim_side > 0 else OrderSide.BUY
        qty = self._instrument.make_qty(self._sim_qty)
        order = self.order_factory.market(
            instrument_id=self._instrument.id,
            order_side=side,
            quantity=qty,
            reduce_only=True,
        )
        self._close_order_ids.add(order.client_order_id)
        self._events.append(
            LedgerEvent(
                EventType.ORDER_SUBMITTED,
                self._hist_ts,
                self._iid,
                order_id=str(order.client_order_id),
                side=-self._sim_side,
                quantity=float(qty.as_double()),
            )
        )
        self.submit_order(order)
        self.log.info(f"Submitted close {side} qty={qty}")


__all__ = ["UbePaperConfig", "UbePaperStrategy"]

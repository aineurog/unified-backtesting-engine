"""UbeActor — the Nautilus-specific trading piece (requirements §4.5/§4.6/§6).

The Actor translates canonical 4-column signals plus :class:`RiskConfig.exit` levels into
native Nautilus order flow. It is a :class:`~nautilus_trader.trading.strategy.Strategy`
(in this version ``Strategy`` *is* ``Actor`` and is the only class exposing the trading
surface). All exit levels are **computed with the ``core`` pure functions** and injected
into Nautilus's native mechanism — no strategy re-implementation (requirements §4.1).

Behavior summary (each item maps to a requirements §6 bullet):

* **Entries** — on a bar whose signals demand a position, size via
  :func:`~ube.core.risk.sizing.size_position` against the venue account balance and the
  bar's close, submit a market order (fills at the same-bar close — the entry is
  evaluated at bar close, requirements §4.4/§5.2).
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
  stop triggers are moved to the just-completed bar's precomputed level.
* **exit_reason stamping** — the reason (``signal``, ``take_profit``, ``stop_loss``,
  ``atr_stop``, ``trailing_stop``, ``chandelier``, ``time_exit``) is recorded per closing
  fill in
  :attr:`exit_reasons` (keyed by client-order id) so the adapter can stamp the ledger's
  closing fill (§4.6). :attr:`position_reasons` mirrors the reference netting pattern.

Trigger timing notes (probe-verified against Nautilus 1.221.0 backtest matching):

* Orders submitted during ``on_bar(bar_i)`` become active for bar ``i+1`` matching; so a
  touched stop is submitted at ``level[entry_bar]`` (the most recent level computable at
  submission) and re-targeted to the precomputed ``level[i]`` during each subsequent
  ``on_bar(bar_i)``. The venue rejects a stop whose trigger is already "in the market" at
  submission/modify time, so the trigger can only ever be a level the *completed* history
  supports — the precomputed arrays let each bar ratchet the trigger to the value the
  reference per-bar rule would use for bar ``i``, evaluated one bar forward (the level for
  bar ``i+1`` uses only data through bar ``i`` — no lookahead, matching the reference's
  ATR "shifted one bar forward" note in ``strategy/core.py``).
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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.events import OrderFilled, OrderRejected
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders.base import Order
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from ube.core.cost import CostModel
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, EngineError
from ube.core.instrument import allows_short
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    ExitPlan,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    atr,
    exit_level,
    scale_out_plan,
)
from ube.core.risk.sizing import (
    SizeModel,
    _check_affordable,
    _entry_fee_rate,
    floor_to_step,
    size_position,
)

__all__ = [
    "UbeActor",
    "UbeActorConfig",
]

#: The 4-column signal row for "no signal on this bar".
_NO_SIGNAL: tuple[bool, bool, bool, bool] = (False, False, False, False)

#: `RiskConfig.exit` type -> §4.6 exit_reason string.
_EXIT_REASONS: dict[type, str] = {
    TakeProfit: "take_profit",
    StopLoss: "stop_loss",
    ATRStop: "atr_stop",
    TrailingStop: "trailing_stop",
    ChandelierExit: "chandelier",
    TimeExit: "time_exit",
}


class UbeActorConfig(StrategyConfig):  # type: ignore[misc]
    """Actor configuration — only msgspec-serializable scalars (requirements §4.2).

    The heavyweight runtime inputs (``market_data``, ``atr_series``, ``sizing``,
    ``exits``) are passed as constructor kwargs so Nautilus never serializes them.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    signal_map: dict[int, tuple[bool, bool, bool, bool]]
    asset_class: str = "crypto_perp"

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
    """The exit book anchored at one position's entry (requirements §6.4)."""

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
        aux_atr: Mapping[str, Any] | None = None,
        sizing: SizeModel | None = None,
        exits: tuple[Exit, ...] = (),
        leverage: float = 1.0,
        cost_model: CostModel | None = None,
    ) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self._market_data = market_data
        self._aux_atr = dict(aux_atr) if aux_atr is not None else None
        self._sizing = sizing if sizing is not None else SizeModel()
        # Cost model drives fee-aware sizing (§7.1): the sizer reserves the entry fee so
        # the order can't push the account negative when the venue charges on top.
        self._cost_model = cost_model
        # Exposure multiplier (§3.2): the reference nautilus_trader strategy scales
        # ``capital * leverage``; the adapter forces this to 1.0 for cash accounts.
        self._leverage = float(leverage)
        self._exits = exits
        # Fail fast: ATR-based exits and ``volatility_target`` sizing require their named
        # aux_data series, and raw aux OHLCV must span the data/signal period (§5.2/§6.3).
        self._validate_aux()
        # The volatility estimate for ``volatility_target`` sizing is resolved from the
        # named ``aux_data`` series (§6.3) — never computed from the signal bars.
        self._vol = self._resolve_vol() if self._sizing.kind == "volatility_target" else None

        ts_ns = market_data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
        self._ts_to_index: dict[int, int] = {
            int(ts_ns[i]): i for i in range(market_data.n_bars)
        }
        self._bar_count = 0
        self._instrument: Any = None
        self._book: _Book | None = None
        self._fill_reasons: dict[str, str] = {}
        #: Shorting is not permitted on long-only asset classes (§4.5). The adapter
        #: rejects such signals up-front (``validate_long_only``); this guard is the
        #: actor's own defence when constructed directly.
        self._allow_short = allows_short(config.asset_class)

        #: Closing-fill client_order_id -> exit_reason (§4.6); read by the adapter.
        self.exit_reasons: dict[str, str] = {}
        #: position_id -> last closing exit_reason (reference netting pattern).
        self.position_reasons: dict[str, str] = {}
        #: bar index -> action string (dropdown for the ledger's signal_evaluated rows).
        self.signal_evaluated: dict[int, str] = {}
        #: client_order_id -> (bar_ns, side ±1, quantity) for every order submitted;
        #: folded into ``order_submitted`` ledger events (§4.6). Holds are not recorded.
        self.order_submitted: dict[str, tuple[int, int, float]] = {}
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
                self._submit_close(reason, fraction, ts=int(bar.ts_event))
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

    # -- entries + signal exits (requirements §6) ------------------------

    def _record_order(self, order: Order, ts: int) -> None:
        """Remember an order so the fold can emit a §4.6 ``order_submitted`` event."""
        side = 1 if order.side == OrderSide.BUY else -1
        self.order_submitted[str(order.client_order_id)] = (
            ts,
            side,
            float(order.quantity.as_double()),
        )

    def _validate_aux(self) -> None:
        """Fail fast if an ATR exit or ``volatility_target`` sizing has no usable series.

        An ATR-based exit (``ATRStop`` / ``ChandelierExit``) must name a specific
        ``aux_data`` series via its ``atr`` key (§5.2); ``volatility_target`` sizing names
        one via ``SizeModel.vol`` (§6.3). The library never computes ATR or volatility
        from the signal ``data`` bars — those may be non-time bars that cannot be
        resampled, and even time bars may be at a too-fine level (§5.2). A missing
        reference, or a name absent from ``aux_data``, is a config-shape error surfaced
        here, before the engine runs (§3 fail-fast principle).
        """
        named: set[str] = set()
        for e in self._exits:
            if isinstance(e, (ATRStop, ChandelierExit)):
                atr_name = getattr(e, "atr", None)
                if atr_name is None:
                    raise ConfigError(
                        f"{type(e).__name__} requires an 'atr' key referencing an "
                        "aux_data series (§5.2); ATR is never computed from the signal "
                        "data bars"
                    )
                named.add(str(atr_name))

        if self._sizing.kind == "volatility_target":
            vol_name = self._sizing.vol
            if not vol_name:
                raise ConfigError(
                    "volatility_target sizing requires a 'vol' key referencing an "
                    "aux_data series (§6.3); volatility is never computed from the "
                    "signal data bars"
                )
            named.add(str(vol_name))

        if not named:
            return

        if self._aux_atr is None:
            raise ConfigError(
                "config references aux_data series(es) "
                f"{sorted(named)} but aux_data was not supplied"
            )
        missing = named - set(self._aux_atr)
        if missing:
            raise ConfigError(
                "config references aux_data series(es) "
                f"{sorted(missing)} that are absent from aux_data"
            )

        # Range consistency vs the data/signal grid (named MarketData series only).
        main_ts = self._market_data.timestamps
        main_start, main_end = main_ts[0], main_ts[-1]
        main_step = main_ts[1] - main_ts[0] if len(main_ts) > 1 else pd.Timedelta(0)
        for name in sorted(named):
            value = self._aux_atr[name]
            if not isinstance(value, MarketData):
                continue  # precomputed arrays are length-checked in _atr_from_aux/_vol_from_aux
            aux_ts = value.timestamps
            if aux_ts[0] > main_start + main_step:
                raise DataShapeError(
                    f"aux_data[{name!r}] starts at {aux_ts[0]} which is after the data "
                    f"start {main_start}; the leading main bars would have no value "
                    "(aux_data must start at or before the signal/price period)"
                )
            aux_step = aux_ts[1] - aux_ts[0] if len(aux_ts) > 1 else main_step
            if aux_ts[-1] < main_end - aux_step:
                raise DataShapeError(
                    f"aux_data[{name!r}] ends at {aux_ts[-1]} which is more than one aux "
                    f"bar before the data end {main_end}; aux_data must span the "
                    "signal/price period (the trailing value would be stale)"
                )

    def _resolve_exit_atr(self, cfg: Exit) -> np.ndarray | None:
        """Resolve the ATR series for ``cfg`` (§5.2).

        An ATR-based exit (``ATRStop`` / ``ChandelierExit``) names an ``aux_data`` series
        via its ``atr`` key; that series is resolved by :meth:`_atr_from_aux` (a
        ``MarketData`` value is turned into ATR internally; a precomputed array is used
        verbatim). The missing-input case is a config error caught up front in
        :meth:`_validate_aux`, so this method never raises mid-engine.run().

        Returns ``None`` for exits that need no ATR (e.g. ``TrailingStop``); the core
        level functions compute their running extremes from the bars directly.
        """
        if isinstance(cfg, (ATRStop, ChandelierExit)):
            return self._atr_from_aux(str(cfg.atr), cfg.period)
        return None

    def _atr_from_marketdata(self, value: MarketData, period: int) -> np.ndarray:
        """Compute ATR(``period``) from a raw OHLCV ``MarketData`` (§5.2), no look-ahead.

        1. ATR(``period``) is computed over the aux bars (Wilder, causal — value at bar
           ``h`` uses only bars through ``h``).
        2. The series is shifted forward one aux bar, so a main bar inside hour ``h``
           only ever sees the ATR of the *last completed* aux bar (``h-1``) — no
           look-ahead into the bar being traded.
        3. The aux ATR is forward-filled onto the (possibly finer) main bar grid so it
           is row-aligned with ``data`` (length ``n_bars``).
        """
        atr_aux = atr(value, period)  # causal Wilder ATR, aligned to aux bars
        series = pd.Series(atr_aux, index=value.timestamps)
        # No look-ahead: shift forward one aux bar (bar h uses ATR through h-1).
        series = series.shift(1).fillna(series.iloc[0])
        # Forward-fill the aux ATR onto the main bar grid (length == n_bars).
        aligned = series.reindex(self._market_data.timestamps, method="ffill")
        return aligned.to_numpy(dtype=np.float64)

    def _atr_from_aux(self, name: str, period: int) -> np.ndarray:
        """Resolve an ``aux_data`` entry into an ATR series (§5.2), with no look-ahead.

        A ``MarketData`` value is a raw OHLCV series — typically coarser than the signal
        bars — that the library turns into ATR (via :meth:`_atr_from_marketdata`); a
        precomputed array is used verbatim.
        """
        aux = self._aux_atr
        if aux is None or name not in aux:
            raise ConfigError(f"aux_data[{name!r}] was not supplied")
        value = aux[name]
        if isinstance(value, MarketData):
            arr = self._atr_from_marketdata(value, period)
        else:
            arr = np.asarray(value, dtype=np.float64)
        if arr.shape[0] != self._market_data.n_bars:
            raise DataShapeError(
                f"aux_data[{name!r}] has {arr.shape[0]} bars but data has "
                f"{self._market_data.n_bars}"
            )
        if not np.isfinite(arr).all() or (arr <= 0).any():
            raise ConfigError("atr_series must be finite and positive")
        return arr

    def _resolve_vol(self) -> np.ndarray:
        """Resolve the ``volatility_target`` vol estimate from ``aux_data`` (§6.3).

        The ``SizeModel.vol`` name is validated up front in :meth:`_validate_aux`, so
        this method never raises mid-engine.run() for a missing reference.
        """
        return self._vol_from_aux(str(self._sizing.vol))

    def _vol_from_marketdata(self, value: MarketData) -> np.ndarray:
        """Per-bar volatility (ATR/price) from a raw OHLCV ``MarketData``, no look-ahead.

        Mirrors :meth:`_atr_from_marketdata`: ATR over the aux bars (causal Wilder), then
        shifted forward one aux bar so a main bar inside hour ``h`` only ever sees the
        volatility of the *last completed* aux bar (``h-1``), then forward-filled onto the
        main bar grid. Dividing by the aux close turns ATR into a dimensionless fraction.
        """
        vol_aux = atr(value) / value.close
        series = pd.Series(vol_aux, index=value.timestamps)
        # No look-ahead: shift forward one aux bar (bar h uses vol through h-1).
        series = series.shift(1).fillna(series.iloc[0])
        # Forward-fill the aux vol onto the main bar grid (length == n_bars).
        aligned = series.reindex(self._market_data.timestamps, method="ffill")
        return aligned.to_numpy(dtype=np.float64)

    def _vol_from_aux(self, name: str) -> np.ndarray:
        """Resolve an ``aux_data`` entry into a per-bar volatility fraction (§6.3).

        A ``MarketData`` value is a raw OHLCV series — typically coarser than the signal
        bars — that the library turns into ``ATR/price`` (via
        :meth:`_vol_from_marketdata`); a precomputed array is used verbatim.
        """
        aux = self._aux_atr
        if aux is None or name not in aux:
            raise ConfigError(f"aux_data[{name!r}] was not supplied")
        value = aux[name]
        if isinstance(value, MarketData):
            arr = self._vol_from_marketdata(value)
        else:
            arr = np.asarray(value, dtype=np.float64)
        if arr.shape[0] != self._market_data.n_bars:
            raise DataShapeError(
                f"aux_data[{name!r}] has {arr.shape[0]} bars but data has "
                f"{self._market_data.n_bars}"
            )
        if not np.isfinite(arr).all() or (arr <= 0).any():
            raise ConfigError("vol estimate must be finite and positive")
        return arr

    def _apply_signal(self, signals: tuple[bool, bool, bool, bool], bar: Bar, idx: int) -> None:
        le, lx, se, sx = signals
        position = self._open_position()
        action = "hold"

        if position is not None and position.is_long:
            if lx or se:
                self._submit_close("signal", ts=int(bar.ts_event))
                action = "long_exit"
                if se and self._allow_short:
                    self._open(-1, bar, idx)
                    action = "short_entry"
        elif position is not None and position.is_short:
            if sx or le:
                self._submit_close("signal", ts=int(bar.ts_event))
                action = "short_exit"
                if le:
                    self._open(1, bar, idx)
                    action = "long_entry"
        else:
            if le:
                self._open(1, bar, idx)
                action = "long_entry"
            elif se and self._allow_short:
                self._open(-1, bar, idx)
                action = "short_entry"

        if action != "hold":
            self.signal_evaluated[idx] = action

    def _open(self, side: int, bar: Bar, idx: int) -> None:
        price = float(bar.close)
        qty = self._entry_quantity(price, idx)
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
        self._record_order(order, int(bar.ts_event))
        if self._exits:
            self._setup_book(
                side=side,
                entry_bar=idx,
                entry_price=price,
                size=qty,
                entry_bar_ts=int(bar.ts_event),
            )

    def _entry_quantity(self, price: float, idx: int) -> Quantity | None:
        """Sized entry quantity in tradable units at entry bar ``idx``.

        For ``volatility_target`` the vol estimate is the *entry bar's* value
        (``self._vol[idx]``) — never the whole series or its first element, which
        would silently mis-size every later entry.

        For the fee-aware ``all_in`` / ``equal_weight`` sizers the lot-grid conversion
        floors (never rounds up) and the affordability guard is re-run against the
        **final** quantity (§7.1): ``size_position`` verifies the raw float, but
        ``make_qty`` quantizes to the instrument's size precision and an up-rounded
        quantity could exceed the verified size by up to one lot's notional + fee —
        pushing the account negative and halting the engine silently.
        """
        if price <= 0.0 or self._instrument is None:
            return None
        account = self.cache.account_for_venue(self.instrument_id.venue)
        balance = float(account.balance_total().as_double())
        # Apply the sizing leverage as an exposure multiplier on the allocated
        # capital, mirroring the reference nautilus_trader quantity formula
        # (``leveraged_capital = capital * leverage`` -> ``qty = notional / price``).
        leveraged_balance = balance * self._leverage
        vol = self._vol[idx] if self._vol is not None else None
        units = np.asarray(
            size_position(
                self._sizing,
                capital=leveraged_balance,
                price=price,
                n=1,
                vol=vol,
                cost_model=self._cost_model,
            ),
            dtype=np.float64,
        )
        raw = float(units.ravel()[0])
        # §7.1: for the FEE-AWARE all_in / equal_weight sizers the lot-grid conversion
        # must floor (never round up): make_qty quantizes to the instrument's size
        # precision and an up-rounded quantity can exceed the size the affordability
        # guard verified by up to one lot's notional + fee — pushing the account
        # negative and halting the engine silently. The gate is the RESOLVED entry-fee
        # rate (not ``cost_model is not None``): the adapter always supplies a cost
        # model (falling back to per-asset-class defaults), and a zero-rate model has
        # no reserved fees to protect — fee-less runs keep the legacy conversion.
        step = float(self._instrument.size_increment)
        fee_rate = _entry_fee_rate(self._cost_model)
        fee_aware = self._sizing.kind in ("all_in", "equal_weight") and fee_rate > 0.0
        tradable = (
            float(floor_to_step(raw, step))
            if fee_aware and math.isfinite(step) and step > 0.0
            else raw
        )
        try:
            qty = self._instrument.make_qty(tradable)
        except ValueError:
            return None
        final_units = float(qty.as_double())
        if final_units <= 0.0:
            return None
        # Re-verify with the FINAL venue-quantized quantity (§7.1): nothing after the
        # in-sizer guard may push the outlay past capital without a loud error.
        # Zero-fee runs keep legacy semantics (the venue tolerates a sub-lot notional
        # overhang; there is no fee to turn it into a shortfall).
        if fee_rate > 0.0:
            _check_affordable(
                leveraged_balance,
                price,
                np.asarray([final_units], dtype=np.float64),
                self._cost_model,
            )
        return qty

    def _submit_close(self, reason: str, fraction: float = 1.0, ts: int | None = None) -> None:
        """Submit a reduce-only market close of ``fraction`` of the remaining position.

        A full close (``fraction >= 1.0`` — stops and time exits, §6.4) cancels the
        exit book and closes the position outright. A scale-out partial
        (``fraction < 1.0`` — a take-profit slice) closes only that slice with a
        reduce-only market order and leaves the book alive so the sibling stops
        keep protecting the remainder (§6.4).

        ``ts`` is the submission bar's event timestamp (for the ``order_submitted`` fold).
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
        if ts is not None:
            self._record_order(order, ts)
        self._fill_reasons[str(order.client_order_id)] = reason
        if fraction >= 1.0:
            self._book = None

    # -- exit book (requirements §6.4) ----------------------------------------

    def _setup_book(
        self, *, side: int, entry_bar: int, entry_price: float, size: Quantity, entry_bar_ts: int
    ) -> None:
        # Resolve each exit's ATR series independently (§5.2): a named aux_data series
        # if supplied, else ATR computed from the bars at the exit's period. Reuses the
        # core scale_out_plan/exit_level helpers (calling, not modifying, core).
        sub_plans = [
            scale_out_plan(
                (cfg,),
                market_data=self._market_data,
                side=side,
                entry_price=entry_price,
                entry_bar=entry_bar,
                atr_series=self._resolve_exit_atr(cfg),
            )
            for cfg in self._exits
        ]
        plan = ExitPlan(
            fractions=tuple(p.fractions[0] for p in sub_plans),
            triggered=tuple(p.triggered[0] for p in sub_plans),
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
                atr_series=self._resolve_exit_atr(cfg),
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
                self._record_order(order, entry_bar_ts)
                self._fill_reasons[str(order.client_order_id)] = reason
                book.live[str(order.client_order_id)] = _LiveExit(
                    order=order, reason=reason, fraction=fraction
                )
            else:
                # StopLoss / ATRStop / TrailingStop / Chandelier: a native stop_market
                # whose trigger is placed at the entry-bar level, then ratcheted by
                # _maintain_exits on each later bar. The initial trigger must be *outside*
                # the market at submission (the venue rejects a stop already "in the
                # market", probe-verified on Nautilus 1.221.0), so the entry-bar level —
                # not the next bar's — is used (levels[entry_bar + 1] only exists once
                # bar entry_bar+1 has completed).
                target = float(levels[entry_bar])
                order = self.order_factory.stop_market(
                    self.instrument_id,
                    exit_side,
                    size,
                    trigger_price=self._fmt_price(target),
                    reduce_only=True,
                )
                self.submit_order(order)
                self._record_order(order, entry_bar_ts)
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
            if live.levels is not None and idx < n:
                trigger = self._fmt_price(float(live.levels[idx]))
            qty_same = float(order.quantity.as_double()) == float(new_qty.as_double())
            trigger_same = trigger is not None and order.trigger_price is not None and (
                float(order.trigger_price.as_double()) == float(trigger)
            )
            if qty_same and (trigger is None or trigger_same):
                continue
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
        ts = int(bar.ts_event)
        idx = self._ts_to_index.get(ts)
        if idx is None:
            # A genuine data/logic error: the engine delivered a bar whose timestamp does
            # not match any input bar. Guessing "the bar after the last one" would compute
            # against the wrong bar's data, so fail loudly instead of silently.
            raise EngineError(
                f"bar timestamp {ts} not found in the known bar index "
                f"({len(self._ts_to_index)} bars); the engine produced a bar that does "
                "not correspond to any input bar"
            )
        return idx

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

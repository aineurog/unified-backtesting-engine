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
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from ube.core.cost import fill_cost
from ube.core.data import MarketData
from ube.core.ledger import EventType, LedgerEvent
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    TrailingStop,
    exit_triggered,
    scale_out_fraction,
)
from ube.core.risk.sizing import floor_to_step, size_position
from ube.papertrading.core import decide_action
from ube.papertrading.state import ExitSeed

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
    leverage: float = 1.0
    multiplier: float = 1.0
    exits: tuple[Any, ...] = ()
    funding_rate: float = 0.0
    borrow_rate: float = 0.0
    funding_interval_ns: int = 0
    open_position: Any = None
    exit_seed: Any = None  # ube ExitSeed — seeds trailing/ATR exits on resume (issue C)
    bar_period_ns: int = 0  # median bar period for seeding synthetic timestamps (issue 1)
    last_funding_ns: int | None = None  # persisted funding clock (issue 2)


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
        # each order to its own bar — we cannot override the order clock in nautilus 1.221).
        # The ube ledger must stay on the *historical* (test-clock) timeline, so every
        # ``LedgerEvent`` is tagged with the historical ts looked up from ``SIGNAL_REGISTRY``
        # (keyed by the bar's live ts). ``_hist_ts`` is that historical ts for the bar
        # currently being processed.
        self._hist_ts = 0
        self._sim_side = config.open_position.side if config.open_position else 0
        self._sim_qty = float(config.open_position.quantity) if config.open_position else 0.0
        self._last_close: float | None = None
        self._entry_order_ids: set[Any] = set()
        self._close_order_ids: set[Any] = set()
        self._events: list[LedgerEvent] = []
        self._quote = "USDT"
        self._leverage = float(getattr(config, "leverage", 1.0) or 1.0)
        self._multiplier = float(getattr(config, "multiplier", 1.0) or 1.0)
        self._exits: tuple[Exit, ...] = tuple(getattr(config, "exits", ()))
        self._funding_rate = float(getattr(config, "funding_rate", 0.0) or 0.0)
        self._borrow_rate = float(getattr(config, "borrow_rate", 0.0) or 0.0)
        self._funding_interval_ns = int(getattr(config, "funding_interval_ns", 0) or 0)
        # Live cash balance for fee-aware sizing — mirrors backtest's
        # account.balance_total() * leverage (starts at balance, then
        # updated on each fill's cash leg + commission).
        self._current_balance = float(config.balance)
        # Bar history for exit level computation (vectorized, §8). Stored as
        # raw Bar objects to rebuild MarketData for exit_triggered.
        self._bars: list[Bar] = []
        # For resumed positions, entry bar is 0 relative to the new slice's bar buffer;
        # trailing exits will be approximate until enough bars accumulate. For full
        # fidelity, PaperState would need to persist the bar history (Point 9).
        self._entry_bar: int | None = 0 if config.open_position else None
        self._entry_price: float | None = (
            float(config.open_position.entry_price) if config.open_position else None
        )
        # Aux data for ATR-based exits (from PaperState.aux_data, backtest parity §5.2).
        self._aux_data: dict[str, Any] = {}
        # Funding: track last funding timestamp to avoid double-counting. Persisted in
        # PaperState.last_funding_ns so a resume continues from the exact saved point
        # (issue 2) — not from bar spacing, which would be contaminated by synthetic seed bars.
        _lf = getattr(config, "last_funding_ns", None)
        self._last_funding_ns: int | None = int(_lf) if _lf is not None else None
        self._bar_period_ns: int = int(getattr(config, "bar_period_ns", 0) or 0)
        # Exit seed (issue C): on resume, the persistence layer writes the minimal exit
        # statistics (running extreme price + ATR warmup window) so trailing / ATR exits are
        # not degenerate on the first bars after a restart. Tracks:
        #   _extreme_price — the running peak (long) / trough (short) since entry.
        #   _atr_window    — a bounded rolling [high, low, close] window for ATR warmup.
        # The bars for these are built lazily on the first real bar (needs instrument
        # precision + the first bar's live/hist timestamps to assign proper preceding
        # timestamps — issue 1). The current values are exposed via :meth:`exit_seed` for
        # the backend to persist.
        self._seed: ExitSeed | None = config.exit_seed
        self._seeded: bool = False
        self._extreme_price: float | None = (
            float(config.exit_seed.extreme_price) if config.exit_seed else None
        )
        self._atr_window: list[list[float]] = []
        self._atr_period = self._max_atr_period()

    def _max_atr_period(self) -> int:
        """Largest ``period`` among configured ATR-style exits; 0 if none."""
        period = 0
        for cfg in self._exits:
            if isinstance(cfg, (ATRStop, ChandelierExit)):
                period = max(period, int(getattr(cfg, "period", 14)))
        return period

    def exit_seed(self) -> ExitSeed | None:
        """The current exit seed for the backend to persist (issue C).

        Returns ``None`` when flat or no path-dependent exit needs seeding, so a resumed run
        only carries the minimal statistics an actual exit requires. The extreme price is
        tracked since entry; the ATR window holds the last ``period + 1`` real bars.
        """
        if self._sim_side == 0:
            return None
        extreme = self._extreme_price
        if not self._has_trailing_exit():
            extreme = None
        window: list[list[float]] | None = None
        if self._atr_period > 0 and self._atr_window:
            window = [
                [float(high), float(low), float(c)] for high, low, c in self._atr_window
            ]
        if extreme is None and window is None:
            return None
        return ExitSeed(extreme_price=extreme, atr_window=window)

    def _has_trailing_exit(self) -> bool:
        """Whether any configured exit needs the persisted running extreme (trailing)."""
        for cfg in self._exits:
            if isinstance(cfg, TrailingStop) or (
                isinstance(cfg, (ATRStop, ChandelierExit)) and bool(getattr(cfg, "trailing", False))
            ):
                return True
        return False

    def _reset_exit_seed(self) -> None:
        """Clear the exit-seed tracking (called when the position becomes flat)."""
        self._extreme_price = None
        self._atr_window = []

    # -- lifecycle --------------------------------------------------------- #
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(InstrumentId.from_str(self._venue_iid))
        if self._instrument is None:
            self.log.error(f"Instrument {self._venue_iid} not found in cache")
            self.stop()
            return
        self._quote = str(getattr(self._instrument, "settlement_currency", "USDT"))
        # Issue C: seeding is now lazy in ``on_bar`` (needs the first real bar's
        # live/hist timestamps and the bar period to assign proper preceding
        # timestamps — issue 1). The synthetic bars are used *only* for exit-level
        # computation (via ``_build_market_data``); they are never published to the
        # sandbox, so they cannot generate spurious orders.
        if self._bar_type is not None:
            self.subscribe_bars(self._bar_type)
        get_ready_event().set()
        self.log.info(f"UbePaperStrategy started for {self._venue_iid}")

    def _ensure_seeded(self, bar: Bar, hist: int) -> None:
        """Lazily prepend persisted seed bars with proper preceding timestamps (issue 1)."""
        if self._seeded or self._seed is None or self._sim_side == 0 or self._instrument is None:
            return
        if self._bars:
            # Already have bars (should only happen if on_start seeded); avoid double-seed
            self._seeded = True
            return
        seed = self._seed
        n_atr = len(seed.atr_window) if seed.atr_window else 0
        has_extreme = seed.extreme_price is not None
        n_seed = (1 if has_extreme else 0) + n_atr
        if n_seed == 0:
            self._seeded = True
            return
        period = self._bar_period_ns or 3_600_000_000_000
        pp = self._instrument.price_precision
        sp = self._instrument.size_precision
        assert self._bar_type is not None
        # Create n_seed synthetic bars immediately before the first real bar,
        # spaced by exactly one bar period, so elapsed-time calculations remain
        # 1h→1h→1h and funding is not contaminated by a huge 0→hist gap.
        # Both live (sandbox) and hist (ube) timelines are spaced.
        for i in range(n_seed):
            is_extreme = has_extreme and i == 0
            # Synthetic bars are never published to the sandbox; their live clock is
            # irrelevant. Store the historical (test-clock) timestamp in ts_event so
            # _build_market_data can recover it without a registry entry. Spacing remains
            # exactly one bar period, so no funding contamination (synthetic never accrues
            # funding — funding uses the persisted _last_funding_ns, not _bars spacing).
            h = int(hist) - (n_seed - i) * int(period)
            if is_extreme:
                price = float(seed.extreme_price)  # type: ignore[arg-type]
                p = Price(price, pp)
                b = Bar(
                    bar_type=self._bar_type,
                    open=p,
                    high=p,
                    low=p,
                    close=p,
                    volume=Quantity(0.0, sp),
                    ts_event=h,
                    ts_init=h,
                )
            else:
                idx = i - (1 if has_extreme else 0)
                hlc = seed.atr_window[idx]  # type: ignore[index]
                b = Bar(
                    bar_type=self._bar_type,
                    open=Price(float(hlc[2]), pp),
                    high=Price(float(hlc[0]), pp),
                    low=Price(float(hlc[1]), pp),
                    close=Price(float(hlc[2]), pp),
                    volume=Quantity(0.0, sp),
                    ts_event=h,
                    ts_init=h,
                )
            self._bars.append(b)
        self._entry_bar = 0
        self._seeded = True

    def _seed_exit_bars(self) -> None:
        """Deprecated: seeding is now lazy in ``on_bar`` with proper timestamps."""
        return

    @property
    def events(self) -> list[LedgerEvent]:
        return self._events

    # -- signals + risk exits ---------------------------------------------- #
    def _build_market_data(self) -> MarketData | None:
        """Rebuild MarketData from buffered bars for exit level computation."""
        if not self._bars:
            return None
        # Reconstruct arrays from Bar objects — vectorized, no per-bar loop in core.
        opens = np.array([float(b.open.as_double()) for b in self._bars], dtype=np.float64)
        highs = np.array([float(b.high.as_double()) for b in self._bars], dtype=np.float64)
        lows = np.array([float(b.low.as_double()) for b in self._bars], dtype=np.float64)
        closes = np.array([float(b.close.as_double()) for b in self._bars], dtype=np.float64)
        volumes = np.array([float(b.volume.as_double()) for b in self._bars], dtype=np.float64)
        # Use historical timestamps for MarketData index (so exit levels align).
        # Synthetic seed bars are not in SIGNAL_REGISTRY; their ts_event already
        # is the historical timestamp (see _ensure_seeded), so fall back to it.
        idx_vals = []
        for b in self._bars:
            hist = SIGNAL_REGISTRY.at(b.ts_event)[4]
            if hist == 0:
                hist = int(b.ts_event)
            idx_vals.append(int(hist))
        idx = np.array(idx_vals, dtype=np.int64)
        # Bypass validation — bars are already validated at ingestion.
        md = MarketData.__new__(MarketData)
        object.__setattr__(md, "open", opens)
        object.__setattr__(md, "high", highs)
        object.__setattr__(md, "low", lows)
        object.__setattr__(md, "close", closes)
        object.__setattr__(md, "volume", volumes)
        object.__setattr__(md, "index", np.array(idx, dtype=np.int64))
        # Also need timestamps as DatetimeIndex for ledger — use index as proxy.
        # MarketData expects index to be DatetimeIndex, but for exit computation we
        # only need OHLC arrays, so we store a minimal stub.
        return md

    def _check_risk_exits(self) -> tuple[str, float] | None:
        """Check RiskConfig.exit in order, return (exit_reason, fraction) if triggered."""
        if (
            not self._exits
            or self._sim_side == 0
            or self._entry_price is None
            or self._entry_bar is None
        ):
            return None
        md = self._build_market_data()
        if md is None or md.n_bars == 0:
            return None
        cur_bar = md.n_bars - 1
        # Resolve aux_data for ATR-based exits (backtest parity §5.2).
        # For paper, aux_data is supplied via PaperState.aux_data or via
        # strategy config; we look for an 'atr' series by name or compute
        # ATR from the bar history if not supplied.
        aux_atr: dict[str, Any] | None = None
        if hasattr(self, "_aux_data") and self._aux_data:
            aux_atr = self._aux_data
        for cfg in self._exits:
            try:
                # ATR-based exits need atr_series — resolve from aux_data if cfg.atr is set.
                atr_series = None
                atr_name = getattr(cfg, "atr", None)
                if atr_name is not None and aux_atr is not None and atr_name in aux_atr:
                    raw = aux_atr[atr_name]
                    if isinstance(raw, np.ndarray):
                        atr_series = raw
                    elif isinstance(raw, MarketData):
                        # Compute ATR from aux MarketData, aligned to main grid.
                        from ube.core.risk.exits import atr as _atr

                        atr_series = _atr(raw, getattr(cfg, "period", 14))
                    else:
                        atr_series = np.asarray(raw, dtype=np.float64)
                triggered = exit_triggered(
                    cfg,
                    market_data=md,
                    side=self._sim_side,
                    entry_price=self._entry_price,
                    entry_bar=self._entry_bar,
                    atr_series=atr_series,
                )
            except Exception:
                continue
            if triggered[cur_bar]:
                # Map config type to exit_reason string for ledger.
                name = type(cfg).__name__
                reason = {
                    "TakeProfit": "take_profit",
                    "StopLoss": "stop_loss",
                    "ATRStop": "atr_stop",
                    "TrailingStop": "trailing_stop",
                    "TimeExit": "time_exit",
                    "ChandelierExit": "chandelier_exit",
                }.get(name, name.lower())
                return reason, scale_out_fraction(cfg)
        return None

    def on_bar(self, bar: Bar) -> None:
        if self._instrument is None:
            return
        SIGNAL_REGISTRY.bars_processed += 1
        # Bars carry live-clock timestamps for sandbox matching; recover the historical
        # (test-clock) ts from the registry so every emitted ``LedgerEvent`` stays on the
        # ube timeline (§9.4 comparability).
        le, lx, se, sx, hist = SIGNAL_REGISTRY.at(bar.ts_event)
        self._hist_ts = int(hist)
        # Issue 1: lazily seed exit-level bar buffer with proper preceding timestamps
        # (not 0) so elapsed-time calculations are not contaminated by a huge 0→hist gap.
        # Seeding happens only on the first real bar after a resume, before the real bar
        # is buffered, so spacing remains 1h→1h→1h.
        self._ensure_seeded(bar, self._hist_ts)
        self._last_close = float(bar.close.as_double())
        # Buffer bar for exit level computation.
        self._bars.append(bar)
        # Issue C: maintain the running extreme price and ATR warmup window *before* the risk
        # exit check, so the current bar contributes to the peak/ATR used to seed the next
        # resume. Only tracked while a position is open and a path-dependent exit requires it.
        if self._sim_side != 0:
            hi = float(bar.high.as_double())
            lo = float(bar.low.as_double())
            if self._extreme_price is None:
                self._extreme_price = hi if self._sim_side > 0 else lo
            else:
                self._extreme_price = (
                    max(self._extreme_price, hi)
                    if self._sim_side > 0
                    else min(self._extreme_price, lo)
                )
            if self._atr_period > 0:
                self._atr_window.append([hi, lo, float(bar.close.as_double())])
                if len(self._atr_window) > self._atr_period + 1:
                    self._atr_window.pop(0)

        # — Risk exits (T4) — checked before signals, stop precedence.
        risk_exit = self._check_risk_exits()
        if risk_exit is not None and self._sim_side != 0:
            reason, fraction = risk_exit
            t = self._hist_ts
            self._events.append(
                LedgerEvent(EventType.SIGNAL_EVALUATED, t, self._iid, action=f"exit_{reason}")
            )
            # Scale-out: fraction of remaining position (1.0 = full close).
            qty_to_close = self._sim_qty * fraction
            # For partial closes, keep remaining position; for full, flatten.
            self._submit_close(self._last_close, fraction=fraction, exit_reason=reason)
            if fraction >= 1.0 - 1e-12:
                self._sim_side = 0
                self._sim_qty = 0.0
                self._entry_bar = None
                self._entry_price = None
                self._reset_exit_seed()
            else:
                self._sim_qty -= qty_to_close
            return

        # — Funding carry (T5) — accrues on *every* bar the position is open, regardless of
        # whether a signal fires this bar. Placed before the signal early-return so funding
        # is not limited to signal bars. Separated from synthetic indicator history:
        # funding uses the persisted real-market clock (_last_funding_ns), not bar spacing
        # derived from _bars (which contains synthetic seed bars). Synthetic bars never
        # generate funding (issue 3).
        if self._sim_side != 0 and (self._funding_rate != 0.0 or self._borrow_rate != 0.0):
            # Elapsed-time accrual using the persisted funding clock (issue 2).
            # _last_funding_ns is set at entry and after each funding event; on resume it
            # is restored from PaperState.last_funding_ns, so the next bar accrues
            # hist - last_funding (correct even across a gap).
            if self._last_funding_ns is None:
                bar_span = 0
            else:
                bar_span = self._hist_ts - int(self._last_funding_ns)
                if bar_span < 0:
                    bar_span = 0
            frac = 1.0
            if self._funding_interval_ns > 0 and bar_span > 0:
                frac = bar_span / float(self._funding_interval_ns)
            notional = abs(self._sim_qty) * (self._last_close or 0.0) * self._multiplier
            # Per-period carry, matching the canonical ledger adapter exactly
            # (``ledger.funding_payments``, §24): amount = (funding + borrow * (side < 0))
            # * notional * frac. Both terms are non-negative; positive = the trader pays.
            # The short side adds the borrow cost but does *not* flip the funding sign —
            # this is the single shared convention with the backtest adapter.
            borrow = self._borrow_rate if self._sim_side < 0 else 0.0
            amount = (self._funding_rate + borrow) * notional * frac
            if amount != 0.0:
                self._events.append(
                    LedgerEvent(
                        EventType.FUNDING_PAYMENT,
                        self._hist_ts,
                        self._iid,
                        amount=amount,
                        currency=self._quote,
                    )
                )
                # Update live balance for next sizing. A negative amount is funding the trader
                # *received* — credit it back to the balance; a positive amount is paid — deduct it.
                self._current_balance -= amount
                self._last_funding_ns = self._hist_ts

        if not (le or lx or se or sx):
            return
        allow_short = not self.config.no_short
        action = decide_action(
            self._sim_side,
            long_entry=le,
            long_exit=lx,
            short_entry=se,
            short_exit=sx,
            allow_short=allow_short,
            policy=self.config.on_opposite_signal,
        )
        if action == "hold":
            if not allow_short and (se or sx):
                self.log.warning(
                    f"Short signal ignored for long-only {self._iid}: se={se} sx={sx} "
                    f"at bar {self._hist_ts} (flat or no short allowed)"
                )
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
        # Optimistic close-credit for same-bar reverse sizing: the sandbox fills
        # the close asynchronously, so ``_current_balance`` doesn't yet reflect the
        # close proceeds when ``_submit_open`` calls ``_size_qty``.  Credit the
        # expected net proceeds (notional − commission) before sizing, then restore
        # after — the fill handlers will do the real accounting when they fire.
        _close_credit = 0.0
        if close_first and self._sim_qty > 0:
            _close_notional = self._sim_qty * (self._last_close or 0.0) * self._multiplier
            _close_comm = float(
                fill_cost(self.config.cost_model, notional=_close_notional)
            ) if self.config.cost_model else 0.0
            _close_credit = _close_notional - _close_comm
            self._current_balance += _close_credit

        if close_first:
            self._submit_close(self._last_close)
            self._sim_side = 0
            self._sim_qty = 0.0
            self._entry_bar = None
            self._entry_price = None
        if desired != 0:
            self._submit_open(desired, self._last_close)
            # Record entry for risk-exit level computation (vectorized, §8).
            self._entry_bar = len(self._bars) - 1
            self._entry_price = self._last_close
            # Funding: reset last funding timestamp on new entry.
            self._last_funding_ns = self._hist_ts

        # Restore optimistic credit — fill handlers manage the real balance.
        if _close_credit != 0.0:
            self._current_balance -= _close_credit

    # -- fills ------------------------------------------------------------- #
    def on_order_filled(self, event: Any) -> None:
        coid = event.client_order_id
        is_open = coid in self._entry_order_ids
        is_close = coid in self._close_order_ids
        self._entry_order_ids.discard(coid)
        self._close_order_ids.discard(coid)
        # Risk-exit reason stashed at submit time.
        close_reason: str | None = None
        if hasattr(self, "_close_reasons"):
            close_reason = self._close_reasons.pop(coid, None)

        if is_open:
            exit_reason: str | None = None
            self._sim_side = 1 if event.is_buy else -1
            self._sim_qty = float(event.last_qty.as_double())
            # Record entry for risk-exit level computation.
            self._entry_bar = len(self._bars) - 1
            self._entry_price = float(event.last_px.as_double())
            self._last_funding_ns = self._hist_ts
            # Issue C: a fresh (non-resume) entry starts a new running extreme and ATR window
            # from this bar. The fill arrives asynchronously (one bar after submission), so
            # the next bar's on_bar may have already set _extreme_price/_atr_window from
            # that next bar's high/low. Preserve the more extreme value and prepend the
            # entry bar to the window instead of overwriting (fix 6).
            if len(self._bars):
                seg = self._bars[-1]
                seg_high = float(seg.high.as_double())
                seg_low = float(seg.low.as_double())
                seg_close = float(seg.close.as_double())
                if self._extreme_price is None:
                    self._extreme_price = seg_high if self._sim_side > 0 else seg_low
                else:
                    if self._sim_side > 0:
                        self._extreme_price = max(self._extreme_price, seg_high)
                    else:
                        self._extreme_price = min(self._extreme_price, seg_low)
                if self._atr_period > 0:
                    entry_hlc = [seg_high, seg_low, seg_close]
                    if not self._atr_window:
                        self._atr_window = [entry_hlc]
                    elif self._atr_window[0] != entry_hlc:
                        self._atr_window.insert(0, entry_hlc)
                        if len(self._atr_window) > self._atr_period + 1:
                            self._atr_window.pop()
            else:
                self._extreme_price = None
                self._atr_window = []
        elif is_close:
            exit_reason = close_reason or "signal"
            # For a reverse, the opening order is already in flight; let its fill set the
            # new side/qty, and only flatten here when the close is a true exit.
            if not self._entry_order_ids:
                # Handle partial closes (scale-out): reduce qty, keep side.
                filled_qty = float(event.last_qty.as_double())
                if filled_qty < self._sim_qty - 1e-12:
                    self._sim_qty -= filled_qty
                else:
                    self._sim_side = 0
                    self._sim_qty = 0.0
                    self._entry_bar = None
                    self._entry_price = None
                    self._reset_exit_seed()
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
        notional = qty * price * self._multiplier
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
            fill_event(
                event,
                self._iid,
                exit_reason=exit_reason,
                ts_override=hist,
                multiplier=self._multiplier,
            )
        )
        comm = commission_event(
            event,
            self._iid,
            self.config.cost_model,
            ts_override=hist,
            multiplier=self._multiplier,
            currency=self._quote,
        )
        # Update live balance for next sizing (mirrors account.balance_total).
        self._current_balance += -side * notional
        if comm is not None:
            self._events.append(comm)
            self._current_balance -= float(comm.amount)  # type: ignore[arg-type]
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
        # Mirror backtest's account.balance_total() * leverage — paper uses
        # live cash balance (updated on each fill) scaled by leverage.
        capital = self._current_balance * self._leverage
        raw = size_position(
            self.config.sizing,
            capital=capital,
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

    def _submit_close(
        self, price: float | None, fraction: float = 1.0, exit_reason: str | None = None
    ) -> None:
        if self._instrument is None or self._sim_qty <= 0:
            return
        side = OrderSide.SELL if self._sim_side > 0 else OrderSide.BUY
        qty_val = self._sim_qty * fraction
        qty = self._instrument.make_qty(qty_val)
        order = self.order_factory.market(
            instrument_id=self._instrument.id,
            order_side=side,
            quantity=qty,
            reduce_only=True,
        )
        # Store exit_reason for fill tagging — keyed by client_order_id.
        # For signal exits, on_order_filled uses "signal"; for risk exits, the
        # caller's reason (e.g. "take_profit") is stashed here and read on fill.
        if exit_reason is not None:
            # Use a side dict to map order -> reason; reuse _close exit_reason
            # via a new dict to avoid mixing with signal logic.
            if not hasattr(self, "_close_reasons"):
                self._close_reasons: dict[Any, str] = {}
            self._close_reasons[order.client_order_id] = exit_reason
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

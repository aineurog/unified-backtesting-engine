"""backtrader engine adapter (§4.1, §4.2).

This module is the thin orchestration layer: it validates the contract, translates the inputs
via :mod:`~ube.adapters.backtrader_adapter.adapt_data`, runs the event-driven strategy via
:mod:`~ube.adapters.backtrader_adapter.engine`, and folds the strategy's ordered order/signal
records into the canonical :class:`~ube.core.ledger.EventLedger` (§4.6).

Fold semantics mirror the vectorbt/nautilus adapters exactly — starting-balance cash at the
first bar, ``signal_evaluated`` / ``order_submitted`` at the signal (submit) bar, then
``fill`` + cash leg + ``commission`` (via the core ``fill_cost`` on slipped notional) +
``position_change`` at the fill bar, with slippage applied at the fill-price level (§8). Carry
(funding/swap + short borrow) uses the same ``core.ledger.funding_payments`` generator as both
sibling adapters (§24). A partial scale-out exit is just another exit fill against the open
position — the ledger fold already handles partial closes (§4.6).

The one architectural difference from vectorbt: backtrader's actor loop fills orders at the
*next* bar open, so fill prices differ from vbt's bar-synchronous fills. That divergence is an
expected and documented parity tolerance (requirements §16); the exit *reason* is still
classified against the core ``exit_triggered`` semantics by the strategy's precomputed plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ube.adapters.backtrader_adapter.adapt_data import (
    bar_notional,
    bar_side,
    bar_timestamps_ns,
    step_timestamps,
    to_signal_frame,
)
from ube.adapters.backtrader_adapter.engine import run_backtrader
from ube.adapters.backtrader_adapter.exits import (
    resolve_atr_series_map,
    resolve_vol_for_sizing,
    validate_aux,
)
from ube.adapters.backtrader_adapter.instrument_map import build_instrument
from ube.adapters.backtrader_adapter.overrides import (
    DEFAULT_STARTING_BALANCE,
    FX_ALIASES,
    validate_overrides,
)
from ube.adapters.backtrader_adapter.strategy import (
    BacktraderStrategy,
    BtOrderRecord,
    BtRunContext,
)
from ube.adapters.base import EngineAdapter
from ube.core.config import BacktestConfig
from ube.core.cost import CostModel, fill_cost, resolve_cost_model, slipped_price
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, InvalidSignalError
from ube.core.instrument import allows_short
from ube.core.ledger import (
    EventLedger,
    EventType,
    FXSeries,
    LedgerEvent,
    funding_payments,
)
from ube.core.result import BacktestResult
from ube.core.signals import Signals, validate_long_only

__all__ = ["BacktraderAdapter"]

#: Event sort rank (mirrors the Nautilus/vectorbt fold — §4.6 ordering within a bar).
_KIND_RANK: dict[EventType, int] = {
    EventType.CASH_MOVEMENT: 0,
    EventType.SIGNAL_EVALUATED: 0,
    EventType.ORDER_SUBMITTED: 0,
    EventType.FILL: 1,
    EventType.COMMISSION: 2,
    EventType.POSITION_CHANGE: 3,
    EventType.FUNDING_PAYMENT: 4,
}

#: Margin initial requirement for leveraged classes (fraction of notional) — the reference
#: futures policy, applied as ``automargin = margin_init / leverage`` in the comminfo.
_MARGIN_INIT: float = 0.05


def _neutralize_shorts(signals: Signals) -> Signals:
    """Return ``signals`` with the short leg cleared for a long-only asset class (§4.5).

    ``crypto_spot`` cannot open a short — there is nothing to borrow — so
    ``short_entry``/``short_exit`` are meaningless for it. Zeroing the columns (rather
    than rejecting the series, which the strategy/actor layer gates elsewhere) makes the
    ignore-at-the-engine guarantee caller-independent: a caller that passes short rows
    (a flip encoding from ``from_target``, a provider that emits both sides, ...) still
    gets a strict long-only run, and the parallel long-side action of a flip (the
    ``long_exit``) is preserved so an open long still closes when its exit bar comes.
    Mirrors the vectorbt adapter's gate exactly.
    """
    dead = np.zeros(signals.n_bars, dtype=np.bool_)
    return Signals(
        long_entry=signals.long_entry,
        long_exit=signals.long_exit,
        short_entry=dead,
        short_exit=dead,
    )


def _fx_series_grid(
    overrides: Mapping[str, Any],
    ts_ns: np.ndarray,
    *,
    settlement: str,
    base_currency: str,
) -> dict[str, FXSeries]:
    """The FX rate series for multi-currency normalization (§4.6).

    Mirrors the vectorbt/nautilus adapters: the synthetic-rates family of overrides
    (``synthetic_rates`` / ``fixed_conversions`` / ``currency_fx`` / ``fx_rates``,
    first-declared wins) is turned into a constant-rate ``FXSeries`` on the bar grid, and a
    1:1 USD/USDT pair is auto-seeded when settlement and base differ and are both in
    ``{USD, USDT}`` so common XAUUSD/USDT runs don't raise ``FXRateUnavailableError`` (§4.7).
    """
    raw: dict[str, float] = {}
    for alias in FX_ALIASES:
        val = overrides.get(alias)
        if val is not None:
            raw = dict(val)
            break  # first-declared alias wins
    grid = np.asarray(ts_ns, dtype=np.int64)
    fx: dict[str, FXSeries] = {
        key: FXSeries(index=grid, rate=np.full(len(grid), rate))
        for key, rate in raw.items()
    }
    _sc = (settlement or "").upper()
    _bc = (base_currency or "").upper()
    if _sc and _bc and _sc != _bc and {_sc, _bc} == {"USD", "USDT"}:
        fwd = _sc + _bc
        inv = _bc + _sc
        ones = np.ones(len(grid))
        if fwd not in fx:
            fx[fwd] = FXSeries(index=grid, rate=ones)
        if inv not in fx:
            fx[inv] = FXSeries(index=grid, rate=np.ones(len(grid)))
    return fx


class BacktraderAdapter(EngineAdapter):
    """Adapter for the backtrader backtesting engine (§4.1, §4.2)."""

    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data: Mapping[str, Any] | None = None,
    ) -> BacktestResult:
        """Run a backtest via backtrader (§4.5).

        Args:
            data: The single-instrument OHLCV bars for the traded instrument.
            signals: The 4-column entry/exit signals on the same bar grid as ``data``.
            config: The full :class:`~ube.core.config.BacktestConfig`.
            aux_data: Optional derived-series map (§5.2) referenced by name from ATR exits
                and volatility_target sizing (``vol`` series must be supplied by the caller;
                volatility is never computed from the signal data bars, §6.3).

        Returns:
            The canonical :class:`~ube.core.result.BacktestResult`.
        """
        if not isinstance(data, MarketData):
            raise DataShapeError(
                "BacktraderAdapter.run expects a MarketData of bars; "
                f"got {type(data).__name__}"
            )
        if not isinstance(signals, Signals):
            raise InvalidSignalError(
                "BacktraderAdapter.run expects canonical Signals; "
                f"got {type(signals).__name__}"
            )
        if not isinstance(config, BacktestConfig):
            raise ConfigError(
                "BacktraderAdapter.run expects a BacktestConfig; "
                f"got {type(config).__name__}"
            )
        if signals.n_bars != data.n_bars:
            raise InvalidSignalError(
                f"signals cover {signals.n_bars} bars but market data has "
                f"{data.n_bars}; signal rows must be row-aligned with bars"
            )

        validate_long_only(signals, config.instrument.asset_class)

        # §4.5 long-only gate at the engine layer (mirrors the vectorbt adapter / nautilus
        # actor gates): a long-only asset class has nothing to borrow, so shorting is
        # undefined for it. The adapter ignores short signals entirely here — even when
        # the caller passes ``short_entry``/``short_exit`` rows, only long entries/exits
        # are acted on and backtrader can never open a short position.
        if not allows_short(config.instrument.asset_class):
            signals = _neutralize_shorts(signals)

        overrides = validate_overrides(config.engine_overrides)
        cost_model: CostModel | None = (
            config.cost_model
            if config.cost_model is not None
            else resolve_cost_model(config.instrument)
        )
        inst = build_instrument(config.instrument, overrides)
        instrument_id = config.instrument.symbol
        settlement = inst.settlement_currency
        starting_balance = float(
            overrides.get("starting_balance", DEFAULT_STARTING_BALANCE)
        )
        funding_interval_hours = inst.funding_interval_hours
        multiplier = inst.contract_multiplier
        account_type = str(overrides.get("account_type", "margin"))
        if account_type not in ("margin", "cash"):
            raise ConfigError(
                f"account_type override must be 'margin' or 'cash'; got {account_type!r}"
            )
        sizing = config.risk.sizing
        # §3.2 (mirroring nautilus/vectorbt): the effective leverage is the larger of the
        # sizing leverage and any explicit ``engine_overrides["leverage"]`` margin knob.
        sizing_lev = float(sizing.leverage)
        override_lev = float(overrides.get("leverage") or 0.0)
        lev = max(sizing_lev, override_lev) if override_lev > 0.0 else sizing_lev
        eff_leverage = 1.0 if account_type == "cash" else lev

        exits = config.risk.exit
        validate_aux(exits, aux_data, sizing=sizing, data=data)

        # Volatility-target sizing vol (§6.3): resolved from aux_data by name — never computed
        # from the signal data bars (mirroring the Nautilus actor and the vectorbt adapter).
        vol_arr: np.ndarray | None = None
        if sizing.kind == "volatility_target":
            vol_arr = resolve_vol_for_sizing(sizing, aux_data, data)
        atr_map = resolve_atr_series_map(exits, aux_data, data)
        slip = float(cost_model.slippage if cost_model is not None else 0.0)
        margin_account = _MARGIN_INIT if inst.margin else None

        ctx = BtRunContext(
            data=data,
            exits=tuple(exits),
            atr_map=atr_map,
            inst=inst,
            sizing=sizing,
            cost_model=cost_model,
            eff_leverage=eff_leverage,
            margin_account=margin_account,
            vol_arr=vol_arr,
            slip=slip,
            starting_balance=starting_balance,
        )
        frame = to_signal_frame(data, signals)
        strat = run_backtrader(
            frame,
            ctx,
            starting_cash=starting_balance * eff_leverage,
        )

        ledger = self._fold(
            strat,
            data=data,
            cost_model=cost_model,
            instrument_id=instrument_id,
            settlement=settlement,
            multiplier=multiplier,
            starting_balance=starting_balance,
            funding_interval_hours=funding_interval_hours,
            slip=slip,
        )

        # Multi-currency normalization (§4.6): build FXSeries from the synthetic-rates
        # family of overrides (mirroring the vectorbt/nautilus adapters) and pass through to
        # ``BacktestResult.from_ledger`` so ``equity_curve`` / ``trade_table`` can convert
        # the instrument's settlement currency into ``base_currency``.
        fx_rates = _fx_series_grid(
            overrides,
            bar_timestamps_ns(data),
            settlement=settlement,
            base_currency=str(config.base_currency or ""),
        )

        return BacktestResult.from_ledger(
            ledger,
            config,
            market_data={instrument_id: data},
            instruments={instrument_id: config.instrument},
            fx_rates=fx_rates or None,
        )

    # ------------------------------------------------------------------
    # Ledger fold (§4.6) — builds the canonical event ledger.
    # ------------------------------------------------------------------

    def _fold(
        self,
        strat: BacktraderStrategy,
        *,
        data: MarketData,
        cost_model: CostModel | None,
        instrument_id: str,
        settlement: str,
        multiplier: float,
        starting_balance: float,
        funding_interval_hours: float,
        slip: float,
    ) -> EventLedger:
        """Fold the strategy's signal/order records into a canonical append-only ledger (§4.6)."""
        bar_ts = bar_timestamps_ns(data)

        sorted_events: list[tuple[int, int, int, LedgerEvent]] = []
        seq = 0
        order_seq = 0

        def _order_submitted(ts: int, side: int, quantity: float) -> None:
            nonlocal order_seq
            order_seq += 1
            _add(
                LedgerEvent(
                    EventType.ORDER_SUBMITTED,
                    int(ts),
                    instrument_id,
                    order_id=f"bt-{order_seq}",
                    side=side,
                    quantity=quantity,
                ),
                int(ts),
            )

        def _add(event: LedgerEvent, ts: int) -> None:
            nonlocal seq
            sorted_events.append((int(ts), _KIND_RANK[event.event_type], seq, event))
            seq += 1

        # Starting balance booked as a cash inflow at the first bar boundary (§4.6).
        _add(
            LedgerEvent(
                EventType.CASH_MOVEMENT,
                int(bar_ts[0]),
                instrument_id,
                amount=starting_balance,
                currency=settlement,
            ),
            int(bar_ts[0]),
        )

        # Signal evaluations (§4.6): one per action the strategy took, on the bar it took it.
        for srec in strat.signal_records:
            _add(
                LedgerEvent(
                    EventType.SIGNAL_EVALUATED,
                    int(bar_ts[srec.bar]),
                    instrument_id,
                    action=srec.action,
                    side=srec.side,
                ),
                int(bar_ts[srec.bar]),
            )

        # Orders in strict submission order (chronological for market orders — see
        # strategy docstring; a flip places the exit first, then the entry, so the fold
        # reproduces the exact fill order).
        net: float = 0.0
        for orec in strat.order_records:
            _order_submitted(int(bar_ts[orec.submit_bar]), orec.side, orec.quantity)
            net += orec.side * orec.quantity
            self._fold_fill(
                bar_ts,
                orec,
                _add,
                position_after=net,
                cost_model=cost_model,
                instrument_id=instrument_id,
                settlement=settlement,
                multiplier=multiplier,
                slip=slip,
            )

        # Carry (funding/swap + short borrow) from the core cost model (§24).
        funding_rate = float(cost_model.funding) if cost_model is not None else 0.0
        borrow_rate = float(cost_model.borrow) if cost_model is not None else 0.0
        if funding_rate != 0.0 or borrow_rate != 0.0:
            position_change = [
                e
                for _t, _r, _s, e in sorted_events
                if e.event_type is EventType.POSITION_CHANGE
            ]
            pc_ts = np.asarray(
                [int(e.timestamp) for e in position_change], dtype=np.int64
            )
            pc_val = np.asarray(
                [
                    float(e.position_after if e.position_after is not None else 0.0)
                    for e in position_change
                ],
                dtype=np.float64,
            )
            step_ts, step_value = step_timestamps(pc_ts, pc_val)
            # §4.5/§24: the schedule travels with the instrument metadata, mirroring the
            # nautilus adapter's ``resolve_funding_interval_hours``.
            interval_ns = int(funding_interval_hours * 3600 * 1_000_000_000)
            for event in funding_payments(
                instrument_id=instrument_id,
                timestamps=bar_ts,
                notional=bar_notional(data, step_ts, step_value, multiplier),
                side=bar_side(data, step_ts, step_value),
                funding_rate=funding_rate,
                borrow_rate=borrow_rate,
                currency=settlement,
                interval_ns=interval_ns,
            ):
                _add(event, int(event.timestamp))

        sorted_events.sort(key=lambda item: (item[0], item[1], item[2]))
        return EventLedger(event for _ts, _rank, _seq, event in sorted_events)

    @staticmethod
    def _fold_fill(
        bar_ts: np.ndarray,
        rec: BtOrderRecord,
        _add: Any,
        *,
        position_after: float,
        cost_model: CostModel | None,
        instrument_id: str,
        settlement: str,
        multiplier: float,
        slip: float,
    ) -> None:
        """One fill's events: cash leg, fill (with reason), commission, position change (§4.6)."""
        ts = int(bar_ts[rec.fill_bar])
        price = float(slipped_price(rec.price, rec.side, slip))
        notional = rec.quantity * price * multiplier

        _add(
            LedgerEvent(
                EventType.CASH_MOVEMENT,
                ts,
                instrument_id,
                amount=-rec.side * notional,
                currency=settlement,
            ),
            ts,
        )
        _add(
            LedgerEvent(
                EventType.FILL,
                ts,
                instrument_id,
                side=rec.side,
                quantity=rec.quantity,
                price=price,
                exit_reason=rec.reason,
            ),
            ts,
        )
        # Commission via the core ``fill_cost`` (§8): on slipped notional so the ledger is
        # fully self-consistent with the price-level slippage model.
        if rec.quantity > 0.0 and cost_model is not None:
            fee = float(fill_cost(cost_model, notional=notional))
            if fee != 0.0:
                _add(
                    LedgerEvent(
                        EventType.COMMISSION,
                        ts,
                        instrument_id,
                        amount=fee,
                        currency=settlement,
                    ),
                    ts,
                )
        _add(
            LedgerEvent(
                EventType.POSITION_CHANGE,
                ts,
                instrument_id,
                position_after=position_after,
            ),
            ts,
        )
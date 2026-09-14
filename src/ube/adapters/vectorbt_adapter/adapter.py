"""vectorbt engine adapter (§4.1, §4.2).

This module is the thin orchestration layer: it validates the contract, translates the inputs
via :mod:`~ube.adapters.vectorbt_adapter.adapt_data`, computes exit stop primitives via
:mod:`~ube.adapters.vectorbt_adapter.exits`, runs the vectorized portfolio via
:mod:`~ube.adapters.vectorbt_adapter.engine`, and folds the resulting trade records into the
canonical :class:`~ube.core.ledger.EventLedger` (§4.6). Carry (funding) uses the same
``core.ledger.funding_payments`` generator as the Nautilus adapter, so both engines share one
cost-event stream (§24).

The single architectural difference from Nautilus: vectorbt exits on its own stop primitives
(``sl_stop`` / ``tp_stop`` / ``sl_trail``), parameterised by per-bar fractions derived from the
core exit levels — not an exact replica of the event-driven ratchet. The exit *reason* is still
classified against the core ``exit_triggered`` semantics so a trade is labelled consistently
with Nautilus (divergence in the exact fill price is expected and documented in the parity
tolerance, requirements §16).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ube.adapters.base import EngineAdapter
from ube.adapters.vectorbt_adapter.adapt_data import (
    bar_index,
    bar_notional,
    bar_side,
    bar_timestamps_ns,
    step_timestamps,
    to_vbt_inputs,
)
from ube.adapters.vectorbt_adapter.engine import build_portfolio, vbt
from ube.adapters.vectorbt_adapter.exits import (
    apply_time_exits,
    classify_exit_reason,
    exit_stop_params,
    resolve_vol_for_sizing,
    validate_aux,
)
from ube.adapters.vectorbt_adapter.instrument_map import (
    VbtInstrument,
    build_instrument,
    floor_to_increment,
)
from ube.adapters.vectorbt_adapter.overrides import (
    DEFAULT_STARTING_BALANCE,
    FX_ALIASES,
    validate_overrides,
)
from ube.core.config import BacktestConfig
from ube.core.cost import CostModel, fill_cost, resolve_cost_model, slipped_price
from ube.core.data import MarketData
from ube.core.errors import (
    ConfigError,
    DataShapeError,
    EngineError,
    InvalidSignalError,
)
from ube.core.ledger import EventLedger, EventType, FXSeries, LedgerEvent, funding_payments
from ube.core.result import BacktestResult
from ube.core.risk.sizing import _entry_fee_rate, size_position
from ube.core.signals import Signals, validate_long_only

__all__ = ["VectorbtAdapter"]

#: Event sort rank (mirrors the Nautilus fold — §4.6 ordering within a bar).
_KIND_RANK: dict[EventType, int] = {
    EventType.CASH_MOVEMENT: 0,
    EventType.SIGNAL_EVALUATED: 0,
    EventType.ORDER_SUBMITTED: 0,
    EventType.FILL: 1,
    EventType.COMMISSION: 2,
    EventType.POSITION_CHANGE: 3,
    EventType.FUNDING_PAYMENT: 4,
}


def _target_quantity(
    sizing: Any,
    price: float,
    equity: float,
    vol: float | None,
    cost_model: CostModel | None,
    vbt_inst: VbtInstrument,
    leverage: float = 1.0,
) -> float:
    """Target units for one entry via the core sizer (§6.3).

    Mirrors the Nautilus actor: the sizing ``leverage`` is applied as an exposure multiplier
    on the allocated capital (``leveraged_capital = capital * leverage`` -> ``qty = notional /
    price``), so a leveraged position in vectorbt matches Nautilus for the same config. The
    ``cost_model`` is passed through so fee-aware sizers (``all_in`` / ``equal_weight``) reserve
    the entry fee up front (§7.1) — the same fix applied to the Nautilus actor — instead of
    spending 100% of equity on units and letting the engine's commission push the account
    negative.

    The ``leverage`` parameter is the effective leverage after applying any account-type
    override (cash accounts force ``leverage=1.0``); it replaces the sizing model's own
    ``leverage`` field for this calculation. The resulting quantity is always *floored* to
    the asset-class lot increment — never rounded up — mirroring the Nautilus actor, where an
    up-rounded quantity can exceed the size the affordability guard verified by up to one
    lot's notional + fee, pushing the account negative (§7.1). ``equal_weight`` sizes against
    ``n=1`` (the single-instrument run), matching the actor's call.
    """
    leveraged = equity * leverage
    kwargs: dict[str, Any] = dict(capital=leveraged, price=price, cost_model=cost_model)
    if sizing.kind == "equal_weight":
        kwargs["n"] = 1
    if vol is not None:
        kwargs["vol"] = vol
    qty = float(size_position(sizing, **kwargs))
    return float(floor_to_increment(qty, vbt_inst.size_increment))


def _net_pnl_per_unit(
    side: int,
    entry_price: float,
    exit_price: float,
    cost_model: CostModel | None,
    multiplier: float,
) -> float:
    """Realised net PnL per single unit at slipped fill prices (§6.3/§8).

    This replaces the per-unit ``trade["PnL"]`` from vectorbt's nominal-price first pass so
    the running equity used for sizing reflects the actual cost model (slipped fills +
    commission), matching how the Nautilus actor sizes off the real-time portfolio balance.
    ``fill_cost`` is used for the commission leg so the fee math has a single source of truth.
    """
    slip = float(cost_model.slippage if cost_model is not None else 0.0)
    entry_f = float(slipped_price(entry_price, side, slip))
    exit_f = float(slipped_price(exit_price, -side, slip))
    gross = side * (exit_f - entry_f) * multiplier
    entry_not = entry_f * 1.0 * multiplier
    exit_not = exit_f * 1.0 * multiplier
    if cost_model is not None:
        entry_fee = float(fill_cost(cost_model, notional=entry_not))
        exit_fee = float(fill_cost(cost_model, notional=exit_not))
    else:
        entry_fee = 0.0
        exit_fee = 0.0
    return gross - entry_fee - exit_fee


def _build_size_series(
    records: pd.DataFrame,
    data: MarketData,
    sizing: Any,
    starting_balance: float,
    vol_arr: np.ndarray | None,
    cost_model: CostModel | None,
    vbt_inst: VbtInstrument,
    *,
    leverage: float = 1.0,
    settlement: str = "USD",
) -> pd.Series:
    """Per-bar ``size`` (amount) array: target qty at entry bars, same qty at exit bars.

    Mirrors the reference iterative sizing loop but uses the core sizer and the running account
    equity before each entry (vectorbt cannot size off equity by itself). The equity tracks the
    slipped PnL (§8), and a market-rejection check surfaces an ``EngineError`` when a
    sized entry cannot be funded (matching the Nautilus venue rejection on insufficient
    margin/cash).
    """
    ts = bar_timestamps_ns(data)
    n = data.n_bars
    size = np.full(n, np.nan, dtype=np.float64)
    running = float(starting_balance)
    multiplier = vbt_inst.contract_multiplier
    for _, trade in records.iterrows():
        entry_bar = bar_index(ts, pd.Timestamp(trade["Entry Timestamp"]))
        exit_bar = bar_index(ts, pd.Timestamp(trade["Exit Timestamp"]))
        direction = str(trade["Direction"])
        side = 1 if direction == "Long" else -1
        raw_entry = float(trade["Avg Entry Price"])
        raw_exit = float(trade["Avg Exit Price"])
        slip = float(cost_model.slippage if cost_model is not None else 0.0)
        slipped_entry = float(slipped_price(raw_entry, side, slip))
        vol = float(vol_arr[entry_bar]) if vol_arr is not None and 0 <= entry_bar < n else None
        qty = _target_quantity(sizing, slipped_entry, running, vol, cost_model, vbt_inst, leverage)
        if qty <= 0.0:
            if 0 <= exit_bar < n:
                size[exit_bar] = 0.0
            continue
        # Market-rejection surfacing (§4.5): mirroring the Nautilus actor, re-verify the
        # FINAL venue-quantized quantity against the leveraged capacity only when the entry
        # carries a fee (fee_rate > 0) — a zero-fee run tolerates a sub-lot notional overhang
        # from lot rounding (the §7.1 residual accepted by the actor). ``volatility_target``
        # sizing is exempt — it is a volatility-budget model that intentionally sizes beyond
        # unleveraged equity.
        if sizing.kind != "volatility_target" and _entry_fee_rate(cost_model) > 0.0:
            unit_cash = qty * slipped_entry
            entry_fee = float(fill_cost(cost_model, notional=unit_cash)) if cost_model else 0.0
            required = unit_cash + entry_fee
            capacity = running * leverage
            if required > capacity * (1.0 + 1e-9):
                raise EngineError(
                    f"market order rejected by the venue: insufficient funds at bar "
                    f"{entry_bar} — requires {required:.6g} {settlement}, "
                    f"available {capacity:.6g} {settlement}"
                )
        if 0 <= entry_bar < n:
            size[entry_bar] = qty
        if 0 <= exit_bar < n:
            size[exit_bar] = qty
        # Realised net PnL per single unit at slipped fill prices (§8), replacing
        # ``trade["PnL"]`` which is the nominal-price vectorbt figure.
        pnl = _net_pnl_per_unit(side, raw_entry, raw_exit, cost_model, multiplier)
        running += pnl * qty
    return pd.Series(size, index=data.timestamps)


def _fx_series_grid(
    overrides: Mapping[str, Any],
    ts_ns: np.ndarray,
    *,
    settlement: str = "",
    base_currency: str = "",
) -> dict[str, FXSeries]:
    """Turn a parsed ``{pair: rate}`` dict (from ``validate_overrides``) into FXSeries on the
    bar timestamp grid.  First-declared alias wins (mirroring the nautilus ``or`` chain: only
    the first of ``synthetic_rates``/``fixed_conversions``/``currency_fx``/``fx_rates`` is
    consumed).

    A 1:1 USD/USDT pair is auto-seeded when settlement and base differ and are both in
    ``{USD, USDT}``, mirroring the nautilus adapter so runs don't raise
    ``FXRateUnavailableError`` for the common XAUUSD/USDT case (§4.7).
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
    # USD/USDT 1:1 auto-seed (mirrors nautilus adapter post-run block).
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


class VectorbtAdapter(EngineAdapter):
    """Adapter for the vectorbt backtesting engine (§4.1, §4.2)."""

    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data: Mapping[str, Any] | None = None,
    ) -> BacktestResult:
        """Run a backtest via vectorbt (§4.5).

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
        if vbt is None:  # pragma: no cover - depends on the optional package
            raise EngineError(
                "vectorbt is not installed; install it (e.g. `pip install vectorbt`) "
                "to use the vectorbt engine"
            )
        if not isinstance(data, MarketData):
            raise DataShapeError(
                "VectorbtAdapter.run expects a MarketData of bars; "
                f"got {type(data).__name__}"
            )
        if not isinstance(signals, Signals):
            raise InvalidSignalError(
                "VectorbtAdapter.run expects canonical Signals; "
                f"got {type(signals).__name__}"
            )
        if not isinstance(config, BacktestConfig):
            raise ConfigError(
                "VectorbtAdapter.run expects a BacktestConfig; "
                f"got {type(config).__name__}"
            )
        if signals.n_bars != data.n_bars:
            raise InvalidSignalError(
                f"signals cover {signals.n_bars} bars but market data has "
                f"{data.n_bars}; signal rows must be row-aligned with bars"
            )

        validate_long_only(signals, config.instrument.asset_class)

        overrides = validate_overrides(config.engine_overrides)
        cost_model: CostModel = (
            config.cost_model
            if config.cost_model is not None
            else resolve_cost_model(config.instrument)
        )
        vbt_inst = build_instrument(config.instrument, overrides)
        instrument_id = config.instrument.symbol
        settlement = vbt_inst.settlement_currency
        starting_balance = float(overrides.get("starting_balance", DEFAULT_STARTING_BALANCE))
        funding_interval_hours = vbt_inst.funding_interval_hours
        multiplier = vbt_inst.contract_multiplier
        account_type = str(overrides.get("account_type", "margin"))
        if account_type not in ("margin", "cash"):
            raise ConfigError(
                f"account_type override must be 'margin' or 'cash'; got {account_type!r}"
            )
        sizing = config.risk.sizing
        # §3.2 (mirroring nautilus): the effective leverage is the larger of the sizing
        # leverage and any explicit ``engine_overrides["leverage"]`` margin knob.
        sizing_lev = float(sizing.leverage)
        override_lev = float(overrides.get("leverage") or 0.0)
        lev = max(sizing_lev, override_lev) if override_lev > 0.0 else sizing_lev
        eff_leverage = 1.0 if account_type == "cash" else lev

        exits = config.risk.exit
        validate_aux(exits, aux_data, sizing=sizing, data=data)

        # Volatility-target sizing vol (§6.3): resolved from ``aux_data`` via the ``vol``
        # series name, never from the signal data bars (mirroring the Nautilus actor).
        vol_arr: np.ndarray | None = None
        if sizing.kind == "volatility_target":
            vol_arr = resolve_vol_for_sizing(sizing, aux_data, data)

        # --- Translate to vectorbt inputs + exit primitives -----------------
        inputs = to_vbt_inputs(data, signals)
        sl_stop, tp_stop, sl_trail = exit_stop_params(exits, data, aux_data)
        # Commission only: slippage is applied at the fill price level in the ledger fold,
        # not as a fee fraction in vectorbt (§8 — parity with the Nautilus actor).
        fees = float(cost_model.commission)
        inputs = apply_time_exits(inputs, exits, data)

        # Two-pass sizing: nominal 1-unit run to learn entry prices, then core-sized run.
        pf = build_portfolio(
            inputs,
            init_cash=starting_balance,
            fees=fees,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            size=1.0,
        )
        records = pf.trades.records_readable
        if len(records):
            size_series = _build_size_series(
                records,
                data,
                sizing,
                starting_balance,
                vol_arr,
                cost_model,
                vbt_inst,
                leverage=eff_leverage,
                settlement=settlement,
            )
            pf = build_portfolio(
                inputs,
                init_cash=starting_balance,
                fees=fees,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                sl_trail=sl_trail,
                size=size_series,
            )
            records = pf.trades.records_readable

        ledger = self._fold(
            pf=pf,
            records=records,
            data=data,
            signals=signals,
            cost_model=cost_model,
            exits=exits,
            aux_data=aux_data,
            instrument_id=instrument_id,
            settlement=settlement,
            multiplier=multiplier,
            starting_balance=starting_balance,
            funding_interval_hours=funding_interval_hours,
        )

        # Multi-currency normalization (§4.6): build FXSeries from the synthetic-rates
        # family of overrides (mirroring the nautilus adapter) and pass through to
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
        pf: Any,
        records: pd.DataFrame,
        data: MarketData,
        signals: Signals,
        cost_model: CostModel,
        exits: tuple[Any, ...],
        aux_data: Mapping[str, Any] | None,
        instrument_id: str,
        settlement: str,
        multiplier: float,
        starting_balance: float,
        funding_interval_hours: float,
    ) -> EventLedger:
        """Fold vectorbt trades into a canonical append-only ledger (§4.6)."""
        bar_ts = bar_timestamps_ns(data)
        slip = float(cost_model.slippage if cost_model is not None else 0.0)

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
                    order_id=f"vbt-{order_seq}",
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

        net: float = 0.0
        for _, trade in records.iterrows():
            entry_dt = pd.Timestamp(trade["Entry Timestamp"])
            exit_dt = pd.Timestamp(trade["Exit Timestamp"])
            entry_bar = bar_index(bar_ts, entry_dt)
            exit_bar = bar_index(bar_ts, exit_dt)
            raw_entry = float(trade["Avg Entry Price"])
            raw_exit = float(trade["Avg Exit Price"])
            size = abs(float(trade["Size"]))
            direction = str(trade["Direction"])
            side = 1 if direction == "Long" else -1
            # Slipped fill prices (§8): entry fill at ``price * (1 + side*slip)``, exit
            # fill at ``price * (1 - side*slip)`` — the fill the venue would actually book.
            entry_price = float(slipped_price(raw_entry, side, slip))
            exit_price = float(slipped_price(raw_exit, -side, slip))

            exit_reason = classify_exit_reason(
                exits, data, side, entry_price, entry_bar, exit_bar, aux_data, signals
            )

            # Signal evaluation recorded at the entry bar (§6.1): holds are never
            # emitted, and only real entries reach the ledger — matching the Nautilus fold.
            if bool(signals.long_entry[entry_bar]):
                eval_action = "long_entry"
            elif bool(signals.short_entry[entry_bar]):
                eval_action = "short_entry"
            else:
                eval_action = None
            if eval_action is not None:
                _add(
                    LedgerEvent(
                        EventType.SIGNAL_EVALUATED,
                        int(bar_ts[entry_bar]),
                        instrument_id,
                        action=eval_action,
                    ),
                    int(bar_ts[entry_bar]),
                )

            # Entry fill + cash leg + position change.
            entry_notional = size * entry_price * multiplier
            _add(
                LedgerEvent(
                    EventType.CASH_MOVEMENT,
                    int(bar_ts[entry_bar]),
                    instrument_id,
                    amount=-side * entry_notional,
                    currency=settlement,
                ),
                int(bar_ts[entry_bar]),
            )
            _order_submitted(int(bar_ts[entry_bar]), side, size)
            _add(
                LedgerEvent(
                    EventType.FILL,
                    int(bar_ts[entry_bar]),
                    instrument_id,
                    side=side,
                    quantity=size,
                    price=entry_price,
                ),
                int(bar_ts[entry_bar]),
            )
            net += side * size
            _add(
                LedgerEvent(
                    EventType.POSITION_CHANGE,
                    int(bar_ts[entry_bar]),
                    instrument_id,
                    position_after=net,
                ),
                int(bar_ts[entry_bar]),
            )
            # Commission via the core ``fill_cost`` (§8): on slipped notional so the ledger
            # is fully self-consistent with the price-level slippage model.
            if size > 0.0 and cost_model is not None:
                entry_fee = float(fill_cost(cost_model, notional=entry_notional))
                if entry_fee != 0.0:
                    _add(
                        LedgerEvent(
                            EventType.COMMISSION,
                            int(bar_ts[entry_bar]),
                            instrument_id,
                            amount=entry_fee,
                            currency=settlement,
                        ),
                        int(bar_ts[entry_bar]),
                    )

            # Exit signal-evaluated row (§4.6): mirrors the nautilus actor where every
            # exit (signal bar or risk bar) is stamped with a per-bar action.
            if exit_reason == "signal":
                exit_action = "long_exit" if side == 1 else "short_exit"
            else:
                exit_action = f"exit_{exit_reason}"
            _add(
                LedgerEvent(
                    EventType.SIGNAL_EVALUATED,
                    int(bar_ts[exit_bar]),
                    instrument_id,
                    action=exit_action,
                ),
                int(bar_ts[exit_bar]),
            )

            # Exit ORDER_SUBMITTED (§4.6): mirrors the nautilus actor, which records one
            # per order placed — entry and close/exit alike.
            _order_submitted(int(bar_ts[exit_bar]), -side, size)

            # Exit fill (with reason) + cash leg + commission + position change.
            exit_notional = size * exit_price * multiplier
            _add(
                LedgerEvent(
                    EventType.CASH_MOVEMENT,
                    int(bar_ts[exit_bar]),
                    instrument_id,
                    amount=side * exit_notional,
                    currency=settlement,
                ),
                int(bar_ts[exit_bar]),
            )
            _add(
                LedgerEvent(
                    EventType.FILL,
                    int(bar_ts[exit_bar]),
                    instrument_id,
                    side=-side,
                    quantity=size,
                    price=exit_price,
                    exit_reason=exit_reason,
                ),
                int(bar_ts[exit_bar]),
            )
            if size > 0.0 and cost_model is not None:
                exit_fee = float(fill_cost(cost_model, notional=exit_notional))
                if exit_fee != 0.0:
                    _add(
                        LedgerEvent(
                            EventType.COMMISSION,
                            int(bar_ts[exit_bar]),
                            instrument_id,
                            amount=exit_fee,
                            currency=settlement,
                        ),
                        int(bar_ts[exit_bar]),
                    )
            net += (-side) * size
            _add(
                LedgerEvent(
                    EventType.POSITION_CHANGE,
                    int(bar_ts[exit_bar]),
                    instrument_id,
                    position_after=net,
                ),
                int(bar_ts[exit_bar]),
            )

        # Carry (funding/swap + short borrow) from the core cost model (§24).
        funding_rate = float(cost_model.funding)
        borrow_rate = float(cost_model.borrow)
        if funding_rate != 0.0 or borrow_rate != 0.0:
            position_change = [
                e
                for _t, _r, _s, e in sorted_events
                if e.event_type is EventType.POSITION_CHANGE
            ]
            pc_ts = np.asarray([int(e.timestamp) for e in position_change], dtype=np.int64)
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
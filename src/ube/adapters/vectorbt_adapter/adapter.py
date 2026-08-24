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
    classify_exit_reason,
    exit_stop_params,
    validate_aux,
)
from ube.adapters.vectorbt_adapter.overrides import (
    DEFAULT_FUNDING_INTERVAL_HOURS,
    DEFAULT_STARTING_BALANCE,
    validate_overrides,
)
from ube.core.config import BacktestConfig
from ube.core.cost import CostModel, resolve_cost_model
from ube.core.data import MarketData
from ube.core.errors import (
    ConfigError,
    DataShapeError,
    EngineError,
    InvalidSignalError,
)
from ube.core.ledger import EventLedger, EventType, LedgerEvent, funding_payments
from ube.core.result import BacktestResult
from ube.core.risk.exits import atr
from ube.core.risk.sizing import size_position
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
    sizing: Any, price: float, equity: float, vol: float | None
) -> float:
    """Target units for one entry via the core sizer (§6.3).

    Mirrors the Nautilus actor: the sizing ``leverage`` is applied as an exposure multiplier
    on the allocated capital (``leveraged_capital = capital * leverage`` -> ``qty = notional /
    price``), so a leveraged position in vectorbt matches Nautilus for the same config.
    """
    leveraged = equity * getattr(sizing, "leverage", 1.0)
    if vol is None:
        return float(size_position(sizing, capital=leveraged, price=price))
    return float(size_position(sizing, capital=leveraged, price=price, vol=vol))


def _build_size_series(
    records: pd.DataFrame,
    data: MarketData,
    sizing: Any,
    starting_balance: float,
    vol_arr: np.ndarray | None,
) -> pd.Series:
    """Per-bar ``size`` (amount) array: target qty at entry bars, same qty at exit bars.

    Mirrors the reference iterative sizing loop but uses the core sizer and the running account
    equity before each entry (vectorbt cannot size off equity by itself).
    """
    ts = bar_timestamps_ns(data)
    n = data.n_bars
    size = np.full(n, np.nan, dtype=np.float64)
    running = float(starting_balance)
    for _, trade in records.iterrows():
        entry_bar = bar_index(ts, pd.Timestamp(trade["Entry Timestamp"]))
        exit_bar = bar_index(ts, pd.Timestamp(trade["Exit Timestamp"]))
        entry_price = float(trade["Avg Entry Price"])
        vol = float(vol_arr[entry_bar]) if vol_arr is not None and 0 <= entry_bar < n else None
        qty = _target_quantity(sizing, entry_price, running, vol)
        if 0 <= entry_bar < n:
            size[entry_bar] = qty
        if 0 <= exit_bar < n:
            size[exit_bar] = qty
        pnl = trade.get("PnL", 0.0)
        # The first (size=1.0) pass records PnL per single unit of the asset, so the
        # realized equity impact scales linearly with the *target* quantity actually
        # sized in the second pass. Multiply by qty so running equity tracks the real
        # position PnL (matches Nautilus' iterative sizing).
        running += float(pnl if pnl is not None else 0.0) * qty
    return pd.Series(size, index=data.timestamps)


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
            aux_data: Optional derived-series map (§5.2) referenced by name from ATR exits.

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
        instrument_id = config.instrument.symbol
        settlement = config.instrument.settlement_currency or "USD"
        starting_balance = float(overrides.get("starting_balance", DEFAULT_STARTING_BALANCE))
        funding_interval_hours = float(
            overrides.get("funding_interval_hours", DEFAULT_FUNDING_INTERVAL_HOURS)
        )
        multiplier = (
            1.0
            if config.instrument.contract_multiplier is None
            else float(config.instrument.contract_multiplier)
        )

        exits = config.risk.exit
        validate_aux(exits, aux_data)

        # Volatility-target sizing needs a per-bar vol estimate (atr/close).
        vol_arr: np.ndarray | None = None
        if config.risk.sizing.kind == "volatility_target":
            vol_arr = atr(data) / np.where(data.close > 0, data.close, np.nan)
            vol_arr = np.nan_to_num(vol_arr, nan=0.0)

        # --- Translate to vectorbt inputs + exit primitives -----------------
        inputs = to_vbt_inputs(data, signals)
        sl_stop, tp_stop, sl_trail = exit_stop_params(exits, data, aux_data)
        fees = float(cost_model.commission + cost_model.slippage)

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
                records, data, config.risk.sizing, starting_balance, vol_arr
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

        return BacktestResult.from_ledger(
            ledger,
            config,
            market_data={instrument_id: data},
            instruments={instrument_id: config.instrument},
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
            entry_price = float(trade["Avg Entry Price"])
            exit_price = float(trade["Avg Exit Price"])
            size = abs(float(trade["Size"]))
            direction = str(trade["Direction"])
            side = 1 if direction == "Long" else -1
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
            entry_fees = trade.get("Entry Fees", 0.0)
            entry_fees = float(entry_fees if entry_fees is not None else 0.0)
            if entry_fees != 0.0:
                _add(
                    LedgerEvent(
                        EventType.COMMISSION,
                        int(bar_ts[entry_bar]),
                        instrument_id,
                        amount=entry_fees,
                        currency=settlement,
                    ),
                    int(bar_ts[entry_bar]),
                )

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
            exit_fees = trade.get("Exit Fees", 0.0)
            exit_fees = float(exit_fees if exit_fees is not None else 0.0)
            if exit_fees != 0.0:
                _add(
                    LedgerEvent(
                        EventType.COMMISSION,
                        int(bar_ts[exit_bar]),
                        instrument_id,
                        amount=exit_fees,
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

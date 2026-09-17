"""Backtrader adapter tests — one file per adapter, append-only.

This is the single test file for the backtrader adapter and its shared base-contract
dependencies. It mirrors the structure of ``test_vectorbt_adapter.py`` but exercises the
*event-driven* backtrader engine (§4.1, §4.2): engine registration, the backtrader
``engine_overrides`` registry, the MarketData/Signals translation helpers, the ATR/aux exit
translation (per-entry trigger plans), the cerebro execution loop, and the canonical ledger
fold (§4.6).

Unlike vectorbt, backtrader fills market orders at the *next* bar's open, so the tests compute
fill prices from the raw bar opens (self-consistent with the data) rather than the
bar-synchronous vbt figures. The exit-reason semantics are shared with the siblings: the exit
plan is classified against the core ``exit_triggered`` rules (§8), and the fold books carry
through the shared ``funding_payments`` generator (§24).

It never touches the Nautilus/vectorbt adapters or their tests — every assertion here runs
against :class:`~ube.adapters.backtrader_adapter.adapter.BacktraderAdapter` only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import ube
from ube.adapters import get_engine
from ube.adapters.backtrader_adapter.adapt_data import (
    bar_step_grid,
    bar_timestamps_ns,
    to_signal_frame,
)
from ube.adapters.backtrader_adapter.exits import (
    atr_from_aux,
    build_exit_plan,
    exit_reason_label,
    validate_aux,
)
from ube.adapters.backtrader_adapter.instrument_map import build_instrument, floor_to_increment
from ube.adapters.backtrader_adapter.overrides import (
    DEFAULT_FUNDING_INTERVAL_HOURS,
    DEFAULT_STARTING_BALANCE,
    parse_synthetic_rates,
    validate_overrides,
)
from ube.core.config import BacktestConfig
from ube.core.cost import CostModel
from ube.core.errors import ConfigError, DataShapeError, InvalidSignalError
from ube.core.experiment_log import ExperimentLog
from ube.core.ledger import EventType
from ube.core.result import BacktestResult
from ube.core.risk import RiskConfig, SizeModel
from ube.core.risk.exits import ATRStop, ChandelierExit, TakeProfit, TimeExit, TrailingStop
from ube.core.signals import from_target
from ube.testing.synthetic import PRESETS, synthetic_bars

try:
    import backtrader  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    backtrader = None

# The end-to-end path imports the strategy/engine/adapter modules that require backtrader.
if backtrader is not None:
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter
    from ube.adapters.backtrader_adapter.engine import run_backtrader
    from ube.adapters.backtrader_adapter.strategy import BtRunContext
else:  # pragma: no cover - exercised only without backtrader
    BacktraderAdapter = None  # type: ignore[assignment,misc]
    run_backtrader = None  # type: ignore[assignment,misc]
    BtRunContext = None  # type: ignore[assignment,misc]

_requires_bt = pytest.mark.skipif(
    backtrader is None, reason="backtrader not installed"
)


def _fills(result):
    return [e for e in result.ledger if e.event_type is EventType.FILL]


def _ledger_events(result, event_type):
    return [e for e in result.ledger if e.event_type is event_type]


# ---------------------------------------------------------------------------
# Engine registration (§4.1).
# ---------------------------------------------------------------------------


def test_backtrader_adapter_is_registrable_under_canonical_name():
    from ube.adapters import register_engine

    register_engine("backtrader", BacktraderAdapter)
    assert get_engine("backtrader") is BacktraderAdapter


def test_ensure_builtin_engines_registered_registers_backtrader():
    ube.ensure_builtin_engines_registered()
    assert get_engine("backtrader") is BacktraderAdapter


# ---------------------------------------------------------------------------
# backtrader engine_overrides registry (§4.3, §7.2).
# ---------------------------------------------------------------------------


def test_validate_overrides_accepts_none():
    assert validate_overrides(None) == {}


def test_validate_overrides_returns_fresh_copy():
    raw = {"starting_balance": 1000.0}
    validated = validate_overrides(raw)
    validated["starting_balance"] = 999.0
    assert raw["starting_balance"] == 1000.0


def test_validate_overrides_accepts_valid_keys():
    validated = validate_overrides(
        {"starting_balance": 50000.0, "funding_interval_hours": 4.0}
    )
    assert validated["starting_balance"] == 50000.0
    assert validated["funding_interval_hours"] == 4.0


def test_validate_overrides_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown engine override"):
        validate_overrides({"bogus": 1.0})


def test_validate_overrides_rejects_non_positive_balance():
    with pytest.raises(ConfigError, match="starting_balance"):
        validate_overrides({"starting_balance": 0.0})


def test_validate_overrides_rejects_non_positive_funding_interval():
    with pytest.raises(ConfigError, match="funding_interval_hours"):
        validate_overrides({"funding_interval_hours": -1.0})


def test_defaults_are_exported():
    assert DEFAULT_STARTING_BALANCE > 0.0
    assert DEFAULT_FUNDING_INTERVAL_HOURS > 0.0


def test_validate_overrides_accepts_extended_schema():
    validated = validate_overrides(
        {
            "starting_balance": 50000.0,
            "funding_interval_hours": 4.0,
            "size_precision": 2,
            "size_increment": 0.01,
            "account_type": "cash",
        }
    )
    assert validated["size_precision"] == 2
    assert validated["size_increment"] == 0.01
    assert validated["account_type"] == "cash"


def test_validate_overrides_rejects_bad_extended_schema():
    with pytest.raises(ConfigError, match="size_precision"):
        validate_overrides({"size_precision": -1})
    with pytest.raises(ConfigError, match="size_increment"):
        validate_overrides({"size_increment": 0.0})
    with pytest.raises(ConfigError, match="account_type"):
        validate_overrides({"account_type": "hedged"})
    with pytest.raises(ConfigError, match="fx_rates"):
        validate_overrides({"fx_rates": {"EURUSD": -1.0}})


def test_parse_synthetic_rates_normalizes_and_inverts():
    parsed = parse_synthetic_rates({"EURUSD": 1.2}, "fx_rates")
    assert parsed["EURUSD"] == pytest.approx(1.2)
    assert parsed["USDEUR"] == pytest.approx(1 / 1.2)
    parsed2 = parse_synthetic_rates(
        [{"pair": "GBPUSD", "rate": 1.3}, {"pair": "USDJPY", "rate": 110.0}],
        "synthetic_rates",
    )
    assert parsed2["GBPUSD"] == pytest.approx(1.3)
    assert parsed2["USDGBP"] == pytest.approx(1 / 1.3)
    assert parsed2["USDJPY"] == pytest.approx(110.0)
    assert parsed2["JPYUSD"] == pytest.approx(1 / 110.0)
    with pytest.raises(ConfigError):
        parse_synthetic_rates({"BAD": "x"}, "fx_rates")


# ---------------------------------------------------------------------------
# adapt_data translation helpers (§6.1).
# ---------------------------------------------------------------------------


def test_bar_timestamps_ns_is_int64_ns_axis():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    ts = bar_timestamps_ns(md)
    assert ts.dtype == np.int64
    assert ts.shape[0] == md.n_bars
    assert (np.diff(ts) > 0).all()
    assert ts[0] == int(md.timestamps.as_unit("ns").asi8[0])


def test_bar_step_grid_forward_fills_before_first_change():
    bar_ts = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    step_ts = np.array([2], dtype=np.int64)
    step_value = np.array([5.0])
    grid = bar_step_grid(step_ts, step_value, bar_ts)
    assert grid.tolist() == [0.0, 0.0, 5.0, 5.0, 5.0]


def test_to_signal_frame_carries_four_signal_lines():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    sig = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    frame = to_signal_frame(md, sig)
    assert len(frame.feed_frame) == md.n_bars
    for col in ("open", "high", "low", "close", "volume"):
        assert col in frame.feed_frame.columns
    for col in ("long_entry", "long_exit", "short_entry", "short_exit"):
        assert col in frame.feed_frame.columns
    entry_bar = int(np.argmax(sig.long_entry))
    assert bool(frame.feed_frame["long_entry"].iloc[entry_bar])
    exit_bar = int(np.argmax(sig.long_exit))
    assert bool(frame.feed_frame["long_exit"].iloc[exit_bar])


# ---------------------------------------------------------------------------
# exits translation (§5.2, §4.2, §8).
# ---------------------------------------------------------------------------


def test_validate_aux_raises_when_atr_series_absent():
    exits = (ATRStop(atr="atr_1h", mult=2.0),)
    with pytest.raises(ConfigError, match="aux_data"):
        validate_aux(exits, None)
    with pytest.raises(ConfigError, match="absent from aux_data"):
        validate_aux(exits, {"other": 1.0})


def test_validate_aux_passes_when_atr_series_present():
    exits = (ATRStop(atr="atr_1h", mult=2.0), ChandelierExit(atr="atr_1h", mult=3.0))
    validate_aux(exits, {"atr_1h": object()})


def test_exit_reason_label_maps_canonical_exits():
    assert exit_reason_label(TakeProfit(0.05)) == "take_profit"
    assert exit_reason_label(ATRStop(atr="a", mult=2.0)) == "atr_stop"
    assert exit_reason_label(TrailingStop(0.01)) == "trailing_stop"
    assert exit_reason_label(ChandelierExit(atr="a", mult=3.0)) == "chandelier"
    assert exit_reason_label(TimeExit(3)) == "time_exit"


def test_build_exit_plan_is_causal_and_anchored():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    plan = build_exit_plan(
        (TakeProfit(0.05, scale_out=0.5), TrailingStop(0.01)),
        md,
        side=1,
        entry_price=float(md.open[1]),
        entry_bar=1,
    )
    assert len(plan) == 2
    assert plan[0].reason == "take_profit"
    assert plan[0].fraction == pytest.approx(0.5)
    assert plan[1].reason == "trailing_stop"
    assert plan[1].fraction == pytest.approx(1.0)
    # The plans must never fire before the entry bar (no look-ahead into the past). The
# strategy only evaluates exit plans from the fill bar (entry_bar + 1) onwards, so a
# trailing stop may legitimately read True on the entry bar itself — but never earlier.
    assert not plan[0].triggered[:1].any()
    assert not plan[1].triggered[:1].any()
    assert plan[0].triggered.dtype == np.bool_ or plan[0].triggered.dtype == bool


def test_atr_from_aux_has_no_lookahead_shift():
    main = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    aux = synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=4)
    series = atr_from_aux(main, aux, period=3)
    assert series.shape[0] == main.n_bars
    assert np.isfinite(series[0])


# ---------------------------------------------------------------------------
# End-to-end Backtrader runs — ledger fold + BacktestResult (§4.5, §4.6).
# ---------------------------------------------------------------------------


@_requires_bt
def test_full_loop_futures_signal_roundtrip():
    # Entry signal at bar 1 -> market order fills at bar 2 open; long_exit at bar 4 ->
    # fills at bar 5 open (backtrader's next-bar-open execution).
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    fills = _fills(result)
    assert len(fills) == 2
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None), (-1, "signal")]
    # all_in sizing (fee-less) at the slipped close of the signal bar, floored to whole lots.
    qty = int(100000.0 / float(md.close[1]))
    assert fills[0].quantity == pytest.approx(qty, abs=1e-6)
    assert fills[0].price == pytest.approx(float(md.open[2]), abs=1e-6)
    assert fills[1].price == pytest.approx(float(md.open[5]), abs=1e-6)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert trade.quantity == pytest.approx(qty, abs=1e-6)
    assert trade.entry_price == pytest.approx(float(md.open[2]), abs=1e-6)
    assert trade.exit_price == pytest.approx(float(md.open[5]), abs=1e-6)
    # Flat at the end: final equity is the net cash book (position fully closed).
    cash = _ledger_events(result, EventType.CASH_MOVEMENT)
    assert float(result.equity_curve.equity[0]) == pytest.approx(100000.0)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(
        sum(e.amount for e in cash), abs=1e-6
    )
    # Position holds from bar 2 through bar 5 open; flat and constant afterwards.
    assert float(result.equity_curve.equity[-1]) == pytest.approx(
        float(result.equity_curve.equity[5]), abs=1e-6
    )
    assert not _ledger_events(result, EventType.COMMISSION)
    assert not _ledger_events(result, EventType.FUNDING_PAYMENT)


@_requires_bt
def test_full_loop_flat_run_stays_flat():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = BacktraderAdapter().run(
        md,
        from_target([0] * 12),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    assert _fills(result) == []
    assert result.trades == ()
    assert (result.equity_curve.equity == 100000.0).all()


@_requires_bt
def test_full_loop_crypto_perp_books_fees_and_funding():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0002, funding=0.0001),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    fills = _fills(result)
    assert len(fills) == 2
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None), (-1, "signal")]
    assert fills[0].quantity > 0.0
    assert fills[0].quantity == pytest.approx(fills[1].quantity, abs=1e-6)
    # Fills at slipped prices (§8): entry pays slippage, exit receives less.
    assert fills[0].price == pytest.approx(float(md.open[2]) * 1.0002, rel=1e-9)
    assert fills[1].price == pytest.approx(float(md.open[5]) * 0.9998, rel=1e-9)
    commissions = _ledger_events(result, EventType.COMMISSION)
    fundings = _ledger_events(result, EventType.FUNDING_PAYMENT)
    assert len(commissions) == 2
    assert all(e.amount > 0.0 for e in commissions)
    # Each order emitted an order_submitted on its signal bar and a fill on the next bar.
    submitted = _ledger_events(result, EventType.ORDER_SUBMITTED)
    assert len(submitted) == 2
    assert [int(e.side) for e in submitted] == [1, -1]
    assert len(fundings) == 3  # three open bars between the fills
    assert all(e.amount > 0.0 for e in fundings)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"


@_requires_bt
def test_full_loop_crypto_perp_short():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = BacktraderAdapter().run(
        md,
        from_target([0, -1, -1, -1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0002, funding=0.0001),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    fills = _fills(result)
    assert [(e.side, e.exit_reason) for e in fills] == [(-1, None), (1, "signal")]
    assert len(_ledger_events(result, EventType.FUNDING_PAYMENT)) == 3
    (trade,) = result.trades
    assert trade.side == -1
    assert trade.exit_reason == "signal"


@_requires_bt
def test_trailing_stop_stamps_exit_reason():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = BacktraderAdapter().run(
        md,
        from_target([1] * 10),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(TrailingStop(0.001),)),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    fills = _fills(result)
    assert len(fills) == 2
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None), (-1, "trailing_stop")]
    (trade,) = result.trades
    assert trade.exit_reason == "trailing_stop"
    assert float(result.equity_curve.equity[-1]) > 0.0


@_requires_bt
def test_time_exit_executes_and_stamps_reason():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    # Entry signal bar 1 -> fill bar 2 open; TimeExit(2) fires at entry_bar + 2 = bar 4,
    # filling at bar 5 open.
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(TimeExit(2),)),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    fills = _fills(result)
    assert len(fills) == 2
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None), (-1, "time_exit")]
    (trade,) = result.trades
    assert trade.exit_reason == "time_exit"
    assert fills[1].price == pytest.approx(float(md.open[5]), abs=1e-6)


@_requires_bt
def test_full_loop_flip_within_same_bar_submits_exit_then_entry():
    # 1 -> -1 flip at bar 2: from_target emits long_exit + short_entry on that bar. The
    # strategy submits the exit first, then (gated) the opposing entry — both fill at bar 3
    # open (next bar after submission). The short exits at bar 5 open.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, -1, -1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0002, funding=0.0001),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    fills = _fills(result)
    assert [(e.side, e.exit_reason) for e in fills] == [
        (1, None),
        (-1, "signal"),
        (-1, None),
        (1, "signal"),
    ]
    assert fills[1].price == pytest.approx(float(md.open[3]) * 0.9998, rel=1e-9)
    # Both legs submitted bar 2 -> both fill at bar 3 open; the short entry also sells at
    # the slipped side (price * (1 - slip)).
    assert fills[2].price == pytest.approx(float(md.open[3]) * 0.9998, rel=1e-9)
    assert fills[3].price == pytest.approx(float(md.open[5]) * 1.0002, rel=1e-9)
    # Both legs submitted on the same bar: exit closes the long first, then the short entry.
    submitted = _ledger_events(result, EventType.ORDER_SUBMITTED)
    exit_sub, entry_sub = submitted[1], submitted[2]
    assert exit_sub.timestamp == entry_sub.timestamp
    assert exit_sub.side == -1 and entry_sub.side == -1
    assert exit_sub.quantity == pytest.approx(fills[0].quantity, abs=1e-6)
    assert entry_sub.quantity == pytest.approx(fills[2].quantity, abs=1e-6)
    assert fills[2].quantity > 0
    assert len(_ledger_events(result, EventType.FUNDING_PAYMENT)) > 0
    assert len(result.trades) == 2


@_requires_bt
def test_full_loop_final_bar_exit_signal_realizes_at_last_close():
    # Exit signal on the very last bar (target flat at bar 11): the order can't fill
    # next-bar-open, so stop() realizes it at the final bar's close with the signal reason
    # (reference adapter semantics).
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    fills = _fills(result)
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None), (-1, "signal")]
    assert fills[1].price == pytest.approx(float(md.close[11]), abs=1e-6)
    assert len(result.trades) == 1
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert trade.exit_price == pytest.approx(float(md.close[11]), abs=1e-6)
    cash = sum(e.amount for e in _ledger_events(result, EventType.CASH_MOVEMENT))
    assert float(result.equity_curve.equity[-1]) == pytest.approx(cash, abs=1e-6)


@_requires_bt
def test_full_loop_sizing_uses_realized_net_balance_times_leverage():
    # §6.3 — the reference (paper/nautilus) sizes off ``(starting_balance + realized net
    # PnL) * leverage`` — the *unleveraged* balance scaled afterwards — NOT the levered
    # broker equity, which accrues PnL on the leveraged base and diverges from the
    # reference from the second trade on. After a losing round trip, the next entry must
    # be sized off ``balance * 100`` (with the loss counted once), not off
    # ``1000000 - loss``.  Two longs with a lossy gap between them.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0, funding=0.0),
            risk=RiskConfig(
                sizing=SizeModel(kind="fixed_fraction", value=0.10, leverage=100.0)
            ),
            engine_overrides={"starting_balance": 10000.0},
        ),
    )
    assert len(result.trades) == 2
    first, second = sorted(result.trades, key=lambda t: t.entry_timestamp)
    # Starting balance: qty = floor(0.10 * 10000 * 100 / entry_price, 0.001).
    assert first.quantity == pytest.approx(
        floor_to_increment(0.10 * 10000.0 * 100.0 / first.entry_price, 0.001),
        abs=1e-6,
    )
    # After the (non-zero) first trade: qty = floor(0.10 * (10000 + net_pnl) * 100 / price).
    balance = 10000.0 + float(first.net_pnl)
    assert balance != pytest.approx(10000.0, abs=1e-9)  # PnL non-zero — the gap discriminates
    assert second.quantity == pytest.approx(
        floor_to_increment(0.10 * balance * 100.0 / second.entry_price, 0.001),
        abs=1e-6,
    )


@_requires_bt
def test_full_loop_open_position_stays_open_without_signal():
    # No exit ever signalled: the final trade is left OPEN (realized PnL 0.0), marked to
    # market through the last bar — mirroring the paper/nautilus references, which do not
    # force-close a position merely because the run ended.
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=6)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    fills = _fills(result)
    assert [(e.side, e.exit_reason) for e in fills] == [(1, None)]
    assert fills[0].price == pytest.approx(float(md.open[2]), abs=1e-6)
    # No completed round trip — the open position is excluded from ``trades``.
    assert result.trades == ()
    (row,) = result.trade_table.itertuples()
    assert row.status == "open"
    assert row.realized_pnl == 0.0
    assert pd.Timestamp(row.exit_datetime) == pd.Timestamp(md.timestamps[-1])
    # Marked to market through the last bar: balance (M2M equity) == equity_curve tail.
    assert row.exit_price == pytest.approx(float(md.close[-1]), abs=1e-6)
    assert row.balance == pytest.approx(
        float(result.equity_curve.equity[-1]), abs=1e-6
    )


@_requires_bt
def test_take_profit_scale_out_partial_close():
    # TakeProfit with scale_out=0.5 closes half, then the signal exit closes the rest.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(TakeProfit(0.01, scale_out=0.5),)),
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    fills = _fills(result)
    assert len(fills) >= 2
    # The scale-out close carries the take_profit reason and the lot-floored half.
    tp_fills = [e for e in fills if e.exit_reason == "take_profit"]
    assert tp_fills, "expected at least one scale-out take-profit fill"
    assert tp_fills[0].quantity == pytest.approx(
        floor_to_increment(fills[0].quantity * 0.5, 0.001), abs=1e-6
    )
    # The ledger folds partial closes: total traded quantity matches the entry once combined.
    entry_qty = fills[0].quantity
    assert sum(e.quantity for e in fills[1:]) == pytest.approx(entry_qty, abs=1e-6)


@_requires_bt
def test_volatility_target_sizing_runs():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    aux = {"vol_1h": synthetic_bars(PRESETS["futures"], seed=3, n_bars=12)}
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            risk=RiskConfig(
                sizing=SizeModel(kind="volatility_target", value=0.02, vol="vol_1h")
            ),
            engine_overrides={"starting_balance": 100000.0},
        ),
        aux_data=aux,
    )
    assert len(result.trades) == 1
    assert result.trades[0].quantity > 0


@_requires_bt
def test_volatility_target_missing_vol_key_raises():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    with pytest.raises(ConfigError, match="volatility_target"):
        BacktraderAdapter().run(
            md,
            from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            BacktestConfig(
                instrument=PRESETS["futures"].instrument,
                risk=RiskConfig(
                    sizing=SizeModel(kind="volatility_target", value=0.02)
                ),
                engine_overrides={"starting_balance": 100000.0},
            ),
        )


@_requires_bt
def test_aux_atr_stop_runs_with_aux_data():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    aux = {"atr_1h": synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=20)}
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(ATRStop(atr="atr_1h", mult=2.0),)),
            engine_overrides={"starting_balance": 100000.0},
        ),
        aux_data=aux,
    )
    assert len(result.trades) >= 1


@_requires_bt
def test_aux_guard_raises_without_aux_data():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    with pytest.raises(ConfigError, match="aux_data"):
        BacktraderAdapter().run(
            md,
            from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            BacktestConfig(
                instrument=PRESETS["crypto_perp"].instrument,
                risk=RiskConfig(exit=(ATRStop(atr="atr_1h", mult=2.0),)),
                engine_overrides={"starting_balance": 100000.0},
            ),
        )


# ---------------------------------------------------------------------------
# Input validation (run rejects malformed contracts, §7.1).
# ---------------------------------------------------------------------------


@_requires_bt
def test_run_rejects_non_market_data_input():
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(DataShapeError, match="MarketData"):
        BacktraderAdapter().run(
            object(),  # type: ignore[arg-type]
            sig,
            BacktestConfig(instrument=PRESETS["futures"].instrument),
        )


@_requires_bt
def test_run_rejects_non_signals_input():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=4)
    with pytest.raises(InvalidSignalError, match="Signals"):
        BacktraderAdapter().run(
            md,
            object(),  # type: ignore[arg-type]
            BacktestConfig(instrument=PRESETS["futures"].instrument),
        )


@_requires_bt
def test_run_rejects_non_backtest_config():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=4)
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(ConfigError, match="BacktestConfig"):
        BacktraderAdapter().run(md, sig, object())  # type: ignore[arg-type]


@_requires_bt
def test_run_rejects_row_misaligned_signals():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(InvalidSignalError, match="row-aligned"):
        BacktraderAdapter().run(
            md, sig, BacktestConfig(instrument=PRESETS["futures"].instrument)
        )


@_requires_bt
def test_backtrader_ignores_short_signals_on_long_only_crypto_spot():
    # crypto_spot is the engine's one long-only class (§4.5): shorting is undefined
    # (nothing to borrow), so the backtrader adapter ignores short_entry/short_exit at
    # the engine layer even when the caller passes them — only long entries/exits are
    # acted on. from_target([0, -1, ...]) opens a short at bar 1; the adapter drops it,
    # so the run completes with zero fills and no short-side events (no signal-gate
    # error).  (crypto_spot is not a preset, so build the instrument explicitly.)
    inst = replace(PRESETS["stocks"].instrument, asset_class="crypto_spot")
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    result = BacktraderAdapter().run(
        md,
        from_target([0, -1, -1, -1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    assert _fills(result) == []
    assert len(result.trades) == 0


@_requires_bt
def test_backtrader_crypto_spot_drops_short_leg_of_flip_keeps_long_exit():
    # A flip encoding (long held, then short_entry) is legal (§6.2) but its short leg is
    # dead for a long-only class: the adapter keeps the long_exit (a long-side action)
    # and drops only short_entry/short_exit, so the long closes normally and no short
    # ever opens.  [0, 1, 1, 1, -1, -1] emits long_entry at 1, flip long_exit +
    # short_entry at 4, hold. Backtrader fills one bar later (next-bar-open fills).
    inst = replace(PRESETS["stocks"].instrument, asset_class="crypto_spot")
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    result = BacktraderAdapter().run(
        md,
        from_target([0, 1, 1, 1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    fills = _fills(result)
    assert [e.side for e in fills] == [1, -1]  # one long entry + one long exit
    assert len(result.trades) == 1  # the long trade, not a short


@_requires_bt
def test_backtrader_short_signals_still_accepted_for_shortable_asset_class():
    # Long-only neutralization is scoped to classes that cannot short (§4.5); a
    # shortable class (crypto_perp) still opens and closes shorts normally.
    inst = replace(PRESETS["crypto_perp"].instrument, asset_class="crypto_perp")
    md = synthetic_bars(PRESETS["crypto_perp"], seed=3, n_bars=6)
    result = BacktraderAdapter().run(
        md,
        from_target([0, -1, -1, -1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    assert any(e.side < 0 for e in _fills(result))


# ---------------------------------------------------------------------------
# Orchestrator path (ube.run) with engine="backtrader".
# ---------------------------------------------------------------------------


@_requires_bt
def test_run_end_to_end_via_ube_run_with_backtrader(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    config = BacktestConfig(
        instrument=PRESETS["futures"].instrument,
        engine="backtrader",
        engine_overrides={"starting_balance": 100000.0},
    )
    result = ube.run(md, signals, config, log_path=tmp_path / "experiments.db")
    assert isinstance(result, BacktestResult)
    assert len(result.trades) == 1
    with ExperimentLog(path=tmp_path / "experiments.db") as log:
        record = log.get(result.run_id)
        assert record is not None
        assert record.engine == "backtrader"


# ---------------------------------------------------------------------------
# Asset-class handling (§4.5): lot quantization across asset classes.
# ---------------------------------------------------------------------------


@_requires_bt
def test_handles_all_asset_classes_with_lot_quantization():
    for key in ["stocks", "futures", "commodities", "forex", "crypto_perp"]:
        preset = PRESETS[key]
        md = synthetic_bars(preset, seed=3, n_bars=12)
        signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        config = BacktestConfig(
            instrument=preset.instrument,
            engine="backtrader",
            engine_overrides={"starting_balance": 100000.0},
        )
        result = BacktraderAdapter().run(md, signals, config)
        fills = _fills(result)
        assert fills, f"expected a fill for asset class {key!r}"
        inst = build_instrument(preset.instrument, config.engine_overrides)
        inc = inst.size_increment
        for f in fills:
            if inc > 0:
                ratio = f.quantity / inc
                assert abs(ratio - round(ratio)) < 1e-6, (
                    f"{key}: qty {f.quantity} not a multiple of increment {inc}"
                )
            assert f.quantity > 0
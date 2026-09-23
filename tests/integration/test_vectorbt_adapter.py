"""Vectorbt adapter tests — one file per adapter, append-only.

This is the single test file for the vectorbt adapter and its shared base-contract
dependencies. It mirrors the structure of ``test_nautilus_adapter.py`` but exercises only
the vectorbt engine (§4.1, §4.2): engine registration, the vectorbt ``engine_overrides``
registry, the MarketData/Signals translation helpers, the ATR/aux exit translation, the
vectorized portfolio execution, and the canonical ledger fold (§4.6).

It never touches the Nautilus adapter or its tests — every assertion here runs against
:class:`~ube.adapters.vectorbt_adapter.adapter.VectorbtAdapter` only.
"""

from dataclasses import replace

import numpy as np
import pytest

import ube
from ube.adapters import get_engine
from ube.adapters.base import _REGISTRY
from ube.adapters.vectorbt_adapter.adapt_data import (
    bar_step_grid,
    bar_timestamps_ns,
    to_vbt_inputs,
)
from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter
from ube.adapters.vectorbt_adapter.engine import build_portfolio
from ube.adapters.vectorbt_adapter.exits import (
    atr_from_aux,
    classify_exit_reason,
    exit_stop_params,
    validate_aux,
)
from ube.adapters.vectorbt_adapter.overrides import (
    DEFAULT_FUNDING_INTERVAL_HOURS,
    DEFAULT_STARTING_BALANCE,
    parse_synthetic_rates,
    validate_overrides,
)
from ube.core.config import BacktestConfig
from ube.core.cost import CostModel
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, InvalidSignalError
from ube.core.experiment_log import ExperimentLog
from ube.core.ledger import EventType
from ube.core.result import BacktestResult
from ube.core.risk import RiskConfig, SizeModel
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    TakeProfit,
    TimeExit,
    TrailingStop,
)
from ube.core.signals import from_target
from ube.testing.synthetic import PRESETS, synthetic_bars


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the global registry around every test."""
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _fills(result):
    return [e for e in result.ledger if e.event_type is EventType.FILL]


def _ledger_events(result, event_type):
    return [e for e in result.ledger if e.event_type is event_type]


# ---------------------------------------------------------------------------
# Engine registration (§4.1).
# ---------------------------------------------------------------------------


def test_vectorbt_adapter_is_registrable_under_canonical_name():
    from ube.adapters import register_engine

    register_engine("vectorbt", VectorbtAdapter)
    assert get_engine("vectorbt") is VectorbtAdapter


def test_ensure_builtin_engines_registered_registers_vectorbt():
    ube.ensure_builtin_engines_registered()
    assert get_engine("vectorbt") is VectorbtAdapter


# ---------------------------------------------------------------------------
# vectorbt engine_overrides registry (§4.3, §7.2).
# ---------------------------------------------------------------------------


def test_validate_overrides_accepts_none():
    assert validate_overrides(None) == {}


def test_validate_overrides_returns_fresh_copy():
    raw = {"starting_balance": 1000.0}
    validated = validate_overrides(raw)
    validated["starting_balance"] = 999.0
    # The caller's mutation must not alias the original mapping.
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
    # A single quoted pair also yields its implicit inverse; a mapping form is accepted.
    parsed = parse_synthetic_rates({"EURUSD": 1.2}, "fx_rates")
    assert parsed["EURUSD"] == pytest.approx(1.2)
    assert parsed["USDEUR"] == pytest.approx(1 / 1.2)
    parsed2 = parse_synthetic_rates(
        [{"pair": "GBPUSD", "rate": 1.3}, {"pair": "USDJPY", "rate": 110.0}], "synthetic_rates"
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
    # Monotonic increasing, matching the raw nanosecond timestamps.
    assert (np.diff(ts) > 0).all()
    assert ts[0] == int(md.timestamps.as_unit("ns").asi8[0])


def test_bar_step_grid_forward_fills_before_first_change():
    bar_ts = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    # Position opens at bar 2.
    step_ts = np.array([2], dtype=np.int64)
    step_value = np.array([5.0])
    grid = bar_step_grid(step_ts, step_value, bar_ts)
    assert grid.tolist() == [0.0, 0.0, 5.0, 5.0, 5.0]


def test_to_vbt_inputs_returns_aligned_series():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    sig = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    inp = to_vbt_inputs(md, sig)
    assert len(inp.close) == md.n_bars
    assert len(inp.entries) == md.n_bars
    assert len(inp.short_exits) == md.n_bars
    assert bool(inp.entries.iloc[1])  # long entry at signal bar
    exit_bar = int(np.argmax(sig.long_exit))
    assert bool(inp.long_exits.iloc[exit_bar])


def test_to_vbt_inputs_tolerates_short_windows():
    # The paper engine feeds 1-2 bar windows per step; pd.infer_freq needs >=3 dates,
    # so to_vbt_inputs must fall back to the observed spacing (or a per-bar default).
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=3)
    ts = md.timestamps.as_unit("ns")
    one = MarketData(open=md.open[:1], high=md.high[:1], low=md.low[:1],
                     close=md.close[:1], volume=md.volume[:1], index=ts[:1])
    two = MarketData(open=md.open[:2], high=md.high[:2], low=md.low[:2],
                     close=md.close[:2], volume=md.volume[:2], index=ts[:2])
    assert to_vbt_inputs(one, from_target([0])).freq is not None
    assert to_vbt_inputs(two, from_target([0, 0])).freq is not None
    assert to_vbt_inputs(md, from_target([0, 0, 0])).freq == "h"


# ---------------------------------------------------------------------------
# exits translation (§5.2, §4.2).
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


def test_exit_stop_params_translates_atr_and_take_profit():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    aux = {"atr_1h": synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=4)}
    exits = (ATRStop(atr="atr_1h", mult=2.0), TakeProfit(0.05))
    sl_stop, tp_stop, sl_trail = exit_stop_params(exits, md, aux)
    assert tp_stop == 0.05
    assert sl_trail is None
    assert isinstance(sl_stop, np.ndarray)
    assert sl_stop.shape[0] == md.n_bars


def test_classify_exit_reason_signal_bar():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    sig = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    exit_bar = int(np.argmax(sig.long_exit))
    reason = classify_exit_reason(
        (),
        md,
        side=1,
        entry_price=100.0,
        entry_bar=1,
        exit_bar=exit_bar,
        aux_data=None,
        signals=sig,
    )
    assert reason == "signal"


def test_atr_from_aux_has_no_lookahead_shift():
    # Aux bars are coarser; the main-grid series must be shifted one bar (the value at
    # main bar i comes from an aux bar strictly before i, never the one it sits inside).
    main = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    aux = synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=4)
    series = atr_from_aux(main, aux, period=3)
    assert series.shape[0] == main.n_bars
    # First main bar has no prior aux bar -> resolved to a finite (zeroed) placeholder.
    assert np.isfinite(series[0])


# ---------------------------------------------------------------------------
# vectorized portfolio execution (engine.py).
# ---------------------------------------------------------------------------


def test_build_portfolio_returns_records():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    sig = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    inp = to_vbt_inputs(md, sig)
    pf = build_portfolio(
        inp,
        init_cash=100000.0,
        fees=0.0,
        sl_stop=None,
        tp_stop=None,
        sl_trail=None,
        size=1.0,
    )
    records = pf.trades.records_readable
    assert len(records) >= 1


# ---------------------------------------------------------------------------
# End-to-end Vectorbt runs — ledger fold + BacktestResult (§4.5, §4.6).
# ---------------------------------------------------------------------------


def test_vectorbt_full_loop_futures_signal_roundtrip():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = VectorbtAdapter().run(
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
    assert fills[0].quantity == pytest.approx(19.0, abs=1e-6)
    assert fills[0].price == pytest.approx(5002.5, abs=1e-6)
    assert fills[1].price == pytest.approx(4989.5, abs=1e-6)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert trade.net_pnl == pytest.approx(-12350.0, abs=1e-6)
    assert float(result.equity_curve.equity[0]) == pytest.approx(100000.0)
    # Parity with Nautilus: always-floored qty (19.0 contracts) matches the actor's §7.1
    # floor-to-lot, so both engines book the same final equity.
    assert float(result.equity_curve.equity[-1]) == pytest.approx(87650.0, abs=1e-6)
    assert not _ledger_events(result, EventType.COMMISSION)
    assert not _ledger_events(result, EventType.FUNDING_PAYMENT)


def test_vectorbt_full_loop_flat_run_stays_flat():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = VectorbtAdapter().run(
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


def test_vectorbt_full_loop_crypto_perp_books_fees_and_funding():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
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
    # Fee-aware all_in reserves the entry fee (§7.1), so the filled quantity is slightly
    # below the fee-less 1.6508 the previous test pinned.
    assert fills[0].quantity == pytest.approx(1.65, abs=1e-3)
    commissions = _ledger_events(result, EventType.COMMISSION)
    fundings = _ledger_events(result, EventType.FUNDING_PAYMENT)
    assert len(commissions) >= 1
    assert all(e.amount > 0.0 for e in commissions)
    assert len(fundings) == 3  # three held bars between entry and exit
    assert all(e.amount > 0.0 for e in fundings)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert float(result.equity_curve.equity[-1]) == pytest.approx(100090.59, abs=1e-2)


def test_vectorbt_full_loop_crypto_perp_short_pays_loss():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
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
    # Parity with the Nautilus engine (which pins the same value): fees are booked at the
    # slipped fill notional and funding on the held notional, so the short is less lossy than
    # the legacy vectorbt fold predicted.
    assert float(result.equity_curve.equity[-1]) == pytest.approx(99568.9293, abs=1e-2)


def test_vectorbt_funding_default_8h_accrues_proportionally_per_open_bar():
    # §24 (blocker 2) accrues funding by elapsed wall-clock time, emitting one proportional
    # event per open bar. With the default 8h interval and a ~3h hold, three open bars each
    # accrue ~1/8 of the per-period rate (frac = bar_span / 8h), so three events are emitted
    # whose summed amount equals the true time-weighted carry — rather than the legacy
    # "nothing if under one full interval" behavior.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(funding=0.0001),
            # No funding_interval_hours override -> default 8h.
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    fundings = _ledger_events(result, EventType.FUNDING_PAYMENT)
    assert len(fundings) == 3
    assert all(e.amount > 0.0 for e in fundings)


def test_vectorbt_trailing_stop_stamps_exit_reason():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
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


def test_vectorbt_time_exit_executes_and_stamps_reason():
    # TimeExit(bars) must actually fold into the vectorbt exit masks (it has no native
    # holding-period primitive) and stamp "time_exit" on the fill.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
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


def test_vectorbt_time_exit_yields_to_earlier_signal_exit():
    # A signal exit on the same bar closes first (first-exit-wins); the time mask is idle.
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(TimeExit(10),)),
            engine_overrides={"starting_balance": 100000.0, "funding_interval_hours": 1.0},
        ),
    )
    (trade,) = result.trades
    assert trade.exit_reason == "signal"


def test_vectorbt_volatility_target_sizing_runs():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    # §6.3: vol comes from aux_data (never the signal bars); the SizeModel must name it via
    # the `vol` key. A raw MarketData series is turned into ATR/price on the aux grid.
    aux = {"vol_1h": synthetic_bars(PRESETS["futures"], seed=3, n_bars=12)}
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            risk=RiskConfig(sizing=SizeModel(kind="volatility_target", value=0.02, vol="vol_1h")),
            engine_overrides={"starting_balance": 100000.0},
        ),
        aux_data=aux,
    )
    assert len(result.trades) == 1
    # The volatility budget sizes beyond unleveraged equity: units = target*capital/(price*vol)
    # is not capped at a fixed-fraction of the account.
    assert result.trades[0].quantity > 0


def test_vectorbt_volatility_target_missing_vol_key_raises():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    with pytest.raises(ConfigError, match="volatility_target"):
        VectorbtAdapter().run(
            md,
            from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            BacktestConfig(
                instrument=PRESETS["futures"].instrument,
                risk=RiskConfig(sizing=SizeModel(kind="volatility_target", value=0.02)),
                engine_overrides={"starting_balance": 100000.0},
            ),
        )


def test_vectorbt_aux_atr_stop_runs_with_aux_data():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    # aux must span the signal/price period (range-consistency §5.2), so 20 bars like main.
    aux = {"atr_1h": synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=20)}
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            risk=RiskConfig(exit=(ATRStop(atr="atr_1h", mult=2.0),)),
            engine_overrides={"starting_balance": 100000.0},
        ),
        aux_data=aux,
    )
    # aux_data supplied -> no ConfigError; a trade is produced from the long signal.
    assert len(result.trades) >= 1


def test_vectorbt_aux_guard_raises_without_aux_data():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    with pytest.raises(ConfigError, match="aux_data"):
        VectorbtAdapter().run(
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


def test_vectorbt_run_rejects_non_market_data_input():
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(DataShapeError, match="MarketData"):
        VectorbtAdapter().run(
            object(),  # type: ignore[arg-type]
            sig,
            BacktestConfig(instrument=PRESETS["futures"].instrument),
        )


def test_vectorbt_run_rejects_non_signals_input():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=4)
    with pytest.raises(InvalidSignalError, match="Signals"):
        VectorbtAdapter().run(
            md,
            object(),  # type: ignore[arg-type]
            BacktestConfig(instrument=PRESETS["futures"].instrument),
        )


def test_vectorbt_run_rejects_non_backtest_config():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=4)
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(ConfigError, match="BacktestConfig"):
        VectorbtAdapter().run(md, sig, object())  # type: ignore[arg-type]


def test_vectorbt_run_rejects_row_misaligned_signals():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    sig = from_target([0, 1, 0, 0])
    with pytest.raises(InvalidSignalError, match="row-aligned"):
        VectorbtAdapter().run(
            md, sig, BacktestConfig(instrument=PRESETS["futures"].instrument)
        )


def test_vectorbt_run_tolerates_single_bar_data():
    # 1-bar windows (the paper engine's per-step feed) previously crashed on
    # pd.infer_freq("Need at least 3 dates"); the freq fallback makes them valid.
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=1)
    sig = from_target([0])
    res = VectorbtAdapter().run(
        md, sig, BacktestConfig(instrument=PRESETS["futures"].instrument)
    )
    # Completes (no frequency-inference crash) and only books the opening cash movement.
    assert res.ledger.events and all(
        e.event_type is EventType.CASH_MOVEMENT for e in res.ledger.events
    )


def test_vectorbt_ignores_short_signals_on_long_only_crypto_spot():
    # crypto_spot is the engine's one long-only class (§4.5): shorting is undefined
    # (nothing to borrow), so the vectorbt adapter ignores short_entry/short_exit at
    # the engine layer even when the caller passes them — only long entries/exits are
    # acted on. from_target([0, -1, ...]) opens a short at bar 1; the adapter drops it,
    # so the run completes with zero fills and no short-side events (no signal-gate
    # error).  (crypto_spot is not a preset, so build the instrument explicitly.)
    inst = replace(PRESETS["stocks"].instrument, asset_class="crypto_spot")
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    result = VectorbtAdapter().run(
        md,
        from_target([0, -1, -1, -1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    assert _fills(result) == []
    assert len(result.trades) == 0


def test_vectorbt_crypto_spot_drops_short_leg_of_flip_keeps_long_exit():
    # A flip encoding (long held, then short_entry) is legal (§6.2) but its short leg is
    # dead for a long-only class: the adapter keeps the long_exit (a long-side action)
    # and drops only short_entry/short_exit, so the long closes normally and no short
    # ever opens.  [0, 1, 1, 1, -1, -1] emits long_entry at 1, flip long_exit +
    # short_entry at 4, hold.
    inst = replace(PRESETS["stocks"].instrument, asset_class="crypto_spot")
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    fills = _fills(result)
    assert [e.side for e in fills] == [1, -1]  # one long entry + one long exit
    assert len(result.trades) == 1  # the long trade, not a short


def test_vectorbt_short_signals_still_accepted_for_shortable_asset_class():
    # Long-only neutralization is scoped to classes that cannot short (§4.5); a
    # shortable class (crypto_perp) still opens and closes shorts normally.
    inst = replace(PRESETS["crypto_perp"].instrument, asset_class="crypto_perp")
    md = synthetic_bars(PRESETS["crypto_perp"], seed=3, n_bars=6)
    result = VectorbtAdapter().run(
        md,
        from_target([0, -1, -1, -1, -1, -1]),
        BacktestConfig(instrument=inst),
    )
    assert any(e.side < 0 for e in _fills(result))


# ---------------------------------------------------------------------------
# Orchestrator path (ube.run) with engine="vectorbt".
# ---------------------------------------------------------------------------


def test_run_end_to_end_via_ube_run_with_vectorbt(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    config = BacktestConfig(
        instrument=PRESETS["futures"].instrument,
        engine="vectorbt",
        engine_overrides={"starting_balance": 100000.0},
    )
    result = ube.run(md, signals, config, log_path=tmp_path / "experiments.db")
    assert isinstance(result, BacktestResult)
    assert len(result.trades) == 1
    with ExperimentLog(path=tmp_path / "experiments.db") as log:
        record = log.get(result.run_id)
        assert record is not None
        assert record.engine == "vectorbt"


# ---------------------------------------------------------------------------
# Asset-class handling (§4.5): the vbt adapter must size every asset class
# with the same per-asset-class precision / lot increment that Nautilus applies.
# ---------------------------------------------------------------------------


def test_vbt_handles_all_asset_classes_with_lot_quantization():
    """The vbt adapter resolves asset-class params via build_instrument and quantizes sizes.

    Stocks / futures / commodities trade whole contracts or shares; forex keeps its fine
    precision; crypto_perp keeps fractional lots. This mirrors Nautilus' instrument_map so the
    same config produces the same lot-quantized quantity on either engine.
    """
    from ube.adapters.vectorbt_adapter.instrument_map import build_instrument

    for key in ["stocks", "futures", "commodities", "forex", "crypto_perp"]:
        preset = PRESETS[key]
        md = synthetic_bars(preset, seed=3, n_bars=12)
        signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        config = BacktestConfig(
            instrument=preset.instrument,
            engine="vectorbt",
            engine_overrides={"starting_balance": 100000.0},
        )
        result = VectorbtAdapter().run(md, signals, config)
        fills = _fills(result)
        assert fills, f"expected a fill for asset class {key!r}"
        vbt_inst = build_instrument(preset.instrument, config.engine_overrides)
        inc = vbt_inst.size_increment
        for f in fills:
            # Quantity must be a whole multiple of the asset-class lot increment.
            if inc > 0:
                ratio = f.quantity / inc
                assert abs(ratio - round(ratio)) < 1e-6, (
                    f"{key}: qty {f.quantity} not a multiple of increment {inc}"
                )
            assert f.quantity > 0


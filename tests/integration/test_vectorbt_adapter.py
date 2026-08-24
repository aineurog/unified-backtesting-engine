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
from ube.adapters.vectorbt_adapter.engine import build_portfolio, vbt
from ube.adapters.vectorbt_adapter.exits import (
    atr_from_aux,
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
from ube.core.cost import CostModel
from ube.core.errors import ConfigError, DataShapeError, EngineError, InvalidSignalError
from ube.core.experiment_log import ExperimentLog
from ube.core.ledger import EventType
from ube.core.result import BacktestResult
from ube.core.risk import RiskConfig, SizeModel
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    TakeProfit,
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


@pytest.mark.skipif(vbt is None, reason="vectorbt not installed")
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
    assert fills[0].quantity == pytest.approx(19.99, abs=0.05)
    assert fills[0].price == pytest.approx(5002.5, abs=1e-6)
    assert fills[1].price == pytest.approx(4989.5, abs=1e-6)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert trade.net_pnl == pytest.approx(-12993.5, abs=1.0)
    assert float(result.equity_curve.equity[0]) == pytest.approx(100000.0)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(87006.5, abs=1.0)
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
    assert fills[0].quantity == pytest.approx(1.6508, abs=1e-3)
    commissions = _ledger_events(result, EventType.COMMISSION)
    fundings = _ledger_events(result, EventType.FUNDING_PAYMENT)
    assert len(commissions) >= 1
    assert all(e.amount > 0.0 for e in commissions)
    assert len(fundings) == 3  # three held bars between entry and exit
    assert all(e.amount > 0.0 for e in fundings)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    assert float(result.equity_curve.equity[-1]) == pytest.approx(100090.63, abs=1e-2)


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
    assert float(result.equity_curve.equity[-1]) == pytest.approx(99568.69, abs=1e-2)


def test_vectorbt_funding_default_8h_charges_nothing_for_short_hold():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            cost_model=CostModel(funding=0.0001),
            # No funding_interval_hours override -> default 8h; a <8h hold books nothing.
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    assert len(_ledger_events(result, EventType.FUNDING_PAYMENT)) == 0


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


def test_vectorbt_volatility_target_sizing_runs():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = VectorbtAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            risk=RiskConfig(sizing=SizeModel(kind="volatility_target", value=0.02)),
            engine_overrides={"starting_balance": 100000.0},
        ),
    )
    assert len(result.trades) == 1


def test_vectorbt_aux_atr_stop_runs_with_aux_data():
    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=20)
    aux = {"atr_1h": synthetic_bars(PRESETS["crypto_perp"], seed=1, n_bars=6)}
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


def test_vectorbt_run_rejects_single_bar_data():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=1)
    sig = from_target([0])
    with pytest.raises((EngineError, DataShapeError, ValueError)):
        VectorbtAdapter().run(
            md, sig, BacktestConfig(instrument=PRESETS["futures"].instrument)
        )


def test_vectorbt_run_rejects_short_on_long_only_asset():
    # crypto_spot is the only long-only asset class; it is not a preset, so build the
    # instrument with that asset class explicitly (validate_long_only keys off the string).
    inst = replace(PRESETS["stocks"].instrument, asset_class="crypto_spot")
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    with pytest.raises(InvalidSignalError, match="long-only"):
        VectorbtAdapter().run(
            md,
            from_target([0, -1, -1, -1, -1, -1]),
            BacktestConfig(instrument=inst),
        )


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

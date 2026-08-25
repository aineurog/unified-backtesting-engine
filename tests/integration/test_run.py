"""Top-level ``ube.run`` orchestrator tests (§4.8, §7.1).

The orchestrator ties the Phase-1 core together: standardize the inputs, resolve the
engine, run the adapter, attach the benchmark, and record the run to the experiment log.
These tests exercise that pipeline end-to-end through the Nautilus adapter (the only
implemented engine) and verify the benchmark attachment and the unconditional
experiment-log record.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import ube
from ube.adapters.base import _REGISTRY
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import (
    CalendarMismatchError,
    ConfigError,
    DataShapeError,
    InvalidSignalError,
    UndeclaredConfigError,
)
from ube.core.experiment_log import ExperimentLog
from ube.core.result import BacktestResult, result_hash
from ube.core.risk import RiskConfig
from ube.core.risk.sizing import SizeModel
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


def _config() -> BacktestConfig:
    return BacktestConfig(
        instrument=PRESETS["futures"].instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 100000.0},
    )


def test_run_end_to_end_attaches_benchmark_and_logs(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    log_path = tmp_path / "experiments.db"

    result = ube.run(md, signals, _config(), log_path=log_path)

    assert isinstance(result, BacktestResult)
    assert result.benchmark is not None
    assert result.benchmark.n_bars == md.n_bars
    assert float(result.benchmark.equity[0]) == 1.0  # buy-and-hold, normalized

    with ExperimentLog(path=log_path) as log:
        assert log.count() == 1
        record = log.get(result.run_id)
        assert record is not None
        assert record.engine == "nautilus"
        assert record.instrument == PRESETS["futures"].instrument.symbol
        assert record.result_hash == result_hash(result)


def test_run_standardizes_dataframe_and_signals_dataframe(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    signals = from_target([0, 1, 1, 0, 0, 0, 0, 0])
    result = ube.run(
        md.to_dataframe(),
        signals.to_dataframe(),
        _config(),
        log_path=tmp_path / "experiments.db",
    )
    assert isinstance(result, BacktestResult)
    assert result.benchmark is not None


def test_run_rejects_non_backtest_config():
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    signals = from_target([0, 1, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ConfigError, match="BacktestConfig"):
        ube.run(md, signals, {"instrument": "nope"})  # type: ignore[arg-type]


def test_run_rejects_non_signal_and_non_dataframe_signals(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    with pytest.raises(InvalidSignalError, match="from_target"):
        ube.run(md, [0, 1, 0, 0], _config(), log_path=tmp_path / "experiments.db")


def test_run_rejects_unknown_engine(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=8)
    signals = from_target([0, 1, 0, 0, 0, 0, 0, 0])
    config = replace(_config(), engine="nonexistent")
    with pytest.raises(ConfigError, match="unknown engine"):
        ube.run(md, signals, config, log_path=tmp_path / "experiments.db")


def test_run_rejects_out_of_session_stock_bars(tmp_path):
    # §4.4: a bar whose timestamp falls where the declared trading calendar says the
    # market is closed is a CalendarMismatchError, never silently accepted. This locks in
    # the wiring of the previously-dead calendar check — 2024-01-01 is a New-Year holiday
    # (and overnight) for XNYS, so a stock bar there must be rejected up-front.
    md = MarketData(
        open=np.array([100.0]),
        high=np.array([101.0]),
        low=np.array([99.0]),
        close=np.array([100.0]),
        volume=np.full(1, 1.0),
        index=pd.DatetimeIndex(["2024-01-01 00:00:00+00:00"]).as_unit("ns"),
    )
    with pytest.raises(CalendarMismatchError):
        ube.run(
            md,
            from_target([1]),
            BacktestConfig(
                instrument=PRESETS["stocks"].instrument,
                engine_overrides={"starting_balance": 100000.0},
            ),
            log_path=tmp_path / "e.db",
        )


def test_run_rejects_portfolio_without_base_currency(tmp_path):
    # §4.7 / §7.1: a portfolio run (data is a Mapping of label -> bars) with no declared
    # base_currency is rejected up-front by BacktestConfig.validate(), which run() now
    # calls via config.validate(portfolio=isinstance(data, Mapping)). This locks in the
    # previously-orphaned validate() method — a portfolio run can no longer silently run
    # with an assumed base currency.
    a = synthetic_bars(PRESETS["stocks"], seed=1, n_bars=4)
    b = synthetic_bars(PRESETS["futures"], seed=2, n_bars=4)
    portfolio_data = {"STOCK": a, "FUT": b}
    config = BacktestConfig(
        instrument=PRESETS["stocks"].instrument,
        engine_overrides={"starting_balance": 100000.0},
    )  # base_currency deliberately undeclared
    with pytest.raises(UndeclaredConfigError, match="base_currency"):
        ube.run(
            portfolio_data,
            from_target([1, 0, 1, 0]),
            config,
            log_path=tmp_path / "e.db",
        )


def test_run_single_instrument_does_not_require_base_currency(tmp_path):
    # The §4.7 guard only fires for portfolio runs. A single-instrument run (data is a
    # MarketData, not a Mapping) must NOT raise UndeclaredConfigError for an undeclared
    # base_currency — the lone instrument's settlement currency is used instead (§7.1).
    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=8)
    config = BacktestConfig(
        instrument=PRESETS["stocks"].instrument,
        engine_overrides={"starting_balance": 100000.0},
    )  # no base_currency
    result = ube.run(
        md,
        from_target([1, 0, 1, 0, 1, 0, 1, 0]),
        config,
        log_path=tmp_path / "e.db",
    )
    assert result is not None


def _constant_bars(n: int = 10, start: str = "2024-01-01") -> MarketData:
    """Hourly constant-price bars with a known, tz-aware UTC index (for date_range tests)."""
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    prices = np.arange(10.0, 10.0 + n)
    return MarketData(
        open=prices,
        high=prices + 1.0,
        low=prices - 1.0,
        close=prices,
        volume=np.ones(n),
        index=idx,
    )


def test_run_date_range_restricts_bars_to_window(tmp_path):
    # §7.2: date_range must actually slice the data, not validate-and-ignore. With an
    # hourly 10-bar dataset and date_range ("... 02:00" .. "... 06:00"), only bars 2..6
    # (inclusive) reach the engine, so the resulting equity curve is exactly those 5 bars.
    md = _constant_bars(10)
    config = BacktestConfig(
        instrument=PRESETS["crypto_perp"].instrument,
        date_range=("2024-01-01 02:00", "2024-01-01 06:00"),
        engine_overrides={"starting_balance": 100000.0},
    )
    result = ube.run(
        md,
        from_target([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        config,
        log_path=tmp_path / "e.db",
    )
    eq_idx = result.equity_curve.index
    assert len(eq_idx) == 5
    assert eq_idx[0] == pd.Timestamp("2024-01-01 02:00", tz="UTC")
    assert eq_idx[-1] == pd.Timestamp("2024-01-01 06:00", tz="UTC")


def test_run_date_range_excluding_all_bars_raises_data_shape_error(tmp_path):
    # §7.2: a date_range that excludes every bar must raise DataShapeError — never return
    # an empty/zeroed result silently.
    md = _constant_bars(10)
    config = BacktestConfig(
        instrument=PRESETS["crypto_perp"].instrument,
        date_range=("2030-01-01", "2030-02-01"),
        engine_overrides={"starting_balance": 100000.0},
    )
    with pytest.raises(DataShapeError, match="excludes every bar"):
        ube.run(
            md,
            from_target([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            config,
            log_path=tmp_path / "e.db",
        )


def test_run_date_range_with_open_start_bound(tmp_path):
    # An open lower bound keeps everything up to `end`. date_range (None, "... 04:00") on
    # the 10-bar set keeps bars 0..4 inclusive → 5 bars ending at 04:00.
    md = _constant_bars(10)
    config = BacktestConfig(
        instrument=PRESETS["crypto_perp"].instrument,
        date_range=(None, "2024-01-01 04:00"),
        engine_overrides={"starting_balance": 100000.0},
    )
    result = ube.run(
        md,
        from_target([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        config,
        log_path=tmp_path / "e.db",
    )
    assert len(result.equity_curve.index) == 5
    assert result.equity_curve.index[-1] == pd.Timestamp("2024-01-01 04:00", tz="UTC")

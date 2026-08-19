"""Top-level ``ube.run`` orchestrator tests (§4.8, §7.1).

The orchestrator ties the Phase-1 core together: standardize the inputs, resolve the
engine, run the adapter, attach the benchmark, and record the run to the experiment log.
These tests exercise that pipeline end-to-end through the Nautilus adapter (the only
implemented engine) and verify the benchmark attachment and the unconditional
experiment-log record.
"""

from dataclasses import replace

import pytest

import ube
from ube.adapters.base import _REGISTRY
from ube.core.config import BacktestConfig
from ube.core.errors import ConfigError, InvalidSignalError
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

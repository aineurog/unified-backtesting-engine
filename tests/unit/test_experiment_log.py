"""Tests for the experiment log (§4.9, §18): DataReference + ExperimentLog."""

import importlib.metadata
import json
import sqlite3
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import BacktestRuntimeError, ConfigError, DataShapeError
from ube.core.experiment_log import DataReference, ExperimentLog, RecordInput
from ube.core.instrument import Instrument

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _inst() -> Instrument:
    return Instrument("BTC-USDT", asset_class="crypto_perp")


def _config() -> BacktestConfig:
    return BacktestConfig(instrument=_inst(), engine="vectorbt")


def _df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 100.0 + n, n),
            "high": np.linspace(101.0, 101.0 + n, n),
            "low": np.linspace(99.0, 99.0 + n, n),
            "close": np.linspace(100.5, 100.5 + n, n),
            "volume": np.arange(1, n + 1, dtype=float),
        },
        index=idx,
    )


def _md(n: int = 50) -> MarketData:
    return MarketData.from_dataframe(_df(n))


def _ref(n: int = 5) -> DataReference:
    return DataReference.from_market_data(_md(n), _inst())


def _record(
    log: ExperimentLog,
    run_id: str = "run-1",
    *,
    engine: str = "vectorbt",
    code_version: str | None = None,
    result_hash: str | None = None,
    timestamp: str | None = None,
) -> None:
    log.record(
        run_id=run_id,
        config=_config(),
        engine=engine,
        data_reference=_ref(),
        code_version=code_version,
        result_hash=result_hash,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# DataReference (§4.9).
# ---------------------------------------------------------------------------


def test_data_reference_from_market_data_fields():
    ref = _ref(5)
    assert ref.instrument.symbol == "BTC-USDT"
    assert ref.instrument.asset_class == "crypto_perp"
    assert ref.bar_type == "time"
    assert ref.row_count == 5
    assert ref.date_range[0] == "2024-01-01T00:00:00+00:00"
    assert ref.date_range[1] == "2024-01-01T04:00:00+00:00"
    assert isinstance(ref.content_hash, str) and len(ref.content_hash) == 64


def test_data_reference_event_bars():
    n = 5
    arr = np.column_stack(
        [
            np.linspace(100.0, 100.0 + n, n),
            np.linspace(101.0, 101.0 + n, n),
            np.linspace(99.0, 99.0 + n, n),
            np.linspace(100.5, 100.5 + n, n),
            np.arange(1, n + 1, dtype=float),
        ]
    )
    ref = DataReference.from_market_data(
        MarketData.from_array(arr, bar_type="volume"), _inst()
    )
    assert ref.bar_type == "volume"
    assert ref.row_count == 5
    assert ref.date_range == ("0", "4")


def test_data_reference_is_frozen():
    ref = _ref(5)
    with pytest.raises(FrozenInstanceError):
        ref.row_count = 99  # type: ignore[misc]


def test_data_reference_empty_market_data_raises():
    empty = MarketData.from_array(np.empty((0, 5)), bar_type="volume")
    with pytest.raises(DataShapeError):
        DataReference.from_market_data(empty, _inst())


def test_content_hash_deterministic_and_bounded_window():
    # Identical data -> identical hash.
    h1 = DataReference.from_market_data(_md(50), _inst()).content_hash
    h2 = DataReference.from_market_data(_md(50), _inst()).content_hash
    assert h1 == h2

    # A change in the head (first HASH_WINDOW rows) or tail (last HASH_WINDOW rows)
    # changes the hash; a middle-only change (outside the window) does not. Volume is
    # used so the OHLC invariants are never violated by the edit.
    head = _df(50)
    head.iloc[0, head.columns.get_loc("volume")] += 1.0
    tail = _df(50)
    tail.iloc[-1, tail.columns.get_loc("volume")] += 1.0
    middle = _df(50)
    middle.iloc[25, middle.columns.get_loc("volume")] += 1.0

    assert (
        DataReference.from_market_data(MarketData.from_dataframe(head), _inst()).content_hash
        != h1
    )
    assert (
        DataReference.from_market_data(MarketData.from_dataframe(tail), _inst()).content_hash
        != h1
    )
    # Index 25 sits strictly between the head [0:16) and tail [34:50) windows.
    assert (
        DataReference.from_market_data(MarketData.from_dataframe(middle), _inst()).content_hash
        == h1
    )


# ---------------------------------------------------------------------------
# Path resolution + WAL (§4.9, §18).
# ---------------------------------------------------------------------------


def test_env_var_path_used(monkeypatch, tmp_path):
    expected = tmp_path / "env.db"
    monkeypatch.setenv("BACKTEST_LOG_PATH", str(expected))
    log = ExperimentLog()
    log.close()
    assert expected.exists()


def test_default_path_uses_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BACKTEST_LOG_PATH", raising=False)
    log = ExperimentLog()
    log.close()
    assert (tmp_path / ".backtest" / "experiments.db").exists()


def test_explicit_path_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BACKTEST_LOG_PATH", str(tmp_path / "env.db"))
    explicit = tmp_path / "explicit.db"
    log = ExperimentLog(path=explicit)
    log.close()
    assert explicit.exists()
    assert not (tmp_path / "env.db").exists()


def test_parent_dir_auto_created(tmp_path):
    path = tmp_path / "nested" / "dir" / "exp.db"
    log = ExperimentLog(path=path)
    log.close()
    assert path.exists()


def test_bad_path_raises_backtest_runtime_error(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(BacktestRuntimeError):
        ExperimentLog(path=blocker / "sub" / "exp.db")


def test_wal_mode_enabled(tmp_path):
    db = tmp_path / "exp.db"
    log = ExperimentLog(path=db)
    conn = sqlite3.connect(str(db))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    log.close()
    assert mode == "wal"


# ---------------------------------------------------------------------------
# record / get / list / count round-trip.
# ---------------------------------------------------------------------------


def test_record_get_list_count_roundtrip(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(
        log,
        "run-1",
        result_hash="abc123",
        timestamp="2024-01-01T00:00:00+00:00",
    )
    assert log.count() == 1

    rec = log.get("run-1")
    assert rec is not None
    assert rec.run_id == "run-1"
    assert rec.timestamp == "2024-01-01T00:00:00+00:00"
    assert rec.engine == "vectorbt"
    assert rec.instrument == "BTC-USDT"
    assert rec.asset_class == "crypto_perp"
    assert rec.bar_type == "time"
    assert rec.row_count == 5
    assert rec.result_hash == "abc123"

    entries = log.list()
    assert len(entries) == 1
    assert entries[0].run_id == "run-1"

    assert log.get("missing") is None


def test_timestamp_defaults_to_now(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    assert log.get("run-1").timestamp  # a non-empty ISO-8601 string


def test_result_hash_nullable(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    assert log.get("run-1").result_hash is None


def test_list_most_recent_first_and_limit(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "a", timestamp="2024-01-01T00:00:00+00:00")
    _record(log, "b", timestamp="2024-01-02T00:00:00+00:00")
    assert [e.run_id for e in log.list()] == ["b", "a"]
    assert [e.run_id for e in log.list(limit=1)] == ["b"]
    assert log.count() == 2


def test_list_rejects_bad_limit(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    with pytest.raises(ConfigError):
        log.list(limit=-1)


def test_record_rejects_empty_run_id_and_engine(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    with pytest.raises(ConfigError):
        _record(log, "")
    with pytest.raises(ConfigError):
        _record(log, "run-1", engine="  ")


def test_record_rejects_wrong_types(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    with pytest.raises(ConfigError):
        log.record(run_id="x", config="nope", engine="vectorbt", data_reference=_ref())  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        log.record(run_id="x", config=_config(), engine="vectorbt", data_reference="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# params serialization (§4.9).
# ---------------------------------------------------------------------------


def test_params_json_roundtrip(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    params = json.loads(log.get("run-1").params)
    assert params["instrument"]["symbol"] == "BTC-USDT"
    assert params["instrument"]["asset_class"] == "crypto_perp"
    assert params["engine"] == "vectorbt"
    assert params["bar_type"] == "time"
    assert params["warmup_bars"] == 0
    assert params["base_currency"] is None
    assert params["risk"]["sizing"]["kind"] == "all_in"
    assert params["data_quality"]["missing_bar"] == "fail"


def test_params_deterministic(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    _record(log, "run-2")
    assert log.get("run-1").params == log.get("run-2").params


# ---------------------------------------------------------------------------
# Duplicate run_id — ignored, first write wins.
# ---------------------------------------------------------------------------


def test_duplicate_run_id_ignored_first_wins(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "dup", engine="vectorbt", result_hash="first")
    _record(log, "dup", engine="backtrader", result_hash="second")
    assert log.count() == 1
    rec = log.get("dup")
    assert rec.engine == "vectorbt"
    assert rec.result_hash == "first"


# ---------------------------------------------------------------------------
# record_many (§18 vectorbt batch).
# ---------------------------------------------------------------------------


def test_record_many_batch(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    ref = _ref()
    cfg = _config()
    log.record_many(
        [
            RecordInput(run_id="a", config=cfg, engine="vectorbt", data_reference=ref),
            RecordInput(run_id="b", config=cfg, engine="backtrader", data_reference=ref),
        ]
    )
    assert log.count() == 2
    assert [e.run_id for e in log.list()] == ["b", "a"]


def test_record_many_atomic_on_validation_error(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    ref = _ref()
    cfg = _config()
    good = RecordInput(run_id="g", config=cfg, engine="vectorbt", data_reference=ref)
    bad = RecordInput(run_id="", config=cfg, engine="vectorbt", data_reference=ref)
    with pytest.raises(ConfigError):
        log.record_many([good, bad])
    assert log.count() == 0


# ---------------------------------------------------------------------------
# code_version (§4.9).
# ---------------------------------------------------------------------------


def test_code_version_auto_detect(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    assert log.get("run-1").code_version == importlib.metadata.version(
        "unified-backtesting-engine"
    )


def test_code_version_override(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1", code_version="1.2.3")
    assert log.get("run-1").code_version == "1.2.3"


def test_code_version_fallback_unknown(monkeypatch, tmp_path):
    def _not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _not_found)
    log = ExperimentLog(path=tmp_path / "exp.db")
    _record(log, "run-1")
    assert log.get("run-1").code_version == "unknown"

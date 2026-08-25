"""Tests for the experiment log (§4.9, §18): DataReference + ExperimentLog."""

import importlib.metadata
import json
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import BacktestRuntimeError, ConfigError, DataShapeError
from ube.core.experiment_log import (
    DataReference,
    ExperimentLog,
    PortfolioDataReference,
    RecordInput,
    _expand_home,
    _resolve_path,
)
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
    assert ref.row_count == 5
    assert ref.date_range[0] == "2024-01-01T00:00:00+00:00"
    assert ref.date_range[1] == "2024-01-01T04:00:00+00:00"
    assert isinstance(ref.content_hash, str) and len(ref.content_hash) == 64


def test_data_reference_from_array_with_timestamps():
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
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ref = DataReference.from_market_data(
        MarketData.from_array(arr, timestamps=ts), _inst()
    )
    assert ref.row_count == 5
    assert ref.date_range == ("2024-01-01T00:00:00+00:00", "2024-01-01T04:00:00+00:00")


def test_data_reference_is_frozen():
    ref = _ref(5)
    with pytest.raises(FrozenInstanceError):
        ref.row_count = 99  # type: ignore[misc]


def test_data_reference_empty_market_data_raises():
    # A zero-bar view (head(0)) is the one public path to an empty MarketData now that
    # the constructors reject empty input; DataReference must still refuse it.
    empty = _md(1).head(0)
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
    assert params["warmup_bars"] == 0
    assert params["base_currency"] is None
    assert params["risk"]["sizing"]["kind"] == "all_in"


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


# ---------------------------------------------------------------------------
# _resolve_path / _expand_home (§4.8): explicit > env var > default, `~` honors $HOME.
# ---------------------------------------------------------------------------


def test_expand_home_tilde_honors_home_env(monkeypatch):
    monkeypatch.setenv("HOME", "C:/fake/home")
    assert _expand_home("~") == Path("C:/fake/home")


def test_expand_home_tilde_slash_appends_relative_path(monkeypatch):
    monkeypatch.setenv("HOME", "C:/fake/home")
    assert _expand_home("~/ube/exp.db") == Path("C:/fake/home/ube/exp.db")


def test_expand_home_tilde_falls_back_when_home_unset(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    assert _expand_home("~") == Path.home()


def test_expand_home_plain_path_is_unchanged():
    assert _expand_home("relative/dir/exp.db") == Path("relative/dir/exp.db")
    assert _expand_home("C:/abs/exp.db") == Path("C:/abs/exp.db")


def test_resolve_path_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("BACKTEST_LOG_PATH", "C:/env/exp.db")
    monkeypatch.setenv("HOME", "C:/fake/home")
    assert _resolve_path("~/ube/exp.db") == Path("C:/fake/home/ube/exp.db")


def test_resolve_path_env_var_when_no_explicit(monkeypatch):
    monkeypatch.setenv("BACKTEST_LOG_PATH", "~/env/exp.db")
    monkeypatch.setenv("HOME", "C:/fake/home")
    assert _resolve_path(None) == Path("C:/fake/home/env/exp.db")


def test_resolve_path_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv("BACKTEST_LOG_PATH", raising=False)
    monkeypatch.setenv("HOME", "C:/fake/home")
    assert _resolve_path(None) == Path("C:/fake/home/.backtest/experiments.db")


# ---------------------------------------------------------------------------
# PortfolioDataReference (§4.8 multi-instrument fingerprint).
# ---------------------------------------------------------------------------


def _inst(symbol: str = "BTC-USDT", asset_class: str = "crypto_perp") -> Instrument:
    return Instrument(symbol, asset_class=asset_class)


def _md(n: int = 50) -> MarketData:
    return MarketData.from_dataframe(_df(n))


def test_portfolio_data_reference_from_mapping_basic():
    inst_a = _inst("SYM_A", "crypto_perp")
    inst_b = _inst("SYM_B", "crypto_spot")
    mds = {"SYM_A": _md(10), "SYM_B": _md(20)}
    inst_map = {"SYM_A": inst_a, "SYM_B": inst_b}
    ref = PortfolioDataReference.from_market_data_mapping(mds, inst_map)
    assert ref.symbols == ("SYM_A", "SYM_B")
    assert ref.instruments == (inst_a, inst_b)
    assert ref.row_count == 30
    assert ref.date_range[0] == "2024-01-01T00:00:00+00:00"
    assert ref.date_range[1] == "2024-01-01T19:00:00+00:00"  # 20 bars (0..19) = 19 hours
    assert isinstance(ref.content_hash, str) and len(ref.content_hash) == 64


def test_portfolio_data_reference_content_hash_deterministic():
    mds = {"SYM_A": _md(10), "SYM_B": _md(20)}
    h1 = PortfolioDataReference.from_market_data_mapping(mds).content_hash
    h2 = PortfolioDataReference.from_market_data_mapping(mds).content_hash
    assert h1 == h2
    # Changing one instrument's data changes the combined hash
    mds2 = {"SYM_A": _md(10), "SYM_B": _md(20)}
    df = _df(10)
    df.iloc[0, df.columns.get_loc("open")] += 1.0
    mds2["SYM_A"] = MarketData.from_dataframe(df)
    h3 = PortfolioDataReference.from_market_data_mapping(mds2).content_hash
    assert h3 != h1


def test_portfolio_data_reference_symbol_sorting():
    # Input order must not matter; symbols are sorted for deterministic identity.
    mds = {"SYM_B": _md(5), "SYM_A": _md(5)}
    ref = PortfolioDataReference.from_market_data_mapping(mds)
    assert ref.symbols == ("SYM_A", "SYM_B")


def test_portfolio_data_reference_empty_mapping_raises():
    with pytest.raises(DataShapeError):
        PortfolioDataReference.from_market_data_mapping({})


def test_portfolio_data_reference_zero_bar_raises():
    mds = {"SYM_A": _md(5)}
    mds["SYM_A"] = mds["SYM_A"].head(0)
    with pytest.raises(DataShapeError):
        PortfolioDataReference.from_market_data_mapping(mds)


def test_portfolio_data_reference_non_marketdata_raises():
    mds = {"SYM_A": "not a MarketData"}
    with pytest.raises(DataShapeError):
        PortfolioDataReference.from_market_data_mapping(mds)  # type: ignore[arg-type]


def test_portfolio_data_reference_instrument_map_optional():
    # instruments_map is optional; asset_class falls back to "portfolio" in the log row.
    mds = {"SYM_A": _md(5), "SYM_B": _md(5)}
    ref = PortfolioDataReference.from_market_data_mapping(mds)
    assert ref.symbols == ("SYM_A", "SYM_B")
    assert ref.instruments == ()


def test_portfolio_data_reference_frozen():
    ref = PortfolioDataReference.from_market_data_mapping({"SYM_A": _md(5)})
    with pytest.raises(FrozenInstanceError):
        ref.row_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExperimentLog accepts PortfolioDataReference and stores joined labels.
# ---------------------------------------------------------------------------


def test_experiment_log_records_portfolio_reference(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    mds = {"SYM_A": _md(5), "SYM_B": _md(10)}
    ref = PortfolioDataReference.from_market_data_mapping(mds)
    log.record(
        run_id="run-portfolio",
        config=_config(),
        engine="vectorbt",
        data_reference=ref,
        result_hash="abc123",
    )
    assert log.count() == 1
    rec = log.get("run-portfolio")
    assert rec is not None
    assert rec.instrument == "SYM_A|SYM_B"
    assert rec.asset_class == "portfolio"
    assert rec.row_count == 15
    assert rec.result_hash == "abc123"


def test_experiment_log_records_portfolio_reference_with_instruments(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    inst_a = _inst("SYM_A", "crypto_perp")
    inst_b = _inst("SYM_B", "stocks")
    mds = {"SYM_A": _md(5), "SYM_B": _md(10)}
    inst_map = {"SYM_A": inst_a, "SYM_B": inst_b}
    ref = PortfolioDataReference.from_market_data_mapping(mds, inst_map)
    log.record(
        run_id="run-portfolio",
        config=_config(),
        engine="vectorbt",
        data_reference=ref,
    )
    rec = log.get("run-portfolio")
    assert rec.asset_class == "crypto_perp|stocks"  # sorted union


def test_experiment_log_rejects_bad_data_reference(tmp_path):
    log = ExperimentLog(path=tmp_path / "exp.db")
    with pytest.raises(ConfigError):
        log.record(
            run_id="bad",
            config=_config(),
            engine="vectorbt",
            data_reference="not a reference",  # type: ignore[arg-type]
        )

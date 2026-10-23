"""Unit tests for :mod:`ube.stats` (§10 — StratStat integration).

These stay engine-free and ledger-free: the tiers are exercised through the generic
:func:`ube.stats.compute` / :func:`ube.stats.from_arrays` assembly points, and the sqlite
path through a temporary database. The end-to-end wiring (``ube.run``, paper ``step``) is
covered by ``tests/integration/test_run_stats.py`` and
``tests/integration/test_papertrading_stats.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import ube.stats as stats
from ube.core.benchmark import BenchmarkCurve
from ube.core.ledger import Trade


def _index(n: int = 250, freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def _returns(n: int = 250, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.011, n), index=_index(n))


def _trade(i: int = 0, *, side: int = 1, net: float = 250.0, notional: float = 10_000.0) -> Trade:
    t = int(_index(10)[i].value)
    move = (net / notional * 100.0 * side) if notional else 0.0
    return Trade(
        instrument_id="ES",
        side=side,
        quantity=1.0,
        entry_timestamp=t,
        exit_timestamp=t + 86_400_000_000_000,
        entry_price=100.0,
        exit_price=100.0 + move,
        gross_pnl=net,
        commission=0.0,
        funding=0.0,
        net_pnl=net,
        entry_notional=notional,
        exit_notional=notional,
    )


def _same(left: float, right: float) -> bool:
    """NaN-tolerant float equality (undefined metrics are legitimately ``NaN``)."""
    if math.isnan(left) and math.isnan(right):
        return True
    return left == pytest.approx(right)


def test_compute_returns_tier_only() -> None:
    s = stats.compute(_returns(), periods_per_year=252)
    assert "returns" in s.tiers
    assert s.tiers == ("returns",)
    assert s.errors == {}
    assert s.periods_per_year == pytest.approx(252.0)
    assert s.n_metrics > 40
    assert "sharpe_ratio" in s.values
    assert math.isfinite(s.values["sharpe_ratio"])
    assert s.meta["input"]["returns"] == 250
    assert s.meta["input"]["trades"] == 0


def test_periods_per_year_inferred_from_daily_grid() -> None:
    s = stats.compute(_returns(n=120))
    assert s.periods_per_year == pytest.approx(stats.SECONDS_PER_YEAR / 86_400.0, rel=1e-9)
    assert isinstance(s.meta["stratstat"], dict)


def test_compute_without_returns_degrades_instead_of_raising() -> None:
    s = stats.compute(pd.Series(dtype=float))
    assert s.n_metrics == 0
    assert s.tiers == ()
    assert "compute" in s.errors
    assert s.periods_per_year is None


def test_single_point_grid_cannot_produce_returns() -> None:
    s = stats.from_arrays(_index(1), np.array([10_000.0]), trades=[])
    assert "returns" not in s.tiers
    assert "returns" in s.errors
    assert s.n_metrics == 0


def test_return_policies_handle_insolvent_points() -> None:
    raw = _returns(n=10)
    raw.iloc[4] = np.nan
    raw.iloc[5] = np.inf

    truncated = stats.compute(raw, periods_per_year=252)
    assert truncated.meta["input"]["returns"] == 4
    assert truncated.meta["dropped_returns"] == 6

    zeroed = stats.compute(raw, periods_per_year=252, return_policy="zero")
    assert zeroed.meta["input"]["returns"] == 10
    assert zeroed.meta["dropped_returns"] == 2

    dropped = stats.compute(raw, periods_per_year=252, return_policy="drop")
    assert dropped.meta["input"]["returns"] == 8
    assert dropped.meta["dropped_returns"] == 2

    with pytest.raises(ValueError, match="return_policy"):
        stats.compute(raw, return_policy="keep")


def test_trades_tier_uses_fraction_pnl_and_bar_durations() -> None:
    s = stats.compute(
        _returns(n=100),
        trades=[_trade(0, net=250.0), _trade(1, side=-1, net=-120.0)],
        periods_per_year=252,
    )
    assert "trades" in s.tiers
    assert s.meta["input"]["trades"] == 2
    assert "win_rate" in s.values
    assert s.values["win_rate"] == pytest.approx(0.5)
    # duration is in bar periods, derived from the daily grid
    assert "avg_holding_period" in s.values
    assert s.values["avg_holding_period"] == pytest.approx(1.0, abs=1e-6)


def test_trades_tier_falls_back_to_currency_without_notional() -> None:
    s = stats.compute(
        _returns(n=100),
        trades=[_trade(0, net=250.0, notional=0.0)],
        periods_per_year=252,
    )
    assert "trades" in s.tiers
    assert "win_rate" in s.values


def test_from_arrays_benchmark_tier_and_positional_alignment() -> None:
    rets = _returns(n=60)
    bench = BenchmarkCurve(
        equity=np.cumprod(1.0 + _returns(n=60, seed=9).to_numpy()),
        returns=np.r_[0.0, _returns(n=59, seed=9).to_numpy()],
    )
    equity = np.cumprod(1.0 + rets.to_numpy()) * 10_000.0
    s = stats.from_arrays(rets.index, equity, trades=[], benchmark=bench)
    assert "benchmark" in s.tiers
    assert s.meta["input"]["benchmark"] == 60
    assert "alpha" in s.values
    assert "information_ratio" in s.values


def test_from_arrays_uses_persisted_returns_when_supplied(monkeypatch) -> None:
    expected_returns = np.array([0.0, 0.01, -0.02])
    captured: dict[str, pd.Series] = {}
    original_compute = stats.compute

    def capture_compute(returns, **kwargs):
        captured["returns"] = returns.copy()
        return original_compute(returns, **kwargs)

    monkeypatch.setattr(stats, "compute", capture_compute)
    result = stats.from_arrays(
        _index(3),
        np.array([100.0, 200.0, 50.0]),
        trades=(),
        equity_returns=expected_returns,
    )

    assert "returns" in result.tiers
    np.testing.assert_allclose(captured["returns"].to_numpy(), expected_returns)


def test_with_benchmark_merges_and_keeps_first_wins() -> None:
    rets = _returns(n=60)
    base = stats.compute(rets, periods_per_year=252)
    bench_returns = np.r_[0.0, _returns(n=59, seed=3).to_numpy()]

    merged = base.with_benchmark(
        BenchmarkCurve(bench_returns, bench_returns),
        returns=rets,
        index=rets.index,
    )
    assert "benchmark" in merged.tiers
    assert merged.meta["input"]["benchmark"] == 60
    assert "benchmark" in base.tiers or "benchmark" not in base.tiers  # base untouched
    assert "benchmark" not in base.tiers
    assert len(merged.values) > len(base.values)
    for name, value in base.values.items():
        assert _same(merged.values[name], value)


def test_results_are_json_and_pickle_safe() -> None:
    s = stats.compute(_returns(n=80), trades=[_trade(0)], periods_per_year=252)
    payload = s.to_dict()
    json.dumps(payload)  # raises if an ndarray slipped through
    assert all(not isinstance(v, np.ndarray) for v in s.objects.values())
    assert all(
        isinstance(v, (list, dict, str, int, float, bool, type(None))) for v in s.objects.values()
    )
    assert len(s.to_records()) == s.n_metrics
    import pickle

    restored = pickle.loads(pickle.dumps(s))
    assert restored.values.keys() == s.values.keys()
    assert all(_same(restored.values[name], value) for name, value in s.values.items())


def test_category_table_and_lookup() -> None:
    s = stats.compute(_returns(n=80), periods_per_year=252)
    categories = {s.categories[name].split(".")[0] for name in s.values}
    assert {"risk", "risk_adjusted", "descriptive"} <= categories
    table = s.table()
    assert list(table.columns)[:3] == ["metric", "value", "category"]
    assert len(table) == s.n_metrics
    assert s.get("definitely_not_a_metric") is None
    assert s["sharpe_ratio"] == s.values["sharpe_ratio"]
    with pytest.raises(KeyError):
        s["definitely_not_a_metric"]


def test_save_and_load_round_trip(tmp_path) -> None:
    db = tmp_path / "paper.db"
    first = stats.compute(_returns(n=80), periods_per_year=252, computed_at_ns=111)
    second = stats.compute(_returns(n=90), periods_per_year=252, computed_at_ns=222)

    assert stats.save_stats(db, "run", first, trade_count=1) == 1
    assert stats.save_stats(db, "run", second, trade_count=2) == 2

    latest = stats.load_stats(db, "run")
    assert len(latest) == second.n_metrics
    assert set(latest["seq"]) == {2}
    assert set(latest["trade_count"]) == {2}
    assert set(latest["as_of_ns"]) == {222}
    row = latest[latest["metric"] == "sharpe_ratio"].iloc[0]
    assert row["value"] == pytest.approx(second.values["sharpe_ratio"])
    assert row["ref"]

    old = stats.load_stats(db, "run", seq=1)
    assert set(old["as_of_ns"]) == {111}

    assert stats.load_stats(tmp_path / "missing.db", "run").empty
    with pytest.raises(TypeError, match="StatsResult"):
        stats.save_stats(db, "run", {"not": "a result"})  # type: ignore[arg-type]


def test_save_and_load_persists_equity_point_count(tmp_path) -> None:
    db = tmp_path / "paper.db"
    snapshot = stats.compute(
        _returns(n=3),
        periods_per_year=252,
        extra_meta={"n_equity_points": 3},
    )
    stats.save_stats(db, "run", snapshot, trade_count=2)

    loaded = stats.load_stats(db, "run")
    assert set(loaded["trade_count"]) == {2}
    assert set(loaded["n_equity_points"]) == {3}


def test_load_stats_migrates_legacy_meta_schema(tmp_path) -> None:
    import sqlite3

    db = tmp_path / "legacy-stats.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE stats (run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "metric TEXT NOT NULL, value REAL, value_json TEXT, category TEXT, ref TEXT, "
        "PRIMARY KEY (run_id, seq, metric))"
    )
    conn.execute(
        "CREATE TABLE stats_meta (run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "as_of_ns INTEGER NOT NULL, trade_count INTEGER NOT NULL, periods_per_year REAL, "
        "tiers TEXT, errors_json TEXT, PRIMARY KEY (run_id, seq))"
    )
    conn.execute(
        "INSERT INTO stats_meta VALUES ('run', 1, 10, 2, 252, 'returns', '{}')"
    )
    conn.execute(
        "INSERT INTO stats VALUES ('run', 1, 'sharpe_ratio', 1.5, NULL, 'risk', NULL)"
    )
    conn.commit()
    conn.close()

    loaded = stats.load_stats(db, "run")
    assert set(loaded["trade_count"]) == {2}
    assert set(loaded["n_equity_points"]) == {0}


def test_enable_switch_round_trip(monkeypatch) -> None:
    try:
        stats.set_enabled(False)
        assert stats.is_enabled() is False
        stats.set_enabled(True)
        assert stats.is_enabled() is True
    finally:
        stats.set_enabled(True)


def test_empty_extra_meta_is_ignored_by_from_arrays() -> None:
    s = stats.from_arrays(_index(3), np.array([100.0, 101.0, 102.0]), trades=[], extra_meta=None)
    assert "equity_source" not in s.meta
    s2 = stats.from_arrays(
        _index(3), np.array([100.0, 101.0, 102.0]), trades=[], extra_meta={"equity_source": "x"}
    )
    assert s2.meta["equity_source"] == "x"

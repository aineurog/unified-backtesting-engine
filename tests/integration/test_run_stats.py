"""Integration: ``result.stats`` is computed eagerly by ``ube.run`` (§10).

Exercises the real nautilus path end to end, so the four tiers are assembled from the
cached result views (equity, trades, exposure from the ledger) plus the benchmark that
``run()`` attaches after the adapter returns.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")

import ube  # noqa: E402
from ube.core.config import BacktestConfig  # noqa: E402
from ube.core.risk import RiskConfig  # noqa: E402
from ube.core.risk.sizing import SizeModel  # noqa: E402
from ube.core.signals import from_target  # noqa: E402
from ube.stats import StatsResult  # noqa: E402
from ube.testing.synthetic import PRESETS, synthetic_bars  # noqa: E402


def _config() -> BacktestConfig:
    return BacktestConfig(
        instrument=PRESETS["futures"].instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine="nautilus",
        engine_overrides={"starting_balance": 100_000.0},
    )


def _run(tmp_path):
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=300)
    signals = from_target([0] + [1] * 100 + [0] * 100 + [-1] * 99)
    return ube.run(md, signals, _config(), log_path=tmp_path / "experiments.db")


def test_run_populates_stats_with_all_four_tiers(tmp_path) -> None:
    result = _run(tmp_path)

    assert result.metrics is None  # legacy field stays untouched
    assert isinstance(result.stats, StatsResult)
    s = result.stats
    assert set(s.tiers) == {"returns", "trades", "benchmark", "exposure"}
    assert s.errors == {}
    assert s.n_metrics > 100

    # every tier fed something
    assert s.meta["input"]["returns"] == 300
    assert s.meta["input"]["benchmark"] == 300
    assert s.meta["input"]["exposure"] == 300
    assert s.meta["input"]["trades"] >= 1
    assert s.meta["dropped_returns"] == 0

    # spot-check values from different tiers
    assert s.values["max_drawdown"] <= 0.0
    assert np.isfinite(s.values["sharpe_ratio"])
    assert s.values["beta"] == pytest.approx(0.0, abs=1.0)
    assert 0.0 <= s.values["win_rate"] <= 1.0
    assert "sharpe_ratio" in s.meta["refs"]
    assert s.computed_at_ns > 0


def test_stats_survive_pickle_and_are_json_safe(tmp_path) -> None:
    result = _run(tmp_path)
    assert result.stats is not None

    restored = pickle.loads(pickle.dumps(result))
    assert restored.stats is not None
    assert restored.stats.values.keys() == result.stats.values.keys()
    assert restored.stats.tiers == result.stats.tiers

    import json

    json.dumps(result.stats.to_dict())


def test_stats_can_be_switched_off(tmp_path) -> None:
    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=40)
    signals = from_target([0] + [1] * 20 + [0] * 19)
    try:
        ube.stats.set_enabled(False)
        result = ube.run(md, signals, _config(), log_path=tmp_path / "off.db")
        assert result.stats is None
        assert result.metrics is None
    finally:
        ube.stats.set_enabled(True)

    again = ube.run(md, signals, _config(), log_path=tmp_path / "on.db")
    assert isinstance(again.stats, StatsResult)


def test_from_result_recomputes_from_cached_views(tmp_path) -> None:
    result = _run(tmp_path)
    assert result.stats is not None

    recomputed = ube.stats.from_result(result)
    for name in ("sharpe_ratio", "max_drawdown", "cumulative_return"):
        assert recomputed.values[name] == pytest.approx(result.stats.values[name], nan_ok=True)
    assert "benchmark" in recomputed.tiers

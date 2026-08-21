"""Smoke test: the package imports and exposes its user-facing surface (§7.1)."""

import ube


def test_version_is_non_empty_string():
    assert isinstance(ube.__version__, str)
    assert ube.__version__


def test_top_level_exposes_user_facing_surface():
    # The orchestrator plus every type needed to compose a run is importable from `ube`.
    expected = {
        "run",
        "ensure_builtin_engines_registered",
        "BacktestConfig",
        "SignalConfig",
        "Instrument",
        "MarketData",
        "Signals",
        "from_target",
        "from_callable",
        "RiskConfig",
        "SizeModel",
        "TakeProfit",
        "StopLoss",
        "ATRStop",
        "TrailingStop",
        "TimeExit",
        "ChandelierExit",
        "BenchmarkConfig",
        "CostModel",
    }
    assert expected <= set(ube.__all__)
    for name in expected:
        assert getattr(ube, name) is not None, name

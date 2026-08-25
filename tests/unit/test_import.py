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


def test_signals_submodule_is_exposed():
    # §6.2 documents ``ube.signals.from_target(...)`` — the module itself must be reachable
    # under that name, not just its members (``ube.from_target``).
    import importlib

    assert hasattr(ube, "signals")
    assert hasattr(ube.signals, "from_target")
    assert hasattr(ube.signals, "from_callable")
    assert hasattr(ube.signals, "Signals")
    # And ``import ube.signals`` must resolve (registered in sys.modules).
    mod = importlib.import_module("ube.signals")
    assert mod is ube.signals
    sig = ube.signals.from_target([0, 1, 0, -1, 0])
    assert sig.n_bars == 5


def test_paper_is_not_stubbed():
    # §9 (paper trading) is a separate, larger piece of work and must NOT be stubbed as an
    # empty namespace. Absence is the correct state until real work backs it (per the punch
    # list scope).
    assert not hasattr(ube, "paper")

"""Smoke test: the package imports and exposes its user-facing surface (§7.1)."""

import importlib

import pytest

import ube
from tests._engines import engine_is_installed
from ube.core.errors import EngineError


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
    assert hasattr(ube, "signals")
    assert hasattr(ube.signals, "from_target")
    assert hasattr(ube.signals, "from_callable")
    assert hasattr(ube.signals, "Signals")
    # And ``import ube.signals`` must resolve (registered in sys.modules).
    mod = importlib.import_module("ube.signals")
    assert mod is ube.signals
    sig = ube.signals.from_target([0, 1, 0, -1, 0])
    assert sig.n_bars == 5


def test_ensure_builtin_engines_registered_registers_only_installed_engines():
    # Registration must follow the *engine package*, not the adapter module: the adapters
    # import their engine lazily, so importing ``..._adapter.adapter`` succeeds even when
    # the engine is absent. Registering such an adapter makes "auto" resolve to an engine
    # that cannot run (this is what broke the nautilus-only CI job).
    before = set(ube.registered_engines())
    ube.ensure_builtin_engines_registered()
    after = set(ube.registered_engines())
    installed = {
        name for name in ("vectorbt", "backtrader", "nautilus") if engine_is_installed(name)
    }
    assert after - before <= installed
    assert installed <= after


def test_paper_is_not_stubbed():
    # §9 (paper trading) is implemented under ``ube.paper`` and must NOT be a stub. The
    # engine-agnostic front-end (init/step/run_auto) is core, so it is asserted
    # unconditionally — it has to be there with no engine installed at all.
    assert hasattr(ube, "paper")
    for name in ("init", "step", "run_auto"):
        assert callable(getattr(ube.paper, name))

    from ube.papertrading.core import get_paper_engine

    if not engine_is_installed("nautilus"):
        # "Not a stub" cuts the other way in an engine-less install: resolving the nautilus
        # backend must fail with the §15 error that names the missing extra, never return a
        # placeholder object that would let a paper session start and then do nothing.
        with pytest.raises(EngineError, match="nautilus-trader"):
            get_paper_engine("nautilus")
    else:
        assert get_paper_engine("nautilus") is not None

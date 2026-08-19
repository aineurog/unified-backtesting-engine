"""Unified Backtesting Engine — one interface and one output across engines.

The package exposes its user-facing surface at the top level so a typical run needs only
``import ube`` (§7.1): the orchestrator (:func:`ube.run`), the engine registry, and the
core types/functions needed to compose a :class:`~ube.core.config.BacktestConfig`
(instrument, data, signals, risk/exits, cost, benchmark) and build its signals.
"""

from ube.adapters import (
    AUTO_ENGINE_ORDER,
    EngineAdapter,
    get_engine,
    register_engine,
    registered_engines,
    resolve_engine_name,
)
from ube.core.benchmark import BenchmarkConfig
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.cost import CostModel
from ube.core.data import MarketData
from ube.core.instrument import Instrument
from ube.core.risk import (
    ATRStop,
    ChandelierExit,
    RiskConfig,
    SizeModel,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
)
from ube.core.signals import Signals, from_callable, from_target
from ube.run import ensure_builtin_engines_registered, run

__version__ = "0.1.0"

__all__ = [
    "AUTO_ENGINE_ORDER",
    "ATRStop",
    "BacktestConfig",
    "BenchmarkConfig",
    "ChandelierExit",
    "CostModel",
    "EngineAdapter",
    "Instrument",
    "MarketData",
    "RiskConfig",
    "SignalConfig",
    "Signals",
    "SizeModel",
    "StopLoss",
    "TakeProfit",
    "TimeExit",
    "TrailingStop",
    "__version__",
    "ensure_builtin_engines_registered",
    "from_callable",
    "from_target",
    "get_engine",
    "register_engine",
    "registered_engines",
    "resolve_engine_name",
    "run",
]

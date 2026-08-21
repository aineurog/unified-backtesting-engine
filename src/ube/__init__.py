"""Unified Backtesting Engine — one interface and one output across engines."""

from ube.adapters import (
    AUTO_ENGINE_ORDER,
    EngineAdapter,
    get_engine,
    register_engine,
    registered_engines,
)
from ube.core.benchmark import BenchmarkConfig
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.cost import CostModel
from ube.core.data import MarketData
from ube.core.instrument import Instrument
from ube.core.risk import (
    ATRStop,
    ChandelierExit,
    Exit,
    RiskConfig,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
)
from ube.core.risk.sizing import SizeModel
from ube.core.signals import Signals, from_callable, from_target
from ube.run import run

__version__ = "0.1.0"

__all__ = [
    "AUTO_ENGINE_ORDER",
    "BacktestConfig",
    "BenchmarkConfig",
    "CostModel",
    "EngineAdapter",
    "Instrument",
    "MarketData",
    "RiskConfig",
    "SignalConfig",
    "Signals",
    "SizeModel",
    "ATRStop",
    "ChandelierExit",
    "Exit",
    "StopLoss",
    "TakeProfit",
    "TimeExit",
    "TrailingStop",
    "__version__",
    "from_callable",
    "from_target",
    "get_engine",
    "register_engine",
    "registered_engines",
    "run",
]
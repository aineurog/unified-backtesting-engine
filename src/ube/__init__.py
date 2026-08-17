"""Unified Backtesting Engine — one interface and one output across engines."""

from ube.adapters import (
    AUTO_ENGINE_ORDER,
    EngineAdapter,
    get_engine,
    register_engine,
    registered_engines,
)

__version__ = "0.1.0"

__all__ = [
    "AUTO_ENGINE_ORDER",
    "EngineAdapter",
    "__version__",
    "get_engine",
    "register_engine",
    "registered_engines",
]
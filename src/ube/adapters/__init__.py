"""Engine adapters — the base contract plus one adapter per backtesting engine (§4.1)."""

from ube.adapters.base import (
    AUTO_ENGINE_ORDER,
    EngineAdapter,
    get_engine,
    register_engine,
    registered_engines,
    resolve_engine_name,
)

__all__ = [
    "AUTO_ENGINE_ORDER",
    "EngineAdapter",
    "get_engine",
    "register_engine",
    "registered_engines",
    "resolve_engine_name",
]

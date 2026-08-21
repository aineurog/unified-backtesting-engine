"""NautilusTrader adapter package (dev-guide §4.5)."""

from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
from ube.adapters.nautilus_adapter.overrides import (
    DEFAULT_OMS_TYPE,
    DEFAULT_STARTING_BALANCE,
    DEFAULT_VENUE,
    NautilusEngineOverrides,
    validate_overrides,
)

__all__ = [
    "DEFAULT_OMS_TYPE",
    "DEFAULT_STARTING_BALANCE",
    "DEFAULT_VENUE",
    "NautilusAdapter",
    "NautilusEngineOverrides",
    "validate_overrides",
]
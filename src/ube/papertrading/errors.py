"""Re-exports of the paper-trading error hierarchy (§15).

The exceptions themselves live in :mod:`ube.core.errors` (the single §15 source); this
module just surfaces the paper-relevant ones so callers can ``from ube.paper.errors
import ...``. New paper-specific errors (beyond the existing ``PaperTradingError`` /
``DuplicateBarError`` / ``StateCorruptionError``) would be added in ``core.errors`` and
re-exported here.
"""

from __future__ import annotations

from ube.core.errors import (
    DuplicateBarError,
    EngineError,
    PaperTradingError,
    StateCorruptionError,
)

__all__ = [
    "DuplicateBarError",
    "EngineError",
    "PaperTradingError",
    "StateCorruptionError",
]

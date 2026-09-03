"""Paper trading for ``ube`` (§9).

Engine-agnostic front-end (``init`` / ``step`` / ``run_auto``) plus a Nautilus backend
that drives Nautilus's own ``SandboxExecutionClient`` as its execution substrate (plan
§0). ``core`` never imports the nautilus backend; the ``"nautilus"`` engine is
registered lazily on first use (see :func:`ube.papertrading.core.get_paper_engine`).

The ``"recording"`` engine is a dependency-free reference broker used by unit tests of
the engine-agnostic logic (plan T2).
"""

from ube.papertrading.config import PaperConfig
from ube.papertrading.core import (
    PaperEngine,
    RecordingBackend,
    get_paper_engine,
    init,
    register_paper_engine,
    run,
    run_auto,
    step,
)
from ube.papertrading.errors import (
    DuplicateBarError,
    EngineError,
    PaperTradingError,
    StateCorruptionError,
)
from ube.papertrading.state import OpenPosition, PaperState

# Register the dependency-free test backend (plan T2).
register_paper_engine("recording", RecordingBackend)

__all__ = [
    "DuplicateBarError",
    "EngineError",
    "OpenPosition",
    "PaperConfig",
    "PaperEngine",
    "PaperState",
    "PaperTradingError",
    "RecordingBackend",
    "StateCorruptionError",
    "get_paper_engine",
    "init",
    "register_paper_engine",
    "run",
    "run_auto",
    "step",
]

"""vectorbt paper-trading backend (lazy-imported by ``core``).

Importing this package self-registers the ``"vectorbt"`` engine and its
:class:`~ube.papertrading.vbt.state.VbtPaperState` state class (see ``.backend``). It must
never be imported eagerly by ``core``/``__init__`` — only when
``PaperConfig(engine="vectorbt")`` (or the vectorbt state class) is requested, so the core
never hard-depends on the optional ``vectorbt`` package.
"""

from ube.papertrading.vbt.backend import VbtPaperEngine
from ube.papertrading.vbt.state import VbtPaperState

__all__ = ["VbtPaperEngine", "VbtPaperState"]

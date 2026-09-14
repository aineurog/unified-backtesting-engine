"""Nautilus paper-trading backend (plan T3, lazy-imported by ``core``).

Importing this package self-registers the ``"nautilus"`` backend in the paper-engine
registry (see ``.backend``). It must never be imported eagerly by ``core``/``__init__`` —
only when ``PaperConfig(engine="nautilus")`` is requested.
"""

from ube.papertrading.nautilus.backend import NautilusPaperEngine

__all__ = ["NautilusPaperEngine"]

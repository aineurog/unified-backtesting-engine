"""Backtrader paper-trading backend.

Importing this package self-registers the ``"backtrader"`` engine and its
:class:`BacktraderPaperState` state class, exactly as
``ube.papertrading.vbt`` does for vectorbt.
"""

from __future__ import annotations

from ube.papertrading.core import (  # noqa: F401
    register_paper_engine,
    register_state_class,
)

from .backend import BacktraderPaperEngine  # noqa: F401
from .state import BacktraderPaperState  # noqa: F401

register_paper_engine("backtrader", BacktraderPaperEngine)
register_state_class("backtrader", BacktraderPaperState)
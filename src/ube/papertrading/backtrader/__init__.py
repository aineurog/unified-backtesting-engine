"""Backtrader paper-trading backend.

Importing this package self-registers the ``"backtrader"`` engine and its
:class:`BacktraderPaperState` state class, exactly as
``ube.papertrading.vbt`` does for vectorbt.
"""

from __future__ import annotations

from ube.papertrading.core import register_paper_engine, register_state_class  # type: ignore[attr-defined]  noqa: F401

from .backend import BacktraderPaperEngine  # type: ignore[attr-defined]  noqa: F401
from .state import BacktraderPaperState  # type: ignore[attr-defined]  noqa: F401

register_paper_engine("backtrader", BacktraderPaperEngine)  # type: ignore[attr-defined]
register_state_class("backtrader", BacktraderPaperState)  # type: ignore[attr-defined]
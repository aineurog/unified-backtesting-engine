"""Execution-timing contract — when a decision-time signal becomes a fill (§6.1).

Signals are **decision-time**: a ``True`` flag on bar ``t`` means the decision was made at
the close of bar ``t`` using information available at that close. When that decision is
executed is a separate, explicit configuration choice
(:attr:`~ube.core.config.BacktestConfig.execution_timing`):

``same_close``
    Execute on the decision bar at its close. This is vectorbt's and Nautilus's native
    behaviour and is the historical implicit behaviour of both adapters.

``next_open`` (default)
    Execute on the following bar at its open. This is backtrader's native behaviour, the
    realistic "decide after the close, act at the next open" convention, and the only
    default with no look-ahead.

``next_close``
    Execute on the following bar at its close.

Everything here is a pure translation over the canonical containers: the schedule maps a
**decision index** to an **execution index**, and :func:`shift_signals` turns a
decision-time :class:`~ube.core.signals.Signals` into the row-aligned frame an engine
must consume to execute on the requested bar. Adapters translate; they never re-invent the
schedule.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.signals import Signals

__all__ = [
    "EXECUTION_TIMINGS",
    "ExecutionTiming",
    "decision_index",
    "execution_index",
    "execution_price",
    "is_valid_execution_timing",
    "shift_signals",
    "validate_execution_timing",
]

#: The accepted ``execution_timing`` values (§6.1).
ExecutionTiming = Literal["next_open", "next_close", "same_close"]

#: Canonical order; the first entry is the default (:attr:`BacktestConfig.execution_timing`).
EXECUTION_TIMINGS: tuple[str, ...] = ("next_open", "next_close", "same_close")


def is_valid_execution_timing(value: object) -> bool:
    """True when ``value`` is one of :data:`EXECUTION_TIMINGS`."""
    return isinstance(value, str) and value in EXECUTION_TIMINGS


def validate_execution_timing(value: object) -> None:
    """Raise :class:`ConfigError` unless ``value`` is an accepted timing."""
    if not is_valid_execution_timing(value):
        raise ConfigError(
            f"execution_timing={value!r} is not one of {EXECUTION_TIMINGS}"
        )


def _lag(timing: ExecutionTiming) -> int:
    """Number of bars between the decision bar and the execution bar."""
    validate_execution_timing(timing)
    return 0 if timing == "same_close" else 1


def execution_index(
    decision_idx: int, timing: ExecutionTiming, n_bars: int
) -> int | None:
    """Execution bar for a decision on ``decision_idx``, or ``None`` if none exists.

    A decision on the final bar has no execution bar under ``next_open``/``next_close``:
    the signal is never filled. That is a defined outcome, not an error.
    """
    if decision_idx < 0 or decision_idx >= n_bars:
        raise IndexError(f"decision index {decision_idx} is outside [0, {n_bars})")
    idx = decision_idx + _lag(timing)
    return idx if idx < n_bars else None


def decision_index(execution_idx: int, timing: ExecutionTiming) -> int | None:
    """Decision bar that an execution on ``execution_idx`` came from.

    ``None`` when the execution bar precedes every possible decision (the first bar under
    ``next_open``/``next_close``), i.e. there is no decision behind it.
    """
    idx = execution_idx - _lag(timing)
    return idx if idx >= 0 else None


def execution_price(
    data: MarketData, decision_idx: int, timing: ExecutionTiming
) -> float | None:
    """Reference (unslipped) fill price for a decision, or ``None`` if no bar exists."""
    idx = execution_index(decision_idx, timing, data.n_bars)
    if idx is None:
        return None
    if timing == "next_open":
        return float(data.open[idx])
    return float(data.close[idx])


def shift_signals(signals: Signals, timing: ExecutionTiming) -> Signals:
    """Row-align decision-time signals with their execution bars.

    For ``same_close`` this is the input unchanged. For ``next_*`` each column is moved
    forward one bar, so row ``i`` carries the decision made on row ``i - 1``; row 0 (and
    the final decision, which shifts past the end) carries no signal. The returned
    container is a normal :class:`Signals` — the engine consumes it exactly like any
    other, while the ledger still reports the decision on its original bar.
    """
    lag = _lag(timing)
    if lag == 0:
        return signals
    n = signals.n_bars

    def _shift(column: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        if n > 1:
            out[1:] = column[:-1]
        return out

    return Signals(
        long_entry=_shift(signals.long_entry),
        long_exit=_shift(signals.long_exit),
        short_entry=_shift(signals.short_entry),
        short_exit=_shift(signals.short_exit),
    )

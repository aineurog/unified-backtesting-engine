"""In-process signal registry bridging ube ``Signals`` and the nautilus strategy.

The ``UbeDataClient`` feeds bars (and their per-bar 4-column signal) one at a time; the
``UbePaperStrategy`` reads the signal for a bar by its ``ts_event``. Both run in the same
process, so a module-level singleton is sufficient (ports ``sim_nautilus.signals``).
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Per-bar signal tuple: (long_entry, long_exit, short_entry, short_exit).
SignalTuple = Tuple[bool, bool, bool, bool]


class SignalRegistry:
    """Map ``ts_ns -> (le, lx, se, sx)`` for every bar published by the data client."""

    def __init__(self) -> None:
        self._signals: Dict[int, SignalTuple] = {}
        self.bars_published = 0
        self.bars_processed = 0

    def register(self, ts_ns: int, le: bool, lx: bool, se: bool, sx: bool) -> None:
        self._signals[int(ts_ns)] = (bool(le), bool(lx), bool(se), bool(sx))
        self.bars_published += 1

    def at(self, ts_ns: int) -> SignalTuple:
        return self._signals.get(int(ts_ns), (False, False, False, False))

    def clear(self) -> None:
        self._signals.clear()
        self.bars_published = 0
        self.bars_processed = 0

    def __len__(self) -> int:
        return len(self._signals)


SIGNAL_REGISTRY = SignalRegistry()

__all__ = ["SignalRegistry", "SignalTuple", "SIGNAL_REGISTRY"]

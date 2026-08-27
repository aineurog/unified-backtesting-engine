"""In-process signal registry bridging ube ``Signals`` and the nautilus strategy.

The ``UbeDataClient`` feeds bars (and their per-bar 4-column signal) one at a time; the
``UbePaperStrategy`` reads the signal for a bar by its ``ts_event``. Both run in the same
process, so a module-level singleton is sufficient (ports ``sim_nautilus.signals``).

Each entry also stores the *historical* (test-clock) timestamp of the bar. The bars are
published to the sandbox with **live-clock** timestamps (so the sandbox's own execution
clock advances past the live order timestamps and matches each order to its own bar — the
order factory stamps orders with a read-only live clock we cannot override), but the ube
ledger/trades must stay on the historical timeline for §9.4 comparability, so the strategy
looks the historical ts up here and tags every ``LedgerEvent`` with it.
"""

from __future__ import annotations

#: Per-bar value: ``(long_entry, long_exit, short_entry, short_exit, historical_ts_ns)``.
SignalTuple = tuple[bool, bool, bool, bool, int]


class SignalRegistry:
    """Map ``ts_ns -> (le, lx, se, sx, historical_ts)`` for every bar published."""

    def __init__(self) -> None:
        self._signals: dict[int, SignalTuple] = {}
        self.bars_published = 0
        self.bars_processed = 0

    def register(
        self, ts_ns: int, le: bool, lx: bool, se: bool, sx: bool, historical_ts: int = 0
    ) -> None:
        self._signals[int(ts_ns)] = (
            bool(le),
            bool(lx),
            bool(se),
            bool(sx),
            int(historical_ts),
        )
        self.bars_published += 1

    def at(self, ts_ns: int) -> SignalTuple:
        return self._signals.get(int(ts_ns), (False, False, False, False, 0))

    def clear(self) -> None:
        self._signals.clear()
        self.bars_published = 0
        self.bars_processed = 0

    def __len__(self) -> int:
        return len(self._signals)


SIGNAL_REGISTRY = SignalRegistry()

__all__ = ["SignalRegistry", "SignalTuple", "SIGNAL_REGISTRY"]

"""Pending decision helpers shared by the window-replay paper backends."""

from __future__ import annotations

from typing import Any

from ube.core.execution import ExecutionTiming
from ube.core.ledger import EventType, LedgerEvent
from ube.core.signals import Signals

_SIGNAL_NAMES = ("long_entry", "long_exit", "short_entry", "short_exit")


def pending_decision(
    signals: Signals, *, timestamp: int, timing: ExecutionTiming
) -> dict[str, Any] | None:
    """Capture an actionable decision on a window's final bar for the next poll."""
    if timing == "same_close":
        return None
    flags = {
        name: bool(getattr(signals, name)[-1])
        for name in _SIGNAL_NAMES
    }
    if not any(flags.values()):
        return None
    return {"kind": "signal", "ts": int(timestamp), "signals": flags}


def pending_signal_events(
    pending: dict[str, Any], instrument_id: str, *, include_side: bool = False
) -> list[LedgerEvent]:
    """Create decision-time ledger events when a final-bar decision is first queued."""
    flags = pending.get("signals")
    if not isinstance(flags, dict):
        return []
    timestamp = int(pending["ts"])
    side_by_name = {
        "long_entry": 1,
        "long_exit": -1,
        "short_entry": -1,
        "short_exit": 1,
    }
    return [
        LedgerEvent(
            EventType.SIGNAL_EVALUATED,
            timestamp,
            instrument_id,
            action=name,
            side=side_by_name[name] if include_side else None,
        )
        for name in _SIGNAL_NAMES
        if bool(flags.get(name, False))
    ]

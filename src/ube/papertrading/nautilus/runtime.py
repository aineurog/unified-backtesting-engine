"""Shared runtime singletons for the one-shot nautilus paper node.

The data client must not drain bars until the strategy has subscribed to the bar type.
Nautilus 1.221 removed ``subscribed_bars`` from the data client, so we coordinate with a
module-level ``asyncio.Event`` (one-shot per ``step`` run). It must live outside the
msgspec configs (which JSON-encode their fields and cannot hold an ``Event``).
"""

from __future__ import annotations

import asyncio

READY_EVENT: asyncio.Event | None = None


def get_ready_event() -> asyncio.Event:
    global READY_EVENT
    if READY_EVENT is None:
        READY_EVENT = asyncio.Event()
    return READY_EVENT


def reset_ready_event() -> asyncio.Event:
    global READY_EVENT
    READY_EVENT = asyncio.Event()
    return READY_EVENT


__all__ = ["READY_EVENT", "get_ready_event", "reset_ready_event"]

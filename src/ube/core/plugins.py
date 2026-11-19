"""Generic entry-point discovery for optional plugins.

This module knows nothing about engines or paper backends: it resolves entry points
in a named group and reports per-plugin failures so one broken plugin cannot take the
core down. Engine-specific registration lives in the adapters and paper-trading
packages.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

__all__ = ["ENGINE_GROUP", "PAPER_ENGINE_GROUP", "load_entry_point_objects"]

#: Entry-point group advertising backtest engine adapters.
ENGINE_GROUP = "ube.engines"

#: Entry-point group advertising paper-trading backends.
PAPER_ENGINE_GROUP = "ube.paper_engines"


def load_entry_point_objects(group: str) -> tuple[list[tuple[str, Any]], dict[str, str]]:
    """Load every entry point in ``group``.

    Returns:
        ``(loaded, failures)`` where ``loaded`` is a list of ``(entry_point_name,
        object)`` pairs and ``failures`` maps an entry-point name to a
        ``"<ExceptionType>: <message>"`` string. A failure in one plugin never
        prevents the others from loading.
    """
    loaded: list[tuple[str, Any]] = []
    failures: dict[str, str] = {}
    try:
        entry_points = metadata.entry_points(group=group)
    except Exception as exc:  # noqa: BLE001 - discovery must never break the core
        return loaded, {"*": f"{type(exc).__name__}: {exc}"}
    for entry_point in entry_points:
        name = str(entry_point.name)
        try:
            obj = entry_point.load()
        except Exception as exc:  # noqa: BLE001 - isolate a broken plugin
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        loaded.append((name, obj))
    return loaded, failures

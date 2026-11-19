"""Entry-point discovery for backtest engine adapters.

Plugins advertise a concrete :class:`~ube.adapters.base.EngineAdapter` subclass in the
``ube.engines`` entry-point group. Registration is lazy: it runs only when an engine
name is requested that the built-in registry does not know.
"""

from __future__ import annotations

from ube.core.errors import ConfigError
from ube.core.plugins import ENGINE_GROUP, load_entry_point_objects

from .base import register_engine

__all__ = ["register_entry_point_engines"]


def register_entry_point_engines() -> dict[str, str]:
    """Register adapters advertised in the ``ube.engines`` group.

    Returns a map of entry-point name to failure description for plugins that could
    not be loaded or registered. Registration failures are isolated per plugin.
    """
    loaded, failures = load_entry_point_objects(ENGINE_GROUP)
    for name, adapter in loaded:
        try:
            register_engine(name, adapter)
        except ConfigError as exc:
            failures[name] = str(exc)
    return failures

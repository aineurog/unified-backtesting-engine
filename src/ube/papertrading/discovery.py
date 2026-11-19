"""Entry-point discovery for paper-trading backends.

Plugins advertise a concrete :class:`~ube.papertrading.core.PaperEngine` subclass in
the ``ube.paper_engines`` entry-point group. A backend module may also register its
engine-specific state class at import time (the built-in backends do), which is why
``get_state_class`` runs the same discovery before falling back to ``PaperState``.
"""

from __future__ import annotations

from ube.core.errors import ConfigError
from ube.core.plugins import PAPER_ENGINE_GROUP, load_entry_point_objects

from .core import register_paper_engine

__all__ = ["register_entry_point_paper_engines"]


def register_entry_point_paper_engines() -> dict[str, str]:
    """Register backends advertised in the ``ube.paper_engines`` group.

    Returns a map of entry-point name to failure description for plugins that could
    not be loaded or registered. Registration failures are isolated per plugin.
    """
    loaded, failures = load_entry_point_objects(PAPER_ENGINE_GROUP)
    for name, engine in loaded:
        try:
            register_paper_engine(name, engine)
        except ConfigError as exc:
            failures[name] = str(exc)
    return failures

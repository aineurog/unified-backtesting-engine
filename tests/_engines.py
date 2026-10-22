"""Which optional engines the test environment actually has (§16).

One definition, imported by both :mod:`tests.conftest` (which decides what to collect)
and the tests that assert something *about* an engine's presence — so the two can never
disagree about what "installed" means.
"""

from __future__ import annotations

import importlib.util

__all__ = ["ENGINE_MODULES", "engine_is_installed"]

#: Engine name -> the module that must be importable for that engine to be present. The
#: distribution name and the import name differ for nautilus (``nautilus-trader``).
ENGINE_MODULES: dict[str, str] = {
    "vectorbt": "vectorbt",
    "backtrader": "backtrader",
    "nautilus": "nautilus_trader",
}


def engine_is_installed(engine: str) -> bool:
    """True when ``engine``'s module can be imported in this environment."""
    try:
        return importlib.util.find_spec(ENGINE_MODULES[engine]) is not None
    except ImportError:
        # ``find_spec`` returns ``None`` when nothing on the path claims the name, but
        # raises when a *parent* is missing, and an installed finder may raise too. Any
        # failure to locate the module means the engine is not usable here.
        return False

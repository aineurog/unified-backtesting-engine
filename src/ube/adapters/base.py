"""Engine adapter interface, registry, and base contract (§4.1, §4.2, §7.1).

Engines are not hardcoded anywhere — they register against this interface via
``ube.register_engine("my_engine", MyEngineAdapter)`` (§4.1). That is the same mechanism
for any future engine, including private adapters living in a separate repo that depend on
this library, not the other way around.

Dependency direction is one-way: adapters import from ``core``; ``core`` never imports from
adapters. Anything two engines share lives in ``core`` once (dev-guide §4.1).

The base contract is deliberately minimal: a single :meth:`EngineAdapter.run` taking the
canonical inputs (:class:`MarketData`, :class:`Signals`, :class:`BacktestConfig`) and
returning the canonical :class:`BacktestResult`. Cross-cutting modules (stats, reporting)
read only this contract — never engine-specific fields (§4.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from numpy.typing import ArrayLike

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.result import BacktestResult
from ube.core.signals import Signals

__all__ = [
    "AUTO_ENGINE_ORDER",
    "EngineAdapter",
    "get_engine",
    "register_engine",
    "registered_engines",
    "resolve_engine_name",
]

#: Engine names tried by :func:`get_engine` for ``"auto"``, in order of preference (§7.1).
AUTO_ENGINE_ORDER: tuple[str, ...] = ("vectorbt", "backtrader", "nautilus")


class EngineAdapter(ABC):
    """Abstract base class for backtesting engine adapters (§4.1).

    Each adapter is a *pure translator*: it consumes the canonical inputs and produces the
    canonical :class:`~ube.core.result.BacktestResult`, converting engine-specific types
    only internally — nothing engine-specific crosses this interface (dev-guide §4.1, §4.2).
    """

    @abstractmethod
    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data: Mapping[str, ArrayLike] | None = None,
    ) -> BacktestResult:
        """Run one backtest and return the canonical result (§4.1, §7.1).

        Args:
            data: The single-instrument OHLCV bars for the traded instrument.
            signals: The 4-column entry/exit signals on the same bar grid as ``data``.
            config: The full :class:`~ube.core.config.BacktestConfig` (instrument, cost,
                risk/exits, benchmark, engine overrides).
            aux_data: Named derived series (§5.2) — e.g. a precomputed ATR — keyed by
                name; unused by adapters that only need ``data``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Engine registry.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[EngineAdapter]] = {}


def _normalize(name: str) -> str:
    """Lowercase-and-strip an engine name; reject empty/non-string names."""
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"engine name must be a non-empty string; got {name!r}")
    return name.strip().lower()


def register_engine(name: str, adapter: type[EngineAdapter]) -> None:
    """Register an adapter ``class`` under ``name`` (§4.1).

    Registration is idempotent per name: re-registering the *same* class is a no-op, while
    registering a *different* class under an existing name is rejected with a
    :class:`~ube.core.errors.ConfigError` to prevent silent overwrite.
    """
    key = _normalize(name)
    if not isinstance(adapter, type) or not issubclass(adapter, EngineAdapter):
        raise ConfigError(
            f"register_engine({name!r}, ...) expects an EngineAdapter subclass; "
            f"got {adapter!r}"
        )
    if adapter is EngineAdapter or bool(adapter.__abstractmethods__):
        raise ConfigError(
            f"registered engine {name!r} ({adapter.__name__}) must be a concrete class"
        )
    current = _REGISTRY.get(key)
    if current is not None and current is not adapter:
        raise ConfigError(
            f"engine {name!r} is already registered to {current.__name__}; "
            f"refusing to overwrite with {adapter.__name__}"
        )
    _REGISTRY[key] = adapter


def registered_engines() -> tuple[str, ...]:
    """Return the sorted names of all currently registered engines."""
    return tuple(sorted(_REGISTRY))


def resolve_engine_name(name: str = "auto") -> str:
    """Resolve ``name`` to a canonical (normalized) engine name (§7.1).

    ``"auto"`` resolves to the first *installed* (registered) adapter in
    ``vectorbt → backtrader → nautilus`` order, and raises
    :class:`~ube.core.errors.ConfigError` when none is registered. Any other name is
    normalized (lowercased/stripped) and returned as-is — whether that name is actually
    *registered* is :func:`get_engine`'s concern, not this function's.
    """
    key = _normalize(name)
    if key == "auto":
        for candidate in AUTO_ENGINE_ORDER:
            if candidate in _REGISTRY:
                return candidate
        available = registered_engines()
        raise ConfigError(
            "no engine adapter is installed (auto requested); "
            "install one via ube.register_engine(...)"
            f" — registered: {', '.join(available) if available else 'none'}"
        )
    return key


def get_engine(name: str = "auto") -> type[EngineAdapter]:
    """Resolve the adapter class for ``name`` (§4.1, §7.1).

    ``"auto"`` returns the first *installed* (registered) adapter in
    ``vectorbt → backtrader → nautilus`` order. An unknown name — or ``"auto"`` with no
    adapter registered — raises :class:`~ube.core.errors.ConfigError`.
    """
    key = resolve_engine_name(name)
    adapter = _REGISTRY.get(key)
    if adapter is None:
        available = registered_engines()
        raise ConfigError(
            f"unknown engine {name!r} — registered engines: "
            f"{', '.join(available) if available else 'none'}"
        )
    return adapter

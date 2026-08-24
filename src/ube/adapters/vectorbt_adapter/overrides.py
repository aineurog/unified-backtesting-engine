"""vectorbt engine overrides — the typed ``engine_overrides`` namespace (§4.3, §7.2).

vectorbt-specific knobs live here, never in :class:`~ube.core.config.BacktestConfig` (§4.2):
the base config only carries them as an opaque ``Mapping``, and :func:`validate_overrides`
enforces the per-adapter schema at ``VectorbtAdapter.run`` time. Unknown keys and wrong types
raise :class:`~ube.core.errors.ConfigError` naming the offending field (requirements §7.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from ube.core.errors import ConfigError

__all__ = [
    "DEFAULT_STARTING_BALANCE",
    "VectorbtEngineOverrides",
    "validate_overrides",
]

#: Account starting balance in ``BacktestConfig.base_currency`` when unset (requirements §4.2).
DEFAULT_STARTING_BALANCE = 100_000.0

#: Default per-period funding cadence for crypto perps (hours, §24).
DEFAULT_FUNDING_INTERVAL_HOURS = 8.0


class VectorbtEngineOverrides(TypedDict, total=False):
    """Engine-specific overrides for the vectorbt adapter (dev-guide §4.3, requirements §4.2).

    Every key is optional; vectorbt defaults are applied for anything not provided. Values are
    validated by :func:`validate_overrides` at ``VectorbtAdapter.run`` time.
    """

    starting_balance: float
    funding_interval_hours: float


def _require_positive_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"engine override {field!r} must be a number")
    if value <= 0:
        raise ConfigError(f"engine override {field!r} must be > 0")


_FIELD_VALIDATORS: dict[str, Callable[[Any, str], None]] = {
    "starting_balance": _require_positive_number,
    "funding_interval_hours": _require_positive_number,
}


def validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and return a fresh copy of ``config.engine_overrides`` (§7.2).

    ``None`` (undeclared) is valid and yields an empty dict. Unknown keys and wrong types raise
    :class:`~ube.core.errors.ConfigError` naming the offending field, per requirements §7.2.

    Args:
        overrides: The ``BacktestConfig.engine_overrides`` mapping, or ``None``.

    Returns:
        A fresh dict of validated override values; safe for the caller to mutate.
    """
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ConfigError(
            "engine_overrides must be a Mapping or None; "
            f"got {type(overrides).__name__}"
        )
    validated: dict[str, Any] = {}
    for field, value in overrides.items():
        validator = _FIELD_VALIDATORS.get(field)
        if validator is None:
            valid = ", ".join(_FIELD_VALIDATORS)
            raise ConfigError(
                f"unknown engine override {field!r} for the vectorbt adapter "
                f"(valid keys: {valid})"
            )
        validator(value, field)
        validated[field] = value
    return validated

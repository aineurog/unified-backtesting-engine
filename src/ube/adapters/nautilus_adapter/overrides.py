"""NautilusTrader engine overrides — the typed ``engine_overrides`` namespace (§4.3, §7.2).

Nautilus-specific knobs live here, never in :class:`~ube.core.config.BacktestConfig` (§4.2):
the base config only carries them as an opaque ``Mapping``, and
:func:`validate_overrides` enforces the per-adapter schema at ``NautilusAdapter.run`` time.
Unknown keys and wrong types raise :class:`~ube.core.errors.ConfigError` naming the
offending field (requirements §7.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

from ube.core.errors import ConfigError

__all__ = [
    "DEFAULT_OMS_TYPE",
    "DEFAULT_STARTING_BALANCE",
    "DEFAULT_VENUE",
    "NautilusEngineOverrides",
    "validate_overrides",
]

#: Synthetic venue id used by the backtest environment (reference ``constants.py``).
DEFAULT_VENUE = "SIM"
#: Account starting balance in ``BacktestConfig.base_currency`` when unset (requirements §4.2).
DEFAULT_STARTING_BALANCE = 100_000.0
#: Default OMS type for margin accounts.
DEFAULT_OMS_TYPE = "NETTING"

_ACCOUNT_TYPES: tuple[str, ...] = ("margin", "cash")
_OMS_TYPES: tuple[str, ...] = ("NETTING", "HEDGING")


class NautilusEngineOverrides(TypedDict, total=False):
    """Engine-specific overrides for the NautilusTrader adapter (dev-guide §4.3, requirements §4.2).

    Every key is optional; Nautilus defaults are applied for anything not provided.
    Values are validated by :func:`validate_overrides` at ``NautilusAdapter.run`` time.
    """

    venue: str
    account_type: Literal["margin", "cash"]
    leverage: float
    starting_balance: float
    price_precision: int
    size_precision: int
    oms_type: Literal["NETTING", "HEDGING"]
    maker_fee: float
    taker_fee: float


# ---------------------------------------------------------------------------
# Field validators.
# ---------------------------------------------------------------------------


def _require_str(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"engine override {field!r} must be a non-empty string")


def _validate_account_type(value: Any, field: str) -> None:
    _require_str(value, field)
    if value not in _ACCOUNT_TYPES:
        raise ConfigError(f"engine override {field!r} must be one of {_ACCOUNT_TYPES}")


def _validate_oms_type(value: Any, field: str) -> None:
    _require_str(value, field)
    if value not in _OMS_TYPES:
        raise ConfigError(f"engine override {field!r} must be one of {_OMS_TYPES}")


def _require_positive_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"engine override {field!r} must be a number")
    if value <= 0:
        raise ConfigError(f"engine override {field!r} must be > 0")


def _require_fraction(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"engine override {field!r} must be a number (fee fraction)")
    if value < 0:
        raise ConfigError(f"engine override {field!r} must be >= 0")


def _require_nonneg_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"engine override {field!r} must be an integer")
    if value < 0:
        raise ConfigError(f"engine override {field!r} must be >= 0")


_FIELD_VALIDATORS: dict[str, Callable[[Any, str], None]] = {
    "venue": _require_str,
    "account_type": _validate_account_type,
    "leverage": _require_positive_number,
    "starting_balance": _require_positive_number,
    "price_precision": _require_nonneg_int,
    "size_precision": _require_nonneg_int,
    "oms_type": _validate_oms_type,
    "maker_fee": _require_fraction,
    "taker_fee": _require_fraction,
}


# ---------------------------------------------------------------------------
# Validation entry point.
# ---------------------------------------------------------------------------


def validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and return a fresh copy of ``config.engine_overrides`` (§7.2).

    ``None`` (undeclared) is valid and yields an empty dict. Unknown keys and wrong
    types raise :class:`~ube.core.errors.ConfigError` naming the offending field, per
    requirements §7.2 ("unknown keys or wrong types raise ``ConfigError``").

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
                f"unknown engine override {field!r} for the nautilus adapter "
                f"(valid keys: {valid})"
            )
        validator(value, field)
        validated[field] = value
    return validated

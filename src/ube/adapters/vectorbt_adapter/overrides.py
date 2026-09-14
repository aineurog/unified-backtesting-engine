"""vectorbt engine overrides — the typed ``engine_overrides`` namespace (§4.3, §7.2).

vectorbt-specific knobs live here, never in :class:`~ube.core.config.BacktestConfig` (§4.2):
the base config only carries them as an opaque ``Mapping``, and :func:`validate_overrides`
enforces the per-adapter schema at ``VectorbtAdapter.run`` time. Unknown keys and wrong types
raise :class:`~ube.core.errors.ConfigError` naming the offending field (requirements §7.2).

Overrides that vectorbt actually needs are limited: the account starting balance, the funding
cadence (which otherwise comes from the canonical instrument — §4.5, so the override is a
precedence fallback, see :mod:`ube.adapters.vectorbt_adapter.instrument_map`), the asset-class
lot grid (``size_precision`` / ``size_increment``), the account type (``cash`` vs ``margin`` —
a cash account ignores the sizing ``leverage``, mirroring the nautilus adapter), and the FX
synthetic-rates family (``synthetic_rates`` / ``fixed_conversions`` / ``currency_fx`` /
``fx_rates``) used to normalize multi-currency results into ``base_currency`` (§4.6).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

from ube.core.errors import ConfigError

__all__ = [
    "DEFAULT_STARTING_BALANCE",
    "DEFAULT_FUNDING_INTERVAL_HOURS",
    "VectorbtEngineOverrides",
    "validate_overrides",
    "parse_synthetic_rates",
]

#: Account starting balance in ``BacktestConfig.base_currency`` when unset (requirements §4.2).
DEFAULT_STARTING_BALANCE = 100_000.0

#: Default per-period funding cadence for crypto perps (hours, §24). Kept for compatibility;
#: the authoritative schedule is the canonical instrument's ``funding_interval_hours`` (§4.5).
DEFAULT_FUNDING_INTERVAL_HOURS = 8.0

_ACCOUNT_TYPES: tuple[str, ...] = ("margin", "cash")

#: The synthetic-rates alias family (mirrors the nautilus override namespace).
FX_ALIASES: tuple[str, ...] = (
    "synthetic_rates",
    "fixed_conversions",
    "currency_fx",
    "fx_rates",
)


class VectorbtEngineOverrides(TypedDict, total=False):
    """Engine-specific overrides for the vectorbt adapter (dev-guide §4.3, requirements §4.2).

    Every key is optional; vectorbt defaults are applied for anything not provided. Values are
    validated by :func:`validate_overrides` at ``VectorbtAdapter.run`` time.

    - ``starting_balance`` — account cash in ``base_currency``.
    - ``funding_interval_hours`` — funding cadence; overrides the canonical instrument's
      ``funding_interval_hours`` when provided (default: the instrument metadata, §4.5).
    - ``size_precision`` / ``size_increment`` — the asset-class lot grid (decimal places and
      the smallest tradable unit); defaults are asset-class tables in ``instrument_map``.
    - ``account_type`` — ``"margin"`` (the sizing ``leverage`` is honored) or ``"cash"``
      (leverage is forced to 1.0, mirroring the nautilus cash account).
    - ``fx_rates`` / ``synthetic_rates`` / ``fixed_conversions`` / ``currency_fx`` — fixed
      synthetic exchange rates for multi-currency normalization (§4.6): a ``{pair: rate}``
      mapping or a list of ``{"pair": ..., "rate": ...}`` mappings.
    """

    starting_balance: float
    funding_interval_hours: float
    size_precision: int
    size_increment: float
    account_type: Literal["margin", "cash"]
    leverage: float
    synthetic_rates: dict[str, float] | list[dict[str, Any]]
    fixed_conversions: dict[str, float] | list[dict[str, Any]]
    currency_fx: dict[str, float] | list[dict[str, Any]]
    fx_rates: dict[str, float] | list[dict[str, Any]]


def _require_positive_number(value: Any, field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"engine override {field!r} must be a number")
    if value <= 0:
        raise ConfigError(f"engine override {field!r} must be > 0")
    return value


def _require_nonneg_int(value: Any, field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"engine override {field!r} must be an integer")
    if value < 0:
        raise ConfigError(f"engine override {field!r} must be >= 0")
    return value


def _validate_account_type(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"engine override {field!r} must be a non-empty string")
    if value not in _ACCOUNT_TYPES:
        raise ConfigError(f"engine override {field!r} must be one of {_ACCOUNT_TYPES}")
    return value


def _split_pair(pair_str: Any, field: str) -> tuple[str, str]:
    """Split a pair label (``"USDUSDT"`` / ``"USD/USDT"`` / ``"USD_USDT"``) into two codes."""
    text = str(pair_str or "").strip().upper()
    parts = [p for p in re.split(r"[\s/_\-:]+", text) if p]
    if len(parts) == 2:
        c1, c2 = parts
    elif len(parts) == 1 and text.isalpha():
        if len(text) == 6:
            c1, c2 = text[:3], text[3:]
        elif len(text) == 7 and text.startswith("USDT"):
            c1, c2 = text[:4], text[4:]
        elif len(text) == 7 and text.endswith("USDT"):
            c1, c2 = text[:-4], text[-4:]
        else:
            raise ConfigError(
                f"engine override {field!r} synthetic rate pair {pair_str!r} must be "
                "two currency codes (e.g. 'USDUSDT' or 'USD/USDT')"
            )
    else:
        raise ConfigError(
            f"engine override {field!r} synthetic rate pair {pair_str!r} must be "
            "two currency codes (e.g. 'USDUSDT' or 'USD/USDT')"
        )
    if not (c1.isalpha() and c2.isalpha() and 3 <= len(c1) <= 4 and 3 <= len(c2) <= 4):
        raise ConfigError(
            f"engine override {field!r} synthetic rate pair {pair_str!r} must be "
            "two 3-4 letter currency codes"
        )
    return c1, c2


def parse_synthetic_rates(value: Any, field: str = "fx_rates") -> dict[str, float]:
    """Parse the synthetic-rates family into a canonical ``{pair: rate}`` dict.

    Accepts a ``{pair: rate}`` mapping or a list of ``{"pair": ..., "rate": ...}`` mappings.
    Each 6-letter (or 7-letter ``USDT...``) pair also registers its inverse at ``1/rate``
    unless the inverse is explicitly declared. Malformed entries raise ``ConfigError`` —
    rates are never guessed (§4.7). This is the vectorbt-side mirror of the nautilus
    ``_synthetic_fx_rates`` builder; the canonical dict is what the adapter turns into
    :class:`~ube.core.ledger.FXSeries` on the bar grid (§4.6).
    """
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = []
        for entry in value:
            if not isinstance(entry, Mapping):
                raise ConfigError(
                    f"engine override {field!r} synthetic rates list entries must be "
                    f"mappings, got {type(entry).__name__}"
                )
            pair = entry.get("pair") or entry.get("symbol")
            rate = entry.get("rate", entry.get("value"))
            items.append((pair, rate))
    else:
        raise ConfigError(
            f"engine override {field!r} synthetic rates must be a {{pair: rate}} mapping "
            f"or a list of {{pair, rate}} mappings, got {type(value).__name__}"
        )
    quoted: dict[str, tuple[str, str, float]] = {}
    for pair_str, rate in items:
        bad_rate = (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        )
        if bad_rate:
            raise ConfigError(
                f"engine override {field!r} synthetic rate for {pair_str!r} must be a "
                f"positive finite number, got {rate!r}"
            )
        c1, c2 = _split_pair(pair_str, field)
        quoted[c1 + c2] = (c1, c2, float(rate))
    pairs: dict[str, float] = {}
    for key, (_c1, _c2, _r) in list(quoted.items()):
        pairs[key] = _r
        if _c2 + _c1 not in quoted:
            pairs[_c2 + _c1] = 1.0 / _r
    return pairs


def _validate_fx_rates(value: Any, field: str) -> Any:
    """Validator that replaces the raw override with its canonical parsed form."""
    return parse_synthetic_rates(value, field)


_FIELD_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "starting_balance": _require_positive_number,
    "funding_interval_hours": _require_positive_number,
    "size_precision": _require_nonneg_int,
    "size_increment": _require_positive_number,
    "account_type": _validate_account_type,
    "leverage": _require_positive_number,
    "synthetic_rates": _validate_fx_rates,
    "fixed_conversions": _validate_fx_rates,
    "currency_fx": _validate_fx_rates,
    "fx_rates": _validate_fx_rates,
}


def validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and return a fresh copy of ``config.engine_overrides`` (§7.2).

    ``None`` (undeclared) is valid and yields an empty dict. Unknown keys and wrong types raise
    :class:`~ube.core.errors.ConfigError` naming the offending field, per requirements §7.2.
    A synthetic-rates override is parsed into its canonical ``{pair: rate}`` form (with
    implicit inverses) as part of validation.

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
        validated[field] = validator(value, field)
    return validated
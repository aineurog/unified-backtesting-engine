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
    price_increment: str
    oms_type: Literal["NETTING", "HEDGING"]
    maker_fee: float
    taker_fee: float
    synthetic_rates: dict[str, float] | list[dict[str, Any]]
    fixed_conversions: dict[str, float] | list[dict[str, Any]]
    currency_fx: dict[str, float] | list[dict[str, Any]]
    fx_rates: dict[str, float] | list[dict[str, Any]]


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


def _validate_synthetic_rates(value: Any, field: str) -> None:
    if not isinstance(value, (Mapping, list, tuple)):
        raise ConfigError(
            f"engine override {field!r} must be a dict/Mapping or list/tuple of rates"
        )


_FIELD_VALIDATORS: dict[str, Callable[[Any, str], None]] = {
    "venue": _require_str,
    "account_type": _validate_account_type,
    "leverage": _require_positive_number,
    "starting_balance": _require_positive_number,
    "price_precision": _require_nonneg_int,
    "size_precision": _require_nonneg_int,
    "price_increment": _require_str,
    "oms_type": _validate_oms_type,
    "maker_fee": _require_fraction,
    "taker_fee": _require_fraction,
    "synthetic_rates": _validate_synthetic_rates,
    "fixed_conversions": _validate_synthetic_rates,
    "currency_fx": _validate_synthetic_rates,
    "fx_rates": _validate_synthetic_rates,
}


def apply_synthetic_rates(
    cache: Any,
    overrides: Mapping[str, Any] | None = None,
    settlement_currency: str | None = None,
    base_currency: str | None = None,
) -> None:
    """Pre-load fixed 1:1 or custom synthetic exchange rates into Nautilus Cache.

    Prevents 'insufficient data for USD/USDT' or missing quote map errors when
    converting portfolio account state between USD and USDT or custom pairs.
    """
    if cache is None:
        return

    try:
        from nautilus_trader.model.currencies import USD, USDT, Currency

        # Default USD <-> USDT 1:1 synthetic rate
        try:
            cache.add_currency(USD)
            cache.add_currency(USDT)
            cache.set_mark_xrate(USD, USDT, 1.0)
        except Exception:
            pass

        if settlement_currency and settlement_currency not in ("USD", "USDT"):
            try:
                curr = Currency.from_str(settlement_currency)
                cache.add_currency(curr)
            except Exception:
                pass

        if base_currency and base_currency not in ("USD", "USDT"):
            try:
                curr = Currency.from_str(base_currency)
                cache.add_currency(curr)
            except Exception:
                pass

        overrides = overrides or {}
        raw_rates = (
            overrides.get("synthetic_rates")
            or overrides.get("fixed_conversions")
            or overrides.get("currency_fx")
            or overrides.get("fx_rates")
        )

        if not raw_rates:
            return

        if isinstance(raw_rates, Mapping):
            for pair_str, rate in raw_rates.items():
                try:
                    pair = str(pair_str).replace("/", "").replace("_", "").strip()
                    if len(pair) == 6:
                        c1_str, c2_str = pair[:3], pair[3:]
                    else:
                        parts = str(pair_str).replace("_", "/").split("/")
                        if len(parts) == 2:
                            c1_str, c2_str = parts[0].strip(), parts[1].strip()
                        else:
                            continue
                    c1 = Currency.from_str(c1_str)
                    c2 = Currency.from_str(c2_str)
                    cache.add_currency(c1)
                    cache.add_currency(c2)
                    cache.set_mark_xrate(c1, c2, float(rate))
                except Exception:
                    pass
        elif isinstance(raw_rates, (list, tuple)):
            for item in raw_rates:
                if isinstance(item, Mapping):
                    pair_str = (
                        item.get("pair") or item.get("symbol") or item.get("currencies")
                    )
                    rate = item.get("rate") or item.get("value") or 1.0
                    if pair_str:
                        try:
                            pair = str(pair_str).replace("/", "").replace("_", "").strip()
                            if len(pair) == 6:
                                c1_str, c2_str = pair[:3], pair[3:]
                            else:
                                parts = str(pair_str).replace("_", "/").split("/")
                                if len(parts) == 2:
                                    c1_str, c2_str = parts[0].strip(), parts[1].strip()
                                else:
                                    continue
                            c1 = Currency.from_str(c1_str)
                            c2 = Currency.from_str(c2_str)
                            cache.add_currency(c1)
                            cache.add_currency(c2)
                            cache.set_mark_xrate(c1, c2, float(rate))
                        except Exception:
                            pass
    except Exception:
        pass


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

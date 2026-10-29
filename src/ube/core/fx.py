"""Stablecoin FX peg policy (§4.7, §4.8).

``ube`` never invents a cross-currency rate: the ledger raises
:class:`~ube.core.errors.FXRateUnavailableError` for an undeclared pair. The one
documented exception is the USD/USDT stablecoin peg — USDT is a USD-pegged
stablecoin, so a ``USD``-settled instrument on a ``USDT`` account (or the reverse)
can be converted at parity instead of failing the common crypto run.

The assumption is never silent: :func:`warn_stablecoin_peg` emits a
``UserWarning`` naming the pair and the explicit override that would replace it,
and a caller-declared rate always wins (the adapters only seed a pair that is
absent from ``engine_overrides``). Every adapter resolves the peg through this
module so the three engines apply exactly the same policy.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping
from typing import Any

from ube.core.errors import ConfigError, FXRateUnavailableError

__all__ = [
    "FX_RATE_ALIASES",
    "STABLECOIN_PEGS",
    "declared_fx_rates",
    "parse_declared_fx_rates",
    "resolve_fx_rate",
    "stablecoin_peg_rate",
    "warn_stablecoin_peg",
]

#: The engine-override aliases carrying caller-declared FX rates; first declared wins.
FX_RATE_ALIASES: tuple[str, ...] = (
    "synthetic_rates",
    "fixed_conversions",
    "currency_fx",
    "fx_rates",
)

#: Currency groups whose members trade at parity. Kept deliberately small: a pair is
#: only auto-seeded when both of its currencies appear in one group.
STABLECOIN_PEGS: tuple[frozenset[str], ...] = (frozenset({"USD", "USDT"}),)


def stablecoin_peg_rate(currency_a: str | None, currency_b: str | None) -> float | None:
    """The documented peg rate for the pair, or ``None`` when it is not a pegged pair.

    Both currency codes are compared case-insensitively. Identical or empty codes are
    not a conversion at all, so they return ``None`` (the ledger handles identity FX).
    """
    a = (currency_a or "").strip().upper()
    b = (currency_b or "").strip().upper()
    if not a or not b or a == b:
        return None
    if any(a in group and b in group for group in STABLECOIN_PEGS):
        return 1.0
    return None


def warn_stablecoin_peg(
    currency_a: str,
    currency_b: str,
    *,
    engine: str,
    stacklevel: int = 3,
) -> None:
    """Warn that an undeclared pair is being converted at the documented peg (§4.7).

    Args:
        currency_a: The settlement currency being converted from.
        currency_b: The base currency being converted to.
        engine: The adapter label, so the warning names the run that made the assumption.
        stacklevel: Forwarded to :func:`warnings.warn` so the warning points at the
            caller's ``ube.run`` / adapter invocation.
    """
    a = currency_a.strip().upper()
    b = currency_b.strip().upper()
    warnings.warn(
        f"{engine}: no FX rate declared for {a}->{b}; using the documented 1:1 "
        f"stablecoin peg. Declare the rate explicitly in engine_overrides "
        f"('fx_rates': {{'{a}{b}': <rate>}}) to replace this assumption (§4.7).",
        stacklevel=stacklevel,
    )


def _split_currency_pair(pair: object) -> tuple[str, str]:
    """Split ``"EURUSD"`` / ``"EUR/USD"`` / ``"USDTUSD"`` into two currency codes."""
    text = str(pair or "").strip().upper()
    parts = [part for part in re.split(r"[\s/_\-:]+", text) if part]
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
                f"FX rate pair {pair!r} must be two currency codes "
                "(e.g. 'EURUSD' or 'EUR/USD')"
            )
    else:
        raise ConfigError(
            f"FX rate pair {pair!r} must be two currency codes "
            "(e.g. 'EURUSD' or 'EUR/USD')"
        )
    if not (c1.isalpha() and c2.isalpha() and 3 <= len(c1) <= 4 and 3 <= len(c2) <= 4):
        raise ConfigError(
            f"FX rate pair {pair!r} must be two 3-4 letter currency codes"
        )
    return c1, c2


def parse_declared_fx_rates(value: object) -> dict[str, float]:
    """Parse the synthetic-rate override family into ``{PAIR: rate}`` (§4.7).

    Accepts a ``{pair: rate}`` mapping or a list of ``{"pair": ..., "rate": ...}``
    mappings, and seeds each pair's inverse at ``1/rate`` unless the inverse was
    declared explicitly. Rates must be positive and finite — they are never guessed.
    """
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = []
        for entry in value:
            if not isinstance(entry, Mapping):
                raise ConfigError(
                    "FX rate list entries must be mappings, got "
                    f"{type(entry).__name__}"
                )
            pair = entry.get("pair") or entry.get("symbol")
            rate = entry.get("rate", entry.get("value"))
            items.append((pair, rate))
    else:
        raise ConfigError(
            "FX rates must be a {pair: rate} mapping or a list of {pair, rate} "
            f"mappings, got {type(value).__name__}"
        )
    quoted: dict[str, tuple[str, str, float]] = {}
    for pair, rate in items:
        if (
            isinstance(rate, bool)
            or not isinstance(rate, int | float)
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise ConfigError(
                f"FX rate for {pair!r} must be a positive finite number, got {rate!r}"
            )
        c1, c2 = _split_currency_pair(pair)
        quoted[c1 + c2] = (c1, c2, float(rate))
    pairs: dict[str, float] = {}
    for key, (_c1, _c2, rate) in quoted.items():
        pairs[key] = rate
        if _c2 + _c1 not in quoted:
            pairs[_c2 + _c1] = 1.0 / rate
    return pairs


def declared_fx_rates(overrides: Mapping[str, Any] | None) -> dict[str, float]:
    """The parsed FX rates declared on ``overrides`` (first alias wins); ``{}`` if none."""
    if not overrides:
        return {}
    for alias in FX_RATE_ALIASES:
        value = overrides.get(alias)
        if value is not None:
            return parse_declared_fx_rates(value)
    return {}


def resolve_fx_rate(
    from_currency: str,
    to_currency: str,
    declared: Mapping[str, float],
) -> float:
    """The rate converting ``from_currency`` into ``to_currency`` (§4.7).

    Identity returns ``1.0`` without a lookup; a declared pair (either direction) wins;
    otherwise the documented stablecoin peg applies. An undeclared, unpegged pair raises
    :class:`~ube.core.errors.FXRateUnavailableError` — a rate is never assumed.
    """
    src = (from_currency or "").strip().upper()
    dst = (to_currency or "").strip().upper()
    if not src or not dst:
        raise ConfigError("FX conversion requires non-empty currency codes")
    if src == dst:
        return 1.0
    rate = declared.get(src + dst)
    if rate is not None:
        return float(rate)
    inverse = declared.get(dst + src)
    if inverse is not None:
        if float(inverse) <= 0:
            raise ConfigError(f"FX rate for {dst + src!r} must be positive")
        return 1.0 / float(inverse)
    peg = stablecoin_peg_rate(src, dst)
    if peg is not None:
        return peg
    raise FXRateUnavailableError(
        f"no FX rate for pair {src + dst!r} to convert {src!r} -> {dst!r}"
    )

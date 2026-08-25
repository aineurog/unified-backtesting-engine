"""Canonical :class:`~ube.core.instrument.Instrument` -> vectorbt parameter mapping.

vectorbt does not require a native instrument object the way Nautilus does (it operates on
plain pandas Series), but the engine still needs the per-asset-class *parameters* that drive
correct sizing and accounting: the size precision / lot increment, the contract multiplier, the
settlement currency, and the funding cadence. This module is the vbt analogue of
:mod:`ube.adapters.nautilus_adapter.instrument_map` (requirements §4.5) — a single, explicit
place that resolves those parameters from the canonical :class:`~ube.core.instrument.Instrument`
and the ``engine_overrides`` namespace, so every asset class (futures, commodities, crypto_perp,
crypto_spot, stocks, forex) is handled consistently rather than by ad-hoc field reads in the
adapter.

The tables below mirror Nautilus's ``instrument_map`` defaults; precision/increment fall back to
the canonical ``tick_size`` / ``contract_multiplier`` where present (§4.5). This module never
invents fees — the core :class:`~ube.core.cost.CostModel` is folded in by the adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ube.adapters.vectorbt_adapter.overrides import DEFAULT_FUNDING_INTERVAL_HOURS
from ube.core.errors import InvalidInstrumentError
from ube.core.instrument import Instrument

__all__ = [
    "VbtInstrument",
    "build_instrument",
    "round_to_increment",
    "floor_to_increment",
]

#: Default size precision (decimal places) per asset class.
_DEFAULT_SIZE_PRECISION: dict[str, int] = {
    "crypto_perp": 3,
    "crypto_spot": 3,
    "futures": 0,
    "commodities": 0,
    "stocks": 0,
    "forex": 5,
}

#: Default size (lot) increment per asset class — the smallest tradable unit.
_DEFAULT_SIZE_INCREMENT: dict[str, float] = {
    "crypto_perp": 0.001,
    "crypto_spot": 0.001,
    "futures": 1.0,
    "commodities": 1.0,
    "stocks": 1.0,
    "forex": 0.00001,
}


@dataclass(frozen=True)
class VbtInstrument:
    """The vectorbt-relevant parameters derived from a canonical instrument (§4.5)."""

    asset_class: str
    size_precision: int
    size_increment: float
    contract_multiplier: float
    settlement_currency: str
    funding_interval_hours: float


def round_to_increment(qty: float, increment: float) -> float:
    """Round a quantity to the nearest ``increment`` (half-up), e.g. 19.99 -> 20 for a 1.0 lot.

    Mirrors Nautilus's ``make_qty`` for fee-less runs (it rounds to the size precision), so the
    vbt and Nautilus entry quantities agree at the lot boundary. A non-positive increment is a
    no-op (fractional sizing is allowed).
    """
    if increment is None or increment <= 0.0:
        return float(qty)
    return float(round(qty / increment)) * float(increment)


def floor_to_increment(qty: float, increment: float) -> float:
    """Floor a quantity to the ``increment`` (never round up), e.g. 52.37 -> 52 for a 1.0 lot.

    Used for fee-aware sizers (§7.1): an up-rounded lot could exceed the size the affordability
    guard verified (by up to one lot's notional + fee). The tiny epsilon absorbs float
    representation error so ``3.0`` stored as ``2.9999999996`` floors to 3, not 2.
    """
    if increment is None or increment <= 0.0:
        return float(qty)
    inc = float(increment)
    return math.floor(float(qty) / inc + 1e-9) * inc


def build_instrument(
    canonical: Instrument,
    overrides: Mapping[str, Any] | None = None,
) -> VbtInstrument:
    """Resolve the vectorbt parameters for ``canonical`` (§4.5).

    Args:
        canonical: The canonical :class:`~ube.core.instrument.Instrument`.
        overrides: The validated ``engine_overrides`` mapping (optional). ``size_precision`` and
            ``funding_interval_hours`` may be overridden; the lot increment and contract
            multiplier default from the asset-class tables / canonical fields.

    Returns:
        The resolved :class:`VbtInstrument`.

    Raises:
        InvalidInstrumentError: ``asset_class`` is not supported by the vbt adapter.
    """
    asset_class = canonical.asset_class
    if asset_class not in _DEFAULT_SIZE_PRECISION:
        raise InvalidInstrumentError(
            f"asset_class={asset_class!r} is not supported by the vectorbt adapter; "
            f"supported: {sorted(_DEFAULT_SIZE_PRECISION)}"
        )
    overrides = overrides if overrides is not None else {}
    size_precision = int(overrides.get("size_precision", _DEFAULT_SIZE_PRECISION[asset_class]))
    size_increment = float(
        overrides.get("size_increment", _DEFAULT_SIZE_INCREMENT[asset_class])
    )
    contract_multiplier = (
        float(canonical.contract_multiplier)
        if canonical.contract_multiplier is not None
        else 1.0
    )
    settlement_currency = canonical.settlement_currency or "USD"
    funding_interval_hours = float(
        overrides.get("funding_interval_hours", DEFAULT_FUNDING_INTERVAL_HOURS)
    )
    return VbtInstrument(
        asset_class=asset_class,
        size_precision=size_precision,
        size_increment=size_increment,
        contract_multiplier=contract_multiplier,
        settlement_currency=settlement_currency,
        funding_interval_hours=funding_interval_hours,
    )

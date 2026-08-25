"""Instrument asset-class metadata (§4.5).

An ``Instrument`` carries everything that varies by asset class, kept out of the
core data/config schema itself. It is a frozen dataclass (immutable after
construction, §3 principle 5) and validates itself on construction, raising
``InvalidInstrumentError`` (a ``ConfigError``) for an unsupported specification.

The asset-class-varying fields that reference *other* subsystems are stored here as
plain string references rather than as objects, so that this module stays free of
any dependency on the calendar (item 03) and cost-model (item 05) modules:

- ``calendar`` — a trading-calendar reference (``"24/7"`` for crypto, or an exchange
  calendar name such as ``"XNYS"``/``"CMES"`` for stocks/futures/forex); resolved by
  ``ube.core.calendar``.
- ``funding_model`` / ``borrow_model`` — references to the funding/swap model (perps,
  forex) and the short-side borrow / hard-to-borrow fee model (stocks, futures);
  resolved by ``ube.core.cost``, which also derives a sensible default cost model from
  ``asset_class`` when these are left ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from ube.core.errors import InvalidInstrumentError

__all__ = [
    "Instrument",
    "ASSET_CLASSES",
    "LONG_ONLY_ASSET_CLASSES",
    "allows_short",
    "DEFAULT_FUNDING_INTERVAL_HOURS",
    "resolve_funding_interval_hours",
]

# The asset classes the engine supports. The labels are locked by the per-asset-class
# deterministic fixtures of §16 (``crypto_perp/``, ``futures/``, ``stocks/``,
# ``forex/``, ``commodities/``) and the ``asset_class="crypto_perp"`` example of §4.5.
# ``crypto_spot`` (spot crypto — no funding/borrow) is the perp's funding-free sibling.
ASSET_CLASSES: frozenset[str] = frozenset(
    {"crypto_perp", "crypto_spot", "futures", "stocks", "forex", "commodities"}
)

#: Asset classes that cannot open a short position (§4.5). Spot crypto has nothing
#: to borrow, so "short" is undefined — ``crypto_spot`` is the engine's one long-only
#: class. The signal validation of §6.1 rejects any short signal on these.
LONG_ONLY_ASSET_CLASSES: frozenset[str] = frozenset({"crypto_spot"})


#: Default funding/swap schedule for instruments that do not declare their own
#: ``funding_interval_hours`` (§4.5, §24). Crypto perpetuals settle funding every 8
#: hours, so 8.0 is the sensible cross-asset default; asset classes without funding
#: (stocks, futures, commodities, crypto_spot, forex) resolve to this too but charge
#: nothing because their cost model carries a zero funding rate.
DEFAULT_FUNDING_INTERVAL_HOURS: float = 8.0


def allows_short(asset_class: str) -> bool:
    """Whether an ``asset_class`` can open a short position (§4.5)."""
    return asset_class not in LONG_ONLY_ASSET_CLASSES


def resolve_funding_interval_hours(instrument: Instrument | None) -> float:
    """The funding/swap cadence in hours for ``instrument`` (§4.5, §24).

    Returns ``instrument.funding_interval_hours`` when declared, otherwise
    :data:`DEFAULT_FUNDING_INTERVAL_HOURS`. ``None`` (no instrument) resolves to the
    default as well — the schedule is asset-class metadata, but an absent instrument
    cannot carry anything more specific.
    """
    if instrument is None:
        return DEFAULT_FUNDING_INTERVAL_HOURS
    if instrument.funding_interval_hours is not None:
        return float(instrument.funding_interval_hours)
    return DEFAULT_FUNDING_INTERVAL_HOURS


def _is_positive_number(value: object) -> bool:
    """True if ``value`` is a finite, strictly-positive real number (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


@dataclass(frozen=True)
class Instrument:
    """Asset-class metadata for a single tradable instrument (§4.5).

    ``symbol`` and ``asset_class`` are required. Every asset-class-varying field is
    optional and defaults to ``None``; a ``None`` value means "apply the sensible
    asset-class default", which the cost/calendar modules resolve. This is not a
    silently-guessed default (§4.7): the *asset class* is declared explicitly and
    drives the resolution.

    Attributes:
        symbol: Instrument identifier, e.g. ``"BTC-USDT"``, ``"ES"``, ``"EURUSD"``.
        asset_class: One of :data:`ASSET_CLASSES`.
        tick_size: Minimum price increment (e.g. ``0.01``, ``0.25``). ``None`` if the
            asset-class default applies.
        contract_multiplier: Number of units/contract per point of price (e.g. ``50.0``
            for ES futures, ``1.0`` for spot). ``None`` if the asset-class default
            applies.
        calendar: Trading-calendar reference (``"24/7"`` for crypto; an exchange
            calendar name for stocks/futures/forex).
        settlement_currency: Currency the instrument settles in (e.g. ``"USDT"``,
            ``"USD"``).
        funding_model: Reference to the funding/swap cost model (perps, forex).
        borrow_model: Reference to the short-side borrow / hard-to-borrow fee model
            (stocks, futures).
        funding_interval_hours: Scheduled funding cadence in hours — the wall-clock
            period over which ``CostModel.funding`` (the per-period rate, §24) is accrued
            once. ``None`` applies :data:`DEFAULT_FUNDING_INTERVAL_HOURS`. This is
            asset-class metadata (§4.5) and belongs on the instrument, not on the
            engine overrides: cross-engine parity (§16) requires the schedule to travel
            with the instrument rather than with one adapter's knobs.
    """

    symbol: str
    asset_class: str
    tick_size: float | None = None
    contract_multiplier: float | None = None
    calendar: str | None = None
    settlement_currency: str | None = None
    funding_model: str | None = None
    borrow_model: str | None = None
    funding_interval_hours: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise InvalidInstrumentError("symbol must be a non-empty string")
        if not isinstance(self.asset_class, str) or self.asset_class not in ASSET_CLASSES:
            raise InvalidInstrumentError(
                f"unsupported asset_class={self.asset_class!r}; "
                f"expected one of {sorted(ASSET_CLASSES)}"
            )
        if self.tick_size is not None and not _is_positive_number(self.tick_size):
            raise InvalidInstrumentError("tick_size must be a positive finite number")
        if self.contract_multiplier is not None and not _is_positive_number(
            self.contract_multiplier
        ):
            raise InvalidInstrumentError(
                "contract_multiplier must be a positive finite number"
            )
        if (
            self.funding_interval_hours is not None
            and not _is_positive_number(self.funding_interval_hours)
        ):
            raise InvalidInstrumentError(
                "funding_interval_hours must be a positive finite number"
            )

    @property
    def allows_short(self) -> bool:
        """Whether this instrument's ``asset_class`` can open a short position (§4.5)."""
        return allows_short(self.asset_class)

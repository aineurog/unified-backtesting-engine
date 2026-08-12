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

from dataclasses import dataclass

from ube.core.errors import InvalidInstrumentError

__all__ = ["Instrument", "ASSET_CLASSES"]

# The asset classes the engine supports. The labels are locked by the per-asset-class
# deterministic fixtures of §16 (``crypto_perp/``, ``futures/``, ``stocks/``,
# ``forex/``, ``commodities/``) and the ``asset_class="crypto_perp"`` example of §4.5.
ASSET_CLASSES: frozenset[str] = frozenset(
    {"crypto_perp", "futures", "stocks", "forex", "commodities"}
)


@dataclass(frozen=True)
class Instrument:
    """Asset-class metadata for a single tradable instrument (§4.5).

    ``symbol`` and ``asset_class`` are required. Every asset-class-varying field is
    optional and defaults to ``None``; a ``None`` value means "apply the sensible
    asset-class default", which the cost/calendar modules resolve. This is not a
    silently-guessed default (§4.8): the *asset class* is declared explicitly and
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
    """

    symbol: str
    asset_class: str
    tick_size: float | None = None
    contract_multiplier: float | None = None
    calendar: str | None = None
    settlement_currency: str | None = None
    funding_model: str | None = None
    borrow_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise InvalidInstrumentError("symbol must be a non-empty string")
        if not isinstance(self.asset_class, str) or self.asset_class not in ASSET_CLASSES:
            raise InvalidInstrumentError(
                f"unsupported asset_class={self.asset_class!r}; "
                f"expected one of {sorted(ASSET_CLASSES)}"
            )
        if self.tick_size is not None and self.tick_size <= 0:
            raise InvalidInstrumentError("tick_size must be positive")
        if self.contract_multiplier is not None and self.contract_multiplier <= 0:
            raise InvalidInstrumentError("contract_multiplier must be positive")

"""Canonical :class:`~ube.core.instrument.Instrument` → Nautilus instrument mapping (§3.1, §4.3).

Maps the canonical asset-class metadata onto the concrete Nautilus instrument class per
plan §3.1 / requirements §4.5 / §16:

+----------------+------------------------+---------------------------------------------+
| ``asset_class``| Nautilus class         | Notes                                       |
+================+========================+=============================================+
| ``futures``    | ``FuturesContract``    | multiplier from ``contract_multiplier``     |
| ``commodities``| ``FuturesContract``    | e.g. gold GC (``multiplier=100``)           |
| ``crypto_perp``| ``CryptoPerpetual``    | linear (``is_inverse=False``)               |
| ``stocks``     | ``Equity``             |                                             |
| ``forex``      | ``CurrencyPair``       |                                             |
| ``crypto_spot``| ``CurrencyPair``       | deferred (no fixture yet)                   |
+----------------+------------------------+---------------------------------------------+

Precision/increments come from the canonical ``tick_size`` / ``contract_multiplier``,
with the reference ``constants.py`` values as fallback defaults. Fees default to
``0.0`` and are set from the ``maker_fee`` / ``taker_fee`` overrides — the adapter
folds the core cost model's ``commission + slippage`` into both at construction
(plan §5.3); this module never invents a fee.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import (
    CryptoPerpetual,
    CurrencyPair,
    Equity,
    FuturesContract,
)
from nautilus_trader.model.objects import Currency, Price, Quantity

from ube.adapters.nautilus_adapter.overrides import DEFAULT_VENUE
from ube.core.errors import InvalidInstrumentError
from ube.core.instrument import Instrument

__all__ = [
    "NautilusInstrumentBuild",
    "build_instrument",
]

#: Default price precision per asset class when ``tick_size`` cannot be used.
_DEFAULT_PRICE_PRECISION: dict[str, int] = {
    "crypto_perp": 2,
    "crypto_spot": 2,
    "futures": 2,
    "commodities": 2,
    "stocks": 2,
    "forex": 5,
}

#: Default size precision per asset class (reference ``constants.py``).
_DEFAULT_SIZE_PRECISION: dict[str, int] = {
    "crypto_perp": 3,
    "crypto_spot": 3,
    "futures": 0,
    "commodities": 0,
    "stocks": 0,
    "forex": 5,
}

#: Futures ``expiration_ns`` sentinel. The canonical ``Instrument`` carries no expiry
#: (§4.5 asset-class metadata), but Nautilus rejects matching against a contract whose
#: ``expiration_ns`` is in the past — ``0`` is read as "expired 1970" and every order
#: bounces. A far-future sentinel keeps the synthetic contract tradable over any backtest
#: window while the (single) contract represents the whole series.
_FUTURES_EXPIRATION_NS: int = 4_102_444_800_000_000_000  # 2100-01-01 00:00:00 UTC

#: Default size increment per asset class.
_DEFAULT_SIZE_INCREMENT: dict[str, str] = {
    "crypto_perp": "0.001",
    "crypto_spot": "0.001",
    "futures": "1",
    "commodities": "1",
    "stocks": "1",
    "forex": "0.00001",
}

#: Default tick size when the canonical instrument omits it (rare; fixtures always set it).
_DEFAULT_TICK_SIZE: dict[str, str] = {
    "crypto_perp": "0.01",
    "crypto_spot": "0.01",
    "futures": "0.25",
    "commodities": "0.1",
    "stocks": "0.01",
    "forex": "0.00001",
}


@dataclass(frozen=True)
class NautilusInstrumentBuild:
    """The result of mapping a canonical :class:`Instrument` onto Nautilus."""

    instrument: Any  # a concrete nautilus_trader instrument (FuturesContract, ...)
    instrument_id: InstrumentId


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _decimal_places(value: float) -> int:
    """Number of decimal places in ``value`` (e.g. ``0.25 -> 2``, ``0.1 -> 1``)."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def _base_quote(canonical: Instrument) -> tuple[str, str]:
    """Split the canonical symbol into base/quote currency codes.

    ``BTC-USDT`` -> ``("BTC", "USDT")``; ``EURUSD`` (forex) -> ``("EUR", "USD")`` via the
    declared settlement currency; everything else (ES, GC, AAPL) -> ``(settlement, settlement)``.
    """
    symbol = canonical.symbol
    if "-" in symbol:
        base, quote = symbol.split("-", 1)
        return base, quote
    quote = canonical.settlement_currency or "USD"
    if canonical.asset_class == "forex" and symbol.endswith(quote) and len(symbol) > len(quote):
        return symbol[: -len(quote)], quote
    return quote, quote


def _precision(canonical: Instrument, overrides: Mapping[str, Any]) -> tuple[int, str]:
    """Resolve ``(price_precision, price_increment)`` for the instrument.

    Nautilus requires ``price_precision == price_increment.precision``, so when the
    ``price_precision`` override differs from the tick-derived default the increment
    string is re-formatted to that precision (e.g. tick ``0.25`` with precision 3 ->
    ``"0.250"``).
    """
    tick = canonical.tick_size
    if tick is not None:
        default_precision = _decimal_places(tick)
    else:
        tick = float(_DEFAULT_TICK_SIZE[canonical.asset_class])
        default_precision = _DEFAULT_PRICE_PRECISION[canonical.asset_class]
    precision = int(overrides.get("price_precision", default_precision))
    increment = f"{tick:.{precision}f}"
    return precision, increment


def _size(canonical: Instrument, overrides: Mapping[str, Any]) -> tuple[int, str]:
    """Resolve ``(size_precision, size_increment)`` for the instrument."""
    precision = overrides.get("size_precision", _DEFAULT_SIZE_PRECISION[canonical.asset_class])
    return int(precision), _DEFAULT_SIZE_INCREMENT[canonical.asset_class]


def _margin(overrides: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    """Resolve ``(margin_init, margin_maint)`` from the ``leverage`` override.

    ``margin_init = 1/leverage`` when leverage is declared; otherwise the reference
    defaults (``0.2`` init, half for maintenance).
    """
    leverage = overrides.get("leverage", 0.0)
    if leverage and leverage > 0:
        margin_init = Decimal("1") / Decimal(str(leverage))
    else:
        margin_init = Decimal("0.2")
    return margin_init, margin_init / Decimal("2")


def _fees(overrides: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    """Resolve ``(maker_fee, taker_fee)`` from the override fraction (default ``0``).

    The override fees are *fractions of notional* charged per fill (plan §5.3): the
    adapter folds ``cost_model.commission + slippage`` into both fees at construction,
    so Nautilus's fill accounting reports the same per-fill cost as the core cost
    model. A missing override defaults to ``0`` (zero-cost, §7.1).
    """
    return (
        Decimal(str(overrides.get("maker_fee", 0.0))),
        Decimal(str(overrides.get("taker_fee", 0.0))),
    )


def _symbol_id(canonical: Instrument, overrides: Mapping[str, Any]) -> tuple[InstrumentId, Symbol]:
    """Build the ``InstrumentId`` (canonical symbol verbatim on the synthetic venue)."""
    venue = Venue(str(overrides.get("venue", DEFAULT_VENUE)))
    raw_symbol = Symbol(canonical.symbol)
    return InstrumentId(raw_symbol, venue), raw_symbol


# ---------------------------------------------------------------------------
# Per-asset-class builders.
# ---------------------------------------------------------------------------


def _build_futures(
    canonical: Instrument, overrides: Mapping[str, Any]
) -> tuple[Any, InstrumentId]:
    instrument_id, raw_symbol = _symbol_id(canonical, overrides)
    price_precision, price_increment = _precision(canonical, overrides)
    margin_init, margin_maint = _margin(overrides)
    maker_fee, taker_fee = _fees(overrides)
    _, quote = _base_quote(canonical)
    multiplier = canonical.contract_multiplier if canonical.contract_multiplier is not None else 1.0
    asset_class = (
        AssetClass.INDEX if canonical.asset_class == "futures" else AssetClass.COMMODITY
    )
    instrument = FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        asset_class=asset_class,
        currency=Currency.from_str(quote),
        price_precision=price_precision,
        price_increment=Price.from_str(price_increment),
        multiplier=Quantity.from_str(str(multiplier)),
        lot_size=Quantity.from_int(1),
        underlying=quote,
        activation_ns=0,
        expiration_ns=_FUTURES_EXPIRATION_NS,
        ts_event=0,
        ts_init=0,
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )
    return instrument, instrument_id


def _build_crypto_perp(
    canonical: Instrument, overrides: Mapping[str, Any]
) -> tuple[Any, InstrumentId]:
    instrument_id, raw_symbol = _symbol_id(canonical, overrides)
    price_precision, price_increment = _precision(canonical, overrides)
    size_precision, size_increment = _size(canonical, overrides)
    margin_init, margin_maint = _margin(overrides)
    maker_fee, taker_fee = _fees(overrides)
    base, quote = _base_quote(canonical)
    settlement = canonical.settlement_currency or quote
    instrument = CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        base_currency=Currency.from_str(base),
        quote_currency=Currency.from_str(quote),
        settlement_currency=Currency.from_str(settlement),
        is_inverse=False,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_increment),
        size_increment=Quantity.from_str(size_increment),
        ts_event=0,
        ts_init=0,
        multiplier=Quantity.from_int(1),
        lot_size=Quantity.from_int(1),
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )
    return instrument, instrument_id


def _build_equity(
    canonical: Instrument, overrides: Mapping[str, Any]
) -> tuple[Any, InstrumentId]:
    instrument_id, raw_symbol = _symbol_id(canonical, overrides)
    price_precision, price_increment = _precision(canonical, overrides)
    margin_init, margin_maint = _margin(overrides)
    maker_fee, taker_fee = _fees(overrides)
    _, quote = _base_quote(canonical)
    instrument = Equity(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        currency=Currency.from_str(quote),
        price_precision=price_precision,
        price_increment=Price.from_str(price_increment),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )
    return instrument, instrument_id


def _build_currency_pair(
    canonical: Instrument, overrides: Mapping[str, Any]
) -> tuple[Any, InstrumentId]:
    instrument_id, raw_symbol = _symbol_id(canonical, overrides)
    price_precision, price_increment = _precision(canonical, overrides)
    size_precision, size_increment = _size(canonical, overrides)
    margin_init, margin_maint = _margin(overrides)
    maker_fee, taker_fee = _fees(overrides)
    base, quote = _base_quote(canonical)
    instrument = CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        base_currency=Currency.from_str(base),
        quote_currency=Currency.from_str(quote),
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_increment),
        size_increment=Quantity.from_str(size_increment),
        ts_event=0,
        ts_init=0,
        multiplier=Quantity.from_int(1),
        lot_size=Quantity.from_int(1),
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )
    return instrument, instrument_id


#: asset_class -> builder. ``crypto_spot`` maps to a ``CurrencyPair`` (reference
#: ``constants.py``: ("crypto", "spot") -> "CurrencyPair"); shorting is not permitted
#: on it (the actor's ``_allow_short`` rule, mirroring the reference ``signals.py``).
_BUILDERS: dict[str, Any] = {
    "futures": _build_futures,
    "commodities": _build_futures,
    "crypto_perp": _build_crypto_perp,
    "crypto_spot": _build_currency_pair,
    "stocks": _build_equity,
    "forex": _build_currency_pair,
}


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def build_instrument(
    canonical: Instrument,
    overrides: Mapping[str, Any] | None = None,
) -> NautilusInstrumentBuild:
    """Map a canonical :class:`~ube.core.instrument.Instrument` onto a Nautilus instrument.

    The synthetic venue and per-asset-class Nautilus class follow plan §3.1; precision and
    increments come from the canonical ``tick_size`` / ``contract_multiplier`` (with
    reference defaults as fallback). ``maker_fee`` / ``taker_fee`` are taken from the
    overrides (default ``0.0``) — the adapter folds the core cost model's
    ``commission + slippage`` into both (plan §5.3).

    Args:
        canonical: The canonical :class:`~ube.core.instrument.Instrument`.
        overrides: Validated
            :class:`~ube.adapters.nautilus_adapter.overrides.NautilusEngineOverrides`
            (``venue``, ``price_precision``, ``size_precision``, ``leverage``,
            ``maker_fee``, ``taker_fee``).

    Returns:
        The built Nautilus instrument plus its :class:`InstrumentId`.

    Raises:
        InvalidInstrumentError: ``asset_class`` is not supported by the adapter.
    """
    if canonical.asset_class not in _BUILDERS:
        raise InvalidInstrumentError(
            f"asset_class={canonical.asset_class!r} is not supported by the nautilus "
            f"adapter; supported: {sorted(_BUILDERS)}"
        )
    overrides = overrides if overrides is not None else {}
    instrument, instrument_id = _BUILDERS[canonical.asset_class](canonical, overrides)
    return NautilusInstrumentBuild(instrument=instrument, instrument_id=instrument_id)
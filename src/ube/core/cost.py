"""Cost-model interface and asset-class default resolution (§4.5, §7.1, §7.2, §24).

The cost model is deliberately thin. §24 defers the *full* list of asset-class cost
defaults — crypto-perp funding, futures rollover, forex swap, and short-side
borrow / hard-to-borrow fees — to "filled in per asset class as adapters are built"
(Phase 2). This module therefore provides only:

- A frozen :class:`CostModel` carrying the four cost dimensions the spec names
  (commission/fee, slippage, funding/swap, borrow), each a single explicit rate that
  defaults to ``0.0`` so the default is zero-cost (§7.1).
- Pure, vectorizable cost functions — :func:`fill_cost` for a single fill and
  :func:`carrying_cost` for the per-bar carrying cost — that the ledger (item 07)
  calls. They are deterministic functions of notional/side/bar-span with no state.
- :func:`resolve_cost_model`, mapping an :class:`~ube.core.instrument.Instrument` to a
  default ``CostModel``: zero-cost everywhere, except ``crypto_perp`` which gets a
  documented "reasonable" funding + fee default (§4.5). Every other asset class
  resolves to zero-cost until §24 lands (see the note in :func:`resolve_cost_model`).

Rates are expressed as fractions of notional: ``commission`` and ``slippage`` are
charged per fill; ``funding`` and ``borrow`` are charged per bar held. Keeping the
model bar-agnostic (§4.3) — the bar span is supplied by the caller, so the same model
works for any bar type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ube.core.errors import ConfigError, InvalidInstrumentError
from ube.core.instrument import Instrument

__all__ = [
    "CostModel",
    "ZERO_COST",
    "resolve_cost_model",
    "fill_cost",
    "carrying_cost",
]

#: The cost dimensions in their canonical order (§4.5, §7.1).
RATE_FIELDS: tuple[str, ...] = ("commission", "slippage", "funding", "borrow")


def _validate_rate(value: object, name: str) -> None:
    """Validate a single cost rate: a finite number (bool rejected).

    A negative rate is deliberately *not* rejected — a negative funding rate models
    received funding (the trader is paid), which is legitimate for perps. The sign
    convention is "positive = a cost the trader pays"; the four rates are otherwise
    just numbers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"cost-model rate {name!r} must be a number; got {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"cost-model rate {name!r} must be finite; got {value!r}")


@dataclass(frozen=True)
class CostModel:
    """Frozen cost model: the four cost dimensions of §4.5, each a rate.

    Every field defaults to ``0.0``, so a bare ``CostModel()`` is the zero-cost model
    of §7.1. Rates are fractions of notional:

    - ``commission`` / ``slippage`` — charged per fill (see :func:`fill_cost`).
    - ``funding`` / ``borrow`` — charged per bar held (see :func:`carrying_cost`);
      ``borrow`` applies to the short side only.

    Attributes:
        commission: Per-fill commission/fee rate, as a fraction of notional.
        slippage: Per-fill slippage rate, as a fraction of notional.
        funding: Per-bar funding/swap rate (perps, forex), as a fraction of notional.
        borrow: Per-bar short-side borrow / hard-to-borrow fee rate (stocks, futures),
            as a fraction of notional; applied only while short.
    """

    commission: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    borrow: float = 0.0

    def __post_init__(self) -> None:
        for name in RATE_FIELDS:
            _validate_rate(getattr(self, name), name)


#: The zero-cost model (§7.1) — a shared, immutable default.
ZERO_COST: CostModel = CostModel()

#: The "reasonable" default for ``crypto_perp`` (§4.5): a representative taker fee of
#: 5 bps plus a 1 bp per-bar funding rate. §24 defers the authoritative per-asset-class
#: numbers to Phase 2; these are documented placeholders, not a researched schedule.
_CRYPTO_PERP_COST: CostModel = CostModel(commission=0.0005, funding=0.0001)


def resolve_cost_model(instrument: Instrument | None = None) -> CostModel:
    """Resolve the default cost model for ``instrument`` (§4.5, §7.1, §7.2).

    - ``crypto_perp`` → a documented "reasonable" funding + fee default
      (``commission=0.0005``, ``funding=0.0001``; slippage and borrow stay zero).
    - Everything else (including ``None``) → zero-cost.

    §24: the full per-asset-class cost defaults — futures rollover, forex swap, and
    short-side borrow / hard-to-borrow fees, together with a resolution of an
    ``Instrument``'s ``funding_model`` / ``borrow_model`` string references — are
    filled in per asset class as adapters are built (Phase 2). Until then every
    non-perp asset class resolves to zero-cost rather than a guessed number (§4.7).
    """
    if instrument is None:
        return ZERO_COST
    if not isinstance(instrument, Instrument):
        raise InvalidInstrumentError(
            f"resolve_cost_model expects an Instrument or None; "
            f"got {type(instrument).__name__}"
        )
    if instrument.asset_class == "crypto_perp":
        return _CRYPTO_PERP_COST
    return ZERO_COST


def fill_cost(model: CostModel, *, notional: ArrayLike) -> np.ndarray:
    """Commission + slippage cost of a fill, as a fraction of the fill notional.

    ``notional`` is the non-negative gross value of the fill (``price * size``); it may
    be a scalar or an array. The result is ``(commission + slippage) * notional`` —
    symmetric in side, so direction is not needed. Pure and vectorized.
    """
    n = np.asarray(notional, dtype=np.float64)
    return (model.commission + model.slippage) * n


def carrying_cost(
    model: CostModel,
    *,
    notional: ArrayLike,
    side: ArrayLike,
    bar_span: ArrayLike,
) -> np.ndarray:
    """Per-bar carrying cost (funding/swap + short-side borrow) over ``bar_span`` bars.

    ``notional`` is the non-negative value of the open position, ``side`` the direction
    (``+1`` long / ``-1`` short / ``0`` flat), and ``bar_span`` the number of bars the
    position is held. Any of the three may be a scalar or an array (broadcast
    together). Funding applies to both sides; borrow applies only while short
    (``side < 0``)::

        cost = (funding + borrow * (side < 0)) * notional * bar_span

    Pure and vectorized — no per-bar Python loop.
    """
    n = np.asarray(notional, dtype=np.float64)
    s = np.asarray(side, dtype=np.float64)
    span = np.asarray(bar_span, dtype=np.float64)
    short = s < 0
    return (model.funding + model.borrow * short) * n * span

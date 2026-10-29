"""Multi-asset portfolio configuration and shared-capital sizing (§ portfolio plan).

This module is the canonical home of the *portfolio-level* concepts, following the same
pattern as :class:`~ube.core.risk.sizing.SizeModel`:

* :class:`PortfolioConfig` describes several instruments that share one pool of capital.
* :class:`EqualWeight` / :class:`FixedWeight` decide how that pool is split
  (:data:`AllocationPolicy`).
* :func:`size_portfolio_entry` is the portfolio-aware sibling of
  :func:`~ube.core.risk.sizing.size_position`. It answers "how many units may *this*
  instrument add without breaching its share of the shared pool", given what the other
  instruments have already committed.

The existing single-instrument path is untouched: ``size_position`` keeps its contract,
and ``PortfolioConfig`` is optional on :class:`~ube.core.config.BacktestConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ube.core.cost import CostModel, fill_cost
from ube.core.errors import ConfigError
from ube.core.instrument import Instrument

if TYPE_CHECKING:
    from ube.core.risk.sizing import SizeModel

__all__ = [
    "AllocationPolicy",
    "DEFAULT_PORTFOLIO_BALANCE",
    "EqualWeight",
    "FixedWeight",
    "PortfolioConfig",
    "RebalancePolicy",
    "allocation_weights",
    "size_portfolio_entry",
]

#: Default shared account balance for a portfolio backtest, matching the engine adapters'
#: ``DEFAULT_STARTING_BALANCE``.
DEFAULT_PORTFOLIO_BALANCE = 100_000.0

#: Relative tolerance for the "weights sum to at most 100%" invariant.
_WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class EqualWeight:
    """Split the shared pool evenly across the configured instruments."""

    def weights(self, instrument_ids: Sequence[str]) -> dict[str, float]:
        """Return ``1/n`` for every instrument (``{}`` for an empty list)."""
        ids = tuple(instrument_ids)
        if not ids:
            return {}
        share = 1.0 / float(len(ids))
        return {instrument_id: share for instrument_id in ids}


@dataclass(frozen=True)
class FixedWeight:
    """Explicit, static split of the shared pool.

    Attributes:
        weights: ``instrument_id -> fraction`` of shared equity. Fractions must be finite
            and non-negative; the sum must not exceed ``1.0`` (the remainder stays in
            cash). Every configured instrument must have an entry.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Mapping) or not self.weights:
            raise ConfigError("FixedWeight.weights must be a non-empty mapping")
        cleaned: dict[str, float] = {}
        for instrument_id, weight in self.weights.items():
            if not isinstance(instrument_id, str) or not instrument_id.strip():
                raise ConfigError(
                    f"FixedWeight weights keys must be non-empty strings; got {instrument_id!r}"
                )
            if isinstance(weight, bool) or not isinstance(weight, int | float):
                raise ConfigError(
                    f"FixedWeight[{instrument_id!r}] must be a number; got {weight!r}"
                )
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise ConfigError(
                    f"FixedWeight[{instrument_id!r}] must be finite and >= 0; got {weight!r}"
                )
            cleaned[instrument_id] = value
        total = sum(cleaned.values())
        if total > 1.0 + _WEIGHT_TOL:
            raise ConfigError(
                "FixedWeight weights must sum to at most 1.0; "
                f"got {total!r} — the shared pool cannot be over-allocated"
            )
        object.__setattr__(self, "weights", cleaned)

    def weights_for(self, instrument_ids: Sequence[str]) -> dict[str, float]:
        """Return the declared weights, requiring every instrument to be present."""
        ids = tuple(instrument_ids)
        missing = [iid for iid in ids if iid not in self.weights]
        if missing:
            raise ConfigError(
                f"FixedWeight is missing an allocation for {missing}; declared: "
                f"{sorted(self.weights)}"
            )
        return {iid: float(self.weights[iid]) for iid in ids}


AllocationPolicy = EqualWeight | FixedWeight


@dataclass(frozen=True)
class RebalancePolicy:
    """Reserved for scheduled rebalancing; not implemented in this pass.

    Declared so a config can express intent without silently pretending the feature
    exists: :class:`PortfolioConfig` rejects a non-``None`` value with a clear error.
    """

    every: str | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    """Several instruments sharing one pool of capital.

    Attributes:
        instruments: The traded instruments. Symbols must be unique — the symbol is the
            key used by ``data`` / ``signals`` mappings and by the allocation weights.
        base_currency: The currency the combined equity curve is denominated in
            (explicit-over-default, §4.7 — no silent default).
        allocation: How the shared pool is split. ``EqualWeight()`` by default.
        rebalance: Reserved; must be ``None`` in this pass.
        sizing: Optional portfolio-level sizing override. ``None`` keeps each
            instrument's own ``config.risk.sizing``; the allocation scale is still
            applied to the capital each instrument receives.
    """

    instruments: tuple[Instrument, ...]
    base_currency: str
    allocation: AllocationPolicy = EqualWeight()
    rebalance: RebalancePolicy | None = None
    sizing: SizeModel | None = None

    def __post_init__(self) -> None:
        instruments = tuple(self.instruments)
        if not instruments:
            raise ConfigError("PortfolioConfig.instruments must not be empty")
        for instrument in instruments:
            if not isinstance(instrument, Instrument):
                raise ConfigError(
                    "PortfolioConfig.instruments must contain Instrument instances; "
                    f"got {type(instrument).__name__}"
                )
        symbols = [instrument.symbol for instrument in instruments]
        if len(set(symbols)) != len(symbols):
            raise ConfigError(f"PortfolioConfig instrument symbols must be unique; got {symbols}")
        object.__setattr__(self, "instruments", instruments)
        if not isinstance(self.base_currency, str) or not self.base_currency.strip():
            raise ConfigError("PortfolioConfig.base_currency must be a non-empty string")
        if not isinstance(self.allocation, EqualWeight | FixedWeight):
            raise ConfigError(
                "PortfolioConfig.allocation must be EqualWeight or FixedWeight; "
                f"got {type(self.allocation).__name__}"
            )
        if self.rebalance is not None:
            raise ConfigError(
                "PortfolioConfig.rebalance is reserved and not implemented in this pass; "
                "leave it as None"
            )
        # A FixedWeight split must cover every configured instrument.
        if isinstance(self.allocation, FixedWeight):
            self.allocation.weights_for(symbols)
            unknown = set(self.allocation.weights) - set(symbols)
            if unknown:
                raise ConfigError(
                    f"FixedWeight references instruments not in PortfolioConfig: {sorted(unknown)}"
                )
        if self.sizing is not None:
            from ube.core.risk.sizing import SizeModel

            if not isinstance(self.sizing, SizeModel):
                raise ConfigError(
                    "PortfolioConfig.sizing must be a SizeModel or None; "
                    f"got {type(self.sizing).__name__}"
                )
            if self.sizing.kind == "fixed_units":
                raise ConfigError(
                    "PortfolioConfig.sizing='fixed_units' cannot respect a shared-capital "
                    "allocation; use a capital-proportional SizeModel"
                )

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        """The configured instrument symbols, in declaration order."""
        return tuple(instrument.symbol for instrument in self.instruments)

    @property
    def instrument_map(self) -> dict[str, Instrument]:
        """``symbol -> Instrument`` for the configured instruments."""
        return {instrument.symbol: instrument for instrument in self.instruments}

    def weights(self) -> dict[str, float]:
        """The allocation fraction for every configured instrument."""
        return allocation_weights(self.allocation, self.instrument_ids)

    def weight(self, instrument_id: str) -> float:
        """The allocation fraction for one instrument (``0.0`` when undeclared)."""
        if instrument_id not in self.instrument_ids:
            raise ConfigError(
                f"instrument {instrument_id!r} is not part of the portfolio "
                f"({list(self.instrument_ids)})"
            )
        return self.weights()[instrument_id]

    def validate_data(self, keys: Sequence[str]) -> None:
        """Raise when the supplied data/signal keys do not match the portfolio."""
        supplied = tuple(keys)
        expected = self.instrument_ids
        missing = [iid for iid in expected if iid not in supplied]
        extra = [key for key in supplied if key not in expected]
        if missing or extra:
            raise ConfigError(
                "portfolio data/signals keys must match PortfolioConfig.instruments; "
                f"missing={missing}, unexpected={extra}, expected={list(expected)}"
            )


def allocation_weights(
    allocation: AllocationPolicy, instrument_ids: Sequence[str]
) -> dict[str, float]:
    """Resolve an :data:`AllocationPolicy` to ``instrument_id -> fraction``.

    ``EqualWeight`` splits the pool ``1/n``; ``FixedWeight`` returns its declared split
    (validated to cover every instrument and not exceed 100%).
    """
    ids = tuple(instrument_ids)
    if not ids:
        raise ConfigError("allocation_weights requires at least one instrument")
    if isinstance(allocation, EqualWeight):
        return allocation.weights(ids)
    if isinstance(allocation, FixedWeight):
        return allocation.weights_for(ids)
    raise ConfigError(
        f"unsupported allocation policy {type(allocation).__name__}; "
        "expected EqualWeight or FixedWeight"
    )


def _entry_fee_rate(cost_model: CostModel | None) -> float:
    """Entry commission as a fraction of notional (reuses the core cost model)."""
    if cost_model is None:
        return 0.0
    return float(fill_cost(cost_model, notional=1.0).item())


def _validate_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ConfigError(f"{name} must be finite and non-negative; got {value!r}")
    return number


def size_portfolio_entry(
    allocation: AllocationPolicy,
    *,
    portfolio_equity: float,
    open_commitments: Mapping[str, float],
    instrument_id: str,
    price: float,
    cost_model: CostModel | None = None,
    target_weight: float | None = None,
) -> np.ndarray:
    """Units this instrument may still add from its share of a shared capital pool.

    This is the portfolio analogue of
    :func:`~ube.core.risk.sizing.size_position`: it never sees one instrument's isolated
    balance, only the combined equity and what the other instruments have already
    committed.

    Args:
        allocation: The pool-wide split policy. ``EqualWeight`` reserves an even share
            per instrument that is (or is about to be) open; ``FixedWeight`` uses the
            declared static fraction.
        portfolio_equity: The shared pool's current equity, in the base currency.
        open_commitments: ``instrument_id -> notional`` already committed by every
            instrument (including this one, if it is topping up).
        instrument_id: The instrument requesting capital.
        price: The fill price used to convert the cash budget into units.
        cost_model: Optional cost model; the entry fee is reserved from the budget.
        target_weight: Optional explicit fraction overriding ``allocation``. Used by the
            batch orchestrator to seed each leg with its precomputed share.

    Returns:
        The additional units as a float64 array (0-d for scalar inputs). ``0.0`` when the
        instrument's share is already fully committed.
    """
    equity = _validate_non_negative(portfolio_equity, "portfolio_equity")
    px = _validate_non_negative(price, "price")
    if px <= 0.0:
        raise ConfigError(f"price must be positive; got {price!r}")
    if not isinstance(instrument_id, str) or not instrument_id.strip():
        raise ConfigError("instrument_id must be a non-empty string")
    if not isinstance(open_commitments, Mapping):
        raise ConfigError("open_commitments must be a mapping")

    commitments: dict[str, float] = {}
    for key, value in open_commitments.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"open_commitments keys must be non-empty strings; got {key!r}")
        commitments[key] = _validate_non_negative(value, f"open_commitments[{key!r}]")

    committed_total = sum(commitments.values())
    current = commitments.get(instrument_id, 0.0)
    available_total = max(equity - committed_total, 0.0)

    if target_weight is not None:
        weight = _validate_non_negative(target_weight, "target_weight")
        if weight > 1.0 + _WEIGHT_TOL:
            raise ConfigError(f"target_weight must be <= 1.0; got {target_weight!r}")
    elif isinstance(allocation, FixedWeight):
        weight = float(allocation.weights.get(instrument_id, 0.0))
    elif isinstance(allocation, EqualWeight):
        # An instrument that is already committed keeps its slot; a new instrument adds
        # one more slot to the equal split.
        slots = len(commitments) + (0 if instrument_id in commitments else 1)
        weight = 1.0 / float(max(slots, 1))
    else:
        raise ConfigError(
            f"unsupported allocation policy {type(allocation).__name__}; "
            "expected EqualWeight or FixedWeight"
        )

    target_notional = equity * weight
    budget = min(max(target_notional - current, 0.0), available_total)
    fee = _entry_fee_rate(cost_model)
    units = budget / (px * (1.0 + fee))
    return np.asarray(units, dtype=np.float64)


def portfolio_metadata(config: Any) -> dict[str, Any]:
    """A small JSON-safe description of a portfolio run for logs/experiments."""
    allocation = getattr(config, "allocation", None)
    if isinstance(allocation, EqualWeight):
        policy: dict[str, Any] = {"kind": "equal_weight"}
    elif isinstance(allocation, FixedWeight):
        policy = {"kind": "fixed_weight", "weights": dict(allocation.weights)}
    else:  # pragma: no cover — PortfolioConfig validates the policy
        policy = {"kind": type(allocation).__name__}
    return {
        "instruments": list(getattr(config, "instrument_ids", ())),
        "base_currency": getattr(config, "base_currency", None),
        "allocation": policy,
    }

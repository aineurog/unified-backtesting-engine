"""Position-sizing subsystem (§6.3, §8).

Direction and timing (signals, §6.1) and magnitude (sizing) are decoupled and
extensible independently: the signal says *what* to do, sizing says *how much*. Five
sizing kinds (§6.3):

- ``fixed_fraction``   — allocate ``value`` fraction of capital
  (position notional = ``value * capital``).
- ``fixed_units``      — buy exactly ``value`` units.
- ``volatility_target``— size so position volatility = ``value`` fraction of portfolio.
- ``all_in``           — 100% of capital per position (units = ``capital / price``).
- ``equal_weight``     — split capital equally across ``n`` positions (portfolio).

:class:`SizeModel` is a frozen dataclass carrying the ``kind`` and, where the kind needs
one, a ``value``. All sizers are pure, vectorized functions of capital/price/params that
return position size in *units* (float64), with no per-bar Python loops (§3 principle 1).

The ``fixed_fraction`` interpretation ("risk X% of capital") is deliberately *allocation
of a value-fraction of capital* (notional = ``value * capital``), not a risk-per-stop
scheme. A risk-per-stop variant ("risk X% of capital, stop distance D away") needs a stop
distance input and is out of scope here (flagged in the item report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ube.core.errors import ConfigError

__all__ = [
    "SizeKind",
    "SIZE_KINDS",
    "SizeModel",
    "size_position",
    "fixed_fraction_size",
    "fixed_units_size",
    "volatility_target_size",
    "all_in_size",
    "equal_weight_size",
]

SizeKind = Literal[
    "fixed_fraction", "fixed_units", "volatility_target", "all_in", "equal_weight"
]

#: The valid sizing kinds, in the §6.3 order.
SIZE_KINDS: tuple[str, ...] = (
    "fixed_fraction",
    "fixed_units",
    "volatility_target",
    "all_in",
    "equal_weight",
)

#: Kinds that require a ``value`` (the fraction / unit count / target volatility).
_VALUE_REQUIRED: frozenset[str] = frozenset(
    {"fixed_fraction", "fixed_units", "volatility_target"}
)


def _validate_value(value: object, name: str = "SizeModel.value") -> float:
    """Validate a sizing ``value``: a finite, strictly positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number; got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        raise ConfigError(f"{name} must be finite and positive; got {value!r}")
    return f


@dataclass(frozen=True)
class SizeModel:
    """Frozen position-sizing model (§6.3).

    Attributes:
        kind: One of :data:`SIZE_KINDS` (default ``"all_in"``, the §7.1 default).
        value: The sizing parameter, required for ``fixed_fraction`` /
            ``fixed_units`` / ``volatility_target`` and forbidden otherwise. Its
            meaning depends on ``kind`` (a fraction of capital, a unit count, or a
            target volatility fraction respectively).
    """

    kind: SizeKind = "all_in"
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in SIZE_KINDS:
            raise ConfigError(
                f"unsupported sizing kind={self.kind!r}; expected one of {SIZE_KINDS}"
            )
        if self.kind in _VALUE_REQUIRED:
            if self.value is None:
                raise ConfigError(f"sizing kind={self.kind!r} requires a value")
            object.__setattr__(self, "value", _validate_value(self.value))
        elif self.value is not None:
            raise ConfigError(
                f"sizing kind={self.kind!r} does not take a value; got {self.value!r}"
            )


def _as_float(value: ArrayLike, name: str) -> np.ndarray:
    """Coerce ``value`` to a float64 array and reject non-finite entries."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ConfigError(f"{name} must be finite")
    return arr


def _require_nonnegative(capital: np.ndarray) -> None:
    if (capital < 0).any():
        raise ConfigError("capital must be non-negative")


def _require_positive_price(price: np.ndarray) -> None:
    if (price <= 0).any():
        raise ConfigError("price must be positive")


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division, typed.

    numpy 2.x stubs type ``ndarray / ndarray`` (and ``np.divide``) as returning ``Any``,
    which trips ``mypy --strict``'s ``warn_return_any`` when returned directly. A typed
    local keeps the result honest without changing the (correct) element-wise semantics.
    """
    result: np.ndarray = numerator / denominator
    return result


def fixed_fraction_size(fraction: float, *, capital: ArrayLike, price: ArrayLike) -> np.ndarray:
    """Units for ``fraction`` of capital allocated to a position (§6.3).

    Position notional = ``fraction * capital``; ``units = fraction * capital / price``.
    ``capital`` and ``price`` may be scalars or arrays (broadcast together). ``fraction``
    is a fraction of capital — not a risk-per-stop amount (see the module docstring).
    """
    f = _validate_value(fraction, "fixed_fraction fraction")
    cap = _as_float(capital, "capital")
    px = _as_float(price, "price")
    _require_nonnegative(cap)
    _require_positive_price(px)
    return _divide(f * cap, px)


def fixed_units_size(units: ArrayLike) -> np.ndarray:
    """Units for the exact-unit-count sizer (§6.3): buy exactly ``value`` units."""
    arr = _as_float(units, "fixed_units value")
    if (arr <= 0).any():
        raise ConfigError("fixed_units value must be positive")
    return arr


def all_in_size(capital: ArrayLike, price: ArrayLike) -> np.ndarray:
    """Units for the naive all-in sizer: 100% of capital (units = ``capital / price``)."""
    cap = _as_float(capital, "capital")
    px = _as_float(price, "price")
    _require_nonnegative(cap)
    _require_positive_price(px)
    return _divide(cap, px)


def equal_weight_size(capital: ArrayLike, price: ArrayLike, n: int) -> np.ndarray:
    """Units for an equal split of capital across ``n`` positions (portfolio).

    ``units = (capital / n) / price``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ConfigError(f"equal_weight n must be a positive integer; got {n!r}")
    cap = _as_float(capital, "capital")
    px = _as_float(price, "price")
    _require_nonnegative(cap)
    _require_positive_price(px)
    return _divide(cap / float(n), px)


def volatility_target_size(
    target: float, *, capital: ArrayLike, price: ArrayLike, vol: ArrayLike
) -> np.ndarray:
    """Units so that position volatility equals ``target`` fraction of portfolio.

    The position's currency volatility is ``units * price * vol``, where ``vol`` is the
    instrument's per-bar volatility expressed as a fraction of price (dimensionless —
    e.g. ``0.02`` = 2%, i.e. an ATR/price ratio or a return standard deviation). Setting
    that equal to ``target * capital`` gives::

        units = target * capital / (price * vol)

    ``vol`` is a required *estimate input* (not computed here) — the caller supplies it
    from aux_data / an indicator (§5.2). Pure and vectorized.
    """
    t = _validate_value(target, "volatility_target value")
    cap = _as_float(capital, "capital")
    px = _as_float(price, "price")
    v = _as_float(vol, "vol")
    _require_nonnegative(cap)
    _require_positive_price(px)
    if (v <= 0).any():
        raise ConfigError("vol must be positive")
    return _divide(t * cap, px * v)


def size_position(
    kind_or_model: SizeModel | SizeKind,
    *,
    capital: ArrayLike,
    price: ArrayLike,
    n: int | None = None,
    vol: ArrayLike | None = None,
) -> np.ndarray:
    """Dispatch a :class:`SizeModel` (or a bare ``kind``) to its sizer (§6.3).

    ``kind_or_model`` is either a :class:`SizeModel`, or a bare :data:`SizeKind` string
    (valid only for value-free kinds — ``all_in`` / ``equal_weight``). ``n`` is required
    for ``equal_weight``; ``vol`` is required for ``volatility_target``. Returns units as
    a float64 array (0-d for scalar inputs).
    """
    model = (
        kind_or_model
        if isinstance(kind_or_model, SizeModel)
        else SizeModel(kind=kind_or_model)
    )

    if model.kind == "fixed_fraction":
        return fixed_fraction_size(
            cast(float, model.value), capital=capital, price=price
        )
    if model.kind == "fixed_units":
        return fixed_units_size(cast(float, model.value))
    if model.kind == "volatility_target":
        if vol is None:
            raise ConfigError("volatility_target sizing requires a vol estimate")
        return volatility_target_size(
            cast(float, model.value), capital=capital, price=price, vol=vol
        )
    if model.kind == "all_in":
        return all_in_size(capital, price)
    # equal_weight
    if n is None:
        raise ConfigError("equal_weight sizing requires n")
    return equal_weight_size(capital, price, n)

"""Benchmark curve builder (§10).

A benchmark is a reference return/equity curve computed over the same bar index as the
input data, so a strategy's results can later be compared against it. Three kinds
(§10):

- ``buy_and_hold`` (single-instrument default): the instrument's close-to-close returns
  and a normalized equity curve starting at ``1.0``.
- ``equal_weight`` (portfolio default): the average of each instrument's normalized
  equity curve.
- ``custom`` (portfolio): a weighted sum of each instrument's normalized equity curve.

This module builds the benchmark *curve only*. The benchmark *comparison* metrics of
§10 (total return, Sharpe, max drawdown, excess return, information ratio) are computed
later by the metrics/reporting layer and are deliberately out of scope here.

The curve container is :class:`BenchmarkCurve`, a frozen pair of ``returns`` /
``equity`` arrays. ``returns[i]`` is the period return from bar ``i-1`` to bar ``i``
(``returns[0] = 0.0`` — there is no prior bar), and ``equity[i]`` is the cumulative
normalized value starting at ``equity[0] = 1.0``. Both arrays are length ``n_bars`` and
align to the input bar index positionally. Portfolio inputs are expected to be
positionally aligned (equal ``n_bars``); the union-of-timestamps + forward-fill
alignment of §4.6 is a ledger concern (item 07), not this builder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError

__all__ = [
    "BenchmarkKind",
    "BENCHMARK_KINDS",
    "BenchmarkConfig",
    "BenchmarkCurve",
    "buy_and_hold_curve",
    "portfolio_curve",
    "build_benchmark",
]

BenchmarkKind = Literal["buy_and_hold", "equal_weight", "custom"]

#: The valid benchmark kinds (§10).
BENCHMARK_KINDS: tuple[str, ...] = ("buy_and_hold", "equal_weight", "custom")


def _coerce_weights(value: object) -> tuple[float, ...]:
    """Validate and normalize ``weights`` into an immutable ``tuple[float, ...]``.

    ``weights`` must be a non-empty sequence of finite, non-negative numbers with a
    positive sum. Negative/NaN weights, an empty sequence, and an all-zero sequence all
    raise :class:`~ube.core.errors.ConfigError` (§3 principle 6 — fail fast).
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError("weights must be a non-empty sequence of non-negative numbers")
    raw = tuple(value)
    if not raw:
        raise ConfigError("weights must not be empty")
    out: list[float] = []
    for w in raw:
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ConfigError(f"weights must be numeric; got {w!r}")
        f = float(w)
        if not math.isfinite(f) or f < 0:
            raise ConfigError(f"weights must be finite and non-negative; got {w!r}")
        out.append(f)
    if sum(out) <= 0:
        raise ConfigError("weights must have a positive sum")
    return tuple(out)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration (§10).

    Attributes:
        kind: The benchmark kind — ``"buy_and_hold"`` (single instrument),
            ``"equal_weight"`` (portfolio average), or ``"custom"`` (portfolio weighted
            sum).
        weights: The weight vector, required when ``kind="custom"`` and forbidden
            otherwise. Weights are normalized to sum to one when the curve is built, so
            any non-negative vector with positive sum is accepted (§10 "arbitrary
            weight vector").
    """

    kind: BenchmarkKind = "buy_and_hold"
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in BENCHMARK_KINDS:
            raise ConfigError(
                f"benchmark kind={self.kind!r} is not one of {BENCHMARK_KINDS}"
            )
        if self.kind == "custom":
            object.__setattr__(self, "weights", _coerce_weights(self.weights))
        elif self.weights is not None:
            raise ConfigError(
                f"weights only apply to kind='custom'; got kind={self.kind!r}"
            )


def _coerce_float1d(values: object, name: str) -> np.ndarray:
    """Coerce a column to a fresh 1-D float64 array, raising ``DataShapeError``."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DataShapeError(f"{name} must be 1-D; got shape {arr.shape}")
    try:
        return arr.astype(np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise DataShapeError(f"{name} is not numeric") from exc


@dataclass(frozen=True)
class BenchmarkCurve:
    """A frozen benchmark return/equity curve over the input bar index.

    Attributes:
        returns: float64[n] period returns; ``returns[0] = 0.0`` and
            ``returns[i] = equity[i] / equity[i-1] - 1`` for ``i >= 1``.
        equity: float64[n] normalized cumulative value, ``equity[0] = 1.0``.
    """

    returns: np.ndarray
    equity: np.ndarray

    def __post_init__(self) -> None:
        returns = _coerce_float1d(self.returns, "returns")
        equity = _coerce_float1d(self.equity, "equity")
        if returns.shape[0] != equity.shape[0]:
            raise DataShapeError(
                f"returns and equity must have the same length; got "
                f"{returns.shape[0]} vs {equity.shape[0]}"
            )
        returns.setflags(write=False)
        equity.setflags(write=False)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "equity", equity)

    @property
    def n_bars(self) -> int:
        """Number of bars the curve covers."""
        return int(self.returns.shape[0])


def _normalized_equity(close: np.ndarray) -> np.ndarray:
    """``close / close[0]`` — a cumulative curve starting at ``1.0`` (empty-safe)."""
    if close.shape[0] == 0:
        return close.astype(np.float64, copy=True)
    return close / float(close[0])


def _returns_from_equity(equity: np.ndarray) -> np.ndarray:
    """Period returns from a cumulative curve: ``returns[0] = 0``, then ratios - 1."""
    n = equity.shape[0]
    returns = np.empty(n, dtype=np.float64)
    if n:
        returns[0] = 0.0
    if n > 1:
        returns[1:] = equity[1:] / equity[:-1] - 1.0
    return returns


def buy_and_hold_curve(market_data: MarketData) -> BenchmarkCurve:
    """Single-instrument buy-and-hold benchmark curve (§10 default).

    Returns the instrument's close-to-close returns and a normalized equity curve
    starting at ``1.0``, aligned to ``market_data``'s bar index.
    """
    if not isinstance(market_data, MarketData):
        raise DataShapeError("buy_and_hold_curve expects a MarketData")
    equity = _normalized_equity(market_data.close)
    returns = _returns_from_equity(equity)
    return BenchmarkCurve(returns=returns, equity=equity)


def portfolio_curve(
    data: Sequence[MarketData], weights: Sequence[float]
) -> BenchmarkCurve:
    """Combine multiple instruments' normalized curves with ``weights`` (pure).

    Each instrument contributes its own normalized equity curve (``close / close[0]``);
    the combined curve is the weighted sum, with ``weights`` normalized to sum to one
    (so the result also starts at ``1.0``). The return curve is derived from the
    combined equity curve. All instruments must have the same number of bars (they are
    assumed positionally aligned); the length of ``weights`` must match the number of
    instruments.
    """
    instruments = list(data)
    if not instruments:
        raise DataShapeError("portfolio benchmark requires at least one instrument")
    for md in instruments:
        if not isinstance(md, MarketData):
            raise DataShapeError("portfolio benchmark requires MarketData inputs")
    n = instruments[0].n_bars
    for md in instruments[1:]:
        if md.n_bars != n:
            raise DataShapeError(
                "portfolio benchmark instruments must have the same number of bars; "
                f"got {n} and {md.n_bars}"
            )

    w = np.asarray(tuple(weights), dtype=np.float64)
    if w.shape != (len(instruments),):
        raise ConfigError(
            f"weights length {w.shape[0]} does not match {len(instruments)} instruments"
        )
    if not np.isfinite(w).all() or (w < 0).any():
        raise ConfigError("weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ConfigError("weights must have a positive sum")
    w = w / total

    if n == 0:
        return BenchmarkCurve(
            returns=np.empty(0, dtype=np.float64), equity=np.empty(0, dtype=np.float64)
        )

    equities = np.stack([md.close / md.close[0] for md in instruments])
    equity = w @ equities
    returns = _returns_from_equity(equity)
    return BenchmarkCurve(returns=returns, equity=equity)


def build_benchmark(
    config: BenchmarkConfig, data: MarketData | Sequence[MarketData]
) -> BenchmarkCurve:
    """Build the benchmark curve for ``config`` over ``data`` (§10).

    ``buy_and_hold`` takes a single ``MarketData``; ``equal_weight`` and ``custom``
    take a sequence of ``MarketData`` (``custom`` uses ``config.weights``). This is the
    primary entry point; the per-kind pure functions are also public.
    """
    if not isinstance(config, BenchmarkConfig):
        raise ConfigError("build_benchmark expects a BenchmarkConfig")
    if config.kind == "buy_and_hold":
        if not isinstance(data, MarketData):
            raise ConfigError("kind='buy_and_hold' requires a single MarketData")
        return buy_and_hold_curve(data)

    if isinstance(data, (MarketData, str, bytes)) or not isinstance(data, Sequence):
        raise ConfigError(f"kind={config.kind!r} requires a sequence of MarketData")
    instruments = list(data)
    if config.kind == "equal_weight":
        n = len(instruments)
        return portfolio_curve(instruments, [1.0 / n] * n)

    if config.weights is None:  # validated at construction; defensive
        raise ConfigError("kind='custom' requires weights")
    return portfolio_curve(instruments, config.weights)

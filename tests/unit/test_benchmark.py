"""Tests for the benchmark curve builder (§10)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.benchmark import (
    BENCHMARK_KINDS,
    BenchmarkConfig,
    BenchmarkCurve,
    build_benchmark,
    buy_and_hold_curve,
    portfolio_curve,
)
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError


def _bars(closes: list[float]) -> MarketData:
    """Event bars with the given closes (monotonic OHLC invariants preserved)."""
    close = np.asarray(closes, dtype=float)
    n = close.shape[0]
    return MarketData(
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.ones(n),
        index=pd.RangeIndex(n),
        bar_type="volume",
    )


# ---------------------------------------------------------------------------
# BenchmarkConfig validation (fail-fast).
# ---------------------------------------------------------------------------


def test_benchmark_kinds_constant():
    assert BENCHMARK_KINDS == ("buy_and_hold", "equal_weight", "custom")


def test_default_config_is_buy_and_hold_without_weights():
    cfg = BenchmarkConfig()
    assert cfg.kind == "buy_and_hold"
    assert cfg.weights is None


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        BenchmarkConfig().kind = "custom"  # type: ignore[misc]


def test_unknown_kind_raises():
    with pytest.raises(ConfigError):
        BenchmarkConfig(kind="momentum")  # type: ignore[arg-type]


def test_custom_requires_weights():
    with pytest.raises(ConfigError):
        BenchmarkConfig(kind="custom")


def test_custom_weights_stored_as_tuple():
    cfg = BenchmarkConfig(kind="custom", weights=[0.25, 0.75])
    assert cfg.weights == (0.25, 0.75)
    assert isinstance(cfg.weights, tuple)


def test_weights_forbidden_for_buy_and_hold_and_equal_weight():
    with pytest.raises(ConfigError):
        BenchmarkConfig(kind="buy_and_hold", weights=(0.5, 0.5))
    with pytest.raises(ConfigError):
        BenchmarkConfig(kind="equal_weight", weights=(0.5, 0.5))


@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [-0.1, 1.1], [0.5, float("nan")]])
def test_invalid_custom_weights_raise(bad):
    with pytest.raises(ConfigError):
        BenchmarkConfig(kind="custom", weights=bad)


# ---------------------------------------------------------------------------
# buy-and-hold curve — alignment to the bar index, starting at 1.0.
# ---------------------------------------------------------------------------


def test_buy_and_hold_returns_and_equity():
    curve = buy_and_hold_curve(_bars([100.0, 110.0, 121.0, 108.9]))
    assert curve.n_bars == 4
    np.testing.assert_allclose(curve.returns, [0.0, 0.1, 0.1, -0.1])
    np.testing.assert_allclose(curve.equity, [1.0, 1.1, 1.21, 1.089])


def test_buy_and_hold_starts_at_one():
    curve = buy_and_hold_curve(_bars([42.0, 40.0, 45.0]))
    assert curve.equity[0] == 1.0
    assert curve.returns[0] == 0.0


def test_buy_and_hold_single_bar():
    curve = buy_and_hold_curve(_bars([100.0]))
    assert curve.n_bars == 1
    np.testing.assert_allclose(curve.equity, [1.0])
    np.testing.assert_allclose(curve.returns, [0.0])


def test_buy_and_hold_empty():
    curve = buy_and_hold_curve(_bars([]))
    assert curve.n_bars == 0
    assert curve.equity.size == 0 and curve.returns.size == 0


def test_buy_and_hold_requires_market_data():
    with pytest.raises(DataShapeError):
        buy_and_hold_curve(pd.DataFrame({"close": [1.0, 2.0]}))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Portfolio curves — equal-weight average, custom weighted sum.
# ---------------------------------------------------------------------------


def test_equal_weight_averages_normalized_curves():
    a = _bars([100.0, 110.0, 121.0])  # equity [1, 1.1, 1.21]
    b = _bars([50.0, 60.0, 55.0])  # equity [1, 1.2, 1.1]
    curve = portfolio_curve([a, b], [0.5, 0.5])
    np.testing.assert_allclose(curve.equity, [1.0, 1.15, 1.155])
    assert curve.equity[0] == 1.0
    assert curve.returns[0] == 0.0


def test_custom_weights_weighted_sum():
    a = _bars([100.0, 110.0, 121.0])
    b = _bars([50.0, 60.0, 55.0])
    curve = portfolio_curve([a, b], [0.25, 0.75])
    np.testing.assert_allclose(curve.equity, [1.0, 1.175, 1.1275])


def test_custom_weights_are_normalized():
    a = _bars([100.0, 110.0, 121.0])
    b = _bars([50.0, 60.0, 55.0])
    np.testing.assert_allclose(
        portfolio_curve([a, b], [1.0, 3.0]).equity,
        portfolio_curve([a, b], [0.25, 0.75]).equity,
    )


def test_portfolio_returns_derive_from_combined_equity():
    a = _bars([100.0, 110.0, 121.0])
    b = _bars([50.0, 60.0, 55.0])
    curve = portfolio_curve([a, b], [0.5, 0.5])
    expected_returns = [0.0, 0.15, 1.155 / 1.15 - 1.0]
    np.testing.assert_allclose(curve.returns, expected_returns)


def test_portfolio_requires_non_empty_data():
    with pytest.raises(DataShapeError):
        portfolio_curve([], [])  # type: ignore[list-item]


def test_portfolio_length_mismatch_raises():
    with pytest.raises(DataShapeError):
        portfolio_curve([_bars([1.0, 2.0, 3.0]), _bars([1.0, 2.0])], [0.5, 0.5])


def test_portfolio_weights_length_mismatch_raises():
    with pytest.raises(ConfigError):
        portfolio_curve([_bars([1.0, 2.0]), _bars([1.0, 2.0])], [0.5])


# ---------------------------------------------------------------------------
# build_benchmark dispatcher.
# ---------------------------------------------------------------------------


def test_build_benchmark_buy_and_hold():
    cfg = BenchmarkConfig(kind="buy_and_hold")
    curve = build_benchmark(cfg, _bars([100.0, 110.0]))
    np.testing.assert_allclose(curve.equity, [1.0, 1.1])


def test_build_benchmark_equal_weight():
    cfg = BenchmarkConfig(kind="equal_weight")
    curve = build_benchmark(
        cfg, [_bars([100.0, 110.0]), _bars([50.0, 60.0])]
    )
    np.testing.assert_allclose(curve.equity, [1.0, 1.15])


def test_build_benchmark_equal_weight_empty_data_raises_clean_error():
    # An empty portfolio must surface as a §15 DataShapeError, not a raw
    # ZeroDivisionError from the 1/n equal-weight computation.
    with pytest.raises(DataShapeError):
        build_benchmark(BenchmarkConfig(kind="equal_weight"), [])


def test_build_benchmark_custom():
    cfg = BenchmarkConfig(kind="custom", weights=[0.25, 0.75])
    curve = build_benchmark(
        cfg, [_bars([100.0, 110.0]), _bars([50.0, 60.0])]
    )
    np.testing.assert_allclose(curve.equity, [1.0, 0.25 * 1.1 + 0.75 * 1.2])


def test_build_benchmark_buy_and_hold_rejects_sequence():
    with pytest.raises(ConfigError):
        build_benchmark(
            BenchmarkConfig(kind="buy_and_hold"),
            [_bars([100.0, 110.0]), _bars([50.0, 60.0])],
        )


def test_build_benchmark_portfolio_rejects_single_market_data():
    with pytest.raises(ConfigError):
        build_benchmark(BenchmarkConfig(kind="equal_weight"), _bars([100.0, 110.0]))


def test_build_benchmark_rejects_non_config():
    with pytest.raises(ConfigError):
        build_benchmark("buy_and_hold", _bars([100.0, 110.0]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability — the curve container copies and freezes its arrays.
# ---------------------------------------------------------------------------


def test_curve_is_frozen():
    curve = buy_and_hold_curve(_bars([100.0, 110.0]))
    with pytest.raises(FrozenInstanceError):
        curve.equity = np.array([1.0])  # type: ignore[misc]


def test_curve_arrays_are_read_only():
    curve = buy_and_hold_curve(_bars([100.0, 110.0]))
    with pytest.raises(ValueError):
        curve.equity[0] = 5.0


def test_curve_constructor_copies_callers_arrays():
    returns = np.array([0.0, 0.1])
    equity = np.array([1.0, 1.1])
    curve = BenchmarkCurve(returns=returns, equity=equity)
    returns[1] = 99.0
    equity[1] = 99.0
    assert curve.returns[1] == 0.1
    assert curve.equity[1] == 1.1


def test_curve_constructor_does_not_freeze_callers_arrays():
    returns = np.array([0.0, 0.1])
    equity = np.array([1.0, 1.1])
    BenchmarkCurve(returns=returns, equity=equity)
    returns[1] = 0.2  # caller's buffer stays writable
    assert returns[1] == 0.2


def test_buy_and_hold_curve_does_not_alias_source_data():
    bars = _bars([100.0, 110.0])
    curve = buy_and_hold_curve(bars)
    assert not np.shares_memory(curve.equity, bars.close)
    assert not np.shares_memory(curve.returns, bars.close)


def test_curve_rejects_length_mismatch():
    with pytest.raises(DataShapeError):
        BenchmarkCurve(returns=np.array([0.0]), equity=np.array([1.0, 1.1]))

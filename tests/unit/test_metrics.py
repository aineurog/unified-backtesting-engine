"""Unit tests for the metrics layer (§4.9, §10)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from ube.core.benchmark import BenchmarkCurve
from ube.core.ledger import EquityCurve, Trade
from ube.core.metrics import Metrics, compute_metrics

DAY = 86_400_000_000_000


def _curve(values, freq="D", start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return EquityCurve(index=idx, equity=np.asarray(values, dtype=np.float64))


def _bench(values):
    eq = np.asarray(values, dtype=np.float64)
    ret = np.zeros(len(eq))
    if len(eq) > 1:
        ret[1:] = eq[1:] / eq[:-1] - 1.0
    return BenchmarkCurve(returns=ret, equity=eq)


def test_total_return_and_max_drawdown():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    m = compute_metrics(ec)
    assert m.total_return == pytest.approx(0.2)
    # peak 110 -> trough 99 = -10%
    assert m.max_drawdown == pytest.approx(-0.1)


def test_monotonic_curve_has_zero_drawdown():
    ec = _curve([100.0, 105.0, 110.0, 120.0])
    assert compute_metrics(ec).max_drawdown == pytest.approx(0.0)


def test_sharpe_and_vol_are_finite_for_multi_bar():
    # Daily bars over ~2 years of a gently drifting series with noise.
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    eq = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
    ec = EquityCurve(index=idx, equity=eq)
    m = compute_metrics(ec)
    assert math.isfinite(m.sharpe)
    assert m.volatility > 0.0
    # 500 daily bars span > 1 year -> confident.
    assert m.low_confidence is False
    assert m.periods_per_year == 252


def test_sub_year_sample_flagged_low_confidence():
    ec = _curve([100.0, 110.0, 99.0, 120.0])  # 3 days
    m = compute_metrics(ec)
    assert m.low_confidence is True
    assert any("< 1 year" in note for note in m.notes)


def test_benchmark_excess_and_information_ratio_present():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    bench = _bench([1.0, 1.1, 1.1, 1.2])  # +20% total, matches strategy
    m = compute_metrics(ec, benchmark=bench)
    assert m.excess_return is not None
    assert m.information_ratio is not None


def test_identical_benchmark_gives_zero_excess_and_nan_ir():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    # Benchmark normalized to exactly the strategy curve => equal total return, zero active.
    bench = _bench([100.0, 110.0, 99.0, 120.0])
    m = compute_metrics(ec, benchmark=bench)
    assert m.excess_return == pytest.approx(0.0)
    assert m.information_ratio != m.information_ratio  # NaN


def test_no_benchmark_skips_comparison():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    m = compute_metrics(ec, benchmark=None)
    assert m.excess_return is None
    assert m.information_ratio is None
    assert any("no benchmark" in note for note in m.notes)


def test_avg_trade_duration_in_days():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    trades = [
        Trade("A", 1, 1, 0, 1 * DAY, 1.0, 1.1, 1.0, 0.0, 0.0, 1.0),
        Trade("A", 1, 1, 2 * DAY, 3 * DAY, 1.0, 1.2, 1.0, 0.0, 0.0, 1.0),
    ]
    m = compute_metrics(ec, trades=trades)
    assert m.trade_count == 2
    # Mean duration: (1 day + 1 day) / 2 = 1 day.
    assert m.avg_trade_duration_days == pytest.approx(1.0)


def test_weekly_bars_use_native_frequency():
    ec = _curve([100.0, 110.0, 121.0, 130.0, 140.0], freq="7D")
    m = compute_metrics(ec)
    # 7-day bars -> ~52 periods/year, resampling up to daily is avoided.
    assert m.periods_per_year == 52
    assert m.bar_period_ns > DAY


def test_irregular_intraday_bars_resample_to_daily():
    # 4 bars per day across 3 days, irregular spacing within the day.
    base = pd.Timestamp("2020-01-01", tz="UTC")
    times = []
    for d in range(3):
        for h in (0, 8, 16, 20):
            times.append(base + pd.Timedelta(days=d, hours=h))
    idx = pd.DatetimeIndex(times)
    eq = np.linspace(100.0, 130.0, len(idx))
    ec = EquityCurve(index=idx, equity=eq)
    m = compute_metrics(ec)
    assert m.periods_per_year == 252  # sub-daily -> daily resample
    assert math.isfinite(m.sharpe)
    assert m.volatility > 0.0


def test_empty_equity_curve_is_safe():
    idx = pd.DatetimeIndex([], tz="UTC")
    ec = EquityCurve(index=idx, equity=np.asarray([], dtype=np.float64))
    m = compute_metrics(ec)
    assert m.total_return == 0.0
    assert m.trade_count == 0
    assert m.low_confidence is True


def test_metrics_is_frozen_and_serializable():
    ec = _curve([100.0, 110.0, 99.0, 120.0])
    m = compute_metrics(ec)
    assert isinstance(m, Metrics)
    # Frozen dataclass -> attribute assignment raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.total_return = 1.0  # type: ignore[misc]
    # Plain dict view round-trips the scalar fields.
    d = m.as_dict()
    assert set(d) >= {"total_return", "sharpe", "max_drawdown", "trade_count"}

"""Performance metrics layer (§4.9, §10).

The metrics layer is the subsystem that turns a :class:`~ube.core.result.BacktestResult`
into the standard performance report. It was the last ``None`` placeholder in the result
container — ``BacktestResult.metrics`` shipped as ``None`` with the comparison math
deferred (§10). This module builds that math.

Pipeline (per the §4.9 / §10 requirements):

1. **Infer the bar period.** Reuse :func:`~ube.core.data.derive_bar_period_ns` on the
   equity-curve grid (the same inference fix from the wiring pass, §4.9). This is the
   cadence the backtest actually ran at, independent of any engine label.
2. **Resample irregular-bar returns to a daily grid before annualizing.** Strategy and
   benchmark period returns are compounded into calendar-day buckets so that irregularly
   spaced bars (event/session/synthetic bars) don't bias the mean/variance. The benchmark
   is positionally aligned to the strategy curve (run.py builds it from the same warm-up /
   date-range-sliced bars), so it shares the strategy's time axis.
3. **Low-confidence flag.** A sample spanning fewer than one calendar year is flagged
   ``low_confidence`` — annualized statistics computed from a sub-year window are
   extrapolations, not measurements.
4. **Standard metric set.** Total return, annualized return (CAGR), Sharpe, volatility,
   max drawdown, trade count, average trade duration.
5. **Benchmark comparison.** Excess return (annualized strategy minus benchmark) and
   information ratio (active return mean / active return std, annualized). Both are
   ``None`` when no benchmark curve is present.

Annualization conventions (documented so they aren't silently magic numbers):

* **Daily grid (sub-daily and daily bars).** Resample to calendar days, then annualize
  with ``ANNUALIZATION_PERIODS_DAILY = 252`` (the standard trading-day Sharpe convention).
* **≥ daily bars (weekly, monthly, …).** Resampling *up* to daily would invent data, so we
  keep the native bar frequency and annualize with ``periods_per_year`` derived from the
  inferred bar period (``YEAR_NS / bar_period_ns``).
* **Returns.** ``annualized_return`` is CAGR (time-based, ``(1+tr)^(1/years)-1``); Sharpe
  and volatility are frequency-based (``mean/std * sqrt(periods_per_year)``). This matches
  conventional reporting: "annualized return" = CAGR, "annualized vol" = ``std*sqrt(ppy)``.
* **Risk-free rate.** ``0.0`` (no risk-free series is carried by the result); the Sharpe is
  therefore the excess-over-cash Sharpe with cash = 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ube.core.benchmark import BenchmarkCurve
from ube.core.data import derive_bar_period_ns
from ube.core.ledger import EquityCurve, Trade

__all__ = ["Metrics", "compute_metrics"]

#: Standard trading-day annualization factor for Sharpe / volatility / IR.
ANNUALIZATION_PERIODS_DAILY = 252

#: Nanoseconds in one calendar day.
DAY_NS = 86_400_000_000_000

#: Nanoseconds in one (average) calendar year.
YEAR_NS = 365.25 * DAY_NS

#: Single day in nanoseconds, reused for duration math.
_DAY_NS_F = float(DAY_NS)


@dataclass(frozen=True)
class Metrics:
    """The populated performance report attached to ``BacktestResult.metrics`` (§4.9, §10).

    All return/risk figures are fractions (``0.12`` = 12%), not percentages. Benchmark
    comparison fields are ``None`` when the run had no benchmark curve.

    Attributes:
        total_return: Cumulative return over the whole sample (``equity[-1]/equity[0] - 1``).
        annualized_return: CAGR over the sample span (``(1+total_return)^(1/years) - 1``).
        sharpe: Annualized Sharpe ratio (risk-free = 0); ``nan`` when volatility is 0.
        volatility: Annualized return standard deviation.
        max_drawdown: Worst peak-to-trough decline of the equity curve (``<= 0``).
        trade_count: Number of completed round-trip trades.
        avg_trade_duration_days: Mean trade holding period, in calendar days.
        excess_return: Annualized strategy return minus annualized benchmark return; ``None``
            without a benchmark.
        information_ratio: Mean active return / std active return, annualized; ``None``
            without a benchmark, ``nan`` when tracking-error volatility is 0.
        bar_period_ns: Inferred median inter-bar spacing (nanoseconds).
        periods_per_year: Annualization factor used for vol / Sharpe / IR.
        sample_days: Sample span in calendar days (first to last bar).
        low_confidence: ``True`` when the sample spans less than one year.
        notes: Human-readable warnings (e.g. low-confidence, missing benchmark).
    """

    total_return: float
    annualized_return: float
    sharpe: float
    volatility: float
    max_drawdown: float
    trade_count: int
    avg_trade_duration_days: float
    excess_return: float | None
    information_ratio: float | None
    bar_period_ns: int
    periods_per_year: int
    sample_days: float
    low_confidence: bool
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """A plain-dict view (for logging / experiment records)."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe": self.sharpe,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "avg_trade_duration_days": self.avg_trade_duration_days,
            "excess_return": self.excess_return,
            "information_ratio": self.information_ratio,
            "bar_period_ns": self.bar_period_ns,
            "periods_per_year": self.periods_per_year,
            "sample_days": self.sample_days,
            "low_confidence": self.low_confidence,
        }


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _period_returns(equity: np.ndarray) -> np.ndarray:
    """Period returns from a cumulative equity curve: ``r[0] = 0``, then ratios - 1."""
    n = equity.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n > 1:
        out[1:] = equity[1:] / equity[:-1] - 1.0
    return out


def _resample_to_daily(returns: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Compound irregular period returns into calendar-day returns (§4.9).

    Bars sharing a calendar day are multiplied together; days with no bar contribute a
    zero return (no observation = no change). The result is a daily-frequency series.
    """
    series = pd.Series(returns, index=index)
    daily = (1.0 + series).resample("D").prod() - 1.0
    return daily.fillna(0.0)


def _max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough decline (``<= 0``); ``0.0`` for empty/flat curves."""
    if equity.shape[0] == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    # Guard against a zero running max (degenerate equity) to avoid div-by-zero.
    safe_max = np.where(running_max == 0.0, 1.0, running_max)
    drawdown = equity / safe_max - 1.0
    return float(drawdown.min())


def _annualized_return(total_return: float, years: float) -> float:
    """CAGR: ``(1+tr)^(1/years) - 1``.

    Falls back to the raw ``total_return`` when annualization is undefined (non-finite or
    zero span) or numerically explosive — a sub-day sample extrapolated to a year overflows,
    and the ``low_confidence`` flag already warns that such annualized figures are not
    measurements. Returning the raw total return keeps the field finite and honest.
    """
    if not math.isfinite(years) or years <= 0 or not math.isfinite(total_return):
        return float(total_return)
    if years < 1.0 / 365.25:
        return float(total_return)
    try:
        return float((1.0 + total_return) ** (1.0 / years) - 1.0)
    except (OverflowError, ValueError):
        return float(total_return)


def _sharpe_and_vol(returns: pd.Series | np.ndarray, periods_per_year: int) -> tuple[float, float]:
    """Annualized Sharpe (rf = 0) and volatility from a return series.

    Sharpe is ``nan`` when volatility is zero (undefined, not 0); volatility is ``0.0`` in
    that case.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.shape[0] < 2:
        return (float("nan"), 0.0)
    std = float(np.std(arr, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return (float("nan"), 0.0)
    mean = float(np.mean(arr))
    ppy = float(periods_per_year)
    vol = std * math.sqrt(ppy)
    sharpe = (mean / std) * math.sqrt(ppy)
    return (sharpe, vol)


def _information_ratio(active: pd.Series | np.ndarray, periods_per_year: int) -> float:
    """Annualized information ratio: mean(active) / std(active) * sqrt(ppy).

    ``nan`` when the active-return volatility is zero.
    """
    arr = np.asarray(active, dtype=np.float64)
    if arr.shape[0] < 2:
        return float("nan")
    std = float(np.std(arr, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return float("nan")
    mean = float(np.mean(arr))
    return float((mean / std) * math.sqrt(periods_per_year))


def _avg_trade_duration_days(trades: Sequence[Trade]) -> float:
    """Mean holding period across completed trades, in calendar days."""
    if not trades:
        return 0.0
    total_ns = sum(float(t.exit_timestamp - t.entry_timestamp) for t in trades)
    return (total_ns / len(trades)) / _DAY_NS_F


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def compute_metrics(
    equity_curve: EquityCurve,
    benchmark: BenchmarkCurve | None = None,
    trades: Sequence[Trade] = (),
) -> Metrics:
    """Compute the full performance report for a result (§4.9, §10).

    Args:
        equity_curve: The strategy equity curve (carries the tz-aware UTC ``DatetimeIndex``
            used for bar-period inference and daily resampling).
        benchmark: Optional benchmark curve, positionally aligned to ``equity_curve`` (run.py
            builds it from the same warm-up / date-range-sliced bars). When ``None``, the
            benchmark-comparison fields are ``None``.
        trades: Completed round-trip trades (for ``trade_count`` / ``avg_trade_duration_days``).

    Returns:
        A frozen :class:`Metrics` populated with the standard set plus the benchmark
        comparison (or ``None`` comparison fields when there is no benchmark).
    """
    notes: list[str] = []
    equity = equity_curve.equity
    index = cast(pd.DatetimeIndex, equity_curve.index)

    n = equity.shape[0]
    if n == 0:
        notes.append("empty equity curve: metrics are zero/undefined")
        return Metrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe=float("nan"),
            volatility=0.0,
            max_drawdown=0.0,
            trade_count=len(trades),
            avg_trade_duration_days=_avg_trade_duration_days(trades),
            excess_return=None,
            information_ratio=None,
            bar_period_ns=0,
            periods_per_year=ANNUALIZATION_PERIODS_DAILY,
            sample_days=0.0,
            low_confidence=True,
            notes=tuple(notes),
        )

    # 1. Infer bar period from the equity-curve grid.
    bar_period_ns = derive_bar_period_ns(index) if n >= 2 else 0

    # Sample span (calendar days) and low-confidence flag.
    first_ts = pd.Timestamp(index[0])
    last_ts = pd.Timestamp(index[-1])
    span_ns = float(last_ts.value - first_ts.value)
    sample_days = span_ns / _DAY_NS_F
    years = sample_days / 365.25
    low_confidence = years < 1.0
    if low_confidence:
        notes.append(
            f"sample spans {sample_days:.1f} days (< 1 year): annualized metrics are "
            "extrapolations, not measurements"
        )

    # 2. Resample to a daily grid for sub-daily/daily bars; keep native frequency for
    #    >= daily bars (resampling up would invent data).
    strat_period = _period_returns(equity)
    bench_period = _period_returns(benchmark.equity) if benchmark is not None else None

    if bar_period_ns > 0 and bar_period_ns <= DAY_NS:
        # Sub-daily or daily: resample to calendar days, annualize with 252.
        strat_series = _resample_to_daily(strat_period, index)
        periods_per_year = ANNUALIZATION_PERIODS_DAILY
        bench_series = _resample_to_daily(bench_period, index) if bench_period is not None else None
    else:
        # >= daily (weekly/monthly/…): use native bar frequency, derive ppy from period.
        strat_series = pd.Series(strat_period[1:], index=index[1:])
        if bar_period_ns > 0:
            periods_per_year = max(1, int(round(YEAR_NS / bar_period_ns)))
        else:
            periods_per_year = ANNUALIZATION_PERIODS_DAILY
        bench_series = (
            pd.Series(bench_period[1:], index=index[1:]) if bench_period is not None else None
        )

    # 4./5. Standard set.
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] != 0 else 0.0
    annualized_return = _annualized_return(total_return, years)
    sharpe, volatility = _sharpe_and_vol(strat_series, periods_per_year)
    max_drawdown = _max_drawdown(equity)

    # Benchmark comparison.
    excess_return: float | None = None
    information_ratio: float | None = None
    if benchmark is not None:
        if benchmark.equity.shape[0] >= 2 and benchmark.equity[0] != 0:
            bench_total = float(benchmark.equity[-1] / benchmark.equity[0] - 1.0)
        else:
            bench_total = 0.0
        bench_ann = _annualized_return(bench_total, years)
        excess_return = annualized_return - bench_ann
        if bench_series is not None:
            aligned = pd.concat([strat_series, bench_series], axis=1).dropna()
            if aligned.shape[0] >= 2:
                active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                information_ratio = _information_ratio(active, periods_per_year)
            else:
                information_ratio = float("nan")
                notes.append("too few overlapping strategy/benchmark days for information ratio")
    else:
        notes.append("no benchmark curve: excess_return and information_ratio skipped")

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        volatility=volatility,
        max_drawdown=max_drawdown,
        trade_count=len(trades),
        avg_trade_duration_days=_avg_trade_duration_days(trades),
        excess_return=excess_return,
        information_ratio=information_ratio,
        bar_period_ns=bar_period_ns,
        periods_per_year=periods_per_year,
        sample_days=sample_days,
        low_confidence=low_confidence,
        notes=tuple(notes),
    )

"""Strategy statistics for UBE results — the StratStat integration (§10).

``ube`` computes no metrics itself (§10): this module is the bridge that feeds the
canonical §4.6 views — equity, closed trades, benchmark and position history — to
`stratstat` and packages what comes back as a frozen :class:`StatsResult`.

Four tiers are fed whenever their inputs exist (§10 "feed all four"):

=========  =========================================================================
tier       input
=========  =========================================================================
returns    equity-curve per-bar returns (optionally resampled via ``resample_to``)
trades     closed round-trip trades: fractional PnL, side, duration in bar periods
benchmark  benchmark returns, aligned **positionally** with the strategy curve
exposure   per-instrument position notional as a fraction of equity
=========  =========================================================================

Safety rules — a run never fails because of statistics:

* a missing ``stratstat`` install, a stratstat exception, or an unusable input is
  recorded in :attr:`StatsResult.errors`; the caller always gets a result;
* :attr:`~ube.core.ledger.EquityCurve.returns` marks insolvent periods ``NaN``, and
  ``return_policy`` decides how they are handled (``"truncate"`` by default — never
  compound through an insolvency);
* annualisation is inferred from the equity grid with
  :func:`~ube.core.data.derive_bar_period_ns`, never hard-coded (§4.9);
* the exposure tier only runs when the supplied bars actually span the equity grid, so
  a live ``step()`` window can never masquerade as full-history exposure.

Typical use::

    result = ube.run(data, signals, config)
    result.stats["sharpe_ratio"]     # scalar metric
    result.stats.table()             # every metric as a DataFrame
    ube.stats.save_stats(db, run_id, result.stats, trade_count=len(result.trades))
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ube.core.benchmark import BenchmarkCurve
from ube.core.data import MarketData, derive_bar_period_ns
from ube.core.ledger import EquityCurve, EventLedger, Positions, Trade, positions

if TYPE_CHECKING:
    from ube.core.result import BacktestResult

__all__ = [
    "SECONDS_PER_YEAR",
    "StatsResult",
    "compute",
    "from_arrays",
    "from_result",
    "is_enabled",
    "load_stats",
    "save_stats",
    "set_enabled",
]

#: Julian seconds per year (365.25 days) — the same constant
#: ``ube.testing.synthetic`` uses, so annualisation agrees across §4.9 and the
#: parity harness.
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

#: Accepted values of the ``return_policy`` argument (§10 safety rule).
RETURN_POLICIES = ("truncate", "zero", "drop")

#: Columns of the persisted-latest snapshot returned by :func:`load_stats`.
LOAD_COLUMNS = [
    "metric",
    "value",
    "value_json",
    "category",
    "ref",
    "seq",
    "as_of_ns",
    "trade_count",
    "n_equity_points",
    "periods_per_year",
    "tiers",
    "errors_json",
]

_STATS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stats (
    run_id     TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    metric     TEXT    NOT NULL,
    value      REAL,
    value_json TEXT,
    category   TEXT,
    ref        TEXT,
    PRIMARY KEY (run_id, seq, metric)
)
"""

_STATS_META_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stats_meta (
    run_id           TEXT    NOT NULL,
    seq              INTEGER NOT NULL,
    as_of_ns         INTEGER NOT NULL,
    trade_count      INTEGER NOT NULL,
    n_equity_points  INTEGER NOT NULL DEFAULT 0,
    periods_per_year REAL,
    tiers            TEXT,
    errors_json      TEXT,
    PRIMARY KEY (run_id, seq)
)
"""

_ENABLED = True


def set_enabled(flag: bool) -> None:
    """Enable or disable the eager statistics computed by ``from_ledger``.

    Statistics are enabled by default; disabling them leaves
    ``BacktestResult.stats`` as ``None`` (nothing else about a run changes).
    """
    global _ENABLED
    _ENABLED = bool(flag)


def is_enabled() -> bool:
    """Whether eager statistics are enabled (default ``True``)."""
    return _ENABLED


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatsResult:
    """The frozen statistics of one run (§10).

    Attributes:
        values: Scalar metrics, ``metric -> float`` (including ``inf``/``NaN`` when a
            metric is undefined for the run, e.g. ``calmar_ratio`` on a flat curve).
        objects: Non-scalar metrics (distributions, fits, …), already converted to
            JSON-serialisable lists/dicts.
        categories: ``metric -> "category[.subcategory]"`` as stratstat tags them.
        periods_per_year: The annualisation factor actually used, or ``None`` when the
            grid carried no time information.
        tiers: The tiers that were computed (``returns``/``trades``/``benchmark``/
            ``exposure``).
        meta: Provenance — stratstat's own meta, the citation refs, the return policy,
            how many non-finite returns were handled, and any warnings raised.
        errors: ``input -> reason`` for everything that was skipped. Never raises.
        computed_at_ns: UTC timestamp of the computation, int64 nanoseconds.
    """

    values: dict[str, float]
    objects: dict[str, Any]
    categories: dict[str, str]
    periods_per_year: float | None
    tiers: tuple[str, ...]
    meta: dict[str, Any]
    errors: dict[str, str]
    computed_at_ns: int

    def __post_init__(self) -> None:
        # Copy every mapping so the frozen result never aliases a caller-supplied dict.
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "objects", dict(self.objects))
        object.__setattr__(self, "categories", dict(self.categories))
        object.__setattr__(self, "meta", dict(self.meta))
        object.__setattr__(self, "errors", dict(self.errors))
        object.__setattr__(self, "tiers", tuple(self.tiers))

    # -- lookup ---------------------------------------------------------

    def __getitem__(self, metric: str) -> float:
        """Return a scalar metric, raising ``KeyError`` when stratstat did not emit it."""
        try:
            return self.values[metric]
        except KeyError:
            raise KeyError(
                f"{metric!r} is not in this result "
                f"({self.n_metrics} metrics, tiers={list(self.tiers)}); "
                f"metrics: {sorted(self.values)}"
            ) from None

    def get(self, metric: str, default: float | None = None) -> float | None:
        """Return a scalar metric, or ``default`` when it is absent."""
        return self.values.get(metric, default)

    @property
    def n_metrics(self) -> int:
        """Number of metrics available (scalar + non-scalar)."""
        return len(self.values) + len(self.objects)

    def category(self, name: str) -> dict[str, float]:
        """Scalar metrics whose stratstat category is ``name`` or a subcategory of it."""
        prefix = f"{name}."
        return {
            metric: value
            for metric, value in self.values.items()
            if self.categories.get(metric, "") == name
            or self.categories.get(metric, "").startswith(prefix)
        }

    # -- views ----------------------------------------------------------

    def table(self) -> pd.DataFrame:
        """Every metric as a ``[metric, value, category, kind]`` DataFrame.

        ``kind`` is ``"scalar"`` for floats and ``"object"`` for distributions/fits.
        """
        rows = [
            {"metric": m, "value": v, "category": self.categories.get(m, ""), "kind": "scalar"}
            for m, v in self.values.items()
        ]
        rows += [
            {"metric": m, "value": v, "category": self.categories.get(m, ""), "kind": "object"}
            for m, v in self.objects.items()
        ]
        frame = pd.DataFrame(rows, columns=["metric", "value", "category", "kind"])
        return frame.sort_values(["category", "metric"], kind="stable").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """All metrics as one ``metric -> value`` mapping (JSON-serialisable)."""
        out: dict[str, Any] = dict(self.values)
        out.update(self.objects)
        return out

    def to_records(self) -> list[dict[str, Any]]:
        """Metric rows shaped like the ``stats`` table, ready for :func:`save_stats`.

        Finite scalars go to ``value``; everything else (``inf``/``NaN`` scalars and
        non-scalar metrics) goes to ``value_json``, which round-trips through Python's
        ``json`` module.
        """
        refs: dict[str, str] = dict(self.meta.get("refs", {}))
        rows: list[dict[str, Any]] = []
        for metric, value in self.values.items():
            finite = math.isfinite(value)
            rows.append(
                {
                    "metric": metric,
                    "value": float(value) if finite else None,
                    "value_json": None if finite else json.dumps(value),
                    "category": self.categories.get(metric, ""),
                    "ref": refs.get(metric),
                }
            )
        for metric, obj in self.objects.items():
            rows.append(
                {
                    "metric": metric,
                    "value": None,
                    "value_json": json.dumps(obj),
                    "category": self.categories.get(metric, ""),
                    "ref": refs.get(metric),
                }
            )
        rows.sort(key=lambda row: (row["category"], row["metric"]))
        return rows

    # -- benchmark tier -------------------------------------------------

    def with_benchmark(
        self,
        benchmark: BenchmarkCurve,
        *,
        returns: np.ndarray,
        index: pd.Index,
        rf: float = 0.0,
    ) -> StatsResult:
        """Return a copy of this result with the benchmark tier recomputed and merged.

        ``ube.run`` attaches the benchmark *after* the adapter has built the result, so
        the benchmark tier is computed here from the result's own equity curve and merged
        into the tiers that already ran. Existing metrics win on any name collision
        (the same first-wins rule stratstat's ``deduplicate`` uses). When nothing can be
        merged the input result is returned unchanged.
        """
        policy = str(self.meta.get("return_policy", "truncate"))
        errors = dict(self.errors)
        # Copy the nested provenance dicts too: this method adds a "refs" entry, and a
        # shallow copy would mutate the original result's mapping.
        meta = {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in self.meta.items()
        }
        values = dict(self.values)
        objects = dict(self.objects)
        categories = dict(self.categories)
        tiers = list(self.tiers)

        rs, _ = _clean_returns(
            pd.Series(np.asarray(returns, dtype=np.float64), index=index), policy
        )
        rs, benchmark_returns, note = _align_benchmark(rs, benchmark)
        if note:
            meta["benchmark_note"] = note

        if len(rs) < 2:
            errors["benchmark"] = f"only {len(rs)} aligned bar(s); benchmark tier skipped"
            return StatsResult(
                values=values,
                objects=objects,
                categories=categories,
                periods_per_year=self.periods_per_year,
                tiers=tuple(tiers),
                meta=meta,
                errors=errors,
                computed_at_ns=_now_ns(),
            )

        ppy = self.periods_per_year
        ppy_int = int(round(ppy)) if ppy is not None and math.isfinite(ppy) and ppy > 0 else None
        try:
            ss = _stratstat()
        except ImportError as exc:
            errors["benchmark"] = str(exc)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    ms = ss.compute_benchmark(
                        rs, benchmark_returns, periods_per_year=ppy_int, rf=rf
                    )
                except Exception as exc:  # noqa: BLE001 — degrade, never fail the run
                    errors["benchmark"] = f"{type(exc).__name__}: {exc}"
                    ms = None
            _record_warnings(meta, caught)
            if ms is not None:
                v, o, c, refs = _unpack(ms)
                for metric, value in v.items():
                    values.setdefault(metric, value)
                for metric, obj in o.items():
                    objects.setdefault(metric, obj)
                for metric, cat in c.items():
                    categories.setdefault(metric, cat)
                existing_refs = meta.setdefault("refs", {})
                for metric, ref in refs.items():
                    existing_refs.setdefault(metric, ref)
                meta["stratstat"] = {**meta.get("stratstat", {}), **dict(ms.meta)}
                input_meta = meta.get("input")
                if isinstance(input_meta, dict):
                    input_meta["benchmark"] = int(len(benchmark_returns))
                if "benchmark" not in tiers:
                    tiers.append("benchmark")

        return StatsResult(
            values=values,
            objects=objects,
            categories=categories,
            periods_per_year=ppy,
            tiers=tuple(tiers),
            meta=meta,
            errors=errors,
            computed_at_ns=_now_ns(),
        )


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def compute(
    returns: np.ndarray | pd.Series | None = None,
    *,
    trades: Sequence[Trade] | None = None,
    benchmark: BenchmarkCurve | np.ndarray | pd.Series | None = None,
    exposure: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
    return_policy: str = "truncate",
    period_ns: int | None = None,
    rf: float = 0.0,
    computed_at_ns: int | None = None,
    extra_meta: Mapping[str, Any] | None = None,
    extra_errors: Mapping[str, str] | None = None,
) -> StatsResult:
    """Feed tier inputs to stratstat and wrap what comes back (§10).

    This is the low-level entry point: the inputs are already assembled, so only the
    §10 safety rules are applied (return cleaning, annualisation inference, positional
    benchmark alignment, exposure grid alignment) before the call.

    Args:
        returns: Per-bar returns. A ``pd.Series`` with a ``DatetimeIndex`` lets the bar
            period be inferred; a bare array requires ``periods_per_year`` for the
            annualised metrics.
        trades: Closed round-trip :class:`~ube.core.ledger.Trade` objects. PnL is fed as
            a fraction of entry notional whenever every trade has one (so Kelly /
            geometric-mean metrics can run), falling back to raw currency otherwise.
        benchmark: The benchmark curve (or its returns), aligned positionally.
        exposure: Per-instrument position **weights** (notional / equity), reindexed onto
            the returns grid.
        periods_per_year: Annualisation override; inferred from ``returns``' index when
            omitted.
        return_policy: How the ``NaN`` marks written by
            :attr:`~ube.core.ledger.EquityCurve.returns` are handled — ``"truncate"``
            (default, drop everything from the first non-finite value), ``"zero"``
            (replace them) or ``"drop"`` (remove only those points).
        period_ns: Bar period override for trade durations, in nanoseconds. Derived from
            ``periods_per_year`` when omitted.
        rf: Risk-free rate passed through to stratstat.
        computed_at_ns: Override for the computation timestamp (tests).
        extra_meta: Extra entries merged into :attr:`StatsResult.meta`.
        extra_errors: Extra entries merged into :attr:`StatsResult.errors`.

    Returns:
        A frozen :class:`StatsResult`; failures are recorded in ``errors``, never raised.
    """
    if return_policy not in RETURN_POLICIES:
        raise ValueError(f"return_policy must be one of {RETURN_POLICIES}; got {return_policy!r}")

    errors: dict[str, str] = dict(extra_errors or {})
    meta: dict[str, Any] = {"return_policy": return_policy}
    if extra_meta:
        meta.update(dict(extra_meta))

    # -- returns --------------------------------------------------------
    returns_input: pd.Series | None = None
    if returns is not None:
        rs = returns if isinstance(returns, pd.Series) else pd.Series(returns)
        rs, handled = _clean_returns(rs, return_policy)
        meta["dropped_returns"] = int(handled)
        if len(rs) >= 2:
            returns_input = rs.astype(np.float64)
        else:
            errors["returns"] = (
                f"only {len(rs)} usable return(s) after {return_policy!r} cleaning"
            )

    # -- annualisation --------------------------------------------------
    ppy = float(periods_per_year) if periods_per_year else None
    if ppy is None and returns_input is not None:
        ppy = _infer_periods_per_year(returns_input.index, errors)
    meta["periods_per_year"] = ppy
    ppy_int = int(round(ppy)) if ppy is not None and math.isfinite(ppy) and ppy > 0 else None
    if period_ns is None and ppy_int is not None:
        period_ns = int(round(SECONDS_PER_YEAR * 1e9 / ppy_int))

    # -- trades ---------------------------------------------------------
    trades_input: dict[str, list[Any]] | None = None
    pnl_unit = "fraction"
    n_trades = len(trades) if trades else 0
    if n_trades:
        trades_input, pnl_unit = _trade_input(trades or (), period_ns)

    # -- benchmark ------------------------------------------------------
    benchmark_input: pd.Series | None = None
    if benchmark is not None:
        if returns_input is None:
            errors["benchmark"] = "the benchmark tier requires strategy returns"
        else:
            returns_input, benchmark_input, note = _align_benchmark(returns_input, benchmark)
            if note:
                meta["benchmark_note"] = note

    # -- exposure -------------------------------------------------------
    exposure_input = exposure
    if exposure_input is not None and returns_input is not None:
        exposure_input = (
            exposure_input.reindex(exposure_input.index.union(returns_input.index))
            .ffill()
            .reindex(returns_input.index)
            .fillna(0.0)
        )

    # -- call stratstat -------------------------------------------------
    kwargs: dict[str, Any] = {}
    tiers: list[str] = []
    if returns_input is not None:
        kwargs["returns"] = returns_input
        tiers.append("returns")
    if trades_input is not None:
        kwargs["trades"] = trades_input
        tiers.append("trades")
        if pnl_unit != "fraction":
            kwargs["pnl_unit"] = pnl_unit
    if benchmark_input is not None:
        kwargs["benchmark"] = benchmark_input
        tiers.append("benchmark")
    if exposure_input is not None:
        kwargs["exposure"] = exposure_input
        tiers.append("exposure")

    meta["input"] = {
        "returns": int(len(returns_input)) if returns_input is not None else 0,
        "trades": n_trades,
        "benchmark": int(len(benchmark_input)) if benchmark_input is not None else 0,
        "exposure": int(len(exposure_input)) if exposure_input is not None else 0,
    }

    values: dict[str, float] = {}
    objects: dict[str, Any] = {}
    categories: dict[str, str] = {}
    if kwargs:
        try:
            ss = _stratstat()
        except ImportError as exc:
            errors["compute"] = str(exc)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    ms = ss.compute_all(periods_per_year=ppy_int, rf=rf, **kwargs)
                except Exception as exc:  # noqa: BLE001 — degrade, never fail the run
                    errors["compute"] = f"{type(exc).__name__}: {exc}"
                    ms = None
                else:
                    values, objects, categories, refs = _unpack(ms)
                    meta["refs"] = refs
                    meta["stratstat"] = dict(ms.meta)
            _record_warnings(meta, caught)
        tiers = tiers if values or objects else []
    else:
        errors.setdefault("compute", "no usable tier input")

    return StatsResult(
        values=values,
        objects=objects,
        categories=categories,
        periods_per_year=ppy,
        tiers=tuple(tiers),
        meta=meta,
        errors=errors,
        computed_at_ns=_now_ns(computed_at_ns),
    )


def from_arrays(
    index: pd.Index,
    equity: np.ndarray,
    *,
    trades: Sequence[Trade],
    equity_returns: Sequence[float] | None = None,
    ledger: EventLedger | None = None,
    market_data: Mapping[str, MarketData] | None = None,
    instruments: Mapping[str, Any] | None = None,
    benchmark: BenchmarkCurve | None = None,
    periods_per_year: float | None = None,
    resample_to: str | None = None,
    return_policy: str = "truncate",
    rf: float = 0.0,
    computed_at_ns: int | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> StatsResult:
    """Build every tier from an equity grid, its trades, and (optionally) a ledger.

    The generic assembly point: the backtest path passes the cached
    :class:`~ube.core.result.BacktestResult` views, and the paper path passes the equity
    it persisted (bars are not stored, §9.5), so both produce the same statistics.

    Args:
        index: The equity grid (tz-aware UTC, normally bar timestamps).
        equity: Equity values on that grid, in the run's base currency.
        trades: The closed round-trip trades.
        equity_returns: Optional returns already calculated and persisted alongside
            ``equity``. Used directly when the curve is not resampled; for a resampled
            curve, returns are derived from the resampled equity points.
        ledger: Needed for the exposure tier (position history); ``None`` skips it.
        market_data: ``instrument_id -> MarketData`` bars for pricing the exposure tier.
            A caller that cannot supply full history (a live ``step()`` window) gets
            exposure over the bars it did supply, flagged in ``meta["exposure_window"]``.
        instruments: ``instrument_id -> Instrument`` for the contract multiplier.
        benchmark: Precomputed benchmark curve, if any.
        periods_per_year: Annualisation override (otherwise inferred from ``index``).
        resample_to: Optional pandas offset alias (e.g. ``"1D"``) applied to the equity
            curve *before* annualisation, so periods, durations and exposure agree.
        return_policy: See :func:`compute`.
        rf: Risk-free rate passed through to stratstat.
        computed_at_ns: Override for the computation timestamp (tests).
        extra_meta: Additional provenance merged into :attr:`StatsResult.meta` (the paper
            path records which equity source it used).

    Returns:
        A frozen :class:`StatsResult`.
    """
    errors: dict[str, str] = {}
    curve = EquityCurve(index=index, equity=equity)
    was_resampled = False
    if resample_to is not None:
        try:
            curve = curve.resample(resample_to)
            was_resampled = True
        except Exception as exc:  # noqa: BLE001 — degrade to the unresampled grid
            errors["resample"] = f"{resample_to!r}: {type(exc).__name__}: {exc}"

    returns_values = curve.returns
    if equity_returns is not None:
        persisted_returns = np.asarray(equity_returns, dtype=np.float64)
        if persisted_returns.ndim != 1 or len(persisted_returns) != len(index):
            errors["equity_returns"] = (
                f"expected a one-dimensional array of {len(index)} return(s) aligned "
                f"to the equity grid, got shape {persisted_returns.shape}"
            )
        elif not was_resampled:
            returns_values = persisted_returns

    exposure = None
    notes: dict[str, Any] = {}
    if ledger is not None and market_data and instruments:
        exposure = _exposure_frame(
            ledger, market_data, instruments, curve.index, curve.equity, errors, notes
        )
    if extra_meta:
        notes.update(dict(extra_meta))

    return compute(
        pd.Series(returns_values, index=curve.index),
        trades=trades,
        benchmark=benchmark,
        exposure=exposure,
        periods_per_year=periods_per_year,
        return_policy=return_policy,
        rf=rf,
        computed_at_ns=computed_at_ns,
        extra_errors=errors,
        extra_meta=notes or None,
    )


def from_result(
    result: BacktestResult,
    *,
    benchmark: BenchmarkCurve | None = None,
    market_data: Mapping[str, MarketData] | None = None,
    instruments: Mapping[str, Any] | None = None,
    periods_per_year: float | None = None,
    resample_to: str | None = None,
    return_policy: str = "truncate",
    rf: float = 0.0,
    computed_at_ns: int | None = None,
) -> StatsResult:
    """Recompute the statistics of an existing :class:`~ube.core.result.BacktestResult`.

    Uses the result's cached views (no ledger re-derivation). ``benchmark`` defaults to
    the curve already attached to the result; ``market_data``/``instruments`` are only
    needed when the exposure tier should be included — the result does not store them
    (§7.3), so :meth:`BacktestResult.from_ledger` passes them at construction time.
    """
    return from_arrays(
        result.equity_curve.index,
        result.equity_curve.equity,
        trades=result.trades,
        ledger=result.ledger,
        market_data=market_data,
        instruments=instruments,
        benchmark=result.benchmark if benchmark is None else benchmark,
        periods_per_year=periods_per_year,
        resample_to=resample_to,
        return_policy=return_policy,
        rf=rf,
        computed_at_ns=computed_at_ns,
    )


# ---------------------------------------------------------------------------
# Persistence (paper trading, §9.5/§18 — same auto-create/override conventions).
# ---------------------------------------------------------------------------


def save_stats(
    db_path: str | Path, run_id: str, stats: StatsResult, *, trade_count: int = 0
) -> int:
    """Persist one statistics snapshot for ``run_id``; returns its ``seq``.

    The snapshot is append-only: every call allocates the next ``seq`` for the run so
    :func:`load_stats` can always answer "the latest refresh". Tables are created
    idempotently, mirroring :func:`ube.papertrading.state.save_trades`.
    """
    if not isinstance(stats, StatsResult):
        raise TypeError(f"save_stats expects a StatsResult; got {type(stats).__name__}")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute(_STATS_SCHEMA_SQL)
        conn.execute(_STATS_META_SCHEMA_SQL)
        _ensure_stats_meta_equity_points(conn)
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM stats_meta WHERE run_id = ?", (run_id,)
        ).fetchone()
        seq = int(row[0]) if row else 1
        ppy = stats.periods_per_year
        conn.execute(
            "INSERT INTO stats_meta (run_id, seq, as_of_ns, trade_count, n_equity_points, "
            "periods_per_year, tiers, errors_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                seq,
                int(stats.computed_at_ns),
                int(trade_count),
                int(stats.meta.get("n_equity_points", 0) or 0),
                float(ppy) if ppy is not None and math.isfinite(ppy) else None,
                ",".join(stats.tiers),
                json.dumps(stats.errors, sort_keys=True),
            ),
        )
        for record in stats.to_records():
            conn.execute(
                "INSERT OR REPLACE INTO stats (run_id, seq, metric, value, value_json, "
                "category, ref) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    seq,
                    record["metric"],
                    record["value"],
                    record["value_json"],
                    record["category"],
                    record["ref"],
                ),
            )
        return seq
    finally:
        conn.close()


def load_stats(
    db_path: str | Path, run_id: str = "default", *, seq: int | None = None
) -> pd.DataFrame:
    """Load a persisted statistics snapshot as a DataFrame (empty when there is none).

    Args:
        db_path: The paper-trading sqlite file.
        run_id: The strategy run to read.
        seq: Snapshot to read; the latest one when omitted.

    Returns:
        One row per metric, joined with its snapshot metadata (``seq``, ``as_of_ns``,
        ``trade_count``, ``periods_per_year``, ``tiers``, ``errors_json``). ``value`` is
        the finite scalar; ``value_json`` carries ``inf``/``NaN`` scalars and non-scalar
        metrics as JSON text.
    """
    if not Path(db_path).exists():
        return pd.DataFrame(columns=LOAD_COLUMNS)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(_STATS_SCHEMA_SQL)
        conn.execute(_STATS_META_SCHEMA_SQL)
        _ensure_stats_meta_equity_points(conn)
        target = seq
        if target is None:
            row = conn.execute(
                "SELECT MAX(seq) FROM stats_meta WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None or row[0] is None:
                return pd.DataFrame(columns=LOAD_COLUMNS)
            target = int(row[0])
        rows = conn.execute(
            "SELECT s.metric, s.value, s.value_json, s.category, s.ref, m.seq, m.as_of_ns, "
            "m.trade_count, m.n_equity_points, m.periods_per_year, m.tiers, m.errors_json "
            "FROM stats AS s JOIN stats_meta AS m ON s.run_id = m.run_id AND s.seq = m.seq "
            "WHERE s.run_id = ? AND s.seq = ? ORDER BY s.category, s.metric",
            (run_id, int(target)),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=LOAD_COLUMNS)
        return pd.DataFrame(rows, columns=LOAD_COLUMNS)
    finally:
        conn.close()


def _ensure_stats_meta_equity_points(conn: sqlite3.Connection) -> None:
    """Add the equity-grid size needed to resume incomplete snapshot retries."""
    columns = conn.execute("PRAGMA table_info(stats_meta)").fetchall()
    if not any(column[1] == "n_equity_points" for column in columns):
        conn.execute(
            "ALTER TABLE stats_meta ADD COLUMN n_equity_points INTEGER NOT NULL DEFAULT 0"
        )


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _stratstat() -> Any:
    """Import stratstat lazily so ``import ube`` never hard-requires the package."""
    try:
        import stratstat
    except ImportError as exc:  # pragma: no cover — depends on the environment
        raise ImportError(
            "ube.stats requires the 'stratstat' package; install it with "
            "`pip install 'stratstat>=0.1.0'`"
        ) from exc
    return stratstat


def _now_ns(value: int | None = None) -> int:
    if value is not None:
        return int(value)
    return int(pd.Timestamp.now(tz="UTC").value)


def _record_warnings(meta: dict[str, Any], caught: list[warnings.WarningMessage]) -> None:
    if caught:
        meta["warnings"] = [
            f"{w.category.__name__}: {w.message}" for w in caught[:20]  # noqa: B007
        ]


def _jsonable(value: Any) -> Any:
    """Convert a non-scalar metric to a JSON-serialisable Python structure."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _unpack(
    metric_set: Any,
) -> tuple[dict[str, float], dict[str, Any], dict[str, str], dict[str, str]]:
    """Split a stratstat :class:`MetricSet` into scalars, objects, categories, refs."""
    values: dict[str, float] = {}
    objects: dict[str, Any] = {}
    categories: dict[str, str] = {}
    refs: dict[str, str] = {}
    for metric in metric_set.results:
        name = str(metric.name)
        category = metric.category
        if isinstance(category, (tuple, list)):
            categories[name] = ".".join(str(part) for part in category)
        else:
            categories[name] = str(category)
        if isinstance(metric.meta, Mapping):
            ref = metric.meta.get("ref")
            if ref:
                refs[name] = str(ref)
        value = metric.value
        if isinstance(value, (int, float, np.integer, np.floating)) and np.ndim(value) == 0:
            values[name] = float(value)
        else:
            objects[name] = _jsonable(value)
    return values, objects, categories, refs


def _clean_returns(series: pd.Series, policy: str) -> tuple[pd.Series, int]:
    """Apply the ``NaN``-handling policy to a return series (§10 safety rule).

    Returns the cleaned series and the number of non-finite points the policy handled
    (dropped, or replaced with ``0.0`` under ``"zero"``).
    """
    if policy not in RETURN_POLICIES:
        raise ValueError(f"return_policy must be one of {RETURN_POLICIES}; got {policy!r}")
    raw = series.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(raw)
    if finite.all():
        return series, 0
    if policy == "truncate":
        cut = int(np.argmax(~finite))
        return series.iloc[:cut], len(series) - cut
    if policy == "drop":
        kept = series[finite]
        return kept, len(series) - len(kept)
    return (
        pd.Series(np.where(finite, raw, 0.0), index=series.index, name=series.name),
        int((~finite).sum()),
    )


def _infer_periods_per_year(index: pd.Index, errors: dict[str, str]) -> float | None:
    """Annualisation factor from the equity grid (§4.9), or ``None`` when unknowable."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    try:
        period_ns = float(derive_bar_period_ns(index))
    except Exception as exc:  # noqa: BLE001 — a degenerate grid is not fatal
        errors["periods_per_year"] = f"{type(exc).__name__}: {exc}"
        return None
    if not math.isfinite(period_ns) or period_ns <= 0:
        errors["periods_per_year"] = f"derived bar period {period_ns!r} is not positive"
        return None
    return SECONDS_PER_YEAR * 1e9 / period_ns


def _trade_input(
    trade_list: Sequence[Trade], period_ns: int | None
) -> tuple[dict[str, list[Any]], str]:
    """Map closed UBE trades onto stratstat's canonical trade-log columns."""
    net = np.asarray([float(t.net_pnl) for t in trade_list], dtype=np.float64)
    notional = np.asarray([float(t.entry_notional) for t in trade_list], dtype=np.float64)
    usable = np.isfinite(net) & np.isfinite(notional) & (np.abs(notional) > 0.0)
    if usable.all():
        pnl = np.divide(net, notional, out=np.full_like(net, np.nan), where=usable)
        unit = "fraction"
    else:
        # At least one trade has no entry notional: fall back to raw currency. Stratstat
        # refuses the metrics that need a fraction (Kelly, geometric mean) rather than
        # guessing, which is the honest outcome.
        pnl = net
        unit = "currency"

    columns: dict[str, list[Any]] = {
        "pnl": [float(v) for v in pnl],
        "side": ["long" if int(t.side) > 0 else "short" for t in trade_list],
    }
    if period_ns:
        entries = np.asarray([int(t.entry_timestamp) for t in trade_list], dtype=np.int64)
        exits = np.asarray([int(t.exit_timestamp) for t in trade_list], dtype=np.int64)
        durations = np.rint((exits - entries) / float(period_ns)).astype(np.int64)
        columns["duration"] = [int(max(v, 0)) for v in durations]
    return columns, unit


def _align_benchmark(
    returns: pd.Series, benchmark: BenchmarkCurve | np.ndarray | pd.Series
) -> tuple[pd.Series, pd.Series, str | None]:
    """Align strategy and benchmark returns bar-for-bar (§10 positional contract).

    ``BenchmarkCurve`` carries no index — ``ube.run`` builds it from the same
    warm-up/date-range-sliced bars, so alignment is positional by construction; a length
    mismatch means something upstream drifted, and the honest reaction is to truncate to
    the common length and say so.
    """
    if isinstance(benchmark, BenchmarkCurve):
        bench = np.asarray(benchmark.returns, dtype=np.float64)
    elif isinstance(benchmark, pd.Series):
        bench = np.asarray(benchmark.to_numpy(dtype=np.float64), dtype=np.float64)
    else:
        bench = np.asarray(benchmark, dtype=np.float64).reshape(-1)

    note = None
    if len(bench) != len(returns):
        common = min(len(returns), len(bench))
        note = (
            f"strategy has {len(returns)} bar(s), benchmark {len(bench)}; "
            f"aligned on the first {common}"
        )
        returns = returns.iloc[:common]
        bench = bench[:common]
    return returns, pd.Series(bench, index=returns.index), note


def _exposure_frame(
    ledger: EventLedger,
    market_data: Mapping[str, MarketData],
    instruments: Mapping[str, Any],
    index: pd.Index,
    equity: np.ndarray,
    errors: dict[str, str],
    notes: dict[str, Any],
) -> pd.DataFrame | None:
    """Per-instrument position weights (notional / equity) on ``index``.

    Returns ``None`` — with the reason in ``errors["exposure"]`` — when the ledger has no
    position history or an instrument cannot be priced over the grid. When the bars start
    after the grid (a live ``step()`` window) the tile is still computed, but over the bars
    supplied; ``notes["exposure_window"]`` records that so a partial series is never
    mistaken for full-history exposure.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.size == 0 or not instruments:
        return None
    try:
        grid_start = int(index.as_unit("ns").asi8[0])  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        grid_start = int(index[0].value)

    notionals: dict[str, pd.Series] = {}
    has_positions = False
    try:
        for instrument_id in instruments:
            md = market_data.get(instrument_id)
            if md is None:
                errors["exposure"] = f"no market data for {instrument_id!r}; exposure skipped"
                return None
            try:
                bar_start = int(md.timestamps.as_unit("ns").asi8[0])  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                bar_start = int(md.index[0].value)
            if bar_start > grid_start:
                notes["exposure_window"] = (
                    f"bars for {instrument_id!r} start after the equity grid "
                    f"({bar_start} > {grid_start}); exposure covers the supplied window only"
                )

            pos: Positions = positions(ledger, instrument_id=instrument_id)
            if pos.timestamps.size == 0:
                continue
            has_positions = True
            units = (
                pd.Series(
                    np.asarray(pos.position, dtype=np.float64),
                    index=pd.DatetimeIndex(pd.to_datetime(pos.timestamps, unit="ns", utc=True)),
                )
                .reindex(index, method="ffill")
                .fillna(0.0)
            )
            close = pd.Series(np.asarray(md.close, dtype=np.float64), index=md.index)
            close = close.reindex(index, method="ffill").fillna(0.0)
            multiplier = float(
                getattr(instruments[instrument_id], "contract_multiplier", 1.0) or 1.0
            )
            notionals[instrument_id] = units * close.fillna(0.0) * multiplier
    except Exception as exc:  # noqa: BLE001 — a malformed ledger is not fatal here
        errors["exposure"] = f"{type(exc).__name__}: {exc}"
        return None

    if not has_positions:
        errors["exposure"] = "no position changes in the ledger; exposure skipped"
        return None

    frame = pd.DataFrame(notionals, index=index)
    safe_equity = np.where(np.isfinite(equity) & (equity > 0.0), equity, np.nan)
    return frame.div(safe_equity, axis=0).fillna(0.0)

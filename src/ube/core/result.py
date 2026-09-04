"""BacktestResult — the frozen result container with cached derived views (§7.3, §4.6).

:class:`BacktestResult` is the canonical *output* of one backtest run. It is immutable
after construction (§3 principle 5) and carries the event ledger (the single source of
truth, §4.6) plus the derived views — ``trades``, ``trade_table``, ``positions``,
``equity_curve`` and ``equity_curve_by_instrument`` — each computed **once** at
construction from the ledger by the pure item-07 functions
(:func:`~ube.core.ledger.trades`,
:func:`~ube.core.ledger.trade_table`,
:func:`~ube.core.ledger.positions`,
:func:`~ube.core.ledger.equity_curve`,
:func:`~ube.core.ledger.equity_curve_by_instrument`) and cached. They are never
recomputed on attribute access (§4.6: "cached views derived from the ledger once").

Design notes:

* **Construction.** The primary entry point is :meth:`BacktestResult.from_ledger`, which
  takes the ledger, the config, and the mark-to-market inputs (``market_data`` +
  ``instruments`` + optional ``fx_rates``) needed to derive the equity curves, then
  computes every derived view once. The plain dataclass constructor is also available
  for callers that already have the views.
* **Positions, single vs portfolio.** A combined position across instruments is undefined
  (§4.6/§7.3), so ``.positions`` holds the combined :class:`~ube.core.ledger.Positions`
  only when the ledger's ``position_change`` events reference a single instrument; it is
  ``None`` for a multi-instrument ledger (consumers needing a per-instrument breakdown
  call the public :func:`~ube.core.ledger.positions(ledger, instrument_id=...)`).
* **base_currency.** The equity curves are denominated in ``config.base_currency``. For a
  single-instrument run with no declared ``base_currency`` (§7.1), the curve is
  denominated in the lone instrument's ``settlement_currency`` (identity FX — the
  natural denomination, not a silently-guessed cross-currency rate). A portfolio run
  with no ``base_currency`` raises :class:`~ube.core.errors.UndeclaredConfigError`
  (§4.7).
* **Warm-up.** When ``config.warmup_bars > 0`` (§7.2), the first N bars are sliced off
  ``equity_curve`` / ``equity_curve_by_instrument`` and completed trades whose entry bar
  falls inside the warm-up window are dropped. Flagging individual ledger events
  ``warmup=True`` is **deferred** (a Phase-2 refinement); only the derived views are
  filtered here.
* **Persistence.** :meth:`save` / :meth:`load` use ``pickle`` — the only format that
  round-trips the full event ledger (parquet cannot carry the heterogeneous sealed event
  schema cleanly). Opt-in and user-driven (§7.3); results are never auto-stored.

``metrics`` is left as ``None`` here: performance metrics are computed by an external
library, not by ``ube`` (§10). The result still exposes the clean data the external
metrics layer consumes — ``ledger``, ``trades``, ``trade_table``, ``positions``,
``equity_curve`` (+ ``.returns`` / ``.resample``) and the (optional) benchmark curve.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ube.core.benchmark import BenchmarkCurve
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import DataShapeError, UndeclaredConfigError
from ube.core.instrument import Instrument
from ube.core.ledger import (
    EquityCurve,
    EventLedger,
    EventType,
    FXSeries,
    Positions,
    Trade,
    equity_curve,
    equity_curve_by_instrument,
    positions,
    trade_table,
    trades,
)

__all__ = ["BacktestResult", "result_hash"]


@dataclass(frozen=True)
class BacktestResult:
    """The frozen, canonical result of one backtest run (§7.3).

    Attributes:
        run_id: A unique string identifier for the run (§4.8); defaulted to a fresh
            UUID4 by :meth:`from_ledger` when not supplied.
        ledger: The append-only event ledger — the single source of truth (§4.6).
        config: The :class:`~ube.core.config.BacktestConfig` this run was produced with.
        trades: The completed round-trip trades, derived once from the ledger (§4.6).
        trade_table: The §4.6 trades table — one row per trade (closed *and* open), the
            spec'd per-trade columns joined with the equity curve (``position_size_pct``,
            ``trade_return_pct``, ``realized_pnl``/``realized_pnl_pct``,
            ``cum_return_pct``, ``balance``, fee splits, ``reason``; the ``_pct`` columns
            are percentages — ``5.0`` = 5%). A convenience / reporting view — not a
            canonical source of truth (see :func:`~ube.core.ledger.trade_table`).
        positions: The combined position-over-time series when the ledger is
            single-instrument, else ``None`` (see the module docstring).
        equity_curve: The combined equity curve in ``base_currency`` (§4.6).
        equity_curve_by_instrument: The per-instrument equity breakdown, on the same
            combined grid (§4.6).
        metrics: Performance metrics — always ``None`` here. Metrics are computed by an
            external library (§10), not by ``ube``; this field is the hand-off slot the
            external layer may populate.
        benchmark: The benchmark curve for this run (``None`` when not computed).
    """

    run_id: str
    ledger: EventLedger
    config: BacktestConfig
    trades: tuple[Trade, ...]
    trade_table: pd.DataFrame
    positions: Positions | None
    equity_curve: EquityCurve
    equity_curve_by_instrument: dict[str, EquityCurve]
    metrics: Any | None = None
    benchmark: BenchmarkCurve | None = None

    def __post_init__(self) -> None:
        # Copy the per-instrument breakdown into a fresh dict so the frozen result never
        # aliases a caller-supplied mapping (immutability rule). The values are already
        # frozen :class:`~ube.core.ledger.EquityCurve` instances from item 07.
        object.__setattr__(
            self, "equity_curve_by_instrument", dict(self.equity_curve_by_instrument)
        )

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    @classmethod
    def from_ledger(
        cls,
        ledger: EventLedger,
        config: BacktestConfig,
        *,
        market_data: Mapping[str, MarketData],
        instruments: Mapping[str, Instrument],
        fx_rates: Mapping[str, FXSeries] | None = None,
        benchmark: BenchmarkCurve | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        """Build a result from a ledger, deriving and caching every view once (§4.6).

        The ``trades`` / ``positions`` / ``equity_curve`` / ``equity_curve_by_instrument``
        views are computed here (via the item-07 pure functions) and cached — they are
        *not* recomputed on later attribute access. ``market_data`` and ``instruments``
        are required only to mark open positions to market (§4.6 step 2); they are not
        stored on the result.

        Args:
            ledger: The append-only event ledger (single source of truth).
            config: The run's ``BacktestConfig`` (supplies ``base_currency`` and
                ``warmup_bars``).
            market_data: ``instrument_id -> MarketData`` (bars for mark-to-market).
            instruments: ``instrument_id -> Instrument`` (provides
                ``settlement_currency``).
            fx_rates: Optional historical FX rates keyed by currency pair.
            benchmark: Optional precomputed :class:`~ube.core.benchmark.BenchmarkCurve`.
            run_id: Optional run identifier; a fresh UUID4 is generated when ``None``.

        Returns:
            A frozen :class:`BacktestResult` with all derived views cached.
        """
        if not isinstance(ledger, EventLedger):
            raise DataShapeError("from_ledger expects an EventLedger")
        if not isinstance(config, BacktestConfig):
            raise DataShapeError("from_ledger expects a BacktestConfig")

        base_currency = _resolve_base_currency(config, instruments)

        trades_view = trades(ledger, instruments=instruments)
        trade_table_view = trade_table(
            ledger,
            market_data,
            instruments,
            base_currency=base_currency,
            fx_rates=fx_rates,
        )
        pc_ids = {e.instrument_id for e in ledger if e.event_type is EventType.POSITION_CHANGE}
        positions_view = positions(ledger) if len(pc_ids) <= 1 else None

        combined = equity_curve(
            ledger, market_data, instruments, base_currency=base_currency, fx_rates=fx_rates
        )
        by_instrument = equity_curve_by_instrument(
            ledger, market_data, instruments, base_currency=base_currency, fx_rates=fx_rates
        )

        if config.warmup_bars > 0:
            trades_view = _drop_warmup_trades(
                trades_view, combined.index, config.warmup_bars
            )
            trade_table_view = _drop_warmup_table_rows(
                trade_table_view, combined.index, config.warmup_bars
            )
            combined = _slice_curve(combined, config.warmup_bars)
            by_instrument = {
                iid: _slice_curve(curve, config.warmup_bars)
                for iid, curve in by_instrument.items()
            }

        return cls(
            run_id=run_id if run_id is not None else str(uuid.uuid4()),
            ledger=ledger,
            config=config,
            trades=trades_view,
            trade_table=trade_table_view,
            positions=positions_view,
            equity_curve=combined,
            equity_curve_by_instrument=by_instrument,
            metrics=None,
            benchmark=benchmark,
        )

    # ------------------------------------------------------------------
    # Persistence (opt-in, §7.3).
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist this result to ``path`` as a pickle (§7.3, opt-in).

        Pickle is used because parquet cannot round-trip the full heterogeneous event
        ledger cleanly. Nothing is stored automatically; the user calls this explicitly.
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> BacktestResult:
        """Load a :class:`BacktestResult` previously written by :meth:`save`.

        Pickle does not preserve ``ndarray.flags.writeable``, so the read-only lock on
        the derived-view arrays is re-applied here (§3 principle 5 — a loaded result is
        frozen, same as a freshly built one).
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise DataShapeError(
                f"load({path}) did not produce a BacktestResult; got {type(obj).__name__}"
            )
        _refreeze(obj.equity_curve)
        _refreeze(obj.equity_curve_by_instrument)
        _refreeze(obj.positions)
        _refreeze(obj.benchmark)
        return obj


def result_hash(result: BacktestResult) -> str:
    """SHA-256 fingerprint of a result's material output (§4.8, §16).

    Hashes the completed ``trades`` — a canonical, deterministic projection sorted by
    entry timestamp (the same field set as the §16 parity ``trades_hash``) — together
    with the combined ``equity_curve`` values, so two runs that agree on both are treated
    as identical. This is the ``result_hash`` stored in the experiment log (§4.8) and the
    basis of cross-engine parity (§16).

    Args:
        result: The run's :class:`BacktestResult`.

    Returns:
        The hex SHA-256 digest.

    Raises:
        DataShapeError: If ``result`` is not a :class:`BacktestResult`.
    """
    if not isinstance(result, BacktestResult):
        raise DataShapeError(
            f"result_hash expects a BacktestResult; got {type(result).__name__}"
        )
    digest = hashlib.sha256()
    records = []
    for trade in sorted(result.trades, key=lambda t: t.entry_timestamp):
        records.append(
            {
                "instrument_id": trade.instrument_id,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_timestamp": trade.entry_timestamp,
                "exit_timestamp": trade.exit_timestamp,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "gross_pnl": trade.gross_pnl,
                "commission": trade.commission,
                "funding": trade.funding,
                "net_pnl": trade.net_pnl,
                "exit_reason": trade.exit_reason,
            }
        )
    digest.update(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(result.equity_curve.equity.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------

#: Per-asset-class default settlement currency used only when neither
#: ``config.base_currency`` nor ``Instrument.settlement_currency`` is declared (§4.7).
#: Keeps the minimal ``ube.Instrument(symbol, asset_class=...)`` usage runnable
#: without forcing a settlement currency on the caller.
_DEFAULT_SETTLEMENT: Mapping[str, str] = {
    "crypto_spot": "USDT",
    "crypto_perp": "USDT",
    "forex": "USD",
    "futures": "USD",
    "stocks": "USD",
    "commodities": "USD",
}


def _resolve_base_currency(
    config: BacktestConfig, instruments: Mapping[str, Instrument]
) -> str:
    """Resolve the currency the equity curves are denominated in (§4.6, §4.7).

    ``config.base_currency`` when declared; otherwise (single-instrument run) the lone
    instrument's ``settlement_currency`` (identity FX — the natural denomination). A
    portfolio run with no declared ``base_currency`` raises ``UndeclaredConfigError``.
    """
    base = config.base_currency
    if base is not None:
        return base

    if len(instruments) != 1:
        raise UndeclaredConfigError(
            "base_currency is required to derive the combined equity curve for a "
            "portfolio run but was not declared on BacktestConfig (§4.7 — never "
            "silently assume a base currency)"
        )
    only = next(iter(instruments.values()))
    settlement = only.settlement_currency
    if settlement is None:
        # No explicit settlement currency was declared. Rather than raise on the
        # simplest-usage path (``ube.Instrument(symbol, asset_class=...)`` with no
        # settlement), fall back to the asset-class default so the equity curve has a
        # denomination. A single-instrument run keeps a 1:1 identity FX; this only
        # picks the currency label. ``config.base_currency`` (when declared) still
        # wins, and portfolio runs still require an explicit base (see above).
        settlement = _DEFAULT_SETTLEMENT.get(only.asset_class, "USD")
    return settlement


def _slice_curve(curve: EquityCurve, n: int) -> EquityCurve:
    """Drop the first ``n`` bars of ``curve`` (warm-up exclusion, §7.2)."""
    return EquityCurve(index=curve.index[n:], equity=curve.equity[n:])


def _drop_warmup_trades(
    trades_view: tuple[Trade, ...],
    index: pd.Index,
    warmup_bars: int,
) -> tuple[Trade, ...]:
    """Drop completed trades whose entry bar falls inside the warm-up window (§7.2).

    The warm-up window is the first ``warmup_bars`` bars of the combined grid; a trade
    entering on a warm-up bar is excluded from statistics. The cutoff is derived from
    the grid's own bar-boundary timestamps (a tz-aware ``DatetimeIndex`` → int64
    nanoseconds since the epoch), matching how the ledger's ``entry_timestamp`` is
    populated, so the two are directly comparable.
    """
    if warmup_bars >= len(index):
        return ()
    if isinstance(index, pd.DatetimeIndex):
        asi8: np.ndarray = index.as_unit("ns").asi8  # type: ignore[attr-defined]
        cutoff = int(asi8[warmup_bars])
    else:
        cutoff = int(index[warmup_bars])
    return tuple(t for t in trades_view if t.entry_timestamp >= cutoff)


def _drop_warmup_table_rows(
    table: pd.DataFrame, index: pd.Index, warmup_bars: int
) -> pd.DataFrame:
    """Drop trades-table rows whose entry bar falls inside the warm-up window (§7.2).

    Mirrors :func:`_drop_warmup_trades` for the §4.6 trades table, which also includes
    open rows (those carry their entry bar in ``entry_datetime``). Rows are compared on
    the tz-aware ``entry_datetime`` column against the same grid-derived cutoff.
    """
    if warmup_bars >= len(index):
        return table.iloc[0:0]
    if isinstance(index, pd.DatetimeIndex):
        asi8: np.ndarray = index.as_unit("ns").asi8  # type: ignore[attr-defined]
        cutoff = int(asi8[warmup_bars])
    else:
        cutoff = int(index[warmup_bars])
    cutoff_ts = pd.Timestamp(cutoff, unit="ns", tz="UTC")
    return table[table["entry_datetime"] >= cutoff_ts]


def _refreeze(obj: Any) -> None:
    """Re-apply ``write=False`` to every derived-view array reachable from ``obj``.

    Pickle drops ``ndarray.flags.writeable``, so a loaded result's derived-view arrays
    come back writable; this restores the immutability guarantee (§3 principle 5). It
    walks only the known array-bearing containers (``EquityCurve``/``Positions``/
    ``BenchmarkCurve``/``pd.DataFrame``) plus dict/tuple/list aggregations — never
    arbitrary ``__dict__`` attributes, which would descend into pandas internals that
    contain reference cycles. The containers reached here are freshly deserialized
    (owned by the loaded result), so locking them is safe — unlike construction, where
    caller-owned buffers must never be frozen.
    """
    if isinstance(obj, np.ndarray):
        obj.setflags(write=False)
    elif isinstance(obj, pd.DataFrame):
        for column in obj.columns:
            values = obj[column]._values
            if isinstance(values, np.ndarray):
                values.setflags(write=False)
    elif isinstance(obj, EquityCurve):
        obj.equity.setflags(write=False)
    elif isinstance(obj, Positions):
        obj.timestamps.setflags(write=False)
        obj.position.setflags(write=False)
    elif isinstance(obj, BenchmarkCurve):
        obj.returns.setflags(write=False)
        obj.equity.setflags(write=False)
    elif isinstance(obj, dict):
        for value in obj.values():
            _refreeze(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _refreeze(value)

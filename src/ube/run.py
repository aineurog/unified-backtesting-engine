"""Top-level ``ube.run`` orchestrator (§4.8, §7.1, §5.2).

:func:`run` is the single user-facing entry point that ties the Phase-1 core together
behind one call: it standardizes the inputs, resolves the engine, runs the adapter,
attaches the benchmark, and records the run to the experiment log — in that order. It is
deliberately a thin orchestrator: every piece of real work is a Phase-1 core function or
adapter, so the orchestration itself carries no engine-specific logic (§4.2).

The inputs mirror the canonical contract:

* ``data`` — any of the four shapes :meth:`~ube.core.data.MarketData.standardize`
  accepts (DataFrame / array / dict / records) or an already-standardized
  :class:`~ube.core.data.MarketData`; a ``{instrument_id: MarketData}`` mapping for a
  portfolio run (forwarded as-is — the adapter decides support).
* ``signals`` — a :class:`~ube.core.signals.Signals` or a 4-column signal
  :class:`pandas.DataFrame`; richer encodings (a ``-1/0/1`` target array or a per-bar
  callable) are converted *first* via :func:`~ube.core.signals.from_target` /
  :func:`~ube.core.signals.from_callable`, not guessed here.
* ``config`` — the full :class:`~ube.core.config.BacktestConfig`.
* ``aux_data`` — a flat ``name -> value`` mapping referenced by name from exit configs
  (e.g. ``ATRStop(atr="atr_12h")``). A value may be a precomputed 1-D array (length
  must match ``data``'s bar count), or a :class:`~ube.core.data.MarketData` of the
  *signal-timeframe OHLCV* — in which case the adapter computes the ATR internally (with
  no look-ahead bias) and aligns it to ``data``'s bar grid. When a referenced name is
  absent the library computes the series from ``data``.

Engines are registered lazily (:func:`ensure_builtin_engines_registered`) so that
``import ube`` never hard-requires an optional engine dependency (§6 pyproject).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ube.adapters import get_engine, register_engine, resolve_engine_name
from ube.core.benchmark import build_benchmark
from ube.core.calendar import resolve_calendar, validate_in_session
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, InvalidSignalError
from ube.core.experiment_log import DataReference, ExperimentLog
from ube.core.result import BacktestResult, result_hash
from ube.core.signals import Signals


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    """Coerce a ``date_range`` bound to a tz-aware UTC ``Timestamp`` (§7.2)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _window_mask(index: pd.DatetimeIndex, start: Any, end: Any) -> np.ndarray:
    """Boolean mask of bars inside the inclusive ``[start, end]`` window."""
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= _as_utc_timestamp(start)
    if end is not None:
        mask &= index <= _as_utc_timestamp(end)
    return mask


def _slice_market_data(md: MarketData, start: Any, end: Any) -> tuple[MarketData, np.ndarray]:
    """Restrict ``md`` to the inclusive ``[start, end]`` window (§7.2).

    Returns the sliced ``MarketData`` and the boolean mask (over the original bars) so
    aligned ``aux_data`` can be sliced by the same positions. Raises ``DataShapeError``
    when the window excludes every bar — never a silent empty run.
    """
    # MarketData guarantees a tz-aware DatetimeIndex (raised at construction otherwise),
    # so this narrows the declared ``pd.Index`` to what _window_mask needs.
    index = md.index
    assert isinstance(index, pd.DatetimeIndex), "MarketData.index must be a DatetimeIndex"
    mask = _window_mask(index, start, end)
    if not mask.any():
        raise DataShapeError(
            f"date_range {start!r}..{end!r} excludes every bar "
            f"(data spans {md.index[0]}..{md.index[-1]})"
        )
    sliced = MarketData(
        open=md.open[mask],
        high=md.high[mask],
        low=md.low[mask],
        close=md.close[mask],
        volume=md.volume[mask],
        index=md.index[mask],
    )
    return sliced, mask


def _slice_aux_data(
    aux_data: Mapping[str, Any],
    mask: np.ndarray,
    start: object,
    end: object,
) -> Mapping[str, Any]:
    """Slice ``aux_data`` to match a date_range-sliced single-instrument run.

    Precomputed arrays are sliced by ``mask`` (length-aligned to the data bars); an
    ``aux_data`` ``MarketData`` is sliced to the same timestamp window.
    """
    out: dict[str, Any] = {}
    for name, val in aux_data.items():
        if isinstance(val, MarketData):
            out[name] = _slice_market_data(val, start, end)[0]
        else:
            out[name] = np.asarray(val)[mask]
    return out


def _slice_signals(sig: Signals, mask: np.ndarray) -> Signals:
    """Slice the four boolean signal columns by the same bar ``mask`` (§6.1, §7.2)."""
    return Signals(
        long_entry=sig.long_entry[mask],
        long_exit=sig.long_exit[mask],
        short_entry=sig.short_entry[mask],
        short_exit=sig.short_exit[mask],
    )


__all__ = ["run", "ensure_builtin_engines_registered"]


def ensure_builtin_engines_registered() -> None:
    """Register the built-in engine adapters whose dependency is importable (§4.1).

    Only Nautilus is implemented today (the vectorbt / backtrader adapters are stubs whose
    ``run`` raises ``NotImplementedError``), and it depends on the *optional*
    ``nautilus-trader`` package. Registration is therefore guarded by an ``ImportError``
    check so ``import ube`` never hard-requires it. Idempotent — re-registering the same
    class is a no-op — so calling this repeatedly is safe.
    """
    try:
        from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    except ImportError:
        return
    register_engine("nautilus", NautilusAdapter)


def _standardize_data(data: object) -> MarketData:
    """Coerce ``data`` to a canonical :class:`MarketData` (§5.1)."""
    if isinstance(data, MarketData):
        return data
    return MarketData.standardize(data)


def _standardize_signals(signals: object) -> Signals:
    """Coerce ``signals`` to canonical :class:`Signals` (§6.1), or fail with guidance."""
    if isinstance(signals, Signals):
        return signals
    if isinstance(signals, pd.DataFrame):
        return Signals.from_dataframe(signals)
    raise InvalidSignalError(
        "run expects canonical Signals or a 4-column signal DataFrame; got "
        f"{type(signals).__name__} — build one with ube.signals.from_target(...), "
        "ube.signals.from_callable(...), or ube.signals.Signals.from_dataframe(...)"
    )


#: Per-asset-class default settlement currency, applied when an instrument declares
#: none (mirrors :data:`ube.core.result._DEFAULT_SETTLEMENT`).
_DEFAULT_SETTLEMENT: dict[str, str] = {
    "crypto_spot": "USDT",
    "crypto_perp": "USDT",
    "forex": "USD",
    "futures": "USD",
    "stocks": "USD",
    "commodities": "USD",
}


def _default_settlement_currency(config: BacktestConfig) -> BacktestConfig:
    """Return a config whose instrument has a settlement currency when omitted.

    The minimal ``ube.Instrument(symbol, asset_class=...)`` usage declares no
    settlement currency; the downstream ledger normalization (core) requires one. It is
    defaulted per asset class here (the canonical entry point) so the simple
    construction path runs without forcing a settlement currency on the caller.
    """
    instrument = config.instrument
    if instrument.settlement_currency is not None:
        return config
    settled = _DEFAULT_SETTLEMENT.get(instrument.asset_class, "USD")
    return replace(
        config,
        instrument=replace(instrument, settlement_currency=settled),
    )


def run(
    data: object,
    signals: object,
    config: BacktestConfig,
    *,
    aux_data: Mapping[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> BacktestResult:
    """Run one backtest end-to-end and return the canonical result (§4.8, §7.1).

    Orchestrates the Phase-1 pipeline in order:

    1. Validate ``config`` and default any omitted settlement currency.
    2. Apply ``config.date_range`` (§7.2): restrict the bars (and aligned ``aux_data``)
       to the declared ``(start, end)`` window before the engine sees them — a window
       that excludes every bar raises ``DataShapeError``. The full dataset is never
       silently run when a ``date_range`` is set.
    3. For a single-instrument run, standardize ``data``/``signals`` to the canonical
       containers; a portfolio (dict-keyed) run is forwarded to the adapter as-is.
    3. Lazily register the installed built-in engine, then resolve ``config.engine``
       (``"auto"`` → first installed) to the adapter class.
    4. Run the adapter, producing a :class:`~ube.core.result.BacktestResult`.
    5. For a single-instrument run, compute and attach the
       :class:`~ube.core.benchmark.BenchmarkCurve` (§10) engine-agnostically and record
       the run to the experiment log (§4.8) — unconditional — with the
       :func:`~ube.core.result.result_hash` fingerprint.

    Args:
        data: The single-instrument OHLCV bars (any :meth:`MarketData.standardize`
            shape, or an already-standardized :class:`~ube.core.data.MarketData`); a
            ``{instrument_id: MarketData}`` mapping for a portfolio run.
        signals: A :class:`~ube.core.signals.Signals` or 4-column signal
            :class:`pandas.DataFrame`; build richer encodings with
            :func:`~ube.core.signals.from_target` /
            :func:`~ube.core.signals.from_callable` first.
        config: The full :class:`~ube.core.config.BacktestConfig`.
        aux_data: Named derived series (§5.2) passed to the engine adapter; a value may
            be a precomputed array or a signal-timeframe :class:`MarketData` the adapter
            turns into ATR internally.
        log_path: Optional experiment-log path override (§4.8); ``None`` uses the normal
            precedence (explicit → ``BACKTEST_LOG_PATH`` → default).

    Returns:
        The frozen :class:`~ube.core.result.BacktestResult` with the benchmark attached
        (single-instrument runs).

    Raises:
        ConfigError: ``config`` is not a :class:`~ube.core.config.BacktestConfig`, or
            ``config.engine`` names no installed adapter.
        DataShapeError: ``data`` cannot be standardized to a :class:`MarketData`.
        InvalidSignalError: ``signals`` cannot be standardized to :class:`Signals`.
        EngineError: The underlying engine failed (wrapped, original preserved).
    """
    if not isinstance(config, BacktestConfig):
        raise ConfigError(f"run expects a BacktestConfig; got {type(config).__name__}")

    # Run-mode cross-field validation (§4.7): reject portfolios without a declared
    # base_currency (and paper-trading without an on_opposite_signal policy) up-front,
    # before any computation — these have no safe default and guessing wrong yields
    # confidently-wrong results.
    config.validate(portfolio=isinstance(data, Mapping))

    config = _default_settlement_currency(config)

    # Portfolio (dict-keyed) inputs are forwarded as-is; the adapter decides support
    # (§7.1). Benchmark and experiment-log data-reference are single-instrument concepts,
    # so they are only attached on the single-asset path below.
    if isinstance(data, Mapping):
        # §7.2: restrict each instrument's bars to the declared date_range window before
        # the adapter sees them. A window that excludes every bar raises DataShapeError.
        if config.date_range is not None:
            start, end = config.date_range
            data = {k: _slice_market_data(v, start, end)[0] for k, v in data.items()}
        ensure_builtin_engines_registered()
        engine_name = resolve_engine_name(config.engine)
        adapter_cls = get_engine(engine_name)
        # Forwarded as-is: the single-instrument adapter contract types ``data``/``signals``
        # as MarketData/Signals, but a portfolio run passes dict-keyed inputs that only a
        # portfolio-capable adapter understands (it validates at runtime, §7.1).
        return adapter_cls().run(data, signals, config, aux_data=aux_data)  # type: ignore[arg-type]

    md = _standardize_data(data)
    sig = _standardize_signals(signals)

    # §7.2: restrict the bars to the declared date_range window before the adapter sees
    # them, so the backtest only covers the requested bounds. Aligned aux_data (e.g. a
    # named ATR series) is sliced by the same positions to keep lengths consistent.
    if config.date_range is not None:
        start, end = config.date_range
        md, mask = _slice_market_data(md, start, end)
        sig = _slice_signals(sig, mask)
        if aux_data:
            aux_data = _slice_aux_data(aux_data, mask, start, end)

    # Calendar validation (§4.4): a bar whose timestamp falls where the declared trading
    # calendar says the market is closed is a §15 data error, never silently accepted.
    # Crypto's "24/7" calendar is a no-op. This is the single integration point so the
    # check runs for every single-instrument run, regardless of adapter.
    validate_in_session(md, resolve_calendar(config.instrument))

    ensure_builtin_engines_registered()
    engine_name = resolve_engine_name(config.engine)
    adapter_cls = get_engine(engine_name)

    result = adapter_cls().run(md, sig, config, aux_data=aux_data)

    benchmark = build_benchmark(config.benchmark, md)
    result = replace(result, benchmark=benchmark)

    with ExperimentLog(path=log_path) as log:
        log.record(
            run_id=result.run_id,
            config=config,
            engine=engine_name,
            data_reference=DataReference.from_market_data(md, config.instrument),
            result_hash=result_hash(result),
        )

    return result

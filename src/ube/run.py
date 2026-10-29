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

import importlib.util
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ube.adapters import get_engine, register_engine, resolve_engine_name
from ube.core.benchmark import build_benchmark
from ube.core.calendar import resolve_calendar, validate_in_session, validate_timestamps
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, InvalidInstrumentError, InvalidSignalError
from ube.core.experiment_log import DataReference, ExperimentLog, PortfolioDataReference
from ube.core.fx import stablecoin_peg_rate, warn_stablecoin_peg
from ube.core.instrument import resolve_settlement_currency
from ube.core.ledger import EventLedger, FXSeries
from ube.core.portfolio import (
    DEFAULT_PORTFOLIO_BALANCE,
    PortfolioConfig,
    allocation_weights,
)
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
    # but a date_range window is a time-bar concept: reject an event-bar index explicitly
    # rather than relying on an assertion (which ``python -O`` removes).
    index = md.index
    if not isinstance(index, pd.DatetimeIndex):
        raise DataShapeError(
            "date_range slicing requires time bars (a DatetimeIndex); got "
            f"{type(index).__name__} — event bars have no wall-clock window (§7.2)"
        )
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


def _warmup_slice_market_data(md: MarketData, n: int) -> MarketData:
    """Drop the first ``n`` bars of ``md`` positionally, to mirror the engine's warm-up
    exclusion (§7.2) so a benchmark built from the result lines up bar-for-bar with the
    strategy curve.
    """
    if n <= 0:
        return md
    return MarketData(
        open=md.open[n:],
        high=md.high[n:],
        low=md.low[n:],
        close=md.close[n:],
        volume=md.volume[n:],
        index=md.index[n:],
    )


__all__ = ["run", "ensure_builtin_engines_registered"]

#: The FX override aliases every adapter accepts, first-declared wins (§4.7).
_FX_ALIASES: tuple[str, ...] = (
    "synthetic_rates",
    "fixed_conversions",
    "currency_fx",
    "fx_rates",
)


def _portfolio_key(key: object) -> str:
    """Normalize a portfolio data/signal key to its instrument symbol."""
    symbol = getattr(key, "symbol", None)
    if isinstance(symbol, str) and symbol.strip():
        return symbol
    return str(key)


def _normalize_portfolio_mapping(mapping: Mapping[object, Any]) -> dict[str, Any]:
    """Key a portfolio mapping by instrument symbol, rejecting collisions."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        symbol = _portfolio_key(key)
        if symbol in out:
            raise ConfigError(
                f"duplicate portfolio key {symbol!r} (instrument labels must be unique)"
            )
        out[symbol] = value
    return out


def _resolve_portfolio_config(config: BacktestConfig) -> BacktestConfig:
    """Resolve settlement currencies and the base currency for a portfolio config."""
    portfolio = config.portfolio
    if portfolio is None:
        return config
    instruments = tuple(
        instrument
        if instrument.settlement_currency is not None
        else replace(
            instrument,
            settlement_currency=resolve_settlement_currency(instrument),
        )
        for instrument in portfolio.instruments
    )
    resolved = replace(portfolio, instruments=instruments)
    base_currency = config.base_currency or resolved.base_currency
    instrument = resolved.instrument_map[config.instrument.symbol]
    return replace(
        config,
        instrument=instrument,
        base_currency=base_currency,
        portfolio=resolved,
    )


def _standardize_portfolio_signals(
    signals: object, portfolio: PortfolioConfig
) -> dict[str, Signals]:
    """Standardize the per-instrument signal inputs for a portfolio run."""
    if isinstance(signals, Mapping):
        keyed = _normalize_portfolio_mapping(signals)
        portfolio.validate_data(list(keyed))
        return {symbol: _standardize_signals(value) for symbol, value in keyed.items()}
    if len(portfolio.instrument_ids) == 1:
        only = portfolio.instrument_ids[0]
        return {only: _standardize_signals(signals)}
    raise InvalidSignalError(
        "a multi-asset portfolio run expects signals as a "
        "{instrument_id: Signals} mapping; got "
        f"{type(signals).__name__}"
    )


def _portfolio_aux_for(
    aux_data: Mapping[str, Any] | None, symbol: str
) -> Mapping[str, Any] | None:
    """Select one leg's aux map, accepting either a flat map or ``symbol -> map``."""
    if aux_data is None:
        return None
    nested = aux_data.get(symbol)
    if isinstance(nested, Mapping):
        return dict(nested)
    return aux_data


def _portfolio_fx_rates(
    config: BacktestConfig,
    market_data: Mapping[str, MarketData],
    instruments: Mapping[str, Any],
) -> dict[str, FXSeries] | None:
    """Build the constant FX series needed to combine several settlement currencies."""
    overrides = dict(config.engine_overrides or {})
    declared: dict[str, float] = {}
    for alias in _FX_ALIASES:
        value = overrides.get(alias)
        if value is not None:
            declared = {str(key): float(rate) for key, rate in dict(value).items()}
            break
    if not market_data:
        return None
    grid = np.unique(
        np.concatenate(
            [
                md.timestamps.as_unit("ns").to_numpy(dtype=np.int64)
                for md in market_data.values()
            ]
        )
    )
    base_currency = str(config.base_currency or "")
    fx: dict[str, FXSeries] = {
        pair: FXSeries(index=grid, rate=np.full(len(grid), rate))
        for pair, rate in declared.items()
    }
    seeded = False
    for instrument in instruments.values():
        settlement = getattr(instrument, "settlement_currency", None)
        peg = stablecoin_peg_rate(settlement, base_currency)
        if peg is None:
            continue
        for pair in (str(settlement) + base_currency, base_currency + str(settlement)):
            if pair not in fx:
                fx[pair] = FXSeries(index=grid, rate=np.full(len(grid), peg))
                seeded = True
    if seeded:
        currencies = sorted(
            {
                str(getattr(instrument, "settlement_currency", ""))
                for instrument in instruments.values()
            }
            - {base_currency}
        )
        for settlement in currencies:
            warn_stablecoin_peg(settlement, base_currency, engine="portfolio")
    return fx or None


def _merge_portfolio_ledgers(
    legs: list[tuple[str, EventLedger]],
) -> EventLedger:
    """Merge per-instrument ledgers into one chronologically ordered ledger."""
    rows: list[tuple[int, int, int, Any]] = []
    for leg_index, (_symbol, ledger) in enumerate(legs):
        for seq, event in enumerate(ledger.events):
            rows.append((int(event.timestamp), leg_index, seq, event))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return EventLedger(event for _ts, _leg, _seq, event in rows)


def _run_portfolio_backtest(
    data: Mapping[object, Any],
    signals: object,
    config: BacktestConfig,
    *,
    aux_data: Mapping[str, Any] | None,
    adapter_cls: type[Any],
) -> BacktestResult:
    """Run one shared-capital portfolio backtest across every configured instrument.

    Each leg is delegated to the engine adapter exactly as a single-instrument run, but
    receives its precomputed share of the *one* shared account as its starting balance
    (``shared_balance * weight``). The legs are then folded into one combined ledger and
    one :class:`~ube.core.result.BacktestResult`, so the reported equity curve is the
    portfolio's, and every fill stays traceable to its instrument.
    """
    portfolio = config.portfolio
    if portfolio is None:  # pragma: no cover — caller guarantees this
        raise ConfigError("_run_portfolio_backtest requires config.portfolio")

    keyed_data = _normalize_portfolio_mapping(data)
    portfolio.validate_data(list(keyed_data))
    market_data = {
        symbol: _standardize_data(value) for symbol, value in keyed_data.items()
    }
    signal_map = _standardize_portfolio_signals(signals, portfolio)

    weights = allocation_weights(portfolio.allocation, portfolio.instrument_ids)
    overrides = dict(config.engine_overrides or {})
    shared_balance = float(overrides.get("starting_balance", DEFAULT_PORTFOLIO_BALANCE))
    if not np.isfinite(shared_balance) or shared_balance <= 0.0:
        raise ConfigError(
            f"portfolio starting_balance must be finite and > 0; got {shared_balance!r}"
        )

    sizing = portfolio.sizing or config.risk.sizing
    if sizing.kind == "fixed_units":
        raise ConfigError(
            "portfolio runs cannot use fixed_units sizing: units are not scaled by the "
            "shared-capital allocation. Use all_in / equal_weight / fixed_fraction / "
            "volatility_target, or set PortfolioConfig.sizing explicitly."
        )

    instruments = portfolio.instrument_map
    legs: list[tuple[str, EventLedger]] = []
    for symbol in portfolio.instrument_ids:
        weight = float(weights[symbol])
        if weight <= 0.0:
            continue
        leg_overrides = dict(overrides)
        leg_overrides["starting_balance"] = shared_balance * weight
        leg_config = replace(
            config,
            instrument=instruments[symbol],
            risk=replace(config.risk, sizing=sizing),
            engine_overrides=leg_overrides,
            portfolio=None,
        )
        result = adapter_cls().run(
            market_data[symbol],
            signal_map[symbol],
            leg_config,
            aux_data=_portfolio_aux_for(aux_data, symbol),
        )
        legs.append((symbol, result.ledger))

    if not legs:
        raise ConfigError(
            "every portfolio instrument has a zero allocation; nothing to run"
        )

    merged = _merge_portfolio_ledgers(legs)
    return BacktestResult.from_ledger(
        merged,
        config,
        market_data=market_data,
        instruments=instruments,
        fx_rates=_portfolio_fx_rates(config, market_data, instruments),
    )


def _optional_dependency_installed(module: str) -> bool:
    """Whether the optional engine package ``module`` is installed.

    The vectorbt/backtrader adapter modules import their engine lazily (guarding the
    import so ``ube`` stays importable without extras), so a successful adapter import
    does **not** prove the engine is present. Registration must check the dependency
    itself — otherwise ``"auto"`` would resolve to an engine whose package is missing and
    fail at run time instead of falling through to an installed engine.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def ensure_builtin_engines_registered() -> None:
    """Register the built-in engine adapters whose dependency is importable (§4.1).

    Nautilus, vectorbt and backtrader are implemented and depend on their respective
    *optional* packages (``nautilus-trader``, ``vectorbt`` and ``backtrader``).
    Registration is guarded so ``import ube`` never hard-requires any of them; an adapter
    whose optional dependency is missing is simply not registered (its ``run`` would raise
    ``EngineError`` if somehow invoked). Idempotent — re-registering the same class is a
    no-op — so calling this repeatedly is safe.
    """
    for name, module_name, dependency, class_name in (
        (
            "nautilus",
            "ube.adapters.nautilus_adapter.adapter",
            "nautilus_trader",
            "NautilusAdapter",
        ),
        (
            "vectorbt",
            "ube.adapters.vectorbt_adapter.adapter",
            "vectorbt",
            "VectorbtAdapter",
        ),
        (
            "backtrader",
            "ube.adapters.backtrader_adapter.adapter",
            "backtrader",
            "BacktraderAdapter",
        ),
    ):
        if not _optional_dependency_installed(dependency):
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        register_engine(name, getattr(module, class_name))


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


def _default_settlement_currency(config: BacktestConfig) -> BacktestConfig:
    """Return a config whose instrument has a settlement currency when omitted.

    The minimal ``ube.Instrument(symbol, asset_class=...)`` usage declares no settlement
    currency; the downstream ledger normalization (core) requires one. The fallback is
    the asset-class default from :func:`~ube.core.instrument.resolve_settlement_currency`
    (§4.7) — the single definition shared by the result builder and the adapters.
    """
    instrument = config.instrument
    if instrument.settlement_currency is not None:
        return config
    settled = resolve_settlement_currency(instrument)
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
    if isinstance(data, Mapping) and config.portfolio is not None:
        config = _resolve_portfolio_config(config)

    # Portfolio (dict-keyed) inputs are forwarded as-is; the adapter decides support
    # (§7.1). Benchmark comparison is a single-instrument concept (it lines up two
    # curves bar-for-bar), so it is only attached on the single-asset path below. The
    # experiment log, however, is unconditional (§4.8) and must record BOTH branches.
    data_reference: DataReference | PortfolioDataReference
    if isinstance(data, Mapping):
        # §7.2: restrict each instrument's bars to the declared date_range window before
        # the adapter sees them. A window that excludes every bar raises DataShapeError.
        if config.date_range is not None:
            start, end = config.date_range
            data = {k: _slice_market_data(v, start, end)[0] for k, v in data.items()}
        # Calendar validation (§4.4): each portfolio leg is checked against its *own*
        # declared calendar — keys that are Instruments (or bare calendar names) resolve
        # to their calendar; any other label carries no calendar and is skipped (no
        # declared calendar ⇒ no calendar constraint, the §4.5 "24/7"/None reading).
        for key, md in data.items():
            try:
                calendar = resolve_calendar(key)
            except InvalidInstrumentError:
                continue
            if calendar.is_always_open:
                continue
            validate_timestamps(md.timestamps, calendar)
        ensure_builtin_engines_registered()
        engine_name = resolve_engine_name(config.engine)
        adapter_cls = get_engine(engine_name)
        if config.portfolio is not None:
            # Shared-capital portfolio path: every leg is delegated to the adapter, but
            # sized from one pre-split pool, and the ledgers fold into one combined result.
            result = _run_portfolio_backtest(
                data, signals, config, aux_data=aux_data, adapter_cls=adapter_cls
            )
            reference_data = _normalize_portfolio_mapping(data)
            data_reference = PortfolioDataReference.from_market_data_mapping(
                reference_data, config.portfolio.instrument_map
            )
        else:
            # Legacy mapping path: forwarded as-is to a portfolio-capable adapter. The
            # single-instrument adapter contract types ``data``/``signals`` as
            # MarketData/Signals, so the adapter validates the mapping at runtime (§7.1).
            result = adapter_cls().run(data, signals, config, aux_data=aux_data)  # type: ignore[arg-type]
            # §4.8: a multi-instrument run is still a run — log it so the true attempt
            # count (PBO/DSR honesty, §11) is complete. Build the portfolio fingerprint.
            data_reference = PortfolioDataReference.from_market_data_mapping(data)
    else:
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

        # §10: the benchmark comparison (excess return, information ratio) assumes the
        # benchmark and strategy curves line up positionally, bar for bar. The adapter
        # already warm-up-slices the strategy outputs, so build the benchmark from the same
        # warmup-excluded bars — not the raw (or merely date_range-sliced) input.
        benchmark_md = _warmup_slice_market_data(md, config.warmup_bars)
        benchmark = build_benchmark(config.benchmark, benchmark_md)
        # Attach the benchmark to the result. Benchmark and strategy curves are
        # positionally aligned (both warm-up / date-range sliced), so the strategy
        # statistics can compare them (§10); ``ube`` computes no metrics of its own.
        # The adapter-built result already carries the return/trade/exposure tiers, so the
        # benchmark tier is computed now and merged — no ledger work is repeated.
        stats_view = result.stats
        if stats_view is not None:
            stats_view = stats_view.with_benchmark(
                benchmark,
                returns=result.equity_curve.returns,
                index=result.equity_curve.index,
            )
        result = replace(result, benchmark=benchmark, stats=stats_view)
        data_reference = DataReference.from_market_data(md, config.instrument)

    # §4.8: unconditional, every run — single-instrument or portfolio. This is reached on
    # both branches above (the early `return` was removed) so no run category is silently
    # exempt from logging.
    with ExperimentLog(path=log_path) as log:
        log.record(
            run_id=result.run_id,
            config=config,
            engine=engine_name,
            data_reference=data_reference,
            result_hash=result_hash(result),
        )
    return result

"""Top-level ``ube.run`` orchestrator (§4.8, §7.1).

:func:`run` is the single user-facing entry point that ties the Phase-1 core together
behind one call: it standardizes the inputs, resolves the engine, runs the adapter,
attaches the benchmark, and records the run to the experiment log — in that order. It is
deliberately a thin orchestrator: every piece of real work is a Phase-1 core function or
adapter, so the orchestration itself carries no engine-specific logic (§4.2).

The inputs mirror the canonical contract:

* ``data`` — any of the four shapes :meth:`~ube.core.data.MarketData.standardize`
  accepts (DataFrame / array / dict / records) or an already-standardized
  :class:`~ube.core.data.MarketData`.
* ``signals`` — a :class:`~ube.core.signals.Signals` or a 4-column signal
  :class:`pandas.DataFrame`; richer encodings (a ``-1/0/1`` target array or a per-bar
  callable) are converted *first* via :func:`~ube.core.signals.from_target` /
  :func:`~ube.core.signals.from_callable`, not guessed here.
* ``config`` — the full :class:`~ube.core.config.BacktestConfig`.

Engines are registered lazily (:func:`ensure_builtin_engines_registered`) so that
``import ube`` never hard-requires an optional engine dependency (§6 pyproject).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd
from numpy.typing import ArrayLike

from ube.adapters import get_engine, register_engine, resolve_engine_name
from ube.core.benchmark import build_benchmark
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError, InvalidSignalError
from ube.core.experiment_log import DataReference, ExperimentLog
from ube.core.result import BacktestResult, result_hash
from ube.core.signals import Signals

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


def run(
    data: object,
    signals: object,
    config: BacktestConfig,
    *,
    aux_data: Mapping[str, ArrayLike] | None = None,
    log_path: str | Path | None = None,
) -> BacktestResult:
    """Run one backtest end-to-end and return the canonical result (§4.8, §7.1).

    Orchestrates the Phase-1 pipeline in order:

    1. Validate ``config`` and standardize ``data``/``signals`` to the canonical
       containers.
    2. Lazily register the installed built-in engine, then resolve ``config.engine``
       (``"auto"`` → first installed) to the adapter class.
    3. Run the adapter, producing a :class:`~ube.core.result.BacktestResult`.
    4. Compute and attach the :class:`~ube.core.benchmark.BenchmarkCurve` (§10)
       engine-agnostically.
    5. Record the run to the experiment log (§4.8) — unconditional — with the
       :func:`~ube.core.result.result_hash` fingerprint.

    Args:
        data: The single-instrument OHLCV bars (any :meth:`MarketData.standardize`
            shape, or an already-standardized :class:`~ube.core.data.MarketData`).
        signals: A :class:`~ube.core.signals.Signals` or 4-column signal
            :class:`pandas.DataFrame`; build richer encodings with
            :func:`~ube.core.signals.from_target` /
            :func:`~ube.core.signals.from_callable` first.
        config: The full :class:`~ube.core.config.BacktestConfig`.
        aux_data: Named derived series (§5.2) passed to the engine adapter.
        log_path: Optional experiment-log path override (§4.8); ``None`` uses the normal
            precedence (explicit → ``BACKTEST_LOG_PATH`` → default).

    Returns:
        The frozen :class:`~ube.core.result.BacktestResult` with the benchmark attached.

    Raises:
        ConfigError: ``config`` is not a :class:`~ube.core.config.BacktestConfig`, or
            ``config.engine`` names no installed adapter.
        DataShapeError: ``data`` cannot be standardized to a :class:`MarketData`.
        InvalidSignalError: ``signals`` cannot be standardized to :class:`Signals`.
        EngineError: The underlying engine failed (wrapped, original preserved).
    """
    if not isinstance(config, BacktestConfig):
        raise ConfigError(f"run expects a BacktestConfig; got {type(config).__name__}")

    md = _standardize_data(data)
    sig = _standardize_signals(signals)

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

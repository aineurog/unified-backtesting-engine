"""vectorbt portfolio execution (mirrors ``actor.py`` for the Nautilus adapter).

vectorbt is *vectorized*: there is no per-bar actor. This module builds the
``vbt.Portfolio.from_signals`` call once from the translated signal frame (§6.1) and the
exit stop primitives (§4.2), wrapping any engine exception in
:class:`~ube.core.errors.EngineError`. The caller drives the two-pass sizing loop and folds
the resulting trade records into the canonical ledger.
"""

from __future__ import annotations

from typing import Any

from ube.adapters.vectorbt_adapter.adapt_data import VbtSignalInputs
from ube.core.errors import EngineError

__all__ = ["build_portfolio", "vbt"]

# vectorbt is an optional dependency (installed by the user). Guard the import so the module
# does not hard-fail when the package is absent; callers validate ``vbt is not None`` first.
try:  # pragma: no cover - exercised only when vectorbt is installed
    import vectorbt as vbt  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    vbt = None


def build_portfolio(
    signals: VbtSignalInputs,
    *,
    init_cash: float,
    fees: float,
    sl_stop: Any,
    tp_stop: Any,
    sl_trail: Any,
    size: Any,
) -> Any:
    """Call ``vbt.Portfolio.from_signals`` once and return the resulting portfolio.

    Args:
        signals: The translated signal frame from :func:`to_vbt_inputs`.
        init_cash: Starting cash (from the ``starting_balance`` override).
        fees: Per-trade fee fraction (commission + slippage from the cost model).
        sl_stop: Per-bar stop-loss fraction array/scalar, or ``None``.
        tp_stop: Take-profit fraction, or ``None``.
        sl_trail: Trailing-stop fraction, or ``None``.
        size: Per-bar ``size`` (amount) series, or ``1.0`` for the nominal pass.

    Returns:
        The ``vbt.Portfolio`` object (typed ``Any`` — vectorbt ships no stubs).

    Raises:
        EngineError: If vectorbt raises for any reason (wrapped with context).
    """
    if vbt is None:  # pragma: no cover - depends on the optional package
        raise EngineError(
            "vectorbt is not installed; install it (e.g. `pip install vectorbt`) "
            "to use the vectorbt engine"
        )
    try:
        return vbt.Portfolio.from_signals(
            close=signals.close,
            entries=signals.entries,
            exits=signals.long_exits,
            short_entries=signals.short_entries,
            short_exits=signals.short_exits,
            init_cash=init_cash,
            fees=fees,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            open=signals.open,
            high=signals.high,
            low=signals.low,
            size=size,
            size_type="amount",
            freq=signals.freq,
        )
    except Exception as exc:  # noqa: BLE001 - wrap whatever vectorbt raised
        raise EngineError(f"vectorbt backtest failed: {exc}") from exc

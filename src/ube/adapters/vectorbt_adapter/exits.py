"""Exit translation for the vectorbt engine (§5.2, §4.2).

vectorbt exits on its own stop primitives (``sl_stop`` / ``tp_stop`` / ``sl_trail``), so this
module translates the canonical exit configs into per-bar stop *fractions* and labels each
closed trade with the core ``exit_triggered`` semantics so a trade is stamped identically to
the Nautilus engine (requirements §4.6, §8). The ATR-based exits resolve their named
``aux_data`` series here, including the no-look-ahead forward-fill described in §5.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    atr,
    exit_triggered,
)
from ube.core.signals import Signals

__all__ = [
    "validate_aux",
    "atr_from_aux",
    "atr_series_for_exit",
    "exit_stop_params",
    "classify_exit_reason",
]


def validate_aux(exits: tuple[Any, ...], aux_data: Mapping[str, Any] | None) -> None:
    """Fail fast if an ATR-based exit names an ``aux_data`` series that is absent (§5.2)."""
    required: set[str] = set()
    for e in exits:
        if isinstance(e, (ATRStop, ChandelierExit)):
            atr_name = getattr(e, "atr", None)
            if atr_name is not None:
                required.add(str(atr_name))
    if not required:
        return
    if aux_data is None:
        raise ConfigError(
            "exits reference ATR series(es) "
            f"{sorted(required)} but aux_data was not supplied; "
            "ATR-based stops/chandelier require aux_data"
        )
    missing = required - set(aux_data)
    if missing:
        raise ConfigError(
            "exits reference ATR series(es) "
            f"{sorted(missing)} that are absent from aux_data; "
            "ATR-based stops/chandelier require the named aux_data series"
        )


def atr_from_aux(data: MarketData, aux_md: MarketData, period: int) -> np.ndarray:
    """Resolve a main-grid ATR series from a raw OHLCV aux ``MarketData`` (§5.2).

    ATR is computed on the (coarser) aux bars and forward-filled onto the main grid, then
    shifted one main bar so a main bar never sees the aux bar it is inside (no look-ahead).
    """
    atr_aux = atr(aux_md, period)
    aux_series = pd.Series(atr_aux, index=aux_md.timestamps)
    ffill = aux_series.reindex(data.timestamps, method="ffill")
    out: np.ndarray = ffill.to_numpy(dtype=np.float64)
    out = np.roll(out, 1)
    out[0] = np.nan
    first_valid = int(np.argmax(~np.isnan(out))) if np.any(~np.isnan(out)) else 0
    if first_valid < out.shape[0]:
        out[: first_valid + 1] = out[first_valid]
    out = np.where(np.isnan(out), 0.0, out)
    return out


def atr_series_for_exit(
    exit: ATRStop | ChandelierExit,
    aux_data: Mapping[str, Any] | None,
    data: MarketData,
) -> np.ndarray:
    """The main-grid ATR series for one ATR/Chandelier exit (§5.2)."""
    if exit.atr is None:
        return atr(data, exit.period)
    value = aux_data.get(exit.atr) if aux_data is not None else None
    if value is None:
        raise ConfigError(
            f"exit references aux_data[{exit.atr!r}] which was not supplied"
        )
    if isinstance(value, MarketData):
        return atr_from_aux(data, value, exit.period)
    series = np.asarray(value, dtype=np.float64)
    if series.ndim != 1 or series.shape[0] != data.n_bars:
        raise DataShapeError(
            f"aux_data[{exit.atr!r}] must be 1-D with length {data.n_bars}; "
            f"got shape {series.shape}"
        )
    return series


def exit_stop_params(
    exits: tuple[Any, ...], data: MarketData, aux_data: Mapping[str, Any] | None
) -> tuple[Any, Any, Any]:
    """Translate the exit configs into vectorbt ``sl_stop`` / ``tp_stop`` / ``sl_trail``.

    ``sl_stop`` is the tightest per-bar stop fraction over every stop-type exit (a fixed
    ``StopLoss`` fraction and any ATR/Chandelier fractions); ``tp_stop`` is the first
    ``TakeProfit``; ``sl_trail`` is the first ``TrailingStop``. ATR/Chandelier fractions are
    ``mult * atr / close`` — an approximation of the absolute core level (which is anchored to
    entry/running peak), accepted for the vectorized engine (§16 parity tolerance).
    """
    sl_scalar: float | None = None
    tp_stop: float | None = None
    trail_scalar: float | None = None
    # Fixed stops -> vbt ``sl_stop`` (anchored to entry, like StopLoss / ATRStop).
    fixed_fractions: list[np.ndarray] = []
    # Trailing stops -> vbt ``sl_trail`` (anchored to running peak, like TrailingStop /
    # ChandelierExit).
    trail_fractions: list[np.ndarray] = []

    for exit in exits:
        if isinstance(exit, TakeProfit):
            tp_stop = exit.percent if tp_stop is None else tp_stop
        elif isinstance(exit, StopLoss):
            sl_scalar = exit.percent if sl_scalar is None else sl_scalar
        elif isinstance(exit, TrailingStop):
            trail_scalar = exit.percent if trail_scalar is None else trail_scalar
        elif isinstance(exit, ATRStop):
            series = atr_series_for_exit(exit, aux_data, data)
            fixed_fractions.append(
                exit.mult * series / np.where(data.close > 0, data.close, np.nan)
            )
        elif isinstance(exit, ChandelierExit):
            series = atr_series_for_exit(exit, aux_data, data)
            trail_fractions.append(
                exit.mult * series / np.where(data.close > 0, data.close, np.nan)
            )

    # Combine fixed stops: per-bar minimum of the scalar StopLoss and every ATRStop series.
    sl_stop: Any = None
    if fixed_fractions:
        combined = fixed_fractions[0]
        for frac in fixed_fractions[1:]:
            combined = np.minimum(combined, frac)
        if sl_scalar is not None:
            combined = np.minimum(combined, sl_scalar)
        sl_stop = np.nan_to_num(combined, nan=0.0)
    elif sl_scalar is not None:
        sl_stop = sl_scalar

    # Combine trailing stops: per-bar minimum of the scalar TrailingStop and Chandelier series.
    sl_trail: Any = None
    if trail_fractions:
        combined = trail_fractions[0]
        for frac in trail_fractions[1:]:
            combined = np.minimum(combined, frac)
        if trail_scalar is not None:
            combined = np.minimum(combined, trail_scalar)
        sl_trail = np.nan_to_num(combined, nan=0.0)
    elif trail_scalar is not None:
        sl_trail = trail_scalar

    return sl_stop, tp_stop, sl_trail


def classify_exit_reason(
    exits: tuple[Any, ...],
    data: MarketData,
    side: int,
    entry_price: float,
    entry_bar: int,
    exit_bar: int,
    aux_data: Mapping[str, Any] | None,
    signals: Signals,
) -> str:
    """Label a closing fill using the core ``exit_triggered`` semantics (§4.6/§8)."""
    if side == 1 and bool(signals.long_exit[exit_bar]):
        return "signal"
    if side == -1 and bool(signals.short_exit[exit_bar]):
        return "signal"
    # A flip (opposite-side entry) also closes the position via a signal: a long exited on
    # the bar a short opens, or a short exited where a long opens. Nautilus journals these as
    # "signal"; without this branch they would be mislabeled "end_of_run".
    if side == 1 and bool(signals.short_entry[exit_bar]):
        return "signal"
    if side == -1 and bool(signals.long_entry[exit_bar]):
        return "signal"

    for exit in exits:
        if isinstance(exit, TakeProfit):
            if bool(
                exit_triggered(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                )[exit_bar]
            ):
                return "take_profit"
        elif isinstance(exit, ATRStop):
            series = atr_series_for_exit(exit, aux_data, data)
            if bool(
                exit_triggered(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                    atr_series=series,
                )[exit_bar]
            ):
                return "atr_stop"
        elif isinstance(exit, ChandelierExit):
            series = atr_series_for_exit(exit, aux_data, data)
            if bool(
                exit_triggered(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                    atr_series=series,
                )[exit_bar]
            ):
                return "chandelier"
        elif isinstance(exit, TrailingStop):
            if bool(
                exit_triggered(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                )[exit_bar]
            ):
                return "trailing_stop"
        elif isinstance(exit, StopLoss):
            if bool(
                exit_triggered(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                )[exit_bar]
            ):
                return "stop_loss"
        elif isinstance(exit, TimeExit) and bool(
            exit_triggered(
                exit,
                market_data=data,
                side=side,
                entry_price=entry_price,
                entry_bar=entry_bar,
            )[exit_bar]
        ):
            return "time_exit"
    return "end_of_run"

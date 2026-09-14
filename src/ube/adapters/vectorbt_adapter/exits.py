"""Exit translation for the vectorbt engine (§5.2, §4.2).

vectorbt exits on its own stop primitives (``sl_stop`` / ``tp_stop`` / ``sl_trail``) plus
the signal exit masks, so this module translates the canonical exit configs into per-bar
stop *fractions*, folds a holding-period ``TimeExit`` into the vectorbt exit masks (vectorbt
has no native time exit), and labels each closed trade with the core ``exit_triggered``
semantics so a trade is stamped identically to the Nautilus engine (requirements §4.6, §8).
The ATR-based exits resolve their named ``aux_data`` series here, including the
no-look-ahead forward-fill described in §5.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ube.adapters.vectorbt_adapter.adapt_data import VbtSignalInputs
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
    "resolve_vol_for_sizing",
    "apply_time_exits",
    "exit_stop_params",
    "classify_exit_reason",
]


def validate_aux(
    exits: tuple[Any, ...],
    aux_data: Mapping[str, Any] | None,
    *,
    sizing: Any | None = None,
    data: MarketData | None = None,
) -> None:
    """Fail fast if an ATR exit or ``volatility_target`` sizing has no usable series (§5.2, §6.3).

    Mirrors the Nautilus actor ``_validate_aux``: an ATR-based exit (``ATRStop`` /
    ``ChandelierExit``) must name a specific ``aux_data`` series via its ``atr`` key —
    the library never computes ATR from the signal ``data`` bars — and ``volatility_target``
    sizing must name one via ``SizeModel.vol``. Enforces that every named series is present.
    When ``data`` is supplied, a named ``MarketData`` aux series must also span the main
    period (start at or before it and reach within one aux bar of its end), so leading and
    trailing main bars always have a value.
    """
    named: set[str] = set()
    for e in exits:
        if isinstance(e, (ATRStop, ChandelierExit)):
            atr_name = getattr(e, "atr", None)
            if atr_name is None:
                raise ConfigError(
                    f"{type(e).__name__} requires an 'atr' key referencing an "
                    "aux_data series (§5.2); ATR is never computed from the signal "
                    "data bars"
                )
            named.add(str(atr_name))
    if sizing is not None and getattr(sizing, "kind", None) == "volatility_target":
        vol_name = getattr(sizing, "vol", None)
        if not vol_name:
            raise ConfigError(
                "volatility_target sizing requires a 'vol' key referencing an "
                "aux_data series (§6.3); volatility is never computed from the "
                "signal data bars"
            )
        named.add(str(vol_name))

    if not named:
        return
    if aux_data is None:
        raise ConfigError(
            "exits/sizing reference aux_data series(es) "
            f"{sorted(named)} but aux_data was not supplied; "
            "ATR-based stops/chandelier require aux_data"
        )
    missing = named - set(aux_data)
    if missing:
        raise ConfigError(
            "exits/sizing reference aux_data series(es) "
            f"{sorted(missing)} that are absent from aux_data; "
            "ATR-based stops/chandelier require the named aux_data series"
        )

    if data is not None:
        main_ts = data.timestamps
        main_start, main_end = main_ts[0], main_ts[-1]
        main_step = main_ts[1] - main_ts[0] if len(main_ts) > 1 else pd.Timedelta(0)
        for name in sorted(named):
            value = aux_data[name]
            if not isinstance(value, MarketData):
                continue  # precomputed arrays are length-checked at resolution
            aux_ts = value.timestamps
            if aux_ts[0] > main_start + main_step:
                raise DataShapeError(
                    f"aux_data[{name!r}] starts at {aux_ts[0]} which is after the data "
                    f"start {main_start}; the leading main bars would have no value "
                    "(aux_data must start at or before the signal/price period)"
                )
            aux_step = aux_ts[1] - aux_ts[0] if len(aux_ts) > 1 else main_step
            if aux_ts[-1] < main_end - aux_step:
                raise DataShapeError(
                    f"aux_data[{name!r}] ends at {aux_ts[-1]} which is more than one aux "
                    f"bar before the data end {main_end}; aux_data must span the "
                    "signal/price period (the trailing value would be stale)"
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


def resolve_vol_for_sizing(
    sizing: Any, aux_data: Mapping[str, Any] | None, data: MarketData
) -> np.ndarray:
    """Resolve the ``volatility_target`` per-bar vol estimate from ``aux_data`` (§6.3).

    Mirrors the Nautilus actor ``_vol_from_aux``: a ``MarketData`` value (a raw OHLCV
    series, typically coarser than the signal bars) is turned into ``ATR/price`` on the aux
    grid, shifted forward one aux bar so a main bar inside hour ``h`` only ever sees the
    volatility of the *last completed* aux bar (no look-ahead), then forward-filled onto the
    main bar grid. A precomputed array is used verbatim. Length and positivity are enforced
    here (fail fast); the name is validated up front in :func:`validate_aux`.
    """
    aux = aux_data if aux_data is not None else {}
    name = str(getattr(sizing, "vol", None))
    if name not in aux:
        raise ConfigError(f"sizing references aux_data[{name!r}] which was not supplied")
    value = aux[name]
    if isinstance(value, MarketData):
        vol_aux = atr(value) / value.close
        series = pd.Series(vol_aux, index=value.timestamps)
        series = series.shift(1).fillna(series.iloc[0])
        aligned = series.reindex(data.timestamps, method="ffill")
        arr = aligned.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != data.n_bars:
        raise DataShapeError(
            f"aux_data[{name!r}] has {arr.shape[0]} bars but data has {data.n_bars}"
        )
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ConfigError("vol estimate must be finite and positive")
    return arr


def _time_exit_mask(entries: np.ndarray, bars: int) -> np.ndarray:
    """Holding-period exit mask: fire at the first bar at/after ``entry_bar + bars``.

    Each fresh ``entries`` bar re-arms the countdown; after firing, the mask goes idle
    until a new entry (a subsequent stale ``True`` is a no-op for vectorbt — first-exit-wins).
    """
    n = entries.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    armed = False
    fire_at: int | None = None
    for i in range(n):
        if bool(entries[i]):
            armed = True
            fire_at = i + bars
        if armed and fire_at is not None and i >= fire_at:
            mask[i] = True
            armed = False
            fire_at = None
    return mask


def apply_time_exits(
    inputs: VbtSignalInputs, exits: tuple[Any, ...], data: MarketData
) -> VbtSignalInputs:
    """Fold ``TimeExit`` into the vectorbt exit masks so vectorbt actually exits (§6.4).

    vectorbt has no native holding-period primitive, so a ``TimeExit(bars)`` is translated
    into an exit-signal mask with the core :func:`time_exit_mask` semantics: a position
    opened at bar ``e`` exits at the first bar at or after ``e + bars``. The mask is applied
    to both the long and the short exit columns (a time exit closes whichever side is open);
    a signal exit or stop that already closed the position makes a later mask bar idle, and
    a new entry re-arms the countdown. Returns a (possibly mutated) copy of ``inputs``.
    """
    time_bars = [e.bars for e in exits if isinstance(e, TimeExit)]
    if not time_bars:
        return inputs
    n = data.n_bars
    long_mask = np.zeros(n, dtype=np.bool_)
    short_mask = np.zeros(n, dtype=np.bool_)
    for bars in time_bars:
        long_mask |= _time_exit_mask(inputs.entries.to_numpy(dtype=np.bool_), bars)
        short_mask |= _time_exit_mask(inputs.short_entries.to_numpy(dtype=np.bool_), bars)
    idx = inputs.close.index
    inputs.long_exits = inputs.long_exits | pd.Series(long_mask, index=idx, dtype=bool)
    inputs.short_exits = inputs.short_exits | pd.Series(short_mask, index=idx, dtype=bool)
    return inputs


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

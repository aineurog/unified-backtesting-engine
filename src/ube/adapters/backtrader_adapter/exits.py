"""Exit translation for the backtrader engine (§5.2, §4.2).

backtrader is an event-driven engine, so unlike vectorbt it does not exit through vectorized
stop primitives: the strategy checks its position bar-by-bar. This module therefore translates
the canonical exit configs into per-entry *per-bar trigger arrays* (via the core
``exit_triggered`` semantics of §4.7/§8), plus the ATR/volatility aux_data resolution shared
with the other adapters (§5.2, §6.3). The strategy recomputes the plan once at entry fill time
(anchored to the actual fill price and bar) and then checks the precomputed arrays on every
subsequent bar — the levels are causal (running peaks, EWM ATR), so there is no look-ahead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    atr,
    exit_triggered,
    scale_out_fraction,
)

__all__ = [
    "validate_aux",
    "atr_from_aux",
    "atr_series_for_exit",
    "resolve_vol_for_sizing",
    "resolve_atr_series_map",
    "BtExitLine",
    "build_exit_plan",
    "exit_reason_label",
]


def validate_aux(
    exits: tuple[Any, ...],
    aux_data: Mapping[str, Any] | None,
    *,
    sizing: Any | None = None,
    data: MarketData | None = None,
) -> None:
    """Fail fast if an ATR exit or ``volatility_target`` sizing has no usable series (§5.2, §6.3).

    Mirrors the vectorbt/nautilus adapters: an ATR-based exit (``ATRStop`` /
    ``ChandelierExit``) must name a specific ``aux_data`` series via its ``atr`` key — the
    library never computes ATR from the signal ``data`` bars — and ``volatility_target``
    sizing must name one via ``SizeModel.vol``. Enforces that every named series is present;
    a named ``MarketData`` aux series must also span the main period.
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

    Mirrors the vectorbt/nautilus adapters: a ``MarketData`` value is turned into
    ``ATR/price`` on the aux grid, shifted forward one aux bar so a main bar inside hour
    ``h`` only ever sees the volatility of the *last completed* aux bar (no look-ahead),
    then forward-filled onto the main bar grid. A precomputed array is used verbatim.
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


def resolve_atr_series_map(
    exits: tuple[Any, ...],
    aux_data: Mapping[str, Any] | None,
    data: MarketData,
) -> dict[int, np.ndarray]:
    """The main-grid ATR series per exit index, for the ATR/Chandelier exits that need one."""
    out: dict[int, np.ndarray] = {}
    for idx, exit in enumerate(exits):
        if isinstance(exit, (ATRStop, ChandelierExit)):
            out[idx] = atr_series_for_exit(exit, aux_data, data)
    return out


@dataclass(frozen=True)
class BtExitLine:
    """One exit's per-bar trigger plan, anchored to a real entry (§8).

    Attributes:
        index: The exit's position in the configured ``RiskConfig.exit`` tuple.
        fraction: The scale-out fraction this exit exits when it fires (§6.4).
        reason: The canonical reason string stamped on the closing fill.
        triggered: Causal per-bar bool array — ``True`` where this exit fires.
    """

    index: int
    fraction: float
    reason: str
    triggered: np.ndarray


def exit_reason_label(cfg: Exit) -> str:
    """The canonical §4.6 reason label for an exit config."""
    if isinstance(cfg, ATRStop):
        return "atr_stop"
    if isinstance(cfg, ChandelierExit):
        return "chandelier"
    if isinstance(cfg, TrailingStop):
        return "trailing_stop"
    if isinstance(cfg, TakeProfit):
        return "take_profit"
    if isinstance(cfg, StopLoss):
        return "stop_loss"
    if isinstance(cfg, TimeExit):
        return "time_exit"
    raise ConfigError(f"unknown exit type {type(cfg).__name__}")


def build_exit_plan(
    exits: tuple[Any, ...],
    data: MarketData,
    *,
    side: int,
    entry_price: float,
    entry_bar: int,
    atr_map: Mapping[int, np.ndarray] | None = None,
) -> tuple[BtExitLine, ...]:
    """Build the per-bar trigger plan for every configured exit, anchored to one entry (§8).

    Pure and causal: each line is the core ``exit_triggered`` result for the exit with this
    trade's side/entry reference. The strategy checks the plan arrays on every bar while the
    position is open; the first firing exit (in configured order) exits its ``fraction``.
    """
    atr_map = atr_map if atr_map is not None else {}
    out: list[BtExitLine] = []
    for idx, cfg in enumerate(exits):
        series = atr_map.get(idx)
        triggered = exit_triggered(
            cfg,
            market_data=data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=series,
        )
        out.append(
            BtExitLine(
                index=idx,
                fraction=float(scale_out_fraction(cfg)),
                reason=exit_reason_label(cfg),
                triggered=np.asarray(triggered, dtype=np.bool_),
            )
        )
    return tuple(out)
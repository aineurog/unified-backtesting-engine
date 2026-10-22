"""Resolution of caller-supplied auxiliary series shared by every engine (§5.2, §6.3).

ATR-based exits and ``volatility_target`` sizing never derive their series from the
signal bars: the caller names a series in ``aux_data``, and this module is the single
implementation of how that series is validated and aligned onto the main bar grid. All
engines (vectorbt, backtrader, nautilus, paper trading) call these helpers so they see
bit-for-bit identical ATR/volatility arrays.

The alignment rule is causal: a series from coarser aux bars is shifted forward one
**aux** bar before being forward-filled onto the main grid, so a main bar inside aux bar
``h`` only ever sees the value through ``h - 1`` (no look-ahead).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError
from ube.core.risk.exits import ATRStop, ChandelierExit, atr

__all__ = [
    "validate_aux",
    "atr_from_aux",
    "atr_series_for_exit",
    "resolve_vol_for_sizing",
]


def _ns_index(ts: Any) -> pd.Index:
    """A timestamp index normalized to int64 epoch-ns.

    ``MarketData`` indexes can be ``datetime64`` (at whatever unit pandas stored) or
    already int64 nanoseconds (the paper strategy's rebuilt grid). Reindexing only
    aligns them when both sides use the same representation.
    """
    # ``as_unit`` is present in pandas itself but typed inconsistently across the
    # pandas-stubs/numpy combinations CI resolves, so each use is conditional.
    if isinstance(ts, pd.DatetimeIndex):
        return pd.Index(ts.as_unit("ns").asi8)  # type: ignore[attr-defined, unused-ignore]
    idx = pd.Index(np.asarray(ts))
    if isinstance(idx, pd.DatetimeIndex):
        return pd.Index(idx.as_unit("ns").asi8)  # type: ignore[attr-defined, unused-ignore]
    if str(idx.dtype).startswith("datetime64"):
        dt = cast(pd.DatetimeIndex, idx)
        return pd.Index(dt.as_unit("ns").asi8)  # type: ignore[attr-defined, unused-ignore]
    return pd.Index(np.asarray(idx, dtype=np.int64))


def _align_aux(
    values: np.ndarray, aux_ts: Any, main_ts: Any
) -> np.ndarray:
    """Shift ``values`` one aux bar, then forward-fill them onto the main grid."""
    series = pd.Series(np.asarray(values, dtype=np.float64), index=_ns_index(aux_ts))
    if not len(series):
        return np.full(len(main_ts), np.nan)
    # Causal: aux bar h contributes from h + 1 onwards. Seeding the first aux value
    # keeps the leading main bars finite instead of NaN.
    series = series.shift(1).fillna(series.iloc[0])
    aligned = series.reindex(_ns_index(main_ts), method="ffill")
    return np.asarray(aligned.to_numpy(dtype=np.float64))


def validate_aux(
    exits: tuple[Any, ...],
    aux_data: Mapping[str, Any] | None,
    *,
    sizing: Any | None = None,
    data: MarketData | None = None,
) -> None:
    """Fail fast if an ATR exit or ``volatility_target`` sizing has no usable series (§5.2, §6.3).

    An ATR-based exit (``ATRStop`` / ``ChandelierExit``) must name a specific
    ``aux_data`` series via its ``atr`` key — the library never computes ATR from the
    signal ``data`` bars — and ``volatility_target`` sizing must name one via
    ``SizeModel.vol``. Enforces that every named series is present. When ``data`` is
    supplied, a named ``MarketData`` aux series must also span the main period (start
    at or before it and reach within one aux bar of its end), so leading and trailing
    main bars always have a value.
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

    ATR is computed on the (coarser) aux bars, shifted one aux bar so a main bar never
    sees the aux bar it is inside, then forward-filled onto the main grid.
    """
    return _align_aux(atr(aux_md, period), aux_md.timestamps, data.timestamps)


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
        arr = atr_from_aux(data, value, exit.period)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != data.n_bars:
        raise DataShapeError(
            f"aux_data[{exit.atr!r}] must be 1-D with length {data.n_bars}; "
            f"got shape {arr.shape}"
        )
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ConfigError("atr_series must be finite and positive")
    return arr


def resolve_vol_for_sizing(
    sizing: Any, aux_data: Mapping[str, Any] | None, data: MarketData
) -> np.ndarray:
    """Resolve the ``volatility_target`` per-bar vol estimate from ``aux_data`` (§6.3).

    A ``MarketData`` value is turned into ``ATR/price`` on the aux grid, shifted one aux
    bar so a main bar only sees the last completed aux bar (no look-ahead), then
    forward-filled onto the main bar grid. A precomputed array is used verbatim.
    """
    aux = aux_data if aux_data is not None else {}
    name = str(getattr(sizing, "vol", None))
    if name not in aux:
        raise ConfigError(f"sizing references aux_data[{name!r}] which was not supplied")
    value = aux[name]
    if isinstance(value, MarketData):
        arr = _align_aux(atr(value) / value.close, value.timestamps, data.timestamps)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != data.n_bars:
        raise DataShapeError(
            f"aux_data[{name!r}] has {arr.shape[0]} bars but data has {data.n_bars}"
        )
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ConfigError("vol estimate must be finite and positive")
    return arr

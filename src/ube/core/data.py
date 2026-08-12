"""Canonical ``MarketData`` container and the bar-agnostic bar model (§4.3, §5.1).

The library accepts bars in any of four widely-used shapes and standardizes them to one
internal form (one door in, one canonical form out — §5.1):

1. pandas ``DataFrame`` — the index or a column holds bar boundaries; columns hold OHLCV.
2. numpy array — 2-D, shape ``(n_bars, 5)``, columns ``[open, high, low, close, volume]``.
3. dict of columns — ``{"open": [...], "high": [...], ...}``.
4. list of records — ``[{"open": ..., "high": ...}, ...]``.

``column_map`` and ``timestamp_col`` let users keep their own column names. Everything
lands in a frozen ``MarketData`` with a tz-aware UTC ``DatetimeIndex`` (time bars) or an
integer ``RangeIndex`` (event bars), columns ``open/high/low/close/volume``, sorted and
unique.

This module only *standardizes* — it never fetches data and never resamples (the bar
type is metadata, not a resampling instruction — §4.3). Structural validation raises the
§15 ``DataError`` subtypes (chiefly ``DataShapeError``); semantic correctness
(adjustments, bad ticks) is the user's responsibility and is not validated (§5.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import cast

import numpy as np
import pandas as pd

from ube.core.errors import DataShapeError

__all__ = [
    "MarketData",
    "BAR_TYPES",
    "EVENT_BAR_TYPES",
    "OHLC_COLUMNS",
    "VOLUME_COLUMN",
    "OHLCV_COLUMNS",
]

# §4.3 — bar types are metadata on the data, not a resampling instruction.
BAR_TYPES: tuple[str, ...] = ("time", "volume", "dollar", "tick")
EVENT_BAR_TYPES: tuple[str, ...] = ("volume", "dollar", "tick")

OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN: str = "volume"
OHLCV_COLUMNS: tuple[str, ...] = OHLC_COLUMNS + (VOLUME_COLUMN,)

_DEFAULT_TIMESTAMP_COL: str = "timestamp"


def _validate_bar_type(bar_type: str) -> str:
    if bar_type not in BAR_TYPES:
        raise DataShapeError(
            f"unsupported bar_type={bar_type!r}; expected one of {BAR_TYPES}"
        )
    return bar_type


def _canonical_to_source(column_map: Mapping[str, str] | None) -> dict[str, str]:
    """Resolve canonical OHLCV names to the user's column names via ``column_map``.

    ``column_map`` maps user column name -> canonical name (``{"O": "open", ...}``).
    Any canonical name not mentioned in the map defaults to its own name.
    """
    if column_map is None:
        return {c: c for c in OHLCV_COLUMNS}
    source: dict[str, str] = {}
    for user_name, canonical in column_map.items():
        if canonical not in OHLCV_COLUMNS:
            raise DataShapeError(
                f"column_map maps {user_name!r} to unknown canonical column "
                f"{canonical!r}; expected one of {OHLCV_COLUMNS}"
            )
        source[canonical] = user_name
    for canonical in OHLCV_COLUMNS:
        source.setdefault(canonical, canonical)
    return source


def _coerce_numeric(values: object, name: str) -> np.ndarray:
    """Coerce a column to a fresh 1-D float64 array, raising ``DataShapeError``."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DataShapeError(f"column {name!r} must be 1-D; got shape {arr.shape}")
    try:
        return arr.astype(np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise DataShapeError(f"column {name!r} is not numeric") from exc


def _validate_ohlc(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> None:
    """Validate the structural OHLC invariants listed in §15, vectorized."""
    if np.isnan(open_).any() or np.isnan(high).any() or np.isnan(low).any() or np.isnan(
        close
    ).any():
        raise DataShapeError("NaN values in OHLC columns")
    if (open_ <= 0).any() or (high <= 0).any() or (low <= 0).any() or (close <= 0).any():
        raise DataShapeError("non-positive prices in OHLC columns")
    bad = high < low
    bad |= high < np.maximum(open_, close)
    bad |= low > np.minimum(open_, close)
    if bad.any():
        i = int(np.argmax(bad))
        raise DataShapeError(
            f"OHLC invariant violated at bar {i}: "
            f"open={open_[i]}, high={high[i]}, low={low[i]}, close={close[i]}"
        )


def _parse_timestamps(timestamps: object) -> pd.DatetimeIndex:
    """Build a ``DatetimeIndex`` from a timestamp sequence, preserving tz-awareness."""
    try:
        # pandas accepts any of Index/Series/ndarray/sequence here; the broad `object`
        # type is intentional because each factory hands off a different source type.
        return pd.DatetimeIndex(timestamps)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise DataShapeError("timestamps could not be parsed as datetimes") from exc


def _timestamp_from_columns(
    df: pd.DataFrame, timestamp_col: str | None
) -> pd.Series | None:
    """Resolve the timestamp column for columnar (dict/records) inputs."""
    name = timestamp_col if timestamp_col is not None else _DEFAULT_TIMESTAMP_COL
    if name in df.columns:
        return df[name]
    return None


@dataclass(frozen=True)
class MarketData:
    """Frozen, canonical bar container: OHLCV columns plus a bar index (§4.3, §5.1).

    Attributes:
        open: float64 array of open prices.
        high: float64 array of high prices.
        low: float64 array of low prices.
        close: float64 array of close prices.
        volume: float64 array of volumes (``NaN`` where volume was not provided).
        index: tz-aware UTC ``DatetimeIndex`` for time bars; integer ``RangeIndex`` for
            event bars.
        bar_type: One of ``"time"``, ``"volume"``, ``"dollar"``, ``"tick"``.
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    index: pd.Index
    bar_type: str

    def __post_init__(self) -> None:
        bar_type = _validate_bar_type(self.bar_type)

        open_ = _coerce_numeric(self.open, "open")
        high = _coerce_numeric(self.high, "high")
        low = _coerce_numeric(self.low, "low")
        close = _coerce_numeric(self.close, "close")
        n = open_.shape[0]
        for name, arr in (("high", high), ("low", low), ("close", close)):
            if arr.shape[0] != n:
                raise DataShapeError(
                    f"column {name!r} has {arr.shape[0]} rows, expected {n}"
                )
        volume = _coerce_numeric(self.volume, "volume")
        if volume.shape[0] != n:
            raise DataShapeError(f"column 'volume' has {volume.shape[0]} rows, expected {n}")

        _validate_ohlc(open_, high, low, close)

        index = self.index
        if len(index) != n:
            raise DataShapeError(f"index has {len(index)} entries, expected {n}")
        if not index.is_monotonic_increasing:
            raise DataShapeError("bar index is not sorted ascending")
        if index.has_duplicates:
            raise DataShapeError("bar index contains duplicate entries")
        if bar_type == "time":
            if not isinstance(index, pd.DatetimeIndex):
                raise DataShapeError("time bars require a DatetimeIndex")
            if index.tz is None:
                raise DataShapeError("time-bar timestamps must be tz-aware")
            if index.tz != UTC:
                index = index.tz_convert("UTC")
        elif index.inferred_type != "integer":
            raise DataShapeError("event-bar index must be an integer index")

        for arr in (open_, high, low, close, volume):
            arr.setflags(write=False)

        object.__setattr__(self, "open", open_)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "bar_type", bar_type)

    # -- properties -----------------------------------------------------------

    @property
    def n_bars(self) -> int:
        """Number of bars."""
        return int(self.open.shape[0])

    @property
    def is_time_bars(self) -> bool:
        """True if ``bar_type == "time"``."""
        return self.bar_type == "time"

    @property
    def timestamps(self) -> pd.DatetimeIndex | None:
        """The tz-aware UTC ``DatetimeIndex`` for time bars, else ``None``."""
        if self.bar_type == "time":
            return cast(pd.DatetimeIndex, self.index)
        return None

    # -- constructors ---------------------------------------------------------

    @classmethod
    def _construct(
        cls,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray | None,
        timestamps: object | None,
        bar_type: str,
    ) -> MarketData:
        bar_type = _validate_bar_type(bar_type)
        if volume is None:
            volume = np.full(len(open_), np.nan, dtype=np.float64)
        if bar_type in EVENT_BAR_TYPES:
            index: pd.Index = pd.RangeIndex(len(open_))
        else:
            if timestamps is None:
                raise DataShapeError(
                    "time bars require timestamps (a timestamp column, an index, "
                    "or timestamps=...)"
                )
            index = _parse_timestamps(timestamps)
        return cls(
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            index=index,
            bar_type=bar_type,
        )

    @classmethod
    def _extract(
        cls,
        df: pd.DataFrame,
        column_map: Mapping[str, str] | None,
        timestamps: object | None,
        bar_type: str,
    ) -> MarketData:
        source = _canonical_to_source(column_map)
        missing = [c for c in OHLC_COLUMNS if source[c] not in df.columns]
        if missing:
            raise DataShapeError(
                f"missing required columns: {[source[c] for c in missing]}"
            )
        open_ = df[source["open"]].to_numpy()
        high = df[source["high"]].to_numpy()
        low = df[source["low"]].to_numpy()
        close = df[source["close"]].to_numpy()
        volume = df[source["volume"]].to_numpy() if source["volume"] in df.columns else None
        return cls._construct(open_, high, low, close, volume, timestamps, bar_type)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        column_map: Mapping[str, str] | None = None,
        timestamp_col: str | None = None,
        bar_type: str = "time",
    ) -> MarketData:
        """Standardize a pandas ``DataFrame`` (§5.1 shape 1).

        ``timestamp_col`` names the column holding bar boundaries; when ``None``, the
        ``DataFrame`` index is used (relevant for time bars only — event bars ignore it).
        """
        if not isinstance(df, pd.DataFrame):
            raise DataShapeError("from_dataframe expects a pandas DataFrame")
        if timestamp_col is not None:
            if timestamp_col not in df.columns:
                raise DataShapeError(f"timestamp column {timestamp_col!r} not found")
            timestamps: object | None = df[timestamp_col]
        else:
            timestamps = df.index
        return cls._extract(df, column_map=column_map, timestamps=timestamps, bar_type=bar_type)

    @classmethod
    def from_array(
        cls,
        arr: object,
        *,
        timestamps: object | None = None,
        bar_type: str = "time",
    ) -> MarketData:
        """Standardize a 2-D numpy array of shape ``(n_bars, 5)`` (§5.1 shape 2).

        Columns are fixed: ``[open, high, low, close, volume]``. Time bars additionally
        require ``timestamps`` (a tz-aware datetime sequence).
        """
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[1] != 5:
            raise DataShapeError(
                "numpy input must be 2-D with shape (n_bars, 5) "
                "[open, high, low, close, volume]; "
                f"got shape {a.shape}"
            )
        return cls._construct(
            a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], timestamps, bar_type
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        column_map: Mapping[str, str] | None = None,
        timestamp_col: str | None = None,
        bar_type: str = "time",
    ) -> MarketData:
        """Standardize a dict of columns (§5.1 shape 3).

        Time bars take their timestamps from ``timestamp_col`` (default ``"timestamp"``).
        """
        if not isinstance(data, Mapping):
            raise DataShapeError("from_dict expects a mapping of column name -> values")
        try:
            df = pd.DataFrame(dict(data))
        except ValueError as exc:
            raise DataShapeError("columns have mismatched lengths") from exc
        timestamps = _timestamp_from_columns(df, timestamp_col)
        return cls._extract(df, column_map=column_map, timestamps=timestamps, bar_type=bar_type)

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, object]],
        *,
        column_map: Mapping[str, str] | None = None,
        timestamp_col: str | None = None,
        bar_type: str = "time",
    ) -> MarketData:
        """Standardize a list of record dicts (§5.1 shape 4).

        Time bars take their timestamps from ``timestamp_col`` (default ``"timestamp"``).
        """
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise DataShapeError("from_records expects a sequence of record dicts")
        try:
            df = pd.DataFrame(list(records))
        except ValueError as exc:
            raise DataShapeError("records could not be aligned into columns") from exc
        timestamps = _timestamp_from_columns(df, timestamp_col)
        return cls._extract(df, column_map=column_map, timestamps=timestamps, bar_type=bar_type)

    @classmethod
    def standardize(
        cls,
        data: object,
        *,
        column_map: Mapping[str, str] | None = None,
        timestamp_col: str | None = None,
        bar_type: str = "time",
    ) -> MarketData:
        """Dispatch on the input shape and standardize to a canonical ``MarketData``."""
        if isinstance(data, pd.DataFrame):
            return cls.from_dataframe(
                data, column_map=column_map, timestamp_col=timestamp_col, bar_type=bar_type
            )
        if isinstance(data, np.ndarray):
            return cls.from_array(data, bar_type=bar_type)
        if isinstance(data, Mapping):
            return cls.from_dict(
                data, column_map=column_map, timestamp_col=timestamp_col, bar_type=bar_type
            )
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return cls.from_records(
                data, column_map=column_map, timestamp_col=timestamp_col, bar_type=bar_type
            )
        raise DataShapeError(f"unsupported input type {type(data).__name__}")

    # -- helpers --------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Reconstruct a pandas ``DataFrame`` with canonical columns and index."""
        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            index=self.index,
        )

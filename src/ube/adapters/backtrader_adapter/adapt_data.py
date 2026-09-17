"""MarketData / Signals translation for the backtrader engine (§6.1).

backtrader consumes a ``pd.DataFrame`` through its ``PandasData`` feed, so this module converts
the canonical :class:`~ube.core.data.MarketData` and :class:`~ube.core.signals.Signals`
containers into the frame the feed loads (OHLCV plus the four entry/exit signal columns) and
exposes the bar-grid arithmetic used by the ledger fold and the carry step (§24).

The index passed to backtrader is made naive-UTC (backtrader's ``date2num`` round-trip is UTC
based, but the adapter never needs the engine's datetimes back — it captures the bar *index* of
every fill in the strategy, so the fold maps indices straight onto the canonical nanosecond
grid without a datetime conversion).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.signals import Signals

__all__ = [
    "bar_timestamps_ns",
    "bar_index",
    "step_timestamps",
    "bar_step_grid",
    "bar_notional",
    "bar_side",
    "BtSignalFrame",
    "to_signal_frame",
]


def bar_timestamps_ns(data: MarketData) -> np.ndarray:
    """The bar-boundary axis as int64 nanoseconds (matching :mod:`ube.core.ledger`)."""
    idx = pd.DatetimeIndex(data.timestamps)
    return cast(
        np.ndarray,
        idx.as_unit("ns").to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )


def bar_index(ts_ns: np.ndarray, dt: pd.Timestamp) -> int:
    """Map a tz-aware ``Timestamp`` to its bar index via bisection."""
    return int(np.searchsorted(ts_ns, int(dt.value)))


def step_timestamps(
    ts: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(timestamps, values)`` to a minimal ascending step series."""
    if ts.shape[0] == 0:
        return ts, values
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    values = values[order]
    unique = np.unique(ts)
    last = np.searchsorted(ts, unique, side="right") - 1
    return ts[last], values[last]


def bar_step_grid(
    step_ts: np.ndarray, step_value: np.ndarray, bar_ts: np.ndarray
) -> np.ndarray:
    """Forward-filled step value at each bar boundary (flat before the first change)."""
    if step_ts.shape[0] == 0:
        return np.zeros(bar_ts.shape[0], dtype=np.float64)
    idx = np.searchsorted(step_ts, bar_ts, side="right") - 1
    valid = idx >= 0
    return np.where(valid, step_value[np.clip(idx, 0, None)], 0.0)


def bar_notional(
    data: MarketData, step_ts: np.ndarray, step_value: np.ndarray, multiplier: float
) -> np.ndarray:
    """Per-bar open notional: ``abs(position) * close * contract_multiplier``."""
    pos = bar_step_grid(step_ts, step_value, bar_timestamps_ns(data))
    return cast(np.ndarray, np.abs(pos) * data.close * multiplier)


def bar_side(
    data: MarketData, step_ts: np.ndarray, step_value: np.ndarray
) -> np.ndarray:
    """Per-bar direction (``+1`` / ``-1`` / ``0``) from the position step series."""
    return cast(
        np.ndarray,
        np.sign(bar_step_grid(step_ts, step_value, bar_timestamps_ns(data))),
    )


class BtSignalFrame:
    """The backtrader-typed signal/price frame for one instrument (§6.1).

    ``index`` is the canonical tz-aware ``data.timestamps``; ``feed_frame`` is the pane
    handed to ``PandasData`` (naive-UTC index, OHLCV + four int signal columns).
    """

    index: pd.DatetimeIndex
    feed_frame: pd.DataFrame


def to_signal_frame(data: MarketData, signals: Signals) -> BtSignalFrame:
    """Translate canonical ``MarketData`` + ``Signals`` into the backtrader feed frame (§6.1).

    The four signal columns are taken verbatim from the canonical container
    (``long_entry`` / ``long_exit`` / ``short_entry`` / ``short_exit``) as ints
    (``True`` -> ``1``), so the feed exposes one int line per signal kind.
    """
    idx = pd.DatetimeIndex(data.timestamps)
    frame = pd.DataFrame(
        {
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume,
            "long_entry": (
                np.asarray(signals.long_entry, dtype=bool).astype(np.int64)
            ),
            "long_exit": np.asarray(signals.long_exit, dtype=bool).astype(np.int64),
            "short_entry": (
                np.asarray(signals.short_entry, dtype=bool).astype(np.int64)
            ),
            "short_exit": np.asarray(signals.short_exit, dtype=bool).astype(np.int64),
        },
        index=idx,
    )
    out = BtSignalFrame()
    out.index = idx.as_unit("ns")
    out.feed_frame = frame.set_index(idx.tz_convert("UTC").tz_localize(None))
    return out
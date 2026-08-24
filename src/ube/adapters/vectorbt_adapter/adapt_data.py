"""MarketData / Signals translation for the vectorbt engine (§6.1).

vectorbt consumes plain pandas Series aligned to the bar index, so this module converts the
canonical :class:`~ube.core.data.MarketData` and :class:`~ube.core.signals.Signals` containers
into the exact Series it expects (close/open/high/low plus the four entry/exit columns) and
exposes the bar-grid arithmetic used by the ledger fold and the carry step (§24).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.signals import Signals

__all__ = [
    "VbtSignalInputs",
    "bar_timestamps_ns",
    "bar_index",
    "step_timestamps",
    "bar_step_grid",
    "bar_notional",
    "bar_side",
    "to_vbt_inputs",
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


class VbtSignalInputs:
    """The vectorbt-typed signal frame for one instrument (§6.1)."""

    close: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    freq: str | None


def to_vbt_inputs(data: MarketData, signals: Signals) -> VbtSignalInputs:
    """Translate canonical ``MarketData`` + ``Signals`` into vectorbt pandas Series (§6.1).

    All Series share the bar index (``data.timestamps``); ``freq`` is inferred for
    ``vbt.Portfolio.from_signals``. The four signal columns are taken verbatim from the
    canonical container (``long_entry`` / ``long_exit`` / ``short_entry`` / ``short_exit``).
    """
    idx = data.timestamps
    out = VbtSignalInputs()
    out.close = pd.Series(data.close, index=idx)
    out.open = pd.Series(data.open, index=idx)
    out.high = pd.Series(data.high, index=idx)
    out.low = pd.Series(data.low, index=idx)
    out.entries = pd.Series(signals.long_entry, index=idx)
    out.long_exits = pd.Series(signals.long_exit, index=idx)
    out.short_entries = pd.Series(signals.short_entry, index=idx)
    out.short_exits = pd.Series(signals.short_exit, index=idx)
    out.freq = pd.infer_freq(idx)
    return out

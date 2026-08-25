"""Unit tests for the date_range slicing helpers in ``ube.run`` (§7.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.core.errors import DataShapeError
from ube.run import (
    _as_utc_timestamp,
    _slice_aux_data,
    _slice_market_data,
    _window_mask,
)


def _bars(n: int = 6, start: str = "2024-01-01") -> MarketData:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    prices = np.arange(10.0, 10.0 + n)
    return MarketData(
        open=prices,
        high=prices + 1.0,
        low=prices - 1.0,
        close=prices,
        volume=np.ones(n),
        index=idx,
    )


def test_as_utc_timestamp_localizes_naive_and_converts_aware():
    assert _as_utc_timestamp("2024-03-01").tzinfo is not None
    assert _as_utc_timestamp("2024-03-01 00:00+02:00") == pd.Timestamp("2024-02-29 22:00", tz="UTC")


def test_window_mask_inclusive_bounds():
    md = _bars(5)  # 00:00 .. 04:00
    mask = _window_mask(md.index, "2024-01-01 01:00", "2024-01-01 03:00")
    assert mask.tolist() == [False, True, True, True, False]


def test_window_mask_open_bounds():
    md = _bars(5)
    lo = _window_mask(md.index, "2024-01-01 02:00", None)
    assert lo.tolist() == [False, False, True, True, True]
    hi = _window_mask(md.index, None, "2024-01-01 02:00")
    assert hi.tolist() == [True, True, True, False, False]


def test_slice_market_data_returns_subset_and_mask():
    md = _bars(6)
    sliced, mask = _slice_market_data(md, "2024-01-01 02:00", "2024-01-01 04:00")
    assert len(sliced.index) == 3
    assert sliced.index[0] == pd.Timestamp("2024-01-01 02:00", tz="UTC")
    assert mask.sum() == 3


def test_slice_market_data_empty_raises_data_shape_error():
    md = _bars(6)
    with pytest.raises(DataShapeError, match="excludes every bar"):
        _slice_market_data(md, "2030-01-01", "2030-02-01")


def test_slice_aux_data_slices_arrays_and_marketdata_by_window():
    md = _bars(6)
    aux = {
        "atr": np.arange(6.0),
        "vol": _bars(6),
    }
    start, end = "2024-01-01 02:00", "2024-01-01 04:00"
    _, mask = _slice_market_data(md, start, end)
    out = _slice_aux_data(aux, mask, start, end)
    assert out["atr"].tolist() == [2.0, 3.0, 4.0]
    assert len(out["vol"].index) == 3
    assert out["vol"].index[0] == pd.Timestamp("2024-01-01 02:00", tz="UTC")

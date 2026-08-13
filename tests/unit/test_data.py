"""Tests for the ``MarketData`` container and the four input standardizations (§4.3, §5.1)."""

from dataclasses import FrozenInstanceError
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.core.errors import DataError, DataShapeError


def _ohlcv_df(n: int = 5, tz: str | None = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz=tz)
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 100.0 + n, n),
            "high": np.linspace(101.0, 101.0 + n, n),
            "low": np.linspace(99.0, 99.0 + n, n),
            "close": np.linspace(100.5, 100.5 + n, n),
            "volume": np.arange(1, n + 1, dtype=float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# The four input formats standardize to the same canonical form.
# ---------------------------------------------------------------------------


def test_from_dataframe_time_bars():
    df = _ohlcv_df(5)
    md = MarketData.from_dataframe(df)
    assert md.n_bars == 5
    assert isinstance(md.index, pd.DatetimeIndex)
    assert md.index.equals(df.index)
    assert md.timestamps is not None
    np.testing.assert_allclose(md.open, df["open"].to_numpy())
    np.testing.assert_allclose(md.close, df["close"].to_numpy())


def test_from_dataframe_column_map_and_timestamp_col():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "O": [1.0, 2.0, 3.0],
            "H": [1.5, 2.5, 3.5],
            "L": [0.5, 1.5, 2.5],
            "C": [1.2, 2.2, 3.2],
            "V": [10.0, 20.0, 30.0],
        }
    )
    md = MarketData.from_dataframe(
        df,
        column_map={"O": "open", "H": "high", "L": "low", "C": "close", "V": "volume"},
        timestamp_col="ts",
    )
    assert md.n_bars == 3
    assert md.index.equals(pd.DatetimeIndex(df["ts"]))
    np.testing.assert_allclose(md.open, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(md.volume, [10.0, 20.0, 30.0])


def test_from_array_with_timestamps():
    arr = np.array(
        [
            [1.0, 1.5, 0.5, 1.2, 100.0],
            [1.2, 1.8, 1.0, 1.6, 80.0],
            [1.6, 2.0, 1.4, 1.9, 120.0],
        ]
    )
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    md = MarketData.from_array(arr, timestamps=ts)
    assert md.n_bars == 3
    assert isinstance(md.index, pd.DatetimeIndex)
    assert md.index.equals(ts)
    np.testing.assert_allclose(md.volume, [100.0, 80.0, 120.0])


def test_from_dict_with_timestamps():
    md = MarketData.from_dict(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC"),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 20.0],
        },
    )
    assert md.n_bars == 2
    assert isinstance(md.index, pd.DatetimeIndex)
    np.testing.assert_allclose(md.close, [1.2, 2.2])


def test_from_records_time_bars():
    records = [
        {
            "timestamp": "2024-01-01 09:00:00+00:00",
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": 1.2,
            "volume": 10.0,
        },
        {
            "timestamp": "2024-01-01 09:05:00+00:00",
            "open": 1.2,
            "high": 1.8,
            "low": 1.0,
            "close": 1.6,
            "volume": 20.0,
        },
    ]
    md = MarketData.from_records(records)
    assert md.n_bars == 2
    assert md.timestamps is not None
    assert md.index[0] == pd.Timestamp("2024-01-01 09:00:00+00:00")
    np.testing.assert_allclose(md.open, [1.0, 1.2])


def test_standardize_dispatches():
    df = _ohlcv_df(3)
    assert MarketData.standardize(df).n_bars == 3
    arr = np.array([[1.0, 1.5, 0.5, 1.2, 10.0]] * 2)
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    assert MarketData.standardize(arr, timestamps=ts).n_bars == 2
    d = {
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC"),
        "open": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.2],
    }
    assert MarketData.standardize(d).n_bars == 2
    recs = [
        {
            "timestamp": "2024-01-01 00:00:00+00:00",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
        }
    ]
    assert MarketData.standardize(recs).n_bars == 1


# ---------------------------------------------------------------------------
# Volume is optional for some bar types (§4.3).
# ---------------------------------------------------------------------------


def test_volume_optional_defaults_to_nan():
    df = _ohlcv_df(3).drop(columns=["volume"])
    md = MarketData.from_dataframe(df)
    assert np.isnan(md.volume).all()


def test_volume_nan_is_allowed():
    df = _ohlcv_df(3)
    df.loc[df.index[1], "volume"] = np.nan
    md = MarketData.from_dataframe(df)
    assert np.isnan(md.volume[1])


# ---------------------------------------------------------------------------
# No resampling: bars are preserved exactly (§4.3).
# ---------------------------------------------------------------------------


def test_no_resampling_preserves_irregular_time_bars():
    idx = pd.to_datetime(
        ["2024-01-01 09:00", "2024-01-01 09:07", "2024-01-01 09:13", "2024-01-01 10:00"]
    ).tz_localize("UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [1.0, 2.0, 3.0, 4.0],
        },
        index=idx,
    )
    md = MarketData.from_dataframe(df)
    assert md.n_bars == 4
    assert md.index.equals(idx)


# ---------------------------------------------------------------------------
# Structural validation raises the correct DataError subtype (§15).
# ---------------------------------------------------------------------------


def test_unsorted_index_raises_data_shape_error():
    df = _ohlcv_df(4).sort_index(ascending=False)
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_duplicate_index_raises_data_shape_error():
    df = _ohlcv_df(3)
    df.index = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-01", "2024-01-02"], tz="UTC"
    )
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_tz_naive_timestamps_raise_data_shape_error():
    df = _ohlcv_df(3)
    df.index = df.index.tz_localize(None)
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_tz_aware_timestamps_are_normalized_to_utc():
    df = _ohlcv_df(3, tz="America/New_York")
    md = MarketData.from_dataframe(df)
    assert md.index.tz == UTC
    assert md.index.equals(df.index.tz_convert("UTC"))


@pytest.mark.parametrize("col", ["open", "high", "low", "close"])
def test_nan_ohlc_raises_data_shape_error(col):
    df = _ohlcv_df(3)
    df.loc[df.index[1], col] = np.nan
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_high_below_low_raises_data_shape_error():
    df = _ohlcv_df(3)
    df.loc[df.index[1], "low"] = 1000.0  # low > high
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_high_below_max_open_close_raises_data_shape_error():
    df = _ohlcv_df(3)
    df.loc[df.index[1], "high"] = 50.0  # below both open and close
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_low_above_min_open_close_raises_data_shape_error():
    df = _ohlcv_df(3)
    df.loc[df.index[1], "low"] = 200.0  # above both open and close
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


@pytest.mark.parametrize("col", ["open", "high", "low", "close"])
def test_non_positive_price_raises_data_shape_error(col):
    df = _ohlcv_df(3)
    df.loc[df.index[1], col] = 0.0
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_missing_required_column_raises_data_shape_error():
    df = _ohlcv_df(3).drop(columns=["close"])
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_wrong_numpy_shape_raises_data_shape_error():
    with pytest.raises(DataShapeError):
        MarketData.from_array(np.zeros((3, 3)))


def test_from_array_four_columns_volume_defaults_to_nan():
    arr = np.array([[1.0, 1.5, 0.5, 1.2], [1.2, 1.8, 1.0, 1.6]])
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    md = MarketData.from_array(arr, timestamps=ts)
    assert md.n_bars == 2
    assert np.isnan(md.volume).all()


def test_from_array_requires_timestamps():
    arr = np.array([[1.0, 1.5, 0.5, 1.2, 10.0]] * 2)
    with pytest.raises(DataShapeError):
        MarketData.from_array(arr)


def test_empty_records_raise_clear_error():
    with pytest.raises(DataShapeError, match="no bars"):
        MarketData.from_records([])


def test_empty_dataframe_raises_clear_error():
    with pytest.raises(DataShapeError, match="no bars"):
        MarketData.from_dataframe(_ohlcv_df(0))


def test_empty_dict_raises_clear_error():
    with pytest.raises(DataShapeError, match="no bars"):
        MarketData.from_dict({})


def test_non_numeric_column_raises_data_shape_error():
    df = _ohlcv_df(3)
    df["close"] = ["a", "b", "c"]
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(df)


def test_time_bars_from_dict_without_timestamp_raises_data_shape_error():
    with pytest.raises(DataShapeError):
        MarketData.from_dict(
            {"open": [1, 2], "high": [1.5, 2.5], "low": [0.5, 1.5], "close": [1.2, 2.2]}
        )


def test_unknown_column_map_target_raises_data_shape_error():
    with pytest.raises(DataShapeError):
        MarketData.from_dataframe(_ohlcv_df(3), column_map={"O": "not_a_column"})


def test_unsupported_input_type_raises_data_shape_error():
    with pytest.raises(DataShapeError):
        MarketData.standardize(42)  # type: ignore[arg-type]


def test_data_shape_error_is_a_data_error():
    assert issubclass(DataShapeError, DataError)


# ---------------------------------------------------------------------------
# Immutability (§3 principle 5).
# ---------------------------------------------------------------------------


def test_marketdata_is_frozen():
    md = MarketData.from_dataframe(_ohlcv_df(3))
    with pytest.raises(FrozenInstanceError):
        md.close = np.zeros(3)  # type: ignore[misc]


def test_ohlcv_arrays_are_read_only():
    md = MarketData.from_dataframe(_ohlcv_df(3))
    with pytest.raises(ValueError):
        md.close[0] = 999.0


def test_to_dataframe_round_trips():
    df = _ohlcv_df(4)
    md = MarketData.from_dataframe(df)
    out = md.to_dataframe()
    assert out.index.equals(df.index)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    np.testing.assert_allclose(out["close"].to_numpy(), df["close"].to_numpy())

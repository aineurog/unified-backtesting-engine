"""Unit tests for the Nautilus adapter's aux_data → ATR-series resolution (§5.2)."""

import numpy as np
import pytest

from ube.adapters.nautilus_adapter.adapter import _resolve_atr_series
from ube.core.errors import ConfigError
from ube.core.risk.exits import ATRStop, ChandelierExit, TakeProfit


def test_no_atr_exits_resolves_to_none() -> None:
    assert _resolve_atr_series((), None) is None
    assert _resolve_atr_series((TakeProfit(percent=0.05),), None) is None


def test_unnamed_atr_resolves_to_none() -> None:
    # An ATR exit without an ``atr`` name auto-computes from its own ``period``.
    assert _resolve_atr_series((ATRStop(mult=2.0),), None) is None


def test_named_atr_present_resolves_the_series() -> None:
    series = np.array([1.0, 2.0, 3.0])
    result = _resolve_atr_series((ATRStop(mult=2.0, atr="atr_12h"),), {"atr_12h": series})
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, series)


def test_named_atr_absent_falls_back_to_none() -> None:
    assert _resolve_atr_series((ATRStop(mult=2.0, atr="atr_12h"),), None) is None
    assert _resolve_atr_series((ATRStop(mult=2.0, atr="atr_12h"),), {}) is None
    assert _resolve_atr_series((ATRStop(mult=2.0, atr="atr_12h"),), {"other": [1.0]}) is None


def test_same_name_across_exits_is_a_single_series() -> None:
    series = np.array([5.0, 6.0])
    exits = (ATRStop(mult=2.0, atr="atr"), ChandelierExit(mult=3.0, atr="atr"))
    result = _resolve_atr_series(exits, {"atr": series})
    np.testing.assert_array_equal(result, series)


def test_multiple_distinct_names_raise() -> None:
    exits = (ATRStop(mult=2.0, atr="a"), ChandelierExit(mult=3.0, atr="b"))
    with pytest.raises(ConfigError):
        _resolve_atr_series(exits, {"a": [1.0], "b": [2.0]})


def test_non_1d_series_raises() -> None:
    with pytest.raises(ConfigError):
        _resolve_atr_series(
            (ATRStop(mult=2.0, atr="atr"),),
            {"atr": np.zeros((3, 2))},
        )


def test_array_like_is_coerced_to_ndarray() -> None:
    result = _resolve_atr_series((ATRStop(mult=2.0, atr="atr"),), {"atr": [1, 2, 3]})
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

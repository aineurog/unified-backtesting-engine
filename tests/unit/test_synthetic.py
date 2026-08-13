"""Tests for the deterministic synthetic-data generator (§16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.testing.synthetic import (
    DEFAULT_N_BARS,
    PRESETS,
    AssetClassPreset,
    synthetic_bars,
)

CANONICAL = ("stocks", "futures", "commodities", "forex", "crypto_perp")
OHLCV = ("open", "high", "low", "close", "volume")


def test_presets_cover_canonical_asset_classes():
    assert set(PRESETS) == set(CANONICAL)


def test_presets_carry_an_instrument():
    for name, preset in PRESETS.items():
        assert isinstance(preset, AssetClassPreset)
        assert preset.instrument.asset_class == name


def test_contract_multiplier_presets():
    assert PRESETS["futures"].instrument.contract_multiplier == 50.0
    assert PRESETS["commodities"].instrument.contract_multiplier == 100.0
    assert PRESETS["stocks"].instrument.contract_multiplier is None
    assert PRESETS["forex"].instrument.contract_multiplier is None


def test_synthetic_bars_is_deterministic():
    a = synthetic_bars("stocks", seed=42)
    b = synthetic_bars("stocks", seed=42)
    for name in OHLCV:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    pd.testing.assert_index_equal(a.index, b.index)


def test_synthetic_bars_differs_by_seed():
    a = synthetic_bars("stocks", seed=1)
    b = synthetic_bars("stocks", seed=2)
    assert not np.array_equal(a.close, b.close)


def test_synthetic_bars_accepts_preset_object():
    a = synthetic_bars(PRESETS["forex"], seed=5)
    b = synthetic_bars("forex", seed=5)
    np.testing.assert_array_equal(a.close, b.close)


def test_synthetic_bars_shape_and_index():
    md = synthetic_bars("futures", seed=0)
    assert isinstance(md, MarketData)
    assert md.n_bars == DEFAULT_N_BARS
    assert isinstance(md.index, pd.DatetimeIndex)
    assert md.index.tz is not None
    assert md.index.is_monotonic_increasing
    assert not md.index.has_duplicates
    assert md.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_synthetic_bars_ohlc_invariants():
    for name in CANONICAL:
        md = synthetic_bars(name, seed=7)
        assert (md.high >= md.open).all() and (md.high >= md.close).all(), name
        assert (md.low <= md.open).all() and (md.low <= md.close).all(), name
        assert (md.high >= md.low).all(), name
        assert (md.close > 0).all(), name


def test_synthetic_bars_volume_positive():
    md = synthetic_bars("stocks", seed=3)
    assert (md.volume > 0).all()


def test_synthetic_bars_respects_tick_size():
    md = synthetic_bars("futures", seed=11)  # ES tick = 0.25
    tick = PRESETS["futures"].instrument.tick_size
    assert tick == 0.25
    for name in ("open", "high", "low", "close"):
        arr = getattr(md, name)
        np.testing.assert_allclose(arr, np.round(arr / tick) * tick, rtol=1e-12, atol=1e-12)


def test_synthetic_bars_unknown_asset_class_raises():
    with pytest.raises(ValueError):
        synthetic_bars("bonds")

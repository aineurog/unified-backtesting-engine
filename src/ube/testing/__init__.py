"""Deterministic synthetic-data generators for adapter-parity tests (§16).

This sub-package ships the seeded generator behind the per-asset-class fixture
data of §16, so tests (and end users) can reproduce the exact OHLCV bars the
parity baselines are built on.
"""

from ube.testing.synthetic import (
    DEFAULT_N_BARS,
    DEFAULT_SEED,
    DEFAULT_START,
    PRESETS,
    AssetClassPreset,
    synthetic_bars,
)

__all__ = [
    "AssetClassPreset",
    "DEFAULT_N_BARS",
    "DEFAULT_SEED",
    "DEFAULT_START",
    "PRESETS",
    "synthetic_bars",
]

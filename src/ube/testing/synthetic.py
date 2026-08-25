"""Deterministic synthetic bar generator for adapter-parity tests (§16).

The parity layer (§16) needs the *same* market data to flow through every engine
adapter, so that any disagreement between adapters is attributable to the adapter
— never to the data. This module produces that data.

Two properties matter and are guaranteed here:

* **Determinism.** Bars are drawn from ``np.random.default_rng(seed)`` (PCG64).
  A pinned seed is platform-stable and reproducible, unlike ``time()`` or the
  module-global RNG. The same ``(asset_class, seed, n_bars, start, freq)`` always
  yields byte-identical OHLCV. The RNG call *order* is fixed in code, so adding a
  call later would change the stream — the committed parquet fixtures (§16) are
  the backstop that catches such a drift.
* **Validity.** Output is a :class:`~ube.core.data.MarketData`, so it satisfies
  the §15 structural checks (positive, NaN-free, ``high >= max(open, close)``,
  ``low <= min(open, close)``) on a regular tz-aware UTC grid. Bars are a
  geometric-Brownian-motion path with per-bar drift/vol derived from the preset's
  *annual* volatility, rounded to the instrument's ``tick_size`` (high rounds up,
  low rounds down) so ticks are respected without ever violating the invariants.

The per-asset-class presets (:data:`PRESETS`) cover the five canonical fixture
classes of §16 — ``stocks``/AAPL, ``futures``/ES, ``commodities``/GC,
``forex``/EURUSD, ``crypto_perp``/BTC-USDT — each carrying the ``Instrument``
metadata (``contract_multiplier``, ``calendar``, ``settlement_currency``) the
adapters need, so one call produces both the bars and the instrument they belong
to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ube.core.calendar import resolve_calendar
from ube.core.data import MarketData
from ube.core.instrument import Instrument

__all__ = [
    "AssetClassPreset",
    "DEFAULT_N_BARS",
    "DEFAULT_SEED",
    "DEFAULT_START",
    "PRESETS",
    "synthetic_bars",
]

#: Canonical generator seed. The committed fixtures of §16 use this seed.
DEFAULT_SEED: int = 0

#: Canonical bar count — one month (31 days) of hourly bars, i.e. January 2024.
DEFAULT_N_BARS: int = 744

#: Canonical start timestamp (naive; localized to UTC by :func:`synthetic_bars`).
DEFAULT_START: str = "2024-01-01"

#: Canonical bar frequency.
DEFAULT_FREQ: str = "1h"

#: Seconds in a year (Julian) — the denominator for annualized vol/drift.
_SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class AssetClassPreset:
    """Generation parameters + ``Instrument`` metadata for one asset class (§16).

    Attributes:
        instrument: The :class:`~ube.core.instrument.Instrument` the bars belong
            to — carries ``contract_multiplier``/``calendar``/``settlement_currency``
            etc. for the adapter layer.
        start_price: The path's starting (and first bar's open) price.
        annual_vol: Annualized volatility (fraction); converted to a per-bar sigma
            from ``bar_freq``.
        drift: Annualized drift (fraction); defaults to ``0.0``.
        bar_freq: Bar frequency for the default grid (e.g. ``"1h"``).
    """

    instrument: Instrument
    start_price: float
    annual_vol: float
    drift: float = 0.0
    bar_freq: str = DEFAULT_FREQ


#: The five canonical asset-class presets of §16, keyed by ``asset_class``.
PRESETS: dict[str, AssetClassPreset] = {
    "stocks": AssetClassPreset(
        instrument=Instrument(
            "AAPL",
            "stocks",
            tick_size=0.01,
            calendar="XNYS",
            settlement_currency="USD",
        ),
        start_price=190.0,
        annual_vol=0.30,
    ),
    "futures": AssetClassPreset(
        instrument=Instrument(
            "ES",
            "futures",
            tick_size=0.25,
            contract_multiplier=50.0,
            calendar="CMES",
            settlement_currency="USD",
        ),
        start_price=5000.0,
        annual_vol=0.15,
    ),
    "commodities": AssetClassPreset(
        instrument=Instrument(
            "GC",
            "commodities",
            tick_size=0.1,
            contract_multiplier=100.0,
            calendar="CMES",
            settlement_currency="USD",
        ),
        start_price=2000.0,
        annual_vol=0.18,
    ),
    "forex": AssetClassPreset(
        instrument=Instrument(
            "EURUSD",
            "forex",
            tick_size=0.0001,
            calendar="24/7",
            settlement_currency="USD",
        ),
        start_price=1.08,
        annual_vol=0.08,
    ),
    "crypto_perp": AssetClassPreset(
        instrument=Instrument(
            "BTC-USDT",
            "crypto_perp",
            tick_size=0.1,
            calendar="24/7",
            settlement_currency="USDT",
        ),
        start_price=60000.0,
        annual_vol=0.60,
    ),
}


def synthetic_bars(
    asset_class: str | AssetClassPreset,
    *,
    seed: int = DEFAULT_SEED,
    n_bars: int = DEFAULT_N_BARS,
    start: str | pd.Timestamp = DEFAULT_START,
    freq: str | None = None,
) -> MarketData:
    """Generate deterministic synthetic OHLCV bars for ``asset_class`` (§16).

    Produces a geometric-Brownian-motion price path (drift + annual vol from the
    preset), opens equal to the previous close, an intra-bar high/low spread, and
    lognormal volume — all drawn from a seeded PCG64 RNG, then rounded to the
    instrument's ``tick_size``. The result is a valid
    :class:`~ube.core.data.MarketData` on a regular tz-aware UTC grid.

    Args:
        asset_class: An ``AssetClassPreset`` or one of its keys in :data:`PRESETS`.
        seed: RNG seed; identical seeds yield identical bars.
        n_bars: Number of bars to generate.
        start: First bar's timestamp (naive or tz-aware; localized to UTC).
        freq: Bar frequency (e.g. ``"1h"``); defaults to the preset's ``bar_freq``.

    Returns:
        A :class:`~ube.core.data.MarketData` with ``n_bars`` bars.
    """
    preset = _resolve_preset(asset_class)
    freq = freq if freq is not None else preset.bar_freq
    rng = np.random.default_rng(seed)

    bars_per_year = _bars_per_year(freq)
    sigma = preset.annual_vol / np.sqrt(bars_per_year)
    mu = preset.drift / bars_per_year

    # Log-normal GBM path (fixed RNG order — part of the determinism contract).
    log_returns = rng.normal(mu - 0.5 * sigma * sigma, sigma, size=n_bars)
    close = preset.start_price * np.exp(np.cumsum(log_returns))
    open_ = np.empty_like(close)
    open_[0] = preset.start_price
    open_[1:] = close[:-1]

    # Intra-bar high/low spread (0.05%–2% of the open/close extremes).
    range_frac = rng.uniform(0.0005, 0.02, size=n_bars)
    raw_high = np.maximum(open_, close) * (1.0 + range_frac)
    raw_low = np.minimum(open_, close) * (1.0 - range_frac)
    high = np.maximum(raw_high, np.maximum(open_, close))
    low = np.minimum(raw_low, np.minimum(open_, close))

    tick = preset.instrument.tick_size
    if tick is not None:
        open_ = _round_to_tick(open_, tick)
        close = _round_to_tick(close, tick)
        # High rounds up, low rounds down, then re-clamp so the invariants
        # (high >= max(open, close), low <= min(open, close)) are never broken.
        high = np.maximum(_round_to_tick(high, tick, "up"), np.maximum(open_, close))
        low = np.minimum(_round_to_tick(low, tick, "down"), np.minimum(open_, close))

    volume = rng.lognormal(mean=11.0, sigma=1.0, size=n_bars)
    index = _utc_index(start, n_bars, freq)
    # Sessioned instruments (stocks/futures/commodities) must only carry bars that fall
    # inside the declared trading calendar — §4.4 requires the data to respect it. Filter
    # the 24/7 grid to in-session timestamps, padding the candidate window so we still
    # return exactly ``n_bars`` bars. Prices/volumes/OHLC are RNG-derived independently of
    # the index, so this only changes timestamps, never the price path.
    cal = resolve_calendar(preset.instrument)
    if not cal.is_always_open:
        days = math.ceil(n_bars / 6.5) + 90
        candidate = pd.date_range(start=start, periods=days * 24, freq=freq)
        candidate = (
            candidate.tz_localize("UTC")
            if candidate.tz is None
            else candidate.tz_convert("UTC")
        )
        mask = cal.session_mask(candidate)
        index = candidate[mask][:n_bars]

    return MarketData(open=open_, high=high, low=low, close=close, volume=volume, index=index)


def _resolve_preset(asset_class: str | AssetClassPreset) -> AssetClassPreset:
    """Return the preset for ``asset_class``, passing a preset through unchanged."""
    if isinstance(asset_class, AssetClassPreset):
        return asset_class
    if not isinstance(asset_class, str) or asset_class not in PRESETS:
        raise ValueError(
            f"unknown asset_class {asset_class!r}; expected one of {sorted(PRESETS)}"
        )
    return PRESETS[asset_class]


def _bars_per_year(freq: str) -> float:
    """Number of ``freq`` bars in a Julian year."""
    seconds = pd.Timedelta(freq).total_seconds()
    if seconds <= 0:
        raise ValueError(f"invalid bar frequency {freq!r}")
    return _SECONDS_PER_YEAR / seconds


def _round_to_tick(values: np.ndarray, tick: float, mode: str = "nearest") -> np.ndarray:
    """Round ``values`` to multiples of ``tick`` (``nearest``/``up``/``down``)."""
    scaled = values / tick
    if mode == "up":
        rounded = np.ceil(scaled - 1e-9)
    elif mode == "down":
        rounded = np.floor(scaled + 1e-9)
    else:
        rounded = np.round(scaled)
    return rounded * tick


def _utc_index(start: str | pd.Timestamp, n_bars: int, freq: str) -> pd.DatetimeIndex:
    """Build a regular tz-aware UTC ``DatetimeIndex`` for the bars."""
    index = pd.date_range(start=start, periods=n_bars, freq=freq)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")

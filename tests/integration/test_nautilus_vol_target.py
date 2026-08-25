"""Nautilus adapter ``volatility_target`` sizing / aux_data resolution tests (§6.3).

``SizeModel(kind="volatility_target", ...)`` must name an ``aux_data`` series via its
``vol`` key, supplying the per-bar volatility estimate (a dimensionless fraction of
price). Volatility is never computed from the signal ``data`` bars — those may be
non-time bars or a too-fine timeframe (§5.2). A missing ``vol`` reference, or a name
absent from ``aux_data``, is a *config* error (``ConfigError``) surfaced up-front in
validation (fail-fast, §3) — never an ``EngineError`` mid-run (§15).
"""

import numpy as np
import pandas as pd
import pytest

from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError, EngineError
from ube.core.risk import RiskConfig, SizeModel
from ube.core.signals import from_target
from ube.testing.synthetic import PRESETS

_DATA_ROWS = [
    "2024-01-01 00:00:00+00:00,100.00,101.00,99.50,100.50",
    "2024-01-01 01:00:00+00:00,100.50,102.00,99.00,99.50",
    "2024-01-01 02:00:00+00:00,99.50,100.00,95.00,95.50",
    "2024-01-01 03:00:00+00:00,95.50,96.00,94.00,94.50",
]


def _data() -> MarketData:
    arrays = {"open": [], "high": [], "low": [], "close": []}
    timestamps = []
    for row in _DATA_ROWS:
        parts = row.split(",")
        timestamps.append(parts[0])
        for i, col in enumerate(("open", "high", "low", "close"), start=1):
            arrays[col].append(float(parts[i]))
    return MarketData(
        open=np.asarray(arrays["open"]),
        high=np.asarray(arrays["high"]),
        low=np.asarray(arrays["low"]),
        close=np.asarray(arrays["close"]),
        volume=np.full(len(_DATA_ROWS), 100.0),
        index=pd.DatetimeIndex(timestamps).as_unit("ns"),
    )


def _instrument():
    return PRESETS["stocks"].instrument


def _config(vol: str | None) -> BacktestConfig:
    return BacktestConfig(
        instrument=_instrument(),
        risk=RiskConfig(sizing=SizeModel(kind="volatility_target", value=0.01, vol=vol)),
    )


# ---------------------------------------------------------------------------
# Missing `vol` reference: config error (§15 / §3 fail-fast), never auto-compute.
# ---------------------------------------------------------------------------


def test_volatility_target_no_vol_raises_config_error_not_engine_error():
    with pytest.raises(ConfigError) as exc:
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 0, 0]),
            _config(vol=None),
            aux_data=None,
        )
    assert not isinstance(exc.value, EngineError)
    assert "vol" in str(exc.value)


def test_volatility_target_named_vol_missing_raises_config_error():
    with pytest.raises(ConfigError):
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 0, 0]),
            _config(vol="vol_1d"),
            aux_data=None,
        )


# ---------------------------------------------------------------------------
# Named `vol` reference resolves: precomputed series + raw MarketData.
# ---------------------------------------------------------------------------


def test_named_vol_series_resolves_verbatim():
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 0, 0]),
        _config(vol="vol_1d"),
        aux_data={"vol_1d": np.full(data.n_bars, 0.02)},
    )
    assert len(result.trades) == 1


def test_named_vol_marketdata_computes_vol_internally():
    # A raw MarketData aux value is turned into ATR/price internally (§6.3) — the
    # escape hatch for a coarser-timeframe volatility than the signal bars.
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 0, 0]),
        _config(vol="vol_1d"),
        aux_data={"vol_1d": data},
    )
    assert len(result.trades) == 1

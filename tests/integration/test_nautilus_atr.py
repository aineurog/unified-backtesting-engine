"""Nautilus adapter ATR / aux_data resolution tests (§5.2).

Covers the ATR auto-compute fallback: an ATR-based exit (``ATRStop`` / ``ChandelierExit``)
with no ``atr`` key must compute ATR internally from price data present in ``aux_data``
rather than erroring. The missing-input case is a *config* error (``ConfigError``),
surfaced up-front in validation (fail-fast, §3) — never an ``EngineError`` mid-run (§15).
"""

import numpy as np
import pandas as pd
import pytest

from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError, EngineError
from ube.core.risk import ATRStop, ChandelierExit, RiskConfig
from ube.core.risk.exits import atr
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


# ---------------------------------------------------------------------------
# §5.2 auto-compute fallback: no `atr` key, price data supplied in aux_data.
# ---------------------------------------------------------------------------


def test_atr_stop_no_key_auto_computes_from_aux_price_data():
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2))),
        aux_data={"price": data},
    )
    assert len(result.trades) >= 1
    assert result.trades[0].exit_reason == "atr_stop"


def test_chandelier_no_key_auto_computes_from_aux_price_data():
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ChandelierExit(2))),
        aux_data={"price": data},
    )
    assert len(result.trades) >= 1


# ---------------------------------------------------------------------------
# Missing input: config error (§15 / §3 fail-fast), NOT an engine error.
# ---------------------------------------------------------------------------


def test_atr_stop_no_key_no_aux_raises_config_error_not_engine_error():
    with pytest.raises(ConfigError) as exc:
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 1, 1]),
            BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2))),
            aux_data=None,
        )
    assert not isinstance(exc.value, EngineError)
    assert "aux_data" in str(exc.value)


def test_named_atr_missing_raises_config_error():
    with pytest.raises(ConfigError):
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 1, 1]),
            BacktestConfig(
                instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2, atr="atr_14"))
            ),
            aux_data=None,
        )


# ---------------------------------------------------------------------------
# Named `atr` key still resolves verbatim (precomputed series path).
# ---------------------------------------------------------------------------


def test_named_atr_series_resolves_verbatim():
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(
            instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2, atr="atr_14"))
        ),
        aux_data={"atr_14": atr(data, 14)},
    )
    assert len(result.trades) >= 1
    assert result.trades[0].exit_reason == "atr_stop"

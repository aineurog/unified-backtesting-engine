"""Nautilus adapter ATR / aux_data resolution tests (§5.2).

An ATR-based exit (``ATRStop`` / ``ChandelierExit``) must name an ``aux_data`` series
via its ``atr`` key. ATR is never computed from the signal ``data`` bars — those may be
non-time bars (which cannot be resampled to a time-based ATR) or a too-fine timeframe.
A missing ``atr`` reference, or a name absent from ``aux_data``, is a *config* error
(``ConfigError``) surfaced up-front in validation (fail-fast, §3) — never an
``EngineError`` mid-run (§15).
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
# Missing `atr` reference: config error (§15 / §3 fail-fast), never auto-compute.
# ---------------------------------------------------------------------------


def test_atr_stop_no_key_raises_config_error_not_engine_error():
    with pytest.raises(ConfigError) as exc:
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 1, 1]),
            BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2))),
            aux_data=None,
        )
    assert not isinstance(exc.value, EngineError)
    assert "aux_data" in str(exc.value)


def test_chandelier_no_key_raises_config_error():
    with pytest.raises(ConfigError):
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 1, 1]),
            BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ChandelierExit(2))),
            aux_data=None,
        )


def test_atr_stop_no_key_raises_config_error_even_with_price_aux_data():
    # Supplying an *unrelated* named series must not silently satisfy the exit: the
    # `atr` reference is mandatory (§5.2), and the library never guesses which aux
    # entry to use.
    with pytest.raises(ConfigError):
        NautilusAdapter().run(
            _data(),
            from_target([1, 1, 1, 1]),
            BacktestConfig(instrument=_instrument(), risk=RiskConfig(exit=ATRStop(2))),
            aux_data={"price": _data()},
        )


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
# Named `atr` reference resolves: precomputed series + raw MarketData.
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


def test_named_atr_marketdata_computes_atr_internally():
    # A raw MarketData aux value is turned into ATR internally by the adapter (§5.2) —
    # the escape hatch for a coarser-timeframe ATR than the signal bars.
    data = _data()
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(
            instrument=_instrument(),
            risk=RiskConfig(exit=ATRStop(2, atr="atr_1h")),
        ),
        aux_data={"atr_1h": data},
    )
    assert len(result.trades) >= 1
    assert result.trades[0].exit_reason == "atr_stop"

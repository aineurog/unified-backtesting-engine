"""Unit tests for the §6.1 execution-timing contract.

Signals are decision-time; ``execution_timing`` chooses the bar they fill on. These
tests pin the pure helpers, the config validation, and the end-to-end rule that all
three engines honour the same schedule (same price, same execution timestamp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.execution import (
    EXECUTION_TIMINGS,
    decision_index,
    execution_index,
    execution_price,
    is_valid_execution_timing,
    shift_signals,
    validate_execution_timing,
)
from ube.core.signals import Signals, from_target
from ube.testing.synthetic import PRESETS


def _signals(**columns: list[bool]) -> Signals:
    n = max(len(v) for v in columns.values())
    empty = [False] * n
    return Signals(
        long_entry=np.array(columns.get("long_entry", empty), dtype=bool),
        long_exit=np.array(columns.get("long_exit", empty), dtype=bool),
        short_entry=np.array(columns.get("short_entry", empty), dtype=bool),
        short_exit=np.array(columns.get("short_exit", empty), dtype=bool),
    )


def _md(n: int = 6) -> MarketData:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = np.arange(1, n + 1, dtype=float) * 100.0
    return MarketData(
        open=base,
        high=base + 5.0,
        low=base - 5.0,
        close=base + 2.5,
        volume=np.full(n, 1000.0),
        index=idx,
    )


# ---------------------------------------------------------------------------
# contract constants + config validation
# ---------------------------------------------------------------------------


def test_contract_values_and_default() -> None:
    assert EXECUTION_TIMINGS == ("next_open", "next_close", "same_close")
    assert BacktestConfig(instrument=PRESETS["crypto_perp"].instrument).execution_timing == (
        "next_open"
    )


@pytest.mark.parametrize("value", [None, "", "open", "next_Open", 1, "same-bar"])
def test_invalid_timing_rejected(value: object) -> None:
    assert not is_valid_execution_timing(value)
    with pytest.raises(ConfigError, match="execution_timing"):
        validate_execution_timing(value)
    with pytest.raises(ConfigError, match="execution_timing"):
        BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            execution_timing=value,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# index / price schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("timing", "expected"),
    [("same_close", {0: 0, 1: 1, 4: 4, 5: 5}), ("next_open", {0: 1, 1: 2, 4: 5, 5: None}),
     ("next_close", {0: 1, 1: 2, 4: 5, 5: None})],
)
def test_execution_index(timing: str, expected: dict[int, int | None]) -> None:
    for decision, exec_idx in expected.items():
        assert execution_index(decision, timing, n_bars=6) == exec_idx  # type: ignore[arg-type]


def test_execution_index_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        execution_index(-1, "next_open", n_bars=6)
    with pytest.raises(IndexError):
        execution_index(6, "same_close", n_bars=6)


def test_decision_index_is_the_inverse() -> None:
    assert decision_index(0, "same_close") == 0
    assert decision_index(0, "next_open") is None
    assert decision_index(3, "next_open") == 2
    assert decision_index(3, "next_close") == 2


def test_execution_price_uses_open_for_next_open_and_close_otherwise() -> None:
    md = _md()
    assert execution_price(md, 1, "next_open") == pytest.approx(md.open[2])
    assert execution_price(md, 1, "next_close") == pytest.approx(md.close[2])
    assert execution_price(md, 1, "same_close") == pytest.approx(md.close[1])
    assert execution_price(md, md.n_bars - 1, "next_open") is None


# ---------------------------------------------------------------------------
# shift_signals
# ---------------------------------------------------------------------------


def test_shift_signals_moves_each_column_one_bar() -> None:
    sig = _signals(
        long_entry=[True, False, True, False],
        long_exit=[False, True, False, False],
        short_entry=[False, False, False, True],
        short_exit=[False, False, True, False],
    )
    out = shift_signals(sig, "next_open")
    assert list(out.long_entry) == [False, True, False, True]
    assert list(out.long_exit) == [False, False, True, False]
    assert list(out.short_entry) == [False, False, False, False]  # last drops past the end
    assert list(out.short_exit) == [False, False, False, True]


def test_shift_signals_same_close_is_identity() -> None:
    sig = _signals(long_entry=[True, False, True])
    out = shift_signals(sig, "same_close")
    assert out is sig


def test_shift_signals_single_bar_drops_everything() -> None:
    out = shift_signals(_signals(long_entry=[True]), "next_open")
    assert out.long_entry.tolist() == [False]


# ---------------------------------------------------------------------------
# end-to-end: every engine honours the configured schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timing", EXECUTION_TIMINGS)
def test_all_engines_honour_execution_timing(timing: str) -> None:
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter
    from ube.core.risk import RiskConfig
    from ube.core.risk.sizing import SizeModel

    md = _md()
    # Decisions: long entry on bar 1, exit on bar 3.
    signals = from_target(np.array([0, 1, 1, 0, 0, 0]))
    config = BacktestConfig(
        instrument=PRESETS["crypto_perp"].instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        execution_timing=timing,  # type: ignore[arg-type]
        engine_overrides={"starting_balance": 100_000.0},
    )

    if timing == "next_open":
        entry_bar, exit_bar = 2, 4
        entry_px, exit_px = md.open[2], md.open[4]
    elif timing == "next_close":
        entry_bar, exit_bar = 2, 4
        entry_px, exit_px = md.close[2], md.close[4]
    else:  # same_close
        entry_bar, exit_bar = 1, 3
        entry_px, exit_px = md.close[1], md.close[3]

    results = [
        adapter().run(md, signals, config)
        for adapter in (VectorbtAdapter, NautilusAdapter, BacktraderAdapter)
    ]
    for result in results:
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(entry_px, rel=1e-12)
        assert trade.exit_price == pytest.approx(exit_px, rel=1e-12)
        assert trade.entry_timestamp == int(md.timestamps.as_unit("ns").asi8[entry_bar])
        assert trade.exit_timestamp == int(md.timestamps.as_unit("ns").asi8[exit_bar])
    assert results[0].trades[0] == results[1].trades[0] == results[2].trades[0]

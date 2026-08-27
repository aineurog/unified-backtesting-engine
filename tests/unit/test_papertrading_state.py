"""Unit tests for ``ube.papertrading.state`` (T1 / §9.5 / §18)."""

from __future__ import annotations

import numpy as np
import pytest

from ube.core.ledger import EventLedger, EventType, LedgerEvent
from ube.core.errors import StateCorruptionError
from ube.papertrading.state import OpenPosition, PaperState


def _sample_state() -> PaperState:
    ledger = EventLedger(
        [
            LedgerEvent(EventType.FILL, 1_000, "BTC-USDT", side=1, quantity=1.0, price=100.0),
            LedgerEvent(EventType.FILL, 2_000, "BTC-USDT", side=-1, quantity=1.0, price=110.0, exit_reason="signal"),
            LedgerEvent(EventType.COMMISSION, 1_000, "BTC-USDT", amount=0.05, currency="USDT"),
        ]
    )
    return PaperState(
        instrument_id="BTC-USDT",
        ledger=ledger,
        last_processed_ns=2_000,
        open_position=OpenPosition(side=1, quantity=1.0, entry_price=100.0, entry_ns=1_000, trade_id="t1"),
        pending_levels={"t1": {"tp": 120.0}},
        signal_fn_state={"targets": [1, 1, 0]},
        config_ref="run-1",
    )


def test_save_load_round_trip(tmp_path) -> None:
    path = tmp_path / "state.db"
    state = _sample_state()
    state.save(str(path), run_id="run-1")

    loaded = PaperState.load(str(path), run_id="run-1")
    assert loaded.instrument_id == "BTC-USDT"
    assert loaded.last_processed_ns == 2_000
    assert loaded.config_ref == "run-1"
    assert loaded.pending_levels == {"t1": {"tp": 120.0}}
    assert loaded.signal_fn_state == {"targets": [1, 1, 0]}
    assert loaded.open_position == state.open_position
    events = loaded.ledger.events
    assert len(events) == 3
    assert events[0].event_type is EventType.FILL
    assert events[0].price == 100.0
    assert events[2].amount == 0.05


def test_load_missing_file(tmp_path) -> None:
    with pytest.raises(StateCorruptionError):
        PaperState.load(str(tmp_path / "nope.db"), run_id="x")


def test_load_missing_row(tmp_path) -> None:
    path = tmp_path / "state.db"
    _sample_state().save(str(path), run_id="run-1")
    with pytest.raises(StateCorruptionError):
        PaperState.load(str(path), run_id="other")


def test_load_corrupt_ledger(tmp_path) -> None:
    path = tmp_path / "state.db"
    state = _sample_state()
    state.save(str(path), run_id="run-1")
    # Corrupt the ledger blob directly.
    import sqlite3

    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("UPDATE paper_state SET ledger = 'not json' WHERE run_id = 'run-1'")
    conn.close()
    with pytest.raises(StateCorruptionError):
        PaperState.load(str(path), run_id="run-1")


def test_empty_ledger_round_trip(tmp_path) -> None:
    path = tmp_path / "state.db"
    state = PaperState(instrument_id="ETH-USDT", ledger=EventLedger(), last_processed_ns=None)
    state.save(str(path))
    loaded = PaperState.load(str(path))
    assert loaded.instrument_id == "ETH-USDT"
    assert loaded.last_processed_ns is None
    assert len(loaded.ledger) == 0
    assert loaded.open_position is None

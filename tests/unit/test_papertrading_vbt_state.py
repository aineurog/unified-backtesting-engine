"""Unit tests for the vectorbt paper state + derivations (§4/§5 of the vbt paper plan).

Covers the validated checkpoint formula (including the same-bar flip and the funding
exclusion), the window-start rules, the incremental fill scan / checkpoint memo, and the
engine-tag guard on ``VbtPaperState.load``.
"""

from __future__ import annotations

import pytest

from ube.core.cost import CostModel, fill_cost
from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType, LedgerEvent
from ube.papertrading.core import get_state_class
from ube.papertrading.state import OpenPosition, PaperState
from ube.papertrading.vbt.state import (
    VbtPaperState,
    checkpoint_balance,
    last_close_ns,
    window_start_ns,
)

CM = CostModel(commission=0.0005, funding=0.0001)
IID = "BTC-USDT"
START = 10_000.0
MULT = 1.0


def _cash(ts: int, amount: float) -> LedgerEvent:
    return LedgerEvent(EventType.CASH_MOVEMENT, ts, IID, amount=amount, currency="USDT")


def _fill(ts: int, side: int, qty: float, price: float) -> LedgerEvent:
    return LedgerEvent(EventType.FILL, ts, IID, side=side, quantity=qty, price=price)


def _comm(ts: int, notional: float) -> LedgerEvent:
    return LedgerEvent(
        EventType.COMMISSION, ts, IID, amount=float(fill_cost(CM, notional=notional)),
        currency="USDT",
    )


def _funding(ts: int, amount: float) -> LedgerEvent:
    return LedgerEvent(EventType.FUNDING_PAYMENT, ts, IID, amount=amount, currency="USDT")


# ---------------------------------------------------------------------------
# last_close_ns
# ---------------------------------------------------------------------------


def test_last_close_none_until_a_close() -> None:
    ledger = EventLedger([_fill(10, 1, 1.0, 100.0)])
    assert last_close_ns(ledger, IID) is None


def test_last_close_on_flat_close() -> None:
    ledger = EventLedger([_fill(10, 1, 1.0, 100.0), _fill(20, -1, 1.0, 110.0)])
    assert last_close_ns(ledger, IID) == 20


def test_last_close_on_flip() -> None:
    # A flip is two fills on the same bar: close the long, open the short.
    ledger = EventLedger(
        [
            _fill(10, 1, 1.0, 100.0),
            _fill(20, -1, 1.0, 110.0),
            _fill(20, -1, 1.0, 110.0),
        ]
    )
    assert last_close_ns(ledger, IID) == 20


def test_last_close_only_when_position_is_flattened() -> None:
    # A scale-in (bar 15) followed by a partial reduce (bar 20) leaves the position open:
    # there is no close until the running position returns to / crosses zero.
    partial = EventLedger(
        [_fill(10, 1, 1.0, 100.0), _fill(15, 1, 1.0, 101.0), _fill(20, -1, 1.0, 110.0)]
    )
    assert last_close_ns(partial, IID) is None
    full = EventLedger(
        [_fill(10, 1, 1.0, 100.0), _fill(15, 1, 1.0, 101.0), _fill(20, -1, 2.0, 110.0)]
    )
    assert last_close_ns(full, IID) == 20


# ---------------------------------------------------------------------------
# window_start_ns
# ---------------------------------------------------------------------------


def test_window_start_open_is_entry_bar() -> None:
    open_pos = OpenPosition(side=1, quantity=1.0, entry_price=100.0, entry_ns=500, trade_id="")
    assert (
        window_start_ns(
            EventLedger(), IID, open_pos, last_processed_ns=9_000, warmup_ns=100, run_start_ns=1
        )
        == 500
    )


def test_window_start_flat_no_warmup_collapses_to_cursor() -> None:
    ledger = EventLedger([_fill(10, 1, 1.0, 100.0), _fill(1_000, -1, 1.0, 110.0)])
    assert (
        window_start_ns(
            ledger, IID, None, last_processed_ns=2_000, warmup_ns=0, run_start_ns=10
        )
        == 2_000
    )


def test_window_start_flat_with_warmup_pins_to_last_close() -> None:
    ledger = EventLedger([_fill(10, 1, 1.0, 100.0), _fill(1_000, -1, 1.0, 110.0)])
    assert (
        window_start_ns(
            ledger, IID, None, last_processed_ns=2_000, warmup_ns=30, run_start_ns=10
        )
        == 970
    )


def test_window_start_flat_no_close_uses_run_start() -> None:
    assert (
        window_start_ns(
            EventLedger(), IID, None, last_processed_ns=500, warmup_ns=0, run_start_ns=100
        )
        == 500
    )
    assert (
        window_start_ns(
            EventLedger(), IID, None, last_processed_ns=500, warmup_ns=10, run_start_ns=100
        )
        == 90
    )


# ---------------------------------------------------------------------------
# checkpoint_balance
# ---------------------------------------------------------------------------


def test_checkpoint_empty_is_starting_capital() -> None:
    assert checkpoint_balance(EventLedger(), IID, None, START) == pytest.approx(START)


def test_checkpoint_open_first_trade_removes_carried_legs() -> None:
    ledger = EventLedger(
        [
            _cash(0, START),
            _cash(10, -6_000.0),
            _comm(10, 6_000.0),
            _fill(10, 1, 0.1, 60_000.0),
        ]
    )
    pos = OpenPosition(side=1, quantity=0.1, entry_price=60_000.0, entry_ns=10, trade_id="")
    # The balance *before* the carried entry is the starting capital (the fold already
    # contains the opening cash + the entry legs; the entry legs are removed).
    assert checkpoint_balance(ledger, IID, pos, START, cost_model=CM) == pytest.approx(START)


def test_checkpoint_non_flip_carry_is_balance_after_close() -> None:
    ledger = EventLedger(
        [
            _cash(0, START),
            _cash(10, -6_000.0),
            _comm(10, 6_000.0),
            _fill(10, 1, 0.1, 60_000.0),
            _cash(20, 6_100.0),
            _comm(20, 6_100.0),
            _fill(20, -1, 0.1, 61_000.0),
            # carried long re-opened at bar 40 (strictly after the close):
            _cash(40, -6_200.0),
            _comm(40, 6_200.0),
            _fill(40, 1, 0.1, 62_000.0),
        ]
    )
    pos = OpenPosition(side=1, quantity=0.1, entry_price=62_000.0, entry_ns=40, trade_id="")
    expected = (
        START
        - 6_000.0
        - float(fill_cost(CM, notional=6_000.0))
        + 6_100.0
        - float(fill_cost(CM, notional=6_100.0))
    )
    # balance after the bar-20 close (entry at bar 40 is outside the ts<=20 fold).
    assert checkpoint_balance(ledger, IID, pos, START, cost_model=CM) == pytest.approx(
        expected
    )


def test_checkpoint_same_bar_flip_is_balance_after_close() -> None:
    ledger = EventLedger(
        [
            _cash(0, START),
            _cash(10, -6_000.0),
            _comm(10, 6_000.0),
            _fill(10, 1, 0.1, 60_000.0),
            _cash(20, 6_100.0),
            _comm(20, 6_100.0),
            _fill(20, -1, 0.1, 61_000.0),  # close long
            _cash(20, 6_100.0),
            _comm(20, 6_100.0),
            _fill(20, -1, 0.1, 61_000.0),  # short entry on the same bar
        ]
    )
    pos = OpenPosition(side=-1, quantity=0.1, entry_price=61_000.0, entry_ns=20, trade_id="")
    expected = (
        START
        - 6_000.0
        - float(fill_cost(CM, notional=6_000.0))
        + 6_100.0
        - float(fill_cost(CM, notional=6_100.0))
    )
    assert checkpoint_balance(ledger, IID, pos, START, cost_model=CM) == pytest.approx(
        expected
    )


def test_checkpoint_excludes_funding() -> None:
    ledger = EventLedger(
        [
            _cash(0, START),
            _funding(5, 12.5),
            _cash(10, -6_000.0),
            _comm(10, 6_000.0),
            _fill(10, 1, 0.1, 60_000.0),
        ]
    )
    pos = OpenPosition(side=1, quantity=0.1, entry_price=60_000.0, entry_ns=10, trade_id="")
    assert checkpoint_balance(ledger, IID, pos, START, cost_model=CM) == pytest.approx(START)


# ---------------------------------------------------------------------------
# VbtPaperState (incremental scan + memo + engine tag)
# ---------------------------------------------------------------------------


def test_state_incremental_last_close() -> None:
    state = VbtPaperState(instrument_id=IID, ledger=EventLedger())
    assert state.last_close_ns() is None
    state.ledger.append(_fill(10, 1, 1.0, 100.0))
    assert state.last_close_ns() is None
    state.ledger.append(_fill(20, -1.0, 1.0, 110.0))
    assert state.last_close_ns() == 20
    # Appending an unrelated funding event does not move the close.
    state.ledger.append(_funding(25, 1.0))
    assert state.last_close_ns() == 20


def test_state_checkpoint_memo_invalidates_on_new_close() -> None:
    state = VbtPaperState(instrument_id=IID, ledger=EventLedger())
    state.ledger.append(_cash(0, START))
    state.ledger.append(_fill(10, 1, 0.1, 60_000.0))
    state.ledger.append(_cash(10, -6_000.0))
    state.ledger.append(_comm(10, 6_000.0))
    state.open_position = OpenPosition(
        side=1, quantity=0.1, entry_price=60_000.0, entry_ns=10, trade_id=""
    )
    assert state.checkpoint_balance(START, cost_model=CM) == pytest.approx(START)

    state.ledger.append(_fill(20, -1, 0.1, 61_000.0))
    state.ledger.append(_cash(20, 6_100.0))
    state.ledger.append(_comm(20, 6_100.0))
    state.open_position = OpenPosition(
        side=-1, quantity=0.1, entry_price=61_000.0, entry_ns=20, trade_id=""
    )
    first = state.checkpoint_balance(START, cost_model=CM)
    assert state._ckpt_memo is not None
    # A later event (ts > close) must not change the memoized checkpoint.
    state.ledger.append(_funding(30, 3.0))
    assert state.checkpoint_balance(START, cost_model=CM) == pytest.approx(first)


def test_state_window_start_uses_open_entry_or_cursor_warmup() -> None:
    state = VbtPaperState(instrument_id=IID, ledger=EventLedger())
    state.ledger.append(_fill(10, 1, 1.0, 100.0))
    state.last_processed_ns = 2_000
    # Flat (no close): window start collapses to the cursor with warmup 0…
    assert state.window_start_ns() == 2_000
    state.ledger.append(_fill(1_000, -1, 1.0, 110.0))  # close at 1000
    assert state.window_start_ns() == 2_000  # max(last_close, cursor) with warmup 0
    assert state.window_start_ns(warmup_ns=30) == 970  # pinned to last_close - warmup
    # Open position: carried entry bar wins over everything.
    state.open_position = OpenPosition(
        side=1, quantity=1.0, entry_price=120.0, entry_ns=500, trade_id=""
    )
    assert state.window_start_ns() == 500
    assert state.window_start_ns(warmup_ns=30) == 500


def test_state_window_start_fresh_session_uses_ledger_start() -> None:
    state = VbtPaperState(instrument_id=IID, ledger=EventLedger())
    assert state.window_start_ns() == 0  # no events and no cursor yet
    state.ledger.append(_cash(0, START))
    assert state.window_start_ns() == 0


def test_vbt_state_save_load_round_trip(tmp_path) -> None:
    path = tmp_path / "s.db"
    state = VbtPaperState(instrument_id=IID, ledger=EventLedger(), last_processed_ns=20)
    state.save(str(path), run_id="r")
    loaded = VbtPaperState.load(str(path), run_id="r")
    assert loaded.instrument_id == IID
    assert loaded.last_processed_ns == 20
    assert loaded.aux_data["vbt"]["engine"] == "vectorbt"


def test_vbt_state_load_rejects_foreign_row(tmp_path) -> None:
    path = tmp_path / "s.db"
    PaperState(instrument_id=IID, ledger=EventLedger()).save(str(path), run_id="r")
    with pytest.raises(StateCorruptionError, match="not written by the vectorbt engine"):
        VbtPaperState.load(str(path), run_id="r")


def test_get_state_class_registry() -> None:
    assert get_state_class("vectorbt") is VbtPaperState
    assert get_state_class("nautilus") is PaperState
    assert get_state_class(None) is PaperState
    assert get_state_class("") is PaperState
    assert get_state_class("something-unknown") is PaperState

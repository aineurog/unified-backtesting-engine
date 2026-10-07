"""Unit tests for the vectorbt paper state + derivations (§4/§5 of the vbt paper plan).

Covers the validated checkpoint formula (including the same-bar flip and the funding
exclusion), the window-start rules, the incremental fill scan / checkpoint memo, and the
engine-tag guard on ``VbtPaperState.load``.
"""

from __future__ import annotations

from math import inf, nextafter

import pytest

from ube.core.cost import CostModel, fill_cost
from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType, LedgerEvent
from ube.papertrading.core import _open_position_from_ledger, get_state_class
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


def _close(ts: int, side: int, qty: float, price: float, reason: str = "atr_stop") -> LedgerEvent:
    return LedgerEvent(
        EventType.FILL, ts, IID, side=side, quantity=qty, price=price, exit_reason=reason
    )


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


def test_last_close_ignores_a_ulp_residue_on_the_close_bar() -> None:
    # A cover fill can settle a position one float-ulp off the size it opened. A live
    # GBPUSD poll closed a 75382.2076-unit short with a cover exactly 1 ulp larger, leaving
    # a 1.46e-11 residue -- ~1e4x an absolute 1e-12 floor. The residue was folded as a live
    # long, so the real close bar was skipped and the *next* fill was misread as the close,
    # which shifted the replay window and checkpoint bound onto the wrong bars. Judging
    # flatness relative to the size being settled keeps the real close bar.
    short = 75_382.2076
    cover = nextafter(short, inf)
    assert cover - short > 1e-12  # guard: the fixture must leave an above-floor residue
    ledger = EventLedger(
        [
            _fill(10, -1, short, 1.32867),
            _fill(20, 1, cover, 1.32897915),
            _fill(30, -1, 75_191.531, 1.32943),
        ]
    )
    assert last_close_ns(ledger, IID) == 20


def test_open_position_entry_ns_is_not_inherited_from_a_residue_cover() -> None:
    # Same residue, but the *open position* derivation must agree with last_close_ns.
    # It judged flatness with a hardcoded absolute 1e-12, so a 1-ulp over-cover left a
    # phantom position and the next entry was folded into it, inheriting the stop bar's
    # entry_ns/entry_price. A carried vbt short therefore reported its entry bar as the
    # 08:03 stop instead of the real 08:05 entry.
    long_qty = 75_924.78287
    stop = nextafter(long_qty, inf)
    assert stop - long_qty > 1e-12  # guard: the fixture must leave an above-floor residue
    ledger = EventLedger(
        [
            _fill(1780841600, 1, long_qty, 1.32354),          # long entry 08:00
            _fill(1780841780, -1, stop, 1.323033878384278),   # ATR stop cover 08:03
            _fill(1780841900, -1, 75_643.59094, 1.32338),     # short entry 08:05
        ]
    )
    pos = _open_position_from_ledger(ledger, IID)
    assert pos is not None
    assert pos.side == -1
    assert pos.entry_ns == 1780841900  # the 08:05 entry, not the 08:03 stop bar
    assert pos.entry_price == pytest.approx(1.32338)


def test_checkpoint_survives_a_carried_short_after_a_residue_close() -> None:
    # The live GBPUSD stall: the phantom carry made entry_ns (08:03) <= the checkpoint
    # fold bound (08:03), so checkpoint_balance added _entry_legs for an entry the fold
    # had not booked yet and aborted the worker with "balance must be > 0 ... got
    # -90094.69". The realized balance (10010.52) must survive as the seed instead.
    long_qty = 75_924.78287
    stop = nextafter(long_qty, inf)
    ledger = EventLedger(
        [
            _cash(0, START),
            _cash(1780841600, -long_qty * 1.32354),  # long entry debit
            _cash(1780841780, +long_qty * 1.323033878384278),  # stop credit
            _cash(1780841900, +75_643.59094 * 1.32338),  # short entry credit
            _fill(1780841600, 1, long_qty, 1.32354),
            _fill(1780841780, -1, stop, 1.323033878384278),
            _fill(1780841900, -1, 75_643.59094, 1.32338),
        ]
    )
    pos = _open_position_from_ledger(ledger, IID)
    assert pos is not None
    bound = last_close_ns(ledger, IID)
    assert bound == 1780841780
    assert bound < pos.entry_ns  # the carry is *after* the bound
    # realized = everything folded through the close; the 08:05 entry leg is not in it
    realized = START - long_qty * 1.32354 + long_qty * 1.323033878384278
    seed = checkpoint_balance(ledger, IID, pos, START, multiplier=MULT, cost_model=None)
    assert seed == pytest.approx(realized, abs=1e-6)
    assert seed > 0.0  # the guard the vbt backend asserts on


def test_residue_flip_ledger_recovers_true_carried_short_and_positive_checkpoint() -> None:
    # The live crypto-perp crash: the adapters' end-of-run restore kept the 07:45 short
    # "held", suppressing its buyback fill, so the stream is --1.187, +1.194, --1.194
    # (marked close), --1.188. The flip's two legs are sized off *different* equity
    # (1.187 vs 1.194), so a residue-only fold read a phantom "short 2.375 @ 07:56",
    # entry_ns <= checkpoint bound bent the old fold to -89,684.34 and the worker aborted
    # with "vectorbt paper checkpoint balance must be > 0". The reason-marked fold must
    # recover the true carried short (1.188 @ 08:09) and a positive realized seed.
    t45, t54, t56, t809 = (1_745_000_000_000_000_000, 1_745_000_540_000_000_000,
                           1_745_000_660_000_000_000, 1_745_000_940_000_000_000)
    p45, p54, p56, p809 = 84_226.19, 84_101.04, 84_040.90, 84_034.15
    ledger = EventLedger(
        [
            _cash(0, START),
            _fill(t45, -1, 1.187, p45),        # short open 07:45 (buyback suppressed)
            _fill(t54, 1, 1.194, p54),         # flip open long 07:54 (unmarked)
            _close(t56, -1, 1.194, p56),       # atr stop closes the long 07:56 (marked)
            _fill(t809, -1, 1.188, p809),      # short open 08:09 (the carried trade)
        ]
    )
    pos = _open_position_from_ledger(ledger, IID)
    assert pos is not None
    assert pos.side == -1
    assert pos.quantity == pytest.approx(1.188, abs=1e-6)
    assert pos.entry_ns == t809  # the 08:09 entry, NOT the phantom 07:56 flip bar
    assert pos.entry_price == pytest.approx(p809)

    bound = last_close_ns(ledger, IID)
    assert bound == t56  # the marked close, not the flip bar
    assert bound < pos.entry_ns

    # Realized equity: the short matured by the 07:54 flip (+), the long by the 07:56
    # close (-); the 08:09 short is carried and excluded. That is ~10k, not the ~110k the
    # un-returned short credit would leave in a margin-cash fold, and positive (no crash).
    expected = START + (p45 - p54) * 1.187 + (p56 - p54) * 1.194
    seed = checkpoint_balance(ledger, IID, pos, START, multiplier=MULT, cost_model=None)
    assert seed == pytest.approx(expected, abs=1e-6)
    assert seed > 0.0
    assert seed < 2.0 * START  # ~10k, not the ~110k inflated fold


def test_checkpoint_seed_is_the_realized_equity_not_twice_inflated_cash() -> None:
    # Sizing-basis regression: a carried trade's seed must be the equity the adapter's
    # sizing fold uses. With the short's opening credit stuck in cash (its buyback
    # suppressed), the margin-cash fold reads ~109,922 and the warm window re-sizes the
    # carried short at ~11x. The realized-fill seed stays ~10k.
    t45, t54, t56, t809 = (1_745_000_000_000_000_000, 1_745_000_540_000_000_000,
                           1_745_000_660_000_000_000, 1_745_000_940_000_000_000)
    p45, p54, p56, p809 = 84_226.19, 84_101.04, 84_040.90, 84_034.15
    ledger = EventLedger(
        [
            _cash(0, START),
            _cash(t45, +1.187 * p45),      # short open credit (buyback never booked)
            _cash(t54, -1.194 * p54),      # long open debit (flip)
            _cash(t56, +1.194 * p56),      # long close credit
            _cash(t809, +1.188 * p809),    # short open credit (carried)
            _fill(t45, -1, 1.187, p45),
            _fill(t54, 1, 1.194, p54),
            _close(t56, -1, 1.194, p56),
            _fill(t809, -1, 1.188, p809),
        ]
    )
    pos = _open_position_from_ledger(ledger, IID)
    assert pos is not None and pos.side == -1
    seed = checkpoint_balance(ledger, IID, pos, START, multiplier=MULT, cost_model=None)
    assert seed > 0.0
    assert seed < 2.0 * START  # ~10k, not ~110k
    # Rough equity sanity: START plus the two realized legs, minus nothing for the carry.
    realized = (p45 - p54) * 1.187 + (p56 - p54) * 1.194
    assert seed == pytest.approx(START + realized, abs=1e-6)


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

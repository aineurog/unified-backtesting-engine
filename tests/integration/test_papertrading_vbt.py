"""Integration tests for the vectorbt paper backend (``ube.papertrading`` ``engine="vectorbt"``).

Drives the real :class:`~ube.adapters.vectorbt_adapter.adapter.VectorbtAdapter` through the
paper ``init``/``run`` API over synthetic ``crypto_perp`` bars, and asserts the defining
property of the recomputable backend: folding a long history in *multiple* windows produces
exactly the ledger a single full-window run produces (including same-bar flips and carried
positions), because every window re-derives its start and checkpoint balance from state.
"""

from __future__ import annotations

import numpy as np
import pytest

from ube.core.config import BacktestConfig, SignalConfig
from ube.core.data import MarketData
from ube.core.errors import EngineError
from ube.core.ledger import trades
from ube.core.risk.exits import ATRStop, TakeProfit
from ube.core.signals import Signals, from_target
from ube.papertrading import get_paper_engine, get_state_class, init, run
from ube.papertrading.config import PaperConfig
from ube.papertrading.vbt.backend import (
    VbtPaperEngine,
    _apply_policy,
    _max_atr_period,
    _zero_at_or_before,
)
from ube.papertrading.vbt.state import VbtPaperState
from ube.testing.synthetic import PRESETS, synthetic_bars

AC = "crypto_perp"


def _config(**kw: object) -> PaperConfig:
    instr = PRESETS[AC].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    return PaperConfig(base=bc, engine="vectorbt", starting_balance=10_000.0, **kw)


def _slice_md(md: MarketData, sl: slice) -> MarketData:
    sliced = MarketData.__new__(MarketData)  # type: ignore[call-arg]
    object.__setattr__(sliced, "open", md.open[sl])
    object.__setattr__(sliced, "high", md.high[sl])
    object.__setattr__(sliced, "low", md.low[sl])
    object.__setattr__(sliced, "close", md.close[sl])
    object.__setattr__(sliced, "volume", md.volume[sl])
    object.__setattr__(sliced, "index", md.index[sl])
    return sliced


def _slice_sig(sig: Signals, sl: slice) -> Signals:
    return Signals(
        long_entry=sig.long_entry[sl],
        long_exit=sig.long_exit[sl],
        short_entry=sig.short_entry[sl],
        short_exit=sig.short_exit[sl],
    )


def _summary(state: VbtPaperState, cfg: PaperConfig) -> list[tuple[int, float, int, int]]:
    instr = cfg.base.instrument
    return [
        (t.side, round(t.net_pnl, 6), t.entry_timestamp, t.exit_timestamp)
        for t in trades(state.ledger, instruments={instr.symbol: instr})
    ]


def _run_windows(
    bounds: list[tuple[int, int]], *, n: int = 24, target: np.ndarray | None = None
) -> tuple[list[tuple[int, float, int, int]], object]:
    if target is None:
        target = np.zeros(n, dtype=int)
        target[4:10] = 1
        target[10:18] = -1
        target[20:24] = 1
    data = synthetic_bars(PRESETS[AC], n_bars=n)
    signals = from_target(target)
    cfg = _config()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "s.db")
        init(cfg, run_id="x", db_path=db)
        state: VbtPaperState | None = None
        for a, b in bounds:
            state, _ = run(
                "x",
                _slice_md(data, slice(a, b)),
                _slice_sig(signals, slice(a, b)),
                cfg,
                db_path=db,
            )
    assert state is not None
    return _summary(state, cfg), state.open_position


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_engine_and_state_class_registered() -> None:
    assert get_paper_engine("vectorbt") is VbtPaperEngine
    assert get_state_class("vectorbt") is VbtPaperState


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_apply_policy_opens_and_holds_long() -> None:
    sig = Signals(
        long_entry=np.array([True, True, False, False]),
        long_exit=np.zeros(4, dtype=bool),
        short_entry=np.zeros(4, dtype=bool),
        short_exit=np.zeros(4, dtype=bool),
    )
    out = _apply_policy(sig, allow_short=True, policy="reverse")
    # The first bar opens; the repeated entry (already long) is a hold.
    assert list(out.long_entry) == [True, False, False, False]
    assert not out.long_exit.any()


def test_apply_policy_reverse_emits_close_and_opposite_entry() -> None:
    sig = Signals(
        long_entry=np.array([True, False, False]),
        long_exit=np.zeros(3, dtype=bool),
        short_entry=np.array([False, True, False]),
        short_exit=np.zeros(3, dtype=bool),
    )
    out = _apply_policy(sig, allow_short=True, policy="reverse")
    assert list(out.long_entry) == [True, False, False]
    assert list(out.long_exit) == [False, True, False]
    assert list(out.short_entry) == [False, True, False]


def test_apply_policy_exit_only_does_not_open_opposite() -> None:
    sig = Signals(
        long_entry=np.array([True, False, False]),
        long_exit=np.zeros(3, dtype=bool),
        short_entry=np.array([False, True, False]),
        short_exit=np.zeros(3, dtype=bool),
    )
    out = _apply_policy(sig, allow_short=True, policy="exit_only")
    assert list(out.long_exit) == [False, True, False]
    assert not out.short_entry.any()


def test_apply_policy_respects_allow_short() -> None:
    sig = Signals(
        long_entry=np.zeros(1, dtype=bool),
        long_exit=np.zeros(1, dtype=bool),
        short_entry=np.array([True]),
        short_exit=np.zeros(1, dtype=bool),
    )
    out = _apply_policy(sig, allow_short=False, policy="reverse")
    assert not out.short_entry.any()


def test_zero_at_or_before_drops_bound_bar() -> None:
    sig = Signals(
        long_entry=np.array([True, False, True]),
        long_exit=np.zeros(3, dtype=bool),
        short_entry=np.zeros(3, dtype=bool),
        short_exit=np.zeros(3, dtype=bool),
    )
    ts = np.array([10, 20, 30])
    out = _zero_at_or_before(sig, ts, 20)
    assert list(out.long_entry) == [False, False, True]


def test_max_atr_period() -> None:
    class Risk:
        exit = (ATRStop(14, period=14), TakeProfit(percent=0.05))

    assert _max_atr_period(Risk()) == 14
    assert _max_atr_period(None) == 0


# ---------------------------------------------------------------------------
# full run vs. multi-window resume
# ---------------------------------------------------------------------------


def test_full_run_produces_flip_trades() -> None:
    got, open_pos = _run_windows([(0, 24)])
    assert [t[0] for t in got] == [1, -1]
    assert got[0][2] == 1_704_081_600_000_000_000  # entry bar 4
    assert open_pos is not None and open_pos.side == 1  # bar-20 long still open


@pytest.mark.parametrize(
    "bounds",
    [
        [(0, 11), (4, 24)],  # window 1 ends exactly on the flip bar
        [(0, 10), (4, 18), (9, 24)],  # open carry; resume at the entry bar
        [(0, 12), (4, 20), (9, 24)],
        [(0, 6), (4, 14), (9, 24)],
        [(0, 11), (10, 20), (20, 24)],
    ],
)
def test_split_windows_match_full_run(bounds: list[tuple[int, int]]) -> None:
    full, full_open = _run_windows([(0, 24)])
    got, open_pos = _run_windows(bounds)
    assert got == full
    assert open_pos == full_open


def test_resume_requires_entry_bar_in_window() -> None:
    # After window 1 (bars 0..9) the long entered at bar 4 is still open. A window that
    # starts after bar 4 (i.e. omits the entry bar) cannot re-open the carried trade.
    data = synthetic_bars(PRESETS[AC], n_bars=24)
    target = np.zeros(24, dtype=int)
    target[4:10] = 1
    target[10:18] = -1
    signals = from_target(target)
    cfg = _config()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "s.db")
        init(cfg, run_id="x", db_path=db)
        run("x", _slice_md(data, slice(0, 10)), _slice_sig(signals, slice(0, 10)), cfg, db_path=db)
        with pytest.raises(EngineError, match="carried entry"):
            run(
                "x",
                _slice_md(data, slice(5, 24)),
                _slice_sig(signals, slice(5, 24)),
                cfg,
                db_path=db,
            )


def test_resume_carries_entry_when_signal_fn_does_not_reemit_it() -> None:
    # Crash-2 regression: the live GBPUSD engine died on `EngineError: vectorbt window
    # does not re-emit the carried entry signal` because the *stateful* signal function
    # (a wall-clock minute rule) did not reproduce the entry when the window restarted at
    # the entry bar. The persisted ledger is authoritative, so the backend must carry the
    # open trade forward (with a warning) instead of aborting the live worker.
    data = synthetic_bars(PRESETS[AC], n_bars=24)
    # Window 1 (bars 0..11): the wall-clock fn originally emitted a long entry at bar 3
    # and holds it — the reference leg we must resume.
    first = from_target(np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # Window 2: the same "wall-clock minute" re-derives a FLAT signal at bar 3 (the entry
    # is not re-emitted at its own bar), but the next leg — a short at bar 15 — is.
    signal_fn_target = np.zeros(24, dtype=int)
    signal_fn_target[15:20] = -1
    signals = from_target(signal_fn_target)
    cfg = _config()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "s.db")
        init(cfg, run_id="x", db_path=db)
        run("x", _slice_md(data, slice(0, 12)), _slice_sig(first, slice(0, 12)), cfg, db_path=db)
        # Window 2 restarts at the carried entry bar (bar 3); the wall-clock fn re-derives
        # a flat signal there, so the window must be carried from the ledger, not crash.
        with pytest.warns(UserWarning, match="not re-emitted"):
            state, _ = run(
                "x",
                _slice_md(data, slice(3, 24)),
                _slice_sig(signals, slice(3, 24)),
                cfg,
                db_path=db,
            )

    got = _summary(state, cfg)
    # Long entered bar 3, closed by the short at bar 15 — both legs survive the carry.
    assert [t[0] for t in got] == [1, -1]
    assert got[0][2] == 1_704_078_000_000_000_000  # entry bar 3
    assert state.open_position is None  # the bar-15 short is closed by bar 20's 0


def test_gapped_resume_does_not_double_book_starting_balance() -> None:
    # Regression: a warm window whose first bar falls strictly after the cursor (a data
    # gap right after the flat last-close bar, e.g. the missing 1m bars observed in real
    # XAUUSD history) used to leak the window-start ``+starting_balance`` cash event into
    # the persisted ledger on every resume — inflating the ledger and the next checkpoint
    # by exactly the checkpoint amount, on top of the single initial deposit.
    data = synthetic_bars(PRESETS[AC], n_bars=24)
    # Window 1: long entered at bar 4, closed (flat) at bar 9.
    first = from_target(np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]))
    cfg = _config()
    import tempfile
    from pathlib import Path

    from ube.core.ledger import EventType

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "s.db")
        init(cfg, run_id="x", db_path=db)
        run("x", _slice_md(data, slice(0, 12)), _slice_sig(first, slice(0, 12)), cfg, db_path=db)
        # Window 2 SKIPS bars 10-11 (10m missing histogram) and re-enters at bar 12 — the
        # resume window starts at bar 12, strictly after the cursor at bar 11.
        second_full = np.zeros(24, dtype=int)
        second_full[12:18] = 1
        second = from_target(second_full)
        state, _ = run(
            "x",
            _slice_md(data, slice(12, 24)),
            _slice_sig(second, slice(12, 24)),
            cfg,
            db_path=db,
        )

    # The +10000 initial deposit may be booked EXACTLY ONCE. Entry/exit cash legs are
    # legitimate; the bug was a second window-start re-seed (amount == the checkpoint,
    # not the initial 10000) when a warm window's first bar fell strictly after the
    # cursor.
    seeds = [
        e
        for e in state.ledger.events
        if e.event_type is EventType.CASH_MOVEMENT
        and e.amount is not None
        and abs(float(e.amount) - 10_000.0) < 1e-9
    ]
    assert len(seeds) == 1, (
        f"initial deposit booked {len(seeds)} times (double-booking leak): "
        f"{[c.amount for c in seeds]}"
    )

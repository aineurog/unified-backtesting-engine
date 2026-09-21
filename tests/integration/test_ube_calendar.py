"""The §4.4 calendar guard, end-to-end, across every engine surface.

The declared trading calendar must have authority over *every* engine path
— backtest and paper alike — even when off-session data is provided:

* **Backtest** (``ube.run``): a bar whose timestamp the declared calendar calls
  closed is a §15 ``CalendarMismatchError``, never silently accepted. The check
  runs before engine dispatch (``run.py``), so all three adapter names must
  reject the same weekend-bar dataset. "24/7" crypto is a no-op.
* **Paper** (``ube.paper.init``/``step``): by default (``calendar_strict=False``)
  the off-session bars are skipped with a ``UserWarning`` so one bad timestamp
  never kills a long-running session; ``calendar_strict=True`` opts into the hard
  ``CalendarMismatchError`` for backtest parity. Both engines (vectorbt and
  nautilus) enforce it, independently of their window/bar mechanics.

The instruments mirror the live paper configs: XAUUSD/GBPUSD on "24/5" and
AAPL on "NASDAQ" — all weekend-closed, so a Fri→Mon dataset carrying Saturday
and Sunday bars is off-session for each. BTC-USDT "24/7" is the control that
proves the guard fires on *calendar*, not on bar count or weekday.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ube
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.data import MarketData
from ube.core.errors import CalendarMismatchError
from ube.core.instrument import Instrument
from ube.core.ledger import EventType, trades
from ube.core.signals import from_target
from ube.papertrading import get_state_class, init, run, run_auto, step
from ube.papertrading.config import PaperConfig
from ube.testing.synthetic import PRESETS

#: The weekend-spanning dataset: Fri 15:00 UTC (in-session for "24/5" and NASDAQ),
#: then the closed Saturday/Sunday, then Mon 15:00 UTC (in-session again). Any engine
#: must treat the middle pair as off-session for every sessioned instrument below.
WEEKEND = [
    "2024-01-05 15:00",  # Friday — market day, session open
    "2024-01-06 15:00",  # Saturday — closed
    "2024-01-07 15:00",  # Sunday — closed
    "2024-01-08 15:00",  # Monday — market day, session open
]

#: ``("24/5"|"24/7")`` instruments as declared in the live paper configs.
SESSIONED: dict[str, Instrument] = {
    "commodities": Instrument(
        "XAUUSD", "commodities", calendar="24/5", settlement_currency="USD"
    ),
    "forex": Instrument("GBPUSD", "forex", calendar="24/5", settlement_currency="USD"),
    "stocks": Instrument("AAPL", "stocks", calendar="NASDAQ", settlement_currency="USD"),
}

BACKTEST_ENGINES = ["vectorbt", "backtrader", "nautilus"]
PAPER_ENGINES = ["vectorbt", "nautilus"]


def _md(timestamps: list[str]) -> MarketData:
    """Hourly-rise OHLC bars over the given UTC timestamps."""
    idx = pd.to_datetime(timestamps).tz_localize("UTC")
    n = len(idx)
    base = np.arange(1, n + 1, dtype=float)
    return MarketData(
        open=base,
        high=base + 0.5,
        low=base - 0.5,
        close=base + 0.25,
        volume=np.full(n, 10.0),
        index=idx,
    )


def _in_session_ns(ts_index: pd.DatetimeIndex) -> set[int]:
    """The set of timestamps (ns epochs) the calendar declares open."""
    return {int(v) for v in ts_index.as_unit("ns").asi8}  # type: ignore[attr-defined]


def _backtest_config(engine: str, instrument: Instrument) -> BacktestConfig:
    return BacktestConfig(
        instrument=instrument,
        engine=engine,
        engine_overrides={"starting_balance": 100000.0},
    )


def _paper_config(engine: str, instrument: Instrument, **kw: object) -> PaperConfig:
    bc = BacktestConfig(
        instrument=instrument,
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    return PaperConfig(base=bc, engine=engine, starting_balance=10_000.0, **kw)


# ---------------------------------------------------------------------------
# Backtest surface (§4.4 / §7.1): every engine rejects off-session bars.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", BACKTEST_ENGINES)
@pytest.mark.parametrize("asset_class", list(SESSIONED))
def test_backtest_rejects_off_session_bars(tmp_path, engine: str, asset_class: str) -> None:
    # §4.4: a bar whose timestamp the declared calendar calls closed is a
    # CalendarMismatchError, never silently accepted. The check runs inside run.py's
    # single-instrument branch *before* engine dispatch, so every adapter rejects the
    # same dataset — engine availability is irrelevant to the guarantee.
    data = _md(WEEKEND)
    signals = from_target([1, 1, 1, 0])
    with pytest.raises(CalendarMismatchError, match="outside the declared trading calendar"):
        ube.run(
            data,
            signals,
            _backtest_config(engine, SESSIONED[asset_class]),
            log_path=tmp_path / "e.db",
        )


def test_backtest_24_7_trades_through_weekend(tmp_path) -> None:
    # "24/7" declares no calendar constraint, so the same weekend bars are *data*, not
    # a contradiction: the run must succeed. This proves the guard fires on the calendar,
    # not on bar count or weekday.
    data = _md(WEEKEND)
    signals = from_target([1, 1, 1, 0])
    result = ube.run(
        data,
        signals,
        _backtest_config("vectorbt", PRESETS["crypto_perp"].instrument),
        log_path=tmp_path / "e.db",
    )
    assert result is not None
    assert len(result.trades) == 1  # the round trip across the weekend still books


def test_backtest_all_in_session_bars_pass(tmp_path) -> None:
    # Control: identical prices on trading days only — no holiday, no weekend — must
    # run cleanly (proves the reject path is not rejecting everything under a sessioned
    # calendar).
    data = _md(["2024-01-05 15:00", "2024-01-08 15:00"])
    signals = from_target([1, 0])
    for engine in BACKTEST_ENGINES:
        result = ube.run(
            data,
            signals,
            _backtest_config(engine, SESSIONED["commodities"]),
            log_path=tmp_path / f"{engine}.db",
        )
        assert result is not None
        assert len(result.trades) == 1


# ---------------------------------------------------------------------------
# Paper surface (§9.x): skip by default, raise when strict.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", PAPER_ENGINES)
@pytest.mark.parametrize("asset_class", list(SESSIONED))
def test_paper_skips_off_session_bars_by_default(engine: str, asset_class: str) -> None:
    # Default mode (calendar_strict=False): a signal whose only entry sits on a closed
    # Saturday must be dropped (with a UserWarning), never traded — the weekend entry
    # bar is filtered out before the engine sees it.
    if engine == "nautilus":
        pytest.importorskip("nautilus_trader")
    data = _md(WEEKEND)
    signals = from_target([0, 1, 1, 0])  # entry on bar 1 (Saturday) — off-session
    cfg = _paper_config(engine, SESSIONED[asset_class])

    with pytest.warns(UserWarning, match="calendar"):
        state = init(cfg)
        _, events = step(data, signals, state, cfg)

    fills = [e for e in events if e.event_type == EventType.FILL]
    assert fills == [], "the off-session Saturday entry must never produce a fill"
    sym = SESSIONED[asset_class].symbol
    assert trades(state.ledger, instruments={sym: SESSIONED[asset_class]}) == ()
    assert state.open_position is None


@pytest.mark.parametrize("engine", PAPER_ENGINES)
@pytest.mark.parametrize("asset_class", list(SESSIONED))
def test_paper_round_trip_survives_weekend_by_default(engine: str, asset_class: str) -> None:
    # Default skip mode must not *over*-drop: a Fri entry + Mon exit (both in-session),
    # with the weekend gap in between, still books exactly one round-trip — fills land
    # only on the in-session timestamps.
    if engine == "nautilus":
        pytest.importorskip("nautilus_trader")
    data = _md(WEEKEND)
    signals = from_target([1, 1, 1, 0])  # entry Fri, exit Mon; weekend held
    cfg = _paper_config(engine, SESSIONED[asset_class])

    with pytest.warns(UserWarning, match="calendar"):
        state = init(cfg)
        _, events = step(data, signals, state, cfg)

    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) >= 1, "the in-session round trip must still trade"
    in_session = _in_session_ns(data.timestamps[[0, 3]])
    for fill in fills:
        assert fill.timestamp in in_session, "a fill landed on a closed Saturday/Sunday bar"
    # Every fill (plus position change) stays on the historical in-session timeline.
    for event in events:
        assert event.timestamp in in_session | {0}, "event escaped the in-session timeline"


@pytest.mark.parametrize("engine", PAPER_ENGINES)
def test_paper_strict_mode_raises_calendar_mismatch(engine: str) -> None:
    # calendar_strict=True opts into hard CalendarMismatchError — backtest parity — so a
    # paper session can refuse rather than silently skip.
    if engine == "nautilus":
        pytest.importorskip("nautilus_trader")
    data = _md(WEEKEND)
    signals = from_target([1, 1, 1, 0])
    cfg = _paper_config(
        engine,
        SESSIONED["commodities"],
        calendar_strict=True,
    )
    state = init(cfg)
    with pytest.raises(CalendarMismatchError, match="outside the declared trading calendar"):
        step(data, signals, state, cfg)


@pytest.mark.parametrize("engine", PAPER_ENGINES)
def test_paper_all_off_session_warning_names_symbol_and_next_open(engine: str) -> None:
    # The live AAPL scenario (crash 3): a poll whose bars all precede the NASDAQ session
    # (e.g. 09:00 UTC, before the ~14:30 open) must not blind-skip — the operator needs to
    # see *which* instrument is closed and *when* it reopens, per the declared calendar.
    if engine == "nautilus":
        pytest.importorskip("nautilus_trader")
    data = _md(["2024-01-08 09:00", "2024-01-08 09:30", "2024-01-08 10:00"])
    signals = from_target([1, 1, 0])
    cfg = _paper_config(engine, SESSIONED["stocks"])  # AAPL on NASDAQ

    with pytest.warns(UserWarning) as rec:
        state = init(cfg)
        _, events = step(data, signals, state, cfg)

    msg = str(rec[0].message)
    assert "market closed for 'AAPL'" in msg, msg
    assert "next open 2024-01-08 14:30:00+00:00" in msg, msg
    assert all(e.event_type != EventType.FILL for e in events)
    assert state.open_position is None


@pytest.mark.parametrize("engine", PAPER_ENGINES)
def test_paper_24_7_trades_through_weekend(engine: str) -> None:
    # "24/7" is a no-op gate: the same weekend-spanning data is unconstrained, so an
    # entry explicitly placed on the Saturday bar must fill — with no calendar warning.
    if engine == "nautilus":
        pytest.importorskip("nautilus_trader")
    data = _md(WEEKEND)
    signals = from_target([0, 1, 1, 0])  # same off-session-sounding Saturday entry
    cfg = _paper_config(engine, PRESETS["crypto_perp"].instrument)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any calendar warning would fail the test
        state = init(cfg)
        _, events = step(data, signals, state, cfg)

    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) >= 1, "24/7 must trade the weekend entry as ordinary data"


# ---------------------------------------------------------------------------
# The other public entry points (run / run_auto) funnel through step → the same gate.
# ---------------------------------------------------------------------------


def test_paper_run_entry_point_gates_calendar(tmp_path) -> None:
    # ``run(strategy_name, ...)`` (the scheduled-run API) delegates to ``step``, so the
    # §4.4 gate must fire there too: strict mode raises on a weekend bar, and the
    # default skips it (with a warning) without breaking the session.
    data = _md(WEEKEND)
    signals = from_target([0, 1, 1, 0])  # entry on Saturday — off-session
    db = str(tmp_path / "s.db")

    cfg = _paper_config("vectorbt", SESSIONED["commodities"])
    with pytest.warns(UserWarning, match="calendar"):
        run("xau_cal", data, signals, cfg, db_path=db)
    state = get_state_class("vectorbt").load(db, run_id="xau_cal")
    sym = SESSIONED["commodities"].symbol
    assert trades(state.ledger, instruments={sym: SESSIONED["commodities"]}) == ()
    assert state.open_position is None

    # Strict mode through the same public entry point raises CalendarMismatchError.
    strict = _paper_config("vectorbt", SESSIONED["commodities"], calendar_strict=True)
    fresh_db = str(tmp_path / "strict.db")
    with pytest.raises(CalendarMismatchError, match="outside the declared trading calendar"):
        run("xau_cal", data, signals, strict, db_path=fresh_db)


def test_paper_run_auto_entry_point_gates_calendar() -> None:
    # ``run_auto(data, signal_fn, ...)`` also delegates to ``step``, so the gate applies
    # to the streaming "call me" surface too: an entry produced by the signal function on
    # a Saturday bar is dropped with a warning, never traded.
    data = _md(WEEKEND)
    cfg = _paper_config("vectorbt", SESSIONED["stocks"])
    sym = SESSIONED["stocks"].symbol

    def fn(window) -> int:
        i = window.n_bars - 1
        if i == 1:  # bar 1 is the Saturday (off-session) — would be an entry pre-gate
            return 1
        return 0

    state = init(cfg)
    with pytest.warns(UserWarning, match="calendar"):
        events = run_auto(data, fn, cfg, state=state)

    assert all(e.event_type != EventType.FILL for e in events)
    assert trades(state.ledger, instruments={sym: SESSIONED["stocks"]}) == ()
    assert state.open_position is None
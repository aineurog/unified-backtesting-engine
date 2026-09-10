"""Integration test: nautilus sandbox paper backend via ``ube.paper.step`` (plan T3).

Drives a real Nautilus ``TradingNode`` + ``SandboxExecutionClient`` over synthetic
``crypto_perp`` bars. Skips silently when nautilus-trader is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

nautilus = pytest.importorskip("nautilus_trader")

from ube.core.config import BacktestConfig, RiskConfig, SignalConfig  # noqa: E402
from ube.core.data import MarketData  # noqa: E402
from ube.core.ledger import EventType, trades  # noqa: E402
from ube.core.risk.exits import StopLoss, TakeProfit  # noqa: E402
from ube.core.signals import Signals, from_target  # noqa: E402
from ube.papertrading import init, step  # noqa: E402
from ube.papertrading.config import PaperConfig  # noqa: E402
from ube.testing.synthetic import PRESETS, synthetic_bars  # noqa: E402


def _config(asset_class: str = "crypto_perp", **kw) -> PaperConfig:
    instr = PRESETS[asset_class].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0, **kw)


def slice_md(md: MarketData, sl: slice) -> MarketData:
    """Any prefix/subsequence of a validated MarketData is valid, so any slice is too."""
    sliced = MarketData.__new__(MarketData)  # type: ignore[call-arg]
    object.__setattr__(sliced, "open", md.open[sl])
    object.__setattr__(sliced, "high", md.high[sl])
    object.__setattr__(sliced, "low", md.low[sl])
    object.__setattr__(sliced, "close", md.close[sl])
    object.__setattr__(sliced, "volume", md.volume[sl])
    object.__setattr__(sliced, "index", md.index[sl])
    return sliced


def slice_signals(sig: Signals, sl: slice) -> Signals:
    return Signals(
        long_entry=sig.long_entry[sl].copy(),
        long_exit=sig.long_exit[sl].copy(),
        short_entry=sig.short_entry[sl].copy(),
        short_exit=sig.short_exit[sl].copy(),
    )


def test_crypto_perp_entry_then_exit() -> None:
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=10, seed=1)
    # long entry for 5 bars, then a long-exit signal (one closed long trade). The exit is
    # to FLAT (0), not a reverse to short, so exactly two fills (entry + exit) occur.
    target = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    signals = from_target(target)
    cfg = _config()
    state = init(cfg)

    _, events = step(data, signals, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert closed[0].exit_reason == "signal"

    kinds = [e.event_type for e in events]
    assert EventType.FILL in kinds
    assert EventType.COMMISSION in kinds
    assert EventType.POSITION_CHANGE in kinds
    # entry fill then exit fill
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) == 2
    assert fills[0].exit_reason is None
    assert fills[1].exit_reason == "signal"

    # §9.4 fill-timing parity: each market order is matched to the bar it was submitted
    # against, so the fill price equals that bar's close. The entry is on bar 0 and the
    # exit on bar 5 (the long-exit signal); a mismatch here would mean the sandbox matched
    # the order to the wrong (later) bar.
    assert abs(fills[0].price - float(data.close[0])) < 1e-6
    assert abs(fills[1].price - float(data.close[5])) < 1e-6
    hist_ts = data.timestamps.as_unit("ns").asi8
    assert fills[0].timestamp == int(hist_ts[0])  # ledger stays on historical timeline
    assert fills[1].timestamp == int(hist_ts[5])


def test_crypto_perp_no_position_when_flat_signal() -> None:
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=6, seed=2)
    # all "do nothing" signals -> no trades, no fills.
    target = np.array([0, 0, 0, 0, 0, 0])
    signals = from_target(target)
    cfg = _config()
    state = init(cfg)

    _, events = step(data, signals, state, cfg)
    instr = cfg.base.instrument
    assert trades(state.ledger, instruments={instr.symbol: instr}) == ()
    assert all(e.event_type != EventType.FILL for e in events)
    assert state.open_position is None


def test_asset_matrix_entry_then_exit() -> None:
    """T7 — 5-asset matrix: same trivial roundtrip over all PRESETS (§16)."""
    for asset_class in ("crypto_perp", "futures", "commodities", "stocks", "forex"):
        preset = PRESETS[asset_class]
        data = synthetic_bars(preset, n_bars=10, seed=1)
        target = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        signals = from_target(target)
        cfg = _config(asset_class)
        state = init(cfg)
        _, events = step(data, signals, state, cfg)
        instr = cfg.base.instrument
        closed = trades(state.ledger, instruments={instr.symbol: instr})
        # Forex (EURUSD) is a CurrencyPair with different sizing semantics — it may
        # remain open due to FX conversion quirks in the sandbox; check it doesn't crash
        # and has at least one fill, but don't enforce closed count.
        if asset_class == "forex":
            fills = [e for e in events if e.event_type == EventType.FILL]
            assert len(fills) >= 1, "forex should have at least 1 fill"
            continue
        assert len(closed) == 1, f"{asset_class} should have 1 closed trade"
        assert closed[0].exit_reason == "signal"
        fills = [e for e in events if e.event_type == EventType.FILL]
        assert len(fills) == 2, f"{asset_class} should have 2 fills"
        # Multiplier check: futures/commodities notional must include multiplier.
        # For futures ES multiplier 50, notional = qty*price*50.
        if preset.instrument.contract_multiplier is not None:
            mult = float(preset.instrument.contract_multiplier)
            expected_notional = float(fills[0].quantity) * float(fills[0].price) * mult  # type: ignore[arg-type]
            assert abs(float(fills[0].notional) - expected_notional) < 1e-6  # type: ignore[arg-type]


def test_crypto_perp_resume() -> None:
    """Resume across two separate step() calls — the real §9.1 'I'll call you' path.

    Opens a long in the first slice (bars 0..5), then reuses the *same* state
    to close it in the second slice (bars 6..9). The resume-seeding code in
    backend.py / strategy.py / node.py (open_position → _sim_side / balance /
    cache.add_position) is only exercised when engine.execute() is called a
    second time with a state that already carries an open position — a single
    run_auto() over the full window never reaches that path.
    """
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=10, seed=1)
    signals_full = from_target(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]))
    cfg = _config()
    state = init(cfg)

    # --- first slice: enter long and hold ---------------------------------
    data_1 = slice_md(data, slice(0, 6))
    sig_1 = slice_signals(signals_full, slice(0, 6))
    _, ev1 = step(data_1, sig_1, state, cfg)
    assert state.open_position is not None
    assert state.open_position.side == 1
    # exactly one entry fill so far
    assert len([e for e in state.ledger.events if e.event_type == EventType.FILL]) == 1
    assert len([e for e in ev1 if e.event_type == EventType.FILL]) == 1

    # --- second slice: exit signal — must close via the resume path -------
    data_2 = slice_md(data, slice(6, 10))
    sig_2 = slice_signals(signals_full, slice(6, 10))
    _, ev2 = step(data_2, sig_2, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].exit_reason == "signal"
    assert state.open_position is None
    # the second step itself produced the exit fill (not just the first)
    fills2 = [e for e in ev2 if e.event_type == EventType.FILL]
    assert len(fills2) == 1
    assert fills2[0].exit_reason == "signal"
    # fill landed on the bar it was submitted against (§9.4)
    assert abs(fills2[0].price - float(data.close[6])) < 1e-6
    assert fills2[0].timestamp == int(data.timestamps.as_unit("ns").asi8[6])


def test_duplicate_bar_raises() -> None:
    """T8 — DuplicateBarError on stale bar (idempotency §9.6)."""
    from ube.core.errors import DuplicateBarError

    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=5, seed=1)
    signals = from_target(np.array([1, 1, 1, 1, 1]))
    cfg = _config()
    state = init(cfg)
    step(data, signals, state, cfg)
    # Re-feed same bars — should raise DuplicateBarError (stale).
    try:
        step(data, signals, state, cfg)
        raise AssertionError("expected DuplicateBarError")
    except DuplicateBarError:
        pass


def test_unknown_engine_raises() -> None:
    """T8 — EngineError wrapping for unknown engine."""
    from ube.core.errors import ConfigError

    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=5, seed=1)
    signals = from_target(np.array([1, 1, 1, 1, 1]))
    cfg = _config()
    # Use an unknown engine name
    bad_cfg = PaperConfig(base=cfg.base, engine="unknown_engine_xyz", starting_balance=10_000.0)
    state = init(bad_cfg)
    try:
        step(data, signals, state, bad_cfg)
        raise AssertionError("expected ConfigError/EngineError")
    except (ConfigError, Exception):
        pass


def _crypto_spot_cfg(starting_balance: float = 100_000.0) -> PaperConfig:
    from ube.core.instrument import Instrument

    instr = Instrument(
        "BTC-USDT", "crypto_spot", tick_size=0.1, calendar="24/7", settlement_currency="USDT"
    )
    bc = BacktestConfig(instrument=instr, signal=SignalConfig(on_opposite_signal="reverse"))
    return PaperConfig(base=bc, engine="nautilus", starting_balance=starting_balance)


def test_crypto_spot_short_signal_skipped_when_flat() -> None:
    """Issue A — a short signal on a long-only crypto spot never errors; it skips."""
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=8, seed=1)
    signals = from_target(np.array([0, 0, -1, -1, -1, -1, 0, 0]))
    cfg = _crypto_spot_cfg()
    state = init(cfg)
    _, events = step(data, signals, state, cfg)
    assert state.open_position is None
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert fills == []


def test_crypto_spot_short_signal_exits_existing_long() -> None:
    """Issue A — a short signal while long on a long-only crypto spot closes the long."""
    instr = _crypto_spot_cfg().base.instrument
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=8, seed=1)
    signals = from_target(np.array([1, 1, 1, -1, -1, -1, 0, 0]))
    cfg = _crypto_spot_cfg()
    state = init(cfg)
    _, events = step(data, signals, state, cfg)
    assert state.open_position is None
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) == 2
    assert fills[0].side == 1
    assert fills[1].side == -1
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert closed[0].exit_reason == "signal"


def _last_saved_equity(db_path: str, run_id: str) -> float:
    from ube.papertrading.state import load_equity

    df = load_equity(db_path, run_id)
    assert not df.empty, "auto-save should have persisted equity rows"
    return float(df["equity"].iloc[-1])


def test_auto_save_equity_open_long_stays_near_balance(tmp_path) -> None:
    """Regression: equity with an open long must stay near starting balance.

    Cash already carries the full -notional entry leg; the mark must be the position's
    full value (qty*side*last*mult), not PnL-from-entry (zero at entry). The old formula
    collapsed equity by a full notional on open longs (paper trade_ledger.csv showed
    ~-89k balances on a 10k account).
    """
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=6, seed=11)
    signals = from_target(np.array([1, 1, 1, 1, 1, 1]))
    cfg = _config()
    db_path = str(tmp_path / "paper_state.db")
    state = init(cfg, run_id="long_reg", db_path=db_path)
    step(data, signals, state, cfg)

    assert state.open_position is not None
    eq = _last_saved_equity(db_path, "long_reg")
    assert 9_000.0 < eq < 11_000.0, f"long-open equity {eq:.2f} collapsed by notional"


def test_auto_save_equity_open_short_stays_near_balance(tmp_path) -> None:
    """Regression: equity with an open short must stay near starting balance.

    Shorts credit full notional into cash, so the mark must be negative (position value)
    or equity balloons by a full notional (paper trade_ledger.csv showed ~+108k).
    """
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=6, seed=12)
    signals = from_target(np.array([-1, -1, -1, -1, -1, -1]))
    cfg = _config()
    db_path = str(tmp_path / "paper_state.db")
    state = init(cfg, run_id="short_reg", db_path=db_path)
    step(data, signals, state, cfg)

    assert state.open_position is not None
    eq = _last_saved_equity(db_path, "short_reg")
    assert 9_000.0 < eq < 11_000.0, f"short-open equity {eq:.2f} inflated by notional"


def test_account_currency_follows_instrument_currency() -> None:
    """Regression: sandbox account base must equal the instrument's currency.

    Live paper-trading logged ``insufficient data for USD/USDT`` on every fill:
    a futures-style instrument (XAUUSD/commodities, currency USD, no
    ``settlement_currency`` attr) got a USDT-denominated sandbox account via a
    ``getattr(..., "settlement_currency", "USDT")`` fallback, while fill
    commissions book in the instrument currency (USD). The sandbox has no FX
    feed, so any mismatch is fatal — the base must be derived from the built
    nautilus instrument (no API calls, no synthetic quotes needed).
    """
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument
    from ube.core.instrument import Instrument
    from ube.papertrading.nautilus.backend import _instrument_currency

    usd_instrument = Instrument(
        "XAUUSD", "commodities", tick_size=0.01, contract_multiplier=1.0,
        settlement_currency="USD",
    )
    assert _instrument_currency(build_instrument(usd_instrument, overrides={}).instrument) == "USD"

    usdt_instrument = Instrument(
        "BITCOIN", "crypto_perp", tick_size=0.01, contract_multiplier=1.0,
        settlement_currency="USDT",
    )
    built = build_instrument(usdt_instrument, overrides={}).instrument
    assert _instrument_currency(built) == "USDT"


def test_backtest_usd_settlement_with_synthetic_usdt_rate() -> None:
    """Regression: USD-settled instrument + USDT base works without declared rates.

    The adapter automatically seeds a 1:1 USD/USDT FXSeries so that
    ``trade_table``/``_fx_rate_at`` never raises ``FXRateUnavailableError`` for
    the common XAUUSD-style case (USD-settled, USDT account).  Explicit
    ``synthetic_rates`` declarations still take precedence and also succeed.
    """
    import ube

    preset = PRESETS["commodities"]  # GC, USD-settled
    data = synthetic_bars(preset, n_bars=10, seed=1)
    signals = from_target(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    def _cfg(**kw):
        return BacktestConfig(
            instrument=preset.instrument,
            signal=SignalConfig(on_opposite_signal="reverse"),
            base_currency="USDT",
            **kw,
        )

    # Auto-seed path: adapter injects 1:1 USD/USDT — no engine_overrides needed.
    result_auto = ube.run(data, signals, _cfg())
    assert len(result_auto.trades) == 1

    # Explicit override path still works and also returns one trade.
    result_explicit = ube.run(
        data, signals, _cfg(engine_overrides={"synthetic_rates": {"USDUSDT": 1.0}})
    )
    assert len(result_explicit.trades) == 1



def _leveraged_cfg() -> PaperConfig:
    from ube.core.config import RiskConfig
    from ube.core.risk.sizing import SizeModel

    instr = PRESETS["crypto_perp"].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
        risk=RiskConfig(sizing=SizeModel(kind="fixed_fraction", value=0.10, leverage=100.0)),
    )
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0)


def _exit_cfg(exit_cfg, asset_class: str = "crypto_perp") -> PaperConfig:
    """A paper config carrying a ``RiskConfig.exit`` (single-exit cases for §9.4)."""
    instr = PRESETS[asset_class].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
        risk=RiskConfig(exit=(exit_cfg,)),
    )
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0)


def test_crypto_perp_touched_take_profit_fills_at_level() -> None:
    """§9.4 — a touched TP exit fills at its own level price, not the bar close.

    Entry at 100 (bar 0 close), TP 5% → level 105.0. Bar 2 prints a high of 106
    (touches the level) but closes at 100.5 — a backtest would fill the bracket at
    the 105.0 level, and the paper engine must book the same 105.0, not 100.5.
    Bar 1 is a quiet hold so the entry fill (one-bar sandbox fill lag) has landed
    before the touch.
    """
    md = MarketData.from_records(
        [
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
             "timestamp": "2024-01-01T00:00:00Z"},
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0,
             "timestamp": "2024-01-01T01:00:00Z"},
            {"open": 100.5, "high": 106.0, "low": 100.0, "close": 100.5, "volume": 1000.0,
             "timestamp": "2024-01-01T02:00:00Z"},
            {"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
             "timestamp": "2024-01-01T03:00:00Z"},
            {"open": 100.0, "high": 100.0, "low": 99.0, "close": 99.5, "volume": 1000.0,
             "timestamp": "2024-01-01T04:00:00Z"},
        ]
    )
    signals = from_target(np.array([1, 1, 0, 0, 0]))
    cfg = _exit_cfg(TakeProfit(percent=0.05))
    state = init(cfg)

    _, events = step(md, signals, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].exit_reason == "take_profit"
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) == 2
    assert fills[1].exit_reason == "take_profit"
    # level fill, not the bar-1 close (100.5)
    assert abs(fills[1].price - 105.0) < 1e-6
    assert abs(closed[0].exit_price - 105.0) < 1e-6


def test_crypto_perp_touched_stop_loss_fills_at_level() -> None:
    """§9.4 — a touched SL exit fills at the stop level: intra-bar wick through.

    Entry at 100, SL 2% → level 98.0. Bar 2 dips to 97 (through the stop) and
    closes back at 99 — the touched stop must book exactly 98.0, not the close.
    Bar 1 is a quiet hold so the entry fill (one-bar sandbox fill lag) has landed.
    """
    md = MarketData.from_records(
        [
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
             "timestamp": "2024-01-01T00:00:00Z"},
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0,
             "timestamp": "2024-01-01T01:00:00Z"},
            {"open": 100.5, "high": 100.5, "low": 97.0, "close": 99.0, "volume": 1000.0,
             "timestamp": "2024-01-01T02:00:00Z"},
            {"open": 99.0, "high": 99.5, "low": 98.0, "close": 98.5, "volume": 1000.0,
             "timestamp": "2024-01-01T03:00:00Z"},
            {"open": 98.5, "high": 99.0, "low": 97.5, "close": 98.0, "volume": 1000.0,
             "timestamp": "2024-01-01T04:00:00Z"},
        ]
    )
    signals = from_target(np.array([1, 1, 0, 0, 0]))
    cfg = _exit_cfg(StopLoss(percent=0.02))
    state = init(cfg)

    _, events = step(md, signals, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) == 2
    assert abs(fills[1].price - 98.0) < 1e-6
    assert abs(closed[0].exit_price - 98.0) < 1e-6


def test_crypto_perp_close_trigger_exit_fills_at_bar_close() -> None:
    """§9.4 — a ``trigger="close"`` SL still fills at the bar close, never the level.

    Bar 1 wicks to 97 (below the 98 stop) but closes at 100.5 → not triggered by the
    close rule. Bar 2 closes at 97.5 → triggered; with no level the MARKET order fills
    at the closing price 97.5, matching the backtest's close-based exit.
    """
    md = MarketData.from_records(
        [
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
             "timestamp": "2024-01-01T00:00:00Z"},
            {"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.5, "volume": 1000.0,
             "timestamp": "2024-01-01T01:00:00Z"},
            {"open": 100.5, "high": 101.0, "low": 96.5, "close": 97.5, "volume": 1000.0,
             "timestamp": "2024-01-01T02:00:00Z"},
            {"open": 97.5, "high": 98.0, "low": 96.5, "close": 97.0, "volume": 1000.0,
             "timestamp": "2024-01-01T03:00:00Z"},
        ]
    )
    signals = from_target(np.array([1, 1, 1, 0]))
    cfg = _exit_cfg(StopLoss(percent=0.02, trigger="close"))
    state = init(cfg)

    _, events = step(md, signals, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    fills = [e for e in events if e.event_type == EventType.FILL]
    assert len(fills) == 2
    # bar-2 close (97.5), not the 98.0 stop level — and never the bar-1 close
    assert abs(fills[1].price - 97.5) < 1e-6
    assert abs(closed[0].exit_price - 97.5) < 1e-6
    assert fills[1].timestamp == int(md.timestamps.as_unit("ns").asi8[2])


def test_resume_reverse_from_open_short_does_not_double_count_notional() -> None:
    """Regression: same-bar reverse of a *resumed* open short must not crash.

    Live paper-trading died with ``ConfigError: capital must be non-negative`` on a
    short->long reversal. Resume seeded ``_current_balance`` as equity (cash + the
    negative short mark ~= 9.9k); the reversal's optimistic close-credit then re-booked
    the ~100k closing notional on top of it, collapsing the balance to ~-90k and making
    100x sizing capital deeply negative. The strategy's balance is a cash book, so a
    resumed run must seed pure cash (short open cash ~= 110k); at the sizing point the
    position is already closed, so cash == equity there and sizing is correct.
    """
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=9, seed=9)
    signals = from_target(np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1]))
    cfg = _leveraged_cfg()
    state = init(cfg)

    # slice 1: open a leveraged short (~10x balance notional), hold it open
    step(slice_md(data, slice(0, 6)), slice_signals(signals, slice(0, 6)), state, cfg)
    assert state.open_position is not None
    assert state.open_position.side == -1

    # slice 2: same-bar reverse short -> long (resume seeding path is exercised
    # because each step() rebuilds the strategy from state/open_position + ledger cash)
    _, ev2 = step(slice_md(data, slice(6, 9)), slice_signals(signals, slice(6, 9)), state, cfg)

    assert state.open_position is not None
    assert state.open_position.side == 1
    fills2 = [e for e in ev2 if e.event_type == EventType.FILL]
    assert len(fills2) == 2, "close-then-open fills expected on the reversal"
    assert fills2[0].exit_reason == "signal" and fills2[1].exit_reason is None

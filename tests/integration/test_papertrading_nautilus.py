"""Integration test: nautilus sandbox paper backend via ``ube.paper.step`` (plan T3).

Drives a real Nautilus ``TradingNode`` + ``SandboxExecutionClient`` over synthetic
``crypto_perp`` bars. Skips silently when nautilus-trader is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

nautilus = pytest.importorskip("nautilus_trader")

from ube.core.config import BacktestConfig, SignalConfig  # noqa: E402
from ube.core.data import MarketData  # noqa: E402
from ube.core.ledger import EventType, trades  # noqa: E402
from ube.core.signals import from_target  # noqa: E402
from ube.papertrading import init, run_auto, step  # noqa: E402
from ube.papertrading.config import PaperConfig  # noqa: E402
from ube.testing.synthetic import PRESETS, synthetic_bars  # noqa: E402


def _config(asset_class: str = "crypto_perp", **kw) -> PaperConfig:
    instr = PRESETS[asset_class].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0, **kw)


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


def test_crypto_perp_resume() -> None:
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=10, seed=1)

    def sig_fn(d: MarketData) -> int:
        return 1 if d.n_bars <= 6 else 0

    cfg = _config()
    state = init(cfg)
    run_auto(data, sig_fn, cfg, state)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].exit_reason == "signal"
    assert state.open_position is None

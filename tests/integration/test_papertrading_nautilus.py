"""Integration test: nautilus sandbox paper backend via ``ube.paper.step`` (plan T3).

Drives a real Nautilus ``TradingNode`` + ``SandboxExecutionClient`` over synthetic
``crypto_perp`` bars. Skips silently when nautilus-trader is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

nautilus = pytest.importorskip("nautilus_trader")

from ube.core.config import BacktestConfig, SignalConfig
from ube.core.ledger import EventType, trades
from ube.core.signals import from_target
from ube.papertrading import init, step
from ube.papertrading.config import PaperConfig
from ube.testing.synthetic import PRESETS, synthetic_bars


def _config(asset_class: str = "crypto_perp", **kw) -> PaperConfig:
    instr = PRESETS[asset_class].instrument
    bc = BacktestConfig(
        instrument=instr,
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    return PaperConfig(base=bc, engine="nautilus", starting_balance=10_000.0, **kw)


def test_crypto_perp_entry_then_exit() -> None:
    data = synthetic_bars(PRESETS["crypto_perp"], n_bars=10, seed=1)
    # long entry for 5 bars, then a long-exit signal (one closed long trade).
    target = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    signals = from_target(target)
    cfg = _config()
    state = init(cfg)

    _, events = step(data, signals, state, cfg)

    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert closed[0].exit_reason == "signal"

    kinds = [e.kind for e in events]
    assert EventType.FILL in kinds
    assert EventType.COMMISSION in kinds
    assert EventType.POSITION_CHANGE in kinds
    # entry fill then exit fill
    fills = [e for e in events if e.kind == EventType.FILL]
    assert len(fills) == 2
    assert fills[0].exit_reason is None
    assert fills[1].exit_reason == "signal"


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
    assert all(e.kind != EventType.FILL for e in events)
    assert state.open_position is None

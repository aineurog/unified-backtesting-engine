"""Unit tests for the engine-agnostic paper front-end (T2 / §9.1, §9.3, §9.6).

Uses the dependency-free ``recording`` backend (plan T2) — no nautilus import.
"""

from __future__ import annotations

import numpy as np
import pytest

from ube.core.config import BacktestConfig, SignalConfig
from ube.core.errors import InvalidSignalError
from ube.core.instrument import Instrument
from ube.core.ledger import trades
from ube.core.signals import from_target
from ube.papertrading.config import PaperConfig
from ube.papertrading.core import init, run_auto, step
from ube.papertrading.errors import DuplicateBarError
from ube.papertrading.state import OpenPosition
from ube.testing.synthetic import PRESETS, synthetic_bars


def _config(
    asset_class: str = "crypto_perp",
    *,
    engine: str = "recording",
    policy: str = "reverse",
) -> PaperConfig:
    instrument = PRESETS[asset_class].instrument
    base = BacktestConfig(
        instrument=instrument,
        signal=SignalConfig(on_opposite_signal=policy),
    )
    return PaperConfig(base=base, engine=engine)


def _data(asset_class: str = "crypto_perp", n_bars: int = 20):
    return synthetic_bars(asset_class, n_bars=n_bars)


def test_step_opens_and_closes_one_trade() -> None:
    data = _data(n_bars=12)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    signals = from_target(target)
    cfg = _config()
    state = init(cfg)
    _state, events = step(data, signals, state, cfg)

    assert state.last_processed_ns is not None
    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert state.open_position is None
    # The open fill + close fill + commission(s) + position changes exist.
    kinds = {e.event_type for e in events}
    assert "fill" in kinds and "position_change" in kinds


def test_idempotency_rejects_duplicate_bars() -> None:
    data = _data(n_bars=10)
    signals = from_target(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0]))
    cfg = _config()
    state = init(cfg)
    step(data, signals, state, cfg)
    # Re-feeding the same bars must be rejected (§9.6).
    with pytest.raises(DuplicateBarError):
        step(data, signals, state, cfg)


def test_run_auto_streams_signal_fn() -> None:
    data = _data(n_bars=12)
    cfg = _config()

    def fn(window):
        i = window.n_bars - 1
        if 4 <= i <= 7:
            return 1
        return 0

    state = init(cfg)
    run_auto(data, fn, cfg, state=state)
    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert state.last_processed_ns is not None


def test_on_opposite_signal_reverse() -> None:
    data = _data(n_bars=8)
    target = np.array([1, 1, 1, -1, -1, -1, -1, -1])
    signals = from_target(target)
    cfg = _config(policy="reverse")
    state = init(cfg)
    step(data, signals, state, cfg)
    # reverse: long then short → the long trade is closed and a short is open.
    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1
    assert state.open_position is not None
    assert state.open_position.side == -1


def test_on_opposite_signal_exit_only() -> None:
    data = _data(n_bars=8)
    target = np.array([1, 1, 1, -1, -1, -1, -1, -1])
    signals = from_target(target)
    cfg = _config(policy="exit_only")
    state = init(cfg)
    step(data, signals, state, cfg)
    # exit_only: long then flat (no short opened).
    instr = cfg.base.instrument
    closed = trades(state.ledger, instruments={instr.symbol: instr})
    assert len(closed) == 1
    assert closed[0].side == 1


def test_on_opposite_signal_ignore() -> None:
    data = _data(n_bars=8)
    target = np.array([1, 1, 1, -1, -1, -1, -1, -1])
    signals = from_target(target)
    cfg = _config(policy="ignore")
    state = init(cfg)
    step(data, signals, state, cfg)
    # ignore: the short signal is dropped; only the long trade (still open) exists.
    assert state.open_position is not None
    assert state.open_position.side == 1


def test_long_only_rejects_short() -> None:
    # A crypto_spot instrument is long-only: a short_entry is a configuration error
    # raised before the engine runs (validate_long_only, §4.5/§4.7).
    data = _data(n_bars=8)
    target = np.array([0, 0, 0, -1, -1, -1, -1, -1])
    signals = from_target(target)
    instr = Instrument("BTC", "crypto_spot")
    cfg = PaperConfig(
        base=BacktestConfig(
            instrument=instr, signal=SignalConfig(on_opposite_signal="reverse")
        ),
        engine="recording",
    )
    state = init(cfg)
    with pytest.raises(InvalidSignalError):
        step(data, signals, state, cfg)


def test_open_position_recomputed_from_ledger() -> None:
    data = _data(n_bars=6)
    target = np.array([1, 1, 1, 1, 1, 1])
    signals = from_target(target)
    cfg = _config()
    state = init(cfg)
    step(data, signals, state, cfg)
    assert isinstance(state.open_position, OpenPosition)
    assert state.open_position.side == 1
    assert state.open_position.quantity == pytest.approx(1.0)

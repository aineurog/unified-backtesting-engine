"""Integration: paper-trading statistics persisted in sqlite (§9.5, §10).

Bars are not stored in paper mode, so the returns tier is rebuilt from the equity points
``step()`` persists and refreshed whenever a trade closes. These tests drive a real
nautilus sandbox over synthetic ``crypto_perp`` bars.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")

import ube.papertrading.state as paper_state_module  # noqa: E402
from ube.core.config import BacktestConfig, SignalConfig  # noqa: E402
from ube.core.data import MarketData  # noqa: E402
from ube.core.ledger import trades as ledger_trades  # noqa: E402
from ube.core.signals import Signals, from_target  # noqa: E402
from ube.papertrading import init, step  # noqa: E402
from ube.papertrading.config import PaperConfig  # noqa: E402
from ube.papertrading.state import load_equity  # noqa: E402
from ube.stats import StatsResult, load_stats  # noqa: E402
from ube.testing.synthetic import PRESETS, synthetic_bars  # noqa: E402

TARGET = np.array([1] * 5 + [0] * 3 + [1] * 5 + [0] * 5)


def _slice_md(md: MarketData, sl: slice) -> MarketData:
    sliced = MarketData.__new__(MarketData)  # type: ignore[call-arg]
    for field in ("open", "high", "low", "close", "volume", "index"):
        object.__setattr__(sliced, field, getattr(md, field)[sl])
    return sliced


def _slice_signals(sig: Signals, sl: slice) -> Signals:
    return Signals(
        long_entry=sig.long_entry[sl].copy(),
        long_exit=sig.long_exit[sl].copy(),
        short_entry=sig.short_entry[sl].copy(),
        short_exit=sig.short_exit[sl].copy(),
    )


def _setup(tmp_path):
    md = synthetic_bars(PRESETS["crypto_perp"], n_bars=18, seed=1)
    sig = from_target(TARGET)
    cfg = PaperConfig(
        base=BacktestConfig(
            instrument=PRESETS["crypto_perp"].instrument,
            signal=SignalConfig(on_opposite_signal="reverse"),
        ),
        engine="nautilus",
        starting_balance=10_000.0,
        state_path=str(tmp_path / "paper.db"),
    )
    state = init(cfg, db_path=str(tmp_path / "paper.db"))
    return md, sig, cfg, state


def _drive(md, sig, cfg, state, *, until_trades: int, start: int = 1) -> int:
    """Feed one new bar per step (the window must start after the idempotency cursor)."""
    for n in range(start, md.n_bars + 1):
        step(_slice_md(md, slice(n - 1, n)), _slice_signals(sig, slice(n - 1, n)), state, cfg)
        if state.stats_trade_count >= until_trades:
            return n
    return md.n_bars


def test_step_persists_stats_snapshot_on_trade_close(tmp_path) -> None:
    md, sig, cfg, state = _setup(tmp_path)
    db = str(tmp_path / "paper.db")

    _drive(md, sig, cfg, state, until_trades=1)
    assert state.stats_trade_count == 1
    first = load_stats(db, "default")
    assert not first.empty
    assert set(first["trade_count"]) == {1}
    assert set(first["tiers"]) == {"returns,trades,exposure"}
    assert set(first["errors_json"]) == {"{}"}
    assert "sharpe_ratio" in set(first["metric"])

    _drive(md, sig, cfg, state, until_trades=2, start=len(load_equity(db, "default")) + 1)
    assert state.stats_trade_count == 2
    latest = load_stats(db, "default")
    assert set(latest["trade_count"]) == {2}
    assert int(latest["seq"].iloc[0]) == 2  # append-only, newest wins
    assert set(load_stats(db, "default", seq=1)["trade_count"]) == {1}


def test_stats_snapshot_equity_stops_at_latest_closed_trade(tmp_path) -> None:
    md, sig, cfg, state = _setup(tmp_path)
    db = str(tmp_path / "paper.db")
    closed_on = _drive(md, sig, cfg, state, until_trades=1)

    # Continue into the next open position without closing another trade.
    for n in range(closed_on + 1, min(closed_on + 4, md.n_bars + 1)):
        step(
            _slice_md(md, slice(n - 1, n)),
            _slice_signals(sig, slice(n - 1, n)),
            state,
            cfg,
        )

    assert state.open_position is not None
    closed = ledger_trades(
        state.ledger,
        {cfg.base.instrument.symbol: cfg.base.instrument},
    )
    latest_exit_ns = max(trade.exit_timestamp for trade in closed)
    equity = load_equity(db, "default")
    equity_ns = equity["timestamp"].astype("int64")
    expected_snapshot_points = int((equity_ns <= latest_exit_ns).sum())
    assert len(equity) > expected_snapshot_points

    snapshot = load_stats(db, "default")
    assert set(snapshot["n_equity_points"]) == {expected_snapshot_points}


def test_snapshot_is_not_rewritten_when_no_trade_closes(tmp_path) -> None:
    md, _, cfg, state = _setup(tmp_path)
    db = str(tmp_path / "paper.db")
    sig = from_target(np.array([1] * 5 + [0] * 13))

    n = _drive(md, sig, cfg, state, until_trades=1)
    assert state.stats_trade_count == 1

    # keep stepping bars that close nothing: the latest snapshot stays seq 1
    for k in range(n + 1, md.n_bars + 1):
        step(_slice_md(md, slice(k - 1, k)), _slice_signals(sig, slice(k - 1, k)), state, cfg)
    assert int(load_stats(db, "default")["seq"].iloc[0]) == 1


def test_resume_does_not_write_duplicate_trade_count_snapshot(tmp_path) -> None:
    md, sig, cfg, state = _setup(tmp_path)
    db = str(tmp_path / "paper.db")
    closed_on = _drive(md, sig, cfg, state, until_trades=1)
    assert state.stats_trade_count == 1

    resumed = type(state).load(db, run_id="default")
    resumed.db_path = db
    resumed.run_id = "default"
    no_new_trades = from_target(np.zeros(md.n_bars, dtype=np.int64))
    for n in range(closed_on + 1, md.n_bars + 1):
        step(
            _slice_md(md, slice(n - 1, n)),
            _slice_signals(no_new_trades, slice(n - 1, n)),
            resumed,
            cfg,
        )

    latest = load_stats(db, "default")
    assert int(latest["seq"].iloc[0]) == 1
    assert set(latest["trade_count"]) == {1}


def test_state_stats_uses_returns_persisted_with_equity(tmp_path, monkeypatch) -> None:
    md, sig, cfg, state = _setup(tmp_path)
    db = str(tmp_path / "paper.db")
    _drive(md, sig, cfg, state, until_trades=1)

    equity = load_equity(db, "default")
    equity_values = equity["equity"].to_numpy()
    expected_returns = np.r_[0.0, equity_values[1:] / equity_values[:-1] - 1.0]
    np.testing.assert_allclose(equity["returns"].to_numpy(), expected_returns)

    captured: dict[str, np.ndarray] = {}
    original_from_arrays = paper_state_module.stats_from_arrays

    def capture_from_arrays(*args, **kwargs):
        captured["returns"] = np.asarray(kwargs["equity_returns"])
        return original_from_arrays(*args, **kwargs)

    monkeypatch.setattr(paper_state_module, "stats_from_arrays", capture_from_arrays)
    snapshot = state.stats(md, cfg)
    assert isinstance(snapshot, StatsResult)
    assert snapshot.meta["equity_source"] == "persisted"
    assert snapshot.meta["n_equity_points"] == len(load_equity(db, "default"))
    np.testing.assert_allclose(captured["returns"], expected_returns)
    assert "returns" in snapshot.tiers
    assert "benchmark" not in snapshot.tiers  # no benchmark is attached in paper mode
    assert snapshot.n_metrics > 0

    # the inner BacktestConfig is accepted too
    inner = state.stats(md, cfg.base)
    assert inner.n_metrics == snapshot.n_metrics


def test_state_stats_without_history_degrades(tmp_path) -> None:
    md, _, cfg, state = _setup(tmp_path)
    snapshot = state.stats(md, cfg)
    assert isinstance(snapshot, StatsResult)
    assert snapshot.n_metrics == 0
    assert snapshot.tiers == ()
    assert "compute" in snapshot.errors
    assert snapshot.meta["equity_source"] == "none"

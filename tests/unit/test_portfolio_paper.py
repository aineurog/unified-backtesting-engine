"""Shared-capital paper trading, exercised with the dependency-free recording broker.

The recording backend is deliberately engine-agnostic, so these tests pin the portfolio
front-end (``init`` / ``step`` / ``run`` / ``run_auto`` / persistence) without importing
any optional engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ube
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.errors import ConfigError, DuplicateBarError, FXRateUnavailableError
from ube.core.instrument import Instrument
from ube.core.ledger import trades
from ube.core.portfolio import EqualWeight, FixedWeight, PortfolioConfig
from ube.core.risk import RiskConfig, SizeModel
from ube.papertrading import (
    PaperConfig,
    PortfolioPaperState,
    init,
    run,
    run_auto,
    step,
)
from ube.papertrading.core import _refresh_portfolio_stats_snapshot
from ube.papertrading.state import load_equity, load_trades


def _bars(step: float, n: int = 24, start: str = "2025-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    path = np.linspace(100.0, 100.0 + step, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": path,
            "high": path + 1.0,
            "low": path - 1.0,
            "close": path + 0.2,
            "volume": 500,
        }
    )


def _market(step: float = 10.0, n: int = 24, start: str = "2025-01-01"):
    return ube.MarketData.from_dataframe(
        _bars(step, n, start=start), timestamp_col="timestamp"
    )


def _targets(n: int = 24) -> np.ndarray:
    # Three round trips: flat, long, flat, long, flat, long, flat.
    values = np.zeros(n, dtype=np.int64)
    values[2:5] = 1
    values[8:10] = 1
    values[14:17] = 1
    return values


def _setup(*, allocation=None, engine: str = "recording", starting_balance: float = 100_000.0):
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    portfolio = PortfolioConfig(
        instruments=(btc, eth),
        base_currency="USDT",
        allocation=allocation or EqualWeight(),
    )
    base = BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=portfolio,
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        signal=SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": starting_balance},
    )
    config = PaperConfig(
        base=base,
        engine=engine,
        starting_balance=starting_balance,
    )
    return config, portfolio, ube.from_target(_targets())


def test_init_creates_one_leg_state_per_instrument(tmp_path):
    config, _, _ = _setup()
    state = init(config, run_id="p1", db_path=str(tmp_path / "paper.db"))

    assert isinstance(state, PortfolioPaperState)
    assert state.instrument_ids == ("BTC-USDT", "ETH-USDT")
    assert set(state.states) == {"BTC-USDT", "ETH-USDT"}
    assert state.last_processed_ns is None
    assert state.open_positions == {"BTC-USDT": None, "ETH-USDT": None}


def test_step_trades_every_leg_and_combines_the_ledger(tmp_path):
    config, portfolio, signals = _setup()
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="p1", db_path=db)
    btc, eth = _market(10.0), _market(20.0)

    state, events = step(
        {"BTC-USDT": btc, "ETH-USDT": eth},
        {"BTC-USDT": signals, "ETH-USDT": signals},
        state,
        config,
    )

    assert events
    combined = list(trades(state.ledger, instruments=portfolio.instrument_map))
    assert len(combined) == 6
    assert {trade.instrument_id for trade in combined} == {"BTC-USDT", "ETH-USDT"}
    assert state.open_positions == {"BTC-USDT": None, "ETH-USDT": None}
    # The parent run persists one trade table and one combined equity point.
    assert len(load_trades(db, "p1")) == 6
    assert len(load_equity(db, "p1")) == 1


def test_step_rejects_mismatched_mapping_keys(tmp_path):
    config, _, signals = _setup()
    state = init(config, run_id="p1", db_path=str(tmp_path / "paper.db"))
    with pytest.raises(ConfigError):
        step(
            {"BTC-USDT": _market()},
            {"BTC-USDT": signals, "ETH-USDT": signals},
            state,
            config,
        )


def test_fixed_weight_zero_allocation_skips_that_leg(tmp_path):
    config, portfolio, signals = _setup(
        allocation=FixedWeight(weights={"BTC-USDT": 1.0, "ETH-USDT": 0.0})
    )
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="p1", db_path=db)
    state, _ = step(
        {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)},
        {"BTC-USDT": signals, "ETH-USDT": signals},
        state,
        config,
    )
    combined = list(trades(state.ledger, instruments=portfolio.instrument_map))
    assert {trade.instrument_id for trade in combined} == {"BTC-USDT"}
    assert state.states["ETH-USDT"].ledger.events == ()


def test_fixed_weight_init_tracks_unallocated_cash(tmp_path):
    config, _, _ = _setup(
        allocation=FixedWeight(weights={"BTC-USDT": 0.5, "ETH-USDT": 0.3})
    )
    state = init(config, run_id="p1", db_path=str(tmp_path / "paper.db"))

    assert state.unallocated_cash == pytest.approx(20_000.0)


def test_paper_config_rejects_currency_disagreement():
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    base_portfolio = PortfolioConfig(instruments=(btc, eth), base_currency="USDT")
    base = BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=base_portfolio,
        engine_overrides={"starting_balance": 100_000.0},
    )
    # Only the paper-level portfolio disagrees, so BacktestConfig alone cannot catch it.
    paper_portfolio = PortfolioConfig(instruments=(btc, eth), base_currency="USD")

    with pytest.raises(ConfigError):
        PaperConfig(base=base, portfolio=paper_portfolio)


def test_portfolio_init_rejects_unconvertible_settlement(tmp_path):
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-EUR", asset_class="crypto_perp", settlement_currency="EUR")
    portfolio = PortfolioConfig(instruments=(btc, eth), base_currency="USDT")
    base = BacktestConfig(
        instrument=btc,
        portfolio=portfolio,
        engine_overrides={"starting_balance": 100_000.0},
    )
    config = PaperConfig(base=base, engine="recording")

    with pytest.raises(FXRateUnavailableError):
        init(config, run_id="p1", db_path=str(tmp_path / "paper.db"))


def test_portfolio_stats_uses_stablecoin_peg_for_usd_base(tmp_path):
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    portfolio = PortfolioConfig(instruments=(btc, eth), base_currency="USD")
    base = BacktestConfig(
        instrument=btc,
        base_currency="USD",
        portfolio=portfolio,
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        signal=SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": 100_000.0},
    )
    config = PaperConfig(base=base, engine="recording")
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="p1", db_path=db)
    data = {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)}
    signals = {"BTC-USDT": ube.from_target(_targets()), "ETH-USDT": ube.from_target(_targets())}

    state, _ = step(data, signals, state, config)
    stats = state.stats(data, config)

    # USDT -> USD is the documented peg, so the combined curve derives without raising.
    assert stats.meta["equity_source"] == "ledger"
    assert stats.meta["is_portfolio"] is True


def test_portfolio_snapshot_skipped_when_trade_count_unchanged(tmp_path, monkeypatch):
    config, _, _ = _setup()
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="p1", db_path=db)
    data = {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)}
    first_bar_ns = int(_market(10.0).timestamps.as_unit("ns").asi8[0])
    saved: list[int] = []
    monkeypatch.setattr("ube.stats.save_stats", lambda *a, **k: saved.append(1))

    # No snapshot yet: the first closed trade triggers a recompute.
    monkeypatch.setattr("ube.stats.load_stats", lambda *a, **k: pd.DataFrame())
    _refresh_portfolio_stats_snapshot(
        db,
        "p1",
        data,
        config,
        state,
        [SimpleNamespace(exit_timestamp=first_bar_ns)],
    )
    assert saved == [1]
    assert state.stats_trade_count == 1

    # A snapshot already covering that trade count is not recomputed.
    snapshot = pd.DataFrame(
        [{"trade_count": 1, "n_equity_points": 0, "tiers": "returns,trades"}]
    )
    monkeypatch.setattr("ube.stats.load_stats", lambda *a, **k: snapshot)
    _refresh_portfolio_stats_snapshot(
        db,
        "p1",
        data,
        config,
        state,
        [SimpleNamespace(exit_timestamp=first_bar_ns)],
    )
    assert saved == [1]


def test_portfolio_state_round_trips_through_sqlite(tmp_path):
    config, _, signals = _setup()
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="p1", db_path=db)
    state, _ = step(
        {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)},
        {"BTC-USDT": signals, "ETH-USDT": signals},
        state,
        config,
    )

    loaded = PortfolioPaperState.load(db, "p1")
    assert loaded.instrument_ids == state.instrument_ids
    assert loaded.last_processed_ns == state.last_processed_ns
    assert loaded.unallocated_cash == pytest.approx(state.unallocated_cash)
    assert len(list(trades(loaded.ledger))) == len(list(trades(state.ledger)))


def test_run_loads_an_existing_session_and_processes_new_bars(tmp_path):
    config, _, _ = _setup()
    db = str(tmp_path / "paper.db")
    early_btc, early_eth = _market(10.0, n=12), _market(20.0, n=12)
    new_btc = _market(10.0, n=12, start="2025-01-01 12:00")
    new_eth = _market(20.0, n=12, start="2025-01-01 12:00")
    early_signals = ube.from_target(_targets(12))
    new_signals = ube.from_target(_targets(24)[12:])

    state, first = run(
        "scheduled",
        {"BTC-USDT": early_btc, "ETH-USDT": early_eth},
        {"BTC-USDT": early_signals, "ETH-USDT": early_signals},
        config,
        db_path=db,
    )
    assert first

    state, second = run(
        "scheduled",
        {"BTC-USDT": new_btc, "ETH-USDT": new_eth},
        {"BTC-USDT": new_signals, "ETH-USDT": new_signals},
        config,
        db_path=db,
    )
    assert isinstance(state, PortfolioPaperState)
    assert second


def test_run_rejects_stale_bars(tmp_path):
    config, _, signals = _setup()
    db = str(tmp_path / "paper.db")
    data = {"BTC-USDT": _market(), "ETH-USDT": _market(20.0)}
    signals_map = {"BTC-USDT": signals, "ETH-USDT": signals}
    run("scheduled", data, signals_map, config, db_path=db)
    with pytest.raises(DuplicateBarError):
        run("scheduled", data, signals_map, config, db_path=db)


def test_run_auto_streams_a_callable_per_leg(tmp_path):
    config, _, _ = _setup()
    db = str(tmp_path / "paper.db")
    state = init(config, run_id="auto", db_path=db)

    def signal_fn(md):
        return 1 if md.n_bars >= 3 else 0

    events = run_auto(
        {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)},
        {"BTC-USDT": signal_fn, "ETH-USDT": signal_fn},
        config,
        state,
        run_id="auto",
    )
    assert events
    # A second call with no new bars is a no-op.
    assert (
        run_auto(
            {"BTC-USDT": _market(10.0), "ETH-USDT": _market(20.0)},
            {"BTC-USDT": signal_fn, "ETH-USDT": signal_fn},
            config,
            state,
            run_id="auto",
        )
        == []
    )

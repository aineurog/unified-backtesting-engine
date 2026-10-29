"""Shared-capital portfolio backtest and paper trading across all three engines.

The single-instrument path is asserted alongside the portfolio path so the new mode is
additive: a plain run and a ``PortfolioConfig`` run both work for Nautilus, VectorBT and
Backtrader, over the same canonical inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ube
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.instrument import Instrument
from ube.core.ledger import trades
from ube.core.portfolio import EqualWeight, FixedWeight, PortfolioConfig
from ube.core.risk import RiskConfig, SizeModel
from ube.papertrading import PaperConfig, PortfolioPaperState
from ube.papertrading import run as paper_run

ENGINES = ("vectorbt", "backtrader", "nautilus")


def _bars(drift: float, n: int = 24) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    path = np.linspace(100.0, 100.0 + drift, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": path,
            "high": path + 1.0,
            "low": path - 1.0,
            "close": path + 0.2,
            "volume": 1_000,
        }
    )


def _targets(n: int = 24) -> np.ndarray:
    values = np.zeros(n, dtype=np.int64)
    values[2:5] = 1
    values[9:12] = 1
    values[16:20] = 1
    return values


def _portfolio(allocation=None) -> PortfolioConfig:
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    return PortfolioConfig(
        instruments=(btc, eth),
        base_currency="USDT",
        allocation=allocation or EqualWeight(),
    )


def _data() -> dict[str, ube.MarketData]:
    return {
        "BTC-USDT": ube.MarketData.from_dataframe(_bars(12.0), timestamp_col="timestamp"),
        "ETH-USDT": ube.MarketData.from_dataframe(_bars(22.0), timestamp_col="timestamp"),
    }


def _config(engine: str, portfolio: PortfolioConfig | None = None) -> BacktestConfig:
    portfolio = portfolio or _portfolio()
    return BacktestConfig(
        instrument=portfolio.instruments[0],
        base_currency=portfolio.base_currency,
        portfolio=portfolio,
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        signal=SignalConfig(on_opposite_signal="reverse"),
        engine=engine,
        engine_overrides={"starting_balance": 100_000.0},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_single_asset_run_is_unchanged(engine: str):
    data = _data()["BTC-USDT"]
    signals = ube.from_target(_targets())
    cfg = BacktestConfig(
        instrument=_portfolio().instruments[0],
        base_currency="USDT",
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        engine=engine,
        engine_overrides={"starting_balance": 100_000.0},
    )

    result = ube.run(data, signals, cfg)

    assert cfg.portfolio is None
    assert result.equity_curve.equity[0] == pytest.approx(100_000.0)
    assert set(result.equity_curve_by_instrument) == {"BTC-USDT"}
    assert len(result.trades) == 3


@pytest.mark.parametrize("engine", ENGINES)
def test_portfolio_backtest_combines_every_leg(engine: str):
    data = _data()
    signals = {symbol: ube.from_target(_targets()) for symbol in data}
    config = _config(engine)

    result = ube.run(data, signals, config)

    assert len(result.equity_curve.equity) == 24
    assert result.equity_curve.equity[0] == pytest.approx(100_000.0)
    assert set(result.equity_curve_by_instrument) == set(data)
    assert {trade.instrument_id for trade in result.trades} == set(data)
    # Three round trips per leg, both legs traded.
    assert len(result.trades) == 6
    assert result.stats is not None
    assert result.stats.n_metrics > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_fixed_weight_portfolio_runs(engine: str):
    allocation = FixedWeight(weights={"BTC-USDT": 0.75, "ETH-USDT": 0.25})
    data = _data()
    signals = {symbol: ube.from_target(_targets()) for symbol in data}

    result = ube.run(data, signals, _config(engine, _portfolio(allocation)))

    assert {trade.instrument_id for trade in result.trades} == set(data)
    assert result.equity_curve.equity[0] == pytest.approx(100_000.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_portfolio_paper_session_combines_legs(engine: str, tmp_path):
    portfolio = _portfolio()
    data = _data()
    signals = {symbol: ube.from_target(_targets()) for symbol in data}
    base = BacktestConfig(
        instrument=portfolio.instruments[0],
        base_currency=portfolio.base_currency,
        portfolio=portfolio,
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        signal=SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": 100_000.0},
    )
    config = PaperConfig(
        base=base,
        engine=engine,
        starting_balance=100_000.0,
        state_path=str(tmp_path / f"{engine}.db"),
    )

    state, events = paper_run("portfolio", data, signals, config)

    assert isinstance(state, PortfolioPaperState)
    assert events
    combined = list(trades(state.ledger, instruments=portfolio.instrument_map))
    assert {trade.instrument_id for trade in combined} == set(data)
    assert len(combined) == 6
    assert state.stats(data, config).meta["is_portfolio"] is True

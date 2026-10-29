"""Cross-engine parity for the shared-capital portfolio path (§16).

The existing parity suite pins the single-instrument contract. This file does the same
job for a portfolio: the same two instruments, the same allocation and the same signals
must produce a comparable *combined* equity curve through Nautilus, VectorBT and
Backtrader, because every leg still folds through the one canonical ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ube
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.instrument import Instrument
from ube.core.portfolio import EqualWeight, FixedWeight, PortfolioConfig
from ube.core.risk import RiskConfig, SizeModel

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


def _run(engine: str, allocation):
    btc = Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    eth = Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")
    portfolio = PortfolioConfig(
        instruments=(btc, eth), base_currency="USDT", allocation=allocation
    )
    config = BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=portfolio,
        risk=RiskConfig(sizing=SizeModel(kind="all_in")),
        signal=SignalConfig(on_opposite_signal="reverse"),
        engine=engine,
        engine_overrides={"starting_balance": 100_000.0},
    )
    data = {
        "BTC-USDT": ube.MarketData.from_dataframe(_bars(12.0), timestamp_col="timestamp"),
        "ETH-USDT": ube.MarketData.from_dataframe(_bars(22.0), timestamp_col="timestamp"),
    }
    signals = {symbol: ube.from_target(_targets()) for symbol in data}
    return ube.run(data, signals, config)


@pytest.mark.parametrize(
    "allocation",
    [EqualWeight(), FixedWeight({"BTC-USDT": 0.75, "ETH-USDT": 0.25})],
)
def test_portfolio_equity_is_comparable_across_engines(allocation):
    results = {engine: _run(engine, allocation) for engine in ENGINES}
    finals = {engine: float(result.equity_curve.equity[-1]) for engine, result in results.items()}
    reference = finals["vectorbt"]

    for engine, final in finals.items():
        assert final == pytest.approx(reference, rel=1e-3), (engine, final, reference)
        assert len(results[engine].trades) == len(results["vectorbt"].trades)

    for _engine, result in results.items():
        assert set(result.equity_curve_by_instrument) == {"BTC-USDT", "ETH-USDT"}
        assert result.equity_curve.equity[0] == pytest.approx(100_000.0)

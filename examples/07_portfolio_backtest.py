"""Portfolio backtests: single-asset vs multi-asset on every engine.

A *single-asset* run uses one instrument and one capital bucket, exactly as before.
A *multi-asset portfolio* run sets ``BacktestConfig.portfolio`` and passes a
``{instrument_id: MarketData}`` mapping plus per-instrument signals. Every leg is
delegated to the same engine adapter, but each leg is seeded with its share of one
shared account and the resulting ledgers are merged into one combined equity curve.

Run from the repository root:

    python examples/07_portfolio_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ube  # noqa: E402

STARTING_BALANCE = 100_000.0


def make_bars(drift: float, periods: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="h", tz="UTC")
    path = np.linspace(100.0, 100.0 + drift, periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": path,
            "high": path + 1.0,
            "low": path - 1.0,
            "close": path + 0.2,
            "volume": 1_000,
        }
    )


def make_signals(periods: int = 30) -> ube.Signals:
    target = np.zeros(periods, dtype=np.int64)
    target[3:7] = 1
    target[13:17] = 1
    target[23:27] = 1
    return ube.from_target(target)


def instruments() -> tuple[ube.Instrument, ube.Instrument]:
    return (
        ube.Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
        ube.Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
    )


def base_config(engine: str, instrument: ube.Instrument) -> ube.BacktestConfig:
    return ube.BacktestConfig(
        instrument=instrument,
        base_currency="USDT",
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        engine=engine,
        engine_overrides={"starting_balance": STARTING_BALANCE},
    )


def run_single_asset(engine: str) -> None:
    btc, _ = instruments()
    data = ube.MarketData.from_dataframe(make_bars(12.0), timestamp_col="timestamp")
    result = ube.run(data, make_signals(), base_config(engine, btc))

    print(f"  single asset  [{engine}]")
    print(f"    start/end equity : {result.equity_curve.equity[0]:,.2f} -> {result.equity_curve.equity[-1]:,.2f}")
    print(f"    trades           : {len(result.trades)}")


def run_multi_asset(engine: str, allocation: ube.AllocationPolicy) -> None:
    btc, eth = instruments()
    portfolio = ube.PortfolioConfig(
        instruments=(btc, eth),
        base_currency="USDT",
        allocation=allocation,
    )
    config = ube.BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=portfolio,
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        engine=engine,
        engine_overrides={"starting_balance": STARTING_BALANCE},
    )
    data = {
        "BTC-USDT": ube.MarketData.from_dataframe(make_bars(12.0), timestamp_col="timestamp"),
        "ETH-USDT": ube.MarketData.from_dataframe(make_bars(22.0), timestamp_col="timestamp"),
    }
    signals = {symbol: make_signals() for symbol in data}
    result = ube.run(data, signals, config)

    by_instrument = ", ".join(
        f"{symbol}={sum(1 for trade in result.trades if trade.instrument_id == symbol)} trades"
        for symbol in result.equity_curve_by_instrument
    )
    print(f"  multi asset   [{engine}] {type(allocation).__name__}")
    print(f"    combined equity  : {result.equity_curve.equity[0]:,.2f} -> {result.equity_curve.equity[-1]:,.2f}")
    print(f"    per instrument   : {by_instrument}")
    print(f"    trades (total)   : {len(result.trades)}")


def main() -> None:
    ube.ensure_builtin_engines_registered()
    engines = [
        name
        for name in ("vectorbt", "backtrader", "nautilus")
        if name in ube.registered_engines()
    ]
    if not engines:
        raise RuntimeError("At least one engine must be installed: pip install 'ube[vectorbt]'")

    for engine in engines:
        print(f"\n=== {engine} ===")
        run_single_asset(engine)
        run_multi_asset(engine, ube.EqualWeight())
        run_multi_asset(
            engine,
            ube.FixedWeight(weights={"BTC-USDT": 0.7, "ETH-USDT": 0.3}),
        )


if __name__ == "__main__":
    main()

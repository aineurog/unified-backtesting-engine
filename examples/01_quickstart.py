from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import ube


def make_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=12, freq="h", tz="UTC")
    base = np.linspace(100.0, 112.0, 12)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.3,
            "volume": 1000,
        }
    )


def resolve_engine() -> str:
    ube.ensure_builtin_engines_registered()
    for name in ("vectorbt", "backtrader", "nautilus"):
        if name in ube.registered_engines():
            return name
    raise RuntimeError("No supported backtesting engine is installed.")


def main() -> None:
    bars = make_bars()
    signals = ube.from_target(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int))
    market_data = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")

    config = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="DEMO", asset_class="stocks", settlement_currency="USD"),
        engine=resolve_engine(),
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 10_000.0},
    )

    result = ube.run(market_data, signals, config)
    print(f"Engine: {config.engine}")
    print(f"Final equity: {result.equity_curve.equity[-1]:.2f}")
    print(result.trade_table[["entry_datetime", "exit_datetime", "side", "quantity", "realized_pnl"]].head())


if __name__ == "__main__":
    main()

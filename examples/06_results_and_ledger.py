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
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=20, freq="h", tz="UTC")
    path = np.linspace(100.0, 110.0, 20)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": path,
            "high": path + 1.0,
            "low": path - 1.0,
            "close": path + 0.5,
            "volume": 800,
        }
    )


def main() -> None:
    bars = make_bars()
    signals = ube.from_target(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))

    config = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="DEMO", asset_class="stocks", settlement_currency="USD"),
        engine="backtrader" if "backtrader" in ube.registered_engines else "vectorbt",
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 10_000.0},
    )

    result = ube.run(bars, signals, config)
    print("First rows of trade table:")
    print(result.trade_table[["entry_datetime", "exit_datetime", "side", "quantity", "realized_pnl"]].head())
    print("Final equity curve value:", result.equity_curve.equity[-1])
    print("Ledger events:", len(result.ledger.events))


if __name__ == "__main__":
    ube.ensure_builtin_engines_registered()
    main()

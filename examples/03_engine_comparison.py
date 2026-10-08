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
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=30, freq="h", tz="UTC")
    path = np.linspace(100.0, 120.0, 30)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": path,
            "high": path + 1.0,
            "low": path - 1.0,
            "close": path + 0.5,
            "volume": 500,
        }
    )


def main() -> None:
    ube.ensure_builtin_engines_registered()
    engines = [name for name in ("backtrader", "vectorbt", "nautilus") if name in ube.registered_engines]
    if not engines:
        raise RuntimeError("At least one engine must be installed to compare results.")

    bars = make_bars()
    signals = ube.from_target(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]))

    print("Comparing engines:", engines)
    for engine in engines:
        config = ube.BacktestConfig(
            instrument=ube.Instrument(symbol="BTCUSDT", asset_class="crypto_perp", settlement_currency="USDT"),
            engine=engine,
            risk=ube.RiskConfig(sizing=ube.SizeModel(kind="fixed_units", value=1.0)),
            engine_overrides={"starting_balance": 10_000.0},
        )
        result = ube.run(bars, signals, config)
        print(f"{engine:>10}: final_equity={result.equity_curve.equity[-1]:.2f}, trades={len(result.trade_table)}")


if __name__ == "__main__":
    main()

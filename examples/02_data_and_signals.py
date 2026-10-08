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
    closes = np.linspace(100.0, 115.0, 20)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes - 0.5,
            "high": closes + 1.5,
            "low": closes - 1.5,
            "close": closes,
            "volume": 2000,
        }
    )


def trend_signal(md: ube.MarketData) -> int:
    close = md.close
    if len(close) < 3:
        return 0
    if close[-1] > close[-3]:
        return 1
    if close[-1] < close[-3]:
        return -1
    return 0


def main() -> None:
    bars = make_bars()
    market_data = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")

    target_signals = ube.from_target(np.array([0, 1, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    callable_signals = ube.from_callable(trend_signal, market_data)

    print("target signal columns:", target_signals.columns)
    print("callable signal bars:", callable_signals.n_bars)
    print(callable_signals.to_dataframe().head())


if __name__ == "__main__":
    main()

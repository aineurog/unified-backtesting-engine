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
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=48, freq="h", tz="UTC")
    close = np.linspace(100.0, 148.0, 48)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.8,
            "high": close + 1.4,
            "low": close - 1.4,
            "close": close,
            "volume": 1_000,
        }
    )


def compute_atr_14(md: ube.MarketData) -> np.ndarray:
    if md.n_bars == 0:
        return np.empty(0, dtype=float)
    prev_close = np.empty_like(md.close)
    prev_close[0] = md.close[0]
    prev_close[1:] = md.close[:-1]
    true_range = np.maximum(md.high - md.low, np.maximum(np.abs(md.high - prev_close), np.abs(md.low - prev_close)))
    true_range[0] = md.high[0] - md.low[0]
    kernel = np.ones(14, dtype=float) / 14.0
    return np.convolve(true_range, kernel, mode="same")


def resolve_engine() -> str:
    ube.ensure_builtin_engines_registered()
    for name in ("backtrader", "vectorbt", "nautilus"):
        if name in ube.registered_engines:
            return name
    raise RuntimeError("No supported backtesting engine is installed.")


def run_precomputed_atr() -> None:
    bars = make_bars()
    md = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
    atr_series = compute_atr_14(md)
    target = np.array([0] * 10 + [1] * 12 + [0] * 26, dtype=int)
    signals = ube.from_target(target)

    config = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="XAUUSD", asset_class="commodities", settlement_currency="USD"),
        engine=resolve_engine(),
        risk=ube.RiskConfig(
            sizing=ube.SizeModel(kind="fixed_fraction", value=0.10),
            exit=(ube.ATRStop(mult=2.0, atr="atr_14h"), ube.TrailingStop(percent=0.04)),
        ),
        engine_overrides={"starting_balance": 10_000.0},
    )

    result = ube.run(md, signals, config, aux_data={"atr_14h": atr_series})
    print("Precomputed ATR example:")
    print(f"Final equity: {result.equity_curve.equity[-1]:.2f}")
    print(result.trade_table[["entry_datetime", "exit_datetime", "side", "quantity", "realized_pnl"]].head())


def run_engine_computed_atr() -> None:
    bars = make_bars()
    md = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
    target = np.array([0] * 10 + [1] * 12 + [0] * 26, dtype=int)
    signals = ube.from_target(target)

    config = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="XAUUSD", asset_class="commodities", settlement_currency="USD"),
        engine=resolve_engine(),
        risk=ube.RiskConfig(
            sizing=ube.SizeModel(kind="fixed_fraction", value=0.10),
            exit=(ube.ATRStop(mult=2.0, period=14),),
        ),
        engine_overrides={"starting_balance": 10_000.0},
    )

    result = ube.run(md, signals, config)
    print("Engine-computed ATR example:")
    print(f"Final equity: {result.equity_curve.equity[-1]:.2f}")
    print(result.trade_table[["entry_datetime", "exit_datetime", "side", "quantity", "realized_pnl"]].head())


def main() -> None:
    run_precomputed_atr()
    run_engine_computed_atr()


if __name__ == "__main__":
    main()

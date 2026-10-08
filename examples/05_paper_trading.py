from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import ube


def make_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=24, freq="h", tz="UTC")
    path = np.linspace(100.0, 118.0, 24)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": path,
            "high": path + 1.2,
            "low": path - 1.2,
            "close": path + 0.4,
            "volume": 500,
        }
    )


def compute_atr_14(md: ube.MarketData) -> np.ndarray:
    prev_close = np.empty_like(md.close)
    prev_close[0] = md.close[0]
    prev_close[1:] = md.close[:-1]
    true_range = np.maximum(md.high - md.low, np.maximum(np.abs(md.high - prev_close), np.abs(md.low - prev_close)))
    true_range[0] = md.high[0] - md.low[0]
    kernel = np.ones(14, dtype=float) / 14.0
    return np.convolve(true_range, kernel, mode="same")


def build_paper_config(*, use_precomputed_atr: bool = False) -> ube.paper.PaperConfig:
    base = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="BTCUSDT", asset_class="crypto_perp", settlement_currency="USDT"),
        signal=ube.SignalConfig(on_opposite_signal="reverse"),
        risk=ube.RiskConfig(
            sizing=ube.SizeModel(kind="fixed_units", value=1.0),
            exit=(ube.ATRStop(mult=2.0, period=14, atr="atr_14m") if use_precomputed_atr else ube.ATRStop(mult=2.0, period=14),),
        ),
    )
    return ube.paper.PaperConfig(base=base, engine="recording")


def step_example() -> None:
    bars = make_bars()
    md = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
    signals = ube.from_target(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int))
    config = build_paper_config(use_precomputed_atr=True)
    state = ube.paper.init(config, run_id="atr-paper-step")
    atr = compute_atr_14(md)
    state, events = ube.paper.step(md, signals, state, config, aux_data={"atr_14m": atr})
    print("step() example:")
    print(f"Open position: {state.open_position}")
    print(f"Events: {len(events)}")


def run_example() -> None:
    bars = make_bars()
    md = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
    signals = ube.from_target(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int))
    config = build_paper_config(use_precomputed_atr=False)
    # A fresh database per invocation: ``run`` resumes from a persisted state, so a fixed
    # path would make a second example run replay already-processed bars (DuplicateBarError).
    db_path = str(Path(tempfile.mkdtemp(prefix="ube_example_paper_")) / "run.sqlite")
    state, events = ube.paper.run("demo-run", md, signals, config, db_path=db_path)
    print("run() example:")
    print(f"Open position: {state.open_position}")
    print(f"Events: {len(events)}")


def run_auto_example() -> None:
    bars = make_bars()
    md = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
    config = build_paper_config(use_precomputed_atr=False)

    def signal_fn(window: ube.MarketData) -> int:
        if window.n_bars < 5:
            return 0
        return 1 if window.close[-1] > window.close[-5] else 0

    events = ube.paper.run_auto(md, signal_fn, config)
    print("run_auto() example:")
    print(f"Events: {len(events)}")


def main() -> None:
    step_example()
    run_example()
    run_auto_example()


if __name__ == "__main__":
    main()

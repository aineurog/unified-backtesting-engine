"""Portfolio paper trading: single-asset vs multi-asset on every engine.

Single-asset paper trading is unchanged (``ube.paper.run`` with one bar series). For a
multi-asset session, set ``PaperConfig.portfolio`` and pass ``{instrument_id: ...}``
mappings: each leg keeps its own engine state under ``<run_id>::<symbol>`` while the
parent session persists one combined trade table, one combined equity point, and one
combined statistics snapshot.

Run from the repository root:

    python examples/08_portfolio_paper_trading.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ube  # noqa: E402

STARTING_BALANCE = 100_000.0


def make_bars(drift: float, periods: int = 24) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="h", tz="UTC")
    path = np.linspace(100.0, 100.0 + drift, periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": path,
            "high": path + 1.2,
            "low": path - 1.2,
            "close": path + 0.3,
            "volume": 500,
        }
    )


def make_signals(periods: int = 24) -> ube.Signals:
    target = np.zeros(periods, dtype=np.int64)
    target[3:7] = 1
    target[12:16] = 1
    target[19:22] = 1
    return ube.from_target(target)


def instruments() -> tuple[ube.Instrument, ube.Instrument]:
    return (
        ube.Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
        ube.Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
    )


def run_single_asset(engine: str) -> None:
    btc, _ = instruments()
    base = ube.BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        signal=ube.SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": STARTING_BALANCE},
    )
    config = ube.paper.PaperConfig(
        base=base,
        engine=engine,
        starting_balance=STARTING_BALANCE,
    )
    data = ube.MarketData.from_dataframe(make_bars(12.0), timestamp_col="timestamp")
    db_path = str(Path(tempfile.mkdtemp(prefix=f"ube_paper_{engine}_")) / "single.sqlite")

    state, events = ube.paper.run(
        f"single-{engine}", data, make_signals(), config, db_path=db_path
    )
    print(f"  single asset  [{engine}]")
    print(f"    state / events   : {type(state).__name__} / {len(events)}")
    print(f"    open position    : {state.open_position}")


def run_multi_asset(engine: str, allocation: ube.AllocationPolicy) -> None:
    btc, eth = instruments()
    portfolio = ube.PortfolioConfig(
        instruments=(btc, eth),
        base_currency="USDT",
        allocation=allocation,
    )
    base = ube.BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=portfolio,
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        signal=ube.SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": STARTING_BALANCE},
    )
    config = ube.paper.PaperConfig(
        base=base,
        engine=engine,
        starting_balance=STARTING_BALANCE,
    )
    data = {
        "BTC-USDT": ube.MarketData.from_dataframe(make_bars(12.0), timestamp_col="timestamp"),
        "ETH-USDT": ube.MarketData.from_dataframe(make_bars(22.0), timestamp_col="timestamp"),
    }
    signals = {symbol: make_signals() for symbol in data}
    db_path = str(Path(tempfile.mkdtemp(prefix=f"ube_paper_{engine}_")) / "multi.sqlite")

    state, events = ube.paper.run(
        f"multi-{engine}", data, signals, config, db_path=db_path
    )
    snapshot = state.stats(data, config)
    print(f"  multi asset   [{engine}] {type(allocation).__name__}")
    print(f"    state / events   : {type(state).__name__} / {len(events)}")
    print(f"    open positions   : {state.open_positions}")
    print(f"    combined stats   : {snapshot.n_metrics} metrics, equity_source={snapshot.meta['equity_source']}")


def main() -> None:
    engines = []
    for name in ("vectorbt", "backtrader", "nautilus"):
        try:
            ube.paper.get_paper_engine(name)
        except Exception:  # optional dependency not installed
            continue
        engines.append(name)
    if not engines:
        raise RuntimeError("Install at least one paper engine: pip install 'ube[paper-trading]'")

    for engine in engines:
        print(f"\n=== {engine} ===")
        run_single_asset(engine)
        run_multi_asset(engine, ube.EqualWeight())
        run_multi_asset(
            engine,
            ube.FixedWeight(weights={"BTC-USDT": 0.6, "ETH-USDT": 0.4}),
        )


if __name__ == "__main__":
    main()

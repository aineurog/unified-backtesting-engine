# Tutorials

Step-by-step guides for backtesting and paper trading with UBE.

---

## Tutorial 1: Your First Backtest

### 1. Install

```powershell
python -m pip install -e ".[backtrader]"
```

### 2. Prepare Data

UBE accepts DataFrames, dicts, or arrays. The simplest is a DataFrame with
canonical OHLCV columns:

```python
import pandas as pd

bars = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
    "open":   [100 + i * 0.1 for i in range(100)],
    "high":   [101 + i * 0.1 for i in range(100)],
    "low":    [99 + i * 0.1 for i in range(100)],
    "close":  [100.5 + i * 0.1 for i in range(100)],
    "volume": [1000] * 100,
})
```

### 3. Define Signals

Use `from_target` for a position-based strategy:

```python
import numpy as np
import ube

# Go long for bars 10-30, short for 50-70, flat otherwise
target = np.zeros(100, dtype=int)
target[10:30] = 1
target[50:70] = -1
signals = ube.from_target(target)
```

Or use `from_callable` for a dynamic strategy:

```python
def momentum(md: ube.MarketData) -> int:
    if md.close[-1] > md.close[-5]:
        return 1
    if md.close[-1] < md.close[-5]:
        return -1
    return 0

market_data = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
signals = ube.from_callable(momentum, market_data)
```

### 4. Configure the Run

```python
config = ube.BacktestConfig(
    instrument=ube.Instrument(
        symbol="DEMO",
        asset_class="stocks",
        settlement_currency="USD",
    ),
    engine="backtrader",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=1.0),
        exit=(
            ube.StopLoss(percent=0.02),
            ube.TakeProfit(percent=0.04),
        ),
    ),
    engine_overrides={"starting_balance": 10_000.0},
)
```

### 5. Run and Inspect

```python
result = ube.run(bars, signals, config)

print(result.trade_table[["entry_datetime", "exit_datetime", "side",
                           "quantity", "realized_pnl", "balance"]])
print(f"Final equity: {result.equity_curve.equity[-1]:.2f}")
```

---

## Tutorial 2: Comparing Engines

Run the same strategy on all three engines and compare:

```python
import ube

engines = ["vectorbt", "backtrader", "nautilus"]
results = {}

for eng in engines:
    cfg = ube.BacktestConfig(
        instrument=ube.Instrument(symbol="BTCUSDT", asset_class="crypto_perp",
                                 settlement_currency="USDT"),
        engine=eng,
        risk=ube.RiskConfig(
            sizing=ube.SizeModel(kind="fixed_fraction", value=0.10),
        ),
        engine_overrides={"starting_balance": 10_000.0},
    )
    results[eng] = ube.run(bars, signals, cfg)

for eng, res in results.items():
    fills = [e for e in res.ledger.events
             if e.event_type.name == "FILL"]
    print(f"{eng:12s}: {len(fills)} fills, "
          f"equity={res.equity_curve.equity[-1]:.2f}")
```

Compare fill events first when results differ — see
[troubleshooting](troubleshooting.md).

---

## Tutorial 3: ATR-Based Exits with Auxiliary Data

ATR exits need a named `aux_data` series:

```python
import numpy as np
import ube

# Compute ATR(14) from bars
def atr(md: ube.MarketData, period: int = 14) -> np.ndarray:
    tr = np.maximum(
        md.high - md.low,
        np.maximum(
            np.abs(md.high - np.roll(md.close, 1)),
            np.abs(md.low - np.roll(md.close, 1)),
        ),
    )
    tr[0] = md.high[0] - md.low[0]
    return np.convolve(tr, np.ones(period) / period, mode="same")

market_data = ube.MarketData.from_dataframe(bars, timestamp_col="timestamp")
atr_series = atr(market_data)

config = ube.BacktestConfig(
    instrument=ube.Instrument(symbol="XAUUSD", asset_class="commodities",
                             settlement_currency="USD"),
    engine="vectorbt",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=10.0),
        exit=(
            ube.ATRStop(mult=2.0, period=14, atr="atr_1h"),
            ube.TimeExit(bars=20),
        ),
    ),
    engine_overrides={"starting_balance": 10_000.0},
)

result = ube.run(
    market_data,
    signals,
    config,
    aux_data={"atr_1h": atr_series},
)
```

---

## Tutorial 4: Paper Trading (Single Window)

Process a batch of bars through the paper engine:

```python
import ube

paper_cfg = ube.PaperConfig(
    base=ube.BacktestConfig(
        instrument=ube.Instrument(symbol="GBPUSD", asset_class="forex",
                                 settlement_currency="USD"),
        risk=ube.RiskConfig(
            sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=10.0),
            exit=(ube.ATRStop(mult=1.5, period=14, atr="atr_1m"),),
        ),
        signal=ube.SignalConfig(on_opposite_signal="reverse"),
        engine_overrides={"starting_balance": 10_000.0},
    ),
    engine="vectorbt",
    starting_balance=10_000.0,
)

state = ube.paper.init(paper_cfg, run_id="gbpusd-demo", db_path="paper.db")
state, events = ube.paper.step(market_data, signals, paper_cfg,
                               aux_data={"atr_1m": atr_series})
print(f"Open position: {state.open_position}")
print(f"New events: {len(events)}")
```

---

## Tutorial 5: Paper Trading (Incremental / Live-style)

For live-style processing where bars arrive incrementally:

```python
import ube

state = ube.paper.init(paper_cfg, run_id="gbpusd-live", db_path="paper.db")

# Each poll: fetch new bars, step, save
for new_bars in bar_feed:
    md = ube.MarketData.from_dataframe(new_bars, timestamp_col="timestamp")
    sig = ube.from_callable(my_strategy, md)
    state, events = ube.paper.step(md, sig, paper_cfg,
                                   aux_data={"atr_1m": atr_series})
    state.save()
    print(f"cursor={state.last_processed_ns} events={len(events)} "
          f"open={state.open_position}")
```

Key rules:
- Always pass bars **after** the cursor for incremental engines.
- Replay backends (VectorBT, Backtrader) need context bars including the
  carried entry bar when a position is open.
- Use the same `run_id` and `db_path` for the same logical session.

---

## Tutorial 6: Paper Trading with `run_auto`

`run_auto` evaluates a signal callable over growing prefixes automatically:

```python
import ube

def signal_fn(md: ube.MarketData) -> int:
    if len(md.close) < 2:
        return 0
    return 1 if md.close[-1] > md.close[-2] else -1

events = ube.paper.run_auto(
    market_data,
    signal_fn,
    paper_cfg,
    run_id="auto-demo",
)
print(f"Processed {len(events)} events")
```

---

## Tutorial 7: Inspecting the Ledger

The ledger is the source of truth:

```python
import ube

result = ube.run(bars, signals, config)

for e in result.ledger.events:
    kind = e.event_type.name
    if kind == "FILL":
        print(f"FILL  ts={e.timestamp} side={e.side} qty={e.quantity} "
              f"px={e.price} reason={e.exit_reason}")
    elif kind == "COMMISSION":
        print(f"COMMISSION amount={e.amount}")
    elif kind == "FUNDING_PAYMENT":
        print(f"FUNDING amount={e.amount}")
```

Rebuild a result from a ledger:

```python
from ube.core.result import BacktestResult

result2 = BacktestResult.from_ledger("rebuild", result.ledger, config)
```

---

## Tutorial 8: Custom Sizing

```python
import ube

config = ube.BacktestConfig(
    instrument=ube.Instrument(symbol="ETHUSDT", asset_class="crypto_perp",
                             settlement_currency="USDT"),
    engine="backtrader",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(
            kind="volatility_target",
            value=0.15,        # 15% annualized vol target
            leverage=1.0,
            vol="realized_vol",  # name in aux_data
        ),
    ),
    engine_overrides={"starting_balance": 50_000.0},
)

result = ube.run(market_data, signals, config,
                 aux_data={"realized_vol": vol_series})
```

---

## Tutorial 9: Multi-Instrument Portfolio (Nautilus)

```python
import ube

config = ube.BacktestConfig(
    instrument=ube.Instrument(symbol="PORTFOLIO", asset_class="stocks",
                             settlement_currency="USD"),
    engine="nautilus",
    base_currency="USD",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(kind="equal_weight", value=3),
    ),
    benchmark=ube.BenchmarkConfig(kind="buy_and_hold"),
)

# Portfolio input: mapping of instrument_id -> MarketData
portfolio_data = {"AAPL": md_aapl, "MSFT": md_msft, "GOOG": md_goog}
result = ube.run(portfolio_data, portfolio_signals, config)
```

---

## Tutorial 10: Saving and Loading Results

```python
import ube

result = ube.run(bars, signals, config)
result.save("my_result.pkl")

# Later
from ube.core.result import BacktestResult
loaded = BacktestResult.load("my_result.pkl")
print(loaded.trade_table)
```

Only load pickles from trusted sources.
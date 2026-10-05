# Getting Started

UBE is pre-release software. Python 3.11 or newer is required. Install the core
package and only the engine you plan to run:

```powershell
python -m pip install -e ".[backtrader]"
```

Choose `[vectorbt]`, `[nautilus]`, or `[all]` instead as needed. Add `[dev]` for
the test and lint tools. The core package does not force-install any backtesting
engine.

## First Backtest

`MarketData` contains aligned OHLCV bars. `from_target` converts a target
position series (`-1`, `0`, `1`) into the four canonical entry/exit signals.

```python
import numpy as np
import pandas as pd

import ube

bars = pd.DataFrame(
    {
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [1, 1, 1, 1, 1],
    }
)

instrument = ube.Instrument(
    symbol="DEMO",
    asset_class="stocks",
    settlement_currency="USD",
)
config = ube.BacktestConfig(
    instrument=instrument,
    engine="backtrader",
    risk=ube.RiskConfig(sizing=ube.SizeModel(kind="fixed_units", value=1)),
)
signals = ube.from_target(np.array([0, 1, 1, 0, 0]))
result = ube.run(bars, signals, config)

print(result.trade_table)
print(result.equity_curve.equity[-1])
```

Install the matching optional extra before choosing an engine. Use an explicit
engine name for reproducible runs; `engine="auto"` chooses the first installed
adapter in VectorBT, Backtrader, NautilusTrader order.

Next: [Core concepts](core-concepts.md), [data and signals](data-and-signals.md),
and [engine behavior](engines/overview.md).
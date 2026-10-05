# Unified Backtesting Engine

UBE standardizes market data, signals, risk configuration, and results across
VectorBT, Backtrader, and NautilusTrader. Each engine remains responsible for its
own execution model; UBE translates between those models and a common event ledger.

The package is pre-release. Treat the API as unstable and check the installed
version's behavior before relying on engine parity in production.

## Choose a path

| You want to... | Start here |
|---|---|
| Run your first backtest | [Getting started](getting-started.md) |
| Understand the shared data and event model | [Core concepts](core-concepts.md) |
| Configure an instrument and a run | [Configuration](configuration.md) |
| Compare engine behavior | [Engine overview](engines/overview.md) |
| Run a paper session | [Paper trading](paper-trading.md) |
| Follow a step-by-step guide | [Tutorials](tutorials.md) |
| Inspect output and ledger semantics | [Results and ledger](results-ledger.md) |
| Debug a mismatch | [Troubleshooting](troubleshooting.md) |
| Extend UBE or run its test suite | [Contributing](contributing.md), [Testing](testing.md) |

## Architecture

```text
MarketData + Signals + BacktestConfig
			|
		    ube.run
	    /         |         \
    VectorBT   Backtrader   NautilusTrader
	    \         |         /
	     Canonical event ledger
			  |
	Trades, positions, equity, trade table
```

The canonical core is engine-independent. Adapters translate the common contract
to engine-specific execution and fold resulting events into the same ledger
schema. The shared output does not imply identical native fill behavior in every
situation; consult the engine pages and [parity notes](engines/overview.md).

## Documentation map

- [Installation and first backtest](getting-started.md)
- [Core concepts](core-concepts.md)
- [Market data and signals](data-and-signals.md)
- [Configuration](configuration.md)
- [Sizing, exits, and costs](risk-and-costs.md)
- [Engine overview](engines/overview.md): [VectorBT](engines/vectorbt.md), [Backtrader](engines/backtrader.md), [NautilusTrader](engines/nautilus.md)
- [Paper trading](paper-trading.md)
- [Results and ledger](results-ledger.md)
- [API reference](api-reference.md)
- [Tutorials](tutorials.md)
- [Testing and fixtures](testing.md)

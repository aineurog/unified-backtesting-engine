# Engine Overview

Install the optional engine dependency matching the adapter: `[vectorbt]`,
`[backtrader]`, or `[nautilus]`. The adapters are registered lazily when their
dependencies are importable. `engine="auto"` selects VectorBT, then Backtrader,
then NautilusTrader; specify the engine to make a run reproducible.

| Capability | VectorBT | Backtrader | NautilusTrader |
|---|---|---|---|
| Backtest adapter | Implemented | Implemented | Implemented |
| Single-instrument OHLCV + canonical signals | Supported | Supported | Supported |
| Canonical result and event ledger | Yes | Yes | Yes |
| Built-in paper backend | Implemented; window replay | Implemented; window replay | Implemented; incremental bars |
| Native execution model | Vectorized signal portfolio | Event-driven strategy | Event-driven engine and actor |
| Cross-engine parity | Exercise with `tests/parity/` and inspect tested cases | Same | Reference for the current parity tests |

The table describes the current code, not a guarantee of identical fills for
every input. Built-in backtest adapters document a single-instrument contract;
mapping-shaped portfolio input to `ube.run()` is forwarded to adapters and does
not mean every built-in engine supports portfolio execution.

## Shared Contract

All adapters consume canonical `MarketData`, `Signals`, and `BacktestConfig`,
then produce `BacktestResult`. The shared `EventLedger` schema makes downstream
trade and equity projections comparable. Differences in native order execution,
fill timing, price precision, lot sizes, and intrabar exit collisions can still
produce meaningful result differences.

## Choosing an Adapter

- Choose VectorBT when the strategy maps naturally to bar-aligned vectorized
  signals and you want to use its portfolio engine.
- Choose Backtrader when an event-driven strategy loop and its broker model fit
  the workflow.
- Choose NautilusTrader when its event-driven backtest and native instrument,
  account, and order abstractions are needed.

These are selection heuristics, not performance or live-trading guarantees. UBE's
paper APIs process supplied bars; the package does not fetch market data or
connect these adapters to a live brokerage account.

Details: [VectorBT](vectorbt.md), [Backtrader](backtrader.md),
[NautilusTrader](nautilus.md). Run the [parity tests](../testing.md) for the
specific asset classes and cases under consideration.
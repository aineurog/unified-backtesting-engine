# Core Concepts

UBE standardizes the inputs and the output around existing backtesting engines.
It does not provide a strategy class or make the engines' execution models
identical.

## Run Contract

The common backtest call is:

```text
MarketData + Signals + BacktestConfig -> EngineAdapter -> BacktestResult
```

- `MarketData` is validated, timestamped OHLCV for one instrument.
- `Signals` carries four boolean action arrays: long/short entry and exit.
- `BacktestConfig` combines instrument metadata, risk/sizing, costs, engine
  selection, and run options.
- An `EngineAdapter` maps that contract into one engine and returns a canonical
  result.
- `BacktestResult.ledger` is the event source of truth; trade and equity views
  are derived from it.

## Actions, Not Position State

Each signal column describes a new action on a bar. A `False` value means “no
action”, not “flat”. An exit and opposite-side entry may both be true on one bar
to represent a flip. `from_target()` instead accepts position states, where `0`
does mean flat and emits the appropriate actions when the target changes.

## Lifecycle

For backtests, UBE standardizes inputs, resolves an adapter, runs it, folds fills
and account events into the ledger, and derives result views. Paper trading
reuses the same canonical inputs and ledger concepts but adds a persistent
`PaperState` and a cursor that prevents processing the same bar twice.

## What Is Shared

Data validation, signal encoding, position sizing, exit configuration, costs,
ledger schema, and result projections are shared concepts. Actual order
submission, broker behavior, bar timing, and some exit/fill details remain
engine-specific. Read the [engine overview](engines/overview.md) before relying
on cross-engine equality for a particular strategy.
# Paper Trading

Paper trading is a local simulation over caller-supplied bars. It shares the
canonical `MarketData`, `Signals`, risk configuration, and ledger, and adds
`PaperConfig` and `PaperState`. It does not fetch market data, run a polling
service, or route orders to a live broker.

## Configuration and State

`PaperConfig` wraps a `BacktestConfig` and adds `engine`, optional `state_path`,
starting-balance convenience, and calendar policy. Declare
`BacktestConfig.signal.on_opposite_signal` as `reverse`, `exit_only`, or `ignore`;
paper runs reject an undeclared policy.

`init(config, run_id=..., db_path=...)` creates an in-memory state or a state
associated with SQLite persistence. `PaperState` contains the append-only event
ledger, bar cursor, last mark, and open position. Persisted configuration is a
reference for inspection; loading state does not replace the config passed to a
later `step` call.

## Processing Bars

```python
state = ube.paper.init(config, run_id="demo", db_path="paper.db")
state, new_events = ube.paper.step(data, signals, state, config)
```

Bars and signals must be aligned and strictly increasing. For incremental
engines, every bar must be newer than the state cursor. VectorBT and Backtrader
paper backends replay context windows and allow older context bars only when the
window advances beyond the cursor. Bar gaps are allowed. Duplicate or stale
input raises `DuplicateBarError`.

`run_auto(data, signal_fn, config, state=...)` evaluates a target callable over
each growing prefix and processes only the suffix after the cursor. `run()` is a
strategy-keyed load-or-initialize wrapper around `step()` and requires a
`db_path` argument or `config.state_path`.

Persistence helpers store state, reconstructed trades, and equity snapshots.
The package currently records equity points per step rather than rebuilding a
complete curve across missing bar history. A persistence failure should be
verified from the database; persistence exceptions in the `step` path are not
always surfaced to callers.

## CLI Status

The CLI is invoked as `python -m ube.papertrading.cli`. `step` currently
initializes or resumes state and prints a summary; it does not consume market
bars. `daemon` is a stub. For actual supplied-bar processing, use the Python
`step`, `run_auto`, or `run` APIs above.
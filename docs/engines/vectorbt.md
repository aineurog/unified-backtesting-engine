# VectorBT Adapter

Use `engine="vectorbt"` with the optional `vectorbt` package installed. The
adapter translates canonical bars and signals into a VectorBT portfolio, then
folds the resulting execution records into UBE's event ledger.

## Mapping and Behavior

- The adapter supports the canonical single-instrument backtest input.
- Common UBE sizing, exits, costs, instrument metadata, and optional auxiliary
  series are translated by the adapter; lot increments and account constraints
  can change the final size.
- UBE derives a running size series so later entries use equity after realized
  costs. Native VectorBT output is not itself the canonical ledger.
- Paper trading uses replay windows so the backend can reconstruct position and
  indicator context from supplied history. Callers must provide enough historical
  bars for the open position and configured warmup needs.

## Limits

Vectorized execution is not tick-by-tick execution. OHLC bars do not reveal the
true intrabar path when multiple exits are touched in one bar. UBE normalizes
records to its ledger semantics, but this does not make the intrabar path known
or equivalent to a live venue.

For available engine overrides, see validation in
`src/ube/adapters/vectorbt_adapter/overrides.py`. The adapter is covered by
`tests/integration/test_vectorbt_adapter.py`,
`tests/integration/test_papertrading_vbt.py`, and
`tests/parity/test_vectorbt_parity.py`.
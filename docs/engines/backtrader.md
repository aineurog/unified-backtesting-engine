# Backtrader Adapter

Use `engine="backtrader"` with the optional `backtrader` package installed. UBE
builds a feed containing the four canonical signal lines, runs its event-driven
strategy, then folds recorded signals and fills into the canonical event ledger.

## Mapping and Behavior

- The adapter accepts single-instrument OHLCV data and aligned `Signals`.
- A signal or exit plan drives entry/exit orders. The UBE ledger records
  bar-synchronous fills at the signal bar close, or at the touched exit level for
  a level-triggered risk exit; slippage and commission are applied in the fold.
- Position sizing uses the shared sizing model, effective leverage, asset-class
  lot increment, and an affordability check. A running net cash balance is used
  for subsequent sizing rather than treating the broker's leveraged equity as
  the account balance.
- Paper trading uses replay windows and carries open positions at their original
  quantity. The replay must include the carried entry bar.

## Diagnosing Differences

Compare canonical fill events first: timestamp, side, quantity, raw/slipped
price, commission, and exit reason. Then compare derived `Trade.net_pnl` and the
equity curve. A trade-table `balance` difference commonly follows an earlier
fill-size or cost difference; it is not by itself proof that the ledger fold's
balance accumulation is the source of the mismatch.

Backtrader's internal broker order execution is used to drive its strategy loop;
the canonical UBE ledger applies its documented bar-close/touched-level policy.
Do not interpret this adapter as an exchange simulator or live execution route.

Adapter options are validated in
`src/ube/adapters/backtrader_adapter/overrides.py`. Coverage is in
`tests/integration/test_backtrader_adapter.py`,
`tests/integration/test_papertrading_backtrader.py`, and
`tests/parity/test_backtrader_parity.py`.
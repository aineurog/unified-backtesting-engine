# NautilusTrader Adapter

Use `engine="nautilus"` with the optional `nautilus-trader` package installed.
The adapter builds Nautilus instruments and bars, runs the UBE actor through a
Nautilus backtest engine, and folds account/order events into the canonical UBE
ledger.

## Mapping and Behavior

- The adapter accepts single-instrument OHLCV data with aligned canonical
  signals.
- The actor uses UBE sizing and exits while submitting Nautilus-native orders.
  Fees, slippage, funding, and position changes are represented in the UBE
  ledger.
- Price precision and order quantity are constrained by the constructed
  instrument and its increments. Explicit adapter overrides may affect this
  mapping.
- Paper trading runs supplied bars through a sandbox execution client and
  persists resumable state when configured. It does not connect to a real
  brokerage account.

## Limits

Native engine event timing and intrabar trigger ordering are not inferred from
information OHLC bars do not contain. Several orders may be submitted around a
single bar and venue fill details can differ from another adapter. Check the
specific parity and integration tests for established behavior.

Options are validated in `src/ube/adapters/nautilus_adapter/overrides.py`.
Coverage is in `tests/integration/test_nautilus_adapter.py`,
`tests/integration/test_papertrading_nautilus.py`, and
`tests/parity/test_nautilus_parity.py`.
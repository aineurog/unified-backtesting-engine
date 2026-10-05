# Results and Ledger

## Event Ledger

`BacktestResult.ledger` is the canonical append-only record. Events include
signal evaluations, submitted orders, fills, commissions, funding payments,
cash movements, position changes, and futures rollovers. Each event uses a UTC
nanosecond timestamp and a canonical instrument id.

The ledger is the source for the result projections. Prefer comparing fill and
cost events before comparing a derived table or CSV export.

## Derived Views

- `result.trades` contains round-trip trades reconstructed from fill and position
  events.
- `result.trade_table` is a reporting DataFrame with one row per trade and
  derived return, balance, size, fee, and exit-reason fields.
- `result.positions` is the position series for a single-instrument run; it is
  `None` when a combined multi-instrument position is undefined.
- `result.equity_curve` and `result.equity_curve_by_instrument` are marked-to-
  market equity views. Currency conversion requires available FX rates when
  settlement and base currencies differ.
- `result.benchmark` is attached by `ube.run()` for supported single-instrument
  runs. `result.metrics` is currently `None`; metrics are not computed by UBE.

`trade_table.balance` is a convenience projection derived from ledger trades and
equity context. It is not an independent account-state store. CSV exporters in
applications may further transform that projection; compare their output with
the underlying events and `BacktestResult` before attributing a mismatch to the
adapter's event ledger.

`BacktestResult.save()` and `.load()` persist the full result using pickle. Only
load pickles from trusted sources.
# Troubleshooting

## Engine Is Not Registered

Install the optional dependency for the requested engine in the same Python
environment used to run UBE. Engine adapters are imported lazily. Check
`ube.registered_engines()` and select an explicit `BacktestConfig.engine`.

## Data or Signal Shape Errors

Check that timestamps are timezone-aware, strictly increasing, and that OHLCV
columns contain aligned rows. Every signal array must be boolean and exactly the
same length as the bars. Use `ube.from_target()` for `-1/0/1` targets rather
than passing targets where `Signals` are expected.

## Different Engine Results

Compare in this order:

1. Confirm the same bars, timestamp alignment, instrument metadata, costs,
   sizing model, leverage, and starting balance were used.
2. Compare `order_submitted` and `fill` events, including quantity, timestamp,
   price, and exit reason.
3. Compare commission, funding, and cash movement events.
4. Compare reconstructed `Trade.net_pnl`, equity, then reporting tables.

If a Backtrader CSV starts with a different first-trade quantity while later
quantities match another engine, the later balance difference is downstream of
the first fill/PnL difference. First verify the exact config, state database,
and installed UBE source used by each runner. The CSV `balance` field alone
cannot establish whether the adapter fold or an earlier sizing/fill differs.

## Paper Resume Repeats or Skips Bars

Use the same run id and state database for the same logical session. Do not pass
bars at or before the cursor to an incremental backend. Replay backends need
context bars (including the carried entry bar when a position remains open) and
must receive a window that advances the cursor.

## Missing FX or Auxiliary Series

Portfolio equity conversion requires rates for the needed currency pairs.
`volatility_target` sizing and ATR-based exits may require a named `aux_data`
entry. Supply a series aligned to the required bars and use the exact configured
name.

## Calendar Rejections

Check the instrument calendar and UTC timestamps. Paper sessions can warn and
skip out-of-session bars by default; strict mode raises a calendar mismatch.

See the [engine guides](engines/overview.md) and [tests](testing.md) for
backend-specific coverage.
# Risk and Costs

## Position Sizing

`RiskConfig` contains a `SizeModel` and an ordered tuple of exits. Available
sizing kinds are:

| Kind | Meaning |
|---|---|
| `fixed_fraction` | Allocate `value` fraction of capital to position notional. This is not risk-per-stop sizing. |
| `fixed_units` | Use exactly `value` units before adapter lot constraints. |
| `volatility_target` | Size to a target portfolio-volatility fraction using a supplied volatility series. |
| `all_in` | Allocate available capital, reserving entry fees when applicable. |
| `equal_weight` | Divide capital across `n` positions; adapters currently run the single-instrument sizing path. |

`SizeModel.leverage` scales exposure for margin-capable account configurations.
Cash account overrides can force effective leverage to one. Adapters quantize
orders to asset-class lot increments; the final quantity may therefore be lower
than the raw sizing formula.

`volatility_target` requires a named `aux_data` series via `SizeModel.vol`. UBE
does not silently estimate volatility from the signal bars. ATR-based exits also
reference named auxiliary data when configured with an `atr` key.

## Exits

The core exposes `StopLoss`, `TakeProfit`, `TrailingStop`, `TimeExit`, `ATRStop`,
and `ChandelierExit`. Configure exits through `RiskConfig(exit=(...))`. The
adapter evaluates them according to its execution model; ordered exit rules and
same-bar triggers can affect which exit is selected. Refer to the relevant
[engine guide](engines/overview.md) for behavior rather than assuming native
engine parity.

## Cost Model

`CostModel` rates are fractions, not percentages: `0.001` is 0.1%. It models
commission, slippage, funding, and borrow cost. Slippage adjusts fill prices;
commission and carry are separate ledger events. When no model is supplied,
`resolve_cost_model()` provides the library default for the instrument's asset
class. Review those defaults in `src/ube/core/cost.py` before using them as a
proxy for a venue's actual fee schedule.

For engine-specific account, lot-size, FX, and funding overrides, see the
[adapter references](engines/overview.md).
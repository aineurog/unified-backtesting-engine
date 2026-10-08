# Risk and Costs

Risk configuration has two independent halves: **how much** to trade
(`SizeModel`) and **when to leave** (`RiskConfig.exit`). Signals still open and
close positions; the configured exits are *additional* protective/target rules
that fire between signal bars.

```python
config = ube.BacktestConfig(
    instrument=ube.Instrument(
        symbol="XAUUSD", asset_class="commodities", settlement_currency="USD",
    ),
    engine="vectorbt",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=10.0),
        exit=(
            ube.StopLoss(percent=0.02),               # hard stop, 2% from entry
            ube.TakeProfit(percent=0.06, scale_out=1.0),
            ube.TrailingStop(percent=0.015),          # trails the running peak
            ube.TimeExit(bars=48),                    # bail out after 48 bars
        ),
    ),
    engine_overrides={"starting_balance": 10_000.0},
)
```

## Position Sizing

`RiskConfig` contains a `SizeModel` and an ordered tuple of exits. Available
sizing kinds are:

| Kind | `value` | Meaning |
|---|---|---|
| `fixed_fraction` | required | Allocate `value` fraction of capital to position notional. This is *allocation*, not risk-per-stop sizing. |
| `fixed_units` | required | Use exactly `value` units before adapter lot constraints. |
| `volatility_target` | required | Size to a target portfolio-volatility fraction using a supplied volatility series. |
| `all_in` | forbidden | Allocate available capital, reserving entry fees when applicable. |
| `equal_weight` | forbidden | Divide capital across `n` positions; adapters currently run the single-instrument sizing path. |

`SizeModel.leverage` scales exposure for margin-capable account configurations.
Cash account overrides can force effective leverage to one. Adapters quantize
orders to asset-class lot increments (flooring, never rounding up), so the final
quantity may be lower than the raw sizing formula.

`volatility_target` requires a named `aux_data` series via `SizeModel.vol`:

```python
ube.SizeModel(kind="volatility_target", value=0.15, vol="realized_vol")
# ... ube.run(..., aux_data={"realized_vol": vol_series})
```

The library never estimates volatility from the signal bars; if the named series
is missing the adapter raises `ConfigError` before the engine runs.

## Exits

`RiskConfig.exit` accepts a single exit or an ordered sequence of exits. The
core exposes six exit configs:

| Class | Signature | Description |
|---|---|---|
| `StopLoss` | `(percent, trigger="touched")` | Static stop a fixed fraction from entry; never ratchets. |
| `TakeProfit` | `(percent, scale_out=1.0, trigger="touched")` | Fixed target a fraction beyond entry; can partially scale out. |
| `TrailingStop` | `(percent, trigger="touched")` | Percentage stop that trails the running peak (long) / trough (short) since entry. |
| `ATRStop` | `(mult, trigger="touched", trailing=False, period=14, atr=None)` | Stop `mult × ATR` from entry; `trailing=True` ratchets it. Requires a named `atr` aux series. |
| `ChandelierExit` | `(mult, trigger="touched", period=14, atr=None)` | Running high/low offset by `mult × ATR`. Requires a named `atr` aux series. |
| `TimeExit` | `(bars)` | Close once the position has been held `bars` bars. |

### The trigger rule

Every price-level exit has a `trigger`:

- `"touched"` (default) — the bar fires if its **high/low** reached the level,
  even if the close did not.
- `"close"` — only the bar's **close** is compared to the level.

`TimeExit` has no `trigger`; time is discrete, so there is no touched/close
ambiguity.

```python
ube.StopLoss(percent=0.02)                     # intrabar touch
ube.StopLoss(percent=0.02, trigger="close")    # close-only confirmation
```

### Stop types, one by one

```python
# Static stop: long stop = entry * (1 - percent), short stop = entry * (1 + percent)
ube.StopLoss(percent=0.02)

# Fixed target: long target = entry * (1 + percent), short target = entry * (1 - percent)
ube.TakeProfit(percent=0.04)

# Trailing percentage stop: long stop = running_high * (1 - percent)
ube.TrailingStop(percent=0.02)

# ATR stop, entry-anchored: stop = entry -/+ mult * ATR
ube.ATRStop(mult=2.0, period=14, atr="atr_1h")

# ATR stop, trailing variant: ratchets with the favourable extreme, never loosens
# below the entry-anchored stop
ube.ATRStop(mult=2.0, period=14, atr="atr_1h", trailing=True)

# Chandelier: running_high - mult * ATR (long), running_low + mult * ATR (short)
ube.ChandelierExit(mult=3.0, period=14, atr="atr_1h")

# Holding-period exit
ube.TimeExit(bars=24)
```

Stops (`StopLoss`, `TrailingStop`, `ATRStop`, `ChandelierExit`, `TimeExit`)
always exit **100%** of the remaining position. Only `TakeProfit` can exit a
fragment; see scale-out below.

### ATR and Chandelier need auxiliary data

`ATRStop` and `ChandelierExit` never compute ATR from the signal bars. They must
name an `aux_data` entry via the `atr` key, and you must supply it. A missing
name raises `ConfigError` before the engine starts:

```text
ConfigError: ATRStop requires an 'atr' key referencing an aux_data series (§5.2);
ATR is never computed from the signal data bars
```

The named series can be either form:

```python
# (a) A precomputed ATR array on the main bar grid: 1-D, length == number of bars.
ube.ATRStop(mult=2.0, atr="atr_1h")
ube.run(md, signals, config, aux_data={"atr_1h": atr_series})

# (b) A signal-timeframe OHLCV MarketData: the adapter computes Wilder's ATR
#     with the exit's `period` on those bars, forward-fills it onto the bar grid,
#     and shifts it one bar so a bar never sees the ATR it is inside (no look-ahead).
ube.ChandelierExit(mult=3.0, period=14, atr="atr_4h")
ube.run(md, signals, config, aux_data={"atr_4h": market_data_4h})
```

When a `MarketData` is supplied, it must span the signal/price period (start at
or before the first bar and reach within one aux bar of the last), or the adapter
raises `DataShapeError`. See
[Tutorial 3](tutorials.md#tutorial-3-atr-based-exits-with-auxiliary-data) and
[Tutorial 11](tutorials.md#tutorial-11-engine-computed-atr-from-aux-bars) for
worked examples.

### Scale-out (partial take-profits)

`TakeProfit.scale_out` is the fraction of the position to exit when that target
fires, in `(0, 1]`. Layer several targets to scale out of a winner:

```python
risk=ube.RiskConfig(
    sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=10.0),
    exit=(
        ube.TakeProfit(percent=0.02, scale_out=0.50),  # bank half at +2%
        ube.TakeProfit(percent=0.04, scale_out=0.50),  # bank the rest at +4%
        ube.StopLoss(percent=0.02),                    # protects whatever remains
    ),
)
```

Constraints:

- The `TakeProfit.scale_out` fractions must sum to at most `1.0`; a larger sum
  raises `ConfigError` at `RiskConfig` construction.
- Each `TakeProfit` fires at most once; the fractions are evaluated on the
  position remaining when it fires.
- Stops always take the whole remaining position regardless of scale-out.

### Ordering and same-bar collisions

Exits are evaluated in the order given. OHLC bars do not reveal the true
intrabar path, so when a single bar touches more than one level, UBE resolves
the winner deterministically by proximity to the bar's **open** (the level price
reaches first). Exact ties fall back to the configured order. This makes "which
exit fired" a function of the bar's geometry, not the config order, but two
engines can still report different prices/timing because their native fill models
differ. Compare `fill` events before comparing derived tables (see
[Troubleshooting](troubleshooting.md)).

## Cost Model

`CostModel` rates are fractions, not percentages: `0.001` is 0.1%. Each field
defaults to `0.0` (a bare `CostModel()` is zero-cost):

| Field | Charged | Notes |
|---|---|---|
| `commission` | per fill, fraction of notional | |
| `slippage` | per fill, fraction of fill price | A BUY pays `price * (1 + slippage)`, a SELL receives `price * (1 - slippage)`. |
| `funding` | per funding period, fraction of notional | Applies to both sides (perps, forex swap). A negative rate models funding *received*. |
| `borrow` | per funding period, fraction of notional | Short side only (stocks, futures). |

The funding/borrow period is the instrument's `funding_interval_hours` schedule
(default `8.0` hours; see the [Instrument reference](api-reference.md)). When no
model is supplied,
`resolve_cost_model()` provides the library default for the instrument's asset
class: `crypto_perp` resolves to `commission=0.0005`, `funding=0.0001`; every
other asset class resolves to zero-cost until the per-asset-class defaults land.
Review those defaults and `resolve_cost_model()` in `src/ube/core/cost.py`.

```python
cost_model=ube.CostModel(commission=0.0004, slippage=0.0002, funding=0.0001)
```

For engine-specific account, lot-size, FX, and funding overrides, see the
[adapter references](engines/overview.md).

# Configuration

`BacktestConfig` is the immutable input contract for a backtest. It combines a
required `Instrument` with optional `CostModel`, `RiskConfig`, `SignalConfig`,
`BenchmarkConfig`, engine selection and run options.

```python
config = ube.BacktestConfig(
    instrument=ube.Instrument(
        symbol="BTCUSDT",
        asset_class="crypto_perp",
        settlement_currency="USDT",
    ),
    engine="nautilus",
    risk=ube.RiskConfig(
        sizing=ube.SizeModel(kind="fixed_fraction", value=0.10, leverage=2.0),
    ),
    engine_overrides={"starting_balance": 10_000.0},
)
```

## Important Fields

- `instrument` describes symbol, asset class, settlement currency, tick size,
  multiplier, calendar, and funding interval.
- `cost_model=None` resolves the library's asset-class default; provide an
  explicit `CostModel` when the run's fees and carry should be controlled.
- `engine` is `"auto"` or a registered engine name. `auto` selects the first
  installed engine in the documented preference order.
- `engine_overrides` passes adapter-specific settings. The core only verifies
  that it is a mapping; each adapter validates its own supported keys.
- `date_range=(start, end)` bounds the run inclusively.
- `warmup_bars` excludes leading bars from derived result views.
- `base_currency` is required for portfolio input. Single-instrument runs can
  use the instrument's settlement currency.
- `signal.on_opposite_signal` is required for paper trading. Select `reverse`,
  `exit_only`, or `ignore` explicitly.

`BacktestConfig` is Python-native; there is no public YAML loader in the package.
The CLI has its own YAML loader, documented under [paper trading](paper-trading.md),
and it expects a serialized `BacktestConfig` shape, not a data-provider config.

See [risk and costs](risk-and-costs.md) for sizing and exit parameters.
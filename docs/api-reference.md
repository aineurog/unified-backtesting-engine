# API Reference

Complete public API surface for UBE. All signatures are current as of the latest
source. Import from the top-level `ube` package unless noted otherwise.

---

## Top-Level Functions

### `ube.run(data, signals, config, *, aux_data=None, log_path=None) -> BacktestResult`

Run a backtest. `data` is a `MarketData`, DataFrame, or dict (auto-standardized).
`signals` is a `Signals` or target array. `config` is a `BacktestConfig`.

```python
result = ube.run(bars_df, ube.from_target(target), config)
```

### `ube.from_target(target) -> Signals`

Convert a `-1/0/1` position array to four boolean signal columns. `0` means flat.

```python
signals = ube.from_target(np.array([0, 1, 1, -1, 0]))
```

### `ube.from_callable(fn, bars) -> Signals`

Evaluate `fn(prefix_market_data) -> int` over growing prefixes of `bars` and
convert the `-1/0/1` results to signals.

```python
def my_strategy(md: ube.MarketData) -> int:
    if md.close[-1] > md.close[-2]:
        return 1
    return 0

signals = ube.from_callable(my_strategy, market_data)
```

### `ube.register_engine(name, adapter_class) -> None`

Register a custom `EngineAdapter` subclass under `name`.

### `ube.get_engine(name='auto') -> type[EngineAdapter]`

Resolve an adapter class. `'auto'` picks the first installed engine in
VectorBT → Backtrader → NautilusTrader order.

### `ube.registered_engines() -> tuple[str, ...]`

Tuple of registered engine names.

### `ube.resolve_engine_name(name='auto') -> str`

Resolve `'auto'` to a concrete engine name.

### `ube.ensure_builtin_engines_registered() -> None`

Force-import and register all built-in adapters.

---

## Data

### `ube.MarketData(open, high, low, close, volume, index)`

Canonical single-instrument bar container. All arrays must be equal length.
`index` is a timezone-aware `pd.Index`.

**Constructors:**

| Method | Signature | Description |
|---|---|---|
| `from_dataframe` | `(df, *, column_map=None, timestamp_col=None)` | Build from a DataFrame |
| `from_dict` | `(data, *, column_map=None, timestamp_col=None)` | Build from a mapping |
| `from_records` | `(records, *, column_map=None, timestamp_col=None)` | Build from a sequence of mappings |
| `from_array` | `(arr, *, timestamps=None)` | Build from a 2-D array `(n, 5)` |
| `standardize` | `(data, *, column_map=None, timestamp_col=None, timestamps=None)` | Auto-detect input format |

**Instance methods:**

| Method | Signature | Description |
|---|---|---|
| `to_dataframe` | `() -> pd.DataFrame` | Export to DataFrame |
| `head` | `(n=5) -> MarketData` | First `n` bars |

---

## Signals

### `ube.Signals(long_entry, long_exit, short_entry, short_exit)`

Four boolean arrays, one element per bar. `False` = no action.

**Constructors:**

| Method | Signature | Description |
|---|---|---|
| `from_dataframe` | `(df) -> Signals` | Read four canonical columns |
| `from_array` | `(arr) -> Signals` | Build from a 2-D boolean array `(n, 4)` |

**Instance methods:**

| Method | Signature | Description |
|---|---|---|
| `to_dataframe` | `() -> pd.DataFrame` | Export to DataFrame |

---

## Configuration

### `ube.BacktestConfig(instrument, cost_model=None, risk=..., signal=..., benchmark=..., engine='auto', engine_overrides=None, date_range=None, base_currency=None, warmup_bars=0)`

The immutable run contract.

| Parameter | Type | Description |
|---|---|---|
| `instrument` | `Instrument` | Required. Symbol, asset class, tick size, etc. |
| `cost_model` | `CostModel \| None` | Fees/slippage/funding. `None` = asset-class default. |
| `risk` | `RiskConfig` | Sizing model + ordered exits. |
| `signal` | `SignalConfig` | Opposite-signal policy. |
| `benchmark` | `BenchmarkConfig` | Benchmark for comparison. |
| `engine` | `str` | `'auto'` or a registered name. |
| `engine_overrides` | `Mapping \| None` | Adapter-specific settings. |
| `date_range` | `tuple \| None` | Inclusive `(start, end)` bounds. |
| `base_currency` | `str \| None` | Required for portfolio runs. |
| `warmup_bars` | `int` | Leading bars excluded from result views. |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `validate` | `(*, portfolio=False, paper_trading=False) -> None` | Validate the config |

### `ube.Instrument(symbol, asset_class, tick_size=None, contract_multiplier=None, calendar=None, settlement_currency=None, funding_model=None, borrow_model=None, funding_interval_hours=None)`

| Parameter | Description |
|---|---|
| `symbol` | Instrument symbol (e.g. `"BTCUSDT"`). |
| `asset_class` | One of `stocks`, `forex`, `commodities`, `crypto_spot`, `crypto_perp`. |
| `tick_size` | Minimum price increment. |
| `contract_multiplier` | Contract size (futures). |
| `calendar` | Trading calendar name. |
| `settlement_currency` | Account settlement currency. |
| `funding_model` | Funding model name. |
| `borrow_model` | Borrow model name. |
| `funding_interval_hours` | Hours between funding payments. |

### `ube.RiskConfig(sizing=SizeModel(...), exit=())`

| Parameter | Description |
|---|---|
| `sizing` | A `SizeModel` instance. |
| `exit` | Ordered tuple of exit configs. |

### `ube.SizeModel(kind='all_in', value=None, leverage=1.0, vol=None)`

| Kind | Description |
|---|---|
| `fixed_fraction` | Allocate `value` fraction of capital to notional. |
| `fixed_units` | Use exactly `value` units. |
| `volatility_target` | Size to target portfolio volatility using `vol` series. |
| `all_in` | Allocate available capital. |
| `equal_weight` | Divide capital across `n` positions. |

| Parameter | Description |
|---|---|
| `kind` | Sizing kind string. |
| `value` | Fraction or unit count. |
| `leverage` | Exposure multiplier. |
| `vol` | Name of volatility series in `aux_data`. |

### `ube.CostModel(commission=0.0, slippage=0.0, funding=0.0, borrow=0.0)`

All rates are fractions (`0.001` = 0.1%).

### `ube.SignalConfig(on_opposite_signal=None)`

| Value | Description |
|---|---|
| `'reverse'` | Flip position on opposite signal. |
| `'exit_only'` | Close on opposite signal, do not re-enter. |
| `'ignore'` | Ignore opposite signals. |

Required for paper trading.

### `ube.BenchmarkConfig(kind='buy_and_hold', weights=None)`

---

## Exits

All exits are configured via `RiskConfig(exit=(...))`.

| Class | Signature | Description |
|---|---|---|
| `StopLoss` | `(percent, trigger='touched')` | Fixed percentage stop. |
| `TakeProfit` | `(percent, scale_out=1.0, trigger='touched')` | Fixed percentage target. |
| `TrailingStop` | `(percent, trigger='touched')` | Trailing percentage stop. |
| `TimeExit` | `(bars)` | Close after N bars. |
| `ATRStop` | `(mult, trigger='touched', trailing=False, period=14, atr=None)` | ATR-based stop. `atr` names an `aux_data` series. |
| `ChandelierExit` | `(mult, trigger='touched', period=14, atr=None)` | Chandelier exit. |

---

## Results

### `BacktestResult(run_id, ledger, config, trades, trade_table, positions, equity_curve, equity_curve_by_instrument, metrics=None, benchmark=None)`

| Attribute | Type | Description |
|---|---|---|
| `run_id` | `str` | Run identifier. |
| `ledger` | `EventLedger` | Canonical append-only event ledger. |
| `config` | `BacktestConfig` | The config used. |
| `trades` | `tuple[Trade, ...]` | Reconstructed round-trip trades. |
| `trade_table` | `pd.DataFrame` | One row per trade with derived fields. |
| `positions` | `Positions \| None` | Position series (single-instrument). |
| `equity_curve` | `EquityCurve` | Marked-to-market equity. |
| `equity_curve_by_instrument` | `dict[str, EquityCurve]` | Per-instrument equity. |
| `metrics` | `Any \| None` | Currently `None`. |
| `benchmark` | `BenchmarkCurve \| None` | Benchmark curve if configured. |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `save` | `(path) -> None` | Pickle the full result. |
| `load` | `(path) -> BacktestResult` | Load a pickled result (trusted sources only). |
| `from_ledger` | `(run_id, ledger, config, ...) -> BacktestResult` | Rebuild from a ledger. |

---

## Ledger

### `LedgerEvent`

Single event in the append-only ledger. Fields: `event_type`, `timestamp`,
`instrument_id`, `action`, `side`, `quantity`, `price`, `amount`, `currency`,
`exit_reason`, `position_after`, `order_id`, `trade_id`.

### `EventType` (enum)

`SIGNAL_EVALUATED`, `ORDER_SUBMITTED`, `FILL`, `FUNDING_PAYMENT`,
`FUTURES_ROLLOVER`, `COMMISSION`, `CASH_MOVEMENT`, `POSITION_CHANGE`

### `trades(ledger, instruments=None) -> tuple[Trade, ...]`

Reconstruct round-trip trades from fill/position events.

### `trade_table(ledger, market_data, instruments, *, base_currency, fx_rates=None, initial_capital=None, leverage=1.0, position_size=None) -> pd.DataFrame`

Build the reporting trade table.

### `equity_curve(ledger, market_data, instruments, *, base_currency, fx_rates=None) -> EquityCurve`

Build the equity curve.

---

## Paper Trading

### `ube.paper.init(config, *, run_id='default', db_path=None) -> PaperState`

Initialize paper state. `config` is a `PaperConfig`. `db_path` enables SQLite
persistence.

### `ube.paper.step(data, signals, state, config, aux_data=None) -> tuple[PaperState, list[LedgerEvent]]`

Process a window of bars. Returns updated state and new events.

### `ube.paper.run(strategy_name, data, signals, config, *, db_path=None, run_id=None, aux_data=None) -> tuple[PaperState, list[LedgerEvent]]`

Load-or-initialize wrapper around `step()`. Requires `db_path` or
`config.state_path`.

### `ube.paper.run_auto(data, signal_fn, config, state=None, *, run_id='default') -> list[LedgerEvent]`

Evaluate `signal_fn` over growing prefixes and process only the suffix after
the cursor.

### `PaperConfig(base, state_path=None, starting_balance=None, engine='nautilus', calendar_validate=True, calendar_strict=False)`

| Parameter | Description |
|---|---|
| `base` | A `BacktestConfig`. |
| `state_path` | Path for SQLite state persistence. |
| `starting_balance` | Convenience override. |
| `engine` | Paper backend: `'vectorbt'`, `'backtrader'`, or `'nautilus'`. |
| `calendar_validate` | Warn on out-of-session bars. |
| `calendar_strict` | Raise on out-of-session bars. |

### `PaperState`

| Attribute | Description |
|---|---|
| `instrument_id` | Canonical instrument id. |
| `ledger` | Append-only event ledger. |
| `last_processed_ns` | Bar cursor (ns). |
| `last_price` | Last mark price. |
| `open_position` | Current `OpenPosition` or `None`. |
| `pending_levels` | Pending exit levels. |
| `signal_fn_state` | Stateful signal function state. |
| `config_ref` | Persisted config reference. |
| `aux_data` | Auxiliary series. |
| `exit_seed` | Exit seed state. |
| `last_funding_ns` | Last funding timestamp. |
| `run_id` | Run identifier. |
| `db_path` | SQLite path. |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `save` | `() -> None` | Persist state. |
| `load` | `(db_path, run_id) -> PaperState` | Load persisted state. |
| `trade_table` | `(config, db_path, run_id) -> pd.DataFrame` | Build trade table. |
| `window_start_ns` | `() -> int` | Replay window start. |
| `get_config_dict` | `() -> dict` | Export config as dict. |

### `OpenPosition`

| Attribute | Description |
|---|---|
| `side` | `1` (long) or `-1` (short). |
| `quantity` | Position size. |
| `entry_price` | Fill price. |
| `entry_ns` | Entry bar timestamp (ns). |
| `trade_id` | Trade identifier. |

---

## Engine Adapter Extension

Implement a concrete subclass of `EngineAdapter` with
`run(data, signals, config, *, aux_data=...) -> BacktestResult`, then register
it with `ube.register_engine("name", AdapterClass)`. Keep engine-specific types
inside the adapter and add integration and parity tests. See
[contributing](contributing.md) and the existing adapter packages under
`src/ube/adapters/`.
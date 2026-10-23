# Statistics

UBE computes a full statistical profile of every run with
[StratStat](https://pypi.org/project/stratstat/) (`stratstat>=0.1.0`, a core
dependency). The same four tiers are produced for backtests and for paper
sessions:

| Tier | Input | Source in UBE |
|---|---|---|
| `returns` | Per-bar strategy returns | `result.equity_curve.returns`; paper: persisted equity points |
| `trades` | Closed round-trip trades | `result.trades`; paper: `trades(state.ledger, ...)` |
| `exposure` | Position weights over time | `positions(ledger)` priced with the run's bars |
| `benchmark` | Benchmark returns | `result.benchmark` (attached by `ube.run()`) |

## Backtests

`result.stats` is computed eagerly, before `ube.run()` returns, so it is part of
the result you log, pickle, or compare:

```python
result = ube.run(data, signals, config)

result.stats.n_metrics            # 151 with all four tiers
result.stats.tiers                # ('returns', 'trades', 'exposure', 'benchmark')
result.stats.values["sharpe_ratio"]
result.stats.values["max_drawdown"]
result.stats.objects["gross_exposure"]   # non-scalar metrics are lists/dicts
result.stats.to_dict()            # JSON-serialisable payload
result.stats.table()              # metric / value / category DataFrame
```

`result.metrics` remains `None`; `stats` is the populated field.

### `StatsResult`

| Attribute | Description |
|---|---|
| `values` | Scalar metrics (`float`, may be `NaN`/`inf` when undefined). |
| `objects` | Non-scalar metrics (distributions, fits) as JSON-able lists/dicts. |
| `categories` | `metric -> "category[.subcategory]"` as stratstat tags them. |
| `periods_per_year` | Annualisation actually used, or `None` when unknown. |
| `tiers` | Tiers that were computed, in order. |
| `meta` | Provenance: stratstat meta, citation `refs`, return policy, `input` counts, warnings. |
| `errors` | `input -> reason` for everything skipped. Computing never raises. |
| `computed_at_ns` | UTC timestamp of the computation (ns). |

Methods: `get(name, default=None)`, `category(name)`, `table()`, `to_dict()`,
`to_records()` (sqlite rows), `with_benchmark(curve, returns=..., index=...)`.

## Paper trading

Paper sessions have no bars stored, so `PaperState.stats()` rebuilds the returns
tier from the equity points written by `step()`:

```python
snapshot = state.stats(data, config)     # PaperConfig or its inner BacktestConfig
snapshot.meta["equity_source"]           # 'persisted' | 'ledger' | 'none'
```

`step()` also persists a snapshot on every trade close into two append-only
tables in the session database:

| Table | Columns |
|---|---|
| `stats` | `run_id`, `seq`, `metric`, `value`, `value_json`, `category`, `ref` |
| `stats_meta` | `run_id`, `seq`, `as_of_ns`, `trade_count`, `periods_per_year`, `tiers`, `errors_json` |

```python
from ube.stats import load_stats

latest = load_stats("paper.db", "default")           # newest snapshot, one row per metric
previous = load_stats("paper.db", "default", seq=1)  # an earlier snapshot
```

Snapshots are written only when the closed-trade count changes (or when an
earlier snapshot was taken before enough equity points existed to compute
returns), so a session that books no trades does not rewrite the database.

## Assembly API

| Function | Purpose |
|---|---|
| `compute(returns, trades=None, benchmark=None, exposure=None, *, periods_per_year=None, return_policy='truncate', rf=0.0, ...)` | Tiers from already-built inputs. |
| `from_arrays(index, equity, *, trades, ledger=None, market_data=None, instruments=None, benchmark=None, ...)` | Build from an equity grid (used by `BacktestResult.from_ledger`). |
| `from_result(result, **kwargs)` | Recompute from a result's cached views. |
| `save_stats(db_path, run_id, stats, *, trade_count=0) -> int` | Persist a snapshot, returns its `seq`. |
| `load_stats(db_path, run_id='default', *, seq=None) -> pd.DataFrame` | Read a snapshot. |
| `set_enabled(bool)` / `is_enabled()` | Global switch; disabled runs leave `result.stats is None`. |

## Conventions and caveats

- **Trade PnL** is passed to stratstat as a fraction of entry notional
  (`net_pnl / entry_notional`). If any trade lacks an entry notional the tier
  falls back to raw currency, and stratstat skips the metrics that require a
  fraction.
- **Trade duration** is measured in bar periods, derived from the median bar
  spacing of the equity grid.
- **Exposure** input is a portfolio weight (`units * close * contract_multiplier
  / equity`), the definition stratstat expects. When the supplied bars start
  after the equity grid — the normal case for a live `step()` window — the
  exposure series covers only that window; `meta["exposure_window"]` says so.
- **Annualisation** is inferred from the median bar spacing
  (`365.25` days per year); pass `periods_per_year=` to override.
- **Benchmark** alignment is positional: `BenchmarkCurve` carries no index and
  is built from the same sliced bars. A length mismatch truncates to the common
  length and records `meta["benchmark_note"]`.
- **Insolvent returns.** `EquityCurve.returns` yields `NaN` once equity drops to
  zero. `return_policy` chooses `"truncate"` (default: stop at the first
  non-finite return), `"zero"` (treat it as `0.0`), or `"drop"` (remove the
  point); the count is recorded in `meta["dropped_returns"]`.
- Failures never raise: a tier that cannot be computed is reported in
  `result.stats.errors`, and the remaining tiers are still produced.

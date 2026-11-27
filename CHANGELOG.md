# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-11-27

### Added

- `BacktestConfig.compute_stats` (default `True`): a per-run switch that skips the
  eager `result.stats` pass. When `False`, `BacktestResult.stats` is `None` while the
  benchmark, equity, trades, positions, and trades table are still built and attached.
  It is per-run and concurrency-safe, unlike the process-global
  `ube.stats.set_enabled()`.

### Changed

- Requires `stratstat>=0.2.0`. New stratstat metrics flow through `StatsResult`
  automatically (scalars in `values`, everything else in `objects`); no per-metric UBE
  change is needed. `result.metrics` remains `None`.
- Percentage metrics adopted from stratstat now report on a 0–100 scale, matching
  their `_pct` names: `long_exposure_pct`, `short_exposure_pct`, `long_short_trade_pct`,
  and the percentage element of `underperforming_periods`. Consumers that assumed 0–1
  fractions must rescale. Paper-trading stats rows written before stratstat 0.2.0 store
  the old 0–1 values, so old and new rows in one session are not directly comparable.

## [0.5.1] - 2026-11-25

### Fixed

- vectorbt fold: a phantom re-entry opened while the core engine was still holding a leg
  is now dropped. vectorbt's entry-anchored fraction stop can close a trade *earlier*
  than the core rule; the fold re-times that close later (`core_stop_exit_window`), but
  vectorbt had already freed the slot and opened any later entry signal as a fresh record.
  The re-entry guard only suppressed a record entering on the re-timed exit bar itself, so
  an entry one or more bars after vectorbt's native stop leaked into the ledger as a second
  concurrent position. The guard now drops every record entering at or before the core exit
  bar. Symptoms were a summed `position_after` (double-counted size, e.g. `position_size_pct`
  ~271%), equity collapsing to zero or below, and NaN bar returns.
- vectorbt fold: `position_after` is now derived from the fills in `(timestamp, emission)`
  order rather than from the record-iteration running total. When re-timing moved an exit
  past another record's entry, the record-order total stamped a chronologically
  inconsistent position stream (and a correspondingly wrong funding step series).

## [0.5.0] - 2026-11-25

### Added

- `TimeExit(duration=timedelta(...))`: a wall-clock holding-period exit alongside the
  existing `TimeExit(bars=n)` bar-count form. Exactly one form is required. The
  duration form exits at the first bar whose timestamp is at or after
  `entry_bar.timestamp + duration`, which makes it meaningful on irregular
  (event-driven/custom) bars where a bar count does not correspond to a fixed holding
  period. The duration is exact elapsed UTC time, with no calendar/session/DST
  adjustment. Supported by vectorbt, backtrader, Nautilus, and all paper engines;
  durations are stored as ISO-8601 strings in paper configs and the experiment log.
  The bar-count form is unchanged.

## [0.4.0] - 2026-11-19

### Added

- Generic engine discovery through Python entry points. An installed package can
  advertise a backtest adapter in the `ube.engines` group and a paper-trading
  backend in the `ube.paper_engines` group; `ube` resolves the entry point lazily
  when a config names that engine. Registration failures are isolated per plugin,
  and plugin engines are never selected by `engine="auto"`. In-process
  `ube.register_engine(...)` is unchanged. Documented in
  `docs/engines/overview.md`.

## [0.3.0] - 2026-11-06

### Added

- `ATRTakeProfit(mult, trigger="touched", scale_out=1.0, period=14, atr=None)`:
  an ATR-scaled take-profit target (`entry ± mult * ATR`) that mirrors the fixed
  `ATRStop` level and can scale out like `TakeProfit`. It requires a named `atr`
  `aux_data` series and is honoured by vectorbt, backtrader, Nautilus, and all
  paper engines.
- `RiskRewardTP(ratio, stop, trigger="touched", scale_out=1.0)`: a take-profit
  anchored at a multiple of the linked stop's initial risk, set once at entry.
  `stop` is a `StopLoss`, `ATRStop`, `TrailingStop`, or `ChandelierExit`; a nested
  ATR stop resolves the same named `atr` `aux_data` series as a top-level one.
  Honoured by vectorbt (via the core re-timing fold), backtrader, Nautilus, and all
  paper engines; vectorbt rejects partial `scale_out`, as it does for the other
  take-profits. `risk_reward_initial_risk_distance` /
  `risk_reward_take_profit_level` expose the risk and target computations, and
  `exit_atr_config` resolves a nested ATR config.
- `TrailingStop(..., activation=...)`: the stop now holds a static entry-anchored
  level (`entry * (1 - percent)` long / `entry * (1 + percent)` short) until price
  moves `activation` in the favourable direction, then trails the running
  peak/trough. The default `activation=None` preserves the previous behaviour. The
  level is computed in core, so vectorbt, backtrader, Nautilus, and all three paper
  engines share it.
- Statistics: integrate StratStat (`stratstat>=0.1.0`). `BacktestResult.stats`
  is computed eagerly by `ube.run()` across the returns, trades, exposure, and
  benchmark tiers; paper sessions replace the current wide-format stats row in
  sqlite on every trade close and expose `PaperState.stats()` with
  `load_stats()` / `save_stats()`. `result.metrics` is unchanged (`None`).

## [0.2.0] - 2026-11-05

### Changed

- **Breaking:** signals are decision-time. A flag on bar `t` means "decided at the
  close of bar `t`", and a new `BacktestConfig.execution_timing` chooses the fill
  bar: `"next_open"` (default — bar `t+1`'s open), `"next_close"` (bar `t+1`'s
  close), or `"same_close"` (bar `t`'s close). Previously vectorbt and nautilus
  filled on the signal bar's close while backtrader filled on the next bar's open,
  so the same signals produced different results per engine. All engines now honour
  the same setting. Set `execution_timing="same_close"` to restore the old
  vectorbt/nautilus behaviour.
- Paper trading on Nautilus, vectorbt, and backtrader carries a decision taken on a
  step's final bar in `PaperState.pending_decision` and executes it on the first
  subsequent bar using the configured timing. This preserves final-bar signals across
  one-bar polls; vectorbt/backtrader still require the replay history needed to rebuild
  an open position. The paper-state sqlite schema migrates automatically.
- `run_auto()` passes vectorbt/backtrader the full supplied series instead of the
  post-cursor suffix, so a carried position can resume across one-bar polls instead of
  failing when the fetch window excludes the entry bar.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  take-profits. `initial_risk_distance` / `risk_reward_take_profit_level` expose the
  risk and target computations, and `exit_atr_config` resolves a nested ATR config.
- `TrailingStop(..., activation=...)`: the stop now holds a static entry-anchored
  level (`entry * (1 - percent)` long / `entry * (1 + percent)` short) until price
  moves `activation` in the favourable direction, then trails the running
  peak/trough. The default `activation=None` preserves the previous behaviour. The
  level is computed in core, so vectorbt, backtrader, Nautilus, and all three paper
  engines share it.
- Initial project scaffold (structure only, no implementation).
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

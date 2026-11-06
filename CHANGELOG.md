# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

- Initial project scaffold (structure only, no implementation).
- Statistics: integrate StratStat (`stratstat>=0.1.0`). `BacktestResult.stats`
  is computed eagerly by `ube.run()` across the returns, trades, exposure, and
  benchmark tiers; paper sessions replace the current wide-format stats row in
  sqlite on every trade close and expose `PaperState.stats()` with
  `load_stats()` / `save_stats()`. `result.metrics` is unchanged (`None`).

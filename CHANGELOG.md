# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Initial project scaffold (structure only, no implementation).
- Statistics: integrate StratStat (`stratstat>=0.1.0`). `BacktestResult.stats`
  is computed eagerly by `ube.run()` across the returns, trades, exposure, and
  benchmark tiers; paper sessions replace the current wide-format stats row in
  sqlite on every trade close and expose `PaperState.stats()` with
  `load_stats()` / `save_stats()`. `result.metrics` is unchanged (`None`).

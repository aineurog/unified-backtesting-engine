# Examples

These examples mirror the public `ube` API documented in [docs/getting-started.md](../docs/getting-started.md), [docs/core-concepts.md](../docs/core-concepts.md), and [docs/paper-trading.md](../docs/paper-trading.md).

Run them from the repository root:

```bash
python examples/01_quickstart.py
python examples/02_data_and_signals.py
python examples/03_engine_comparison.py
python examples/04_risk_and_costs.py
python examples/05_paper_trading.py
python examples/06_results_and_ledger.py
```

## Included coverage

- Canonical `BacktestConfig` and `PaperConfig` initialization
- `from_target` and `from_callable` signal construction
- Backtest exits using `ATRStop` and `TrailingStop`
- Both ATR modes: precomputed aux data and engine-computed ATR from the bars
- Paper trading through `step`, `run`, and `run_auto`
- Result inspection via trade tables, ledger output, and equity curves

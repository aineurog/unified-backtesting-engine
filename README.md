# Unified Backtesting Engine

A Python library that standardizes the input contract, risk/exit layer, and
post-backtest analysis (statistics and reporting) across multiple backtesting
engines — vectorbt, backtrader, and NautilusTrader. It does not replace those
engines; it provides one interface and one canonical output format on top of
them, plus forward/paper testing for engines that lack it natively.

> **Note:** This library is **not yet released**. It is under active
> development and the public API is unstable.

## Install

```bash
pip install -e ".[dev]"
```

## Development quickstart

```bash
pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).

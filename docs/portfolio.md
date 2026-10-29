# Multi-Asset Portfolios

A multi-asset portfolio is several instruments trading from **one shared pool of
capital** and reporting **one combined equity curve**. It is different from passing
several instruments to an engine as isolated buckets: the portfolio mode splits the
shared balance by an explicit allocation policy and folds every leg's fills into the
canonical ledger, so the result's equity curve is the portfolio's, not a collection of
independent curves.

The single-instrument path is unchanged. Portfolio mode is opt-in through
`BacktestConfig.portfolio` (backtests) or `PaperConfig.portfolio` (paper trading).

- Backtest example: [`examples/07_portfolio_backtest.py`](../examples/07_portfolio_backtest.py)
- Paper-trading example: [`examples/08_portfolio_paper_trading.py`](../examples/08_portfolio_paper_trading.py)

Both examples run single-asset and multi-asset cases for NautilusTrader, VectorBT and
Backtrader.

## Configuration

`PortfolioConfig` is the portfolio-level counterpart of `SizeModel`:

```python
import ube

btc = ube.Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT")
eth = ube.Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT")

portfolio = ube.PortfolioConfig(
    instruments=(btc, eth),
    base_currency="USDT",
    allocation=ube.EqualWeight(),          # or ube.FixedWeight(...)
)
```

| Field | Meaning |
|---|---|
| `instruments` | Every traded instrument. Symbols must be unique and are the keys used by `data`, `signals` and `weights`. |
| `base_currency` | Currency of the combined equity curve. Required; never silently defaulted. |
| `allocation` | How the shared pool is split: `EqualWeight()` or `FixedWeight(weights={...})`. |
| `sizing` | Optional portfolio-wide `SizeModel`. `None` keeps each leg's `config.risk.sizing`. |
| `rebalance` | Reserved. Must be `None` in this release; forced rebalancing is not implemented. |

### Allocation policies

| Policy | Split |
|---|---|
| `EqualWeight()` | `1/n` for each configured instrument. |
| `FixedWeight(weights={"BTC-USDT": 0.7, "ETH-USDT": 0.3})` | The declared static fractions. Fractions must be finite, non-negative, cover every instrument, and sum to at most `1.0`; any remainder stays in cash. |

Unknown keys, negative weights, over-allocation, duplicate instrument symbols and an
empty instrument list all raise `ConfigError`.

### Sizing rules

Portfolio runs reject `SizeModel(kind="fixed_units")`: units do not scale with an
allocation, so they cannot respect the shared pool. Use a capital-proportional sizer
(`all_in`, `equal_weight`, `fixed_fraction`, or `volatility_target`) or set
`PortfolioConfig.sizing` explicitly.

## Backtest

Pass a `{instrument_id: MarketData}` mapping and a `{instrument_id: Signals}` mapping:

```python
import ube

config = ube.BacktestConfig(
    instrument=btc,              # any configured instrument (required by the contract)
    base_currency="USDT",
    portfolio=portfolio,
    risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
    engine="vectorbt",           # or "backtrader" / "nautilus"
    engine_overrides={"starting_balance": 100_000.0},
)

data = {
    "BTC-USDT": ube.MarketData.from_dataframe(btc_bars, timestamp_col="timestamp"),
    "ETH-USDT": ube.MarketData.from_dataframe(eth_bars, timestamp_col="timestamp"),
}
signals = {symbol: ube.from_target(target) for symbol in data}

result = ube.run(data, signals, config)
result.equity_curve                    # one combined portfolio curve
result.equity_curve_by_instrument      # per-instrument mark-to-market breakdown
result.trades                          # every leg, tagged by instrument_id
result.benchmark                       # equal-weight buy-and-hold by default (§10)
result.stats                           # statistics over the combined curve
```

What happens internally:

1. The shared account balance comes from `engine_overrides["starting_balance"]`
   (default `100_000.0`) and is denominated in `base_currency`.
2. Each leg is delegated to the selected engine adapter exactly as a single-instrument
   run. Its seed is that leg's share, converted from `base_currency` into the leg's
   settlement currency: `starting_balance = shared_balance * weight / rate`.
3. When `FixedWeight` leaves part of the pool unallocated, that remainder is booked as
   a base-currency cash position on the first bar, so the combined equity starts at the
   full shared balance rather than the sum of the leg seeds.
4. The per-leg ledgers are merged chronologically into one ledger.
5. `BacktestResult.from_ledger` derives the combined curve, per-instrument breakdown,
   trades and statistics from that merged ledger. Settlement currencies are converted to
   `base_currency` using declared FX rates (`fx_rates` / `synthetic_rates` /
   `fixed_conversions` / `currency_fx` in `engine_overrides`); the documented USD/USDT
   stablecoin peg applies when no explicit rate is declared. Any other undeclared pair
   raises `FXRateUnavailableError` — a rate is never guessed.
6. A portfolio benchmark is attached to the result. `result.benchmark` defaults to an
   equal-weight buy-and-hold of the configured instruments (each leg bought on its first
   bar, held to the end); pass a `BenchmarkConfig` to override it. Custom weights are
   ordered by `PortfolioConfig.instrument_ids` and normalized to sum to one.

Because each leg receives only its share of the pool, aggregate open commitments cannot
exceed the shared balance. The allocation is static for the life of the run — this is a
shared *capital pool*, not a rebalancing or cross-instrument risk model.

### Benchmark

`result.benchmark` is a `BenchmarkCurve` over the same timestamp grid as
`result.equity_curve`, so `result.stats` reports benchmark-relative metrics
(alpha, beta, information ratio, …) alongside the portfolio's own. The default is
equal-weight buy-and-hold; `BenchmarkConfig(kind="custom", weights=(0.6, 0.4))` selects
an explicit weight vector ordered by `PortfolioConfig.instrument_ids`, and
`kind="buy_and_hold"` is only meaningful for a single-instrument portfolio (it is
rewritten to equal-weight when several instruments are configured).

### Single-asset vs multi-asset on each engine

```python
for engine in ("vectorbt", "backtrader", "nautilus"):
    # Single asset: no PortfolioConfig.
    single = ube.BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        engine=engine,
        engine_overrides={"starting_balance": 100_000.0},
    )
    ube.run(btc_data, btc_signals, single)

    # Multi asset: same engine, shared pool.
    multi = ube.BacktestConfig(
        instrument=btc,
        base_currency="USDT",
        portfolio=portfolio,
        risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
        engine=engine,
        engine_overrides={"starting_balance": 100_000.0},
    )
    ube.run(data, signals, multi)
```

The same two calls work for every adapter. Engine differences in fill timing, price
precision and lot constraints still apply, so compare combined curves with the same
tolerance rules used by the single-instrument parity suite.

## Paper Trading

Set `PaperConfig.portfolio` (or `BacktestConfig.portfolio` inside the base config). The
session state is a `PortfolioPaperState`, which owns one ordinary engine state per
instrument:

```python
import ube

base = ube.BacktestConfig(
    instrument=btc,
    base_currency="USDT",
    portfolio=portfolio,
    risk=ube.RiskConfig(sizing=ube.SizeModel(kind="all_in")),
    signal=ube.SignalConfig(on_opposite_signal="reverse"),
    engine_overrides={"starting_balance": 100_000.0},
)
config = ube.paper.PaperConfig(
    base=base,
    engine="vectorbt",
    starting_balance=100_000.0,
    state_path="paper.db",
)

state, events = ube.paper.run(
    "multi-asset-demo",
    {
        "BTC-USDT": btc_data,
        "ETH-USDT": eth_data,
    },
    {
        "BTC-USDT": btc_signals,
        "ETH-USDT": eth_signals,
    },
    config,
)
```

`init`, `step` and `run_auto` accept the same mapping shape:

```python
state = ube.paper.init(config, run_id="demo", db_path="paper.db")

state, events = ube.paper.step(
    {"BTC-USDT": btc_data, "ETH-USDT": eth_data},
    {"BTC-USDT": btc_signals, "ETH-USDT": eth_signals},
    state,
    config,
)

events = ube.paper.run_auto(
    {"BTC-USDT": btc_data, "ETH-USDT": eth_data},
    {"BTC-USDT": btc_fn, "ETH-USDT": eth_fn},   # one callable per leg
    config,
    state,
)
```

Key properties:

- `state.instrument_ids` is the ordered instrument list.
- `state.open_positions` maps each instrument to its open position or `None`.
- `state.ledger` is the combined, chronologically ordered ledger.
- Persistence writes each leg under `<run_id>::<symbol>` plus one portfolio manifest
  row under `<run_id>`; the parent run stores one combined trade table, one combined
  equity point per step, and one combined statistics snapshot.
- `state.stats(data_map, config)` returns statistics over the combined curve with
  `meta["is_portfolio"] == True`.
- An all-zero allocation leg is skipped rather than run with zero capital.

The same static allocation rule as the backtest applies: each leg is seeded with
`shared_balance * weight / rate`, where `rate` converts `base_currency` into that leg's
settlement currency. Unallocated `FixedWeight` cash is tracked on the session and folded
back into the combined equity and statistics snapshot. `rebalance` remains reserved, and
no cross-instrument netting or hedging is performed.

**Currency limitation.** A leg whose settlement currency differs from `base_currency`
needs a declared FX rate in `engine_overrides` (the same `fx_rates` family as the
backtest) unless the pair is the documented 1:1 USD/USDT stablecoin peg. Any other
undeclared pair raises `FXRateUnavailableError` when the session is created. `base.base_currency`
and `PortfolioConfig.base_currency` must agree; a disagreement raises `ConfigError`.

## Engine notes

| Engine | Portfolio behavior |
|---|---|
| **VectorBT** | Each leg is a normal VectorBT run seeded with its share; the merged ledger drives the portfolio curve. Paper trading keeps the window-replay resume semantics per leg. |
| **Backtrader** | Each leg runs as a single-instrument adapter instance seeded with its pre-split, FX-converted share, so one asset cannot silently starve another; the merged ledger drives the portfolio curve. Paper trading keeps the replay/checkpoint semantics per leg. |
| **NautilusTrader** | Each leg runs through the Nautilus adapter with its share and the standard sandbox execution model. Paper trading reuses the incremental per-instrument backend and existing resume behavior. |

Details: [VectorBT](engines/vectorbt.md), [Backtrader](engines/backtrader.md),
[NautilusTrader](engines/nautilus.md).

## Limitations

- No scheduled or threshold rebalancing (`PortfolioConfig.rebalance` must be `None`).
- No `RiskParity` or volatility-weighted allocation; `EqualWeight` and `FixedWeight`
  only.
- No cross-instrument hedging/netting: this is a shared capital pool, not a shared risk
  model.
- Allocation is applied to the starting shared balance. Leg profits stay in that leg for
  the remainder of the run.
- The portfolio layer reuses each engine's existing fill simulation; it does not erase
  engine-specific execution differences.

## API summary

| Object | Purpose |
|---|---|
| `ube.PortfolioConfig` | Instruments, base currency, allocation, optional sizing. |
| `ube.EqualWeight` / `ube.FixedWeight` | Allocation policies. |
| `ube.AllocationPolicy` | Type alias for the two policies. |
| `ube.size_portfolio_entry(...)` | Portfolio-aware sibling of `size_position`: units available from a share of a shared pool. |
| `ube.paper.PortfolioPaperState` | Resumable multi-asset paper session. |
| `ube.paper.init_portfolio(...)` | Create a fresh portfolio paper session (called by `init`). |
| `ube.paper.step_portfolio(...)` | Process one mapping-shaped portfolio slice (called by `step`). |

# Testing

Install the docs, development, and engine extras, then run from the repository
root:

```powershell
python -m pip install -e ".[all,dev,docs]"
pytest
mkdocs build --strict
```

The test suite is organized as:

- `tests/unit/`: core contracts, sizing, exits, ledger, results, and state.
- `tests/integration/`: adapters, runner, paper backends, and calendar behavior.
- `tests/parity/`: locked cross-engine fixture runs and backtest/paper consistency.
- `tests/fixtures/`: deterministic synthetic bars, instrument metadata, and
  expected parity results.

Regenerate deterministic bar fixtures with:

```powershell
python tests/fixtures/generate.py
```

Review manifest and expected-result changes deliberately. A changed dependency
may alter synthetic data or floating-point output; do not update parity locks
without checking the resulting trades and equity.

The current parity suite covers selected strategies and asset-class fixtures; it
does not prove that all adapter features or every real market scenario produce
identical results. VectorBT, Backtrader, and NautilusTrader have different native
execution models. Always add a focused test for new behavior and run the relevant
adapter tests.

## Engine parity locks (§16)

Each asset class has one `expected_results.json` holding an `engines` block per
adapter (`vectorbt`, `nautilus`, `backtrader`). Every block is a real, reviewed
run — no placeholders — and `tests/parity/test_fixtures.py` fails if one is left
zeroed.

`tests/parity/_parity.py` is the only definition of the trade-table hash and the
fixture loader. All three engines run the trivial `trivial_long_roundtrip`
strategy over the same committed parquet bars and must agree **exactly** on the
trade count, the trade table hash and the final equity
(`tests/parity/test_cross_engine_parity.py`). That equality is only meaningful
because every adapter snaps OHLC onto the instrument's venue price grid before
running: without `core.snap_prices`, vectorbt would price the raw float64 bar
while Nautilus prices the rounded venue price, and the reported fills would
differ in the last float bits. Each engine's own parity file pins it to the
locked baseline independently.

Engine extras are optional. `tests/conftest.py` collects an engine test module
only when all of its engines are importable, so an install without extras runs
the core suite cleanly and each CI engine job runs exactly its own files.

## Synthetic data

`ube.testing` exposes the seeded generator behind the parity fixtures, so tests
(and callers) can reproduce identical bars:

```python
from ube.testing import PRESETS, synthetic_bars

md = synthetic_bars("crypto_perp", seed=0, n_bars=200)
print(PRESETS["crypto_perp"].instrument)  # the matching Instrument metadata
```

`synthetic_bars(asset_class, *, seed=0, n_bars=744, start="2024-01-01", freq=None)`
accepts an asset class string (a key of `PRESETS`) or an `AssetClassPreset`, and
returns a validated `MarketData` on a tz-aware UTC grid. The same
`(asset_class, seed, n_bars, start, freq)` always yields byte-identical OHLCV;
sessioned instruments are filtered to in-session bars. `DEFAULT_SEED`,
`DEFAULT_N_BARS`, and `DEFAULT_START` are exported for convenience.
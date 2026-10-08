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
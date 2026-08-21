# Deterministic fixtures (§16)

Per-asset-class synthetic market data used as the ground truth for adapter-parity
tests. Every engine adapter must produce the same result from these bars; any
disagreement is an adapter bug, never a data bug.

## Layout

```
tests/fixtures/
├── generate.py              # regenerates everything below (see below)
├── manifest.json            # seed / bar count / per-asset content hash
├── stocks/aapl_2024_01.parquet        + aapl_instrument.json
├── futures/es_2024_01.parquet         + es_instrument.json
├── commodities/gc_2024_01.parquet     + gc_instrument.json
├── forex/eurusd_2024_01.parquet       + eurusd_instrument.json
└── crypto_perp/btc_usdt_2024_01.parquet + btc_usdt_instrument.json
```

Each `<asset_class>/<stem>_2024_01.parquet` holds one month (31 days) of hourly
OHLCV bars from `ube.testing.synthetic.synthetic_bars`; the sibling
`<stem>_instrument.json` is the matching `Instrument` metadata (asdict).

## Regenerating

```bash
python tests/fixtures/generate.py
```

Output is deterministic: the same `ube`/numpy/pandas versions reproduce the
committed bytes exactly. If a dependency upgrade changes `default_rng` output,
the `manifest.json` content hashes drift — review and re-commit the fixtures
deliberately rather than letting parity baselines shift silently.

## `expected_results.json`

One file per asset class, produced by running the trivial parity strategy
(``trivial_long_roundtrip``) through each engine, reviewing for agreement, and locking
the result. Schema:

```json
{
  "asset_class": "futures",
  "instrument": "ES",
  "strategy": "trivial_long_roundtrip",
  "locked_at": "<ISO date>",
  "tolerance": { "final_equity_rtol": 1e-6 },
  "engines": {
    "vectorbt":    { "final_equity": 0.0, "n_trades": 0, "trades_hash": "" },
    "backtrader":  { "final_equity": 0.0, "n_trades": 0, "trades_hash": "" },
    "nautilus":    { "final_equity": 91725.0, "n_trades": 1, "trades_hash": "…" }
  }
}
```

All five asset classes have a locked **nautilus** baseline (the reference engine).
The ``vectorbt`` and ``backtrader`` blocks are zeroed placeholders — they are filled
with real values once those adapters are implemented, at which point cross-engine
parity is actually exercised. A baseline is a real, manually-reviewed run, never a
hand-invented number; the test asserts the engine reproduces it within the stated
tolerance (see ``tests/parity/test_nautilus_parity.py``).

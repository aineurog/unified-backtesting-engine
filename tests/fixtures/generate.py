"""Regenerate the deterministic per-asset-class fixtures of §16.

Usage::

    python tests/fixtures/generate.py

For each canonical asset class in ``ube.testing.synthetic.PRESETS`` this writes::

    <asset_class>/<basename>_2024_01.parquet    # deterministic OHLCV bars
    <asset_class>/<basename>_instrument.json    # Instrument metadata (asdict)

and a top-level ``manifest.json`` recording the generator seed, bar count, start
timestamp and a per-asset SHA-256 content hash — so a numpy/pandas upgrade that
silently changes ``default_rng`` output shows up as a manifest diff, not a
silent parity-baseline change.

The parquet files are committed (§16). ``expected_results.json`` (the per-engine
locked results of the trivial parity strategy) is *not* produced here — a baseline is
a reviewed value locked by hand, and ``generate.py`` must never regenerate (and thus
silently overwrite) it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ube import __version__
from ube.testing.synthetic import (
    DEFAULT_N_BARS,
    DEFAULT_SEED,
    DEFAULT_START,
    PRESETS,
    synthetic_bars,
)

FIXTURES_DIR = Path(__file__).resolve().parent


def _basename(symbol: str) -> str:
    """Fixture filename stem for a symbol (``BTC-USDT`` -> ``btc_usdt``)."""
    return symbol.lower().replace("-", "_")


def _content_hash(open_, high, low, close, volume, index) -> str:
    """SHA-256 over the raw OHLCV bytes and the bar timestamps (exact fingerprint)."""
    h = hashlib.sha256()
    for arr in (open_, high, low, close, volume):
        h.update(arr.tobytes())
    h.update(index.asi8.tobytes())
    return h.hexdigest()


def main() -> None:
    manifest: dict[str, object] = {
        "generator": "ube.testing.synthetic",
        "ube_version": __version__,
        "seed": DEFAULT_SEED,
        "n_bars": DEFAULT_N_BARS,
        "start": DEFAULT_START,
        "assets": {},
    }
    assets: dict[str, dict[str, object]] = {}
    for asset_class, preset in PRESETS.items():
        md = synthetic_bars(
            asset_class, seed=DEFAULT_SEED, n_bars=DEFAULT_N_BARS, start=DEFAULT_START
        )
        out_dir = FIXTURES_DIR / asset_class
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _basename(preset.instrument.symbol)

        md.to_dataframe().to_parquet(out_dir / f"{stem}_2024_01.parquet")
        with (out_dir / f"{stem}_instrument.json").open("w") as fh:
            json.dump(asdict(preset.instrument), fh, indent=2, sort_keys=True)
            fh.write("\n")

        assets[asset_class] = {
            "symbol": preset.instrument.symbol,
            "parquet": f"{asset_class}/{stem}_2024_01.parquet",
            "instrument": f"{asset_class}/{stem}_instrument.json",
            "row_count": md.n_bars,
            "content_hash": _content_hash(
                md.open, md.high, md.low, md.close, md.volume, md.index
            ),
        }

    manifest["assets"] = assets
    with (FIXTURES_DIR / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    for asset_class in sorted(assets):
        info = assets[asset_class]
        print(f"{asset_class:<12} {info['symbol']:<10} rows={info['row_count']} "
              f"sha256={str(info['content_hash'])[:16]}…")


if __name__ == "__main__":
    main()

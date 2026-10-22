"""Shared machinery for the §16 cross-engine parity suite.

Every parity test file runs *the same* trivial strategy over *the same* committed
fixtures and compares against *the same* lock file — so the fixture loading, the
target construction, the trade-table hash and the lock-file lookup live here exactly
once (§3), and each ``test_<engine>_parity.py`` file holds only its engine's run and
its engine's assertions.

The trade-table hash is the parity hinge: it is the canonical, sorted, field-by-field
fingerprint of a result's trades. It must be byte-for-byte reproducible across files,
because cross-engine equality is asserted through it (:mod:`test_cross_engine_parity`).
Serializing a timestamp as ``str(...)`` in one file and as the raw ``int`` in another
silently makes two *identical* trade tables hash differently — which is exactly the
bug this shared definition removes.

The lock files under ``tests/fixtures/<asset_class>/expected_results.json`` are the
single source of truth (§16): one ``engines.<engine>`` block per engine, captured from
a real run and never invented (dev-guide §4.6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.instrument import Instrument
from ube.core.result import BacktestResult
from ube.core.risk import RiskConfig
from ube.core.risk.sizing import SizeModel
from ube.core.signals import from_target

__all__ = [
    "FIXTURES_DIR",
    "PARITY_ASSETS",
    "LOCK_KEYS",
    "TRADE_FIELDS",
    "ENGINES",
    "load_instrument",
    "load_fixture",
    "fixture_hash",
    "manifest_info",
    "parity_target",
    "parity_config",
    "run_parity",
    "trades_hash",
    "lock_doc",
    "locked_engine",
    "equity_of",
]

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

#: The five canonical asset classes of §16 — each has a committed parquet fixture, an
#: instrument manifest entry, and a locked ``expected_results.json`` baseline.
#: ``asset_class -> (fixture folder, symbol, parquet stem)``.
PARITY_ASSETS: dict[str, tuple[str, str, str]] = {
    "futures": ("futures", "ES", "es"),
    "crypto_perp": ("crypto_perp", "BTC-USDT", "btc_usdt"),
    "commodities": ("commodities", "GC", "gc"),
    "forex": ("forex", "EURUSD", "eurusd"),
    "stocks": ("stocks", "AAPL", "aapl"),
}

#: The §16 lock schema — every ``engines.<name>`` block carries exactly these.
LOCK_KEYS: tuple[str, ...] = ("final_equity", "n_trades", "trades_hash")

#: The trade-table columns that make up the parity fingerprint, in serialization order.
TRADE_FIELDS: tuple[str, ...] = (
    "instrument_id",
    "side",
    "quantity",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "commission",
    "funding",
    "net_pnl",
    "exit_reason",
)

#: Every engine name that must appear in every lock file, and that the cross-engine
#: suite compares. All three adapters fill on the signal bar at its close (§9.4), so
#: the same bars necessarily produce the same trade table in each — equality is the
#: assertion, not a tolerance.
ENGINES: tuple[str, ...] = ("vectorbt", "nautilus", "backtrader")


def load_instrument(asset_class: str) -> Instrument:
    """Load only the fixture's instrument metadata (no parquet read)."""
    folder, _symbol, stem = PARITY_ASSETS[asset_class]
    with (FIXTURES_DIR / folder / f"{stem}_instrument.json").open() as fh:
        return Instrument(**json.load(fh))


def load_fixture(asset_class: str) -> tuple[MarketData, Instrument]:
    """Load a committed fixture pair (parquet bars + instrument metadata)."""
    folder, _symbol, stem = PARITY_ASSETS[asset_class]
    md = MarketData.from_dataframe(
        pd.read_parquet(FIXTURES_DIR / folder / f"{stem}_2024_01.parquet")
    )
    return md, load_instrument(asset_class)


def fixture_hash(md: MarketData) -> str:
    """SHA-256 over the raw OHLCV bytes and bar timestamps (as in ``generate.py``)."""
    h = hashlib.sha256()
    for arr in (md.open, md.high, md.low, md.close, md.volume):
        h.update(arr.tobytes())
    h.update(md.timestamps.asi8.tobytes())
    return h.hexdigest()


def manifest_info(asset_class: str) -> dict[str, Any]:
    """The ``manifest.json`` entry for ``asset_class`` (fixture provenance record)."""
    with (FIXTURES_DIR / "manifest.json").open() as fh:
        manifest = json.load(fh)
    return manifest["assets"][asset_class]


def parity_target(n_bars: int) -> np.ndarray:
    """Trivial parity strategy target: long from bar 1 to the last bar (flat ends).

    Deliberately the simplest possible strategy — one round trip, no stops, no sizing
    model, no warm-up. §16 calls for "a trivial strategy that *must* produce the same
    result through every engine"; anything richer would make a divergence ambiguous
    between strategy semantics and engine semantics.
    """
    target = np.zeros(n_bars, dtype=int)
    target[1:-1] = 1
    return target


def parity_config(instrument: Instrument) -> BacktestConfig:
    """The canonical parity config: ``fixed_units`` 1, 100k account, no cost model."""
    return BacktestConfig(
        instrument=instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 100000.0},
    )


def run_parity(adapter_cls: type, asset_class: str) -> BacktestResult:
    """Run the trivial parity strategy through ``adapter_cls`` on the fixture bars."""
    md, instrument = load_fixture(asset_class)
    return adapter_cls().run(
        md, from_target(parity_target(md.n_bars)), parity_config(instrument)
    )


def trades_hash(trades: Any) -> str:
    """Deterministic SHA-256 over the trade table (canonical, sorted by entry).

    The single definition for every engine: field values are serialized *as stored*, so
    two engines holding the same trade table hash identically regardless of which test
    file computed it.
    """
    records = [
        {field: getattr(trade, field) for field in TRADE_FIELDS}
        for trade in sorted(trades, key=lambda trade: trade.entry_timestamp)
    ]
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def lock_doc(asset_class: str) -> dict[str, Any]:
    """The whole ``expected_results.json`` document for ``asset_class``."""
    path = FIXTURES_DIR / PARITY_ASSETS[asset_class][0] / "expected_results.json"
    with path.open() as fh:
        return json.load(fh)


def locked_engine(asset_class: str, engine: str) -> dict[str, Any]:
    """The locked ``engines.<engine>`` baseline block for ``asset_class``."""
    return lock_doc(asset_class)["engines"][engine]


def equity_of(result: BacktestResult) -> float:
    """The final equity on a result's equity curve."""
    return float(result.equity_curve.equity[-1])

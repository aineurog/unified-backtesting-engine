"""Vectorbt cross-engine parity tests — one parity file per adapter (§16).

This is the single parity test file for the Vectorbt adapter, mirroring
``test_nautilus_parity.py``. It runs the trivial parity strategy
(``trivial_long_roundtrip`` — a `fixed_units` of 1 long from the second bar to the
last bar, so exactly one round trip) through the real committed fixtures of §16 using
the VectorbtAdapter and asserts the run reproduces the **locked** baseline below.

The locked values were captured from a real, manually-reviewed run — never invented
(dev-guide §4.6). They are immutable: any change that breaks the baseline is caught
here. They are kept inline in this file (rather than in the shared
``expected_results.json`` ``engines.vectorbt`` block) so the Nautilus parity file and
its fixtures stay untouched — cross-engine comparison against the Nautilus block is
intentionally *not* an assertion, since engine fill models and exit primitives differ
(fractional vs absolute stops), so each adapter is locked to its own baseline (§16
tolerance: exact trade tables/reasons may differ for dynamic stops).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.instrument import Instrument
from ube.core.result import BacktestResult
from ube.core.risk import RiskConfig
from ube.core.risk.sizing import SizeModel
from ube.core.signals import from_target

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

#: The five canonical asset classes of §16 — each has a committed parquet fixture, an
#: instrument manifest entry, and a locked ``expected_results.json`` baseline.
PARITY_ASSETS = {
    "futures": ("futures", "ES", "es"),
    "crypto_perp": ("crypto_perp", "BTC-USDT", "btc_usdt"),
    "commodities": ("commodities", "GC", "gc"),
    "forex": ("forex", "EURUSD", "eurusd"),
    "stocks": ("stocks", "AAPL", "aapl"),
}

LOCK_KEYS = ("final_equity", "n_trades", "trades_hash")

#: Locked vectorbt baseline (§16), one entry per parity asset. Captured from a real run;
#: do not edit without re-reviewing the engine output.
_VBT_LOCKED = {
    "futures": {
        "final_equity": 91725.0,
        "n_trades": 1,
        "trades_hash": "0aafa08e8f5d63608c746fb3094cba41584185cfec687c08aafbb956e9dc3810",
    },
    "crypto_perp": {
        "final_equity": 91264.14779999998,
        "n_trades": 1,
        "trades_hash": "45caee561c90ed184d93af48471c460c92a9c95f59bd03a41bf5d9949db856dc",
    },
    "commodities": {
        "final_equity": 92040.0,
        "n_trades": 1,
        "trades_hash": "a721e03098ac1ca54130770da78a6985df21f69d572625b973feb9b7ace0e3e1",
    },
    "forex": {
        "final_equity": 99999.981,
        "n_trades": 1,
        "trades_hash": "5953412d7feb00b02ce31104228ed6f0d720d4ace7c463f68e873081d878a16e",
    },
    "stocks": {
        "final_equity": 99987.29999999999,
        "n_trades": 1,
        "trades_hash": "bfe442a97c74c838d6cf4ce45b4bd9018600261e79a9cc2c6229dd5d91f9156c",
    },
}


def _fixture_hash(md: MarketData) -> str:
    """SHA-256 over the raw OHLCV bytes and bar timestamps (as in generate.py)."""
    h = hashlib.sha256()
    for arr in (md.open, md.high, md.low, md.close, md.volume):
        h.update(arr.tobytes())
    h.update(md.timestamps.asi8.tobytes())
    return h.hexdigest()


def _load_fixture(asset_class: str) -> tuple[MarketData, Instrument]:
    """Load a committed fixture pair (parquet bars + instrument metadata)."""
    folder, _symbol, stem = PARITY_ASSETS[asset_class]
    md = MarketData.from_dataframe(
        pd.read_parquet(FIXTURES_DIR / folder / f"{stem}_2024_01.parquet")
    )
    with (FIXTURES_DIR / folder / f"{stem}_instrument.json").open() as fh:
        instrument = Instrument(**json.load(fh))
    return md, instrument


def _parity_target(n_bars: int) -> np.ndarray:
    """Trivial parity strategy target: long from bar 1 to the last bar (flat ends)."""
    target = np.zeros(n_bars, dtype=int)
    target[1:-1] = 1
    return target


def _trades_hash(trades) -> str:
    """Deterministic SHA-256 over the trades table (canonical, sorted by entry)."""
    records = []
    for t in sorted(trades, key=lambda trade: trade.entry_timestamp):
        records.append(
            {
                "instrument_id": t.instrument_id,
                "side": t.side,
                "quantity": t.quantity,
                "entry_timestamp": str(t.entry_timestamp),
                "exit_timestamp": str(t.exit_timestamp),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "commission": t.commission,
                "funding": t.funding,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
            }
        )
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _parity_result(asset_class: str) -> BacktestResult:
    """Run the trivial parity strategy through the Vectorbt adapter."""
    from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter

    md, instrument = _load_fixture(asset_class)
    signals = from_target(_parity_target(md.n_bars))
    config = BacktestConfig(
        instrument=instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 100000.0},
    )
    return VectorbtAdapter().run(md, signals, config)


def _locked(asset_class: str) -> dict:
    return _VBT_LOCKED[asset_class]


# ---------------------------------------------------------------------------
# Fixture integrity (§16): the committed parquet must not drift silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_fixture_matches_manifest(asset_class: str):
    with (FIXTURES_DIR / "manifest.json").open() as fh:
        manifest = json.load(fh)
    info = manifest["assets"][asset_class]
    md, instrument = _load_fixture(asset_class)
    assert md.n_bars == info["row_count"]
    assert _fixture_hash(md) == info["content_hash"]
    assert instrument.symbol == info["symbol"]


# ---------------------------------------------------------------------------
# Locked baseline reproducibility (§16): vectorbt matches the inline baseline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_vectorbt_parity_matches_locked(asset_class: str):
    result = _parity_result(asset_class)
    locked_engine = _locked(asset_class)

    assert len(result.trades) == locked_engine["n_trades"]
    assert float(result.equity_curve.equity[-1]) == pytest.approx(
        locked_engine["final_equity"],
        rel=1e-6,
    )
    assert _trades_hash(result.trades) == locked_engine["trades_hash"]


def test_locked_baseline_schema_is_intact():
    """The inline vectorbt baseline carries the §16 schema and is fully filled."""
    assert set(_VBT_LOCKED) == set(PARITY_ASSETS)
    for _asset_class, block in _VBT_LOCKED.items():
        assert set(block) == set(LOCK_KEYS)
        assert block["final_equity"] != 0.0
        assert block["n_trades"] > 0
        assert block["trades_hash"]

"""NautilusTrader cross-engine parity tests — one parity file per adapter (§16).

This is the single parity test file for the Nautilus adapter, created in Step 8. It
runs the trivial parity strategy (``trivial_long_roundtrip`` — a `fixed_units` of 1
long from the second bar to the last bar, so exactly one round trip) through the
real committed fixtures of §16 and asserts the run reproduces the **locked** values
in ``tests/fixtures/<asset_class>/expected_results.json`` (``final_equity``,
``n_trades``, ``trades_hash`` — schema in ``tests/fixtures/README.md``).

The locked values were captured from a real, manually-reviewed run — never invented
(dev-guide §4.6). They are immutable: any change that breaks the baseline is caught
here. Cross-engine comparison activates when the vectorbt / backtrader adapters land
and fill their engine blocks — until then their blocks stay placeholder zeros.
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
    "stocks": ("stocks", "AAPL", "aapl"),
    "commodities": ("commodities", "GC", "gc"),
    "forex": ("forex", "EURUSD", "eurusd"),
}

LOCK_KEYS = ("final_equity", "n_trades", "trades_hash")


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
                "entry_timestamp": t.entry_timestamp,
                "exit_timestamp": t.exit_timestamp,
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
    """Run the trivial parity strategy through the Nautilus adapter."""
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter

    md, instrument = _load_fixture(asset_class)
    signals = from_target(_parity_target(md.n_bars))
    config = BacktestConfig(
        instrument=instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 100000.0},
    )
    return NautilusAdapter().run(md, signals, config)


def _locked(asset_class: str) -> dict:
    with (FIXTURES_DIR / PARITY_ASSETS[asset_class][0] / "expected_results.json").open() as fh:
        return json.load(fh)


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
# Locked baseline reproducibility (§16): nautilus matches expected_results.json.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_nautilus_parity_matches_locked(asset_class: str):
    result = _parity_result(asset_class)
    locked_engine = _locked(asset_class)["engines"]["nautilus"]

    assert len(result.trades) == locked_engine["n_trades"]
    assert float(result.equity_curve.equity[-1]) == pytest.approx(
        locked_engine["final_equity"],
        rel=_locked(asset_class)["tolerance"]["final_equity_rtol"],
    )
    assert _trades_hash(result.trades) == locked_engine["trades_hash"]


def test_locked_baseline_schema_is_intact():
    """The lock files carry the §16 schema; absent engines stay placeholders."""
    for asset_class in PARITY_ASSETS:
        doc = _locked(asset_class)
        assert set(doc) >= {
            "asset_class",
            "instrument",
            "strategy",
            "locked_at",
            "tolerance",
            "engines",
        }
        engines = doc["engines"]
        assert set(engines) == {"vectorbt", "backtrader", "nautilus"}
        for _name, block in engines.items():
            assert set(block) == set(LOCK_KEYS)
        nautilus = engines["nautilus"]
        assert nautilus["final_equity"] != 0.0
        assert nautilus["n_trades"] > 0
        assert nautilus["trades_hash"]
        for name in ("vectorbt", "backtrader"):
            placeholder = engines[name]
            assert placeholder["final_equity"] == 0.0
            assert placeholder["n_trades"] == 0
            assert placeholder["trades_hash"] == ""
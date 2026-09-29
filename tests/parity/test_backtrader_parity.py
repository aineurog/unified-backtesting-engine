"""Backtrader cross-engine parity tests — one parity file per adapter (§16).

This is the single parity file for the backtrader adapter. It runs the trivial
parity strategy (``trivial_long_roundtrip`` — a `fixed_units` of 1 long from the
second bar to the last bar, so exactly one round trip) through the real committed
fixtures of §16 and asserts the run reproduces the **locked** values in
``tests/fixtures/<asset_class>/expected_results.json`` (``final_equity``,
``n_trades``, ``trades_hash`` — schema in ``tests/fixtures/README.md``).

The locked values were captured from a real run (backtrader fills market orders at the
*same* bar's close — §9.4 parity with the nautilus reference — so the ``backtrader``
block's ``trades_hash`` is byte-identical to the ``nautilus`` block's for every
asset class; commission/slippage/funding are all zero for the fixture cost model) and
cross-checked against the Nautilus baseline: the same instrument, the same target, the
same cost model, and the same fill timing. The lock is per-engine and immutable: any
change that breaks it is caught here.
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

#: The five canonical asset classes of §16 — matching the Nautilus parity lock.
PARITY_ASSETS = {
    "futures": ("futures", "ES", "es"),
    "crypto_perp": ("crypto_perp", "BTC-USDT", "btc_usdt"),
    "commodities": ("commodities", "GC", "gc"),
    "forex": ("forex", "EURUSD", "eurusd"),
    "stocks": ("stocks", "AAPL", "aapl"),
}

LOCK_KEYS = ("final_equity", "n_trades", "trades_hash")


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
    """Run the trivial parity strategy through the backtrader adapter."""
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter

    md, instrument = _load_fixture(asset_class)
    signals = from_target(_parity_target(md.n_bars))
    config = BacktestConfig(
        instrument=instrument,
        risk=RiskConfig(sizing=SizeModel(kind="fixed_units", value=1.0)),
        engine_overrides={"starting_balance": 100000.0},
    )
    return BacktraderAdapter().run(md, signals, config)


def _locked(asset_class: str) -> dict:
    with (FIXTURES_DIR / PARITY_ASSETS[asset_class][0] / "expected_results.json").open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Locked baseline reproducibility (§16): backtrader matches its own locked block.
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_backtrader_parity_matches_locked(asset_class: str):
    result = _parity_result(asset_class)
    locked_engine = _locked(asset_class)["engines"]["backtrader"]

    assert locked_engine["final_equity"] != 0.0  # real value, not a placeholder
    assert locked_engine["n_trades"] > 0
    assert len(result.trades) == locked_engine["n_trades"]
    assert float(result.equity_curve.equity[-1]) == pytest.approx(
        locked_engine["final_equity"],
        rel=_locked(asset_class)["tolerance"]["final_equity_rtol"],
    )
    assert _trades_hash(result.trades) == locked_engine["trades_hash"]


# ---------------------------------------------------------------------------
# Cross-engine lock identity (§16): backtrader must stay byte-equal to Nautilus.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_backtrader_block_matches_nautilus_block(asset_class: str):
    """The backtrader lock must reproduce the Nautilus lock exactly.

    Same-bar-close fills (§9.4) mean the backtrader vault reproduces the Nautilus
    baseline byte-for-byte (same trades, same recorded floats), so the two blocks in
    ``expected_results.json`` are identical. A divergence here would flag a real
    fill-timing regression, not fixture churn.
    """
    result = _parity_result(asset_class)
    fixtures = _locked(asset_class)
    engines = fixtures["engines"]
    nautilus = engines["nautilus"]
    backtrader = engines["backtrader"]

    assert _trades_hash(result.trades) == nautilus["trades_hash"]
    assert backtrader["trades_hash"] == nautilus["trades_hash"]
    # The trades are byte-identical; the folded running balance may still accumulate
    # float noise (≤ rtol) in each engine's equity aggregation.
    assert float(backtrader["final_equity"]) == pytest.approx(
        float(nautilus["final_equity"]),
        rel=fixtures["tolerance"]["final_equity_rtol"],
    )
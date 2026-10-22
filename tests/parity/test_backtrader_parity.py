"""Backtrader cross-engine parity tests — one parity file per adapter (§16).

This is the single parity file for the backtrader adapter. It runs the trivial
parity strategy (``trivial_long_roundtrip`` — a `fixed_units` of 1 long from the
second bar to the last bar, so exactly one round trip) through the real committed
fixtures of §16 and asserts the run reproduces the **locked** values in
``tests/fixtures/<asset_class>/expected_results.json`` (``engines.backtrader``:
``final_equity``, ``n_trades``, ``trades_hash`` — schema in ``tests/fixtures/README.md``).

The locked values were captured from a real run. The backtrader strategy fills on the
signal bar at that bar's close, matching the nautilus/vectorbt reference (§9.4), and
bar prices are snapped onto the same venue grid before any engine work — so its trade
table is identical to the other two engines' rather than merely comparable within a
tolerance. That equality is asserted in :mod:`test_cross_engine_parity`; the lock is
per-engine and immutable: any change that breaks it is caught here.
"""

import pytest

from tests.parity._parity import (
    PARITY_ASSETS,
    equity_of,
    lock_doc,
    locked_engine,
    run_parity,
    trades_hash,
)


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_backtrader_parity_matches_locked(asset_class: str):
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter

    result = run_parity(BacktraderAdapter, asset_class)
    locked = locked_engine(asset_class, "backtrader")

    assert len(result.trades) == locked["n_trades"]
    assert equity_of(result) == pytest.approx(
        locked["final_equity"],
        rel=lock_doc(asset_class)["tolerance"]["final_equity_rtol"],
    )
    assert trades_hash(result.trades) == locked["trades_hash"]

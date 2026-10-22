"""Vectorbt cross-engine parity tests — one parity file per adapter (§16).

This is the single parity test file for the Vectorbt adapter, mirroring
``test_nautilus_parity.py``. It runs the trivial parity strategy
(``trivial_long_roundtrip`` — a `fixed_units` of 1 long from the second bar to the
last bar, so exactly one round trip) through the real committed fixtures of §16 using
the VectorbtAdapter and asserts the run reproduces the **locked** values in
``tests/fixtures/<asset_class>/expected_results.json`` (``engines.vectorbt``:
``final_equity``, ``n_trades``, ``trades_hash`` — schema in ``tests/fixtures/README.md``).

The locked values were captured from a real, manually-reviewed run — never invented
(dev-guide §4.6). They are immutable: any change that breaks the baseline is caught
here. They live in the shared fixture lock (§16's per-asset × per-engine matrix),
alongside the Nautilus and backtrader blocks, so cross-engine agreement is assertable
rather than merely hoped for — see :mod:`test_cross_engine_parity`.
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
def test_vectorbt_parity_matches_locked(asset_class: str):
    from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter

    result = run_parity(VectorbtAdapter, asset_class)
    locked = locked_engine(asset_class, "vectorbt")

    assert len(result.trades) == locked["n_trades"]
    assert equity_of(result) == pytest.approx(
        locked["final_equity"],
        rel=lock_doc(asset_class)["tolerance"]["final_equity_rtol"],
    )
    assert trades_hash(result.trades) == locked["trades_hash"]

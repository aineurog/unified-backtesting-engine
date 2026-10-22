"""NautilusTrader cross-engine parity tests — one parity file per adapter (§16).

This is the single parity test file for the Nautilus adapter. It runs the trivial
parity strategy (``trivial_long_roundtrip`` — a `fixed_units` of 1 long from the second
bar to the last bar, so exactly one round trip) through the real committed fixtures of
§16 and asserts the run reproduces the **locked** values in
``tests/fixtures/<asset_class>/expected_results.json`` (``engines.nautilus``:
``final_equity``, ``n_trades``, ``trades_hash`` — schema in ``tests/fixtures/README.md``).

The locked values were captured from a real, manually-reviewed run — never invented
(dev-guide §4.6). They are immutable: any change that breaks the baseline is caught
here. Every engine's block is now filled, so cross-engine agreement is asserted in
:mod:`test_cross_engine_parity` rather than deferred.
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
def test_nautilus_parity_matches_locked(asset_class: str):
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter

    result = run_parity(NautilusAdapter, asset_class)
    locked = locked_engine(asset_class, "nautilus")

    assert len(result.trades) == locked["n_trades"]
    assert equity_of(result) == pytest.approx(
        locked["final_equity"],
        rel=lock_doc(asset_class)["tolerance"]["final_equity_rtol"],
    )
    assert trades_hash(result.trades) == locked["trades_hash"]

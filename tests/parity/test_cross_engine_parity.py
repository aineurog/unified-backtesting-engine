"""Cross-engine parity: what the engines must agree on (§16).

§16 requires "a trivial strategy that **must** produce the same result through every
engine". This file is where that requirement is actually asserted. The per-engine files
(``test_nautilus_parity.py`` et al.) each pin their engine to its own locked baseline;
nothing in them would notice if two engines drifted *together*, or if a divergence
appeared that every per-engine lock happily absorbed. This file compares the engines
against each other.

The comparison is exact, not a tolerance. All three adapters fill on the signal bar at
that bar's close and fold through the same core ledger, and the bar prices are snapped
onto the instrument's venue grid before either engine sees them (§4.5). So the same
bars necessarily yield the same trade table — field for field — and the same final
equity in every engine.

Before ``core.snap_prices`` was applied, the engines fed *different numbers* to the
same instrument: vectorbt took the raw float64 bar (``59995.100000000006``) while
nautilus took the venue-grid price (``59995.1``), and the reported fills differed in
the last float bits. This suite is the assertion that catches that class of drift.
"""

import pytest

from tests.parity._parity import (
    ENGINES,
    PARITY_ASSETS,
    equity_of,
    lock_doc,
    run_parity,
    trades_hash,
)


def _run_all(asset_class: str):
    """Run all three engines over one fixture and return ``(vbt, nautilus, backtrader)``."""
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter

    return (
        run_parity(VectorbtAdapter, asset_class),
        run_parity(NautilusAdapter, asset_class),
        run_parity(BacktraderAdapter, asset_class),
    )


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_every_engine_agrees_exactly(asset_class: str):
    """The same bars through three independent engines produce the same result.

    Exact, not approximate: trade count, the full trade table (via its canonical hash)
    and the final equity must all be identical. This is the assertion that fails when
    one adapter prices the bars off a different grid than another, so it is deliberately
    not a tolerance.
    """
    vbt, naut, bt = _run_all(asset_class)

    assert len(vbt.trades) == len(naut.trades) == len(bt.trades)
    assert equity_of(vbt) == equity_of(naut) == equity_of(bt)
    assert trades_hash(vbt.trades) == trades_hash(naut.trades) == trades_hash(bt.trades)


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_locked_blocks_agree(asset_class: str):
    """The committed engine locks encode the same result, so they cannot drift apart.

    The runtime comparison above proves the engines agree *today*; this proves the
    recorded baselines agree too. Without it, two lock files could be edited in
    lockstep into a wrong-but-consistent state that no per-engine test would reject.
    """
    engines = lock_doc(asset_class)["engines"]
    reference = engines[ENGINES[0]]

    for engine in ENGINES[1:]:
        block = engines[engine]
        assert block["n_trades"] == reference["n_trades"], f"{asset_class}/{engine}"
        assert block["final_equity"] == reference["final_equity"], f"{asset_class}/{engine}"
        assert block["trades_hash"] == reference["trades_hash"], f"{asset_class}/{engine}"

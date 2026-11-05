"""Cross-engine parity: what the engines must agree on (§16).

§16 requires "a trivial strategy that **must** produce the same result through every
engine". This file is where that requirement is actually asserted. The per-engine files
(``test_nautilus_parity.py`` et al.) each pin their engine to its own locked baseline;
nothing in them would notice if two engines drifted *together*, or if a divergence
appeared that every per-engine lock happily absorbed. This file compares the engines
against each other.

The comparison is exact, not a tolerance. All three adapters follow the same
``execution_timing`` contract (default ``next_open``: a decision at bar *t* fills at
bar *t+1*'s open, §6.1) and fold through the same core ledger, and the bar prices are
snapped onto the instrument's venue grid before either engine sees them (§4.5). So the
same bars necessarily yield the same trade table — field for field — and the same final
equity in every engine.

Before ``core.snap_prices`` was applied, the engines fed *different numbers* to the
same instrument: vectorbt took the raw float64 bar (``59995.100000000006``) while
nautilus took the venue-grid price (``59995.1``), and the reported fills differed in
the last float bits. This suite is the assertion that catches that class of drift.
"""

import numpy as np
import pandas as pd
import pytest

from tests.parity._parity import (
    ENGINES,
    PARITY_ASSETS,
    equity_of,
    load_instrument,
    lock_doc,
    parity_config,
    run_parity,
    trades_hash,
)
from ube.core.data import MarketData
from ube.core.signals import from_target


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


_GAP_INDEX = pd.to_datetime(
    [
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00",
        "2024-01-01 07:00",
        "2024-01-01 08:00",
    ],
    utc=True,
)


@pytest.fixture(scope="module")
def gapped_bars() -> MarketData:
    """Six hourly bars with a four-hour hole between index 3 and index 4.

    The hole is the point: an adapter that mistakes "the next bar" for "one bar period
    later" lands on a different bar than the others. Cross-engine equality alone cannot
    catch a *uniform* timing shift (all three filling on the wrong bar), so the test
    below also pins the exact fill bar and price.
    """
    open_ = np.array([100.0, 110.0, 125.0, 140.0, 155.0, 170.0])
    close = open_ + 1.0
    return MarketData(
        open=open_,
        high=close + 5.0,
        low=open_ - 5.0,
        close=close,
        volume=np.full(open_.shape[0], 1000.0),
        index=_GAP_INDEX,
    )


def test_gapped_bars_all_engines_fill_on_the_same_next_bar(
    gapped_bars: MarketData,
) -> None:
    """A uniform timing regression must fail even when the engines agree.

    Entry is decided on bar 0 and the exit on bar 2; under the default ``next_open``
    they fill at bar 1's open (110) and bar 3's open (140). Had every engine shifted
    together to ``same_close``, the fills would be bar 0's close (101) and bar 2's
    close (126) and cross-engine equality would still pass — the explicit bar/price
    assertions are what make that regression visible.
    """
    from ube.adapters.backtrader_adapter.adapter import BacktraderAdapter
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter

    instrument = load_instrument("crypto_perp")
    signals = from_target(np.array([1, 1, 0, 0, 0, 0]))
    config = parity_config(instrument)

    results = [
        adapter().run(gapped_bars, signals, config)
        for adapter in (VectorbtAdapter, NautilusAdapter, BacktraderAdapter)
    ]
    ts = gapped_bars.timestamps.as_unit("ns").asi8
    for result in results:
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(110.0)
        assert trade.exit_price == pytest.approx(140.0)
        assert trade.entry_timestamp == int(ts[1])
        assert trade.exit_timestamp == int(ts[3])
    assert results[0].trades == results[1].trades == results[2].trades

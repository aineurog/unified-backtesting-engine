"""§16 fixture integrity — the committed lock files must not drift silently.

This is the one place the fixture *provenance* is checked, independent of any engine:
the parquet bytes must match the recorded content hash, the instrument manifest must
agree with the fixture, and every ``expected_results.json`` must carry the complete
§16 schema — three real engine blocks, none of them a placeholder. A lock file that
still holds zeros is an unfinished lock, and §16's per-asset × per-engine matrix is
not satisfied by it, so it is a failure rather than a skip.
"""

import pytest

from tests.parity._parity import (
    ENGINES,
    LOCK_KEYS,
    PARITY_ASSETS,
    fixture_hash,
    load_fixture,
    lock_doc,
    manifest_info,
)


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_fixture_matches_manifest(asset_class: str):
    """The committed parquet bars match the manifest's recorded row count and hash."""
    info = manifest_info(asset_class)
    md, instrument = load_fixture(asset_class)
    assert md.n_bars == info["row_count"]
    assert fixture_hash(md) == info["content_hash"]
    assert instrument.symbol == info["symbol"]


@pytest.mark.parametrize("asset_class", sorted(PARITY_ASSETS))
def test_lock_file_carries_every_engine(asset_class: str):
    """Every lock file holds the full §16 schema, with no placeholder engine left.

    §16 requires an ``expected_results.json`` per asset class **per engine**. A zeroed
    block means that engine's baseline was never captured, so the matrix is incomplete
    — the run is not "not yet comparable", it is unverified.
    """
    doc = lock_doc(asset_class)
    assert set(doc) >= {
        "asset_class",
        "instrument",
        "strategy",
        "locked_at",
        "tolerance",
        "engines",
    }
    assert doc["asset_class"] == asset_class
    assert doc["strategy"] == "trivial_long_roundtrip"
    assert doc["tolerance"]["final_equity_rtol"] > 0.0
    assert set(doc["engines"]) == set(ENGINES)
    for engine in ENGINES:
        block = doc["engines"][engine]
        assert set(block) == set(LOCK_KEYS), f"{asset_class}/{engine} block schema"
        assert block["final_equity"] != 0.0, f"{asset_class}/{engine} is a placeholder"
        assert block["n_trades"] > 0, f"{asset_class}/{engine} is a placeholder"
        assert block["trades_hash"], f"{asset_class}/{engine} is a placeholder"

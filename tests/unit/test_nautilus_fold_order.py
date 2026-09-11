"""Unit tests for the nautilus adapter fold's fill ordering (§4.6).

The fold builds ``position_change`` events from a running signed net over the
fills report. Nautilus's report is not guaranteed chronological, so a stop fill
reported before its own entry fill would misattribute ``position_after`` to the
wrong timestamps and corrupt the equity curve's marks. ``_fills_chronological``
orders the rows by ``ts_event`` (stable) before that accumulation.
"""

import pandas as pd

from ube.adapters.nautilus_adapter.adapter import (
    _fill_timestamp_ns,
    _fills_chronological,
)


def _fills_report(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.set_index("client_order_id", inplace=True)
    return df


def test_fills_chronological_sorts_by_ts_event():
    # Report-index order is non-chronological: the intra-bar stop SELL at 09:54 is
    # reported before its entry BUY at 09:51.
    fills = _fills_report(
        [
            {
                "client_order_id": "O-2",
                "ts_event": pd.Timestamp("2026-09-11 09:54:00", tz="UTC"),
                "last_qty": 7386.19387,
                "order_side": "SELL",
                "last_px": 1.35058,
            },
            {
                "client_order_id": "O-1",
                "ts_event": pd.Timestamp("2026-09-11 09:51:00", tz="UTC"),
                "last_qty": 7386.19387,
                "order_side": "BUY",
                "last_px": 1.35085,
            },
        ]
    )
    ordered = _fills_chronological(fills)
    assert [cid for cid, _ in ordered] == ["O-1", "O-2"]
    assert _fill_timestamp_ns(ordered[0][1]) <= _fill_timestamp_ns(ordered[1][1])


def test_fills_chronological_is_stable_within_same_ts():
    # Same-timestamp split fills keep the report's relative order: the final running
    # net is order-independent, but the intermediate position_change event is not.
    fills = _fills_report(
        [
            {
                "client_order_id": "O-a",
                "ts_event": pd.Timestamp("2026-09-11 09:56:00", tz="UTC"),
                "last_qty": 7288.3466,
                "order_side": "SELL",
                "last_px": 1.35048,
            },
            {
                "client_order_id": "O-b",
                "ts_event": pd.Timestamp("2026-09-11 09:56:00", tz="UTC"),
                "last_qty": 91.0,
                "order_side": "SELL",
                "last_px": 1.35048,
            },
        ]
    )
    ordered = _fills_chronological(fills)
    assert [cid for cid, _ in ordered] == ["O-a", "O-b"]


def test_fills_chronological_idempotent_when_already_ordered():
    fills = _fills_report(
        [
            {
                "client_order_id": "O-1",
                "ts_event": pd.Timestamp("2026-09-11 09:40:00", tz="UTC"),
                "last_qty": 7403.40408,
                "order_side": "BUY",
                "last_px": 1.35073,
            },
            {
                "client_order_id": "O-2",
                "ts_event": pd.Timestamp("2026-09-11 09:51:00", tz="UTC"),
                "last_qty": 7386.19387,
                "order_side": "BUY",
                "last_px": 1.35085,
            },
            {
                "client_order_id": "O-3",
                "ts_event": pd.Timestamp("2026-09-11 09:54:00", tz="UTC"),
                "last_qty": 7386.19387,
                "order_side": "SELL",
                "last_px": 1.35057983,
            },
        ]
    )
    ordered = _fills_chronological(fills)
    assert [cid for cid, _ in ordered] == ["O-1", "O-2", "O-3"]
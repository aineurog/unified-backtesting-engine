"""Tests for calendar-aware bar indexing and the missing-bar policy (§4.4, §4.7)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.calendar import (
    CALENDAR_ALWAYS_OPEN,
    AlwaysOpenCalendar,
    DataQualityConfig,
    ExchangeCalendar,
    apply_missing_bar_policy,
    detect_gaps,
    resolve_calendar,
)
from ube.core.data import MarketData
from ube.core.errors import (
    CalendarMismatchError,
    ConfigError,
    DataError,
    InvalidInstrumentError,
    MissingBarError,
)
from ube.core.instrument import Instrument


def _bars(timestamps: list[str]) -> MarketData:
    """Minute time bars with monotonically rising OHLC, indexed by the given UTC times."""
    idx = pd.to_datetime(timestamps).tz_localize("UTC")
    n = len(idx)
    base = np.arange(1, n + 1, dtype=float)
    return MarketData(
        open=base,
        high=base + 0.5,
        low=base - 0.5,
        close=base + 0.25,
        volume=np.full(n, 10.0),
        index=idx,
        bar_type="time",
    )


def _event_bars(n: int = 4, bar_type: str = "volume") -> MarketData:
    base = np.arange(1, n + 1, dtype=float)
    return MarketData(
        open=base,
        high=base + 0.5,
        low=base - 0.5,
        close=base + 0.25,
        volume=np.full(n, 10.0),
        index=pd.RangeIndex(n),
        bar_type=bar_type,
    )


# ---------------------------------------------------------------------------
# Calendar resolution (§4.5).
# ---------------------------------------------------------------------------


def test_always_open_resolves_for_none_and_24_7():
    assert isinstance(resolve_calendar(None), AlwaysOpenCalendar)
    assert isinstance(resolve_calendar(CALENDAR_ALWAYS_OPEN), AlwaysOpenCalendar)
    assert isinstance(resolve_calendar("  24/7 "), AlwaysOpenCalendar)


def test_always_open_calendar_is_always_open():
    cal = resolve_calendar("24/7")
    idx = pd.to_datetime(["2024-01-02 03:00", "2024-01-06 23:59"]).tz_localize("UTC")
    assert cal.is_always_open
    assert cal.session_mask(idx).tolist() == [True, True]
    assert cal.session_index(idx).tolist() == [0, 0]


def test_resolve_instrument_calendar_reference():
    ins = Instrument("ES", asset_class="futures", calendar="CMES")
    cal = resolve_calendar(ins)
    assert isinstance(cal, ExchangeCalendar)
    assert cal.name == "CMES"


def test_cme_alias_maps_to_cmes():
    cal = resolve_calendar("CME")
    assert isinstance(cal, ExchangeCalendar)
    assert cal.name == "CMES"


def test_unknown_calendar_name_raises_invalid_instrument_error():
    with pytest.raises(InvalidInstrumentError):
        resolve_calendar("NOT_A_REAL_EXCHANGE")


# ---------------------------------------------------------------------------
# Exchange-calendar session membership (§4.4).
# ---------------------------------------------------------------------------


def test_xnys_session_membership():
    cal = resolve_calendar("XNYS")
    idx = pd.to_datetime(
        [
            "2024-01-08 13:00",  # before open (14:30 UTC)
            "2024-01-08 14:30",  # open
            "2024-01-08 15:00",  # in session
            "2024-01-08 21:00",  # close (inclusive)
            "2024-01-08 21:01",  # after close
            "2024-01-06 15:00",  # Saturday (closed)
        ]
    ).tz_localize("UTC")
    assert cal.session_mask(idx).tolist() == [False, True, True, True, False, False]


# ---------------------------------------------------------------------------
# DataQualityConfig (§4.7).
# ---------------------------------------------------------------------------


def test_data_quality_config_defaults_to_fail():
    assert DataQualityConfig().missing_bar == "fail"


def test_data_quality_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        DataQualityConfig().missing_bar = "skip"  # type: ignore[misc]


def test_data_quality_config_rejects_unknown_policy():
    with pytest.raises(ConfigError):
        DataQualityConfig(missing_bar="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Calendar-expected closures are excluded, not forward-filled (§4.4).
# ---------------------------------------------------------------------------


def test_weekend_closure_is_expected_gap_not_missing():
    cal = resolve_calendar("XNYS")
    md = _bars(
        [
            "2024-01-05 20:58",  # Friday, last minutes of the session
            "2024-01-05 20:59",
            "2024-01-08 14:30",  # Monday open
            "2024-01-08 14:31",
        ]
    )
    report = detect_gaps(md, cal)
    assert not report.out_of_session.any()
    assert not report.unexpected_gap.any()
    assert report.expected_gap.tolist() == [False, True, False]
    # The weekend closure is not treated as a missing bar under any policy.
    assert apply_missing_bar_policy(md, cal, DataQualityConfig("fail")).n_bars == 4
    assert apply_missing_bar_policy(md, cal, DataQualityConfig("forward_fill")).n_bars == 4


# ---------------------------------------------------------------------------
# Genuine unexpected gap -> missing-bar policy (§4.7).
# ---------------------------------------------------------------------------


def test_unexpected_gap_fail_raises_missing_bar_error():
    cal = resolve_calendar("XNYS")
    md = _bars(["2024-01-08 14:30", "2024-01-08 14:31", "2024-01-08 14:33"])
    report = detect_gaps(md, cal)
    assert report.unexpected_gap.tolist() == [False, True]
    assert report.missing_counts.tolist() == [0, 1]
    assert report.has_missing
    with pytest.raises(MissingBarError):
        apply_missing_bar_policy(md, cal, DataQualityConfig("fail"))


def test_unexpected_gap_skip_returns_unchanged():
    cal = resolve_calendar("XNYS")
    md = _bars(["2024-01-08 14:30", "2024-01-08 14:31", "2024-01-08 14:33"])
    out = apply_missing_bar_policy(md, cal, DataQualityConfig("skip"))
    assert out is md
    assert out.n_bars == 3


def test_unexpected_gap_forward_fill_inserts_synthetic_bar():
    cal = resolve_calendar("XNYS")
    md = _bars(["2024-01-08 14:30", "2024-01-08 14:31", "2024-01-08 14:33"])
    out = apply_missing_bar_policy(md, cal, DataQualityConfig("forward_fill"))
    assert out.n_bars == 4
    assert out.index.tolist() == [
        pd.Timestamp("2024-01-08 14:30:00+00:00"),
        pd.Timestamp("2024-01-08 14:31:00+00:00"),
        pd.Timestamp("2024-01-08 14:32:00+00:00"),
        pd.Timestamp("2024-01-08 14:33:00+00:00"),
    ]
    # The synthetic bar carries the previous bar's price, with zero volume.
    assert out.close[2] == out.close[1]
    assert out.open[2] == out.open[1]
    assert out.volume[2] == 0.0


def test_missing_bar_error_is_a_data_error():
    assert issubclass(MissingBarError, DataError)
    assert issubclass(CalendarMismatchError, DataError)


# ---------------------------------------------------------------------------
# CalendarMismatchError — a bar at a closed timestamp (§15).
# ---------------------------------------------------------------------------


def test_bar_at_closed_timestamp_raises_calendar_mismatch():
    cal = resolve_calendar("XNYS")
    # A clearly out-of-session bar: a Saturday timestamp.
    md_sat = _bars(["2024-01-06 15:00", "2024-01-08 14:30"])
    report = detect_gaps(md_sat, cal)
    assert report.out_of_session.tolist() == [True, False]
    with pytest.raises(CalendarMismatchError):
        apply_missing_bar_policy(md_sat, cal, DataQualityConfig("fail"))


# ---------------------------------------------------------------------------
# Event bars bypass calendar logic entirely (§4.3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bar_type", ["volume", "dollar", "tick"])
def test_event_bars_are_a_no_op(bar_type):
    cal = resolve_calendar("XNYS")
    md = _event_bars(5, bar_type=bar_type)
    report = detect_gaps(md, cal)
    assert not report.has_missing
    assert not report.unexpected_gap.any()
    assert not report.out_of_session.any()
    # Event bars are returned unchanged regardless of the policy.
    assert apply_missing_bar_policy(md, cal, DataQualityConfig("fail")) is md
    assert apply_missing_bar_policy(md, cal, DataQualityConfig("forward_fill")) is md

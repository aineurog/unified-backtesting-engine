"""Tests for calendar resolution and session awareness (§4.4, §4.5)."""

import numpy as np
import pandas as pd
import pytest

from ube.core.calendar import (
    CALENDAR_ALWAYS_OPEN,
    AlwaysOpenCalendar,
    ExchangeCalendar,
    resolve_calendar,
    validate_in_session,
)
from ube.core.data import MarketData
from ube.core.errors import CalendarMismatchError, DataError, InvalidInstrumentError
from ube.core.instrument import Instrument


def _bars(timestamps: list[str]) -> MarketData:
    """Minute bars with monotonically rising OHLC, indexed by the given UTC times."""
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
# validate_in_session (§4.4) — the out-of-session check.
# ---------------------------------------------------------------------------


def test_in_session_bars_pass_validation():
    cal = resolve_calendar("XNYS")
    md = _bars(["2024-01-08 14:30", "2024-01-08 14:31", "2024-01-08 15:00"])
    validate_in_session(md, cal)  # must not raise


def test_bar_at_closed_timestamp_raises_calendar_mismatch():
    cal = resolve_calendar("XNYS")
    # A clearly out-of-session bar: a Saturday timestamp.
    md = _bars(["2024-01-06 15:00", "2024-01-08 14:30"])
    with pytest.raises(CalendarMismatchError):
        validate_in_session(md, cal)


def test_always_open_calendar_trivially_passes():
    cal = resolve_calendar("24/7")
    md = _bars(["2024-01-06 15:00", "2024-01-08 14:30"])
    validate_in_session(md, cal)  # must not raise


def test_calendar_mismatch_error_is_a_data_error():
    assert issubclass(CalendarMismatchError, DataError)

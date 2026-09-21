"""Calendar resolution and session awareness (§4.4, §4.5).

This module turns an ``Instrument.calendar`` string reference (item 02) into a
``pandas_market_calendars`` calendar, and uses it to validate that a ``MarketData`` bar
index matches the declared trading calendar.

The calendar says, per timestamp, whether the market is open. The one data-validation
use in Phase 1 is the out-of-session check (:func:`validate_in_session`): a bar at a
timestamp where the calendar says the market is *closed* raises
:class:`~ube.core.errors.CalendarMismatchError` (§15 — the data timestamps contradict
the declared calendar). Session detection (:meth:`TradingCalendar.session_mask` /
:meth:`TradingCalendar.session_index`) is also what Phase-2 funding/rollover scheduling
builds on.

There is deliberately **no missing-bar policy** here: the engine does not try to detect
or repair bars that are absent. Data is taken as the source of truth; a calendar is only
used to reject data that contradicts it (§4.4).

Everything is computed in UTC (§4.4) and vectorized — no per-bar Python loops (§3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal  # type: ignore[import-untyped]

from ube.core.data import MarketData
from ube.core.errors import CalendarMismatchError, InvalidInstrumentError
from ube.core.instrument import Instrument

__all__ = [
    "CALENDAR_ALWAYS_OPEN",
    "TradingCalendar",
    "AlwaysOpenCalendar",
    "ExchangeCalendar",
    "resolve_calendar",
    "validate_in_session",
    "validate_timestamps",
    "is_in_session",
    "next_open_after",
]

#: The calendar reference that means "always open" (crypto — §4.5).
CALENDAR_ALWAYS_OPEN: str = "24/7"

#: Aliases from the names a user is likely to write to the names
#: ``pandas_market_calendars`` actually registers. Extensible.
_CALENDAR_ALIASES: dict[str, str] = {"CME": "CMES"}


def _asi8(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC nanoseconds-since-epoch int64 values for a tz-aware ``DatetimeIndex``.

    Normalized to nanosecond resolution first: pandas 3.x stores tz-aware indexes at
    microsecond resolution by default, and ``asi8`` is the canonical accessor but is
    absent from ``pandas-stubs`` (hence the targeted ignore).
    """
    return cast(np.ndarray, index.as_unit("ns").asi8)  # type: ignore[attr-defined]


class TradingCalendar(ABC):
    """A trading calendar: decides, per timestamp, whether the market is open.

    Concrete calendars are obtained via :func:`resolve_calendar`; do not construct
    these directly. Indexes passed to :meth:`session_mask` / :meth:`session_index`
    must be tz-aware UTC ``DatetimeIndex`` values (as ``MarketData`` guarantees).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_always_open(self) -> bool:
        """True if the market never closes (``"24/7"``)."""
        return False

    @abstractmethod
    def session_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean array: True where the calendar says the market is open."""

    @abstractmethod
    def session_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Integer session ordinal per timestamp; ``-1`` where the market is closed.

        Two timestamps in the same trading session share an ordinal.
        """

    def next_open_after(self, ts: Any) -> pd.Timestamp | None:
        """The next session open (UTC) strictly after ``ts``, or ``None``.

        Used by the paper engines to tell the operator *when* a market that is
        currently closed will reopen (e.g. ``APPL: market closed — next open
        2026-09-21 14:30:00 UTC``). Always-open calendars return ``ts`` itself.
        """
        return None


class AlwaysOpenCalendar(TradingCalendar):
    """A calendar that is always open (crypto ``"24/7"`` — §4.5)."""

    def __init__(self) -> None:
        super().__init__(CALENDAR_ALWAYS_OPEN)

    @property
    def is_always_open(self) -> bool:
        return True

    def session_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.ones(len(index), dtype=bool)

    def session_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        # One logical session spanning all time; every bar is "in session".
        return np.zeros(len(index), dtype=np.int64)

    def next_open_after(self, ts: Any) -> pd.Timestamp | None:
        # Always open — the market is never closed, so it is already open at ``ts``.
        return pd.Timestamp(ts)


class ExchangeCalendar(TradingCalendar):
    """A named ``pandas_market_calendars`` exchange calendar."""

    def __init__(self, name: str, calendar: Any) -> None:
        super().__init__(name)
        self._calendar = calendar

    def _schedule(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # Pad one day on either side so partial sessions at the data edges are covered.
        start = index.min().normalize() - pd.Timedelta(days=1)
        end = index.max().normalize() + pd.Timedelta(days=1)
        return cast(pd.DataFrame, self._calendar.schedule(start, end))

    def session_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.session_index(index) >= 0

    def session_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        schedule = self._schedule(index)
        opens = _asi8(pd.DatetimeIndex(schedule["market_open"]))
        closes = _asi8(pd.DatetimeIndex(schedule["market_close"]))
        n = len(opens)
        if n == 0:
            return np.full(len(index), -1, dtype=np.int64)

        ts = _asi8(index)
        # The session whose open is the last open <= ts.
        pos = np.searchsorted(opens, ts, side="right") - 1
        valid = (pos >= 0) & (pos < n)
        open_candidate = np.where(valid, opens[np.clip(pos, 0, n - 1)], 0)
        close_candidate = np.where(valid, closes[np.clip(pos, 0, n - 1)], 0)
        in_session = valid & (ts >= open_candidate) & (ts <= close_candidate)
        return np.where(in_session, pos, -1).astype(np.int64)

    def next_open_after(self, ts: Any) -> pd.Timestamp | None:
        stamp = pd.Timestamp(ts)
        stamp = (
            stamp.tz_localize("UTC")
            if stamp.tzinfo is None
            else stamp.tz_convert("UTC")
        )
        # Query the schedule across a generous horizon (a week) so weekends and multi-day
        # holiday closures are covered. `_calendar.schedule` needs a range, not an index.
        start = stamp.normalize()
        end = start + pd.Timedelta(days=7)
        try:
            schedule = self._calendar.schedule(start, end)
        except Exception:
            return None
        opens = pd.DatetimeIndex(schedule["market_open"])
        opens = opens.tz_convert("UTC")
        future = opens[opens > stamp]
        if len(future) == 0:
            return None
        return pd.Timestamp(future[0])


def resolve_calendar(reference: str | Instrument | None = None) -> TradingCalendar:
    """Resolve a calendar reference to a :class:`TradingCalendar` (§4.5).

    ``reference`` may be an ``Instrument`` (its ``.calendar`` field is used), a bare
    calendar name string, or ``None``. ``None`` and ``"24/7"`` resolve to the
    always-open calendar; a named exchange calendar (e.g. ``"XNYS"``, ``"CMES"``) is
    looked up in ``pandas_market_calendars``. ``"CME"`` is accepted as an alias for
    ``"CMES"``. An unknown name raises :class:`~ube.core.errors.InvalidInstrumentError`.

    ``None`` means "no calendar was declared", which is treated as always-open — the
    safe reading: no calendar constraints, so no calendar-derived validation. An
    asset-class-to-exchange default table is deliberately *not* invented here (§2/§4.5
    name no exchanges); users trading sessioned markets declare their exchange calendar.
    """
    if isinstance(reference, Instrument):
        reference = reference.calendar
    name = reference if isinstance(reference, str) else None
    if name is None or name.strip() == CALENDAR_ALWAYS_OPEN:
        return AlwaysOpenCalendar()
    name = name.strip()
    canonical = _CALENDAR_ALIASES.get(name, name)
    try:
        calendar = mcal.get_calendar(canonical)
    except Exception as exc:
        raise InvalidInstrumentError(
            f"unknown trading calendar {name!r}; expected {CALENDAR_ALWAYS_OPEN!r} "
            f"or a pandas_market_calendars exchange name"
        ) from exc
    return ExchangeCalendar(canonical, calendar)


def validate_timestamps(ts_index: pd.DatetimeIndex, calendar: TradingCalendar) -> None:
    """Raise :class:`~ube.core.errors.CalendarMismatchError` for out-of-session timestamps.

    A timestamp that falls where the calendar says the market is *closed* means the data
    contradicts the declared calendar — a §15 data error, never silently accepted.
    Always-open calendars (``"24/7"``) trivially pass. This is the calendar's only
    Phase-1 validation: it rejects bars that *should not exist*, and deliberately does
    not try to detect or repair bars that *should exist but do not* (§4.4).

    This is the shared single-timestamp/array validator: the data-driven backtest path
    (:func:`validate_in_session`), the portfolio branch, and the event-driven paper
    engines all run on it, so every engine enforces the *same* declared calendar.

    Args:
        ts_index: A tz-aware UTC ``DatetimeIndex`` of compiled bar/event timestamps.
        calendar: The resolved :class:`TradingCalendar` for the instrument.

    Raises:
        CalendarMismatchError: If any timestamp is outside the declared calendar.
    """
    if calendar.is_always_open:
        return
    session = calendar.session_index(ts_index)
    out_of_session = session < 0
    if out_of_session.any():
        i = int(np.argmax(out_of_session))
        raise CalendarMismatchError(
            f"bar {i} at {ts_index[i]} is outside the declared trading calendar "
            f"{calendar.name!r}"
        )


def is_in_session(ts: Any, calendar: TradingCalendar) -> bool:
    """Whether a single timestamp falls inside the declared calendar.

    The scalar form of :func:`validate_timestamps` for event/bar-driven engines (paper
    backends) that decide per-bar whether to act. An always-open calendar (``"24/7"``)
    short-circuits to ``True``. The timestamp is coerced to the tz-aware UTC grid the
    calendars require (naive timestamps are assumed UTC, which matches ``MarketData``).

    Args:
        ts: A single timestamp (``pd.Timestamp``-coercible, naive or tz-aware).
        calendar: The resolved :class:`TradingCalendar` for the instrument.

    Returns:
        ``True`` when the calendar says the market is open at ``ts``.
    """
    if calendar.is_always_open:
        return True
    stamp = pd.Timestamp(ts)
    stamp = (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )
    index = pd.DatetimeIndex([stamp.as_unit("ns")])
    return bool((calendar.session_index(index) >= 0)[0])


def next_open_after(
    ts: Any,
    calendar: TradingCalendar,
) -> pd.Timestamp | None:
    """The next session open (UTC) strictly after a timestamp, or ``None``.

    The operator-facing form of :meth:`TradingCalendar.next_open_after` — used by the
    paper engines to report *when* a currently-closed market reopens (e.g. the AAPL
    weekend/off-hours skip warning). Always-open calendars return ``ts`` itself; an
    exchange calendar returns the first ``market_open`` at or after the next scheduled
    session (a one-week horizon covers weekends and multi-day holiday closures).

    Args:
        ts: A single timestamp (``pd.Timestamp``-coercible, naive or tz-aware).
        calendar: The resolved :class:`TradingCalendar` for the instrument.

    Returns:
        The UTC open of the next session strictly after ``ts``, or ``None`` when the
        schedule cannot be determined (e.g. an exchange with no future sessions).
    """
    return calendar.next_open_after(ts)


def validate_in_session(market_data: MarketData, calendar: TradingCalendar) -> None:
    """Raise :class:`~ube.core.errors.CalendarMismatchError` for out-of-session bars.

    A bar whose timestamp falls where the calendar says the market is *closed* means the
    data contradicts the declared calendar — a §15 data error, never silently accepted.
    Always-open calendars (``"24/7"``) trivially pass. This is the calendar's only
    Phase-1 validation: it rejects bars that *should not exist*, and deliberately does
    not try to detect or repair bars that *should exist but do not* (§4.4).

    Thin wrapper over :func:`validate_timestamps` so the whole-dataset path shares the
    single validator implementation.

    Args:
        market_data: The canonical bar container.
        calendar: The resolved :class:`TradingCalendar` for the instrument.

    Raises:
        CalendarMismatchError: If any bar timestamp is outside the declared calendar.
    """
    validate_timestamps(market_data.timestamps, calendar)

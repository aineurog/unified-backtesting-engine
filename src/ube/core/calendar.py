"""Calendar-aware bar indexing and the missing-bar policy (§4.4, §4.7).

This module turns an ``Instrument.calendar`` string reference (item 02) into a
``pandas_market_calendars`` calendar, and uses it to validate the bar index of a
time-bar ``MarketData``:

- **Calendar-expected closures** (weekends, holidays, overnight) are excluded from the
  bar sequence entirely and never forward-filled — a 14-bar ATR means "the last 14 bars
  that actually existed" (§4.4).
- **Genuine unexpected gaps** — a missing bar *within* an open session — are handled by
  the missing-bar policy in :class:`DataQualityConfig` (§4.7): ``"fail"`` raises
  :class:`~ube.core.errors.MissingBarError`, ``"forward_fill"`` inserts a synthetic bar
  carrying the previous bar's prices, ``"skip"`` leaves the gap as-is.

Event bars (``volume``/``dollar``/``tick``) have no session concept (§4.3), so all
calendar logic is a no-op for them.

Everything is computed in UTC (§4.4) and vectorized — no per-bar Python loops (§3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal  # type: ignore[import-untyped]

from ube.core.data import MarketData
from ube.core.errors import (
    CalendarMismatchError,
    ConfigError,
    InvalidInstrumentError,
    MissingBarError,
)
from ube.core.instrument import Instrument

__all__ = [
    "CALENDAR_ALWAYS_OPEN",
    "MISSING_BAR_POLICIES",
    "MissingBarPolicy",
    "DataQualityConfig",
    "TradingCalendar",
    "AlwaysOpenCalendar",
    "ExchangeCalendar",
    "resolve_calendar",
    "GapReport",
    "detect_gaps",
    "apply_missing_bar_policy",
]

#: The calendar reference that means "always open" (crypto — §4.5).
CALENDAR_ALWAYS_OPEN: str = "24/7"

#: The valid missing-bar policies (§4.7).
MISSING_BAR_POLICIES: tuple[str, ...] = ("fail", "forward_fill", "skip")

MissingBarPolicy = Literal["fail", "forward_fill", "skip"]

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


@dataclass(frozen=True)
class DataQualityConfig:
    """Missing-bar / data-quality policy for time bars (§4.7).

    This is the single source of truth for the missing-bar policy; the later
    ``BacktestConfig`` (item 08) imports it here rather than re-declaring it.

    Attributes:
        missing_bar: How to handle a genuine unexpected gap (a bar missing where the
            calendar says one should exist). ``"fail"`` raises ``MissingBarError``;
            ``"forward_fill"`` inserts a bar carrying the previous bar's prices;
            ``"skip"`` leaves the gap. Irrelevant for event bars (§4.3).
    """

    missing_bar: MissingBarPolicy = "fail"

    def __post_init__(self) -> None:
        if self.missing_bar not in MISSING_BAR_POLICIES:
            raise ConfigError(
                f"missing_bar={self.missing_bar!r} is not one of {MISSING_BAR_POLICIES}"
            )


#: The default data-quality policy (``missing_bar="fail"``). A module-level singleton
#: is used as a default argument so the immutable default is shared, not re-created per
#: call.
_DEFAULT_DATA_QUALITY_CONFIG = DataQualityConfig()


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

        Two timestamps in the same trading session share an ordinal, which is what lets
        gap detection distinguish an intra-session gap from a session boundary.
        """


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
        # One logical session spanning all time; every pair of bars is "same session",
        # so there are no calendar-expected closures.
        return np.zeros(len(index), dtype=np.int64)


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


def resolve_calendar(reference: str | Instrument | None = None) -> TradingCalendar:
    """Resolve a calendar reference to a :class:`TradingCalendar` (§4.5).

    ``reference`` may be an ``Instrument`` (its ``.calendar`` field is used), a bare
    calendar name string, or ``None``. ``None`` and ``"24/7"`` resolve to the
    always-open calendar; a named exchange calendar (e.g. ``"XNYS"``, ``"CMES"``) is
    looked up in ``pandas_market_calendars``. ``"CME"`` is accepted as an alias for
    ``"CMES"``. An unknown name raises :class:`~ube.core.errors.InvalidInstrumentError`.

    ``None`` means "no calendar was declared", which is treated as always-open — the
    safe reading: no calendar constraints, so no calendar-derived gap detection. An
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


@dataclass(frozen=True)
class GapReport:
    """The result of :func:`detect_gaps`: a classification of a time-bar index.

    Attributes:
        in_session: bool[n] — the bar timestamp is inside a trading session.
        out_of_session: bool[n] — the bar timestamp is outside every session (a bar
            where the calendar says the market is closed).
        expected_gap: bool[n-1] — the gap between consecutive bars is a calendar-expected
            closure (a session boundary was crossed), not a missing bar.
        unexpected_gap: bool[n-1] — the gap between consecutive bars is a genuine missing
            bar (or bars) within an open session.
        missing_counts: int64[n-1] — the number of missing bars per ``unexpected_gap``
            (0 everywhere else).
        period_ns: int | None — the inferred bar period in nanoseconds (``None`` when it
            cannot be determined, e.g. fewer than two bars).
    """

    in_session: np.ndarray
    out_of_session: np.ndarray
    expected_gap: np.ndarray
    unexpected_gap: np.ndarray
    missing_counts: np.ndarray
    period_ns: int | None

    @property
    def has_missing(self) -> bool:
        """True if any genuine unexpected gap was detected."""
        return bool(self.unexpected_gap.any())


def _infer_period(ts: pd.DatetimeIndex) -> int | None:
    """Infer the bar period (ns) as the minimum positive inter-bar spacing.

    Time bars are regular (§4.3), so the finest observed spacing is the bar period; a
    wider same-session spacing means a bar (or bars) is missing.
    """
    if len(ts) < 2:
        return None
    diffs = _asi8(ts)[1:] - _asi8(ts)[:-1]
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None
    return int(positive.min())


def detect_gaps(market_data: MarketData, calendar: TradingCalendar) -> GapReport:
    """Classify the bar index of ``market_data`` against ``calendar`` (§4.4).

    Time bars only — for event bars this returns a no-op report with no gaps. Never
    raises: it is the analysis half of calendar-aware indexing; enforcement (raising
    ``MissingBarError`` / ``CalendarMismatchError``, or repairing) lives in
    :func:`apply_missing_bar_policy`.
    """
    n = market_data.n_bars
    empty_bool = np.zeros(0, dtype=bool)
    empty_int = np.zeros(0, dtype=np.int64)
    if not market_data.is_time_bars:
        return GapReport(
            in_session=np.ones(n, dtype=bool),
            out_of_session=np.zeros(n, dtype=bool),
            expected_gap=empty_bool,
            unexpected_gap=empty_bool,
            missing_counts=empty_int,
            period_ns=None,
        )

    ts = cast(pd.DatetimeIndex, market_data.timestamps)
    session = calendar.session_index(ts)
    in_session = session >= 0
    out_of_session = ~in_session

    if n < 2:
        return GapReport(
            in_session=in_session,
            out_of_session=out_of_session,
            expected_gap=empty_bool,
            unexpected_gap=empty_bool,
            missing_counts=empty_int,
            period_ns=_infer_period(ts),
        )

    period = _infer_period(ts)
    both_in_session = in_session[:-1] & in_session[1:]
    same_session = session[:-1] == session[1:]

    if period is None:
        missing = np.zeros(n - 1, dtype=np.int64)
        unexpected = np.zeros(n - 1, dtype=bool)
    else:
        diffs = _asi8(ts)[1:] - _asi8(ts)[:-1]
        missing = np.maximum(diffs // period - 1, 0)
        unexpected = both_in_session & same_session & (missing >= 1)

    expected = both_in_session & ~same_session
    return GapReport(
        in_session=in_session,
        out_of_session=out_of_session,
        expected_gap=expected,
        unexpected_gap=unexpected,
        missing_counts=np.where(unexpected, missing, 0),
        period_ns=period,
    )


def _runs_arange(counts: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(1, m + 1)`` for each run length ``m`` in ``counts``.

    Vectorized: e.g. ``_runs_arange([3, 2]) == [1, 2, 3, 1, 2]``.
    """
    c = np.asarray(counts, dtype=np.int64)
    total = int(c.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(c)
    starts = ends - c
    idx = np.arange(total, dtype=np.int64)
    run_id = np.searchsorted(ends, idx, side="right")
    return idx - starts[run_id] + 1


def _forward_fill(market_data: MarketData, counts: np.ndarray, period: int) -> MarketData:
    """Insert synthetic bars carrying the previous bar's OHLC for each missing bar.

    ``counts`` is the per-gap missing-bar count (length ``n - 1``); ``period`` is the
    bar period in nanoseconds. Fully vectorized via ``np.repeat`` + a position mask.
    """
    ts = _asi8(cast(pd.DatetimeIndex, market_data.timestamps))
    n = len(ts)
    total = int(counts.sum())

    repeats = np.ones(n, dtype=np.int64)
    repeats[:-1] += counts

    filled_open = np.repeat(market_data.open, repeats)
    filled_high = np.repeat(market_data.high, repeats)
    filled_low = np.repeat(market_data.low, repeats)
    filled_close = np.repeat(market_data.close, repeats)
    filled_volume = np.repeat(market_data.volume, repeats)

    out = np.empty(n + total, dtype=np.int64)
    if total == 0:
        out[:] = ts
    else:
        # Original bar i lands at position i + sum(counts[:i]); the synthetic bars fill
        # the slots between it and the next original bar.
        prefix = np.concatenate(([0], np.cumsum(counts)))
        original_pos = np.arange(n, dtype=np.int64) + prefix
        inserted_mask = np.ones(n + total, dtype=bool)
        inserted_mask[original_pos] = False

        gap_positions = np.nonzero(counts)[0]
        starts = ts[gap_positions]
        runs = counts[gap_positions]
        missing_ts = np.repeat(starts, runs) + _runs_arange(runs) * period

        out[original_pos] = ts
        out[inserted_mask] = missing_ts
        filled_volume[inserted_mask] = 0.0

    index = pd.DatetimeIndex(pd.to_datetime(out, unit="ns", utc=True))
    return MarketData(
        open=filled_open,
        high=filled_high,
        low=filled_low,
        close=filled_close,
        volume=filled_volume,
        index=index,
        bar_type="time",
    )


def apply_missing_bar_policy(
    market_data: MarketData,
    calendar: TradingCalendar,
    config: DataQualityConfig = _DEFAULT_DATA_QUALITY_CONFIG,
) -> MarketData:
    """Apply the missing-bar policy to ``market_data`` (§4.4, §4.7).

    Event bars are returned unchanged (calendar logic is a no-op — §4.3). For time bars:

    - A bar at a closed timestamp always raises :class:`~ube.core.errors.CalendarMismatchError`
      (the data timestamps do not match the declared calendar — §15).
    - A genuine unexpected gap is handled per ``config.missing_bar``:
      ``"fail"`` raises :class:`~ube.core.errors.MissingBarError`; ``"forward_fill"``
      returns a repaired ``MarketData`` with synthetic bars; ``"skip"`` returns the data
      unchanged.
    - Calendar-expected closures are never filled or errored on.
    """
    if not market_data.is_time_bars:
        return market_data

    report = detect_gaps(market_data, calendar)
    ts = cast(pd.DatetimeIndex, market_data.timestamps)

    if report.out_of_session.any():
        i = int(np.argmax(report.out_of_session))
        raise CalendarMismatchError(
            f"bar {i} at {ts[i]} is outside the declared trading calendar "
            f"{calendar.name!r}"
        )

    if not report.unexpected_gap.any():
        return market_data

    if config.missing_bar == "skip":
        return market_data

    period = report.period_ns
    if period is None:
        # Defensive: an unexpected gap implies a period was inferred, but guard anyway.
        raise MissingBarError(
            f"missing bar within an open session of calendar {calendar.name!r}"
        )

    if config.missing_bar == "fail":
        first_gap = int(np.argmax(report.unexpected_gap))
        first_missing = ts[first_gap] + pd.Timedelta(period, unit="ns")
        raise MissingBarError(
            f"missing bar at {first_missing} within an open session of calendar "
            f"{calendar.name!r} (missing_bar={config.missing_bar!r})"
        )

    return _forward_fill(market_data, report.missing_counts, period)

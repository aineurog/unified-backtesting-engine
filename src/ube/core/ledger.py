"""Append-only event ledger — the single source of truth per run (§4.6).

The ledger is the canonical record of *what happened* during a backtest: it is the only
place a run's events are stored, and every downstream artifact (``trades``,
``positions``, ``equity_curve``) is a *derived view* computed from it once — never a
separately-stored aggregate that could drift (§4.6).

Design summary
--------------

* **Sealed event schema.** A single frozen :class:`LedgerEvent` discriminated by an
  :class:`EventType` enum. The enum closes the set of event types (§4.6's list), so no
  ad-hoc event kind can be invented at call time; type-specific payload fields are
  frozen dataclass fields, validated per event type at construction (fail fast, §3
  principle 6). Every event carries a ``timestamp`` (a bar boundary) and an
  ``instrument_id`` so portfolio backtests can filter by instrument (§4.6).
* **``timestamp`` representation.** Stored as ``int``: nanoseconds since the Unix epoch
  (UTC) for time bars, or the integer bar index for event bars. This is the one
  representation that makes the §4.6 "union of timestamps + forward-fill" derivation
  uniform for both bar models. Time-bar ``DatetimeIndex`` inputs are normalized to
  nanosecond resolution via ``.as_unit("ns")`` before int64 math (pandas 3.x stores
  tz-aware indexes at microsecond resolution — see contract-decisions item 03).
* **Append-only container.** :class:`EventLedger` only appends; entries are frozen and
  the snapshot view (``.events``) is an immutable tuple, so already-appended entries
  cannot be mutated or replaced (§3 principle 5, §4.6).
* **Derived views are pure functions**, not attributes (§4.6 "cached views ... exposed
  as convenience attributes on ``BacktestResult``" — item 08 caches them). ``trades`` is
  an inherently sequential fold (round-trip semantics); ``positions`` and the
  mark-to-market equity derivation are vectorized ``searchsorted``/``cumsum``
  operations — no per-bar Python loops (§3 principle 1).
* **Multi-currency normalization (§4.6, §4.8).** The aggregation takes an explicit
  ``base_currency``. Each instrument's mark value is in its ``Instrument.
  settlement_currency`` and is converted to ``base_currency`` via caller-supplied
  historical FX rates (``fx_rates``, a ``Mapping`` keyed by currency pair). There is no
  FX data model yet; a missing rate or an un-normalizable currency raises the §15
  :class:`~ube.core.errors.FXRateUnavailableError` — never a silent 1:1 assumption.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, FXRateUnavailableError
from ube.core.instrument import Instrument

__all__ = [
    "EventType",
    "EVENT_TYPES",
    "LedgerEvent",
    "EventLedger",
    "FXSeries",
    "Trade",
    "Positions",
    "EquityCurve",
    "trades",
    "positions",
    "equity_curve",
    "equity_curve_by_instrument",
]


class EventType(StrEnum):
    """The sealed set of ledger event types (§4.6).

    Membership in this enum is the *only* way to name an event type; the set is closed
    by construction. Values are the stable string names adapters emit and consumers
    filter on.
    """

    SIGNAL_EVALUATED = "signal_evaluated"
    ORDER_SUBMITTED = "order_submitted"
    FILL = "fill"
    FUNDING_PAYMENT = "funding_payment"
    FUTURES_ROLLOVER = "futures_rollover"
    COMMISSION = "commission"
    CASH_MOVEMENT = "cash_movement"
    POSITION_CHANGE = "position_change"


#: All event types, in a stable order for iteration/testing.
EVENT_TYPES: tuple[EventType, ...] = tuple(EventType)

#: Float tolerance for "position is flat" checks in the trade fold.
_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Event-schema validation helpers (module-private, fail fast on malformed events).
# ---------------------------------------------------------------------------


def _require_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataShapeError(
            f"ledger event field {name!r} must be a non-empty string; got {value!r}"
        )


def _require_side(value: object) -> None:
    if isinstance(value, bool) or value not in (1, -1):
        raise DataShapeError(
            f"ledger event field 'side' must be +1 (long) or -1 (short); got {value!r}"
        )


def _require_quantity(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataShapeError(
            f"ledger event field 'quantity' must be a number; got {value!r}"
        )
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise DataShapeError(
            f"ledger event field 'quantity' must be finite and positive; got {value!r}"
        )


def _require_price(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataShapeError(f"ledger event field 'price' must be a number; got {value!r}")
    if not math.isfinite(value) or float(value) <= 0.0:
        raise DataShapeError(
            f"ledger event field 'price' must be finite and positive; got {value!r}"
        )


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataShapeError(
            f"ledger event field {name!r} must be a number; got {value!r}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise DataShapeError(
            f"ledger event field {name!r} must be finite; got {value!r}"
        )
    return v


def _require_nonneg(value: object, name: str) -> None:
    if _require_finite(value, name) < 0.0:
        raise DataShapeError(
            f"ledger event field {name!r} must be non-negative; got {value!r}"
        )


# ---------------------------------------------------------------------------
# Event schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable ledger entry, discriminated by :attr:`event_type` (§4.6).

    Every event carries :attr:`timestamp` (bar boundary: int64 ns since epoch for time
    bars, integer bar index for event bars) and :attr:`instrument_id` (§4.6 — every
    ledger entry is tagged by instrument). The remaining fields are type-specific and
    default to ``None``/``False``; each event type requires its own subset (validated
    here, raising :class:`~ube.core.errors.DataShapeError` on a malformed entry):

    ``signal_evaluated``
        ``action`` — one of ``"long_entry"``, ``"long_exit"``, ``"short_entry"``,
        ``"short_exit"``, or ``"hold"`` (§6.1).

    ``order_submitted``
        ``order_id`` (non-empty), ``side`` (±1), ``quantity`` (positive magnitude).

    ``fill``
        ``side`` (±1), ``quantity`` (positive magnitude — direction is ``side``),
        ``price`` (> 0), optional ``notional`` (≥ 0), and ``gap_fill`` — the §4.6 flag
        marking a fill that jumped the requested level.

    ``funding_payment``
        ``amount`` (signed — negative funding is *received*), ``currency``.

    ``futures_rollover``
        ``rollover_from`` / ``rollover_to`` (contract symbols), optional ``price``.

    ``commission``
        ``amount`` (≥ 0), ``currency``.

    ``cash_movement``
        ``amount`` (signed — inflow positive, outflow negative), ``currency``.

    ``position_change``
        ``position_after`` (signed resulting position quantity); ``side`` is the sign of
        ``position_after``.

    Attributes:
        event_type: The discriminated event type.
        timestamp: Bar-boundary timestamp (int64 ns since epoch UTC, or bar index).
        instrument_id: Instrument the event belongs to (§4.6 tagging).
        action: ``signal_evaluated`` payload.
        order_id: ``order_submitted`` payload (also the link for ``fill``).
        side: Direction, ``+1`` long / ``-1`` short / ``0`` flat.
        quantity: Positive units (direction is :attr:`side`) (fill / order).
        price: Fill or order price (> 0).
        notional: Fill notional (≥ 0), optional.
        gap_fill: ``fill`` flag (§4.6).
        amount: Signed monetary amount (funding / cash) or ≥ 0 (commission).
        currency: Currency of :attr:`amount`.
        rollover_from: Expiring futures contract symbol.
        rollover_to: Incoming futures contract symbol.
        position_after: Resulting signed position quantity.
    """

    event_type: EventType
    timestamp: int
    instrument_id: str

    action: str | None = None
    order_id: str | None = None
    side: int | None = None
    quantity: float | None = None
    price: float | None = None
    notional: float | None = None
    gap_fill: bool = False
    amount: float | None = None
    currency: str | None = None
    rollover_from: str | None = None
    rollover_to: str | None = None
    position_after: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, np.integer)):
            raise DataShapeError(
                f"ledger event timestamp must be an integer; got {self.timestamp!r}"
            )
        object.__setattr__(self, "timestamp", int(self.timestamp))
        _require_str(self.instrument_id, "instrument_id")

        t = self.event_type
        if t is EventType.SIGNAL_EVALUATED:
            _require_str(self.action, "action")
        elif t is EventType.ORDER_SUBMITTED:
            _require_str(self.order_id, "order_id")
            _require_side(self.side)
            _require_quantity(self.quantity)
        elif t is EventType.FILL:
            _require_side(self.side)
            _require_quantity(self.quantity)
            _require_price(self.price)
            if self.notional is not None:
                _require_nonneg(self.notional, "notional")
        elif t is EventType.FUNDING_PAYMENT:
            _require_finite(self.amount, "amount")
            _require_str(self.currency, "currency")
        elif t is EventType.FUTURES_ROLLOVER:
            _require_str(self.rollover_from, "rollover_from")
            _require_str(self.rollover_to, "rollover_to")
            if self.price is not None:
                _require_price(self.price)
        elif t is EventType.COMMISSION:
            _require_nonneg(self.amount, "amount")
            _require_str(self.currency, "currency")
        elif t is EventType.CASH_MOVEMENT:
            _require_finite(self.amount, "amount")
            _require_str(self.currency, "currency")
        elif t is EventType.POSITION_CHANGE:
            if self.position_after is None:
                raise DataShapeError("position_change event requires position_after")
            _require_finite(self.position_after, "position_after")


# ---------------------------------------------------------------------------
# Append-only container.
# ---------------------------------------------------------------------------


class EventLedger:
    """Append-only container of :class:`LedgerEvent` entries (§4.6, §5 principle 5).

    The ledger is the one deliberately-mutable object per run — but only by appending.
    Entries are frozen :class:`LedgerEvent` instances and the snapshot returned by
    :attr:`events` is an immutable tuple, so already-appended entries cannot be mutated
    or replaced. Portfolio backtests filter by instrument via :meth:`by_instrument`.
    """

    def __init__(self, events: Iterable[LedgerEvent] = ()) -> None:
        self._events: list[LedgerEvent] = []
        for event in events:
            self.append(event)

    def append(self, event: LedgerEvent) -> None:
        """Append ``event`` to the end of the ledger (the only mutation allowed)."""
        if not isinstance(event, LedgerEvent):
            raise DataShapeError(
                f"EventLedger.append expects a LedgerEvent; got {type(event).__name__}"
            )
        self._events.append(event)

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """An immutable snapshot of the appended entries, in append order."""
        return tuple(self._events)

    def by_instrument(self, instrument_id: str) -> tuple[LedgerEvent, ...]:
        """The entries tagged with ``instrument_id``, in append order (§4.6)."""
        return tuple(e for e in self._events if e.instrument_id == instrument_id)

    def instrument_ids(self) -> tuple[str, ...]:
        """The distinct ``instrument_id`` values present, in first-appearance order."""
        seen: dict[str, None] = {}
        for e in self._events:
            seen.setdefault(e.instrument_id, None)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[LedgerEvent]:
        return iter(self._events)

    def __getitem__(self, index: int) -> LedgerEvent:
        return self._events[index]

    def __bool__(self) -> bool:
        return bool(self._events)


# ---------------------------------------------------------------------------
# FX series container (no FX data model yet — §4.6 caller-supplied rates).
# ---------------------------------------------------------------------------


def _coerce_int1d(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DataShapeError(f"{name} must be 1-D; got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        raise DataShapeError(f"{name} must be integer; got dtype {arr.dtype}")
    return arr.astype(np.int64, copy=True)


def _coerce_float1d(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DataShapeError(f"{name} must be 1-D; got shape {arr.shape}")
    try:
        return arr.astype(np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise DataShapeError(f"{name} must be numeric; got dtype {arr.dtype}") from exc


@dataclass(frozen=True)
class FXSeries:
    """A single historical FX rate curve (the "curve-like" value of ``fx_rates``).

    Because there is no FX data model yet (§4.6), the equity derivation accepts
    caller-supplied rates as :class:`FXSeries` — a frozen pair of an integer nanosecond
    index and a float64 rate array. A rate value is *units of base currency per one unit
    of the quoted currency* (see :func:`equity_curve`).

    Attributes:
        index: int64[n] timestamps (ns since epoch UTC), strictly ascending.
        rate: float64[n] exchange rates, aligned to ``index``.
    """

    index: np.ndarray
    rate: np.ndarray

    def __post_init__(self) -> None:
        index = _coerce_int1d(self.index, "index")
        rate = _coerce_float1d(self.rate, "rate")
        if index.shape[0] != rate.shape[0]:
            raise DataShapeError(
                f"FXSeries index and rate must have the same length; got "
                f"{index.shape[0]} vs {rate.shape[0]}"
            )
        if index.shape[0] > 1 and not (np.diff(index) > 0).all():
            raise DataShapeError("FXSeries index must be strictly ascending")
        if rate.shape[0] and np.isnan(rate).any():
            raise DataShapeError("FXSeries rate must not contain NaN")
        index.setflags(write=False)
        rate.setflags(write=False)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "rate", rate)

    @classmethod
    def from_series(cls, series: pd.Series) -> FXSeries:
        """Build an :class:`FXSeries` from a pandas ``Series`` (index + values)."""
        if not isinstance(series, pd.Series):
            raise DataShapeError("FXSeries.from_series expects a pandas Series")
        idx = series.index
        if isinstance(idx, pd.DatetimeIndex):
            index = _asi8(idx)
        elif idx.inferred_type == "integer":
            index = idx.to_numpy(dtype=np.int64)
        else:
            raise DataShapeError(
                "FXSeries index must be a DatetimeIndex or an integer index"
            )
        return cls(index=index, rate=series.to_numpy(dtype=np.float64))


# ---------------------------------------------------------------------------
# Derived-view containers (frozen; arrays copied + read-only).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """A completed per-instrument round-trip trade, derived from the ledger (§4.6).

    Attributes:
        instrument_id: Instrument the trade belongs to.
        side: ``+1`` long / ``-1`` short.
        quantity: Absolute units traded (the size of each leg).
        entry_timestamp: Bar boundary of the first fill.
        exit_timestamp: Bar boundary of the closing fill.
        entry_price: Volume-weighted average entry price.
        exit_price: Volume-weighted average exit price.
        gross_pnl: ``-(entry_notional + exit_notional)`` in the settlement currency.
        commission: Total commissions allocated to the trade (≥ 0).
        funding: Net funding allocated to the trade (signed; negative = received).
        net_pnl: ``gross_pnl - commission - funding``.
    """

    instrument_id: str
    side: int
    quantity: float
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    funding: float
    net_pnl: float


@dataclass(frozen=True)
class Positions:
    """A position-over-time step series for one instrument, derived from the ledger.

    ``position_change`` events yield the *resulting* signed position at each bar
    boundary where it changed (§4.6). The series is a step function: the position is
    flat (``0``) before the first change and holds between changes.

    Attributes:
        timestamps: int64[n] change timestamps, strictly ascending.
        position: float64[n] signed resulting position after each change.
    """

    timestamps: np.ndarray
    position: np.ndarray

    def __post_init__(self) -> None:
        timestamps = _coerce_int1d(self.timestamps, "timestamps")
        position = _coerce_float1d(self.position, "position")
        if timestamps.shape[0] != position.shape[0]:
            raise DataShapeError(
                f"Positions timestamps and position must have the same length; got "
                f"{timestamps.shape[0]} vs {position.shape[0]}"
            )
        if timestamps.shape[0] > 1 and not (np.diff(timestamps) > 0).all():
            raise DataShapeError("Positions timestamps must be strictly ascending")
        timestamps.setflags(write=False)
        position.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class EquityCurve:
    """A derived equity curve over a bar grid (§4.6).

    Attributes:
        index: The grid — a tz-aware UTC ``DatetimeIndex`` (time bars) or ``RangeIndex``
            (event bars).
        equity: float64[n] values in ``base_currency``, aligned to ``index``.
    """

    index: pd.Index
    equity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.index, pd.Index):
            raise DataShapeError("EquityCurve.index must be a pandas Index")
        equity = _coerce_float1d(self.equity, "equity")
        if len(self.index) != equity.shape[0]:
            raise DataShapeError(
                f"EquityCurve index and equity must have the same length; got "
                f"{len(self.index)} vs {equity.shape[0]}"
            )
        equity.setflags(write=False)
        object.__setattr__(self, "equity", equity)


# ---------------------------------------------------------------------------
# Trade fold (inherently sequential — round-trip semantics are path-dependent).
# ---------------------------------------------------------------------------


@dataclass
class _MutableTrade:
    """Mutable accumulator for one round trip during the sequential fold."""

    instrument_id: str
    side: int
    entry_timestamp: int
    entry_units: float = 0.0
    entry_notional: float = 0.0
    exit_units: float = 0.0
    exit_notional: float = 0.0
    exit_timestamp: int = 0
    commission: float = 0.0
    funding: float = 0.0


@dataclass
class _FoldState:
    """Per-instrument fold state: net position and the current (open/trailing) trade."""

    position: dict[str, float]
    current: dict[str, _MutableTrade]


def _freeze_trade(mt: _MutableTrade) -> Trade:
    """Convert a completed round-trip accumulator into an immutable :class:`Trade`."""
    entry_price = abs(mt.entry_notional) / mt.entry_units if mt.entry_units > 0.0 else 0.0
    exit_price = abs(mt.exit_notional) / mt.exit_units if mt.exit_units > 0.0 else 0.0
    gross_pnl = -(mt.entry_notional + mt.exit_notional)
    return Trade(
        instrument_id=mt.instrument_id,
        side=mt.side,
        quantity=mt.exit_units,
        entry_timestamp=mt.entry_timestamp,
        exit_timestamp=mt.exit_timestamp,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross_pnl,
        commission=mt.commission,
        funding=mt.funding,
        net_pnl=gross_pnl - mt.commission - mt.funding,
    )


def _process_fill(state: _FoldState, out: list[Trade], event: LedgerEvent) -> None:
    """Fold one ``fill`` event into round-trip trades (sequential, per instrument)."""
    iid = event.instrument_id
    q = float(cast(int, event.side)) * float(cast(float, event.quantity))
    p = float(cast(float, event.price))
    pos = state.position.get(iid, 0.0)

    remaining = q
    while abs(remaining) > _EPS:
        if abs(pos) < _EPS:
            prev = state.current.get(iid)
            if prev is not None and (prev.entry_units > 0.0 or prev.exit_units > 0.0):
                out.append(_freeze_trade(prev))
            mt = _MutableTrade(iid, 1 if remaining > 0.0 else -1, event.timestamp)
            state.current[iid] = mt
        else:
            mt = state.current[iid]
            if mt is None:
                raise DataShapeError(
                    f"ledger invariant broken: open position with no trade for {iid!r}"
                )

        if (remaining > 0.0) == (mt.side > 0):
            # Same direction as the open round trip: add to entry.
            mt.entry_units += abs(remaining)
            mt.entry_notional += remaining * p
            pos += remaining
            remaining = 0.0
        else:
            # Opposite direction: close up to the open size, then (if a flip) reopen.
            open_units = mt.entry_units - mt.exit_units
            close_units = min(abs(remaining), open_units)
            close_q = -float(mt.side) * close_units
            mt.exit_units += close_units
            mt.exit_notional += close_q * p
            pos += close_q
            remaining -= close_q
            if abs(mt.entry_units - mt.exit_units) < _EPS:
                mt.exit_timestamp = event.timestamp

    state.position[iid] = pos


def trades(ledger: EventLedger) -> tuple[Trade, ...]:
    """Fold fills/commissions/funding into per-instrument round-trip trades (§4.6).

    Events are folded in append order. A round trip opens on the first fill after a flat
    position, accumulates same-direction fills into a volume-weighted entry and
    opposite-direction fills into a volume-weighted exit, and closes when the position
    returns to flat (a flip through zero closes the old trade and opens a new one).
    ``commission`` and ``funding_payment`` events are allocated to the open trade (or,
    if the position is already flat, the most-recently-closed trade); a cost with no
    associated round trip is not represented here (it still lives in the ledger and
    affects equity via ``cash_movement``).

    Only *closed* round trips are emitted; an open position at the end of the run is a
    position (see :func:`positions`), not a trade. The fold is inherently sequential
    (round-trip semantics are path-dependent); it is the one sanctioned non-vectorized
    part of this module (§3 principle 1).

    Args:
        ledger: The append-only event ledger.

    Returns:
        The completed round-trip trades, in the order their entries occurred.
    """
    if not isinstance(ledger, EventLedger):
        raise DataShapeError("trades() expects an EventLedger")

    state = _FoldState(position={}, current={})
    out: list[Trade] = []

    for event in ledger:
        t = event.event_type
        iid = event.instrument_id
        if t is EventType.FILL:
            _process_fill(state, out, event)
        elif t is EventType.COMMISSION or t is EventType.FUNDING_PAYMENT:
            mt = state.current.get(iid)
            if mt is not None:
                amount = cast(float, event.amount)
                if t is EventType.COMMISSION:
                    mt.commission += amount
                else:
                    mt.funding += amount

    for mt in state.current.values():
        if mt.exit_units > 0.0 and abs(mt.entry_units - mt.exit_units) < _EPS:
            out.append(_freeze_trade(mt))

    return tuple(out)


# ---------------------------------------------------------------------------
# Positions view (vectorized).
# ---------------------------------------------------------------------------


def _step_series(
    timestamps: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort ``(timestamps, values)`` into a minimal step series.

    Sorts ascending, keeps the *last* value per timestamp, and drops consecutive
    no-op changes (a step function records where the value actually changes).
    """
    if timestamps.shape[0] == 0:
        return timestamps, values
    order = np.argsort(timestamps, kind="stable")
    ts = timestamps[order]
    vals = values[order]
    unique = np.unique(ts)
    last = np.searchsorted(ts, unique, side="right") - 1
    ts = ts[last]
    vals = vals[last]
    if ts.shape[0] > 1:
        keep = np.empty(ts.shape[0], dtype=bool)
        keep[0] = True
        keep[1:] = vals[1:] != vals[:-1]
        ts = ts[keep]
        vals = vals[keep]
    return ts, vals


def positions(ledger: EventLedger, instrument_id: str | None = None) -> Positions:
    """Derive a position-over-time step series from ``position_change`` events (§4.6).

    Each ``position_change`` event records the *resulting* signed position at its bar
    boundary; the result is a step function (flat before the first change, holding
    between changes). The derivation is vectorized (sort + ``searchsorted`` dedupe).

    Args:
        ledger: The append-only event ledger.
        instrument_id: Restrict to one instrument. When ``None``, the ledger must
            contain ``position_change`` events for exactly one ``instrument_id``;
            otherwise :class:`~ube.core.errors.DataShapeError` is raised (a combined
            position across instruments is undefined — filter first).

    Returns:
        The resulting :class:`Positions` step series (possibly empty).
    """
    if not isinstance(ledger, EventLedger):
        raise DataShapeError("positions() expects an EventLedger")

    ts_list: list[int] = []
    pos_list: list[float] = []
    ids: set[str] = set()
    for event in ledger:
        if event.event_type is not EventType.POSITION_CHANGE:
            continue
        if instrument_id is None:
            ids.add(event.instrument_id)
        elif event.instrument_id != instrument_id:
            continue
        ts_list.append(event.timestamp)
        pos_list.append(cast(float, event.position_after))

    if instrument_id is None and len(ids) > 1:
        raise DataShapeError(
            "positions() requires an instrument_id when the ledger contains "
            f"position_change events for multiple instruments ({sorted(ids)})"
        )

    ts = np.asarray(ts_list, dtype=np.int64)
    pos = np.asarray(pos_list, dtype=np.float64)
    ts, pos = _step_series(ts, pos)
    return Positions(timestamps=ts, position=pos)


# ---------------------------------------------------------------------------
# Equity curve (vectorized mark-to-market + multi-currency normalization).
# ---------------------------------------------------------------------------


def _asi8(index: pd.DatetimeIndex) -> np.ndarray:
    """int64 nanosecond view of a ``DatetimeIndex`` (normalized to ns — pandas 3.x)."""
    return cast(np.ndarray, index.as_unit("ns").asi8)  # type: ignore[attr-defined]


def _timestamps_ns(md: MarketData) -> np.ndarray:
    """The bar-boundary axis of ``md`` as int64 (ns for time bars, bar index otherwise)."""
    if md.is_time_bars:
        return _asi8(cast(pd.DatetimeIndex, md.index))
    return md.index.to_numpy(dtype=np.int64)


def _settlement_currency(
    instruments: Mapping[str, Instrument], instrument_id: str, base_currency: str
) -> str:
    instr = instruments.get(instrument_id)
    if instr is None:
        raise DataShapeError(
            f"instrument_id {instrument_id!r} has no Instrument entry (its "
            "settlement_currency is required to normalize to base_currency)"
        )
    if not isinstance(instr, Instrument):
        raise DataShapeError(
            f"instruments[{instrument_id!r}] must be an Instrument; "
            f"got {type(instr).__name__}"
        )
    sc = instr.settlement_currency
    if sc is None:
        raise FXRateUnavailableError(
            f"instrument {instrument_id!r} has no settlement_currency; cannot "
            f"normalize to base_currency {base_currency!r} (§4.8 — never assume)"
        )
    return sc


def _fx_rate_at(
    from_currency: str,
    base_currency: str,
    fx_rates: Mapping[str, FXSeries],
    timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Forward-filled rate converting ``from_currency`` -> ``base_currency`` at each grid point.

    The rate is looked up under the concatenated pair key ``from_currency + base_currency``
    (e.g. ``"EURUSD"``), and its value is *units of base currency per one unit of the
    quoted currency*. When ``from_currency == base_currency`` the identity (``1.0``) is
    returned without any lookup. A missing pair, or a grid timestamp preceding the first
    rate entry, raises :class:`~ube.core.errors.FXRateUnavailableError` (§4.8 — never a
    silent 1:1 assumption).
    """
    if from_currency == base_currency:
        return np.ones(timestamps_ns.shape[0], dtype=np.float64)
    key = from_currency + base_currency
    fx = fx_rates.get(key)
    if fx is None:
        raise FXRateUnavailableError(
            f"no FX rate for pair {key!r} to convert {from_currency!r} -> "
            f"{base_currency!r}"
        )
    idx = np.searchsorted(fx.index, timestamps_ns, side="right") - 1
    if (idx < 0).any():
        raise FXRateUnavailableError(
            f"FX rate for pair {key!r} has no entry on/before one or more grid "
            "timestamps (cannot normalize — never assume a rate, §4.8)"
        )
    return fx.rate[idx]


def _build_equity(
    ledger: EventLedger,
    market_data: Mapping[str, MarketData],
    instruments: Mapping[str, Instrument],
    base_currency: str,
    fx_rates: Mapping[str, FXSeries],
) -> tuple[EquityCurve, dict[str, EquityCurve]]:
    """Compute the combined equity curve and the per-instrument breakdown once."""
    if not isinstance(ledger, EventLedger):
        raise DataShapeError("equity_curve() expects an EventLedger")
    if not isinstance(market_data, Mapping) or not market_data:
        raise DataShapeError(
            "equity_curve() requires a non-empty mapping of instrument_id -> MarketData"
        )
    if not isinstance(base_currency, str) or not base_currency.strip():
        raise ConfigError("base_currency must be a non-empty string")

    mds: dict[str, MarketData] = {}
    for iid, md in market_data.items():
        if not isinstance(md, MarketData):
            raise DataShapeError(f"market_data[{iid!r}] must be a MarketData")
        mds[iid] = md

    bar_types = {md.bar_type for md in mds.values()}
    if len(bar_types) > 1:
        raise DataShapeError(
            "combined equity curve requires a uniform bar_type across instruments; "
            f"got {sorted(bar_types)}"
        )

    # Combined grid: sorted union of all instruments' bar boundaries (§4.6 step 1).
    grid = np.unique(np.concatenate([_timestamps_ns(md) for md in mds.values()]))

    # Position-change step series per instrument.
    pc_ts: dict[str, list[int]] = {}
    pc_val: dict[str, list[float]] = {}
    for event in ledger:
        if event.event_type is EventType.POSITION_CHANGE:
            pc_ts.setdefault(event.instrument_id, []).append(event.timestamp)
            pc_val.setdefault(event.instrument_id, []).append(
                cast(float, event.position_after)
            )
    pc: dict[str, tuple[np.ndarray, np.ndarray]] = {
        iid: _step_series(
            np.asarray(pc_ts[iid], dtype=np.int64),
            np.asarray(pc_val[iid], dtype=np.float64),
        )
        for iid in pc_ts
    }

    # A held position with no market_data cannot be marked to market (§4.6 step 2) —
    # fail loudly rather than silently dropping it from the combined equity (§4.8).
    unmarketable = set(pc_ts) - set(mds)
    if unmarketable:
        raise DataShapeError(
            "position_change events reference instruments with no market_data entry: "
            f"{sorted(unmarketable)} — cannot mark to market"
        )

    n = grid.shape[0]
    marks_base_sum = np.zeros(n, dtype=np.float64)
    per_instrument: dict[str, EquityCurve] = {}

    for iid, md in mds.items():
        t = _timestamps_ns(md)
        close = md.close

        pt, pv = pc.get(iid, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
        if pt.shape[0] == 0:
            pos_grid = np.zeros(n, dtype=np.float64)
        else:
            pidx = np.searchsorted(pt, grid, side="right") - 1
            pos_grid = np.where(pidx >= 0, pv[pidx], 0.0)

        # Mark-to-market with the latest close on/before each grid point (§4.6 step 2):
        # forward-filled between the instrument's own bars (and past its last bar, where
        # "latest close on/before" is the last close); zero before its first bar.
        cidx = np.searchsorted(t, grid, side="right") - 1
        valid = cidx >= 0
        clipped = np.clip(cidx, 0, None)
        close_grid = np.where(valid, close[clipped], 0.0)

        mark_local = pos_grid * close_grid  # settlement-currency value

        # Convert to base currency (§4.6 step 3).
        sc = _settlement_currency(instruments, iid, base_currency)
        mark_base = mark_local * _fx_rate_at(sc, base_currency, fx_rates, grid)

        marks_base_sum += mark_base
        per_instrument[iid] = EquityCurve(index=_grid_index(bar_types, grid), equity=mark_base)

    # Base-currency cash: cumulative sum of converted cash movements (§4.6 step 4).
    cash_ts: list[int] = []
    cash_base: list[float] = []
    for event in ledger:
        if event.event_type is EventType.CASH_MOVEMENT:
            amount = cast(float, event.amount)
            currency = cast(str, event.currency)
            rate = _fx_rate_at(
                currency,
                base_currency,
                fx_rates,
                np.asarray([event.timestamp], dtype=np.int64),
            )
            cash_ts.append(event.timestamp)
            cash_base.append(amount * float(rate[0]))

    cash_grid = np.zeros(n, dtype=np.float64)
    if cash_ts:
        cts = np.asarray(cash_ts, dtype=np.int64)
        cb = np.asarray(cash_base, dtype=np.float64)
        order = np.argsort(cts, kind="stable")
        cts = cts[order]
        cb = cb[order]
        cum = np.cumsum(cb)
        cidx = np.searchsorted(cts, grid, side="right") - 1
        cash_grid = np.where(cidx >= 0, cum[np.clip(cidx, 0, cum.shape[0] - 1)], 0.0)

    combined = EquityCurve(index=_grid_index(bar_types, grid), equity=cash_grid + marks_base_sum)
    return combined, per_instrument


def _grid_index(bar_types: set[str], grid: np.ndarray) -> pd.Index:
    """Reconstruct the public index for the grid from the (uniform) bar type."""
    if "time" in bar_types:
        return cast(pd.Index, pd.to_datetime(grid, unit="ns", utc=True).as_unit("ns"))
    return pd.RangeIndex(grid.shape[0])


def equity_curve(
    ledger: EventLedger,
    market_data: Mapping[str, MarketData],
    instruments: Mapping[str, Instrument],
    *,
    base_currency: str,
    fx_rates: Mapping[str, FXSeries] | None = None,
) -> EquityCurve:
    """Derive the combined portfolio equity curve (§4.6).

    The derivation follows §4.6 exactly and is vectorized (``searchsorted`` /
    ``cumsum``, no per-bar Python loop):

    1. Common index = the sorted union of all instruments' bar timestamps.
    2. Each instrument's open position is marked to market with its latest close on/before
       each grid point (forward-filled between its own bars).
    3. Each instrument's mark value is converted to ``base_currency`` via its
       ``Instrument.settlement_currency`` and the caller-supplied ``fx_rates``.
    4. Total = base-currency cash (cumulative ``cash_movement`` amounts, each converted
       to ``base_currency`` at its timestamp) + the sum of all instrument mark values.

    ``fx_rates`` is a ``Mapping[str, FXSeries]`` keyed by the concatenated currency pair
    ``"<settlement><base>"`` (e.g. ``"EURUSD"``), where each rate is *units of base per
    one unit of settlement*. ``settlement_currency == base_currency`` needs no entry. A
    missing pair — or an instrument without a ``settlement_currency`` — raises
    :class:`~ube.core.errors.FXRateUnavailableError` (§4.8: never silently assume or
    leave unconverted).

    Args:
        ledger: The append-only event ledger.
        market_data: ``instrument_id -> MarketData`` (bars used for mark-to-market).
        instruments: ``instrument_id -> Instrument`` (provides ``settlement_currency``).
        base_currency: The explicit portfolio base currency (§4.8).
        fx_rates: Optional historical FX rates keyed by currency pair.

    Returns:
        The combined equity curve in ``base_currency``.
    """
    combined, _ = _build_equity(
        ledger, market_data, instruments, base_currency, fx_rates or {}
    )
    return combined


def equity_curve_by_instrument(
    ledger: EventLedger,
    market_data: Mapping[str, MarketData],
    instruments: Mapping[str, Instrument],
    *,
    base_currency: str,
    fx_rates: Mapping[str, FXSeries] | None = None,
) -> dict[str, EquityCurve]:
    """Derive the per-instrument equity breakdown (§4.6 "by instrument").

    Returns one :class:`EquityCurve` per instrument, each on the *same* combined grid as
    :func:`equity_curve` and each equal to that instrument's mark-to-market value in
    ``base_currency`` — so the combined curve is ``base-currency cash + sum(breakdown)``.

    Args:
        ledger: The append-only event ledger.
        market_data: ``instrument_id -> MarketData``.
        instruments: ``instrument_id -> Instrument``.
        base_currency: The explicit portfolio base currency (§4.8).
        fx_rates: Optional historical FX rates keyed by currency pair.

    Returns:
        ``instrument_id -> EquityCurve`` (mark value in ``base_currency``).
    """
    _, per_instrument = _build_equity(
        ledger, market_data, instruments, base_currency, fx_rates or {}
    )
    return per_instrument

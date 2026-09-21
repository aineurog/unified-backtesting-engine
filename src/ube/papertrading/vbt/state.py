"""vectorbt paper-trading state (§4/§5 of the vectorbt paper plan).

The vectorbt backend is *stateless and recomputable*: every ``step`` re-runs
``vectorbt.from_signals`` over a window of bars and folds the resulting trades into the
canonical :class:`~ube.core.ledger.EventLedger`. That makes the engine correct only if it
can answer two questions from the persisted state alone:

* **Where does the replay window start?** (:func:`window_start_ns`) — the carried entry bar
  when a position is open, otherwise the last close bar (minus an indicator warmup).
* **What is the account balance *before* the carried entry?** (:func:`checkpoint_balance`) —
  the ``starting_balance`` to seed the window's sizing with, so a carried position is not
  re-sized off the *current* (post-trade) equity.

This module carries the derivation logic as free functions (usable with any
:class:`~ube.papertrading.state.PaperState`) and :class:`VbtPaperState`, whose methods add
an incremental fill-scan cache and a checkpoint fold memo.

The checkpoint formula
----------------------

This is *not* the naive ``starting_capital + Σ(cash) − Σ(commission) − Σ(funding)`` over
``ts < window_start``. That formula was falsified against the adapter's own sizing fold
(``_build_size_series.running``): it double-counts the ledger's opening ``cash_movement``,
drifts by the funding it subtracts (the sizing fold excludes funding), and goes negative on
a same-bar flip because the prior close and the carried entry share a bar. The validated
formula is::

    bound = last_close_ns()                       # exit ts of the most recent close
    total = Σ(cash_movement) − Σ(commission)      # over events with ts <= bound, NO funding
    if no cash/commission event:
        total = starting_capital
    if open_position and (bound is None or entry_ns <= bound):
        enot = quantity * entry_price * multiplier
        total += side * enot + fill_cost(cost_model, notional=enot)   # remove carried legs

The correction is applied only when the carried entry's legs fall *within* the fold — i.e.
on a same-bar flip (``entry_ns == bound``) or before any close (``bound is None``). For an
ordinary carry (entry strictly after the last close) the entry legs are already outside the
fold, so the balance before the entry is simply the balance after the prior close.
"""

from __future__ import annotations

from typing import Any, cast

from ube.core.cost import fill_cost
from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType
from ube.papertrading.state import OpenPosition, PaperState

__all__ = [
    "VBT_ENGINE_TAG",
    "VbtPaperState",
    "checkpoint_balance",
    "fold_cash_commission",
    "last_close_ns",
    "window_start_ns",
]

#: Persisted engine marker written into ``PaperState.aux_data["vbt"]`` (see §4/§5).
VBT_ENGINE_TAG: dict[str, Any] = {"engine": "vectorbt", "schema": 1}

_EPS = 1e-12


def _raw_events(ledger: EventLedger) -> list[Any]:
    """The append-only backing list, for incremental scans.

    ``EventLedger.events`` returns a fresh tuple snapshot (an O(n) copy) on every access;
    the scanners here only ever *append*, so they hold the backing list and resume from the
    last scanned index. Falls back to a snapshot for duck-typed ledgers.
    """
    raw = getattr(ledger, "_events", None)
    return raw if isinstance(raw, list) else list(ledger.events)


def last_close_ns(ledger: EventLedger, instrument_id: str) -> int | None:
    """Bar ts of the most recent fill that flattened or flipped the position.

    Derived from the ``fill`` stream (not stored): a fill closes the current trade when it
    returns the running signed position to zero or crosses through zero (a flip). Returns
    ``None`` when no fill has closed a position.
    """
    pos = 0.0
    last_close: int | None = None
    for event in _raw_events(ledger):
        if event.event_type is not EventType.FILL or event.instrument_id != instrument_id:
            continue
        if event.side is None or event.quantity is None:
            continue
        new = pos + float(event.side) * float(event.quantity)
        if abs(pos) > _EPS and (abs(new) < _EPS or (new > 0.0) != (pos > 0.0)):
            last_close = int(event.timestamp)
        pos = new
    return last_close


def fold_cash_commission(
    ledger: EventLedger, bound_ns: int | None
) -> tuple[float, bool]:
    """``(Σ cash_movement − Σ commission, saw_any)`` over events with ``ts <= bound_ns``.

    Funding is deliberately excluded (the adapter's sizing fold ``_build_size_series``
    excludes it too — see the module docstring). ``bound_ns=None`` folds every event.
    ``saw_any`` is ``False`` for an empty ledger (or one with no cash/commission events),
    where the caller substitutes ``starting_capital``.
    """
    total = 0.0
    saw_any = False
    limit = None if bound_ns is None else int(bound_ns)
    for event in _raw_events(ledger):
        if limit is not None and int(event.timestamp) > limit:
            continue
        if event.event_type is EventType.CASH_MOVEMENT and event.amount is not None:
            saw_any = True
            total += float(event.amount)
        elif event.event_type is EventType.COMMISSION and event.amount is not None:
            saw_any = True
            total -= float(event.amount)
    return total, saw_any


def _entry_legs(
    open_position: OpenPosition, multiplier: float, cost_model: Any
) -> float:
    """``side*entry_notional + fill_cost(entry_notional)`` for the carried entry.

    Subtracting this from a fold that already contains the carried entry's legs yields the
    balance *before* the entry — the quantity to seed window sizing with.
    """
    enot = (
        float(open_position.quantity)
        * float(open_position.entry_price)
        * float(multiplier)
    )
    entry_fee = float(fill_cost(cost_model, notional=enot)) if cost_model is not None else 0.0
    return float(open_position.side) * enot + entry_fee


def checkpoint_balance(
    ledger: EventLedger,
    instrument_id: str,
    open_position: OpenPosition | None,
    starting_capital: float,
    *,
    multiplier: float = 1.0,
    cost_model: Any = None,
) -> float:
    """The validated pre-window account balance (§5) — see the module docstring.

    Args:
        ledger: The persisted event ledger.
        instrument_id: The single traded instrument.
        open_position: The carried open position, or ``None`` when flat.
        starting_capital: The session starting balance (substituted when the ledger holds
            no cash/commission events yet).
        multiplier: The instrument's contract multiplier.
        cost_model: The resolved cost model (for the carried entry's commission leg).

    Returns:
        The balance to seed the window's sizing with.
    """
    bound = last_close_ns(ledger, instrument_id)
    total, saw_any = fold_cash_commission(ledger, bound)
    if not saw_any:
        total = float(starting_capital)
    if open_position is not None and (bound is None or int(open_position.entry_ns) <= bound):
        total += _entry_legs(open_position, multiplier, cost_model)
    return float(total)


def window_start_ns(
    ledger: EventLedger,
    instrument_id: str,
    open_position: OpenPosition | None,
    *,
    last_processed_ns: int | None,
    warmup_ns: int,
    run_start_ns: int,
) -> int:
    """The int64-ns start of the replay window (§6.1).

    * Open position — the carried entry bar (no warmup, no cursor max): the window must
      contain the entry so vectorbt re-opens and manages the carried trade.
    * Flat — the last close bar (``warmup_ns``) of indicator context. With ``warmup_ns=0``
      this collapses to the cursor (``max(last_close, cursor)``); with a warmup it is
      *pinned* to ``last_close - warmup_ns`` regardless of how far the cursor advanced.
    """
    if open_position is not None:
        return int(open_position.entry_ns)
    last_close = last_close_ns(ledger, instrument_id)
    base = int(last_close) if last_close is not None else int(run_start_ns)
    if warmup_ns > 0:
        return base - int(warmup_ns)
    if last_processed_ns is not None:
        base = max(base, int(last_processed_ns))
    return base


class VbtPaperState(PaperState):
    """A :class:`PaperState` with the vectorbt derivation caches (§4/§5).

    Adds an incremental, append-only fill scan for :meth:`last_close_ns` and a checkpoint
    fold :meth:`checkpoint_balance` memoized on the last close (the fold over ``ts <= bound``
    is fixed once the close is known; only a new close invalidates it). Both are transient —
    never persisted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scan_events: list[Any] | None = None
        self._scan_n = 0
        self._scan_pos = 0.0
        self._scan_last_close: int | None = None
        self._ckpt_memo: tuple[int, float] | None = None

    def last_close_ns(self) -> int | None:
        """Incremental :func:`last_close_ns` over this state's ledger."""
        events = _raw_events(self.ledger)
        if events is not self._scan_events:
            self._scan_events = events
            self._scan_n = 0
            self._scan_pos = 0.0
            self._scan_last_close = None
        pos = self._scan_pos
        last_close = self._scan_last_close
        i = self._scan_n
        n = len(events)
        iid = self.instrument_id
        while i < n:
            event = events[i]
            i += 1
            if event.event_type is not EventType.FILL or event.instrument_id != iid:
                continue
            if event.side is None or event.quantity is None:
                continue
            new = pos + float(event.side) * float(event.quantity)
            if abs(pos) > _EPS and (abs(new) < _EPS or (new > 0.0) != (pos > 0.0)):
                last_close = int(event.timestamp)
            pos = new
        self._scan_n = i
        self._scan_pos = pos
        self._scan_last_close = last_close
        return last_close

    def checkpoint_balance(
        self,
        starting_capital: float,
        *,
        multiplier: float = 1.0,
        cost_model: Any = None,
    ) -> float:
        """:func:`checkpoint_balance` with a fold memo keyed on the last close."""
        bound = self.last_close_ns()
        if bound is None:
            total, saw_any = fold_cash_commission(self.ledger, None)
            if not saw_any:
                total = float(starting_capital)
        elif self._ckpt_memo is not None and self._ckpt_memo[0] == bound:
            total = self._ckpt_memo[1]
        else:
            total, saw_any = fold_cash_commission(self.ledger, bound)
            if not saw_any:
                total = float(starting_capital)
            self._ckpt_memo = (bound, total)
        pos = self.open_position
        if pos is not None and (bound is None or int(pos.entry_ns) <= bound):
            total += _entry_legs(pos, multiplier, cost_model)
        return float(total)

    def window_start_ns(self, *, warmup_ns: int = 0) -> int:
        """:func:`window_start_ns` for this state — the fetch window's first bar.

        Callers (the live runner) fetch bars from this timestamp so the window always
        contains the carried entry bar (or the cursor/warmup context when flat); the
        engine re-slices to this exact bound regardless of extra leading bars.
        """
        run_start_ns = (
            int(self.ledger.events[0].timestamp)
            if self.ledger.events
            else (int(self.last_processed_ns) if self.last_processed_ns is not None else 0)
        )
        return window_start_ns(
            self.ledger,
            self.instrument_id,
            self.open_position,
            last_processed_ns=self.last_processed_ns,
            warmup_ns=int(warmup_ns),
            run_start_ns=run_start_ns,
        )

    # -- persistence: write / verify the vectorbt engine tag (aux_data) ----

    def save(self, path: str, run_id: str = "default") -> None:
        """Persist, stamping ``aux_data['vbt']`` so the row is recognizable on reload."""
        aux: dict[str, Any] = dict(self.aux_data) if self.aux_data else {}
        aux["vbt"] = dict(VBT_ENGINE_TAG)
        self.aux_data = aux
        super().save(path, run_id=run_id)

    @classmethod
    def load(cls, path: str, run_id: str = "default") -> VbtPaperState:
        """Load and verify the row was written by the vectorbt engine.

        Raises:
            StateCorruptionError: The row lacks the ``aux_data['vbt']`` engine marker —
                it was written by a different (e.g. nautilus) engine and must not be
                resumed with the vectorbt backend.
        """
        obj = cast("VbtPaperState", super().load(path, run_id=run_id))
        tag = (obj.aux_data or {}).get("vbt")
        if not isinstance(tag, dict) or tag.get("engine") != "vectorbt":
            raise StateCorruptionError(
                f"paper_state run_id={run_id!r} was not written by the vectorbt engine "
                "(missing aux_data['vbt'] engine tag); refusing to resume across engines"
            )
        return obj

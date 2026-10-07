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

The checkpoint is the **realized equity** the adapter's own sizing fold bases the carried
entry on — ``starting_capital + Σ net PnL`` of the round trips that actually *completed* in
the fill stream (a close fill or a flip), at the fills' already-slipped prices, with both
legs' commissions via ``fill_cost``::

    completed = any fill that flattens or flips the carried side (see
                :func:`~ube.papertrading.state.step_position_after_fill`)
    net_pnl   = side * (exit_px − entry_px) * qty * multiplier
                − fill_cost(entry notional) − fill_cost(exit notional)      [fees]
    equity    = starting_capital + Σ net_pnl(completed)

This equals the older ``Σ(cash_movement) − Σ(commission)`` fold over ``ts <= last_close``
whenever every closed position's legs are booked (the cases that fold was "validated"
against). It stays correct when a short's *buyback leg is suppressed*: the adapters'
end-of-run restore can leave a stopped short "held", skipping its cover fill, so that short's
opening credit never leaves the margin-cash fold and inflates it ~11x (the sizing-feed
``running`` figures the seed must match). Realized equity matures only completed round trips,
so a carried entry is *automatically* excluded — no ``_entry_legs``-style correction, no
dependence on the last-close bound. Funding is ignored, matching the sizing fold.
"""

from __future__ import annotations

from typing import Any, cast

from ube.core.cost import fill_cost
from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType
from ube.papertrading.state import (
    OpenPosition,
    PaperState,
    step_position_after_fill,
)

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

# Scale-relative "flat" tolerance for the fill scan (mirrors ``ube.core.ledger`` and
# ``ube.papertrading.core``). A cover fill can settle a position one float-ulp off the
# size it opened — e.g. a 75382.2076-unit forex short covered at 75382.2076 leaves a
# 1.46e-11 residue, orders of magnitude above an absolute 1e-12 floor. Classifying that
# residue as a live position makes ``last_close_ns`` skip the real close bar and shifts the
# replay window and checkpoint bound onto the wrong bars.
_FLAT_REL_TOL = 1e-9


def _flat_tol(*scales: float) -> float:
    """Scale-relative "is flat" tolerance: ``~1e-9`` of the largest quantity involved."""
    return max(max(abs(float(s)) for s in scales), 1.0) * _FLAT_REL_TOL


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

    Derived from the ``fill`` stream (not stored) with the same fold as
    :func:`~ube.papertrading.core._open_position_from_ledger` (see
    :func:`~ube.papertrading.state.step_position_after_fill`): a close fill (``exit_reason``
    set) or an opposite-side fill that fully covers the carried side records a close bar.
    Returns ``None`` when no fill has closed a position.
    """
    pos = 0.0
    entry_px = 0.0
    entry_ns = 0
    order_id = ""
    last_close: int | None = None
    for event in _raw_events(ledger):
        if event.event_type is not EventType.FILL or event.instrument_id != instrument_id:
            continue
        if event.side is None or event.quantity is None or event.price is None:
            continue
        pos, entry_px, entry_ns, order_id, closed = step_position_after_fill(
            pos, entry_px, entry_ns, order_id, event
        )
        if closed:
            last_close = int(event.timestamp)
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


def _realized_pnl(
    side: int,
    qty: float,
    entry: float,
    exit_px: float,
    *,
    multiplier: float,
    cost_model: Any,
) -> float:
    """Net PnL of a matured round trip at the fills' already-slipped prices."""
    pnl = float(side) * (exit_px - entry) * qty * multiplier
    if cost_model is not None:
        pnl -= float(fill_cost(cost_model, notional=qty * entry * multiplier))
        pnl -= float(fill_cost(cost_model, notional=qty * exit_px * multiplier))
    return pnl


def _realized_round_trips(
    events: list[Any],
    instrument_id: str,
    *,
    multiplier: float,
    cost_model: Any,
) -> float:
    """``Σ net PnL`` of the *completed* round trips in an event list (§5).

    Folds the fill stream exactly like
    :func:`~ube.papertrading.state.step_position_after_fill` — a close fill (``exit_reason``
    set) or a fully-covering opposite-side fill matures the carried side — and adds each
    matured side's net PnL at the fills' already-slipped prices. A carried (still-open)
    entry is never matured, so it is automatically excluded from the equity. Funding is
    ignored, matching the adapters' sizing fold.
    """
    total = 0.0
    side = 0
    qty = 0.0
    entry = 0.0
    for event in events:
        if event.event_type is not EventType.FILL or event.instrument_id != instrument_id:
            continue
        if event.side is None or event.quantity is None or event.price is None:
            continue
        q = float(event.side) * float(event.quantity)
        px = float(event.price)
        tol = _flat_tol(q, qty)
        if abs(qty) <= tol:
            if event.exit_reason is not None:
                continue
            side = 1 if q > 0 else -1
            qty, entry = abs(q), px
        elif event.exit_reason is not None:
            # A close fill never opens a side: full/over-size closes mature the round trip,
            # smaller ones partially reduce it.
            if abs(q) >= qty - tol:
                total += _realized_pnl(
                    side, qty, entry, px, multiplier=multiplier, cost_model=cost_model
                )
                side, qty, entry = 0, 0.0, 0.0
            else:
                qty -= abs(q)
        elif (q > 0) == (side > 0):
            new_qty = qty + abs(q)
            entry = (entry * qty + px * abs(q)) / new_qty
            qty = new_qty
        elif abs(q) >= qty - tol:
            total += _realized_pnl(
                side, qty, entry, px, multiplier=multiplier, cost_model=cost_model
            )
            flip = q + float(side) * qty
            if abs(flip) <= tol:
                # Exact (ulp-level) cover: the round trip matured flat.
                side, qty, entry = 0, 0.0, 0.0
            else:
                # Full opposite replacement (same-bar flip): the fill's own size is the new
                # carried side, opened at its own (already-slipped) price.
                side = 1 if q > 0 else -1
                qty, entry = abs(q), px
        else:
            qty -= abs(q)
    return float(total)


def checkpoint_balance(
    ledger: EventLedger,
    instrument_id: str,
    open_position: OpenPosition | None,
    starting_capital: float,
    *,
    multiplier: float = 1.0,
    cost_model: Any = None,
) -> float:
    """The realized-equity pre-window account balance (§5) — see the module docstring.

    Args:
        ledger: The persisted event ledger.
        instrument_id: The single traded instrument.
        open_position: Informational — the carried open position, or ``None`` when flat.
            The derived balance is independent of it: only completed round trips are
            matured, so a carried entry is automatically excluded.
        starting_capital: The session starting balance (substituted when the ledger holds
            no completed round trips yet).
        multiplier: The instrument's contract multiplier.
        cost_model: The resolved cost model (commission legs on the realized notional).

    Returns:
        The balance to seed the window's sizing with.
    """
    realized = _realized_round_trips(
        _raw_events(ledger),
        instrument_id,
        multiplier=multiplier,
        cost_model=cost_model,
    )
    return float(starting_capital) + realized


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
    :meth:`checkpoint_balance` memoized on the fill count (the realized-equity fold is fixed
    once the fill stream stops moving; funding/cash appends can't change it). Both are
    transient — never persisted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scan_events: list[Any] | None = None
        self._scan_n = 0
        self._scan_pos = 0.0
        self._scan_entry_px = 0.0
        self._scan_entry_ns = 0
        self._scan_order_id = ""
        self._scan_last_close: int | None = None
        self._ckpt_memo: tuple[int, float] | None = None

    def last_close_ns(self) -> int | None:
        """Incremental :func:`last_close_ns` over this state's ledger."""
        events = _raw_events(self.ledger)
        if events is not self._scan_events:
            self._scan_events = events
            self._scan_n = 0
            self._scan_pos = 0.0
            self._scan_entry_px = 0.0
            self._scan_entry_ns = 0
            self._scan_order_id = ""
            self._scan_last_close = None
        pos = self._scan_pos
        entry_px = self._scan_entry_px
        entry_ns = self._scan_entry_ns
        order_id = self._scan_order_id
        last_close = self._scan_last_close
        i = self._scan_n
        n = len(events)
        iid = self.instrument_id
        while i < n:
            event = events[i]
            i += 1
            if event.event_type is not EventType.FILL or event.instrument_id != iid:
                continue
            if event.side is None or event.quantity is None or event.price is None:
                continue
            pos, entry_px, entry_ns, order_id, closed = step_position_after_fill(
                pos, entry_px, entry_ns, order_id, event
            )
            if closed:
                last_close = int(event.timestamp)
        self._scan_n = i
        self._scan_pos = pos
        self._scan_entry_px = entry_px
        self._scan_entry_ns = entry_ns
        self._scan_order_id = order_id
        self._scan_last_close = last_close
        return last_close

    def checkpoint_balance(
        self,
        starting_capital: float,
        *,
        multiplier: float = 1.0,
        cost_model: Any = None,
    ) -> float:
        """:func:`checkpoint_balance` with a memo keyed on the state's fill count.

        Equity only moves when a fill matures a round trip, so while the fill stream is
        stable the realized fold is served from the memo (appended funding/cash events skip
        the fold entirely).
        """
        raw = _raw_events(self.ledger)
        iid = self.instrument_id
        fill_n = sum(
            1
            for e in raw
            if e.event_type is EventType.FILL and e.instrument_id == iid
        )
        if self._ckpt_memo is not None and self._ckpt_memo[0] == fill_n:
            return float(starting_capital) + self._ckpt_memo[1]
        realized = _realized_round_trips(
            raw, iid, multiplier=multiplier, cost_model=cost_model
        )
        self._ckpt_memo = (fill_n, realized)
        return float(starting_capital) + realized

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

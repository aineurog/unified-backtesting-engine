"""Paper-trading run state + sqlite persistence (§9.5, §18).

``PaperState`` is the engine-agnostic, resumable snapshot of a single-instrument paper
session. It is *not* the source of truth for trade accounting — that is the
:class:`~ube.core.ledger.EventLedger` it wraps (§4.6) — but it carries the cursor and
open-position pointers a restarted run needs so it never opens a second bracket on a
symbol that already has one open (the standalone sim's ``_resume_open_trade``).

Persistence mirrors :mod:`ube.core.experiment_log`: a single embedded sqlite table, one
row per ``run_id``, auto-created, overridable path. One transaction per save. Bad data on
load raises :class:`~ube.core.errors.StateCorruptionError` (never a silent partial
restore).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType, LedgerEvent

__all__ = [
    "OpenPosition",
    "PaperState",
]


@dataclass(frozen=True)
class OpenPosition:
    """The single open trade a resumed run must reconstruct (§9.5).

    Attributes:
        side: ``+1`` long / ``-1`` short.
        quantity: Positive units still open.
        entry_price: Volume-weighted entry price of the open units.
        entry_ns: Bar-boundary timestamp (int64 ns) of the entry fill.
        trade_id: The ledger key for the open trade (the ``order_id``/fill link).
    """

    side: int
    quantity: float
    entry_price: float
    entry_ns: int
    trade_id: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_state (
    run_id           TEXT PRIMARY KEY,
    instrument_id    TEXT NOT NULL,
    last_processed_ns INTEGER,
    ledger           TEXT NOT NULL,
    open_position    TEXT,
    pending_levels   TEXT,
    signal_fn_state  TEXT,
    config_ref       TEXT
)
"""

_INSERT_SQL = (
    "INSERT OR REPLACE INTO paper_state "
    "(run_id, instrument_id, last_processed_ns, ledger, open_position, "
    "pending_levels, signal_fn_state, config_ref) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def _ledger_to_json(ledger: EventLedger) -> str:
    rows = [
        {k: (v.value if isinstance(v, EventType) else v) for k, v in asdict(e).items()}
        for e in ledger.events
    ]
    return json.dumps(rows)


def _ledger_from_json(text: str) -> EventLedger:
    try:
        rows = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise StateCorruptionError(f"ledger blob is not valid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise StateCorruptionError("ledger blob must be a list of events")
    events: list[LedgerEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            raise StateCorruptionError("ledger event row must be an object")
        try:
            row["event_type"] = EventType(row["event_type"])
            events.append(LedgerEvent(**row))
        except (KeyError, ValueError, TypeError) as exc:
            raise StateCorruptionError(f"malformed ledger event: {exc}") from exc
    return EventLedger(events)


def _json(obj: Any) -> str:
    return json.dumps(obj)


def _from_json(text: str | None, default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise StateCorruptionError(f"state blob is not valid JSON: {exc}") from exc


class PaperState:
    """The resumable snapshot of a paper-trading run (§9.5).

    Attributes:
        instrument_id: The single traded instrument (paper trading is single-instrument,
            §6 of the plan).
        ledger: The append-only :class:`~ube.core.ledger.EventLedger` — the audit trail.
        last_processed_ns: The idempotency cursor; the timestamp of the last bar fed
            (``None`` before any bar, §9.6).
        open_position: The currently-open trade, or ``None`` when flat.
        pending_levels: Per-open-trade TP/SL (and scale-out) levels the backend needs to
            rebuild working orders on resume.
        signal_fn_state: State carried by a streaming signal function (§9.5); ``{}`` when
            stateless.
        config_ref: A pointer (e.g. a run id or config path) to the canonical config used;
            the full config is re-supplied by the caller on resume, not persisted.
    """

    def __init__(
        self,
        instrument_id: str,
        ledger: EventLedger,
        last_processed_ns: int | None = None,
        open_position: OpenPosition | None = None,
        pending_levels: dict[str, Any] | None = None,
        signal_fn_state: dict[str, Any] | None = None,
        config_ref: str | None = None,
    ) -> None:
        self.instrument_id = instrument_id
        self.ledger = ledger
        self.last_processed_ns = last_processed_ns
        self.open_position = open_position
        self.pending_levels: dict[str, Any] = pending_levels if pending_levels is not None else {}
        self.signal_fn_state: dict[str, Any] = (
            signal_fn_state if signal_fn_state is not None else {}
        )
        self.config_ref = config_ref

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------

    def save(self, path: str, run_id: str = "default") -> None:
        """Persist this state to ``path`` (sqlite), upserting ``run_id`` (§18)."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise StateCorruptionError("run_id must be a non-empty string")
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), isolation_level=None)
            conn.execute(_SCHEMA_SQL)
            conn.execute(
                _INSERT_SQL,
                (
                    run_id,
                    self.instrument_id,
                    self.last_processed_ns,
                    _ledger_to_json(self.ledger),
                    _json(asdict(self.open_position)) if self.open_position else None,
                    _json(self.pending_levels),
                    _json(self.signal_fn_state),
                    self.config_ref,
                ),
            )
            conn.close()
        except sqlite3.Error as exc:
            raise StateCorruptionError(f"could not save paper state to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str, run_id: str = "default") -> PaperState:
        """Restore a state previously written by :meth:`save` (§18).

        Raises:
            StateCorruptionError: The file is missing, the row is absent, or any blob
                fails to deserialize.
        """
        if not Path(path).exists():
            raise StateCorruptionError(f"paper state file {path} does not exist")
        try:
            conn = sqlite3.connect(str(path), isolation_level=None)
            row = conn.execute(
                "SELECT instrument_id, last_processed_ns, ledger, open_position, "
                "pending_levels, signal_fn_state, config_ref FROM paper_state "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            conn.close()
        except sqlite3.Error as exc:
            raise StateCorruptionError(f"could not read paper state at {path}: {exc}") from exc
        if row is None:
            raise StateCorruptionError(f"no paper state for run_id={run_id!r} in {path}")

        instrument_id, last_ns, ledger_text, open_text, pending_text, sfn_text, config_ref = row
        ledger = _ledger_from_json(ledger_text)
        open_position = None
        if open_text is not None:
            try:
                open_position = OpenPosition(**_from_json(open_text, {}))
            except TypeError as exc:
                raise StateCorruptionError(f"malformed open_position: {exc}") from exc
        return cls(
            instrument_id=instrument_id,
            ledger=ledger,
            last_processed_ns=last_ns,
            open_position=open_position,
            pending_levels=_from_json(pending_text, {}),
            signal_fn_state=_from_json(sfn_text, {}),
            config_ref=config_ref,
        )

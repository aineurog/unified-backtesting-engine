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

import contextlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import StateCorruptionError
from ube.core.ledger import EventLedger, EventType, LedgerEvent

__all__ = [
    "ExitSeed",
    "OpenPosition",
    "PaperState",
    "get_or_create_run_id",
    "save_trades",
    "save_equity",
    "load_trades",
    "load_equity",
]


@dataclass(frozen=True)
class ExitSeed:
    """Minimal sufficient statistics to rebuild trailing/ATR exits on resume (§8, issue C).

    Persisting only these *bounded* values (not the full bar history — plan blocker #5 /
    Point 9) lets a resumed run re-seed its exit-level computation so trailing / ATR exits
    are not degenerate on the first bars after a resume.

    Attributes:
        extreme_price: The running peak (long) / trough (short) price reached since entry.
            Seeds trailing-stop / chandelier ratcheting so a resume does not reset the stop
            to the current bar. ``None`` when no trailing-style exit requires it.
        atr_window: The last ``period + 1`` bars' high/low/close (as ``[high, low, close]``
            per bar, in bar order) needed to seed Wilder ATR warmup. Bounded, so storage cost
            is independent of history length. ``None`` when no ATR-based exit requires it.
    """

    extreme_price: float | None = None
    atr_window: list[list[float]] | None = None


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
    last_price       REAL,
    ledger           TEXT NOT NULL,
    open_position    TEXT,
    pending_levels   TEXT,
    signal_fn_state  TEXT,
    config_ref       TEXT,
    aux_data         TEXT,
    exit_seed        TEXT,
    last_funding_ns  INTEGER
)
"""

_STRATEGIES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_name TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    created_at    INTEGER NOT NULL
)
"""

_TRADES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    run_id          TEXT NOT NULL,
    instrument_id   TEXT NOT NULL,
    entry_timestamp INTEGER NOT NULL,
    exit_timestamp  INTEGER,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    side            INTEGER NOT NULL,
    quantity        REAL NOT NULL,
    status          TEXT NOT NULL,
    entry_notional  REAL,
    exit_notional   REAL,
    gross_pnl       REAL,
    commission      REAL,
    funding         REAL,
    net_pnl         REAL,
    exit_reason     TEXT,
    PRIMARY KEY (run_id, entry_timestamp, instrument_id)
)
"""

_EQUITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS equity (
    run_id    TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    equity    REAL NOT NULL,
    returns   REAL NOT NULL,
    PRIMARY KEY (run_id, timestamp)
)
"""

_INSERT_SQL = (
    "INSERT OR REPLACE INTO paper_state "
    "(run_id, instrument_id, last_processed_ns, last_price, ledger, open_position, "
    "pending_levels, signal_fn_state, config_ref, aux_data, exit_seed, last_funding_ns) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# -- strategy registry helpers (§9.5 strategy_name → run_id) ----------------


def _ensure_all_tables(conn: sqlite3.Connection) -> None:
    """Create all paper-trading tables if missing (idempotent)."""
    conn.execute(_SCHEMA_SQL)
    conn.execute(_STRATEGIES_SCHEMA_SQL)
    conn.execute(_TRADES_SCHEMA_SQL)
    conn.execute(_EQUITY_SCHEMA_SQL)
    # migrations for old DBs that predate new tables
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE paper_state ADD COLUMN aux_data TEXT")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE paper_state ADD COLUMN exit_seed TEXT")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE paper_state ADD COLUMN last_funding_ns INTEGER")


def get_or_create_run_id(
    db_path: str, strategy_name: str, instrument_id: str
) -> str:
    """Resolve strategy_name → run_id, creating the row if missing.

    The run_id is the strategy_name itself (stable, human-readable) — no
    separate surrogate needed. Stored in `strategies` for fast lookup and
    also as `paper_state.run_id`.
    """
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        raise StateCorruptionError("strategy_name must be a non-empty string")
    strategy_name = strategy_name.strip()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _ensure_all_tables(conn)
        row = conn.execute(
            "SELECT run_id FROM strategies WHERE strategy_name = ?", (strategy_name,)
        ).fetchone()
        if row is not None:
            return str(row[0])
        # create new
        run_id = strategy_name
        conn.execute(
            "INSERT OR REPLACE INTO strategies (strategy_name, run_id, instrument_id, created_at) VALUES (?, ?, ?, ?)",
            (strategy_name, run_id, instrument_id, int(pd.Timestamp.now(tz="UTC").value // 1_000_000_000)),
        )
        return run_id
    finally:
        conn.close()


def save_trades(
    db_path: str, run_id: str, trades_list: list[Any]
) -> None:
    """Upsert trade rows for run_id (from ube.core.ledger.trades)."""
    if not trades_list:
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _ensure_all_tables(conn)
        for t in trades_list:
            # t is ube.core.ledger.Trade
            conn.execute(
                "INSERT OR REPLACE INTO trades (run_id, instrument_id, entry_timestamp, exit_timestamp, entry_price, exit_price, side, quantity, status, entry_notional, exit_notional, gross_pnl, commission, funding, net_pnl, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    t.instrument_id,
                    int(t.entry_timestamp),
                    int(t.exit_timestamp),
                    float(t.entry_price),
                    float(t.exit_price),
                    int(t.side),
                    float(t.quantity),
                    str(t.status),
                    float(t.entry_notional),
                    float(t.exit_notional),
                    float(t.gross_pnl),
                    float(t.commission),
                    float(t.funding),
                    float(t.net_pnl),
                    t.exit_reason,
                ),
            )
    finally:
        conn.close()


def save_equity(
    db_path: str, run_id: str, equity_df: Any
) -> None:
    """Append equity points for run_id (expects DataFrame with timestamp/equity/returns)."""
    if equity_df is None or len(equity_df) == 0:
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _ensure_all_tables(conn)
        for _, r in equity_df.iterrows():
            # timestamp may be int ns or datetime
            raw_ts = r["timestamp"] if "timestamp" in r else r.get("timestamp")
            try:
                ts = int(pd.to_datetime(raw_ts, utc=True).value)  # ns
            except Exception:
                ts = int(raw_ts)  # already int
            conn.execute(
                "INSERT OR REPLACE INTO equity (run_id, timestamp, equity, returns) VALUES (?, ?, ?, ?)",
                (run_id, int(ts), float(r["equity"]), float(r["returns"])),
            )
    finally:
        conn.close()


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


def _aux_to_json(aux: dict[str, Any] | None) -> str | None:
    """Serialize aux_data (may contain MarketData or arrays) to JSON."""
    if not aux:
        return None
    out: dict[str, Any] = {}
    for k, v in aux.items():
        if isinstance(v, MarketData):
            # Store MarketData as dict of columns + index
            out[k] = {
                "_type": "MarketData",
                "open": v.open.tolist(),
                "high": v.high.tolist(),
                "low": v.low.tolist(),
                "close": v.close.tolist(),
                "volume": v.volume.tolist(),
                "index": v.index.astype("int64").tolist(),
            }
        elif isinstance(v, np.ndarray):
            out[k] = {"_type": "ndarray", "data": v.tolist()}
        else:
            out[k] = v
    return json.dumps(out)


def _aux_from_json(text: str | None) -> dict[str, Any]:
    """Deserialize aux_data from JSON, handling MarketData and ndarrays."""
    if text is None:
        return {}
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise StateCorruptionError(f"aux_data blob is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise StateCorruptionError("aux_data blob must be a dict")
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and v.get("_type") == "MarketData":
            out[k] = MarketData(
                open=np.array(v["open"], dtype=np.float64),
                high=np.array(v["high"], dtype=np.float64),
                low=np.array(v["low"], dtype=np.float64),
                close=np.array(v["close"], dtype=np.float64),
                volume=np.array(v["volume"], dtype=np.float64),
                index=pd.to_datetime(v["index"], unit="ns", utc=True),  # type: ignore[arg-type]
            )
        elif isinstance(v, dict) and v.get("_type") == "ndarray":
            out[k] = np.array(v["data"], dtype=np.float64)
        else:
            out[k] = v
    return out


class PaperState:
    """The resumable snapshot of a paper-trading run (§9.5).

    Attributes:
        instrument_id: The single traded instrument (paper trading is single-instrument,
            §6 of the plan).
        ledger: The append-only :class:`~ube.core.ledger.EventLedger` — the audit trail.
        last_processed_ns: The idempotency cursor; the timestamp of the last bar fed
            (``None`` before any bar, §9.6).
        last_price: The close of the last processed bar. Persisted so that equity can be
            derived on resume even though bars are not stored (plan blocker #5) — the open
            position is marked-to-market with this price. Full historical equity curve
            across gaps still requires the bar series (Point 9 — bars are not stored,
            only the last mark).
        open_position: The currently-open trade, or ``None`` when flat.
        pending_levels: Per-open-trade TP/SL (and scale-out) levels the backend needs to
            rebuild working orders on resume.
        signal_fn_state: State carried by a streaming signal function (§9.5); ``{}`` when
            stateless.
        config_ref: YAML dump of the canonical ``BacktestConfig`` at ``init`` time
            (via ``yaml.safe_dump(asdict(base))``). Stored for self-contained resume
            but *not* auto-applied on ``load()`` — caller must pass ``config`` to
            ``step()``/``run_auto()`` or reconstruct via ``get_config_dict()`` (Point 8).
        exit_seed: Minimal exit-level seed (extreme price + ATR warmup window) written by
            the backend at the end of each ``execute`` and re-applied on resume so trailing /
            ATR exits are not degenerate after a restart (issue C). ``None`` when flat or no
            path-dependent exit needs it.
    """

    def __init__(
        self,
        instrument_id: str,
        ledger: EventLedger,
        last_processed_ns: int | None = None,
        last_price: float | None = None,
        open_position: OpenPosition | None = None,
        pending_levels: dict[str, Any] | None = None,
        signal_fn_state: dict[str, Any] | None = None,
        config_ref: str | None = None,
        aux_data: dict[str, Any] | None = None,
        exit_seed: ExitSeed | None = None,
        last_funding_ns: int | None = None,
        run_id: str | None = None,
        db_path: str | None = None,
    ) -> None:
        self.instrument_id = instrument_id
        self.ledger = ledger
        self.last_processed_ns = last_processed_ns
        self.last_price = last_price
        self.open_position = open_position
        self.pending_levels: dict[str, Any] = pending_levels if pending_levels is not None else {}
        self.signal_fn_state: dict[str, Any] = (
            signal_fn_state if signal_fn_state is not None else {}
        )
        self.config_ref = config_ref
        self.aux_data: dict[str, Any] = aux_data if aux_data is not None else {}
        self.exit_seed: ExitSeed | None = exit_seed
        self.last_funding_ns: int | None = last_funding_ns
        self.run_id: str | None = run_id
        self.db_path: str | None = db_path

    def get_config_dict(self) -> dict[str, Any] | None:
        """Return the stored ``BacktestConfig`` as a dict, or ``None`` if absent.

        The dict is the ``yaml.safe_load`` of ``config_ref`` (the ``asdict`` of the
        canonical ``BacktestConfig`` at ``init`` time). It is *not* auto-applied
        on ``load()`` — caller must reconstruct ``BacktestConfig``/``PaperConfig``
        explicitly if they want a fully self-contained resume (see T8).
        """
        if self.config_ref is None:
            return None
        import yaml  # type: ignore[import-untyped]

        try:
            data = yaml.safe_load(self.config_ref)
        except Exception as exc:
            raise StateCorruptionError(f"config_ref is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise StateCorruptionError("config_ref YAML must be a mapping")
        return data

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
            _ensure_all_tables(conn)
            conn.execute(
                _INSERT_SQL,
                (
                    run_id,
                    self.instrument_id,
                    self.last_processed_ns,
                    self.last_price,
                    _ledger_to_json(self.ledger),
                    _json(asdict(self.open_position)) if self.open_position else None,
                    _json(self.pending_levels),
                    _json(self.signal_fn_state),
                    self.config_ref,
                    _aux_to_json(self.aux_data),
                    _json(asdict(self.exit_seed)) if self.exit_seed else None,
                    self.last_funding_ns,
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
        aux_text: str | None = None
        exit_seed_text: str | None = None
        last_funding_ns: int | None = None
        try:
            conn = sqlite3.connect(str(path), isolation_level=None)
            # Ensure aux_data column exists for old DBs
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE paper_state ADD COLUMN aux_data TEXT")
            # Ensure exit_seed column exists for the newest schema
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE paper_state ADD COLUMN exit_seed TEXT")
            # Ensure last_funding_ns column exists (funding clock)
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE paper_state ADD COLUMN last_funding_ns INTEGER")
            try:
                row = conn.execute(
                    "SELECT instrument_id, last_processed_ns, last_price, ledger, "
                    "open_position, pending_levels, signal_fn_state, config_ref, aux_data, "
                    "exit_seed, last_funding_ns "
                    "FROM paper_state "
                    "WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                try:
                    row = conn.execute(
                        "SELECT instrument_id, last_processed_ns, last_price, ledger, "
                        "open_position, pending_levels, signal_fn_state, config_ref, aux_data, "
                        "exit_seed "
                        "FROM paper_state "
                        "WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                except sqlite3.OperationalError:
                    try:
                        row = conn.execute(
                            "SELECT instrument_id, last_processed_ns, last_price, ledger, "
                            "open_position, pending_levels, signal_fn_state, config_ref, aux_data "
                            "FROM paper_state "
                            "WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()
                    except sqlite3.OperationalError:
                        row = conn.execute(
                            "SELECT instrument_id, last_processed_ns, last_price, ledger, "
                            "open_position, pending_levels, signal_fn_state, config_ref "
                            "FROM paper_state "
                            "WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()
            conn.close()
        except sqlite3.Error as exc:
            raise StateCorruptionError(f"could not read paper state at {path}: {exc}") from exc
        if row is None:
            raise StateCorruptionError(f"no paper state for run_id={run_id!r} in {path}")

        # Handle schema variants: newest (11 cols), 10 with aux+exit_seed, 9 with aux only, 8 base.
        (
            instrument_id,
            last_ns,
            last_price,
            ledger_text,
            open_text,
            pending_text,
            sfn_text,
            config_ref,
        ) = row[0:8]
        if len(row) > 8:
            aux_text = row[8]
        if len(row) > 9:
            exit_seed_text = row[9]
        if len(row) > 10:
            last_funding_ns = row[10]
        ledger = _ledger_from_json(ledger_text)
        open_position = None
        if open_text is not None:
            try:
                open_position = OpenPosition(**_from_json(open_text, {}))
            except TypeError as exc:
                raise StateCorruptionError(f"malformed open_position: {exc}") from exc
        exit_seed = None
        if exit_seed_text:
            try:
                exit_seed = ExitSeed(**_from_json(exit_seed_text, {}))
            except (TypeError, ValueError) as exc:
                raise StateCorruptionError(f"malformed exit_seed: {exc}") from exc
        obj = cls(
            instrument_id=instrument_id,
            ledger=ledger,
            last_processed_ns=last_ns,
            last_price=last_price,
            open_position=open_position,
            pending_levels=_from_json(pending_text, {}),
            signal_fn_state=_from_json(sfn_text, {}),
            config_ref=config_ref,
            aux_data=_aux_from_json(aux_text),
            exit_seed=exit_seed,
            last_funding_ns=last_funding_ns,
            run_id=run_id,
            db_path=path,
        )
        return obj

    def trade_table(self, db_path: str) -> Any:
        """Return the persisted trade table for this state's run_id (if any)."""
        # deferred import to avoid circular
        from ube.core.ledger import Trade

        # run_id is not stored on the instance; caller must supply db_path + run_id via load
        raise NotImplementedError("Use load_trades(db_path, run_id) instead")


def load_trades(db_path: str, run_id: str = "default") -> Any:
    """Load persisted trades for run_id as DataFrame (empty if none)."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _ensure_all_tables(conn)
        cur = conn.execute(
            "SELECT instrument_id, entry_timestamp, exit_timestamp, entry_price, exit_price, side, quantity, status, entry_notional, exit_notional, gross_pnl, commission, funding, net_pnl, exit_reason FROM trades WHERE run_id = ? ORDER BY entry_timestamp",
            (run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        # convert ns timestamps to datetime for convenience
        for c in ("entry_timestamp", "exit_timestamp"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], unit="ns", utc=True)
        return df
    finally:
        conn.close()


def load_equity(db_path: str, run_id: str = "default") -> Any:
    """Load persisted equity curve for run_id as DataFrame."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["timestamp", "equity", "returns"])
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _ensure_all_tables(conn)
        cur = conn.execute(
            "SELECT timestamp, equity, returns FROM equity WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "equity", "returns"])
        df = pd.DataFrame(rows, columns=["timestamp", "equity", "returns"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
        return df
    finally:
        conn.close()

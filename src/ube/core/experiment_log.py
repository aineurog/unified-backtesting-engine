"""Experiment log — sqlite-backed, content-addressable metadata for every run (§4.9).

Every backtest run, through any entry point, is *automatically* recorded: the resolved
params, the engine, a :class:`DataReference` identifying the exact data that was run
(instrument, bar type, date range, row count, and a SHA-256 ``content_hash``), the code
version, and an optional result hash. The log is a local embedded sqlite database (WAL
mode, §18) whose trial count — :meth:`ExperimentLog.count` — is the *true* number of
attempts that the overfitting-correction statistics (PBO/DSR, §11) depend on. That is
why logging is unconditional, not opt-in (§4.9: "logging happens as part of ``run()``").

**This module is separate from diagnostic logging (§15, §16).** It does *not* call
:mod:`logging` or ``print``; it writes only to sqlite. The standard diagnostic logger
(``logging.getLogger("ube")``) is a different concern and is unaffected by — and has no
effect on — what this module records.

**No ``ube.run()`` orchestrator exists in Phase 1.** The 9 Phase-1 items are the core
modules only; the orchestrator lands later. "Auto-log as part of ``run()``" is therefore
realized here as the :meth:`ExperimentLog.record` API that the future orchestrator calls
*unconditionally* (never gated behind an opt-in flag) at the end of every run.

**Path resolution precedence** (highest first):

1. Explicit ``ExperimentLog(path=...)``.
2. Environment variable ``BACKTEST_LOG_PATH``.
3. Default ``~/.backtest/experiments.db`` (``~`` expanded, parent auto-created).

**Schema** — a single ``experiments`` table, created if absent (SQLite types):

===============  =========  ===================================================
column           type       meaning
===============  =========  ===================================================
``run_id``       TEXT PK    the run's unique id (``BacktestResult.run_id``)
``timestamp``    TEXT       record time (ISO-8601 UTC, overrideable)
``engine``       TEXT       the engine adapter name
``params``       TEXT       the resolved config, JSON text (deterministic)
``instrument``   TEXT       ``Instrument.symbol``
``asset_class``  TEXT       ``Instrument.asset_class``
``bar_type``     TEXT       ``MarketData.bar_type``
``date_range_start`` TEXT   first bar timestamp (canonical string)
``date_range_end``   TEXT   last bar timestamp (canonical string)
``row_count``    INTEGER    number of bars
``content_hash`` TEXT       SHA-256 over head+tail of each OHLCV column
``code_version`` TEXT       installed package version (or an explicit override)
``result_hash``  TEXT       nullable — hash of ``.trades``/``.equity_curve`` (§16)
===============  =========  ===================================================

**content_hash algorithm** (:class:`DataReference`): SHA-256 over, for each OHLCV column
in order (open, high, low, close, volume), the column name plus the **first and last**
:data:`HASH_WINDOW` rows of that column, as raw little-endian float64 bytes. The window
is bounded — the hash never reads the whole array — so a change confined to the middle
rows (at unchanged length) is *not* caught by the hash; it *is* caught when the length
changes (``row_count``) or the change lands in the head/tail window. Deterministic given
identical data; the raw-bytes representation is platform-endian.

**Duplicate ``run_id``** — ``run_id`` is the primary key. A repeated :meth:`record` with
the same ``run_id`` is **ignored** (first write wins, ``INSERT OR IGNORE``): a retried
orchestration step re-recording the same run must not create a second row or overwrite
the first, keeping :meth:`count` honest.

**Thread/process safety (§18).** Each :class:`ExperimentLog` owns one sqlite connection
opened from its path. Separate processes (e.g. backtrader multiprocessing workers) should
each open their own :class:`ExperimentLog`; SQLite in WAL mode serializes concurrent
writers across connections.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from ube.core.config import BacktestConfig
from ube.core.data import BAR_TYPES, OHLCV_COLUMNS, MarketData
from ube.core.errors import (
    BacktestRuntimeError,
    ConfigError,
    DataShapeError,
    InvalidInstrumentError,
)
from ube.core.instrument import Instrument

__all__ = [
    "DataReference",
    "ExperimentRecord",
    "RecordInput",
    "ExperimentLog",
    "HASH_WINDOW",
    "DEFAULT_LOG_PATH",
]

#: The number of head and tail rows hashed per OHLCV column (§4.9 content_hash window).
HASH_WINDOW: int = 16

#: The default experiment-log path (§4.9); ``~`` is expanded at resolution time.
DEFAULT_LOG_PATH: str = "~/.backtest/experiments.db"

#: The environment variable overriding the default path (§4.9).
_ENV_VAR: str = "BACKTEST_LOG_PATH"

#: The distribution name read for ``code_version`` (matches ``pyproject.toml``).
_PACKAGE_NAME: str = "unified-backtesting-engine"

#: Column order used by both the schema and the read helpers (mirrors the docstring table).
_SELECT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "engine",
    "params",
    "instrument",
    "asset_class",
    "bar_type",
    "date_range_start",
    "date_range_end",
    "row_count",
    "content_hash",
    "code_version",
    "result_hash",
)

_COLUMNS_SQL: str = ", ".join(_SELECT_COLUMNS)

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id           TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    engine           TEXT NOT NULL,
    params           TEXT NOT NULL,
    instrument       TEXT NOT NULL,
    asset_class      TEXT NOT NULL,
    bar_type         TEXT NOT NULL,
    date_range_start TEXT NOT NULL,
    date_range_end   TEXT NOT NULL,
    row_count        INTEGER NOT NULL,
    content_hash     TEXT NOT NULL,
    code_version     TEXT NOT NULL,
    result_hash      TEXT
)
"""

_INSERT_SQL: str = (
    "INSERT OR IGNORE INTO experiments ("
    + _COLUMNS_SQL
    + ") VALUES ("
    + ", ".join("?" for _ in _SELECT_COLUMNS)
    + ")"
)


@dataclass(frozen=True)
class DataReference:
    """The content-addressable data reference for one run (§4.9).

    Identifies *exactly* which data was run: the instrument (symbol + asset class and
    its asset-class metadata), the bar type, the first/last bar timestamp, the row
    count, and a SHA-256 ``content_hash`` over a bounded head+tail window of each OHLCV
    column. Two users who load "the same" data that diverges (an exchange restated
    history, a file overwritten) get different hashes and can detect it immediately.

    Immutable (frozen, §3 principle 5); holds only scalars/strings/immutable tuples, so
    there are no arrays to copy.

    Attributes:
        instrument: The traded instrument — its ``symbol`` + ``asset_class`` are the
            identity recorded in the log (the remaining fields are asset-class metadata).
        bar_type: One of :data:`~ube.core.data.BAR_TYPES` (``"time"``/``"volume"``/
            ``"dollar"``/``"tick"``).
        date_range: ``(first_bar, last_bar)`` timestamp, each as a canonical string
            (ISO-8601 UTC for time bars, the decimal index for event bars).
        row_count: The number of bars.
        content_hash: SHA-256 hex digest over the head+tail of each OHLCV column.
    """

    instrument: Instrument
    bar_type: str
    date_range: tuple[str, str]
    row_count: int
    content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, Instrument):
            raise InvalidInstrumentError(
                "DataReference.instrument must be an Instrument; "
                f"got {type(self.instrument).__name__}"
            )
        if self.bar_type not in BAR_TYPES:
            raise ConfigError(f"bar_type={self.bar_type!r} is not one of {BAR_TYPES}")
        if not isinstance(self.date_range, tuple) or len(self.date_range) != 2:
            raise ConfigError("date_range must be a (start, end) 2-tuple of strings")
        if not all(isinstance(v, str) for v in self.date_range):
            raise ConfigError("date_range entries must be strings")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or (
            self.row_count < 0
        ):
            raise ConfigError(
                f"row_count must be a non-negative integer; got {self.row_count!r}"
            )
        if not isinstance(self.content_hash, str) or not self.content_hash:
            raise ConfigError("content_hash must be a non-empty string")

    @classmethod
    def from_market_data(
        cls, md: MarketData, instrument: Instrument
    ) -> DataReference:
        """Build a :class:`DataReference` from a :class:`~ube.core.data.MarketData`.

        Computes the ``content_hash`` (see the module docstring for the exact window and
        the byte representation) and the ``date_range`` from the market data's index. The
        ``instrument`` supplies the recorded identity (``symbol`` + ``asset_class``).

        Args:
            md: The canonical bar container for this run.
            instrument: The traded instrument (asset-class metadata, §4.5).

        Raises:
            DataShapeError: If ``md`` is not a ``MarketData`` or has zero bars.
            InvalidInstrumentError: If ``instrument`` is not an ``Instrument``.
        """
        if not isinstance(md, MarketData):
            raise DataShapeError("from_market_data expects a MarketData")
        if not isinstance(instrument, Instrument):
            raise InvalidInstrumentError("from_market_data expects an Instrument")
        if md.n_bars == 0:
            raise DataShapeError("cannot build a DataReference from an empty MarketData")
        start = _index_repr(md.index[0])
        end = _index_repr(md.index[md.n_bars - 1])
        return cls(
            instrument=instrument,
            bar_type=md.bar_type,
            date_range=(start, end),
            row_count=md.n_bars,
            content_hash=_compute_content_hash(md),
        )


@dataclass(frozen=True)
class ExperimentRecord:
    """One recorded experiment-log entry, as returned by :meth:`ExperimentLog.get` and
    :meth:`ExperimentLog.list` (a read-only projection of the ``experiments`` table).

    ``params`` is the raw JSON text (parse it with :func:`json.loads`); ``result_hash``
    is ``None`` when not recorded for the run.
    """

    run_id: str
    timestamp: str
    engine: str
    params: str
    instrument: str
    asset_class: str
    bar_type: str
    date_range_start: str
    date_range_end: str
    row_count: int
    content_hash: str
    code_version: str
    result_hash: str | None


@dataclass(frozen=True)
class RecordInput:
    """One pending experiment-log entry — the inputs to :meth:`ExperimentLog.record_many`.

    Mirrors :meth:`ExperimentLog.record`'s arguments so a batch (e.g. a vectorbt
    parameter grid, §18) can be buffered in memory and written in a single transaction.
    """

    run_id: str
    config: BacktestConfig
    engine: str
    data_reference: DataReference
    code_version: str | None = None
    result_hash: str | None = None
    timestamp: str | None = None


class ExperimentLog:
    """A thin wrapper over a sqlite3 database recording one row per run (§4.9, §18).

    Each instance owns one sqlite connection opened from its resolved path (see the
    module docstring for the precedence). The connection runs in WAL mode and the schema
    is created on first construction. Separate processes should open their own
    :class:`ExperimentLog`.

    Example::

        log = ExperimentLog()                       # -> ~/.backtest/experiments.db
        log.record(
            run_id=result.run_id,
            config=result.config,
            engine="vectorbt",
            data_reference=DataReference.from_market_data(md, instrument),
        )
        log.get(result.run_id)
        log.count()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = _resolve_path(path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BacktestRuntimeError(
                f"could not create experiment log directory {self._path.parent}"
            ) from exc
        try:
            # Autocommit mode: single statements are individually durable; batches use an
            # explicit BEGIN/COMMIT below, so behavior is version-independent.
            self._conn = sqlite3.connect(str(self._path), isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise BacktestRuntimeError(
                f"could not initialise experiment log at {self._path}: {exc}"
            ) from exc

    # -- recording -----------------------------------------------------------

    def record(
        self,
        *,
        run_id: str,
        config: BacktestConfig,
        engine: str,
        data_reference: DataReference,
        code_version: str | None = None,
        result_hash: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Insert one run into the log (one INSERT, one transaction — §18).

        The future orchestrator calls this *unconditionally* at the end of every run
        (§4.9); it is never opt-in. ``params`` is serialized deterministically from
        ``config``; ``code_version`` auto-detects the installed package version unless an
        override is supplied; ``timestamp`` defaults to the current UTC time.

        A duplicate ``run_id`` (a retried orchestration step re-recording the same run)
        is ignored — see the module docstring.

        Args:
            run_id: The run's unique id (``BacktestResult.run_id``).
            config: The resolved :class:`~ube.core.config.BacktestConfig`.
            engine: The engine adapter name (e.g. ``"vectorbt"``).
            data_reference: The :class:`DataReference` for the data that was run.
            code_version: Explicit code version override; ``None`` auto-detects via
                ``importlib.metadata`` with a ``"unknown"`` fallback.
            result_hash: Optional hash of ``.trades``/``.equity_curve`` (§16); computed
                by the orchestrator/regression layer, never here.
            timestamp: Record-time override (ISO-8601); defaults to now in UTC.

        Raises:
            ConfigError: If ``run_id``/``engine`` are empty, ``config``/``data_reference``
                are the wrong type, or the config cannot be serialized to JSON.
            BacktestRuntimeError: If the sqlite write fails.
        """
        row = self._build_row(
            RecordInput(
                run_id=run_id,
                config=config,
                engine=engine,
                data_reference=data_reference,
                code_version=code_version,
                result_hash=result_hash,
                timestamp=timestamp,
            )
        )
        self._execute(_INSERT_SQL, row)

    def record_many(self, records: Iterable[RecordInput]) -> None:
        """Record a batch of runs in a single transaction (§18 vectorbt batch).

        Every row is validated and serialized *before* the transaction opens, so a
        validation/``ConfigError`` on any record leaves the log untouched (atomic). A
        sqlite error mid-batch rolls the whole transaction back. Duplicate ``run_id``
        entries within or across batches are ignored (``INSERT OR IGNORE``), consistent
        with :meth:`record`.

        Args:
            records: The pending entries (buffered in memory by the caller).

        Raises:
            ConfigError: If any record is invalid (same conditions as :meth:`record`).
            BacktestRuntimeError: If the sqlite write fails.
        """
        rows = [self._build_row(r) for r in records]
        if not rows:
            return
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise BacktestRuntimeError(
                f"could not begin a transaction on experiment log at {self._path}: {exc}"
            ) from exc
        try:
            self._conn.executemany(_INSERT_SQL, rows)
        except sqlite3.Error as exc:
            self._conn.execute("ROLLBACK")
            raise BacktestRuntimeError(
                f"failed to record batch to experiment log at {self._path}: {exc}"
            ) from exc
        self._conn.execute("COMMIT")

    # -- query helpers (for the §20 CLI) -------------------------------------

    def get(self, run_id: str) -> ExperimentRecord | None:
        """Return the recorded entry for ``run_id``, or ``None`` if it is absent.

        Args:
            run_id: The run's unique id.

        Returns:
            The :class:`ExperimentRecord`, or ``None`` when ``run_id`` is not in the log.

        Raises:
            BacktestRuntimeError: If the sqlite read fails.
        """
        row = self._execute(
            f"SELECT {_COLUMNS_SQL} FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self, limit: int | None = None) -> tuple[ExperimentRecord, ...]:
        """Return recent entries, most recently recorded first.

        "Most recent" is reverse insertion order (``ORDER BY rowid DESC``) — the log is
        append-only, so this is the reverse chronological order.

        Args:
            limit: Optional maximum number of entries to return (``None`` = all).

        Returns:
            A tuple of :class:`ExperimentRecord`, newest first.

        Raises:
            ConfigError: If ``limit`` is not a non-negative integer or ``None``.
            BacktestRuntimeError: If the sqlite read fails.
        """
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ConfigError(
                f"limit must be a non-negative integer or None; got {limit!r}"
            )
        if limit is None:
            rows = self._execute(
                f"SELECT {_COLUMNS_SQL} FROM experiments ORDER BY rowid DESC"
            ).fetchall()
        else:
            rows = self._execute(
                f"SELECT {_COLUMNS_SQL} FROM experiments ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_row_to_record(r) for r in rows)

    def count(self) -> int:
        """Return the number of recorded runs (the true number of attempts, §11 PBO/DSR).

        Duplicate ``run_id`` values count once (they are a single attempt), because the
        table is keyed on ``run_id``.

        Raises:
            BacktestRuntimeError: If the sqlite read fails.
        """
        row = self._execute("SELECT COUNT(*) FROM experiments").fetchone()
        return int(row[0]) if row is not None else 0

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        self._conn.close()

    def __enter__(self) -> ExperimentLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise BacktestRuntimeError(
                f"experiment log operation failed at {self._path}: {exc}"
            ) from exc

    def _build_row(self, inp: RecordInput) -> tuple[object, ...]:
        """Validate one pending record and produce its sqlite parameter tuple."""
        run_id = _require_nonempty(inp.run_id, "run_id")
        engine = _require_nonempty(inp.engine, "engine")
        if not isinstance(inp.config, BacktestConfig):
            raise ConfigError("config must be a BacktestConfig")
        if not isinstance(inp.data_reference, DataReference):
            raise ConfigError("data_reference must be a DataReference")
        params = _config_to_json(inp.config)
        code_version = _resolve_code_version(inp.code_version)
        timestamp = inp.timestamp if inp.timestamp is not None else _utcnow_iso()
        dr = inp.data_reference
        return (
            run_id,
            timestamp,
            engine,
            params,
            dr.instrument.symbol,
            dr.instrument.asset_class,
            dr.bar_type,
            dr.date_range[0],
            dr.date_range[1],
            dr.row_count,
            dr.content_hash,
            code_version,
            inp.result_hash,
        )


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path | None) -> Path:
    """Resolve the log path per §4.9: explicit > env var > default, expanding ``~``."""
    if path is not None:
        candidate = str(path)
    else:
        env = os.environ.get(_ENV_VAR)
        candidate = env if env else DEFAULT_LOG_PATH
    return Path(candidate).expanduser()


def _index_repr(value: object) -> str:
    """Canonical string for a bar-index entry: ISO-8601 for timestamps, else ``str``."""
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _compute_content_hash(md: MarketData) -> str:
    """SHA-256 over the head+tail (``HASH_WINDOW`` rows) of each OHLCV column (§4.9).

    Each column's contribution is ``<column name bytes> + head bytes + tail bytes``, in
    fixed OHLCV order; the float64 arrays are hashed as raw little-endian bytes. The
    window is bounded (``2 * HASH_WINDOW`` rows read per column), so the hash never reads
    the whole array — a middle-only change at unchanged length is not detected (see the
    module docstring).
    """
    digest = hashlib.sha256()
    arrays: tuple[np.ndarray, ...] = (md.open, md.high, md.low, md.close, md.volume)
    for name, arr in zip(OHLCV_COLUMNS, arrays, strict=True):
        digest.update(name.encode("utf-8"))
        digest.update(arr[:HASH_WINDOW].tobytes())
        digest.update(arr[-HASH_WINDOW:].tobytes())
    return digest.hexdigest()


def _config_to_json(config: BacktestConfig) -> str:
    """Serialize a ``BacktestConfig`` to a deterministic JSON string (§4.9 ``params``).

    ``dataclasses.asdict`` recursively flattens the nested frozen sub-configs; a
    JSON-safe ``default`` handles the remaining non-primitive leaves (timestamps, numpy
    scalars, enums); ``sort_keys=True`` makes the output deterministic. The config is
    never mutated.
    """
    data: dict[str, Any] = asdict(config)
    try:
        return json.dumps(data, default=_json_default, sort_keys=True)
    except TypeError as exc:
        raise ConfigError(
            f"BacktestConfig contains a value that cannot be serialized to JSON: {exc}"
        ) from exc


def _json_default(value: object) -> object:
    """JSON ``default`` hook for the leaf values ``asdict`` leaves non-primitive."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot JSON-serialize value of type {type(value).__name__}")


def _resolve_code_version(override: str | None) -> str:
    """Resolve the code version: explicit override wins, else the installed version."""
    if override is not None:
        return override
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _row_to_record(row: Sequence[Any]) -> ExperimentRecord:
    """Project a raw sqlite row (in :data:`_SELECT_COLUMNS` order) to a record."""
    result_hash = row[12]
    return ExperimentRecord(
        run_id=row[0],
        timestamp=row[1],
        engine=row[2],
        params=row[3],
        instrument=row[4],
        asset_class=row[5],
        bar_type=row[6],
        date_range_start=row[7],
        date_range_end=row[8],
        row_count=row[9],
        content_hash=row[10],
        code_version=row[11],
        result_hash=result_hash if result_hash is None else str(result_hash),
    )

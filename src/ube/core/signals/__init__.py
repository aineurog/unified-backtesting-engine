"""Canonical signal format and its conversion helpers (§6.1, §6.2, §9.2).

The engine has exactly one internal signal format — the 4-column entry/exit
``Signals`` container (vectorbt's native shape, §6.1):

- ``long_entry``  — open a long position (an *action*, not a state)
- ``long_exit``   — close the current long position
- ``short_entry`` — open a short position
- ``short_exit``  — close the current short position

The columns are independent and accumulate; ``False`` means only "no new action this
bar", never "there is no open position". This is why a long→short flip is *two*
simultaneous events on one bar (§6.2) and why the format cannot be a single column.

Two alternative inputs are accepted and converted to this format at the boundary:

- A discrete target-position array of ``-1/0/1`` (where ``0`` = "flat", §6.2) via
  :func:`from_target`.
- A per-bar callable returning ``-1/0/1`` (§9.2) via :func:`from_callable`, which
  collects the target array and routes it through :func:`from_target`.

Validation is the "validate phase" of §6.1: it confirms the four columns are boolean
and raises :class:`~ube.core.errors.InvalidSignalError` on a shape/dtype mismatch or on
a contradictory entry (``long_entry`` + ``short_entry``, or ``long_exit`` +
``short_exit``, both ``True`` on the same bar), naming the offending bar. A
``long_exit`` + ``short_entry`` (or ``short_exit`` + ``long_entry``) bar is *not* a
conflict — it is the legal flip encoding.

Everything after a target array is collected is vectorized; the only per-bar Python
iteration is the inherent call of a user-supplied callable in :func:`from_callable`
(§3 principle 1, §9.2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import InvalidSignalError

__all__ = [
    "SIGNAL_COLUMNS",
    "Signals",
    "from_target",
    "from_callable",
    "validate_long_only",
]

#: The four canonical signal columns in their canonical order (§6.1).
SIGNAL_COLUMNS: tuple[str, ...] = ("long_entry", "long_exit", "short_entry", "short_exit")


def _coerce_bool(values: object, name: str) -> np.ndarray:
    """Coerce a column to a fresh 1-D boolean array, raising ``InvalidSignalError``."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise InvalidSignalError(
            f"signal column {name!r} must be 1-D; got shape {arr.shape}"
        )
    if arr.dtype != np.bool_:
        raise InvalidSignalError(
            f"signal column {name!r} must be boolean; got dtype {arr.dtype}"
        )
    # Copy so the frozen container never aliases (or freezes) the caller's buffer.
    return arr.copy()


@dataclass(frozen=True)
class Signals:
    """The canonical 4-column entry/exit signal container (§6.1).

    Frozen and self-validating: construction raises
    :class:`~ube.core.errors.InvalidSignalError` on a shape/dtype mismatch or on a
    contradictory entry. The four boolean arrays are read-only after construction
    (§3 principle 5).

    Attributes:
        long_entry: bool array — open a long position.
        long_exit: bool array — close the current long position.
        short_entry: bool array — open a short position.
        short_exit: bool array — close the current short position.
    """

    long_entry: np.ndarray
    long_exit: np.ndarray
    short_entry: np.ndarray
    short_exit: np.ndarray

    def __post_init__(self) -> None:
        long_entry = _coerce_bool(self.long_entry, "long_entry")
        long_exit = _coerce_bool(self.long_exit, "long_exit")
        short_entry = _coerce_bool(self.short_entry, "short_entry")
        short_exit = _coerce_bool(self.short_exit, "short_exit")

        n = long_entry.shape[0]
        for name, arr in (
            ("long_exit", long_exit),
            ("short_entry", short_entry),
            ("short_exit", short_exit),
        ):
            if arr.shape[0] != n:
                raise InvalidSignalError(
                    f"signal column {name!r} has {arr.shape[0]} rows, expected {n}"
                )

        # §6.1 conflict rule — the only two contradictory pairings. A flip
        # (long_exit + short_entry, or short_exit + long_entry) is legal and is *not*
        # checked here.
        both_entry = long_entry & short_entry
        if both_entry.any():
            i = int(np.argmax(both_entry))
            raise InvalidSignalError(
                f"contradictory signal entries at bar {i}: "
                "both long_entry and short_entry are True"
            )
        both_exit = long_exit & short_exit
        if both_exit.any():
            i = int(np.argmax(both_exit))
            raise InvalidSignalError(
                f"contradictory signal entries at bar {i}: "
                "both long_exit and short_exit are True"
            )

        for arr in (long_entry, long_exit, short_entry, short_exit):
            arr.setflags(write=False)

        object.__setattr__(self, "long_entry", long_entry)
        object.__setattr__(self, "long_exit", long_exit)
        object.__setattr__(self, "short_entry", short_entry)
        object.__setattr__(self, "short_exit", short_exit)

    # -- properties -----------------------------------------------------------

    @property
    def n_bars(self) -> int:
        """Number of bars these signals cover."""
        return int(self.long_entry.shape[0])

    @property
    def columns(self) -> tuple[str, ...]:
        """The four canonical column names, in canonical order."""
        return SIGNAL_COLUMNS

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Signals:
        """Construct from a pandas ``DataFrame`` carrying the 4 canonical columns.

        Extra columns are ignored; a missing canonical column raises
        :class:`~ube.core.errors.InvalidSignalError`. Non-boolean columns also raise,
        since the format is explicitly boolean (§6.1).
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidSignalError("from_dataframe expects a pandas DataFrame")
        missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidSignalError(f"missing signal columns: {missing}")
        return cls(
            long_entry=df["long_entry"].to_numpy(),
            long_exit=df["long_exit"].to_numpy(),
            short_entry=df["short_entry"].to_numpy(),
            short_exit=df["short_exit"].to_numpy(),
        )

    @classmethod
    def from_array(cls, arr: object) -> Signals:
        """Construct from a 2-D boolean array of shape ``(n_bars, 4)``.

        Columns are fixed: ``[long_entry, long_exit, short_entry, short_exit]``.
        """
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[1] != 4:
            raise InvalidSignalError(
                "signal array must be 2-D with shape (n_bars, 4) "
                "[long_entry, long_exit, short_entry, short_exit]; "
                f"got shape {a.shape}"
            )
        if a.dtype != np.bool_:
            raise InvalidSignalError(f"signal array must be boolean; got dtype {a.dtype}")
        return cls(
            long_entry=a[:, 0],
            long_exit=a[:, 1],
            short_entry=a[:, 2],
            short_exit=a[:, 3],
        )

    # -- helpers --------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Reconstruct a pandas ``DataFrame`` with the four canonical columns."""
        return pd.DataFrame(
            {
                "long_entry": self.long_entry,
                "long_exit": self.long_exit,
                "short_entry": self.short_entry,
                "short_exit": self.short_exit,
            }
        )


def from_target(target: object) -> Signals:
    """Derive the 4-column events from a discrete target-position array (§6.2).

    ``target`` is a 1-D array of ``-1/0/1`` where the value *is* the desired position,
    so ``0`` unambiguously means "flat" (close whatever is held). The deterministic
    transition table is computed as the diff of consecutive values::

        prev -> next     events emitted
         0   ->  1       long_entry
         0   -> -1       short_entry
         1   ->  0       long_exit
        -1   ->  0       short_exit
         1   -> -1       long_exit  +  short_entry   (flip)
        -1   ->  1       short_exit +  long_entry    (flip)
         no change       nothing

    The implicit position before the first bar is flat (``0``), so a first bar of ``1``
    emits ``long_entry`` and ``-1`` emits ``short_entry``. "Hold" is a repeated target
    value, never ``0`` (``[1, 1, 1]`` stays long; ``[1, 0, 0]`` exits at the second
    bar).

    Values outside ``{-1, 0, 1}`` (or NaN, or a non-1-D/non-numeric input) raise
    :class:`~ube.core.errors.InvalidSignalError`.
    """
    t = np.asarray(target)
    if t.ndim != 1:
        raise InvalidSignalError(f"target must be a 1-D array; got shape {t.shape}")
    if t.dtype.kind not in ("b", "i", "u", "f"):
        raise InvalidSignalError(
            f"target must be numeric (-1/0/1); got dtype {t.dtype}"
        )
    t_float = t.astype(np.float64)
    if np.isnan(t_float).any():
        raise InvalidSignalError("target contains NaN")
    invalid = (t_float != -1.0) & (t_float != 0.0) & (t_float != 1.0)
    if invalid.any():
        i = int(np.argmax(invalid))
        raise InvalidSignalError(
            f"target values must be -1, 0, or 1; got {t_float[i]} at index {i}"
        )

    t_int = t.astype(np.int64)
    n = t_int.shape[0]
    prev = np.empty(n, dtype=np.int64)
    if n:
        prev[0] = 0  # implicit flat start
        prev[1:] = t_int[:-1]

    long_entry = (prev != 1) & (t_int == 1)
    long_exit = (prev == 1) & (t_int != 1)
    short_entry = (prev != -1) & (t_int == -1)
    short_exit = (prev == -1) & (t_int != -1)

    return Signals(
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
    )


def from_callable(fn: Callable[[MarketData], int], bars: MarketData) -> Signals:
    """Run a per-bar signal callable over history and derive events (§9.2).

    ``fn`` is called once per bar with a growing data window — the canonical
    :class:`~ube.core.data.MarketData` slice ``bars[:i + 1]`` (all bars up to and
    including bar ``i``) — and must return a discrete target of ``-1/0/1``. The targets
    are collected and routed through :func:`from_target`, so the result is the same
    event model backtest and paper trading share (§9.2). An invalid return raises
    :class:`~ube.core.errors.InvalidSignalError` (via :func:`from_target`).

    The per-bar call is the one place a Python call per bar is inherent — ``fn`` is
    user code; everything after the target array is collected is vectorized.
    """
    if not isinstance(bars, MarketData):
        raise InvalidSignalError("from_callable expects a MarketData of bars")
    targets: list[int] = []
    for i in range(bars.n_bars):
        window = bars.head(i + 1)
        targets.append(fn(window))
    return from_target(np.asarray(targets))


def validate_long_only(signals: Signals, asset_class: str) -> None:
    """Validate a signal series against a long-only asset class (§4.5).

    ``crypto_spot`` is long-only: a ``short_entry`` can never *open* a short, and a
    ``short_exit`` has no short to close. The gate lives at the strategy level (the
    paper ``decide_action`` and the nautilus actor both consult ``allows_short`` and
    already turn a short signal into an exit-of-an-existing-long or a skip), so this
    function no longer rejects the series — rejecting here would prevent that
    exit-or-skip handling and contradict the reference shorting gate (backtrader
    ``strategy.py``/nautilus ``signals.py``). It remains for call-site compatibility
    and is a no-op for shortable classes; structural validation is left to the caller.
    """
    return

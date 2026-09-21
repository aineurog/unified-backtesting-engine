"""vectorbt paper-trading backend (§6/§7 of the vectorbt paper plan).

:class:`VbtPaperEngine` is a drop-in :class:`~ube.papertrading.core.PaperEngine` that drives
``vectorbt.from_signals`` as the execution substrate — exactly the way
:class:`~ube.papertrading.nautilus.backend.NautilusPaperEngine` drives Nautilus's sandbox.
Importing this module self-registers both the ``"vectorbt"`` engine and its
:class:`~ube.papertrading.vbt.state.VbtPaperState` state class.

Per ``step`` the engine:

1. Computes the replay window (:func:`~ube.papertrading.vbt.state.window_start_ns`) —
   the carried entry bar when a position is open, otherwise the last close bar (minus an
   ATR warmup) — and slices ``data``/``signals`` to it.
2. Encodes the position-change policy (§9.3) into the 4-column signals *before* the vbt
   call (``decide_action`` with ``sim_side`` seeded flat), skipping bars that carry no
   signal. This is the engine-agnostic decision logic — vbt output is never post-filtered.
3. Computes the pre-window checkpoint balance
   (:func:`~ube.papertrading.vbt.state.checkpoint_balance`) and seeds the vbt run's
   ``starting_balance`` with it (leveraged by the adapter). A carried position is therefore
   carried at its original size, not re-sized off current equity.
4. Runs :meth:`VectorbtAdapter.run` directly on the window (no ExperimentLog/benchmark
   side effects) and returns only events strictly after the persisted cursor — the same
   ``ts > last_processed_ns`` append rule the nautilus backend uses.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np

from ube.adapters.vectorbt_adapter.adapter import VectorbtAdapter
from ube.adapters.vectorbt_adapter.overrides import DEFAULT_STARTING_BALANCE
from ube.core.cost import resolve_cost_model
from ube.core.data import MarketData
from ube.core.errors import EngineError
from ube.core.instrument import Instrument, allows_short
from ube.core.risk.exits import ATRStop, ChandelierExit
from ube.core.signals import Signals
from ube.papertrading.core import (
    PaperEngine,
    apply_calendar_policy,
    decide_action,
    register_paper_engine,
    register_state_class,
)

from .state import (
    VbtPaperState,
    checkpoint_balance,
    last_close_ns,
    window_start_ns,
)

__all__ = ["VbtPaperEngine"]

_PERIOD_FALLBACK_NS = 60_000_000_000


def _max_atr_period(risk: Any) -> int:
    """The largest ATR lookback across the configured exits (0 when none use ATR)."""
    exits = getattr(risk, "exit", None) if risk is not None else None
    if not exits:
        return 0
    period = 0
    for cfg in exits:
        if isinstance(cfg, (ATRStop, ChandelierExit)):
            period = max(period, int(getattr(cfg, "period", 14)))
    return period


def _zero_at_or_before(signals: Signals, ts: np.ndarray, bound_ns: int) -> Signals:
    """Drop every signal at a bar ``ts <= bound_ns`` (the flat re-emit guard)."""
    keep = ts > int(bound_ns)
    return Signals(
        long_entry=signals.long_entry & keep,
        long_exit=signals.long_exit & keep,
        short_entry=signals.short_entry & keep,
        short_exit=signals.short_exit & keep,
    )


def _apply_policy(
    signals: Signals, *, allow_short: bool, policy: str
) -> Signals:
    """Encode the §9.3 position-change policy into fresh signal arrays.

    ``sim_side`` is seeded flat (``0``) — the window starts at the carried entry bar when a
    position is open, so the entry's own raw signal re-opens it. A bar whose four raw
    columns are all ``False`` is skipped (raw ``False`` means "no action", never "close").
    """
    n = signals.n_bars
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    sim_side = 0
    for i in range(n):
        a_le = bool(signals.long_entry[i])
        a_lx = bool(signals.long_exit[i])
        a_se = bool(signals.short_entry[i])
        a_sx = bool(signals.short_exit[i])
        if not (a_le or a_lx or a_se or a_sx):
            continue
        action = decide_action(
            sim_side,
            long_entry=a_le,
            long_exit=a_lx,
            short_entry=a_se,
            short_exit=a_sx,
            allow_short=allow_short,
            policy=policy,
        )
        if action == "hold":
            continue
        if action == "open_long":
            long_entry[i] = True
            sim_side = 1
        elif action == "open_short":
            short_entry[i] = True
            sim_side = -1
        elif action == "close":
            if sim_side == 1:
                long_exit[i] = True
            else:
                short_exit[i] = True
            sim_side = 0
        elif action == "reverse":
            if sim_side == 1:
                long_exit[i] = True
                short_entry[i] = True
            else:
                short_exit[i] = True
                long_entry[i] = True
            sim_side = -sim_side
    return Signals(
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
    )


class VbtPaperEngine(PaperEngine):
    """Runs one ``step`` window through :class:`VectorbtAdapter`."""

    def execute(
        self,
        *,
        state: Any,
        data: MarketData,
        signals: Signals,
        config: Any,
    ) -> list[Any]:
        instrument = config.base.instrument
        if not isinstance(instrument, Instrument):
            raise EngineError("vectorbt paper backend requires a canonical Instrument")
        filtered = apply_calendar_policy(data, signals, config, engine_label="vectorbt paper")
        if filtered is None:
            return []
        data, signals = filtered
        n = data.n_bars
        if n == 0:
            return []

        overrides = dict(config.base.engine_overrides) if config.base.engine_overrides else {}
        iid = state.instrument_id
        ts = np.asarray(data.timestamps.as_unit("ns").asi8, dtype=np.int64)  # type: ignore[attr-defined]
        period_ns = int(np.median(np.diff(ts))) if n > 1 else _PERIOD_FALLBACK_NS
        cursor = state.last_processed_ns
        open_pos = state.open_position
        if isinstance(state, VbtPaperState):
            last_close = state.last_close_ns()
        else:
            last_close = last_close_ns(state.ledger, iid)

        warmup_ns = _max_atr_period(config.base.risk) * period_ns
        if cursor is None:
            run_start_ns = int(ts[0])
        else:
            events = state.ledger.events
            run_start_ns = int(events[0].timestamp) if events else int(ts[0])
        ws = window_start_ns(
            state.ledger,
            iid,
            open_pos,
            last_processed_ns=cursor,
            warmup_ns=warmup_ns,
            run_start_ns=run_start_ns,
        )
        a = int(np.searchsorted(ts, ws, side="left"))
        if open_pos is not None and (a >= n or int(ts[a]) > int(open_pos.entry_ns)):
            raise EngineError(
                f"vectorbt window for the carried entry at {open_pos.entry_ns} starts at "
                f"{int(ts[a]) if a < n else 'past the slice'} — the entry bar is not in the "
                "supplied data (the caller must include it in the fetch window)"
            )
        if a >= n:
            return []

        win_ts = ts[a:]
        win_data = MarketData(
            open=data.open[a:],
            high=data.high[a:],
            low=data.low[a:],
            close=data.close[a:],
            volume=data.volume[a:],
            index=data.timestamps[a:],
        )
        win_sig = Signals(
            long_entry=signals.long_entry[a:],
            long_exit=signals.long_exit[a:],
            short_entry=signals.short_entry[a:],
            short_exit=signals.short_exit[a:],
        )
        if open_pos is None and last_close is not None:
            win_sig = _zero_at_or_before(win_sig, win_ts, last_close)

        policy = config.base.signal.on_opposite_signal
        if policy is None:
            raise EngineError(
                "vectorbt paper backend requires signal.on_opposite_signal (§9.3)"
            )
        encoded = _apply_policy(
            win_sig,
            allow_short=allows_short(instrument.asset_class),
            policy=str(policy),
        )
        if open_pos is not None:
            side = int(open_pos.side)
            entry_here = (
                bool(encoded.long_entry[0]) if side == 1 else bool(encoded.short_entry[0])
            )
            if not entry_here:
                # The persisted ledger is authoritative: a stateful/time-based signal
                # function (e.g. one driven by wall-clock minutes) cannot replay the
                # carried entry over a shortened window, so the recomputed signals will
                # not re-emit it. Re-open the carried side at the entry bar and drop any
                # conflicting bar-0 signals so vectorbt carries the trade forward, instead
                # of aborting the live worker (event-driven strategy still gets bars 1+
                # untouched).
                warnings.warn(
                    f"vectorbt paper: carried {'long' if side == 1 else 'short'} entered "
                    f"at {open_pos.entry_ns} was not re-emitted by the recomputed "
                    "window; carrying it from the persisted ledger (the signal function "
                    "is not recomputable over the window)",
                    stacklevel=3,
                )
                le = np.array(encoded.long_entry, dtype=bool)
                lx = np.array(encoded.long_exit, dtype=bool)
                se = np.array(encoded.short_entry, dtype=bool)
                sx = np.array(encoded.short_exit, dtype=bool)
                le[0] = side == 1
                lx[0] = False
                se[0] = side == -1
                sx[0] = False
                encoded = Signals(
                    long_entry=le,
                    long_exit=lx,
                    short_entry=se,
                    short_exit=sx,
                )

        starting_capital = (
            float(config.starting_balance)
            if config.starting_balance is not None
            else float(overrides.get("starting_balance", DEFAULT_STARTING_BALANCE))
        )
        cost_model = (
            config.base.cost_model
            if config.base.cost_model is not None
            else resolve_cost_model(instrument)
        )
        multiplier = (
            1.0
            if instrument.contract_multiplier is None
            else float(instrument.contract_multiplier)
        )
        if isinstance(state, VbtPaperState):
            checkpoint = state.checkpoint_balance(
                starting_capital, multiplier=multiplier, cost_model=cost_model
            )
        else:
            checkpoint = checkpoint_balance(
                state.ledger,
                iid,
                open_pos,
                starting_capital,
                multiplier=multiplier,
                cost_model=cost_model,
            )
        if not checkpoint > 0.0:
            raise EngineError(
                "vectorbt paper checkpoint balance must be > 0 to seed sizing; got "
                f"{checkpoint!r} (the persisted ledger and starting balance are "
                "inconsistent)"
            )

        vbt_overrides = dict(overrides)
        vbt_overrides["starting_balance"] = float(checkpoint)
        vbt_config = dataclasses.replace(
            config.base, engine="vectorbt", engine_overrides=vbt_overrides
        )
        result = VectorbtAdapter().run(win_data, encoded, vbt_config)

        cutoff = None if cursor is None else int(cursor)
        return [
            event
            for event in result.ledger.events
            if cutoff is None or int(event.timestamp) > cutoff
        ]


register_paper_engine("vectorbt", VbtPaperEngine)
register_state_class("vectorbt", VbtPaperState)

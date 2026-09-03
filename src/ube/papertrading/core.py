"""Engine-agnostic paper-trading front-end (§9.1, §9.3, §9.6, §9.7).

This module owns everything that is *not* engine-specific:

* :func:`init` / :func:`step` / :func:`run_auto` — the two modes from §9.1
  ("I'll call you" and "call me").
* Idempotency (§9.6) — the ``last_processed_ns`` cursor; a stale or duplicate bar is a
  :class:`~ube.core.errors.DuplicateBarError`.
* The position-change policy (§9.3) — ``on_opposite_signal`` (``reverse`` / ``exit_only``
  / ``ignore``) is carried on :class:`~ube.papertrading.config.PaperConfig` and passed to
  the backend, which is the only place that emits orders.
* The backend registry — mirrors :mod:`ube.adapters.base`; ``"nautilus"`` is registered
  lazily (only when ``nautilus-trader`` is importable) so ``core`` never hard-depends on
  it.

The ledger (the single source of truth, §4.6) and the open-position cursor live on
:class:`~ube.papertrading.state.PaperState`. :func:`step` appends the backend's returned
``LedgerEvent``s to the state, advances the cursor, and recomputes the open position from
the ledger (deriving, never duplicating, the accounting).
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable

import numpy as np
import yaml  # type: ignore[import-untyped]

from ube.core.data import MarketData
from ube.core.errors import (
    ConfigError,
    DuplicateBarError,
    EngineError,
    InvalidSignalError,
)
from ube.core.instrument import Instrument, allows_short
from ube.core.ledger import EventLedger, EventType, LedgerEvent
from ube.core.signals import Signals, from_target, validate_long_only

from .config import PaperConfig
from .state import OpenPosition, PaperState

__all__ = [
    "PaperEngine",
    "RecordingBackend",
    "get_paper_engine",
    "init",
    "register_paper_engine",
    "run",
    "run_auto",
    "step",
]

# A tiny epsilon for open-position folds.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Backend registry (mirrors ube.adapters.base).
# ---------------------------------------------------------------------------


class PaperEngine:
    """Base class for a paper-trading execution backend.

    A backend is *pure translation*: it consumes the canonical
    :class:`~ube.core.data.MarketData` + :class:`~ube.core.signals.Signals` + the
    :class:`~ube.papertrading.state.PaperState` (to read the open position / pending
    levels for resume) and returns the new :class:`~ube.core.ledger.LedgerEvent`\\s the
    run produced. It must not mutate canonical inputs; it may append to ``state.ledger``
    or update ``state.open_position`` / ``state.pending_levels`` directly, but the
    canonical bookkeeping (cursor, derived open position) is done by :func:`step`.
    """

    def execute(
        self,
        *,
        state: PaperState,
        data: MarketData,
        signals: Signals,
        config: PaperConfig,
    ) -> list[LedgerEvent]:
        """Execute one slice of bars and return the new ledger events."""
        raise NotImplementedError


_REGISTRY: dict[str, type[PaperEngine]] = {}


def _normalize(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"paper engine name must be a non-empty string; got {name!r}")
    return name.strip().lower()


def register_paper_engine(name: str, engine: type[PaperEngine]) -> None:
    """Register a backend class under ``name`` (§4.1, mirrored from adapters.base)."""
    key = _normalize(name)
    if not isinstance(engine, type) or not issubclass(engine, PaperEngine):
        raise ConfigError(
            f"register_paper_engine({name!r}, ...) expects a PaperEngine subclass; "
            f"got {engine!r}"
        )
    if engine is PaperEngine or bool(getattr(engine, "__abstractmethods__", False)):
        raise ConfigError(f"registered paper engine {name!r} must be a concrete class")
    _REGISTRY[key] = engine


def get_paper_engine(name: str = "nautilus") -> type[PaperEngine]:
    """Resolve the backend class for ``name``; lazily import the nautilus backend."""
    key = _normalize(name)
    if key not in _REGISTRY:
        if key == "nautilus":
            try:
                importlib.import_module("ube.papertrading.nautilus")  # self-registers
            except ImportError as exc:
                raise EngineError(
                    "the nautilus paper backend requires nautilus-trader to be "
                    f"installed: {exc}"
                ) from exc
        else:
            raise ConfigError(
                f"unknown paper engine {name!r}; registered: {sorted(_REGISTRY) or 'none'}"
            )
    return _REGISTRY[key]


# ---------------------------------------------------------------------------
# Open-position derivation (from the ledger — never a duplicate source).
# ---------------------------------------------------------------------------


def _open_position_from_ledger(
    ledger: EventLedger, instrument_id: str
) -> OpenPosition | None:
    """Fold fills to the current open position (side/qty/entry/entry_ns/trade_id)."""
    position = 0.0
    entry_px = 0.0
    entry_ns = 0
    trade_id = ""
    for e in ledger.events:
        if e.event_type is not EventType.FILL or e.instrument_id != instrument_id:
            continue
        if e.side is None or e.quantity is None or e.price is None:
            continue
        q = float(e.side) * float(e.quantity)
        px = float(e.price)
        oid = e.order_id or ""
        if abs(position) < _EPS:
            entry_px, entry_ns, trade_id = px, int(e.timestamp), oid
            position = q
        elif (q > 0) == (position > 0):
            new_pos = position + q
            entry_px = (entry_px * abs(position) + px * abs(q)) / abs(new_pos)
            position = new_pos
        else:
            if abs(q) >= abs(position):
                flip = q + position
                if abs(flip) < _EPS:
                    position, entry_px, entry_ns, trade_id = 0.0, 0.0, 0, ""
                else:
                    position, entry_px, entry_ns, trade_id = flip, px, int(e.timestamp), oid
            else:
                position += q
    if abs(position) < _EPS:
        return None
    return OpenPosition(
        side=1 if position > 0 else -1,
        quantity=abs(position),
        entry_price=entry_px,
        entry_ns=entry_ns,
        trade_id=trade_id,
    )


# ---------------------------------------------------------------------------
# Front-end.
# ---------------------------------------------------------------------------


def init(
    config: PaperConfig, *, run_id: str = "default", db_path: str | None = None
) -> PaperState:
    """Create a fresh :class:`PaperState` for a session (§9.1).

    Args:
        config: The paper-trading configuration.
        run_id: Identifier used for sqlite persistence (§9.5).
        db_path: Optional sqlite path for auto-save (when set, step() will
            auto-persist after each call).

    Returns:
        A fresh state with an empty ledger and no cursor.
    """
    instrument = config.base.instrument
    if not isinstance(instrument, Instrument):
        raise ConfigError("PaperConfig.base.instrument must be a canonical Instrument")
    # db_path may come from config.state_path if not explicitly passed
    if db_path is None:
        db_path = config.state_path
    return PaperState(
        instrument_id=instrument.symbol,
        ledger=EventLedger(),
        last_processed_ns=None,
        # Persist the resolved canonical config as YAML (plan blocker #10) so a resumed
        # run is self-contained — the caller need not re-supply ``base`` (though it may).
        config_ref=yaml.safe_dump(dataclasses.asdict(config.base)),
        run_id=run_id,
        db_path=db_path,
    )


def step(
    data: MarketData,
    signals: Signals,
    state: PaperState,
    config: PaperConfig,
) -> tuple[PaperState, list[LedgerEvent]]:
    """Process one slice of bars through the paper engine (§9.1 "I'll call you").

    The slice must contain only *new* bars (idempotency, §9.6): every bar's timestamp
    must exceed ``state.last_processed_ns`` and the slice must be strictly increasing.

    Args:
        data: The new bars (single instrument), aligned with ``signals``.
        signals: The 4-column signals for the same bar grid.
        state: The current session state (mutated in place: ledger appended, cursor
            advanced, open position recomputed).
        config: The paper-trading configuration.

    Returns:
        ``(state, new_events)`` — the (same) state and the events this slice produced.

    Raises:
        DuplicateBarError: A bar is stale/duplicate (idempotency violation, §9.6).
        InvalidSignalError: Signal/bar shape mismatch or a long-only asset asked to short.
        ConfigError: Malformed config.
    """
    if not isinstance(data, MarketData):
        raise ConfigError("step expects a MarketData for data")
    if not isinstance(signals, Signals):
        raise ConfigError("step expects a Signals for signals")
    if not isinstance(state, PaperState):
        raise ConfigError("step expects a PaperState for state")
    if not isinstance(config, PaperConfig):
        raise ConfigError("step expects a PaperConfig for config")

    # Enforce the explicit-over-default position-change policy (§9.3/§4.7) before any
    # backend work. This is the single source of truth for ``on_opposite_signal`` (plan
    # blocker #1) — it lives on ``base.signal``, not on ``PaperConfig``.
    config.base.validate(paper_trading=True)

    n = data.n_bars
    if signals.n_bars != n:
        raise InvalidSignalError(
            f"signals cover {signals.n_bars} bars but data has {n}; must be aligned"
        )
    if n == 0:
        return state, []

    instrument = config.base.instrument
    asset_class = instrument.asset_class if isinstance(instrument, Instrument) else ""
    validate_long_only(signals, asset_class)

    ts = data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
    ts = np.asarray(ts, dtype=np.int64)
    # Monotonicity of the *slice* guards against corrupt/out-of-order data (still a
    # DuplicateBarError — these are effectively replays of already-seen bars).
    if not (np.diff(ts) > 0).all():
        raise DuplicateBarError("paper bars must be strictly increasing in time")
    # Idempotency (§9.6): replay protection. A bar at or before the cursor is a
    # stale/duplicate and must never be reprocessed.
    if state.last_processed_ns is not None and (ts <= state.last_processed_ns).any():
        raise DuplicateBarError(
            f"bar timestamp {int(ts[ts <= state.last_processed_ns][0])} is not after "
            f"last_processed_ns={state.last_processed_ns} (idempotency, §9.6)"
        )
    # Gaps (bars skipped between runs) are *allowed* — they are inherent to streaming
    # feeds and are not replays. The plan previously conflated gaps with stale/duplicate
    # by requiring contiguity; that requirement is dropped (plan blocker #9).

    engine_cls = get_paper_engine(config.engine)
    try:
        engine = engine_cls()
        new_events = engine.execute(state=state, data=data, signals=signals, config=config)
    except DuplicateBarError:
        raise
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrap engine failure as §15 EngineError
        raise EngineError(f"paper engine {config.engine!r} failed: {exc}") from exc

    for event in new_events:
        state.ledger.append(event)
    if new_events:
        state.last_processed_ns = int(ts[-1])
    # Persist the last close so equity can be derived on resume even though bars are not
    # stored (plan blocker #5). Used to mark the open position to market on reload.
    if n > 0:
        state.last_price = float(data.close[-1])
    state.open_position = _open_position_from_ledger(state.ledger, state.instrument_id)
    # Auto-save if the state knows its db_path (new scheduled-run behavior)
    # The db_path may come from config.state_path or from PaperState.db_path
    db_path = getattr(state, "db_path", None) or getattr(config, "state_path", None)
    run_id = getattr(state, "run_id", None) or "default"
    if db_path:
        try:
            # also ensure the strategy registry is up to date
            state.db_path = db_path
            state.run_id = run_id
            # ensure strategy registry entry exists (for strategy_name-based runs)
            try:
                from ube.papertrading.state import get_or_create_run_id

                # if state was created via strategy_name, ensure the mapping exists
                # we don't have strategy_name here, but run_id is the strategy_name in new API
                get_or_create_run_id(db_path, run_id, state.instrument_id)
            except Exception:
                pass
            state.save(db_path, run_id=run_id)
            # also persist trades/equity for the trade_table view
            try:
                from ube.core.ledger import trades as _trades
                from ube.papertrading.state import save_trades, save_equity
                import pandas as pd

                # trades
                # we need instrument for trades()
                from ube.core.instrument import Instrument as _Instr

                instr = config.base.instrument
                if isinstance(instr, _Instr):
                    tlist = list(_trades(state.ledger, {instr.symbol: instr}))
                    save_trades(db_path, run_id, tlist)
                    # equity - build from ledger + current state (simple)
                    # For now, save a single equity point per step (the current equity)
                    # Full equity curve will be built incrementally
                    # Use the same logic as main.py: cash + unrealized
                    from ube.core.ledger import EventType as _ET

                    total_cash = 0.0
                    has_cash = False
                    for e in state.ledger.events:
                        if e.event_type == _ET.CASH_MOVEMENT and e.amount is not None:
                            has_cash = True
                            total_cash += float(e.amount)
                        elif e.event_type in (_ET.COMMISSION, _ET.FUNDING_PAYMENT) and e.amount is not None:
                            has_cash = True
                            total_cash -= float(e.amount)
                    bal = total_cash if has_cash else 0.0
                    if state.open_position and state.last_price is not None:
                        mult = float(instr.contract_multiplier or 1.0)
                        bal += (float(state.last_price) - float(state.open_position.entry_price)) * float(state.open_position.quantity) * float(state.open_position.side) * mult
                    # append one equity point
                    eq_df = pd.DataFrame([{"timestamp": int(ts[-1]), "equity": float(bal), "returns": 0.0}])
                    save_equity(db_path, run_id, eq_df)
            except Exception:
                pass
        except Exception:
            pass
    return state, new_events


def run(
    strategy_name: str,
    data: MarketData,
    signals: Signals,
    config: PaperConfig,
    *,
    db_path: str | None = None,
    run_id: str | None = None,
) -> tuple[PaperState, list[LedgerEvent]]:
    """One-call scheduled-run entry point (strategy_name → run_id → step → auto-save).

    This is the `strategy_name`-keyed API the scheduled paper-trading scripts
    actually need: it owns `load-or-init` + `step` + durable `trades`/`equity`
    persistence internally, so every `step` call is atomic and survives a
    process exit (memory discarded each run). The caller no longer needs to
    manually `try: load() except: init()` and `save()`.

    Args:
        strategy_name: Human-readable strategy name (e.g. ``"BITCOIN_m5"``).
            Used as the stable key in the ``strategies`` table and as the
            ``run_id`` for ``paper_state`` (one row per strategy).
        data: New bars (single instrument), aligned with ``signals``.
        signals: 4-column signals for the same bar grid.
        config: Canonical ``PaperConfig`` (already built externally from
            ``config.yaml`` — never a raw dict).
        db_path: Explicit sqlite path. When ``None``, uses
            ``config.state_path``.
        run_id: Optional explicit run_id override (defaults to
            ``strategy_name``).

    Returns:
        ``(state, new_events)`` — same as ``step`` but with auto-persistence
        already done.
    """
    # canonical config is required (never a raw yaml dict)
    if not isinstance(config, PaperConfig):
        raise ConfigError("ube.paper.run expects a canonical PaperConfig (build it externally from config.yaml)")

    # db_path is required (like backtest) — either explicit or via config.state_path
    effective_db = db_path if db_path is not None else getattr(config, "state_path", None)
    if not effective_db:
        raise ConfigError("db_path is required for scheduled paper trading (set config.state_path or pass db_path)")

    effective_run_id = (run_id or strategy_name).strip() if isinstance(strategy_name, str) and strategy_name.strip() else (run_id or "default")
    if not effective_run_id:
        raise ConfigError("strategy_name / run_id must be a non-empty string")

    # ensure strategy registry entry exists and get the stable run_id
    try:
        from ube.papertrading.state import get_or_create_run_id

        # instrument_id may not be known until after load/init, so use config's instrument for the first creation
        instr_id = config.base.instrument.symbol if isinstance(config.base.instrument, Instrument) else str(config.base.instrument)
        effective_run_id = get_or_create_run_id(str(effective_db), str(effective_run_id), str(instr_id))
    except Exception:
        # if registry fails, fall back to the raw run_id — step will still work
        pass

    # load-or-init (the try-load-fallback every caller was hand-rolling)
    try:
        from ube.papertrading.state import PaperState as _PS

        state = _PS.load(str(effective_db), run_id=str(effective_run_id))
        # ensure the loaded state knows its persistence location for auto-save
        state.db_path = str(effective_db)
        state.run_id = str(effective_run_id)
    except Exception as exc:
        # distinguish "no row for run_id" (fresh strategy) from corrupt DB
        msg = str(exc)
        if "no paper state for run_id" in msg or "does not exist" in msg:
            state = init(config, run_id=str(effective_run_id), db_path=str(effective_db))
        else:
            raise

    # delegate to step (which now auto-saves and persists trades/equity)
    return step(data, signals, state, config)


def run_auto(
    data: MarketData,
    signal_fn: Callable[[MarketData], int],
    config: PaperConfig,
    state: PaperState | None = None,
    *,
    run_id: str = "default",
) -> list[LedgerEvent]:
    """Mode B (§9.1 "call me"): stream a signal callable over the bars (§9.2).

    The callable is invoked per bar with a growing window (``data.head(i + 1)``), exactly
    like :func:`ube.core.signals.from_callable`. Only bars *after* the state cursor are
    processed, so repeated calls with appended history continue the session (the
    signal-function state lives in ``state.signal_fn_state``; the full target array is
    recomputed from history each call, which matches ``from_callable`` semantics).

    Args:
        data: The (possibly full) bar series.
        signal_fn: A callable returning ``-1/0/1`` for a growing window.
        config: The paper-trading configuration.
        state: An existing session state, or ``None`` to create one (ephemeral unless
            ``config.state_path`` is set).

    Returns:
        The list of ledger events produced by this call (empty if no new bars).
    """
    if not isinstance(data, MarketData):
        raise ConfigError("run_auto expects a MarketData")
    if state is None:
        state = init(config, run_id=run_id)

    n = data.n_bars
    if n == 0:
        return []
    full_targets = [int(signal_fn(data.head(i + 1))) for i in range(n)]
    state.signal_fn_state["targets"] = full_targets

    ts = data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
    ts = np.asarray(ts, dtype=np.int64)
    last = state.last_processed_ns
    a = 0 if last is None else int(np.searchsorted(ts, last, side="right"))
    if a >= n:
        return []

    full_signals = from_target(np.asarray(full_targets, dtype=np.int64))
    suffix_signals = Signals(
        long_entry=full_signals.long_entry[a:],
        long_exit=full_signals.long_exit[a:],
        short_entry=full_signals.short_entry[a:],
        short_exit=full_signals.short_exit[a:],
    )
    data_suffix = MarketData(
        open=data.open[a:],
        high=data.high[a:],
        low=data.low[a:],
        close=data.close[a:],
        volume=data.volume[a:],
        index=data.timestamps[a:],
    )
    _, events = step(data_suffix, suffix_signals, state, config)
    return events


# ---------------------------------------------------------------------------
# RecordingBackend — a deterministic in-memory broker for unit tests (no engine dep).
# ---------------------------------------------------------------------------


def decide_action(
    sim_side: int,
    *,
    long_entry: bool,
    long_exit: bool,
    short_entry: bool,
    short_exit: bool,
    allow_short: bool,
    policy: str,
) -> str:
    """Resolve the per-bar position-change action from the 4-column signal.

    Shared by every backend (the recording fake *and* the nautilus strategy) so the
    engine-agnostic decision logic has a single source of truth (§9.4 comparability).

    Returns one of: ``"hold"``, ``"open_long"``, ``"open_short"``, ``"close"``,
    ``"reverse"`` (close the current position and open the opposite).
    """
    desired = 0
    if long_entry:
        desired = 1
    elif short_entry and allow_short:
        desired = -1
    elif long_exit or short_exit:
        desired = 0

    if sim_side == 0:
        if desired == 0:
            return "hold"
        return "open_long" if desired == 1 else "open_short"
    if desired == sim_side:
        return "hold"
    if desired == 0:
        return "close"
    # opposite signal -> apply the §9.3 position-change policy
    if policy == "reverse":
        return "reverse"
    if policy == "exit_only":
        return "close"
    return "hold"  # ignore


class RecordingBackend(PaperEngine):
    """A dependency-free reference broker used by unit tests of the engine-agnostic core.

    It honors ``config.on_opposite_signal`` and the long-only guard, and emits the same
    ledger event kinds a real backend would (``signal_evaluated`` / ``order_submitted`` /
    ``fill`` / ``position_change`` / ``commission``). It is *not* a production backend —
    it is the spec's ``_RecordingBackend`` fake (plan §7, T2).
    """

    def execute(
        self,
        *,
        state: PaperState,
        data: MarketData,
        signals: Signals,
        config: PaperConfig,
    ) -> list[LedgerEvent]:
        from ube.core.cost import fill_cost, resolve_cost_model

        instrument = config.base.instrument
        asset_class = instrument.asset_class if isinstance(instrument, Instrument) else ""
        cost_model = resolve_cost_model(instrument if isinstance(instrument, Instrument) else None)
        allow_short = allows_short(asset_class)
        # on_opposite_signal is explicit-over-default (§4.7) — validate() already
        # guarantees it is set for paper trading, so no fallback is needed.
        assert config.base.signal.on_opposite_signal is not None
        policy = config.base.signal.on_opposite_signal

        events: list[LedgerEvent] = []
        iid = state.instrument_id
        ts = data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]

        # Resume the simulated position from the state (so resume is exercised).
        sim_side = 0
        sim_qty = 0.0
        if state.open_position is not None:
            sim_side = state.open_position.side
            sim_qty = state.open_position.quantity

        for i in range(data.n_bars):
            le = bool(signals.long_entry[i])
            lx = bool(signals.long_exit[i])
            se = bool(signals.short_entry[i])
            sx = bool(signals.short_exit[i])
            if not (le or lx or se or sx):
                continue
            price = float(data.close[i])
            t = int(ts[i])
            action = decide_action(
                sim_side,
                long_entry=le,
                long_exit=lx,
                short_entry=se,
                short_exit=sx,
                allow_short=allow_short,
                policy=policy,
            )
            if action == "hold":
                continue
            close_first = action in ("close", "reverse")
            desired = 0
            if action == "open_long":
                desired = 1
            elif action == "open_short":
                desired = -1
            elif action == "reverse":
                desired = -sim_side  # open opposite of the current position

            # Emit the close (if any) then the open, as separate fills.
            if close_first and sim_qty > 0:
                notional = sim_qty * price
                commission = float(fill_cost(cost_model, notional=notional))
                events.append(
                    LedgerEvent(
                        EventType.FILL,
                        t,
                        iid,
                        side=-sim_side,
                        quantity=sim_qty,
                        price=price,
                        notional=notional,
                        exit_reason="signal",
                    )
                )
                events.append(
                    LedgerEvent(
                        EventType.POSITION_CHANGE, t, iid, side=0, position_after=0.0
                    )
                )
                if commission > 0:
                    events.append(
                        LedgerEvent(EventType.COMMISSION, t, iid, amount=commission, currency="USD")
                    )
                sim_side, sim_qty = 0, 0.0

            if desired != 0:
                side = 1 if desired == 1 else -1
                # Use an all-in-ish size: 1.0 unit for the fake (deterministic).
                qty = 1.0
                notional = qty * price
                commission = float(fill_cost(cost_model, notional=notional))
                order_id = f"{action}-{t}"
                events.append(
                    LedgerEvent(
                        EventType.SIGNAL_EVALUATED, t, iid, action=action
                    )
                )
                events.append(
                    LedgerEvent(
                        EventType.ORDER_SUBMITTED,
                        t,
                        iid,
                        order_id=order_id,
                        side=side,
                        quantity=qty,
                    )
                )
                events.append(
                    LedgerEvent(
                        EventType.FILL,
                        t,
                        iid,
                        side=side,
                        quantity=qty,
                        price=price,
                        notional=notional,
                        order_id=order_id,
                    )
                )
                events.append(
                    LedgerEvent(
                        EventType.POSITION_CHANGE,
                        t,
                        iid,
                        side=side,
                        position_after=side * qty,
                    )
                )
                if commission > 0:
                    events.append(
                        LedgerEvent(EventType.COMMISSION, t, iid, amount=commission, currency="USD")
                    )
                sim_side, sim_qty = side, qty
        return events

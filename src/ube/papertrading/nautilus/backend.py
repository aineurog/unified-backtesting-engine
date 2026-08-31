"""Nautilus paper backend (plan T3 / §4.3).

:class:`NautilusPaperEngine` drives a real Nautilus ``TradingNode`` whose
``SandboxExecutionClient`` executes MARKET orders against ube ``MarketData`` bars. The
strategy bridges fills into ube ``LedgerEvent``s returned to ``core.step``. Nautilus is the
*execution substrate* (§0) — all decisions/sizing use ``core`` (no re-implementation).

Importing this module self-registers the ``"nautilus"`` backend, so ``core`` never hard-depends
on nautilus-trader (lazy import, A5).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from ube.adapters.nautilus_adapter.adapt_data import build_bar_type
from ube.adapters.nautilus_adapter.instrument_map import build_instrument
from ube.core.cost import resolve_cost_model
from ube.core.errors import EngineError
from ube.core.instrument import Instrument, allows_short
from ube.core.ledger import EventType, LedgerEvent
from ube.papertrading.core import PaperEngine, register_paper_engine

from .node import build_node, run_node
from .runtime import reset_ready_event
from .signals import SIGNAL_REGISTRY
from .strategy import UbePaperConfig, UbePaperStrategy

if TYPE_CHECKING:
    from ube.core.data import MarketData
    from ube.core.signals import Signals
    from ube.papertrading.config import PaperConfig
    from ube.papertrading.state import PaperState


class NautilusPaperEngine(PaperEngine):
    """Drives a Nautilus ``SandboxExecutionClient`` for one ``step`` slice."""

    def execute(
        self,
        *,
        state: PaperState,
        data: MarketData,
        signals: Signals,
        config: PaperConfig,
    ) -> list[LedgerEvent]:

        canonical = config.base.instrument
        asset_class = canonical.asset_class if isinstance(canonical, Instrument) else ""
        overrides = dict(config.base.engine_overrides) if config.base.engine_overrides else {}
        no_short = not allows_short(asset_class)
        # Nautilus closes its event loop on ``node.dispose()``; a second ``step`` in the same
        # process (e.g. a second integration test) would otherwise hit "Event loop is closed".
        # Give every run a fresh live loop before the node is built.
        try:
            _loop = asyncio.get_event_loop()
        except RuntimeError:
            _loop = None
        if _loop is None or _loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            build = build_instrument(canonical, overrides=overrides)
            instrument = build.instrument
            instrument_id = instrument.id
            venue = str(instrument_id.venue)

            cost_model = (
                config.base.cost_model
                if config.base.cost_model is not None
                else resolve_cost_model(canonical)
            )

            ts = data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
            ts = np.asarray(ts, dtype=np.int64)
            n = data.n_bars
            period_ns = int(np.median(np.diff(ts))) if n > 1 else 60_000_000_000
            # Bars are published to the sandbox with **live-clock** timestamps. The sandbox
            # execution client advances its own (Test) clock by each bar's ``ts_init`` and
            # matches an order only when the clock has passed the order's ``ts_init`` — and
            # the order factory stamps orders with a *read-only* live clock we cannot
            # override (it's a C-level slot in nautilus 1.221). So the bars must sit on the
            # live timeline, strictly after "now", for every order to match the bar it was
            # submitted against. The historical (test-clock) ts is preserved separately via
            # ``signal_map`` so the ledger/trades stay comparable (§9.4).
            live_base = time.time_ns() + 10_000_000_000  # +10s headroom, ns precision
            bars = []
            signal_map = {}
            for i in range(n):
                t_live = int(live_base + i * period_ns)
                t_hist = int(ts[i])
                bars.append(
                    {
                        "ts_ns": t_live,
                        "open": float(data.open[i]),
                        "high": float(data.high[i]),
                        "low": float(data.low[i]),
                        "close": float(data.close[i]),
                        "volume": float(data.volume[i]),
                    }
                )
                signal_map[t_live] = (
                    bool(signals.long_entry[i]),
                    bool(signals.long_exit[i]),
                    bool(signals.short_entry[i]),
                    bool(signals.short_exit[i]),
                    t_hist,
                )

            bar_type = build_bar_type(instrument_id, period_ns)

            starting_balance = float(
                config.starting_balance
                if config.starting_balance is not None
                else overrides.get("starting_balance", 100_000.0)
            )
            balance = starting_balance
            if state.ledger.events:
                total_cash = 0.0
                has_cash = False
                for e in state.ledger.events:
                    if e.event_type is EventType.CASH_MOVEMENT and e.amount is not None:
                        has_cash = True
                        total_cash += float(e.amount)
                    elif (
                        e.event_type in (EventType.COMMISSION, EventType.FUNDING_PAYMENT)
                        and e.amount is not None
                    ):
                        has_cash = True
                        total_cash -= float(e.amount)
                if has_cash:
                    balance = total_cash

            # Leverage: mirror backtest's sizing * margin logic — sizing leverage
            # dominates, override is fallback, cash accounts force 1.0 (§3.2).
            sizing_lev = 1.0
            try:
                sizing_lev = float(getattr(config.base.risk.sizing, "leverage", 1.0))
            except Exception:
                sizing_lev = 1.0
            override_lev = float(overrides.get("leverage", 0.0))
            if no_short:
                leverage = 1.0
            else:
                lev = max(sizing_lev, override_lev)
                leverage = lev if lev > 0 else 1.0
            multiplier = (
                1.0
                if canonical.contract_multiplier is None
                else float(canonical.contract_multiplier)
            )
            # Funding: per-period rates from cost model + calendar interval.
            from ube.core.instrument import resolve_funding_interval_hours

            funding_rate = float(getattr(cost_model, "funding", 0.0) or 0.0) if cost_model else 0.0
            borrow_rate = float(getattr(cost_model, "borrow", 0.0) or 0.0) if cost_model else 0.0
            funding_interval_hours = resolve_funding_interval_hours(canonical)
            interval_ns = int(funding_interval_hours * 3600 * 1_000_000_000)
            quote = str(getattr(instrument, "settlement_currency", "USDT"))

            # Reset per-run singletons — TradingNode closes its loop on dispose
            # and UbeDataClient.DONE is a class-level Event that remains set
            # after the previous run; without resetting, the next run's
            # _stop_when_done sees an already-set event and stops before
            # draining any bars.
            from .data_client import UbeDataClient

            UbeDataClient.DONE = None
            reset_ready_event()
            strat_cfg = UbePaperConfig(
                instrument_id=str(instrument_id),
                bar_type=str(bar_type),
                sizing=config.base.risk.sizing,
                cost_model=cost_model,
                no_short=no_short,
                on_opposite_signal=str(config.base.signal.on_opposite_signal),
                balance=balance,
                leverage=leverage,
                multiplier=multiplier,
                exits=tuple(config.base.risk.exit) if config.base.risk.exit else (),
                funding_rate=funding_rate,
                borrow_rate=borrow_rate,
                funding_interval_ns=interval_ns,
                open_position=state.open_position,
            )
            strategy = UbePaperStrategy(config=strat_cfg)

            node = build_node(
                instrument=instrument,
                bars=bars,
                signal_map=signal_map,
                bar_type=str(bar_type),
                balance=balance,
                quote=quote,
                venue=venue,
                leverage=leverage,
                strategy=strategy,
                overrides=overrides,
                open_position=state.open_position,
            )

            SIGNAL_REGISTRY.clear()
            run_node(node)
            events = list(strategy.events)
            # Starting balance booked as a cash inflow at the first bar boundary (§4.6
            # step 4 — the cash leg of the equity curve). Emitted once per session (only
            # on the first ``step`` slice, when the cursor has not advanced yet).
            if state.last_processed_ns is None and len(ts) > 0:
                events.insert(
                    0,
                    LedgerEvent(
                        EventType.CASH_MOVEMENT,
                        int(ts[0]),
                        strategy._iid,
                        amount=balance,
                        currency=quote,
                    ),
                )
            return events
        except Exception as exc:  # pragma: no cover - defensive
            raise EngineError(
                f"nautilus paper backend failed: {exc}"
            ) from exc


register_paper_engine("nautilus", NautilusPaperEngine)

__all__ = ["NautilusPaperEngine"]

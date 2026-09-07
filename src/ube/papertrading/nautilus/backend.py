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
from typing import TYPE_CHECKING, Any

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


def _instrument_currency(instrument: Any) -> str:
    """Account/quote currency for the sandbox, taken from the nautilus instrument.

    The sandbox has no quote feed for conversion pairs, so the account MUST be
    denominated in the instrument's own currency: every fill's commission and
    PnL is booked in that currency (``MakerTakerFeeModel`` uses
    ``quote_currency``), and the account manager converts to base on each fill.
    Any mismatch raises ``insufficient data for USD/USDT`` on every fill
    (``Quote maps must not be empty``). Perp/spot/forex pairs expose
    ``quote_currency``; futures/equity only expose ``currency`` (no
    ``settlement_currency`` attr — a ``getattr(..., "settlement_currency",
    "USDT")`` fallback silently built a USDT account against a USD instrument).
    With base == activity currency, ``get_xrate`` short-circuits to 1.0
    (``from == to``) and no FX lookup — synthetic or otherwise — is ever needed.
    """
    for attr in ("quote_currency", "currency", "settlement_currency"):
        cur = getattr(instrument, attr, None)
        if cur is not None:
            return str(cur)
    return "USDT"


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
        # Explicit account-type mapping (fix 4): long-only (crypto_spot, stocks) => CASH
        # in a true spot model, margined (crypto_perp, futures, etc.) => MARGIN.
        # For paper trading, spot currently retains MARGIN to allow phantom closes on
        # resume (sandbox not seeded — issue A deferred). True CASH would reject a
        # reduce_only close without a real position.
        if "account_type" not in overrides:
            # Ideal: "cash" if no_short else "margin". Keep MARGIN for no_short until seeding.
            overrides["account_type"] = "margin"
            if no_short:
                import warnings

                warnings.warn(
                    "Spot/stocks paper trading defaulting to MARGIN account for resume "
                    "compatibility (sandbox not seeded with open position). For true "
                    "CASH spot, set engine_overrides.account_type='cash'.",
                    stacklevel=2,
                )
        # Warn if resuming a CASH spot with open position (will be rejected without seeding)
        _acct_tmp = overrides.get("account_type", "margin")
        _acct_type_tmp = "MARGIN" if _acct_tmp == "margin" else "CASH"
        if no_short and state.open_position is not None and _acct_type_tmp == "CASH":
            import warnings

            warnings.warn(
                f"Resuming {asset_class!r} (CASH) with open_position {state.open_position} "
                "but sandbox is not seeded — reduce_only close may be rejected. "
                "Consider using MARGIN for paper spot or implementing position seeding.",
                stacklevel=2,
            )
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
            multiplier = (
                1.0
                if canonical.contract_multiplier is None
                else float(canonical.contract_multiplier)
            )
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
            # Resume sizing always seeds the strategy with the *pure cash book* —
            # never cash + open-position mark. ``_current_balance`` is cash-only:
            # each fill books a ±notional cash leg (§4.6), and a same-bar reversal
            # zeroes _sim_side/_sim_qty and applies the optimistic close-credit
            # *before* sizing, so at the exact ``_size_qty`` point cash ≈ post-close
            # equity. Marking the open position here double-counts the notional one
            # bar later: a short open credits ~+notional to cash, the resume re-adds
            # the negative mark on top, and the reversal's close-credit then books the
            # closing notional a second time — equity collapses ~-90k on a 10k account
            # ("capital must be non-negative" crash in live paper-trading, 07:55 bar).
            # An uninterrupted run has _current_balance = 10k + short credit (never a
            # mark), so cash-only seeding is exactly what keeps split-run sizing
            # identical to a single run over the same bars (Issue B).

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
            # Funding: per-period rates from cost model + calendar interval.
            from ube.core.instrument import resolve_funding_interval_hours

            funding_rate = float(getattr(cost_model, "funding", 0.0) or 0.0) if cost_model else 0.0
            borrow_rate = float(getattr(cost_model, "borrow", 0.0) or 0.0) if cost_model else 0.0
            funding_interval_hours = resolve_funding_interval_hours(canonical)
            interval_ns = int(funding_interval_hours * 3600 * 1_000_000_000)
            # Account currency == instrument currency (see _instrument_currency):
            # the sandbox has no FX feed, so any mismatch (e.g. a USDT account
            # against a USD-quoted instrument) fails every fill's account-state
            # update with "insufficient data for USD/USDT".
            quote = _instrument_currency(instrument)

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
                exit_seed=state.exit_seed,
                bar_period_ns=period_ns,
                last_funding_ns=state.last_funding_ns,
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
            )

            SIGNAL_REGISTRY.clear()
            run_node(node)
            events = list(strategy.events)
            # Issue C: persist the exit seed (minimal trailing/ATR statistics) so a future
            # resume re-seeds exit-level computation instead of starting degenerate. Written
            # by the backend because it owns the strategy lifecycle; ``step`` does not see the
            # strategy's internal state. Cleared automatically when the position is flat.
            state.exit_seed = strategy.exit_seed()
            # Issue 2: persist the funding clock so a resume continues from the exact
            # saved point (not from bar spacing, which is contaminated by synthetic seed bars).
            # Synthetic indicator history must never generate funding (issue 3).
            try:
                sim_side = int(getattr(strategy, "_sim_side", 0))
                lf = getattr(strategy, "_last_funding_ns", None)
                state.last_funding_ns = int(lf) if sim_side != 0 and lf is not None else None
            except Exception:
                state.last_funding_ns = None
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

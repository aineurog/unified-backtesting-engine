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
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np

from ube.adapters.nautilus_adapter.adapt_data import build_bar_type
from ube.adapters.nautilus_adapter.instrument_map import build_instrument
from ube.core.cost import resolve_cost_model
from ube.core.data import max_price_decimals
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
            # Resolution-aware price precision (parity with the backtest adapter): the
            # sandbox requires bar OHLC precision to equal the instrument's, and a coarse
            # config tick (0.1) must never round away digits the data actually carries.
            instrument_pp = int(instrument.price_precision)
            data_pp = max_price_decimals(
                chain(data.open, data.high, data.low, data.close),
                base=instrument_pp,
            )
            if data_pp > instrument_pp:
                overrides = dict(overrides)
                overrides["price_precision"] = data_pp
                overrides["price_increment"] = f"{10**-data_pp:.{data_pp}f}"
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

            # Volume scaling for full fills (paper/sandbox parity bug): the sandbox fills
            # MARKET orders against trade ticks synthesized from the bar's volume
            # (``SimulatedExchange._process_trade_ticks_from_bar`` — each tick size is
            # ``max(volume/4, size_increment)``, total fillable == bar volume). MT5 volume
            # for FX is raw *lots* (~100-400 per bar) while our order qty is in base
            # currency units — at 100x leverage a 10% order is ~73,800 units, so the raw
            # bar volume (~116) silently truncated every fill (~116.5). The backtest adapter
            # sizes post-tick and never passes through this fill model, which is why only
            # paper was affected. Scale bar volumes so the cap never binds; volume is unused
            # in the paper accounting (fills are booked by qty*price), so this is safe.
            # Upper bounds from config (starting balance, resolved leverage) — balance only
            # shrinks on resume, so starting_balance is always >= the live sizing capital.
            sizing_val = 1.0
            try:
                sm = config.base.risk.sizing
                sm_val = getattr(sm, "value", None) if sm is not None else None
                if sm is not None and sm_val is not None:
                    sizing_val = float(sm_val)
            except Exception:
                sizing_val = 1.0
            try:
                _slev = float(getattr(config.base.risk.sizing, "leverage", 1.0))
            except Exception:
                _slev = 1.0
            _olev = float(overrides.get("leverage", 0.0))
            _acct_est = str(overrides.get("account_type", "margin")).lower()
            est_lev = (1.0 if _acct_est == "cash" else max(_slev, _olev, 1.0))
            est_balance = float(
                config.starting_balance or overrides.get("starting_balance", 100_000.0)
            )
            max_notional = est_balance * est_lev * max(sizing_val, 1.0)
            min_close = float(np.min(data.close)) if n > 0 else 1.0
            max_order_qty = (max_notional / min_close) if min_close > 0 else max_notional
            max_vol = float(np.max(data.volume)) if n > 0 and len(data.volume) else 0.0
            volume_scale = (
                max(1.0, max_order_qty / max_vol) if max_vol > 0 else max(1.0, max_order_qty)
            )
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
                        "volume": float(data.volume[i]) * volume_scale,
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

            # Leverage: mirror backtest's sizing * margin logic — sizing leverage
            # dominates, override is fallback, CASH accounts force 1.0 (§3.2).
            # ``no_short`` (long-only crypto_spot/stocks) no longer forces 1x: the
            # backtest adapter levers long-only instruments too (actor_leverage =
            # sizing_leverage unless account_type == 'cash'), so paper must match it
            # (e.g. crypto_spot 10% x 100x = 10x balance exposure — trade_ledger vs
            # ledger.csv parity). Long-only stays enforced via allow_short (decide_action);
            # leverage and direction are orthogonal.
            sizing_lev = 1.0
            try:
                sizing_lev = float(getattr(config.base.risk.sizing, "leverage", 1.0))
            except Exception:
                sizing_lev = 1.0
            override_lev = float(overrides.get("leverage", 0.0))
            _acct = str(overrides.get("account_type", "margin")).lower()
            if _acct == "cash":
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

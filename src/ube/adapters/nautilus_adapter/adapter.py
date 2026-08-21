"""NautilusTrader engine adapter (§4.5).

:class:`NautilusAdapter` is the concrete :class:`~ube.adapters.base.EngineAdapter` that
runs one canonical backtest on NautilusTrader's ``BacktestEngine`` and folds the
engine's fills/positions back into the canonical event ledger (§4.6), producing a
:class:`~ube.core.result.BacktestResult`.

Translation summary (plan §5.3):

* **Signals + exits** — the canonical 4-column :class:`~ube.core.signals.Signals` and
  :class:`RiskConfig.exit` levels are injected into Nautilus via :class:`UbeActor`
  (native orders); the core trigger rule computes the levels, Nautilus performs the
  fill simulation (§8).
* **Fees** — the core :class:`~ube.core.cost.CostModel` drives everything. Its
  ``commission + slippage`` is folded into both the instrument's ``maker_fee`` and
  ``taker_fee`` at construction (an explicit ``maker_fee`` / ``taker_fee`` engine
  override wins over the cost model). Nautilus then charges the fee per fill, and the
  fold books each fill's ``commission`` into the ledger as a ``commission`` event —
  the canonical trade/equity math applies it identically to the core cost functions.
* **Carry** — ``funding`` / ``borrow`` rates have no native Nautilus equivalent, so
  the adapter derives ``funding_payment`` events directly from the core
  :func:`~ube.core.ledger.funding_payments` generator, aligned per bar (§24).
* **Ledger fold** — the actor records ``signal_evaluated`` (per-bar action strings for
  real entries/exits/flips only — holds are never emitted), ``order_submitted`` (one
  entry per order the actor actually placed, carrying the order_id/side/quantity at the
  submission bar), ``exit_reasons`` (closing-fill client_order_id → §4.6 reason) and
  ``position_reasons``; the adapter folds those plus the fills/positions reports into
  ``LedgerEvent``\\ s sorted by timestamp — ``signal_evaluated``, ``order_submitted``,
  ``fill`` (with the closing fill's ``exit_reason``), ``commission``,
  ``position_change`` (reconstructed from the fills' running net position, so scale-out
  partials move the position), and ``funding_payment`` — preceded by a ``cash_movement``
  book of the starting balance at the first bar (§4.6 step 4).
* **Result** — :meth:`~ube.core.result.BacktestResult.from_ledger` derives the
  ``trades`` / ``positions`` / ``equity_curve`` views once (§4.6).

Parity tolerance note (§8, §15): the parity contract is *aggregate* — realized P&L,
exit-reason attribution, and round-trip timing — never per-fill minutiae, because
Nautilus's bar-adaptive fill model legitimately differs from the core simulation:

* One logical market order can produce **multiple sub-fills** whose exact quantities
  depend on the bar's range and volume (e.g. ``0.25 @ 65000.0 + 1.288 @ 65000.1`` for a
  single order); the ledger folds them, and the *trade* math (entry/exit average,
  ``net_pnl``) is engine-stable.
* A **touched** stop/target fills at the trigger/target price, whereas the reference
  (native ``trailing_stop_market``) fills at the triggering bar's close — so a
  per-fill price can differ between engines by up to the intra-bar range. Aggregate
  realized P&L and ``exit_reason`` are the locked parity numbers.
* On a bar crossing **both** a stop and a target, Nautilus fills the limit (target)
  *first*; no ``BacktestVenueConfig`` flag flips this (probe-verified on 1.221.0).
  §8 prescribes stop-first; the actor honours §8 precedence for close-time exits and
  documents this as a residual divergence for the parity report.
* Fees: ``commission + slippage`` is folded into the instrument's ``maker_fee`` /
  ``taker_fee``, so Nautilus charges per fill; ``funding`` / ``borrow`` have no native
  equivalent and are re-derived from the core ``CostModel`` by
  :func:`~ube.core.ledger.funding_payments`. Rounding can differ from a purely
  core-computed schedule by sub-cent amounts.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, cast

import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money

from ube.adapters.base import EngineAdapter
from ube.adapters.nautilus_adapter.actor import UbeActor, UbeActorConfig
from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars, to_signal_map
from ube.adapters.nautilus_adapter.instrument_map import build_instrument
from ube.adapters.nautilus_adapter.overrides import (
    DEFAULT_OMS_TYPE,
    DEFAULT_STARTING_BALANCE,
    DEFAULT_VENUE,
    validate_overrides,
)
from ube.core.config import BacktestConfig
from ube.core.cost import resolve_cost_model
from ube.core.data import MarketData
from ube.core.errors import (
    ConfigError,
    DataShapeError,
    EngineError,
    InvalidSignalError,
)
from ube.core.ledger import EventLedger, EventType, LedgerEvent, funding_payments
from ube.core.result import BacktestResult
from ube.core.risk.exits import atr
from ube.core.signals import Signals

__all__ = ["NautilusAdapter"]

#: Stable sort rank per event type within a timestamp (so same-bar events fold in a
#: canonical order: evaluation -> order_submitted -> fills -> commissions ->
#: positions -> carry).
_KIND_RANK: dict[EventType, int] = {
    EventType.CASH_MOVEMENT: 0,
    EventType.SIGNAL_EVALUATED: 0,
    EventType.ORDER_SUBMITTED: 0,
    EventType.FILL: 1,
    EventType.COMMISSION: 2,
    EventType.POSITION_CHANGE: 3,
    EventType.FUNDING_PAYMENT: 4,
}


class NautilusAdapter(EngineAdapter):
    """Adapter for the NautilusTrader backtesting engine."""

    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data: Mapping[str, Any] | None = None,
    ) -> BacktestResult:
        """Run a backtest via NautilusTrader (§4.5).

        Args:
            data: The single-instrument OHLCV bars for the traded instrument.
            signals: The 4-column entry/exit signals on the same bar grid as ``data``.
            config: The full :class:`~ube.core.config.BacktestConfig` (instrument, cost,
                risk/exits, benchmark, engine overrides).
            aux_data: Optional derived-series map referenced by name from exit configs
                (§5.2) — e.g. ``{"atr_12h": <array>}`` for ``ATRStop(atr="atr_12h")``.
                When a referenced name is absent the adapter computes ATR from ``data``.
                A flat ``name -> array`` map for a single instrument.

        Returns:
            The canonical :class:`~ube.core.result.BacktestResult`.

        Raises:
            DataShapeError: ``data`` is not a :class:`~ube.core.data.MarketData`
                (or has fewer than two bars — no bar period to derive, §15).
            InvalidSignalError: ``signals`` is not a
                :class:`~ube.core.signals.Signals`, or its rows do not match
                ``data``'s bar count.
            ConfigError: ``config`` is not a :class:`~ube.core.config.BacktestConfig`.
            EngineError: The underlying Nautilus backtest failed (original preserved).
        """
        if not isinstance(data, MarketData):
            raise DataShapeError(
                "NautilusAdapter.run expects a MarketData of bars; "
                f"got {type(data).__name__}"
            )
        if not isinstance(signals, Signals):
            raise InvalidSignalError(
                "NautilusAdapter.run expects canonical Signals; "
                f"got {type(signals).__name__}"
            )
        if not isinstance(config, BacktestConfig):
            raise ConfigError(
                "NautilusAdapter.run expects a BacktestConfig; "
                f"got {type(config).__name__}"
            )
        if signals.n_bars != data.n_bars:
            raise InvalidSignalError(
                f"signals cover {signals.n_bars} bars but market data has "
                f"{data.n_bars}; signal rows must be row-aligned with bars"
            )

        overrides = validate_overrides(config.engine_overrides)
        cost_model = (
            config.cost_model
            if config.cost_model is not None
            else resolve_cost_model(config.instrument)
        )

        # Fold commission + slippage into the instrument fees (§5.3); an explicit
        # maker_fee/taker_fee override wins over the resolved cost model.
        fee_overrides = dict(overrides)
        rate = cost_model.commission + cost_model.slippage
        fee_overrides.setdefault("maker_fee", rate)
        fee_overrides.setdefault("taker_fee", rate)

        # Leverage (§3.2 single source of truth): it lives on ``SizeModel`` and is the
        # exposure multiplier applied to the order quantity. The venue's margin capacity
        # must cover that exposure, so the *effective* margin leverage is the larger of
        # the sizing leverage and any ``engine_overrides["leverage"]`` (the legacy margin
        # knob). When neither is set, ``_margin`` keeps its reference default (0.2 -> 5x).
        sizing_leverage = float(config.risk.sizing.leverage)
        override_leverage = float(overrides.get("leverage", 0.0) or 0.0)
        margin_leverage = 0.0
        if sizing_leverage > 1.0:
            margin_leverage = sizing_leverage
        if override_leverage > 0.0:
            margin_leverage = max(margin_leverage, override_leverage)
        if margin_leverage > 0.0:
            fee_overrides["leverage"] = margin_leverage

        build = build_instrument(config.instrument, fee_overrides)
        instrument_id = str(build.instrument_id)
        bars, bar_type = to_nautilus_bars(data, build)
        signal_map = to_signal_map(data, signals)
        settlement = config.instrument.settlement_currency or "USD"

        # Cash accounts cannot be leveraged: the reference nautilus_trader forces
        # leverage to 1.0 for cash, so mirror that here (§3.2).
        _acct = overrides.get("account_type", "margin")
        actor_leverage = 1.0 if _acct == "cash" else sizing_leverage

        actor = UbeActor(
            UbeActorConfig(
                instrument_id=build.instrument_id,
                bar_type=bar_type,
                signal_map=signal_map,
                asset_class=config.instrument.asset_class,
            ),
            market_data=data,
            sizing=config.risk.sizing,
            exits=config.risk.exit,
            aux_atr=aux_data,
            leverage=actor_leverage,
            vol=(
                _volatility_estimate(data)
                if config.risk.sizing.kind == "volatility_target"
                else None
            ),
        )

        venue = Venue(str(overrides.get("venue", DEFAULT_VENUE)))
        oms_type = (
            OmsType.NETTING
            if overrides.get("oms_type", DEFAULT_OMS_TYPE) == "NETTING"
            else OmsType.HEDGING
        )
        account_type = (
            AccountType.MARGIN
            if overrides.get("account_type", "margin") == "margin"
            else AccountType.CASH
        )
        starting_balance = float(
            overrides.get("starting_balance", DEFAULT_STARTING_BALANCE)
        )
        currency = Currency.from_str(settlement)

        try:
            # ube reports failures through EngineError exceptions, not the Nautilus
            # console log, so silence even ERROR chatter (e.g. the intentional
            # cash-account short rejection in tests) that would otherwise pollute
            # backtest output as a scary traceback.
            engine = BacktestEngine(
                config=BacktestEngineConfig(logging=LoggingConfig(log_level="OFF"))
            )
            engine.add_venue(
                venue,
                oms_type,
                account_type,
                [Money(Decimal(str(starting_balance)), currency)],
                base_currency=currency,
            )
            engine.add_instrument(build.instrument)
            engine.add_data(bars)
            engine.add_strategy(actor)
            engine.run()
        except Exception as exc:  # noqa: BLE001 — wrap whatever Nautilus raised.
            if not isinstance(exc, EngineError):
                raise EngineError(f"nautilus backtest failed: {exc}") from exc
            raise

        try:
            ledger = self._fold(
                engine,
                actor=actor,
                data=data,
                config=config,
                cost_model=cost_model,
                instrument_id=instrument_id,
                settlement=settlement,
                starting_balance=starting_balance,
            )
        finally:
            engine.dispose()

        return BacktestResult.from_ledger(
            ledger,
            config,
            market_data={instrument_id: data},
            instruments={instrument_id: config.instrument},
        )

    # ------------------------------------------------------------------
    # Ledger fold (§4.6).
    # ------------------------------------------------------------------

    def _fold(
        self,
        engine: BacktestEngine,
        *,
        actor: UbeActor,
        data: MarketData,
        config: BacktestConfig,
        cost_model: object,
        instrument_id: str,
        settlement: str,
        starting_balance: float,
    ) -> EventLedger:
        """Fold Nautilus reports + the actor's records into a canonical ledger."""
        multiplier = (
            1.0
            if config.instrument.contract_multiplier is None
            else float(config.instrument.contract_multiplier)
        )
        bar_ts = _bar_timestamps_ns(data)

        sorted_events: list[tuple[int, int, int, LedgerEvent]] = []
        seq = 0

        def _add(event: LedgerEvent, ts: int) -> None:
            nonlocal seq
            sorted_events.append((int(ts), _KIND_RANK[event.event_type], seq, event))
            seq += 1

        # Starting balance booked as a cash inflow at the first bar boundary (§4.6
        # step 4 — the cash leg of the equity curve).
        _add(
            LedgerEvent(
                EventType.CASH_MOVEMENT,
                int(bar_ts[0]),
                instrument_id,
                amount=starting_balance,
                currency=settlement,
            ),
            int(bar_ts[0]),
        )

        # Per-bar signal evaluations recorded by the actor (§6.1). Holds are never
        # recorded by the actor, so only real entries/exits/flips reach the ledger.
        for bar_idx, action in actor.signal_evaluated.items():
            _add(
                LedgerEvent(
                    EventType.SIGNAL_EVALUATED,
                    int(bar_ts[bar_idx]),
                    instrument_id,
                    action=action,
                ),
                int(bar_ts[bar_idx]),
            )

        # One order_submitted per order the actor actually placed (§4.6): entries,
        # scale-out partials, stop/target exits, and full closes. Submitted at the bar
        # boundary (int(bar.ts_event)) and ordered before the resulting fills.
        for order_id, (ts, side, quantity) in actor.order_submitted.items():
            _add(
                LedgerEvent(
                    EventType.ORDER_SUBMITTED,
                    ts,
                    instrument_id,
                    order_id=order_id,
                    side=side,
                    quantity=quantity,
                ),
                ts,
            )

        # Fills -> fill + cash-leg + commission events; position_change reconstructed
        # from the running net position so scale-out partials move the ledger position.
        fills = engine.trader.generate_fills_report()
        net: float = 0.0
        for client_order_id, row in fills.iterrows():
            ts = _fill_timestamp_ns(row)
            side = 1 if row["order_side"] == "BUY" else -1
            quantity = float(row["last_qty"])
            price = float(row["last_px"])
            notional = quantity * price * multiplier
            # Cash leg of the fill (§4.6): a buy pays the notional out of the account, a
            # sell pays it back in. Together with the starting-balance cash_movement and
            # the mark-to-market of open positions, this keeps the equity curve
            # self-financing: equity == starting balance + accumulated (realized) P&L,
            # with the open position's unrealized P&L carried by the mark.
            _add(
                LedgerEvent(
                    EventType.CASH_MOVEMENT,
                    ts,
                    instrument_id,
                    amount=-side * notional,
                    currency=settlement,
                ),
                ts,
            )
            _add(
                LedgerEvent(
                    EventType.FILL,
                    ts,
                    instrument_id,
                    side=side,
                    quantity=quantity,
                    price=price,
                    notional=notional,
                    exit_reason=actor.exit_reasons.get(str(client_order_id)),
                ),
                ts,
            )
            commission = str(row["commission"])
            if commission:
                amount, currency = _parse_money(commission)
                if amount != 0.0:
                    _add(
                        LedgerEvent(
                            EventType.COMMISSION,
                            ts,
                            instrument_id,
                            amount=amount,
                            currency=currency,
                        ),
                        ts,
                    )
            net += side * quantity
            _add(
                LedgerEvent(
                    EventType.POSITION_CHANGE,
                    ts,
                    instrument_id,
                    position_after=net,
                ),
                ts,
            )

        # Carry (funding/swap + short borrow) from the core cost model (§24): rebuild
        # the per-bar position step series and let funding_payments generate events.
        funding, borrow = _carry_rates(cost_model)
        if funding != 0.0 or borrow != 0.0:
            position_change = _events_of_type(sorted_events, EventType.POSITION_CHANGE)
            pc_ts = np.asarray(
                [int(e.timestamp) for e in position_change],
                dtype=np.int64,
            )
            pc_val = np.asarray(
                [float(cast(float, e.position_after)) for e in position_change],
                dtype=np.float64,
            )
            step_ts, step_value = _step_timestamps(pc_ts, pc_val)
            for event in funding_payments(
                instrument_id=instrument_id,
                timestamps=bar_ts,
                notional=_bar_notional(data, step_ts, step_value, multiplier),
                side=_bar_side(data, step_ts, step_value),
                funding_rate=funding,
                borrow_rate=borrow,
                currency=settlement,
            ):
                _add(event, int(event.timestamp))

        sorted_events.sort(key=lambda item: (item[0], item[1], item[2]))
        return EventLedger(event for _ts, _rank, _seq, event in sorted_events)


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------


def _events_of_type(
    events: list[tuple[int, int, int, LedgerEvent]], event_type: EventType
) -> list[LedgerEvent]:
    """The events of one type, in append order."""
    return [event for _ts, _rank, _seq, event in events if event.event_type is event_type]


def _carry_rates(cost_model: object) -> tuple[float, float]:
    """The ``(funding, borrow)`` per-bar rates of the resolved cost model."""
    funding = float(getattr(cost_model, "funding", 0.0))
    borrow = float(getattr(cost_model, "borrow", 0.0))
    return funding, borrow


def _volatility_estimate(market_data: MarketData) -> np.ndarray:
    """A per-bar volatility estimate (ATR / price) for ``volatility_target`` sizing."""
    result: np.ndarray = atr(market_data) / market_data.close
    return result


def _bar_timestamps_ns(data: MarketData) -> np.ndarray:
    """The bar-boundary axis as int64 nanoseconds (matching :mod:`ube.core.ledger`)."""
    return cast(np.ndarray, data.timestamps.as_unit("ns").asi8)  # type: ignore[attr-defined]


def _fill_timestamp_ns(row: Any) -> int:
    """The int64-nanosecond fill timestamp from a fills-report row."""
    return int(row["ts_event"].value)


def _parse_money(text: str) -> tuple[float, str]:
    """Parse a Nautilus ``Money`` repr (``"1000.20 USD"``) into ``(amount, currency)``."""
    parts = str(text).strip().split()
    return float(parts[0]), parts[1] if len(parts) > 1 else "USD"


def _step_timestamps(
    ts: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(timestamps, values)`` to a minimal ascending step series."""
    if ts.shape[0] == 0:
        return ts, values
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    values = values[order]
    unique = np.unique(ts)
    last = np.searchsorted(ts, unique, side="right") - 1
    return ts[last], values[last]


def _bar_step_grid(
    step_ts: np.ndarray, step_value: np.ndarray, bar_ts: np.ndarray
) -> np.ndarray:
    """Forward-filled step value at each bar boundary (flat before the first change)."""
    if step_ts.shape[0] == 0:
        return np.zeros(bar_ts.shape[0], dtype=np.float64)
    idx = np.searchsorted(step_ts, bar_ts, side="right") - 1
    valid = idx >= 0
    return np.where(valid, step_value[np.clip(idx, 0, None)], 0.0)


def _bar_position(data: MarketData, step_ts: np.ndarray, step_value: np.ndarray) -> np.ndarray:
    """The held position at each bar boundary (``+``/``-``/open-notional sign)."""
    bar_ts = _bar_timestamps_ns(data)
    return _bar_step_grid(step_ts, step_value, bar_ts)


def _bar_notional(
    data: MarketData,
    step_ts: np.ndarray,
    step_value: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Per-bar open notional: ``abs(position) * close * contract_multiplier``."""
    pos = _bar_position(data, step_ts, step_value)
    result: np.ndarray = np.abs(pos) * data.close * multiplier
    return result


def _bar_side(data: MarketData, step_ts: np.ndarray, step_value: np.ndarray) -> np.ndarray:
    """Per-bar direction (``+1`` / ``-1`` / ``0``) from the position step series."""
    result: np.ndarray = np.sign(_bar_position(data, step_ts, step_value))
    return result
"""Nautilus paper-trading strategy (plan T3 / §4.3).

Ports ``sim_nautilus.strategy.SignalBracketStrategy`` but, for the T3 vertical slice, submits
plain MARKET orders driven by the *same* ``decide_action`` logic the recording backend uses
(§9.4 comparability) — no bespoke accounting. Sizing comes from
``core.risk.sizing.size_position`` + ``floor_to_step``; fills are bridged to ube
``LedgerEvent``s (the ``EventLedger`` is the single source of truth, §4.6).

TP/SL bracket exits and funding arrive in later tasks (T4/T5); the slice exits on the
signal (``exit_reason="signal"``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

from ube.core.ledger import EventType, LedgerEvent
from ube.core.risk.sizing import SizeModel, floor_to_step, size_position
from ube.papertrading.core import decide_action

from .bridge import commission_event, fill_event, position_change_event
from .runtime import get_ready_event
from .signals import SIGNAL_REGISTRY


class UbePaperConfig(StrategyConfig):
    """Configuration for :class:`UbePaperStrategy`."""

    instrument_id: str = ""
    bar_type: str = ""
    sizing: Any = None  # ube SizeModel (frozen)
    cost_model: Any = None  # ube CostModel | None
    no_short: bool = False
    on_opposite_signal: str = "reverse"
    balance: float = 0.0


class UbePaperStrategy(Strategy):
    """Submits MARKET orders from the per-bar ``decide_action`` and bridges fills."""

    def __init__(self, config: UbePaperConfig) -> None:
        super().__init__(config)
        self._instrument: Instrument | None = None
        self._iid = config.instrument_id
        self._bar_type = BarType.from_str(config.bar_type) if config.bar_type else None
        self._sim_side = 0
        self._sim_qty = 0.0
        self._pending_open: int | None = None
        self._last_close: float | None = None
        self._entry_order_ids: set = set()
        self._close_order_ids: set = set()
        self._events: list[LedgerEvent] = []

    # -- lifecycle --------------------------------------------------------- #
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(InstrumentId.from_str(self._iid))
        if self._instrument is None:
            self.log.error(f"Instrument {self._iid} not found in cache")
            self.stop()
            return
        if self._bar_type is not None:
            self.subscribe_bars(self._bar_type)
        get_ready_event().set()
        self.log.info(f"UbePaperStrategy started for {self._iid}")

    @property
    def events(self) -> list[LedgerEvent]:
        return self._events

    # -- signals ----------------------------------------------------------- #
    def on_bar(self, bar: Bar) -> None:
        if self._instrument is None:
            return
        SIGNAL_REGISTRY.bars_processed += 1
        self._last_close = float(bar.close.as_double())
        le, lx, se, sx = SIGNAL_REGISTRY.at(bar.ts_event)
        if not (le or lx or se or sx):
            return
        action = decide_action(
            self._sim_side,
            long_entry=le,
            long_exit=lx,
            short_entry=se,
            short_exit=sx,
            allow_short=not self.config.no_short,
            policy=self.config.on_opposite_signal,
        )
        if action == "hold":
            return

        close_first = action in ("close", "reverse")
        desired = 0
        if action == "open_long":
            desired = 1
        elif action == "open_short":
            desired = -1
        elif action == "reverse":
            desired = -self._sim_side

        t = int(bar.ts_event)
        self._events.append(
            LedgerEvent(EventType.SIGNAL_EVALUATED, t, self._iid, action=action)
        )
        if close_first:
            self._submit_close(self._last_close)
        if desired != 0:
            if action == "reverse":
                self._pending_open = desired
            else:
                self._submit_open(desired, self._last_close)

    # -- fills ------------------------------------------------------------- #
    def on_order_filled(self, event) -> None:
        coid = event.client_order_id
        self._entry_order_ids.discard(coid)
        self._close_order_ids.discard(coid)

        if self._sim_side == 0:
            exit_reason: str | None = None
            self._sim_side = 1 if event.is_buy else -1
            self._sim_qty = float(event.last_qty.as_double())
        else:
            exit_reason = "signal"
            self._sim_side = 0
            self._sim_qty = 0.0
            if self._pending_open is not None:
                desired = self._pending_open
                self._pending_open = None
                self._submit_open(desired, self._last_close)

        self._events.append(fill_event(event, self._iid, exit_reason=exit_reason))
        comm = commission_event(event, self._iid, self.config.cost_model)
        if comm is not None:
            self._events.append(comm)
        self._events.append(
            position_change_event(
                event,
                self._iid,
                side=self._sim_side,
                position_after=self._sim_side * self._sim_qty,
            )
        )

    # -- order submission -------------------------------------------------- #
    def _lot_step(self) -> float:
        instr = self._instrument
        if instr is None:
            return 1.0
        prec = getattr(instr, "size_precision", None)
        if prec is not None:
            return 10.0 ** (-int(prec))
        for attr in ("size_increment", "lot_size"):
            val = getattr(instr, attr, None)
            if val is not None:
                try:
                    return float(val.as_double())
                except Exception:  # pragma: no cover - defensive
                    continue
        return 1.0

    def _size_qty(self, price: float):
        if self.config.sizing is None or price is None or price <= 0:
            return self._instrument.make_qty(0.0)
        units = size_position(
            self.config.sizing,
            capital=self.config.balance,
            price=price,
            cost_model=self.config.cost_model,
        )
        step = self._lot_step()
        units = floor_to_step(units, step)
        return self._instrument.make_qty(float(units))

    def _submit_open(self, desired: int, price: float | None) -> None:
        if self._instrument is None or price is None:
            return
        side = OrderSide.BUY if desired == 1 else OrderSide.SELL
        qty = self._size_qty(price)
        if float(qty.as_double()) <= 0.0:
            self.log.warning(f"Skipped open {side}: non-positive quantity")
            return
        order = self.order_factory.market(
            instrument_id=self._instrument.id,
            order_side=side,
            quantity=qty,
            reduce_only=False,
        )
        self._entry_order_ids.add(order.client_order_id)
        self._events.append(
            LedgerEvent(
                EventType.ORDER_SUBMITTED,
                int(order.ts_init),
                self._iid,
                order_id=str(order.client_order_id),
                side=desired,
                quantity=float(qty.as_double()),
            )
        )
        self.submit_order(order)
        self.log.info(f"Submitted open {side} qty={qty}")

    def _submit_close(self, price: float | None) -> None:
        if self._instrument is None or self._sim_qty <= 0:
            return
        side = OrderSide.SELL if self._sim_side > 0 else OrderSide.BUY
        qty = self._instrument.make_qty(self._sim_qty)
        order = self.order_factory.market(
            instrument_id=self._instrument.id,
            order_side=side,
            quantity=qty,
            reduce_only=True,
        )
        self._close_order_ids.add(order.client_order_id)
        self._events.append(
            LedgerEvent(
                EventType.ORDER_SUBMITTED,
                int(order.ts_init),
                self._iid,
                order_id=str(order.client_order_id),
                side=-self._sim_side,
                quantity=float(qty.as_double()),
            )
        )
        self.submit_order(order)
        self.log.info(f"Submitted close {side} qty={qty}")


__all__ = ["UbePaperConfig", "UbePaperStrategy"]

"""Event-driven backtrader strategy (§6.1, §8).

backtrader is an actor loop, not a vector engine: the strategy walks the bars and places
market orders via the broker. This module implements the single-instrument execution policy
that mirrors the other adapters' semantics on that loop:

- **Entry**: a ``long_entry``/``short_entry`` signal bar submits a market order that fills at the
  *next* bar's open. Sizing uses the core :func:`~ube.core.risk.sizing.size_position` against the
  broker equity levered by the effective leverage (§6.3), priced at the slipped current close,
  floored to the asset-class lot increment and additionally floored to what the broker margin
  allows — mirroring the reference ``NotionalSizer`` so a leveraged perp entry can never be
  rejected by the venue for insufficient margin.
- **Exits**: at entry fill time the strategy precomputes the per-bar trigger arrays of every
  configured exit via :func:`~ube.adapters.backtrader_adapter.exits.build_exit_plan` (anchored
  to the actual fill and the slipped entry price, causal — §8). Each bar, the first not-yet-spent
  exit whose array fires (in configured order) exits its ``scale_out_fraction`` at the next open;
  stops always exit the whole remaining position. A spent line never re-fires.
- **Signal exits / flips**: a matching ``long_exit``/``short_exit`` (or an opposite-side entry)
  closes the position via a *signal* exit; a flip then re-enters the opposite side on the bar the
  close order is placed.
- **Accounting**: the strategy records every evaluated signal and submitted order into
  ``signal_records`` / ``order_records``; those (not the broker's cash) are what the adapter
  folds into the canonical ledger — the broker cash is only the *sizing* equity, and fills happen
  at raw open prices, with ``commission`` and ``slippage`` applied by the fold (§8).
- **Final close**: an order placed on the last bar can never fill through the next-bar-open path,
  so ``stop()`` realizes a position still open at the end at the final bar's close (an in-flight
  exit keeps its reason; the remainder closes reasonless) — mirroring the reference adapter,
  which realizes the final bar's exit at the bar close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from backtrader import Order
from backtrader import Strategy as BtStrategyBase

from ube.adapters.backtrader_adapter.exits import BtExitLine, build_exit_plan
from ube.adapters.backtrader_adapter.instrument_map import (
    BtInstrument,
    floor_to_increment,
)
from ube.core.cost import CostModel, fill_cost, slipped_price
from ube.core.data import MarketData
from ube.core.errors import EngineError
from ube.core.risk.exits import Exit
from ube.core.risk.sizing import _entry_fee_rate, size_position

__all__ = [
    "BtRunContext",
    "BtSignalRecord",
    "BtOrderRecord",
    "BacktraderStrategy",
]

#: Signed-size threshold below which the broker position counts as flat.
_FLAT_EPS: float = 1e-12


@dataclass
class BtRunContext:
    """Everything the strategy needs from the adapter, attached to the feed (engine.py).

    Holds the canonical market data (for exit-plan triggers), the configured exits, the per-exit
    ATR series, the sizing model, cost model, resolved instrument and the effective account
    leverage. Attaching via the feed instance avoids backtrader's param deep-copies of the
    strategy object.
    """

    data: MarketData
    exits: tuple[Exit, ...]
    atr_map: dict[int, np.ndarray]
    inst: BtInstrument
    sizing: Any
    cost_model: CostModel | None
    eff_leverage: float
    margin_account: float | None
    vol_arr: np.ndarray | None
    slip: float


@dataclass(frozen=True)
class BtSignalRecord:
    """One evaluated signal (§4.6) — the fold's ``signal_evaluated`` row."""

    bar: int  # the bar the strategy saw the signal on
    action: str
    side: int = 0


@dataclass(frozen=True)
class BtOrderRecord:
    """One submitted + filled order (§4.6) — the fold's ``order_submitted``/``fill`` rows.

    ``submit_bar`` is the bar the order was placed on (the signal/trigger bar); ``fill_bar`` is
    the bar whose open it filled at (``submit_bar + 1`` for a market order). ``price`` is the raw
    fill price (fill-bar open); slippage is applied at the fold.
    """

    submit_bar: int
    fill_bar: int
    side: int  # the fill direction (+1 buy / -1 sell)
    quantity: float
    price: float
    reason: str | None = None


@dataclass(frozen=True)
class _SignalExit:
    """The signal-exit pseudo-line used for full signal closes / flips (§8)."""

    reason: str = "signal"
    fraction: float = 1.0  # signal exits always close the whole remaining position
    action: str = "long_exit"
    index: int = -1  # never tracked in the spent set


@dataclass
class _OpenPosition:
    side: int  # +1 long / -1 short
    entry_bar: int
    entry_price: float  # raw fill price (fill-bar open)
    plan: tuple[BtExitLine, ...] = ()
    spent: set[int] = field(default_factory=set)  # plan line indices already fired


class BacktraderStrategy(BtStrategyBase):  # type: ignore[misc]  # untyped backtrader base
    """The event-driven execution policy for one instrument (§6.1)."""

    def __init__(self) -> None:
        self._ctx: BtRunContext = cast(BtRunContext, self.data._ube_ctx)
        self._ube_orders: dict[int, dict[str, Any]] = {}  # order.ref -> metadata
        self._pending_entry: dict[str, Any] | None = None
        self._pending_exit: dict[str, Any] | None = None
        self._pos: _OpenPosition | None = None
        self.signal_records: list[BtSignalRecord] = []
        self.order_records: list[BtOrderRecord] = []

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _fill_price(self, side: int) -> float:
        """Actual next-bar fill price for an entry (backtrader ``+1`` index = next bar)."""
        ctx = self._ctx
        i = len(self.data) - 1
        if i + 1 >= ctx.data.n_bars:
            return float("nan")
        return float(slipped_price(float(self.data.open[1]), side, ctx.slip))

    def _target_quantity(self, side: int) -> float:
        """Core-sized entry quantity for one direction (§6.3), floored to the lot increment."""
        ctx = self._ctx
        if len(self.data) - 1 + 1 >= ctx.data.n_bars:
            return 0.0  # no next bar to fill on — entry cannot execute
        price = self._fill_price(side)
        equity = float(self.broker.getvalue())
        kwargs: dict[str, Any] = dict(
            capital=equity, price=price, cost_model=ctx.cost_model
        )
        if ctx.sizing.kind == "equal_weight":
            kwargs["n"] = 1
        if ctx.vol_arr is not None:
            kwargs["vol"] = float(ctx.vol_arr[len(self.data) - 1])
        qty = float(size_position(ctx.sizing, **kwargs))
        qty = floor_to_increment(qty, ctx.inst.size_increment)
        if qty <= 0.0:
            return 0.0
        cash = float(self.broker.getcash())
        # §8: mirrored reference ``NotionalSizer`` — size against the actual fill price
        # (next-bar open), not the decision close, and reserve the entry commission, so the
        # bookable notional *plus* fees can never exceed cash/leverage and the venue cannot
        # spuriously reject a sized-for order. The broker equity is already the *leveraged*
        # account (``starting_balance × leverage``), matching the reference sizing capital — the
        # sizer's own ``leverage`` knob is not re-applied here. Cash-like classes book the full
        # notional; futures-like book margin = notional/lev via automargin ``1/lev``.
        fee_rate = (
            _entry_fee_rate(ctx.cost_model) if ctx.cost_model is not None else 0.0
        )
        unit_cost = price * (1.0 + fee_rate)
        if ctx.margin_account is not None:
            lev = max(ctx.margin_account, 1.0)
            max_qty = cash * lev / (unit_cost * ctx.margin_account)
            return float(
                floor_to_increment(min(qty, max_qty), ctx.inst.size_increment)
            )
        max_qty = cash / unit_cost
        return float(
            floor_to_increment(min(qty, max_qty), ctx.inst.size_increment)
        )

    def _check_affordable(self, side: int, qty: float) -> None:
        """Surface a genuine venue shortfall as an ``EngineError`` (§7.1 mirror of vectorbt)."""
        ctx = self._ctx
        if (
            ctx.cost_model is not None
            and ctx.sizing.kind != "volatility_target"
            and _entry_fee_rate(ctx.cost_model) > 0.0
        ):
            price = self._fill_price(side)
            notional = qty * price
            required = notional + float(fill_cost(ctx.cost_model, notional=notional))
            capacity = float(self.broker.getvalue()) * ctx.eff_leverage
            if required > capacity * (1.0 + 1e-9):
                raise EngineError(
                    f"market order rejected by the venue: insufficient funds at bar "
                    f"{len(self.data) - 1} — requires {required:.6g}, available "
                    f"{capacity:.6g} {ctx.inst.settlement_currency}"
                )

    # ------------------------------------------------------------------
    # Next / per-bar event loop
    # ------------------------------------------------------------------

    def next(self) -> None:
        i = len(self.data) - 1
        ctx = self._ctx
        if ctx.data.n_bars < 2:
            return  # no room for a next-bar fill

        # --- Exits: risk plan first, then signal exit / flip -------------
        if (
            self._pos is not None
            and self._pending_exit is None
            and abs(self.position.size) > _FLAT_EPS
        ):
            fired: BtExitLine | _SignalExit | None = None
            for line in self._pos.plan:
                if line.index in self._pos.spent or not bool(line.triggered[i]):
                    continue
                fired = line
                break
            if fired is None:
                side = self._pos.side
                if side == 1 and bool(self.data.long_exit[0]):
                    fired = _SignalExit(action="long_exit")
                elif side == -1 and bool(self.data.short_exit[0]):
                    fired = _SignalExit(action="short_exit")
                elif side == 1 and bool(self.data.short_entry[0]):
                    fired = _SignalExit(action="long_exit")
                elif side == -1 and bool(self.data.long_entry[0]):
                    fired = _SignalExit(action="short_exit")

            if fired is not None:
                open_units = abs(self.position.size)
                fraction = float(fired.fraction)
                exit_qty = min(fraction * open_units, open_units)
                exit_qty = float(
                    floor_to_increment(exit_qty, ctx.inst.size_increment)
                )
                if exit_qty < ctx.inst.size_increment and exit_qty > _FLAT_EPS:
                    exit_qty = min(ctx.inst.size_increment, open_units)
                if exit_qty > _FLAT_EPS:
                    exit_side = -self._pos.side
                    order = (
                        self.buy(size=exit_qty)
                        if exit_side == 1
                        else self.sell(size=exit_qty)
                    )
                    action = (
                        fired.action
                        if isinstance(fired, _SignalExit)
                        else f"exit_{fired.reason}"
                    )
                    self.signal_records.append(
                        BtSignalRecord(i, action, exit_side)
                    )
                    self._pending_exit = {
                        "ref": order.ref,
                        "submit_bar": i,
                        "side": exit_side,
                        "size": exit_qty,
                        "reason": fired.reason,
                        "close_all": isinstance(fired, _SignalExit),
                    }
                    self._ube_orders[order.ref] = {
                        "kind": "exit",
                        "submit_bar": i,
                        "side": exit_side,
                        "size": exit_qty,
                        "reason": fired.reason,
                    }
                    if not isinstance(fired, _SignalExit):
                        self._pos.spent.add(fired.index)

        # --- Entries (fresh, or the flip leg of the exit above) ----------
        if self._pending_entry is None and (
            self._pending_exit is None
            or (
                self._pending_exit.get("close_all") is True
                and self._pending_exit["submit_bar"] == i
            )
        ):
            self._maybe_entry(i)

    def _maybe_entry(self, i: int) -> None:
        pos = self._pos
        pos_flat = pos is None or abs(self.position.size) < _FLAT_EPS
        if pos_flat:
            if bool(self.data.long_entry[0]):
                self._submit_entry(i, 1)
            elif bool(self.data.short_entry[0]):
                self._submit_entry(i, -1)
        elif pos is not None and pos.side == 1 and bool(self.data.short_entry[0]):
            self._submit_entry(i, -1)
        elif pos is not None and pos.side == -1 and bool(self.data.long_entry[0]):
            self._submit_entry(i, 1)

    def _submit_entry(self, i: int, side: int) -> None:
        qty = self._target_quantity(side)
        if qty <= _FLAT_EPS:
            return
        self._check_affordable(side, qty)
        order = self.buy(size=qty) if side == 1 else self.sell(size=qty)
        self.signal_records.append(
            BtSignalRecord(i, "long_entry" if side == 1 else "short_entry", side)
        )
        self._pending_entry = {
            "ref": order.ref,
            "submit_bar": i,
            "side": side,
            "size": qty,
        }
        self._ube_orders[order.ref] = {
            "kind": "entry",
            "submit_bar": i,
            "side": side,
            "size": qty,
        }

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def notify_order(self, order: Any) -> None:
        if order.status in (Order.Margin, Order.Rejected, Order.Canceled):
            status = order.Status[order.status]
            raise EngineError(
                f"backtrader order rejected in broker ({status}): "
                f"ordref={order.ref} side={'buy' if order.isbuy() else 'sell'} "
                f"size={order.created.size} price={order.created.price}"
            )
        if order.status is not Order.Completed:
            return
        meta = self._ube_orders.pop(order.ref, None)
        if meta is None:
            return
        ctx = self._ctx
        submit_bar = int(meta["submit_bar"])
        fill_bar = submit_bar + 1
        side = int(meta["side"])
        qty = abs(float(order.executed.size))
        price = float(order.executed.price)
        if meta["kind"] == "entry":
            self.order_records.append(
                BtOrderRecord(submit_bar, fill_bar, side, qty, price, None)
            )
            plan = build_exit_plan(
                ctx.exits,
                ctx.data,
                side=side,
                entry_price=float(slipped_price(price, side, ctx.slip)),
                entry_bar=fill_bar,
                atr_map=ctx.atr_map,
            )
            self._pos = _OpenPosition(
                side=side,
                entry_bar=fill_bar,
                entry_price=price,
                plan=plan,
            )
            self._pending_entry = None
        else:
            self.order_records.append(
                BtOrderRecord(
                    submit_bar,
                    fill_bar,
                    side,
                    qty,
                    price,
                    meta.get("reason"),
                )
            )
            self._pending_exit = None
            if self._pos is not None and abs(self.position.size) < _FLAT_EPS:
                self._pos = None

    def stop(self) -> None:
        """Close any still-open position at the final bar's close (§4.5, §6.1).

        A market order submitted on the very last bar has no next bar to fill at, so its
        exit would otherwise be silently lost. Mirror the reference (Nautilus) adapter,
        which realizes the final bar's exit at the bar close: an exit already in flight on
        the final bar is booked with its own reason, and any remaining open quantity is
        closed as a plain final close (reason ``None``). Both records use the raw final
        close as fill price — the fold applies slippage like every other fill — and the
        fold's running net returns to zero, so the ledger reproduces an honest closed
        round trip instead of an open mark-to-market.
        """
        if self._pos is None or abs(self.position.size) <= _FLAT_EPS:
            return
        last = self._ctx.data.n_bars - 1
        close_price = float(self.data.close[0])
        total = abs(self.position.size)

        pending = self._pending_exit
        if pending is not None:
            side = int(pending["side"])
            qty = float(pending["size"])
            self.order_records.append(
                BtOrderRecord(last, last, side, qty, close_price, pending.get("reason"))
            )
            total -= qty
            self._pending_exit = None

        if total > _FLAT_EPS:
            self.order_records.append(
                BtOrderRecord(
                    last, last, -self._pos.side, total, close_price, None
                )
            )
        self._pos = None
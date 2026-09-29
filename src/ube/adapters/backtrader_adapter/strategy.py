"""Event-driven backtrader strategy (§6.1, §8).

backtrader is an actor loop, not a vector engine: the strategy walks the bars and places
market orders via the broker. This module implements the single-instrument execution policy
that mirrors the other adapters' semantics on that loop:

- **Entry**: a ``long_entry``/``short_entry`` signal bar submits a market order that fills at that
  bar's close (same-bar, bar-synchronous fills matching the vectorbt/nautilus references). Sizing
  uses the core :func:`~ube.core.risk.sizing.size_position` against the broker equity levered by
  the effective leverage (§6.3), priced at the slipped current close, floored to the asset-class
  lot increment and additionally floored to what the broker margin allows — mirroring the
  reference ``NotionalSizer`` so a leveraged perp entry can never be rejected by the venue for
  insufficient margin.
- **Exits**: at entry fill time the strategy precomputes the per-bar trigger arrays of every
  configured exit via :func:`~ube.adapters.backtrader_adapter.exits.build_exit_plan` (anchored
  to the actual fill and the slipped entry price, causal — §8). Each bar, the first not-yet-spent
  exit whose array fires (in configured order) exits its ``scale_out_fraction`` at the bar's close;
  stops always exit the whole remaining position. A spent line never re-fires. A *touched* stop
  fills at its own level (the reference books the touched level, not the bar close).
- **Signal exits / flips**: a matching ``long_exit``/``short_exit`` (or an opposite-side entry)
  closes the position via a *signal* exit at the bar close; a flip then re-enters the opposite
  side on the same bar.
- **Accounting**: the strategy records every evaluated signal and submitted order into
  ``signal_records`` / ``order_records``; those (not the broker's cash) are what the adapter
  folds into the canonical ledger — the broker cash is only the *sizing* equity, and fills happen
  at raw bar closes (or touched levels), with ``commission`` and ``slippage`` applied by the fold
  (§8).
- **Final bar**: backtrader's broker executes a market order at the *next* bar's open, so an order
  placed on the last bar can never fill through the broker within the run. Because the reference
  fills it at the same bar's close, ``stop()`` books an in-flight exit or entry on the final bar
  at that bar's close (with its reason), so a final-bar signal still trades exactly like the
  references. A position with no exit signal is left *open* — the references keep the final trade
  unrealized (realized PnL 0), so the ledger ends on an open mark-to-market row rather than a
  reasonless close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime as _py_dt
from typing import Any, cast

import numpy as np
import pandas as pd
from backtrader import Order, Position
from backtrader import Strategy as BtStrategyBase

from ube.adapters.backtrader_adapter.exits import BtExitLine, build_exit_plan
from ube.adapters.backtrader_adapter.instrument_map import (
    BtInstrument,
    floor_to_increment,
)
from ube.core.cost import CostModel, fill_cost, slipped_price
from ube.core.data import MarketData
from ube.core.errors import EngineError
from ube.core.risk.exits import Exit, first_reached_exit
from ube.core.risk.sizing import _entry_fee_rate, size_position

__all__ = [
    "BtCarriedFill",
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
    starting_balance: float
    #: The position carried into this run by a paper-trading resume (§3) — seeded into the
    #: broker and the strategy's own tracker at ``start()``, never re-executed. ``None`` for
    #: a fresh backtest (and the cold paper window).
    carried: BtCarriedFill | None = None


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
    the bar it filled on — ``submit_bar`` for a bar-synchronous market order (same-bar close).
    ``price`` is the raw fill price (the fill bar's close, or a *touched* exit's own level);
    slippage is applied at the fold.

    ``fill_bar``/``price`` are ``None`` for an order submitted on the market's final bar that
    ``stop()`` could not book (defensive; the current ``stop()`` always realizes final-bar
    in-flight orders at the final close, so the fold shouldn't see one).
    """

    submit_bar: int
    fill_bar: int | None = None
    side: int = 0  # the fill direction (+1 buy / -1 sell)
    quantity: float = 0.0
    price: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BtCarriedFill:
    """The persisted fill of the position carried across a paper-trading resume (§3).

    The backtrader paper backend re-runs a window starting *at* the carried entry's fill
    bar; the strategy must hold that position from the first bar instead of re-executing
    it. Re-executing would (a) re-book the entry a bar later — double-booking the ledger —
    and (b) re-size it off the resume balance instead of its original quantity. ``price``
    is the raw (pre-slippage) entry fill price — the entry bar's close — which the exit
    plan and the fold slip exactly like every other fill.

    Attributes:
        side: The fill direction (+1 long / -1 short).
        quantity: The exact persisted fill size (positive units).
        price: The raw entry fill price from the entry bar's close.
    """

    side: int
    quantity: float
    price: float


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
    entry_price: float  # raw fill price (fill-bar close)
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
        # §6.3 sizing capital: the reference (nautilus) actor sizes off the *unleveraged*
        # realized net balance times the leverage — not the levered broker equity, which
        # accrues PnL on the leveraged base and diverges from the reference from the second
        # trade on. We track that balance ourselves: ``starting_balance`` plus the realized
        # net PnL of every exited portion, computed at slipped fill prices and the core
        # ``fill_cost`` fee (mirroring ``_net_pnl_per_unit`` in the vectorbt adapter), so
        # quantities match the reference row for row. PnL is realized at exit *submission*:
        # a flip sizes its re-entry in the same ``next()`` that submits the close, and the
        # reference has already booked the closing trade's PnL before sizing the flip leg.
        self._cash_balance = float(self._ctx.starting_balance)
        self.signal_records: list[BtSignalRecord] = []
        self.order_records: list[BtOrderRecord] = []

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _venue_price(self, side: int) -> float:
        """The price the broker will actually execute a market order at.

        backtrader's broker fills a market order at the *next* bar's open (the ``+1``
        line index, already loaded within the feed), not at the recorded close — so
        capacity and affordability checks must be priced at that open. On the final bar
        there is no next open inside this run (a same-bar entry is realized by
        ``stop()`` at the recorded close), so the recorded close is used instead.
        """
        if len(self.data) - 1 + 1 < self._ctx.data.n_bars:
            return float(self.data.open[1])
        return self._fill_price(side)

    def _snap_price(self, price: float) -> float:
        """Snap a float price to the instrument's price grid (§4.5 / nautilus parity).

        The nautilus reference rounds every bar price to ``price_precision`` when it
        ingests the feed (``adapt_data.py``), so its recorded fills carry clean decimals;
        backtrader would otherwise fold raw float64 bar values whose repr is burdened with
        arithmetic noise (``59995.100000000006``). Snapping bar-close fill prices to the
        same grid keeps the recorded trades byte-equal to the reference.
        """
        return float(f"{price:.{self._ctx.inst.price_precision}f}")

    def _fill_price(self, side: int) -> float:
        """Actual fill price for a market order: the current bar's close (§9.4 parity).

        The adapter's folded ledger fills at raw bar closes (or a *touched* exit's own
        level) regardless of the broker's internal next-open execution, so the price used
        for sizing and realized PnL is always available within the decision bar's
        ``next()``.
        """
        return float(
            slipped_price(self._snap_price(float(self.data.close[0])), side, self._ctx.slip)
        )

    def _target_quantity(self, side: int) -> float:
        """Core-sized entry quantity for one direction (§6.3), floored to the lot increment."""
        ctx = self._ctx
        price = self._fill_price(side)
        # §6.3 capital base: the *unleveraged* net cash balance scaled by the account
        # leverage (the reference ``capital = balance × leverage``), tracked per fill in
        # ``notify_order`` — not ``broker.getvalue()``, which is the levered equity that
        # accrues PnL on the leveraged base. The sizer's own ``leverage`` knob is not
        # re-applied here (the base is already leveraged), matching the nautilus actor.
        equity = self._cash_balance * ctx.eff_leverage
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
        # and reserve the entry commission, so the bookable notional *plus* fees can
        # never exceed cash/leverage and the venue cannot spuriously reject a sized-for
        # order. backtrader's broker still executes a market order at the *next* bar's
        # open (the ``+1`` line index, already loaded) even though the folded ledger
        # re-stamps the fill at the submit bar's close, so the capacity hurdle is priced
        # at that next open — sizing against the close under-prices the venue and trips
        # its Margin check on a tight-cash round trip. The sizing capital is already
        # leveraged (``cash_balance × eff_leverage``, see ``_cash_balance``) — the
        # sizer's own ``leverage`` knob is not re-applied here. Cash-like classes book
        # the full notional; futures-like book margin = notional/lev via automargin
        # ``1/lev``.
        fee_rate = (
            _entry_fee_rate(ctx.cost_model) if ctx.cost_model is not None else 0.0
        )
        unit_cost = self._venue_price(side) * (1.0 + fee_rate)
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
            # Priced at the recorded fill (the submit bar's close), so the shortfall
            # check mirrors what the folded ledger actually books. The sizing capacity
            # in ``_target_quantity`` is the venue-side guard priced at the broker's
            # next-bar-open execution; re-pricing the *purchasing power* here at that
            # open as well would phantom-reject a same-bar flip whose exit is credited
            # at the close while the entry executes at the next open.
            notional = qty * self._fill_price(side)
            required = notional + float(fill_cost(ctx.cost_model, notional=notional))
            capacity = self._cash_balance * ctx.eff_leverage
            if required > capacity * (1.0 + 1e-9):
                raise EngineError(
                    f"market order rejected by the venue: insufficient funds at bar "
                    f"{len(self.data) - 1} — requires {required:.6g}, available "
                    f"{capacity:.6g} {ctx.inst.settlement_currency}"
                )

    def _realize_exit_pnl(
        self,
        exit_side: int,
        qty: float,
        level_price: float | None = None,
    ) -> None:
        """Book the realized net PnL of an exiting portion into ``_cash_balance`` (§6.3).

        Mirrors the reference actor, which realizes a closing trade's net PnL (gross at
        slipped fill prices minus the core ``fill_cost`` entry and exit fees) when it
        closes — here at exit submission so a same-bar flip sizes its new leg on the
        already-realized balance, exactly like the reference does before sizing the flip.
        The entry leg is matched against the open position's raw fill price; both prices
        are slipped the same way the fold slips them.

        ``level_price`` is a *touched* risk exit's own level. §9.4 parity: the reference
        fills a touched stop/target **at its level**, slipped like every other fill, not at
        the bar close — so the level is what has to be priced here, or the realized PnL (and
        therefore the running balance) drifts from the reference. Pass ``None`` to keep
        bar-close pricing, which is correct for signal closes, flips, and level-less exits
        (``TimeExit`` / ``trigger="close"``).
        """
        ctx = self._ctx
        if self._pos is None or qty <= _FLAT_EPS or ctx.cost_model is None:
            return
        pos = self._pos
        cost_model = ctx.cost_model
        mult = ctx.inst.contract_multiplier
        entry_slip = float(slipped_price(pos.entry_price, pos.side, ctx.slip))
        exit_slip = (
            self._fill_price(exit_side)
            if level_price is None
            else float(slipped_price(float(level_price), exit_side, ctx.slip))
        )
        if not np.isfinite(exit_slip):
            return
        gross = pos.side * (exit_slip - entry_slip) * qty * mult
        entry_fee = float(fill_cost(cost_model, notional=qty * entry_slip * mult))
        exit_fee = float(fill_cost(cost_model, notional=qty * exit_slip * mult))
        self._cash_balance += gross - entry_fee - exit_fee

    # ------------------------------------------------------------------
    # Start / resume seeding
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Seed the broker and position tracker with a carried position on resume (§3).

        The paper backend passes ``ctx.carried`` on a warm window whose open position was
        persisted by an earlier step. backtrader must hold it from bar 0 — never re-submit
        the entry order — or the fill would re-book a bar later (double-booking the ledger,
        and the position itself) and be re-sized off the resume balance instead of carried
        at its original quantity. Mirror the reference (nautilus) actor, which seeds its
        own position tracker from the persisted open position and only books *new* fills.
        """
        carried = self._ctx.carried
        if carried is None or float(carried.quantity) <= _FLAT_EPS:
            return
        ctx = self._ctx
        raw = float(carried.price)
        pos = Position(size=float(carried.side) * float(carried.quantity), price=raw)
        # The broker's credit-interest pass reads ``pos.datetime.date()`` on every bar, so
        # the seeded position must carry a fill-time ``datetime`` like broker-executed ones
        # (``Position.update`` records it); stamp the entry bar's timestamp.
        pos.datetime = _py_dt.fromtimestamp(
            float(int(pd.Timestamp(ctx.data.timestamps[0]).value)) / 1_000_000_000.0,
            tz=UTC,
        ).replace(tzinfo=None)
        # Anchor the seeded position at the entry bar's close so the broker's end-of-bar
        # ``cashadjust`` steps reproduce the full run's from this bar on (the absolute cash
        # level is sizing-equity only — the fold never reads the broker).
        pos.adjbase = float(ctx.data.close[0])
        self.broker.positions[self.data] = pos
        plan = build_exit_plan(
            ctx.exits,
            ctx.data,
            side=int(carried.side),
            entry_price=float(slipped_price(raw, carried.side, ctx.slip)),
            entry_bar=0,
            atr_map=ctx.atr_map,
        )
        self._pos = _OpenPosition(
            side=int(carried.side),
            entry_bar=0,
            entry_price=raw,
            plan=plan,
        )

    # ------------------------------------------------------------------
    # Next / per-bar event loop
    # ------------------------------------------------------------------

    def next(self) -> None:
        i = len(self.data) - 1
        ctx = self._ctx

        # --- Exits: risk plan first, then signal exit / flip -------------
        if (
            self._pos is not None
            and self._pending_exit is None
            and abs(self.position.size) > _FLAT_EPS
        ):
            fired: BtExitLine | _SignalExit | None = None
            due = [
                line
                for line in self._pos.plan
                if line.index not in self._pos.spent and bool(line.triggered[i])
            ]
            if due:
                # Several exits can trigger on one bar; the one the bar reached first wins
                # (§4.7/§8). Level-less exits (TimeExit, trigger="close") fill at the close,
                # so they rank on the close price.
                ranked = [
                    float(line.level[i]) if line.level is not None else float(self.data.close[0])
                    for line in due
                ]
                pick = first_reached_exit(ranked, open_price=float(self.data.open[0]))
                fired = due[0 if pick is None else pick]
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
                    # A *touched* risk exit fills at its own level (§9.4 parity with the
                    # reference), not at the bar close. ``level`` is None for level-less
                    # exits (TimeExit / trigger="close") and for signal closes, which
                    # correctly book bar-close pricing.
                    level_price = (
                        None
                        if isinstance(fired, _SignalExit) or fired.level is None
                        else float(fired.level[i])
                    )
                    self._realize_exit_pnl(exit_side, exit_qty, level_price)
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
                        "level_price": level_price,
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
        fill_bar = submit_bar  # same-bar close fills (§9.4 parity with the references)
        side = int(meta["side"])
        qty = abs(float(order.executed.size))
        # Bar-synchronous pricing: the folded ledger fills a market order at the *submit*
        # bar's close (the reference's same-bar fill), not at the broker's internal
        # next-bar-open execution, so re-stamp the executed price here (snapped to the
        # instrument's price grid like the reference's ingested bars).
        price = self._snap_price(float(ctx.data.close[submit_bar]))
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
            # A touched risk exit is booked at its own level (slipped like every other
            # fill), not at the submit bar's close — §9.4 parity with the reference. Signal
            # closes and level-less exits keep the bar-close ``price`` stamped above.
            level_price = meta.get("level_price")
            if level_price is not None:
                price = float(slipped_price(float(level_price), side, ctx.slip))
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
        """Realize an order still in flight on the final bar; otherwise hold the position open.

        backtrader's broker executes a market order at the *next* bar's open, so an order
        submitted on the very last bar cannot fill through the broker within the run. The
        reference fills it at the same bar's close, so an exit or entry still in flight is
        booked here at the raw final close (the fold applies slippage like every other
        fill); an exit keeps its reason and any remaining open quantity is held (partial
        scale-outs). This holds for both a standalone backtest and a paper seam window —
        the carried position continues into the next window from the fill bar, exactly as
        the reference backends carry it, so the windowed ledger is identical to a single
        full-window run.

        Any remaining open quantity with no exit signal is *not* force-closed — the
        references leave a position with no exit signal open (realized PnL 0), so the
        fold's running net stays non-zero and the ledger reproduces an open mark-to-market
        row.
        """
        last = self._ctx.data.n_bars - 1
        close_price = self._snap_price(float(self.data.close[0]))
        pending = self._pending_exit
        if pending is not None:
            self.order_records.append(
                BtOrderRecord(
                    last,
                    last,
                    int(pending["side"]),
                    float(pending["size"]),
                    close_price,
                    pending.get("reason"),
                )
            )
            self._pending_exit = None
            if abs(self.position.size) - float(pending["size"]) <= _FLAT_EPS:
                self._pos = None
        pending_entry = self._pending_entry
        if pending_entry is not None:
            self.order_records.append(
                BtOrderRecord(
                    last,
                    last,
                    int(pending_entry["side"]),
                    float(pending_entry["size"]),
                    close_price,
                    None,
                )
            )
            self._pending_entry = None
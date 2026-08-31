"""Bridge: nautilus fill events -> ube ``LedgerEvent``s (plan T3 / §4.3, §4.6).

Keeps nautilus types out of ``core`` (A4.1). Reported commissions are computed by
``core.cost.fill_cost`` (not Nautilus's number) so the paper ledger is reproducible across
engines (nautilus plan A5.3 / §9.4). Per-fill ordering here matches the recording backend:
``FILL`` then ``COMMISSION`` then ``POSITION_CHANGE``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ube.core.cost import fill_cost
from ube.core.ledger import EventType, LedgerEvent

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled


def fill_event(
    fill: OrderFilled,
    instrument_id: str,
    *,
    exit_reason: str | None,
    ts_override: int | None = None,
    multiplier: float = 1.0,
) -> LedgerEvent:
    """A single ube ``FILL`` ledger event from a nautilus ``OrderFilled``.

    ``ts_override`` lets the caller supply the historical (test-clock) timestamp: the
    sandbox bars carry live-clock timestamps (so orders match correctly), but the ube
    ledger must stay on the test-clock timeline for §9.4 comparability.
    ``multiplier`` is the contract multiplier (e.g. 50 for ES futures) — notional is
    ``qty * price * multiplier`` like the backtest (§4.6).
    """
    t = int(ts_override) if ts_override is not None else int(fill.ts_init)
    qty = float(fill.last_qty.as_double())
    price = float(fill.last_px.as_double())
    notional = qty * price * multiplier
    side = 1 if fill.is_buy else -1
    return LedgerEvent(
        EventType.FILL,
        t,
        instrument_id,
        side=side,
        quantity=qty,
        price=price,
        notional=notional,
        order_id=str(fill.client_order_id),
        exit_reason=exit_reason,
    )


def cash_movement_event(
    instrument_id: str,
    *,
    amount: float,
    currency: str,
    ts_override: int | None = None,
) -> LedgerEvent:
    """A ube ``CASH_MOVEMENT`` ledger event — the cash leg of the equity curve (§4.6 step 4).

    ``amount`` is signed (inflow positive); a buy pays notional out of the account
    (``amount = -side * notional``), a sell pays it back in. Mirrors the recording backend's
    cash fold so the paper ledger is self-financing and comparable to mode A (§9.4).
    """
    t = int(ts_override) if ts_override is not None else 0
    return LedgerEvent(
        EventType.CASH_MOVEMENT, t, instrument_id, amount=amount, currency=currency
    )


def commission_event(
    fill: OrderFilled,
    instrument_id: str,
    cost_model: Any,
    *,
    ts_override: int | None = None,
    multiplier: float = 1.0,
    currency: str = "USD",
) -> LedgerEvent | None:
    """A ube ``COMMISSION`` ledger event computed via ``core.cost.fill_cost``.

    ``multiplier`` and ``currency`` mirror the backtest — notional includes the
    contract multiplier and commission is booked in the instrument's settlement
    currency, not hardcoded USD.
    """
    t = int(ts_override) if ts_override is not None else int(fill.ts_init)
    qty = float(fill.last_qty.as_double())
    price = float(fill.last_px.as_double())
    notional = qty * price * multiplier
    commission = float(fill_cost(cost_model, notional=notional))
    if commission <= 0.0:
        return None
    return LedgerEvent(
        EventType.COMMISSION, t, instrument_id, amount=commission, currency=currency
    )


def position_change_event(
    fill: OrderFilled,
    instrument_id: str,
    *,
    side: int,
    position_after: float,
    ts_override: int | None = None,
) -> LedgerEvent:
    """A ube ``POSITION_CHANGE`` ledger event after a fill resolved the net side."""
    t = int(ts_override) if ts_override is not None else int(fill.ts_init)
    return LedgerEvent(
        EventType.POSITION_CHANGE,
        t,
        instrument_id,
        side=side,
        position_after=position_after,
    )


__all__ = [
    "fill_event",
    "cash_movement_event",
    "commission_event",
    "position_change_event",
]

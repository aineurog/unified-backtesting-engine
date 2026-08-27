"""Bridge: nautilus fill events -> ube ``LedgerEvent``s (plan T3 / §4.3, §4.6).

Keeps nautilus types out of ``core`` (A4.1). Reported commissions are computed by
``core.cost.fill_cost`` (not Nautilus's number) so the paper ledger is reproducible across
engines (nautilus plan A5.3 / §9.4). Per-fill ordering here matches the recording backend:
``FILL`` then ``COMMISSION`` then ``POSITION_CHANGE``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ube.core.cost import fill_cost
from ube.core.ledger import EventType, LedgerEvent

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled


def fill_event(
    fill: OrderFilled,
    instrument_id: str,
    *,
    exit_reason: str | None,
) -> LedgerEvent:
    """A single ube ``FILL`` ledger event from a nautilus ``OrderFilled``."""
    t = int(fill.ts_init)
    qty = float(fill.last_qty.as_double())
    price = float(fill.last_px.as_double())
    notional = qty * price
    side = 1 if fill.is_buy else -1
    return LedgerEvent(
        EventType.FILL,
        t,
        instrument_id,
        side=side,
        quantity=qty,
        price=price,
        notional=notional,
        exit_reason=exit_reason,
    )


def commission_event(
    fill: OrderFilled,
    instrument_id: str,
    cost_model,
) -> LedgerEvent | None:
    """A ube ``COMMISSION`` ledger event computed via ``core.cost.fill_cost``."""
    t = int(fill.ts_init)
    qty = float(fill.last_qty.as_double())
    price = float(fill.last_px.as_double())
    notional = qty * price
    commission = float(fill_cost(cost_model, notional=notional))
    if commission <= 0.0:
        return None
    return LedgerEvent(
        EventType.COMMISSION, t, instrument_id, amount=commission, currency="USD"
    )


def position_change_event(
    fill: OrderFilled,
    instrument_id: str,
    *,
    side: int,
    position_after: float,
) -> LedgerEvent:
    """A ube ``POSITION_CHANGE`` ledger event after a fill resolved the net side."""
    t = int(fill.ts_init)
    return LedgerEvent(
        EventType.POSITION_CHANGE,
        t,
        instrument_id,
        side=side,
        position_after=position_after,
    )


__all__ = ["fill_event", "commission_event", "position_change_event"]

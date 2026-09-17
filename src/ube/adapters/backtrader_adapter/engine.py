"""backtrader execution (mirrors ``engine.py`` for the vectorbt adapter).

This module wires the translated signal frame (§6.1) and the strategy context
(:mod:`~ube.adapters.backtrader_adapter.strategy`) into a ``cerebro`` run: it builds the
``PandasData``-derived feed with the signal lines, configures the broker (starting cash, the
commission scheme — fraction commission via ``COMM_PERC``/``percabs``, futures-like margin for
leveraged classes, full notional for cash-like classes), and runs the event loop. The returned
strategy instance carries the ordered ``signal_records`` / ``order_records`` the adapter folds
into the canonical ledger.

The broker cash is *only* the sizing equity — fills are recorded at raw open prices and the
legate flows (commission, slippage, funding) are computed by the fold (§8), so the engine never
emits its own ledger rows.
"""

from __future__ import annotations

from typing import cast

from backtrader import Cerebro, CommInfoBase, feeds

from ube.adapters.backtrader_adapter.adapt_data import BtSignalFrame
from ube.adapters.backtrader_adapter.strategy import (
    BacktraderStrategy,
    BtRunContext,
)
from ube.core.errors import EngineError

__all__ = ["BtSignalFeed", "make_comminfo", "run_backtrader"]


class BtSignalFeed(feeds.PandasData):  # type: ignore[misc]  # untyped backtrader feed
    """One instrument's bar feed with the four signal lines (§6.1)."""

    lines = ("long_entry", "long_exit", "short_entry", "short_exit")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("long_entry", "long_entry"),
        ("long_exit", "long_exit"),
        ("short_entry", "short_entry"),
        ("short_exit", "short_exit"),
    )

    _ube_ctx: BtRunContext


def make_comminfo(
    ctx: BtRunContext,
) -> CommInfoBase:
    """The broker commission scheme for one instrument (§8).

    Commission is a flat fraction of notional (``COMM_PERC`` + ``percabs=True``), from the
    core ``CostModel.commission``. Leveraged classes (crypto perps / forex margin accounts) use
    a futures-like contract with ``automargin = margin_init / leverage`` — the reference
    ``DispatchCommInfo`` policy — so a position sized to ``equity * leverage`` requires exactly
    ``equity * margin_init`` of cash and can never be rejected by the venue. Cash-like classes
    book the full notional as margin.
    """
    margin_init = ctx.margin_account
    fee = float(ctx.cost_model.commission) if ctx.cost_model is not None else 0.0
    if margin_init is None:
        return CommInfoBase(
            commtype=CommInfoBase.COMM_PERC,
            percabs=True,
            commission=fee,
            stocklike=True,
        )
    leverage = max(float(ctx.eff_leverage), 1.0)
    return CommInfoBase(
        commtype=CommInfoBase.COMM_PERC,
        percabs=True,
        commission=fee,
        stocklike=False,
        automargin=margin_init / leverage,
        leverage=leverage,
    )


def run_backtrader(
    frame: BtSignalFrame,
    ctx: BtRunContext,
    *,
    starting_cash: float,
) -> BacktraderStrategy:
    """Build cerebro, run it, and return the strategy holding the records (§6.1).

    ``ctx`` rides on the feed instance (backtrader keeps the feed object), so the strategy
    picks it up untouched. ``starting_cash`` is the *leveraged* account cash used for sizing
    equity only — the fold books the unleveraged ``starting_balance`` (§4.6).

    Raises:
        EngineError: If backtrader is absent or the run raises for any reason.
    """
    feed = BtSignalFeed(dataname=frame.feed_frame)
    feed._ube_ctx = ctx
    cerebro = Cerebro(stdstats=False)
    cerebro.adddata(feed, name="bt")
    cerebro.broker.setcash(float(starting_cash))
    cerebro.broker.addcommissioninfo(make_comminfo(ctx))
    cerebro.addstrategy(BacktraderStrategy)
    try:
        run = cerebro.run()
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap whatever backtrader raised
        raise EngineError(f"backtrader backtest failed: {exc}") from exc
    return cast(BacktraderStrategy, run[0])
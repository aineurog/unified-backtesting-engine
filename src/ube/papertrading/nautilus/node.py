"""TradingNode wiring for the ube ``MarketData``-backed sandbox paper trader (plan T3).

Ports ``sim_nautilus.node``: a ``UbeDataClient`` (pushes ube bars) plus a Nautilus
``SandboxExecutionClient`` (``bar_execution=True``, ``reject_stop_orders=False``,
``oms_type="NETTING"``, ``account_type="MARGIN"``) executing MARKET orders against the bars.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import LoggingConfig
from nautilus_trader.live.node import TradingNode, TradingNodeConfig
from nautilus_trader.model.enums import OrderStatus

from .data_client import UbeDataClient, UbeDataClientConfig, UbeDataClientFactory
from .signals import SIGNAL_REGISTRY


def build_node(
    *,
    instrument,
    bars: list[dict],
    signal_map: dict[int, tuple],
    bar_type: str,
    balance: float,
    quote: str,
    venue: str,
    leverage: float = 1.0,
    strategy=None,
    overrides: dict[str, Any] | None = None,
) -> TradingNode:
    """Assemble a :class:`TradingNode` driving the ube data client + sandbox exec client.

    ``account_type`` / ``oms_type`` have a single source of truth: the
    :class:`~ube.adapters.nautilus_adapter.overrides.NautilusEngineOverrides` (plan
    blocker #11). They are read from ``overrides`` (lowercase ``"margin"/"cash"`` and
    ``"NETTING"/"HEDGING"``) and mapped to the sandbox's uppercase enum spellings.
    """
    overrides = overrides or {}
    _acct = overrides.get("account_type", "margin")
    account_type = "MARGIN" if _acct == "margin" else "CASH"
    oms_type = overrides.get("oms_type", "NETTING")
    data_config = UbeDataClientConfig(
        bars=bars,
        signal_map=signal_map,
        bar_type=bar_type,
        instrument_id=str(instrument.id),
        venue=venue,
    )
    exec_config = SandboxExecutionClientConfig(
        venue=venue,
        starting_balances=[f"{balance} {quote}"],
        base_currency=quote,
        oms_type=oms_type,
        account_type=account_type,
        default_leverage=Decimal(str(leverage)),
        book_type="L1_MBP",
        frozen_account=False,
        bar_execution=True,
        trade_execution=True,
        reject_stop_orders=False,
        support_gtd_orders=True,
        support_contingent_orders=True,
        use_reduce_only=False,
    )
    node_config = TradingNodeConfig(
        trader_id="PAPER-001",
        data_clients={"MYDATA": data_config},
        exec_clients={"SANDBOX": exec_config},
        logging=LoggingConfig(log_level="ERROR", log_colors=False),
        timeout_connection=5.0,
        timeout_reconciliation=2.0,
        timeout_portfolio=2.0,
        timeout_disconnection=2.0,
        timeout_post_stop=2.0,
        timeout_shutdown=2.0,
    )
    node = TradingNode(config=node_config)
    node.add_data_client_factory("MYDATA", UbeDataClientFactory)
    node.add_exec_client_factory("SANDBOX", SandboxLiveExecClientFactory)
    node.build()
    if strategy is not None:
        node.trader.add_strategy(strategy)
    node.cache.add_instrument(instrument)
    return node


def run_node(node: TradingNode) -> None:
    """Run the node until the data client finishes, settle fills, then stop (one-shot)."""
    loop = node.get_event_loop()

    async def _settle() -> None:
        transient = {OrderStatus.INITIALIZED, OrderStatus.SUBMITTED}
        for _ in range(100):  # up to ~10s
            await asyncio.sleep(0.1)
            published = SIGNAL_REGISTRY.bars_published
            if published > 0 and SIGNAL_REGISTRY.bars_processed >= published:
                orders = node.cache.orders(instrument_id=None)
                if all(o.status not in transient for o in orders):
                    return

    async def _stop_when_done() -> None:
        ev = UbeDataClient.DONE
        while ev is None:
            await asyncio.sleep(0.05)
            ev = UbeDataClient.DONE
        await ev.wait()
        await _settle()
        node.stop()

    loop.call_soon(lambda: loop.create_task(_stop_when_done()))
    node.run()
    node.dispose()


__all__ = ["build_node", "run_node"]

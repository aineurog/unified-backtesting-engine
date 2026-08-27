"""Custom ``LiveMarketDataClient`` that drains a slice of ube ``MarketData`` + ``Signals``.

Ports ``sim_nautilus.data_client.GeneratorDataClient``: instead of polling an external
generator it drains the in-memory ube bars passed to ``ube.paper.step`` (deterministic
replay, plan §4.3 / T3). Each bar is converted to a Nautilus ``Bar``, delivered to the
``DataEngine`` (strategy subscriptions via ``_handle_data``) and published on the venue
topic so the ``SandboxExecutionClient`` can match resting orders against it (``bar_execution``).
The bar's 4-column signal is registered for the strategy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nautilus_trader.live.data_client import MarketDataClient
from nautilus_trader.live.factories import LiveDataClientConfig, LiveDataClientFactory
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import ClientId, InstrumentId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity

from .runtime import get_ready_event
from .signals import SIGNAL_REGISTRY

BAR_AGGREGATION = "LAST"
BAR_SPEC = "EXTERNAL"


class UbeDataClientConfig(LiveDataClientConfig):
    """Configuration for the ube ``MarketData``-backed data client."""

    bars: list[dict[str, Any]] = []  # each: {ts_ns, open, high, low, close, volume}
    signal_map: dict[int, tuple] = {}  # ts_ns -> (le, lx, se, sx)
    bar_type: str = ""
    instrument_id: str = ""
    venue: str = ""


class UbeDataClient(MarketDataClient):  # type: ignore[misc]
    """Pushes ube ``MarketData`` bars into the node in a single shot."""

    DONE: asyncio.Event | None = None  # set after the one-shot drain completes

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_id: ClientId,
        venue: Venue | None,
        msgbus,
        cache,
        clock,
        config: UbeDataClientConfig,
    ) -> None:
        super().__init__(
            client_id=client_id,
            venue=venue,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )
        self._bars = list(config.bars)
        self._signal_map = dict(config.signal_map)
        self._bar_type = BarType.from_str(config.bar_type)
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._venue = config.venue
        self._instrument: Instrument | None = None
        self._poll_task: asyncio.Task | None = None
        if type(self).DONE is None:
            type(self).DONE = asyncio.Event()

    # -- lifecycle --------------------------------------------------------- #
    def connect(self) -> None:
        self._cache.add_instrument(self._instrument_cached())
        self._log.info("Connecting UbeDataClient")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # pragma: no cover - defensive
            loop = asyncio.new_event_loop()
        self._poll_task = loop.create_task(self._connect())
        self._set_connected(True)

    async def _connect(self) -> None:
        self._instrument = self._instrument_cached()
        # Wait until the strategy has subscribed to the bar type (nautilus 1.231 removed
        # ``subscribed_bars`` from the client). Best-effort timeout so a misconfigured
        # run cannot hang forever.
        if get_ready_event() is not None:
            try:
                await asyncio.wait_for(get_ready_event().wait(), timeout=5.0)
            except TimeoutError:  # pragma: no cover - defensive
                self._log.warning("Timed out waiting for strategy subscription")
        await self._drain()
        self._log.info("Single-shot ube bar drain complete")

    async def _disconnect(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()

    def disconnect(self) -> None:
        self._set_connected(False)

    def _instrument_cached(self) -> Instrument:
        instr = self._cache.instrument(self._instrument_id)
        if instr is None:  # pragma: no cover - defensive
            raise RuntimeError(f"instrument {self._instrument_id} not found in cache")
        return instr

    # -- single-shot drain ------------------------------------------------ #
    async def _drain(self) -> None:
        try:
            for row in self._bars:
                ts_ns = int(row["ts_ns"])
                bar = self._to_bar(ts_ns, row)
                self._handle_data(bar)
                self._publish_to_venue(bar)
                le, lx, se, sx = self._signal_map.get(
                    ts_ns, (False, False, False, False)
                )
                SIGNAL_REGISTRY.register(ts_ns, le, lx, se, sx)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._log.error(f"UbeDataClient drain failed: {exc}")
        finally:
            if type(self).DONE is not None and not type(self).DONE.is_set():
                type(self).DONE.set()

    def _publish_to_venue(self, bar: Bar) -> None:
        topic = f"data.{type(bar).__name__}.{self._venue}.{self._instrument_id.symbol}"
        self._msgbus.publish(topic=topic, msg=bar)

    def _to_bar(self, ts_ns: int, row: dict[str, Any]) -> Bar:
        instr = self._instrument
        pp = instr.price_precision
        sp = instr.size_precision
        return Bar(
            bar_type=self._bar_type,
            open=Price(float(row["open"]), pp),
            high=Price(float(row["high"]), pp),
            low=Price(float(row["low"]), pp),
            close=Price(float(row["close"]), pp),
            volume=Quantity(float(row.get("volume", 0.0)), sp),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )

    # -- instrument serving ----------------------------------------------- #
    def request_instrument(self, request) -> None:
        self._handle_instrument(self._instrument_cached(), None, None, None, None)

    def request_instruments(self, request) -> None:
        self._handle_instruments(
            request.venue, [self._instrument_cached()], None, None, None, None
        )

    def subscribe_bars(self, command) -> None:
        # Push-based client: subscription is recorded by the DataEngine; we publish bars
        # directly via ``_handle_data`` + the venue topic. Nothing to do remotely.
        return None

    def unsubscribe_bars(self, command) -> None:
        return None


class UbeDataClientFactory(LiveDataClientFactory):
    """Node factory creating the ube ``MarketData``-backed data client."""

    @staticmethod
    def create(loop, name: str, config, msgbus, cache, clock):
        return UbeDataClient(
            loop=loop,
            client_id=ClientId(f"{name}-001"),
            venue=Venue(config.venue) if config.venue else None,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )  # type: ignore[call-arg]


__all__ = ["UbeDataClient", "UbeDataClientConfig", "UbeDataClientFactory"]

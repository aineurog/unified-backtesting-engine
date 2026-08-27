"""Nautilus paper backend (plan T3 / §4.3).

:class:`NautilusPaperEngine` drives a real Nautilus ``TradingNode`` whose
``SandboxExecutionClient`` executes MARKET orders against ube ``MarketData`` bars. The
strategy bridges fills into ube ``LedgerEvent``s returned to ``core.step``. Nautilus is the
*execution substrate* (§0) — all decisions/sizing use ``core`` (no re-implementation).

Importing this module self-registers the ``"nautilus"`` backend, so ``core`` never hard-depends
on nautilus-trader (lazy import, A5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ube.adapters.nautilus_adapter.adapt_data import build_bar_type
from ube.adapters.nautilus_adapter.instrument_map import build_instrument
from ube.core.cost import resolve_cost_model
from ube.core.errors import EngineError
from ube.core.instrument import Instrument, allows_short
from ube.core.ledger import LedgerEvent
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
        try:
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
            bars: list[dict[str, Any]] = []
            signal_map: dict[int, tuple] = {}
            for i in range(n):
                t = int(ts[i])
                bars.append(
                    {
                        "ts_ns": t,
                        "open": float(data.open[i]),
                        "high": float(data.high[i]),
                        "low": float(data.low[i]),
                        "close": float(data.close[i]),
                        "volume": float(data.volume[i]),
                    }
                )
                signal_map[t] = (
                    bool(signals.long_entry[i]),
                    bool(signals.long_exit[i]),
                    bool(signals.short_entry[i]),
                    bool(signals.short_exit[i]),
                )

            period_ns = int(np.median(np.diff(ts))) if n > 1 else 60_000_000_000
            bar_type = build_bar_type(instrument_id, period_ns)

            balance = float(
                config.starting_balance
                if config.starting_balance is not None
                else overrides.get("starting_balance", 10_000.0)
            )
            leverage = 1.0 if no_short else float(overrides.get("leverage", 1.0))
            quote = str(getattr(instrument, "settlement_currency", "USDT"))

            reset_ready_event()
            strat_cfg = UbePaperConfig(
                instrument_id=str(instrument_id),
                bar_type=str(bar_type),
                sizing=config.base.risk.sizing,
                cost_model=cost_model,
                no_short=no_short,
                on_opposite_signal=str(config.base.signal.on_opposite_signal),
                balance=balance,
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
            return list(strategy.events)
        except Exception as exc:  # pragma: no cover - defensive
            raise EngineError(
                f"nautilus paper backend failed: {exc}"
            ) from exc


register_paper_engine("nautilus", NautilusPaperEngine)

__all__ = ["NautilusPaperEngine"]

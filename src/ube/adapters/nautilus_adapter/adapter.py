"""NautilusTrader engine adapter (§4.5)."""

from collections.abc import Mapping

from ube.adapters.base import EngineAdapter
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.result import BacktestResult
from ube.core.signals import Signals


class NautilusAdapter(EngineAdapter):
    """Adapter for the NautilusTrader backtesting engine."""

    def run(
        self,
        data: MarketData,
        signals: Signals,
        config: BacktestConfig,
        *,
        aux_data: Mapping[str, MarketData] | None = None,
    ) -> BacktestResult:
        """Run a backtest via NautilusTrader (§4.5)."""
        raise NotImplementedError
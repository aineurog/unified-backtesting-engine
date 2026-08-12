"""NautilusTrader engine adapter."""

from ube.adapters.base import EngineAdapter


class NautilusAdapter(EngineAdapter):
    """Adapter for the NautilusTrader backtesting engine."""

    def run(self, *args: object, **kwargs: object) -> object:
        """Run a backtest via NautilusTrader."""
        raise NotImplementedError

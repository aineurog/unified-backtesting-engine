"""backtrader engine adapter."""

from ube.adapters.base import EngineAdapter


class BacktraderAdapter(EngineAdapter):
    """Adapter for the backtrader backtesting engine."""

    def run(self, *args: object, **kwargs: object) -> object:
        """Run a backtest via backtrader."""
        raise NotImplementedError

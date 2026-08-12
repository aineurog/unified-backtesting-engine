"""vectorbt engine adapter."""

from ube.adapters.base import EngineAdapter


class VectorbtAdapter(EngineAdapter):
    """Adapter for the vectorbt backtesting engine."""

    def run(self, *args: object, **kwargs: object) -> object:
        """Run a backtest via vectorbt."""
        raise NotImplementedError

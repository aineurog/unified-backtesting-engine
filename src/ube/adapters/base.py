"""Engine adapter abstract base class."""

from abc import ABC, abstractmethod


class EngineAdapter(ABC):
    """Abstract base class for backtesting engine adapters."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run a backtest; each engine adapter implements its own execution."""
        raise NotImplementedError

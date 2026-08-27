"""Paper-trading configuration — a thin wrapper over :class:`BacktestConfig` (§9.1).

Paper trading reuses the canonical :class:`~ube.core.config.BacktestConfig` as the
single source of trading params (instrument, risk/exits, cost, engine overrides) — there
is no second, parallel config (§4.2). :class:`PaperConfig` only adds the few paper-only
fields the backtest contract does not carry: the position-change policy (§9.3), the
sqlite state path (§9.5), and a convenience ``starting_balance`` that flows into the
engine overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ube.core.config import BacktestConfig

__all__ = ["PaperConfig"]

OnOppositeSignal = Literal["reverse", "exit_only", "ignore"]


@dataclass(frozen=True)
class PaperConfig:
    """Configuration for a paper-trading session (§9.1, §9.3, §9.5).

    Attributes:
        base: The canonical :class:`~ube.core.config.BacktestConfig` (instrument,
            risk/exits, cost model, engine overrides). Engine is ``"nautilus"`` for the
            nautilus backend, implied here and not re-stored.
        on_opposite_signal: Position-change policy (§9.3) when a signal opposes the open
            position — ``reverse`` (close + reopen in the new direction), ``exit_only``
            (close, do not reopen), or ``ignore`` (hold the position).
        state_path: sqlite path for :class:`PaperState` persistence (§9.5). ``None`` means
            the run is ephemeral (no resume).
        starting_balance: Convenience override of the venue starting balance; folded into
            ``base.engine_overrides["starting_balance"]`` so the nautilus sandbox seeds the
            account (default ``None`` → the adapter's ``DEFAULT_STARTING_BALANCE``).
        engine: The registered paper engine name (``"nautilus"`` by default). The
            registry is mirrored from :mod:`ube.adapters.base`.
    """

    base: BacktestConfig
    on_opposite_signal: OnOppositeSignal = "reverse"
    state_path: str | None = None
    starting_balance: float | None = None
    engine: str = "nautilus"

    def __post_init__(self) -> None:
        if self.on_opposite_signal not in ("reverse", "exit_only", "ignore"):
            raise ValueError(
                f"on_opposite_signal must be reverse/exit_only/ignore; "
                f"got {self.on_opposite_signal!r}"
            )
        if self.starting_balance is not None and self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        # Fold the convenience balance into the engine overrides so the backend sees a
        # single canonical config (no second source of truth for the balance).
        if self.starting_balance is not None:
            overrides: dict[str, Any] = (
                dict(self.base.engine_overrides) if self.base.engine_overrides else {}
            )
            overrides["starting_balance"] = self.starting_balance
            object.__setattr__(self.base, "engine_overrides", overrides)

    @property
    def overrides(self) -> dict[str, Any]:
        """The engine overrides, as a plain dict (never ``None``)."""
        return dict(self.base.engine_overrides) if self.base.engine_overrides else {}

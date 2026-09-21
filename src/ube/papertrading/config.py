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
from typing import Any

from ube.core.calendar import TradingCalendar, resolve_calendar
from ube.core.config import BacktestConfig

__all__ = ["PaperConfig"]


@dataclass(frozen=True)
class PaperConfig:
    """Configuration for a paper-trading session (§9.1, §9.3, §9.5).

    The position-change policy (§9.3 ``on_opposite_signal``) lives on the canonical
    :class:`~ube.core.config.BacktestConfig.signal` — it is *not* re-declared here (single
    source of truth, plan blocker #1). :meth:`BacktestConfig.validate(paper_trading=True)`
    enforces that it is declared before a paper run.

    Attributes:
        base: The canonical :class:`~ube.core.config.BacktestConfig` (instrument,
            risk/exits, cost model, engine overrides, and the ``signal`` policy). Engine is
            ``"nautilus"`` for the nautilus backend, implied here and not re-stored.
        state_path: sqlite path for :class:`PaperState` persistence (§9.5). ``None`` means
            the run is ephemeral (no resume).
        starting_balance: Convenience override of the venue starting balance; folded into
            ``base.engine_overrides["starting_balance"]`` so the nautilus sandbox seeds the
            account (default ``None`` → the adapter's ``DEFAULT_STARTING_BALANCE``).
        engine: The registered paper engine name (``"nautilus"`` by default). The
            registry is mirrored from :mod:`ube.adapters.base`.
        calendar_validate: Whether to check incoming bars against the instrument's
            declared trading calendar (§4.4). Default ``True`` — paper never silently
            trades bars whose timestamps the declared calendar calls closed.
        calendar_strict: When ``calendar_validate`` finds an out-of-session bar, raise
            :class:`~ube.core.errors.CalendarMismatchError` (``True``, backtest parity)
            instead of skipping the bar and warning (``False``, the paper default so one
            bad bar never kills a long-running session).
    """

    base: BacktestConfig
    state_path: str | None = None
    starting_balance: float | None = None
    engine: str = "nautilus"
    # Calendar gate for paper trading (§4.4): unlike backtests (which reject out-of-session
    # bars hard via run.py), a paper session must not die on a single bad timestamp — by
    # default off-session bars are skipped (and warned), with ``calendar_strict=True``
    # opting into hard CalendarMismatchError parity with the backtest path. "24/7" labels
    # are a no-op either way.
    calendar_validate: bool = True
    calendar_strict: bool = False

    def __post_init__(self) -> None:
        if self.starting_balance is not None and self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if not isinstance(self.calendar_validate, bool):
            raise ValueError("calendar_validate must be a bool")
        if not isinstance(self.calendar_strict, bool):
            raise ValueError("calendar_strict must be a bool")
        # Fold the convenience balance into the engine overrides so the backend sees a
        # single canonical config (no second source of truth for the balance).
        if self.starting_balance is not None:
            overrides: dict[str, Any] = (
                dict(self.base.engine_overrides) if self.base.engine_overrides else {}
            )
            overrides["starting_balance"] = self.starting_balance
            import dataclasses

            new_base = dataclasses.replace(self.base, engine_overrides=overrides)
            object.__setattr__(self, "base", new_base)

    @property
    def overrides(self) -> dict[str, Any]:
        """The engine overrides, as a plain dict (never ``None``)."""
        return dict(self.base.engine_overrides) if self.base.engine_overrides else {}

    @property
    def instrument_calendar(self) -> TradingCalendar:
        """The resolved :class:`TradingCalendar` of ``base.instrument`` (§4.4).

        ``None`` (no declared calendar) and ``"24/7"`` resolve to the always-open
        calendar, so a paper session makes no calendar-derived checks for crypto (§4.5).
        """
        return resolve_calendar(self.base.instrument)

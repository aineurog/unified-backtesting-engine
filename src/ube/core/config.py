"""BacktestConfig and its sub-configs — the canonical input contract (§7, §4.7).

:class:`BacktestConfig` is the single object that fully describes one backtest run. It
is *composed* (§7.2) from the independently-defaultable sub-configs built earlier in
Phase 1 — :class:`~ube.core.instrument.Instrument`,
:class:`~ube.core.cost.CostModel`, :class:`~ube.core.risk.RiskConfig`,
:class:`~ube.core.benchmark.BenchmarkConfig` — plus a few scalar fields. It is frozen
(immutable after construction, §3 principle 5) and self-validating.

:class:`SignalConfig` is new here: it carries the paper-trading position-change policy
(``on_opposite_signal``, §9.3). It is *explicit-over-default* (§4.7): the field defaults
to ``None`` ("undeclared") and :meth:`BacktestConfig.validate(paper_trading=True)` raises
:class:`~ube.core.errors.UndeclaredConfigError` if it is still unset — a paper-trading
run has no safe default position-change policy.

Explicit-over-default fields (§4.7) are *not* silently defaulted at construction:
``base_currency`` stays ``None`` unless declared, and ``signal.on_opposite_signal`` stays
``None`` unless declared. :meth:`BacktestConfig.validate` is what enforces them for the
run modes that need them (portfolio / paper trading), because the run orchestrator (which
knows instrument count and mode) is not built in Phase 1.

**Deferred (not Phase 1, not in the plan acceptance criteria):**

* YAML round-trip (``.from_yaml()`` / ``.to_yaml()``, §7.2) lands with the CLI/cron work.
* Per-adapter ``engine_overrides`` TypedDict key/type validation (§7.2) — no adapters
  exist in Phase 1, so the field is accepted as any ``Mapping`` and validated only for
  being a mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ube.core.benchmark import BenchmarkConfig
from ube.core.cost import CostModel
from ube.core.errors import ConfigError, UndeclaredConfigError
from ube.core.instrument import Instrument
from ube.core.risk import RiskConfig

__all__ = [
    "OppositeSignalPolicy",
    "OPPOSITE_SIGNAL_POLICIES",
    "SignalConfig",
    "BacktestConfig",
]

#: The paper-trading position-change policies (§9.3).
OppositeSignalPolicy = Literal["reverse", "exit_only", "ignore"]

#: The valid ``on_opposite_signal`` values (§9.3).
OPPOSITE_SIGNAL_POLICIES: tuple[str, ...] = ("reverse", "exit_only", "ignore")


@dataclass(frozen=True)
class SignalConfig:
    """Signal-related configuration, chiefly the position-change policy (§9.3, §4.7).

    Attributes:
        on_opposite_signal: What to do when a signal arrives opposite to the current
            position during paper trading: ``"reverse"`` (close and flip),
            ``"exit_only"`` (close and go flat), or ``"ignore"`` (hold). ``None`` means
            "undeclared" (§4.7) — a paper-trading run must declare it, enforced by
            :meth:`BacktestConfig.validate(paper_trading=True)`.
    """

    on_opposite_signal: OppositeSignalPolicy | None = None

    def __post_init__(self) -> None:
        if (
            self.on_opposite_signal is not None
            and self.on_opposite_signal not in OPPOSITE_SIGNAL_POLICIES
        ):
            raise ConfigError(
                f"on_opposite_signal={self.on_opposite_signal!r} is not one of "
                f"{OPPOSITE_SIGNAL_POLICIES} (or None)"
            )


@dataclass(frozen=True)
class BacktestConfig:
    """The canonical input contract for one backtest run (§7.2).

    Composed from the Phase-1 sub-configs; every field is either required, or has a
    sensible asset-class-aware default (§7.1). Explicit-over-default fields (§4.7) —
    ``base_currency`` and ``signal.on_opposite_signal`` — are *not* silently defaulted:
    they stay ``None`` until declared, and :meth:`validate` enforces them for the run
    modes that need them.

    Attributes:
        instrument: The traded instrument (asset-class metadata, §4.5). Required.
        cost_model: Optional explicit cost model; ``None`` means "resolve the asset-class
            default" (via :func:`~ube.core.cost.resolve_cost_model`), §7.1/§7.2.
        risk: Sizing + exits (default :class:`~ube.core.risk.RiskConfig` — ``all_in``
            sizing, no exits, §7.1).
        signal: Signal-related config (default :class:`SignalConfig`).
        benchmark: Benchmark config (default
            :class:`~ube.core.benchmark.BenchmarkConfig`, ``buy_and_hold``, §7.1).
        engine: Engine adapter to use (default ``"auto"`` — first installed). Must be a
            non-empty string.
        engine_overrides: Optional engine-specific overrides (§4.2). Accepted as any
            ``Mapping``; per-adapter TypedDict key/type validation is deferred (no
            adapters exist in Phase 1). Stored as a copy (immutability rule).
        date_range: Optional ``(start, end)`` bound; stored as-is (a 2-tuple). ``start``
            and ``end`` must be comparable (``start <= end``) when both are present.
        base_currency: The explicit portfolio base currency (§4.6, §4.7). ``None`` =
            "undeclared"; required for portfolio runs, optional for single-instrument
            runs (§7.1). Never silently defaulted.
        warmup_bars: Number of leading bars to exclude from the derived views and
            metrics (indicator lookback, §7.2). Non-negative integer; default ``0``.
    """

    instrument: Instrument
    cost_model: CostModel | None = None
    risk: RiskConfig = RiskConfig()
    signal: SignalConfig = SignalConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    engine: str = "auto"
    engine_overrides: Mapping[str, Any] | None = None
    date_range: tuple[Any, Any] | None = None
    base_currency: str | None = None
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, Instrument):
            raise ConfigError(
                "BacktestConfig.instrument must be an Instrument; "
                f"got {type(self.instrument).__name__}"
            )
        if self.cost_model is not None and not isinstance(self.cost_model, CostModel):
            raise ConfigError(
                "BacktestConfig.cost_model must be a CostModel or None; "
                f"got {type(self.cost_model).__name__}"
            )
        if not isinstance(self.risk, RiskConfig):
            raise ConfigError(
                "BacktestConfig.risk must be a RiskConfig; "
                f"got {type(self.risk).__name__}"
            )
        if not isinstance(self.signal, SignalConfig):
            raise ConfigError(
                "BacktestConfig.signal must be a SignalConfig; "
                f"got {type(self.signal).__name__}"
            )
        if not isinstance(self.benchmark, BenchmarkConfig):
            raise ConfigError(
                "BacktestConfig.benchmark must be a BenchmarkConfig; "
                f"got {type(self.benchmark).__name__}"
            )
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ConfigError("engine must be a non-empty string")
        if self.engine_overrides is not None:
            if not isinstance(self.engine_overrides, Mapping):
                raise ConfigError(
                    "engine_overrides must be a Mapping or None; "
                    f"got {type(self.engine_overrides).__name__}"
                )
            object.__setattr__(self, "engine_overrides", dict(self.engine_overrides))
        if self.date_range is not None:
            if not isinstance(self.date_range, (tuple, list)) or len(self.date_range) != 2:
                raise ConfigError("date_range must be a (start, end) pair or None")
            start, end = self.date_range
            object.__setattr__(self, "date_range", (start, end))
            if start is not None and end is not None:
                try:
                    if start > end:
                        raise ConfigError(
                            f"date_range start {start!r} is after end {end!r}"
                        )
                except TypeError:
                    # Not mutually comparable — store as-is (§7.2: do not over-engineer).
                    pass
        if self.base_currency is not None and (
            not isinstance(self.base_currency, str) or not self.base_currency.strip()
        ):
            raise ConfigError("base_currency must be a non-empty string when declared")
        if (
            isinstance(self.warmup_bars, bool)
            or not isinstance(self.warmup_bars, int)
            or self.warmup_bars < 0
        ):
            raise ConfigError(
                f"warmup_bars must be a non-negative integer; got {self.warmup_bars!r}"
            )

    def validate(self, *, portfolio: bool = False, paper_trading: bool = False) -> None:
        """Validate cross-field constraints for a run mode (§4.7).

        Field-level type/value errors are already raised at construction (raising
        :class:`~ube.core.errors.ConfigError`). This method checks the two
        explicit-over-default constraints that depend on the *run mode* — information
        the orchestrator (not yet built in Phase 1) knows, so it is passed in as
        explicit flags:

        * ``portfolio=True`` with ``base_currency`` unset raises
          :class:`~ube.core.errors.UndeclaredConfigError` (§4.7/§7.1 — a portfolio run
          needs an explicit base currency to normalize multi-currency PnL).
        * ``paper_trading=True`` with ``signal.on_opposite_signal`` unset raises
          :class:`~ube.core.errors.UndeclaredConfigError` (§9.3/§4.7 — the
          position-change policy has no safe default).

        Args:
            portfolio: Whether this is a multi-instrument/portfolio run.
            paper_trading: Whether this is a paper-trading run.

        Raises:
            UndeclaredConfigError: A required explicit-over-default field is unset.
        """
        if portfolio and self.base_currency is None:
            raise UndeclaredConfigError(
                "base_currency is required for a portfolio run but was not declared "
                "(§4.7 — never silently assume a base currency)"
            )
        if paper_trading and self.signal.on_opposite_signal is None:
            raise UndeclaredConfigError(
                "signal.on_opposite_signal is required for paper trading but was not "
                "declared (§9.3/§4.7 — the position-change policy has no safe default)"
            )

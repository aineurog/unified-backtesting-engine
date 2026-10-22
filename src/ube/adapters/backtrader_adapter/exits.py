"""Exit translation for the backtrader engine (§5.2, §4.2).

backtrader is an event-driven engine, so unlike vectorbt it does not exit through vectorized
stop primitives: the strategy checks its position bar-by-bar. This module therefore translates
the canonical exit configs into per-entry *per-bar trigger arrays* (via the core
``exit_triggered`` semantics of §4.7/§8), plus the ATR/volatility aux_data resolution shared
with the other adapters (§5.2, §6.3). The strategy recomputes the plan once at entry fill time
(anchored to the actual fill price and bar) and then checks the precomputed arrays on every
subsequent bar — the levels are causal (running peaks, EWM ATR), so there is no look-ahead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.risk.aux import (
    atr_from_aux,
    atr_series_for_exit,
    resolve_vol_for_sizing,
    validate_aux,
)
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    exit_level,
    exit_triggered,
    scale_out_fraction,
)

__all__ = [
    "validate_aux",
    "atr_from_aux",
    "atr_series_for_exit",
    "resolve_vol_for_sizing",
    "resolve_atr_series_map",
    "BtExitLine",
    "build_exit_plan",
    "exit_reason_label",
]


def resolve_atr_series_map(
    exits: tuple[Any, ...],
    aux_data: Mapping[str, Any] | None,
    data: MarketData,
) -> dict[int, np.ndarray]:
    """The main-grid ATR series per exit index, for the ATR/Chandelier exits that need one."""
    out: dict[int, np.ndarray] = {}
    for idx, exit in enumerate(exits):
        if isinstance(exit, (ATRStop, ChandelierExit)):
            out[idx] = atr_series_for_exit(exit, aux_data, data)
    return out


@dataclass(frozen=True)
class BtExitLine:
    """One exit's per-bar trigger plan, anchored to a real entry (§8).

    Attributes:
        index: The exit's position in the configured ``RiskConfig.exit`` tuple.
        fraction: The scale-out fraction this exit exits when it fires (§6.4).
        reason: The canonical reason string stamped on the closing fill.
        triggered: Causal per-bar bool array — ``True`` where this exit fires.
        level: Causal per-bar price level, or ``None`` for a level-less exit
            (:class:`TimeExit` and ``trigger="close"`` exits, which fill at the bar close).
            Carried so the strategy can order same-bar exits by which the bar reached first
            (§4.7/§8) rather than by configured order.
    """

    index: int
    fraction: float
    reason: str
    triggered: np.ndarray
    level: np.ndarray | None = None


def exit_reason_label(cfg: Exit) -> str:
    """The canonical §4.6 reason label for an exit config."""
    if isinstance(cfg, ATRStop):
        return "atr_stop"
    if isinstance(cfg, ChandelierExit):
        return "chandelier"
    if isinstance(cfg, TrailingStop):
        return "trailing_stop"
    if isinstance(cfg, TakeProfit):
        return "take_profit"
    if isinstance(cfg, StopLoss):
        return "stop_loss"
    if isinstance(cfg, TimeExit):
        return "time_exit"
    raise ConfigError(f"unknown exit type {type(cfg).__name__}")


def build_exit_plan(
    exits: tuple[Any, ...],
    data: MarketData,
    *,
    side: int,
    entry_price: float,
    entry_bar: int,
    atr_map: Mapping[int, np.ndarray] | None = None,
) -> tuple[BtExitLine, ...]:
    """Build the per-bar trigger plan for every configured exit, anchored to one entry (§8).

    Pure and causal: each line is the core ``exit_triggered`` result for the exit with this
    trade's side/entry reference, plus that exit's per-bar level so the strategy can pick
    the exit the bar reached *first* when several fire at once (§4.7/§8). Each bar, the
    first-reached unspent exit exits its ``fraction``.
    """
    atr_map = atr_map if atr_map is not None else {}
    out: list[BtExitLine] = []
    for idx, cfg in enumerate(exits):
        series = atr_map.get(idx)
        triggered = exit_triggered(
            cfg,
            market_data=data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=series,
        )
        if isinstance(cfg, TimeExit) or getattr(cfg, "trigger", None) == "close":
            level = None
        else:
            try:
                level = np.asarray(
                    exit_level(
                        cfg,
                        market_data=data,
                        side=side,
                        entry_price=entry_price,
                        entry_bar=entry_bar,
                        atr_series=series,
                    ),
                    dtype=np.float64,
                )
            except ConfigError:
                level = None
        out.append(
            BtExitLine(
                index=idx,
                fraction=float(scale_out_fraction(cfg)),
                reason=exit_reason_label(cfg),
                triggered=np.asarray(triggered, dtype=np.bool_),
                level=level,
            )
        )
    return tuple(out)
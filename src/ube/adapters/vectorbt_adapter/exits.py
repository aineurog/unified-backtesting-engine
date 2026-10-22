"""Exit translation for the vectorbt engine (§5.2, §4.2).

vectorbt exits on its own stop primitives (``sl_stop`` / ``tp_stop`` / ``sl_trail``) plus
the signal exit masks, so this module translates the canonical exit configs into per-bar
stop *fractions*, folds a holding-period ``TimeExit`` into the vectorbt exit masks (vectorbt
has no native time exit), and labels each closed trade with the core ``exit_triggered``
semantics so a trade is stamped identically to the Nautilus engine (requirements §4.6, §8).
The ATR-based exits resolve their named ``aux_data`` series here, including the
no-look-ahead forward-fill described in §5.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ube.adapters.vectorbt_adapter.adapt_data import VbtSignalInputs
from ube.core.data import MarketData
from ube.core.errors import IncompatibleConfigError
from ube.core.risk.aux import (
    atr_from_aux,
    atr_series_for_exit,
    resolve_vol_for_sizing,
    validate_aux,
)
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    exit_level,
    exit_triggered,
    first_reached_exit,
    time_exit_mask_for_entries,
)
from ube.core.signals import Signals

__all__ = [
    "validate_aux",
    "atr_from_aux",
    "atr_series_for_exit",
    "resolve_vol_for_sizing",
    "apply_time_exits",
    "exit_stop_params",
    "classify_exit_reason",
    "core_stop_override",
    "core_stop_exit_window",
]


def apply_time_exits(
    inputs: VbtSignalInputs, exits: tuple[Any, ...], data: MarketData
) -> VbtSignalInputs:
    """Fold ``TimeExit`` into the vectorbt exit masks so vectorbt actually exits (§6.4).

    vectorbt has no native holding-period primitive, so a ``TimeExit(bars)`` is translated
    into an exit-signal mask with the core :func:`time_exit_mask` semantics: a position
    opened at bar ``e`` exits at the first bar at or after ``e + bars``. The mask is applied
    to both the long and the short exit columns (a time exit closes whichever side is open);
    a signal exit or stop that already closed the position makes a later mask bar idle, and
    a new entry re-arms the countdown. Returns a (possibly mutated) copy of ``inputs``.
    """
    time_bars = [e.bars for e in exits if isinstance(e, TimeExit)]
    if not time_bars:
        return inputs
    n = data.n_bars
    long_mask = np.zeros(n, dtype=np.bool_)
    short_mask = np.zeros(n, dtype=np.bool_)
    for bars in time_bars:
        long_mask |= time_exit_mask_for_entries(
            inputs.entries.to_numpy(dtype=np.bool_), bars
        )
        short_mask |= time_exit_mask_for_entries(
            inputs.short_entries.to_numpy(dtype=np.bool_), bars
        )
    idx = inputs.close.index
    inputs.long_exits = inputs.long_exits | pd.Series(long_mask, index=idx, dtype=bool)
    inputs.short_exits = inputs.short_exits | pd.Series(short_mask, index=idx, dtype=bool)
    return inputs


def exit_stop_params(
    exits: tuple[Any, ...], data: MarketData, aux_data: Mapping[str, Any] | None
) -> tuple[Any, Any, Any]:
    """Translate the exit configs into vectorbt ``sl_stop`` / ``tp_stop`` / ``sl_trail``.

    ``sl_stop`` is the tightest per-bar stop fraction over every **non-trailing** stop-type
    exit (fixed ``StopLoss`` and non-trailing ``ATRStop`` fractions — merged via per-bar
    minimum); ``tp_stop`` is the tightest ``TakeProfit`` fraction; ``sl_trail`` is always
    ``None``.

    Take-profit scale-outs (``scale_out < 1.0``, §6.4) cannot be represented: vectorbt's
    single ``tp_stop`` closes the whole position. Rather than silently exiting 100% on the
    first target, this raises :class:`IncompatibleConfigError` — the caller must use an
    engine that simulates partial fills (backtrader or nautilus).

    Trailing stops (``TrailingStop`` / ``ChandelierExit`` / trailing ``ATRStop``) are
    deliberately kept out of vectorbt's native stop. Vectorbt exposes a single
    ``sl_stop`` distance plus one ``sl_trail`` boolean, so the only way to pass a trailing
    stop alongside a non-trailing one is to merge them and trail the merged distance. When
    the non-trailing level is tighter (e.g. an entry-anchored ATR fraction of ~0.11%
    beside a 1% trailing stop) vectorbt then ratchets that tighter, non-trailing level —
    firing intrabar on bars where the core rule, which trails only its own level off the
    running peak, holds (a wrong exit price and a dropped same-bar reversal). Trailing
    exits are instead resolved exactly by the fold's :func:`core_stop_override`, which
    re-times every close to the core first-trigger bar/level, so no trailing distance needs
    to reach vectorbt. ATR fractions are ``mult * atr / close`` — an approximation of the
    absolute core level, accepted for the vectorized engine (§16 parity tolerance).
    """
    # The tightest take-profit owns vectorbt's single ``tp_stop``.
    tp_percents: list[float] = []
    # Non-trailing stop distances merged into one per-bar ``sl_stop``.
    all_fractions: list[np.ndarray] = []
    # Scalar StopLoss broadcast to a per-bar array.
    sl_scalar: float | None = None

    for exit in exits:
        if isinstance(exit, TakeProfit):
            if exit.scale_out < 1.0:
                raise IncompatibleConfigError(
                    "vectorbt cannot simulate take-profit scale-outs (§6.4): "
                    f"TakeProfit(percent={exit.percent}, scale_out={exit.scale_out}) "
                    "closes the whole position on one vectorbt exit; use backtrader or "
                    "nautilus for partial exits"
                )
            tp_percents.append(float(exit.percent))
        elif isinstance(exit, StopLoss):
            sl_scalar = exit.percent if sl_scalar is None else sl_scalar
        elif isinstance(exit, ATRStop):
            if exit.trailing:
                # Trailing ATR — resolved by the core re-timing fold, not by vectorbt.
                continue
            series = atr_series_for_exit(exit, aux_data, data)
            fraction = exit.mult * series / np.where(data.close > 0, data.close, np.nan)
            all_fractions.append(fraction)
        elif isinstance(exit, (TrailingStop, ChandelierExit)):
            # Always trailing — resolved by the core re-timing fold, not by vectorbt.
            continue

    # Merge the scalar StopLoss as a per-bar constant array.
    n = data.n_bars
    if sl_scalar is not None:
        all_fractions.append(np.full(n, sl_scalar))

    # ``sl_stop``: per-bar minimum of every non-trailing stop fraction.
    sl_stop: Any = None
    if all_fractions:
        combined = all_fractions[0]
        for frac in all_fractions[1:]:
            combined = np.minimum(combined, frac)
        sl_stop = np.nan_to_num(combined, nan=0.0)

    tp_stop: float | None = min(tp_percents) if tp_percents else None

    # Never trail natively: vectorbt's single trailing primitive cannot represent a
    # heterogeneous stop set (see the docstring). ``None`` disables the trailing mechanism.
    return sl_stop, tp_stop, None


def _exit_reason_name(exit: Any) -> str | None:
    """The canonical §4.6 reason string for an exit config (None for unknown types)."""
    if isinstance(exit, TakeProfit):
        return "take_profit"
    if isinstance(exit, ATRStop):
        return "atr_stop"
    if isinstance(exit, ChandelierExit):
        return "chandelier"
    if isinstance(exit, TrailingStop):
        return "trailing_stop"
    if isinstance(exit, StopLoss):
        return "stop_loss"
    if isinstance(exit, TimeExit):
        return "time_exit"
    return None


def _exits_fired_at(
    exits: tuple[Any, ...],
    data: MarketData,
    side: int,
    entry_price: float,
    entry_bar: int,
    bar: int,
    aux_data: Mapping[str, Any] | None,
) -> list[tuple[str, float]]:
    """The exits ``(reason, level-or-close)`` that fire on ``bar`` under the core rule (§4.7/§8).

    Shared by :func:`classify_exit_reason` (single-bar label) and
    :func:`core_stop_override` (first-bar re-location) so the winner selection is the same
    everywhere. :class:`TimeExit` has no price level, and a ``"close"``-triggered exit is
    evaluated against the close — both fill at the bar close, so they rank on the close
    price (mirroring the nautilus actor §9.4).
    """
    fired: list[tuple[str, float]] = []
    for exit in exits:
        series = (
            atr_series_for_exit(exit, aux_data, data)
            if isinstance(exit, (ATRStop, ChandelierExit))
            else None
        )
        if not bool(
            exit_triggered(
                exit,
                market_data=data,
                side=side,
                entry_price=entry_price,
                entry_bar=entry_bar,
                atr_series=series,
            )[bar]
        ):
            continue
        reason = _exit_reason_name(exit)
        if reason is None:
            continue
        if isinstance(exit, TimeExit) or getattr(exit, "trigger", None) == "close":
            level = float(data.close[bar])
        else:
            level = float(
                exit_level(
                    exit,
                    market_data=data,
                    side=side,
                    entry_price=entry_price,
                    entry_bar=entry_bar,
                    atr_series=series,
                )[bar]
            )
        fired.append((reason, level))
    return fired


def classify_exit_reason(
    exits: tuple[Any, ...],
    data: MarketData,
    side: int,
    entry_price: float,
    entry_bar: int,
    exit_bar: int,
    aux_data: Mapping[str, Any] | None,
    signals: Signals,
) -> str:
    """Label a closing fill using the core ``exit_triggered`` semantics (§4.6/§8).

    Risk exits are resolved **before** the signal exit, matching the reference (nautilus
    returns from its risk block before it ever evaluates signals) and the backtrader
    strategy (which only falls back to a signal close ``if fired is None``). When a stop or
    target triggers on the same bar as a signal, the stop is what closed the trade and the
    fill price proves it — the level, not the bar close. Checking the signal first
    mislabeled every such trade as ``signal``, e.g. a live crypto-perp exit filled at the
    exact stop level 83003.330925 stamped ``signal`` while nautilus/backtrader stamped
    ``stop_loss``.
    """
    # Among the exits that trigger on this bar, the one the bar *reached first* owns the
    # label (§4.7/§8) — nearest the bar's open, so a bar that ran through the target
    # before the stop is a take_profit, not a stop_loss.
    fired = _exits_fired_at(exits, data, side, entry_price, entry_bar, exit_bar, aux_data)

    if fired:
        pick = first_reached_exit([lvl for _, lvl in fired], open_price=float(data.open[exit_bar]))
        return fired[0 if pick is None else pick][0]

    # No risk exit fired on this bar, so the signal/flip close owns the label. A flip
    # (opposite-side entry) also closes the position: a long exited on the bar a short
    # opens, or a short exited where a long opens. Nautilus journals these as "signal";
    # without this branch they would be mislabeled "end_of_run".
    if _signal_exit_fires_at(signals, side, exit_bar):
        return "signal"

    return "end_of_run"


def _signal_exit_fires_at(signals: Signals, side: int, bar: int) -> bool:
    """Whether the signal columns close a ``side`` position on ``bar`` (§4.6).

    The same acceptance :func:`classify_exit_reason` applies at a native exit bar: the
    side's own exit column, or the flip encoding's opposite-side entry (a long is closed
    by ``long_exit`` or by the ``short_entry`` that opens the other leg, and vice versa).
    """
    if side == 1:
        return bool(signals.long_exit[bar]) or bool(signals.short_entry[bar])
    return bool(signals.short_exit[bar]) or bool(signals.long_entry[bar])


def core_stop_override(
    exits: tuple[Any, ...],
    data: MarketData,
    *,
    side: int,
    entry_price: float,
    entry_bar: int,
    exit_bar: int,
    aux_data: Mapping[str, Any] | None,
) -> tuple[int, float, str] | None:
    """Re-locate a vectorbt stop fill to the core first-trigger ``(bar, level, reason)``.

    vectorbt's native stop primitives are **entry-anchored** (the fraction is taken at the
    entry bar) and **close-triggered** — neither matches the core ``touched`` + moving
    per-bar level rule that backtrader and nautilus run bar-by-bar (e.g. a non-trailing
    ATR stop at ``entry - mult * atr[i]``). The two engines therefore can stop on different
    bars, and at different prices, for the same bars/signals/config — the live divergence
    where vectorbt stayed in a trade backtrader/nautilus had stopped on the same bar.

    This scans ``entry_bar+1..exit_bar`` for the first bar where *any* exit fires under the
    core rule and returns that bar, the winning exit's fill price there (its own level for
    ``trigger="touched"``, the bar close for ``"close"``/``TimeExit`` — §9.4), and its
    reason. Bars strictly after ``exit_bar`` are ignored so a legitimate earlier
    signal/time exit is never re-timed backwards. The scan starts **after** the entry bar
    because the reference engines book a same-bar entry at the bar close and cannot exit
    before the next bar — re-locating a stop onto the entry bar would manufacture a
    same-bar close+open that backtrader/nautilus never produce (and corrupt the sized
    pass). Returns ``None`` when no exit fires on or before ``exit_bar`` — vectorbt's own
    bar/price stands.
    """
    n = data.n_bars
    if not (0 <= entry_bar < n) or not (0 <= exit_bar < n) or entry_bar > exit_bar:
        return None
    for bar in range(entry_bar + 1, exit_bar + 1):
        fired = _exits_fired_at(exits, data, side, entry_price, entry_bar, bar, aux_data)
        if not fired:
            continue
        # §4.7: a bar that sweeps several levels is ambiguous; the one nearest the open was
        # reached first (same rule as the nautilus actor and classify_exit_reason).
        pick = first_reached_exit([lvl for _, lvl in fired], open_price=float(data.open[bar]))
        reason, level = fired[0 if pick is None else pick]
        return bar, float(level), reason
    return None


def core_stop_exit_window(
    exits: tuple[Any, ...],
    data: MarketData,
    *,
    side: int,
    entry_price: float,
    entry_bar: int,
    exit_bar: int,
    aux_data: Mapping[str, Any] | None,
    signals: Signals | None = None,
) -> tuple[int, float, str] | None:
    """Re-time a vectorbt native-stop close that fired **before** the core rule would.

    The mirror image of :func:`core_stop_override`. vectorbt's ``sl_stop``/``tp_stop`` are
    **entry-anchored fractions** — ``mult * atr[entry_bar] / close[entry_bar]`` applied to
    the entry price — so the level is frozen at the entry bar, whereas the core
    :func:`~ube.core.risk.exits.atr_stop_level` recomputes ``entry -/+ mult * atr[i]`` on
    every bar. When ATR *widens* after the entry, the core level moves away from entry while
    the frozen fraction does not, so vectorbt closes the trade on a bar where the core rule
    has not fired yet. :func:`core_stop_override` cannot repair that by construction: it only
    scans ``entry_bar+1..exit_bar`` and the core's real trigger lies one or more bars later,
    past the bound. The fold then labelled the close with the ``end_of_run`` fallback from
    :func:`classify_exit_reason` — a stop exit reported as a run-end close, which corrupted
    the ledger reason and left the run's balance diverging from backtrader/nautilus (which
    held the position and took it out at the core stop one bar later).

    This scans ``exit_bar+1..n-1`` for the first bar the core rule fires and returns that
    ``(bar, level, reason)``, so the fold can book the fill backtrader/nautilus book. The
    core rule includes the **signal exits** when ``signals`` is supplied: vectorbt also
    exits natively on the encoded ``long_exit``/``short_exit`` columns, but a native stop
    that closed the trade *before* the flip bar leaves a carried record open (restored by
    the fold) while the incoming opposite-side record enters on the flip bar — netting the
    ledger to zero and killing the new leg on every cursor replay. Re-timing that close to
    the flip bar with reason ``"signal"`` is what the reference engines book: the close and
    the new entry then both stand. Risk exits win a coincident bar (§4.7/§8, the same
    priority :func:`classify_exit_reason` applies), and the earliest bar wins overall.

    Returns ``None`` when the core rule does not fire later in the window either — the
    caller keeps vectorbt's own close (the legitimate early-exit case), which is only
    correct when something else really did close the trade.
    """
    n = data.n_bars
    if not (0 <= entry_bar < n) or not (0 <= exit_bar < n) or entry_bar > exit_bar:
        return None
    for bar in range(exit_bar + 1, n):
        fired = _exits_fired_at(exits, data, side, entry_price, entry_bar, bar, aux_data)
        if fired:
            pick = first_reached_exit([lvl for _, lvl in fired], open_price=float(data.open[bar]))
            reason, level = fired[0 if pick is None else pick]
            return bar, float(level), reason
        if signals is not None and _signal_exit_fires_at(signals, side, bar):
            return bar, float(data.close[bar]), "signal"
    return None

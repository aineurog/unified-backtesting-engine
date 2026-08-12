"""Engine-agnostic exit subsystem: exit configs, level computation, and scale-out
(§6.4, §8).

Exits are *path-dependent on bar history, not on order-fill state* (§8): each exit is a
pure function that, given the bars (OHLCV), a side, and an entry reference, returns the
per-bar stop/target **level array** (or, for a time exit, a trigger mask). The levels are
fully precomputed so an adapter can inject them into the engine's native stop/target
mechanism (``sl_stop``/``tp_stop`` for vectorbt, a Sizer/order wrapper for backtrader, an
Actor for NautilusTrader). The *fill simulation* — did the bar actually fill at the
level, at what price, with what slippage — is the engine's job (§8); this module only
produces deterministic levels and the deterministic *trigger rule* (§4.8/§9) that must
be identical across backtest and paper trading.

The starting set (§8): ATR stop (fixed multiple + trailing), percentage trailing stop,
time exit, and chandelier exit. Derived series (ATR, running high/low) are computed from
the bars internally (vectorized) when not supplied as ``atr_series`` (§5.2).

Multi-level exits with scale-out (§6.4): :class:`RiskConfig.exit` is an ordered tuple of
exits; ``TakeProfit`` carries a ``scale_out`` fraction (default ``1.0``) while stops
always exit 100% of the remaining position. :func:`scale_out_plan` yields the ordered
sequence of scale-out fractions per bar as a pure, vectorized computation (no fill
simulation).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard, cast

import numpy as np
import pandas as pd

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError
from ube.core.risk.sizing import SizeModel

__all__ = [
    "Trigger",
    "TRIGGERS",
    "TakeProfit",
    "ATRStop",
    "TrailingStop",
    "TimeExit",
    "ChandelierExit",
    "Exit",
    "ExitPlan",
    "RiskConfig",
    "atr",
    "take_profit_level",
    "atr_stop_level",
    "trailing_stop_level",
    "chandelier_level",
    "time_exit_mask",
    "exit_level",
    "is_triggered",
    "exit_triggered",
    "scale_out_fraction",
    "scale_out_plan",
]

#: The TP/SL trigger rule (§4.8/§9): ``"touched"`` = the bar's high/low reached the
#: level intra-bar; ``"close"`` = only the bar's close is compared. Must match between
#: backtest and paper trading.
Trigger = Literal["touched", "close"]

TRIGGERS: tuple[str, ...] = ("touched", "close")

#: A bar-slice direction: ``"above"`` means the level lies above the current price (a
#: long target / short stop); ``"below"`` means below (a long stop / short target).
_AboveBelow = Literal["above", "below"]


def _validate_positive(value: object, name: str) -> float:
    """Validate a finite, strictly positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number; got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        raise ConfigError(f"{name} must be finite and positive; got {value!r}")
    return f


def _validate_scale_out(value: object) -> float:
    """Validate a scale-out fraction: finite, in ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"scale_out must be a number; got {value!r}")
    f = float(value)
    if not math.isfinite(f) or not (0.0 < f <= 1.0):
        raise ConfigError(f"scale_out must be in (0, 1]; got {value!r}")
    return f


def _validate_trigger(value: object) -> str:
    if value not in TRIGGERS:
        raise ConfigError(f"trigger must be one of {TRIGGERS}; got {value!r}")
    return str(value)


def _validate_period(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer; got {value!r}")
    return value


def _validate_atr_name(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"atr must be a non-empty aux_data name or None; got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Exit configs (frozen, fail-fast).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeProfit:
    """Fixed take-profit target (§6.4, §8).

    Target price = ``entry * (1 + percent)`` for a long, ``entry * (1 - percent)`` for a
    short. A positive ``percent`` is a profit target; a stop-loss is a different exit
    (``ATRStop`` / ``TrailingStop``).

    Attributes:
        percent: Distance above (long) / below (short) entry, as a fraction.
        scale_out: Fraction of the position to exit when hit, in ``(0, 1]`` (§6.4).
        trigger: The §4.8 trigger rule.
    """

    percent: float
    scale_out: float = 1.0
    trigger: Trigger = "touched"

    def __post_init__(self) -> None:
        object.__setattr__(self, "percent", _validate_positive(self.percent, "TakeProfit.percent"))
        object.__setattr__(self, "scale_out", _validate_scale_out(self.scale_out))
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))


@dataclass(frozen=True)
class ATRStop:
    """ATR-based stop: fixed multiple off entry, with an optional trailing variant (§8).

    Fixed (``trailing=False``): ``stop = entry - mult * atr`` (long) / ``+ mult * atr``
    (short) — the level moves only because ATR changes. Trailing (``trailing=True``):
    the stop ratchets in the favourable direction, tracking ``high - mult * atr`` (long)
    while never loosening below the entry-anchored stop.

    Attributes:
        mult: ATR multiple.
        trigger: The §4.8 trigger rule.
        trailing: Whether to ratchet the stop in the favourable direction.
        period: ATR lookback, used when ``atr`` (the aux_data reference) is ``None``.
        atr: Optional ``aux_data`` name (§5.2); resolved by the caller into an
            ``atr_series`` passed to :func:`atr_stop_level`.
    """

    mult: float
    trigger: Trigger = "touched"
    trailing: bool = False
    period: int = 14
    atr: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mult", _validate_positive(self.mult, "ATRStop.mult"))
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))
        if not isinstance(self.trailing, bool):
            raise ConfigError(f"ATRStop.trailing must be a bool; got {self.trailing!r}")
        object.__setattr__(self, "period", _validate_period(self.period, "ATRStop.period"))
        object.__setattr__(self, "atr", _validate_atr_name(self.atr))


@dataclass(frozen=True)
class TrailingStop:
    """Trailing stop as a percentage off the running peak/trough (§8).

    Long: ``stop = running_high_since_entry * (1 - percent)``. Short:
    ``stop = running_low_since_entry * (1 + percent)``.

    Attributes:
        percent: Trailing distance off the peak/trough, as a fraction.
        trigger: The §4.8 trigger rule.
    """

    percent: float
    trigger: Trigger = "touched"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "percent", _validate_positive(self.percent, "TrailingStop.percent")
        )
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))


@dataclass(frozen=True)
class TimeExit:
    """Time-based exit: exit ``bars`` bars after entry (§8).

    Chosen signature: ``TimeExit(bars)`` — a holding-period exit that fires once the
    position has been held for ``bars`` bars (i.e. at bar ``entry_bar + bars``). The
    "exit at a bar timestamp" form is not implemented separately (it would need calendar
    machinery and is out of scope for item 06). A time exit has no price level and no
    ``trigger`` — time is discrete, so there is no touched/close ambiguity.

    Attributes:
        bars: Number of bars to hold before exiting (a positive integer).
    """

    bars: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bars", _validate_period(self.bars, "TimeExit.bars"))


@dataclass(frozen=True)
class ChandelierExit:
    """Chandelier exit: running highest-high / lowest-low offset by ``mult * ATR`` (§8).

    Long: ``level = running_high_since_entry - mult * atr``. Short:
    ``level = running_low_since_entry + mult * atr``.

    Attributes:
        mult: ATR multiple.
        trigger: The §4.8 trigger rule.
        period: ATR lookback, used when ``atr`` is ``None``.
        atr: Optional ``aux_data`` name (§5.2); resolved by the caller.
    """

    mult: float
    trigger: Trigger = "touched"
    period: int = 14
    atr: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mult", _validate_positive(self.mult, "ChandelierExit.mult"))
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))
        object.__setattr__(self, "period", _validate_period(self.period, "ChandelierExit.period"))
        object.__setattr__(self, "atr", _validate_atr_name(self.atr))


#: The union of exit config types (§8).
Exit = TakeProfit | ATRStop | TrailingStop | TimeExit | ChandelierExit

_EXIT_NAMES: tuple[str, ...] = (
    "TakeProfit",
    "ATRStop",
    "TrailingStop",
    "TimeExit",
    "ChandelierExit",
)


def _is_exit(obj: object) -> TypeGuard[Exit]:
    return isinstance(obj, (TakeProfit, ATRStop, TrailingStop, TimeExit, ChandelierExit))


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _require_market_data(market_data: object) -> MarketData:
    if not isinstance(market_data, MarketData):
        raise DataShapeError(f"expected a MarketData; got {type(market_data).__name__}")
    return market_data


def _validate_side(side: object) -> int:
    if side not in (1, -1):
        raise ConfigError(f"side must be +1 (long) or -1 (short); got {side!r}")
    return int(side)


def _validate_entry_price(entry_price: object) -> float:
    if isinstance(entry_price, bool) or not isinstance(entry_price, (int, float)):
        raise ConfigError(f"entry_price must be a number; got {entry_price!r}")
    f = float(entry_price)
    if not math.isfinite(f) or f <= 0:
        raise ConfigError(f"entry_price must be finite and positive; got {entry_price!r}")
    return f


def _validate_entry_bar(entry_bar: object, n_bars: int) -> int:
    if isinstance(entry_bar, bool) or not isinstance(entry_bar, int):
        raise ConfigError(f"entry_bar must be an integer; got {entry_bar!r}")
    if entry_bar < 0 or entry_bar >= n_bars:
        raise ConfigError(f"entry_bar={entry_bar} out of range for {n_bars} bars")
    return entry_bar


def _resolve_atr(
    atr_series: object | None, market_data: MarketData, period: int
) -> np.ndarray:
    """Resolve the ATR series: use ``atr_series`` if supplied, else compute (§5.2)."""
    if atr_series is None:
        return atr(market_data, period)
    arr = np.asarray(atr_series, dtype=np.float64)
    if arr.ndim != 1:
        raise DataShapeError(f"atr_series must be 1-D; got shape {arr.shape}")
    if arr.shape[0] != market_data.n_bars:
        raise DataShapeError(
            f"atr_series length {arr.shape[0]} does not match {market_data.n_bars} bars"
        )
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ConfigError("atr_series must be finite and positive")
    return arr


def _running_extreme_since(
    values: np.ndarray, entry_bar: int, *, maximum: bool
) -> np.ndarray:
    """Running max (or min) of ``values`` from ``entry_bar`` onward; NaN before."""
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if entry_bar < n:
        seg = values[entry_bar:]
        out[entry_bar:] = (
            np.maximum.accumulate(seg) if maximum else np.minimum.accumulate(seg)
        )
    return out


# ---------------------------------------------------------------------------
# ATR indicator.
# ---------------------------------------------------------------------------


def atr(market_data: MarketData, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range over the last ``period`` bars (§4.3, §4.4).

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)`` (§4.3),
    smoothed with Wilder's running average (an exponential average with
    ``alpha = 1 / period``), so a ``period``-bar ATR reflects "the last ``period`` bars
    that actually existed" (§4.4). Pure and vectorized; returns a float64 array aligned
    to the bar index.
    """
    md = _require_market_data(market_data)
    _validate_period(period, "ATR period")
    high = md.high
    low = md.low
    close = md.close
    n = high.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return pd.Series(true_range).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


# ---------------------------------------------------------------------------
# Per-exit level functions (pure).
# ---------------------------------------------------------------------------


def take_profit_level(
    cfg: TakeProfit, market_data: MarketData, *, side: object, entry_price: object
) -> np.ndarray:
    """Per-bar take-profit target: a constant ``entry * (1 ± percent)`` level (§6.4)."""
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    mult = 1.0 + cfg.percent if s > 0 else 1.0 - cfg.percent
    return np.full(md.n_bars, entry * mult, dtype=np.float64)


def atr_stop_level(
    cfg: ATRStop,
    market_data: MarketData,
    *,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Per-bar ATR stop level, fixed (entry-anchored) or trailing (§8)."""
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    atr_arr = _resolve_atr(atr_series, md, cfg.period)
    mult = cfg.mult

    if not cfg.trailing:
        return entry - mult * atr_arr if s > 0 else entry + mult * atr_arr

    # Trailing: ratchet in the favourable direction, never loosening past the
    # entry-anchored stop.
    if s > 0:
        base = md.high - mult * atr_arr
        anchor = entry - mult * atr_arr[eb]
        running = _running_extreme_since(base, eb, maximum=True)
        running[eb:] = np.maximum(running[eb:], anchor)
        return running
    base = md.low + mult * atr_arr
    anchor = entry + mult * atr_arr[eb]
    running = _running_extreme_since(base, eb, maximum=False)
    running[eb:] = np.minimum(running[eb:], anchor)
    return running


def trailing_stop_level(
    cfg: TrailingStop,
    market_data: MarketData,
    *,
    side: object,
    entry_bar: int = 0,
) -> np.ndarray:
    """Per-bar trailing stop as a percentage off the running peak/trough (§8)."""
    md = _require_market_data(market_data)
    s = _validate_side(side)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    if s > 0:
        peak = _running_extreme_since(md.high, eb, maximum=True)
        return peak * (1.0 - cfg.percent)
    trough = _running_extreme_since(md.low, eb, maximum=False)
    return trough * (1.0 + cfg.percent)


def chandelier_level(
    cfg: ChandelierExit,
    market_data: MarketData,
    *,
    side: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Per-bar chandelier level: running high/low offset by ``mult * ATR`` (§8)."""
    md = _require_market_data(market_data)
    s = _validate_side(side)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    atr_arr = _resolve_atr(atr_series, md, cfg.period)
    if s > 0:
        high = _running_extreme_since(md.high, eb, maximum=True)
        return high - cfg.mult * atr_arr
    low = _running_extreme_since(md.low, eb, maximum=False)
    return low + cfg.mult * atr_arr


def time_exit_mask(cfg: TimeExit, market_data: MarketData, *, entry_bar: int = 0) -> np.ndarray:
    """Per-bar trigger mask for a holding-period exit (§8).

    ``True`` from the bar at which the position has been held ``cfg.bars`` bars (i.e.
    ``i >= entry_bar + cfg.bars``).
    """
    md = _require_market_data(market_data)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    return np.arange(md.n_bars) >= eb + cfg.bars


# ---------------------------------------------------------------------------
# Trigger rule + dispatch.
# ---------------------------------------------------------------------------


def is_triggered(
    trigger: Trigger,
    level: object,
    *,
    high: object,
    low: object,
    close: object,
    direction: _AboveBelow,
) -> np.ndarray:
    """Apply the §4.8/§9 trigger rule to a level array (pure, vectorized).

    ``direction`` says whether the level lies above or below the price (a long target /
    short stop is "above"; a long stop / short target is "below"). For ``"touched"`` the
    bar's high (above) or low (below) is compared; for ``"close"`` the close is compared
    regardless of direction. This is the single source of truth for the trigger rule; it
    does *not* simulate fills (§8).
    """
    _validate_trigger(trigger)
    lvl = np.asarray(level, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    cl = np.asarray(close, dtype=np.float64)
    if direction not in ("above", "below"):
        raise ConfigError(f"direction must be 'above' or 'below'; got {direction!r}")
    ref = (hi if direction == "above" else lo) if trigger == "touched" else cl
    return ref >= lvl if direction == "above" else ref <= lvl


def _exit_direction(cfg: Exit, side: int) -> _AboveBelow:
    """Whether an exit's level lies above or below the price, given the side."""
    if isinstance(cfg, TakeProfit):
        return "above" if side > 0 else "below"
    return "below" if side > 0 else "above"


def exit_level(
    cfg: Exit,
    *,
    market_data: MarketData,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Dispatch to the per-exit level function for price-level exits.

    :class:`TimeExit` has no price level — use :func:`time_exit_mask` or
    :func:`exit_triggered` instead. ``entry_price`` is the entry reference; trailing /
    chandelier exits ignore it (they are anchored to the bar history since ``entry_bar``).
    """
    if isinstance(cfg, TakeProfit):
        return take_profit_level(cfg, market_data, side=side, entry_price=entry_price)
    if isinstance(cfg, ATRStop):
        return atr_stop_level(
            cfg,
            market_data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=atr_series,
        )
    if isinstance(cfg, TrailingStop):
        return trailing_stop_level(cfg, market_data, side=side, entry_bar=entry_bar)
    if isinstance(cfg, ChandelierExit):
        return chandelier_level(
            cfg, market_data, side=side, entry_bar=entry_bar, atr_series=atr_series
        )
    if isinstance(cfg, TimeExit):
        raise ConfigError("TimeExit has no price level; use time_exit_mask / exit_triggered")
    raise ConfigError(f"unknown exit type {type(cfg).__name__}")


def exit_triggered(
    cfg: Exit,
    *,
    market_data: MarketData,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Whether ``cfg`` fires at each bar (level + trigger rule), for any exit type.

    A bool array aligned to the bar index; the deterministic trigger rule of §4.8/§9 is
    applied via :func:`is_triggered` (for a time exit, the mask is used directly).
    """
    if isinstance(cfg, TimeExit):
        return time_exit_mask(cfg, market_data, entry_bar=entry_bar)
    s = _validate_side(side)
    level = exit_level(
        cfg,
        market_data=market_data,
        side=s,
        entry_price=entry_price,
        entry_bar=entry_bar,
        atr_series=atr_series,
    )
    md = _require_market_data(market_data)
    return is_triggered(
        cfg.trigger,
        level,
        high=md.high,
        low=md.low,
        close=md.close,
        direction=_exit_direction(cfg, s),
    )


# ---------------------------------------------------------------------------
# Multi-level exits with scale-out (§6.4).
# ---------------------------------------------------------------------------


def scale_out_fraction(cfg: Exit) -> float:
    """The scale-out fraction an exit exits when it fires (§6.4).

    ``TakeProfit`` exits ``cfg.scale_out`` of the position; every stop (``ATRStop``,
    ``TrailingStop``, ``ChandelierExit``, ``TimeExit``) always exits 100% of the
    remaining position.
    """
    if isinstance(cfg, TakeProfit):
        return cfg.scale_out
    return 1.0


def _coerce_exits(exits: object) -> tuple[Exit, ...]:
    # A single exit object is accepted as a one-element shorthand (§7.2 shows
    # ``RiskConfig(exit=ATRStop(...))``); a sequence is taken as-is. Strings are rejected
    # (they are not exits).
    if _is_exit(exits):
        return (exits,)
    if isinstance(exits, (str, bytes)) or not isinstance(exits, Sequence):
        raise ConfigError("exits must be an exit config or a sequence of exit configs")
    out = tuple(exits)
    for e in out:
        if not _is_exit(e):
            raise ConfigError(
                f"unknown exit type {type(e).__name__}; expected one of {_EXIT_NAMES}"
            )
    return cast(tuple[Exit, ...], out)


@dataclass(frozen=True)
class ExitPlan:
    """Ordered scale-out plan: which exits fire on which bar, and their fractions (§6.4).

    Attributes:
        fractions: The ordered scale-out fractions, one per exit in the configured order.
        triggered: Per-exit bool arrays (aligned to the bar index) — ``True`` where the
            exit's level was hit under its trigger rule. For bar ``i`` the exits ``j``
            with ``triggered[j][i]`` fire in order, each exiting ``fractions[j]`` of the
            remaining position.
    """

    fractions: tuple[float, ...]
    triggered: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        fractions = tuple(float(f) for f in self.fractions)
        triggered = tuple(_readonly_bool(a) for a in self.triggered)
        if len(fractions) != len(triggered):
            raise DataShapeError(
                f"fractions ({len(fractions)}) and triggered ({len(triggered)}) "
                "lengths differ"
            )
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(self, "triggered", triggered)

    @property
    def n_exits(self) -> int:
        """Number of exits in the plan."""
        return len(self.fractions)


def _readonly_bool(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.bool_).copy()
    out.setflags(write=False)
    return out


def scale_out_plan(
    exits: Sequence[Exit],
    *,
    market_data: MarketData,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> ExitPlan:
    """Compute the ordered scale-out plan for ``exits`` (§6.4).

    Pure and vectorized: for each exit (in order) it computes the level and applies the
    trigger rule, returning an :class:`ExitPlan` of ``fractions`` + per-exit ``triggered``
    bool arrays. Stops always carry fraction ``1.0``. No fill simulation is performed
    (§8).
    """
    ordered = _coerce_exits(exits)
    fractions = tuple(scale_out_fraction(e) for e in ordered)
    triggered = tuple(
        exit_triggered(
            e,
            market_data=market_data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=atr_series,
        )
        for e in ordered
    )
    return ExitPlan(fractions=fractions, triggered=triggered)


# ---------------------------------------------------------------------------
# RiskConfig.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """Risk configuration: sizing plus an ordered tuple of exits (§6.4, §7.1).

    Attributes:
        sizing: The position-sizing model (default :class:`SizeModel` ``all_in``, §7.1).
        exit: The ordered tuple of exits (default empty — exits come from signals only,
            §7.1). May be given as a single exit object (wrapped into a one-tuple, the
            §7.2 shorthand) or a list/tuple (converted to a tuple).
    """

    sizing: SizeModel = SizeModel()
    exit: tuple[Exit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sizing, SizeModel):
            raise ConfigError(
                f"RiskConfig.sizing must be a SizeModel; got {type(self.sizing).__name__}"
            )
        ordered = _coerce_exits(self.exit)
        total = sum(scale_out_fraction(e) for e in ordered if isinstance(e, TakeProfit))
        if total > 1.0 + 1e-12:
            raise ConfigError(
                f"take-profit scale_out fractions sum to {total}; must not exceed 1.0"
            )
        object.__setattr__(self, "exit", ordered)

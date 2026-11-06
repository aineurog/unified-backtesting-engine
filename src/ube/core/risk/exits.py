"""Engine-agnostic exit subsystem: exit configs, level computation, and scale-out
(§6.4, §8).

Exits are *path-dependent on bar history, not on order-fill state* (§8): each exit is a
pure function that, given the bars (OHLCV), a side, and an entry reference, returns the
per-bar stop/target **level array** (or, for a time exit, a trigger mask). The levels are
fully precomputed so an adapter can inject them into the engine's native stop/target
mechanism (``sl_stop``/``tp_stop`` for vectorbt, a Sizer/order wrapper for backtrader, an
Actor for NautilusTrader). The *fill simulation* — did the bar actually fill at the
level, at what price, with what slippage — is the engine's job (§8); this module only
produces deterministic levels and the deterministic *trigger rule* (§4.7/§9) that must
be identical across backtest and paper trading.

The starting set (§8): static stop-loss, ATR stop (fixed multiple + trailing), percentage
trailing stop, time exit, and chandelier exit. Derived series (ATR, running high/low) are
computed from the bars internally (vectorized) when not supplied as ``atr_series`` (§5.2).

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
    "ATRTakeProfit",
    "RiskRewardTP",
    "StopLoss",
    "ATRStop",
    "TrailingStop",
    "TimeExit",
    "ChandelierExit",
    "Exit",
    "ExitPlan",
    "RiskConfig",
    "atr",
    "take_profit_level",
    "atr_take_profit_level",
    "risk_reward_take_profit_level",
    "initial_risk_distance",
    "exit_atr_config",
    "atr_stop_level",
    "trailing_stop_level",
    "chandelier_level",
    "stop_loss_level",
    "time_exit_mask",
    "time_exit_mask_for_entries",
    "exit_level",
    "is_triggered",
    "exit_triggered",
    "first_reached_exit",
    "scale_out_fraction",
    "scale_out_plan",
]

#: The TP/SL trigger rule (§4.7/§9): ``"touched"`` = the bar's high/low reached the
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
    (``StopLoss`` / ``ATRStop`` / ``TrailingStop``).

    Attributes:
        percent: Distance above (long) / below (short) entry, as a fraction.
        scale_out: Fraction of the position to exit when hit, in ``(0, 1]`` (§6.4).
        trigger: The §4.7 trigger rule.
    """

    percent: float
    scale_out: float = 1.0
    trigger: Trigger = "touched"

    def __post_init__(self) -> None:
        object.__setattr__(self, "percent", _validate_positive(self.percent, "TakeProfit.percent"))
        object.__setattr__(self, "scale_out", _validate_scale_out(self.scale_out))
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))


@dataclass(frozen=True)
class ATRTakeProfit:
    """ATR-based take-profit target: a fixed multiple off entry (§8).

    Long: ``target = entry + mult * atr``; short: ``target = entry - mult * atr``. Like
    the fixed :class:`ATRStop`, the level moves only because ATR changes. Unlike a stop,
    it may scale out of the position.

    Attributes:
        mult: ATR multiple.
        trigger: The §4.7 trigger rule.
        scale_out: Fraction of the position to exit when hit, in ``(0, 1]`` (§6.4).
        period: ATR lookback, used when ``atr`` (the aux_data reference) is ``None``.
        atr: Optional ``aux_data`` name (§5.2); resolved by the caller into an
            ``atr_series`` passed to :func:`atr_take_profit_level`.
    """

    mult: float
    trigger: Trigger = "touched"
    scale_out: float = 1.0
    period: int = 14
    atr: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mult", _validate_positive(self.mult, "ATRTakeProfit.mult")
        )
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))
        object.__setattr__(self, "scale_out", _validate_scale_out(self.scale_out))
        object.__setattr__(
            self, "period", _validate_period(self.period, "ATRTakeProfit.period")
        )
        object.__setattr__(self, "atr", _validate_atr_name(self.atr))


@dataclass(frozen=True)
class StopLoss:
    """Static stop-loss: a fixed distance from entry, never ratcheted (§8).

    Long: ``stop = entry * (1 - percent)``; Short: ``stop = entry * (1 + percent)``.
    Unlike :class:`TrailingStop` (which ratchets off the running peak/trough) and
    :class:`ATRStop` (volatility-scaled), this is anchored to entry only and never
    moves. A stop exits 100% of the remaining position (no ``scale_out``).

    Attributes:
        percent: Distance below (long) / above (short) entry, as a fraction of price.
        trigger: The §4.7 trigger rule.
    """

    percent: float
    trigger: Trigger = "touched"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "percent", _validate_positive(self.percent, "StopLoss.percent")
        )
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
        trigger: The §4.7 trigger rule.
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

    With ``activation`` set, the trailing stop does not ratchet until price has made a
    favourable move of that fraction from entry. Until then it is a static,
    entry-anchored stop: ``entry * (1 - percent)`` long / ``entry * (1 + percent)`` short.
    Once activated it trails normally off the running peak/trough. The trailing level is
    always at or above the static level after activation, so the stop never loosens.

    Attributes:
        percent: Trailing distance off the peak/trough, as a fraction.
        trigger: The §4.7 trigger rule.
        activation: Favourable move from entry required before trailing begins, as a
            fraction; ``None`` (default) trails from the entry bar.
    """

    percent: float
    trigger: Trigger = "touched"
    activation: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "percent", _validate_positive(self.percent, "TrailingStop.percent")
        )
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))
        if self.activation is not None:
            object.__setattr__(
                self,
                "activation",
                _validate_positive(self.activation, "TrailingStop.activation"),
            )


@dataclass(frozen=True)
class TimeExit:
    """Time-based exit: exit ``bars`` bars after entry (§8).

    Chosen signature: ``TimeExit(bars)`` — a holding-period exit that fires once the
    position has been held for ``bars`` bars (i.e. at bar ``entry_bar + bars``). The
    "exit at a bar timestamp" form is not implemented separately (it would need calendar
    machinery and is out of scope for item 06). A time exit has no price level and no
    ``trigger`` — time is discrete, so there is no touched/close ambiguity. Durations
    such as ``"12h"`` are converted to a bar count by the caller from the bar interval;
    the library performs no calendar math.

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
        trigger: The §4.7 trigger rule.
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


#: A stop config a :class:`RiskRewardTP` can link its initial risk to.
_RiskStop = StopLoss | ATRStop | TrailingStop | ChandelierExit


@dataclass(frozen=True)
class RiskRewardTP:
    """Take-profit placed at a multiple of the linked stop's initial risk (§6.4, §8).

    The target is anchored once, at entry: ``target = entry ± ratio * risk``, where
    ``risk`` is the distance from ``entry`` to the stop's initial level, computed by
    :func:`initial_risk_distance`. With a ``StopLoss(percent)`` or
    ``TrailingStop(percent)`` the risk is ``entry * percent``; with an ``ATRStop`` it is
    ``mult * ATR``; with a ``ChandelierExit`` it is the distance to the entry bar's
    chandelier level. The target does not move while the linked stop trails — it is a
    fixed risk/reward target set at entry.

    Attributes:
        ratio: Reward-to-risk multiple (``2.0`` means target twice the initial risk).
        stop: The stop whose initial risk anchors the target.
        trigger: The §4.7 trigger rule.
        scale_out: Fraction of the position to exit when hit, in ``(0, 1]`` (§6.4).
    """

    ratio: float
    stop: _RiskStop
    trigger: Trigger = "touched"
    scale_out: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ratio", _validate_positive(self.ratio, "RiskRewardTP.ratio")
        )
        if not isinstance(self.stop, (StopLoss, ATRStop, TrailingStop, ChandelierExit)):
            raise ConfigError(
                "RiskRewardTP.stop must be a StopLoss, ATRStop, TrailingStop, or "
                f"ChandelierExit; got {type(self.stop).__name__}"
            )
        object.__setattr__(self, "trigger", _validate_trigger(self.trigger))
        object.__setattr__(self, "scale_out", _validate_scale_out(self.scale_out))


#: The union of exit config types (§8).
Exit = (
    TakeProfit
    | ATRTakeProfit
    | RiskRewardTP
    | StopLoss
    | ATRStop
    | TrailingStop
    | TimeExit
    | ChandelierExit
)

_EXIT_NAMES: tuple[str, ...] = (
    "TakeProfit",
    "ATRTakeProfit",
    "RiskRewardTP",
    "StopLoss",
    "ATRStop",
    "TrailingStop",
    "TimeExit",
    "ChandelierExit",
)


def _is_exit(obj: object) -> TypeGuard[Exit]:
    return isinstance(
        obj,
        (
            TakeProfit,
            ATRTakeProfit,
            RiskRewardTP,
            StopLoss,
            ATRStop,
            TrailingStop,
            TimeExit,
            ChandelierExit,
        ),
    )


def _is_take_profit(
    cfg: object,
) -> TypeGuard[TakeProfit | ATRTakeProfit | RiskRewardTP]:
    """Whether ``cfg`` is a take-profit family exit (fixed, ATR-scaled, or RR)."""
    return isinstance(cfg, (TakeProfit, ATRTakeProfit, RiskRewardTP))


def exit_atr_config(
    cfg: Exit,
) -> ATRStop | ATRTakeProfit | ChandelierExit | None:
    """The ATR-consuming config ``cfg`` references, or ``None``.

    A :class:`RiskRewardTP` references its nested stop's ATR series; every other ATR
    exit references its own. Engines use this to resolve the aux ``atr_data`` series
    once, regardless of whether ATR sits on the exit or on a nested stop.
    """
    if isinstance(cfg, RiskRewardTP):
        return exit_atr_config(cfg.stop)
    if isinstance(cfg, (ATRStop, ATRTakeProfit, ChandelierExit)):
        return cfg
    return None


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


def atr_take_profit_level(
    cfg: ATRTakeProfit,
    market_data: MarketData,
    *,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Per-bar ATR take-profit target: ``entry ± mult * atr`` (§8).

    Long: ``entry + mult * atr``; short: ``entry - mult * atr``. ``entry_bar`` is
    validated for a uniform dispatch signature but does not affect the level.
    """
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    _validate_entry_bar(entry_bar, md.n_bars)
    atr_arr = _resolve_atr(atr_series, md, cfg.period)
    return entry + cfg.mult * atr_arr if s > 0 else entry - cfg.mult * atr_arr


def initial_risk_distance(
    cfg: RiskRewardTP,
    *,
    market_data: MarketData,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> float:
    """Distance from entry to the linked stop's initial level (§6.4, §8).

    ``StopLoss`` and ``TrailingStop`` are entry-anchored, so the risk is
    ``entry * percent`` regardless of side. ``ATRStop`` (fixed or trailing) uses its
    entry-anchored level ``mult * ATR`` at ``entry_bar``. ``ChandelierExit`` uses the
    distance to the entry bar's chandelier level, which depends on that bar's high/low.
    A non-positive or non-finite distance means the stop sits at or beyond entry and is
    rejected rather than producing an inverted target. A precomputed ``atr_series`` is
    used verbatim: supply a causal (lagged) series so the value read at ``entry_bar``
    was known at entry.
    """
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    stop = cfg.stop

    if isinstance(stop, (StopLoss, TrailingStop)):
        distance = entry * stop.percent
    elif isinstance(stop, ATRStop):
        atr_arr = _resolve_atr(atr_series, md, stop.period)
        distance = stop.mult * atr_arr[eb]
    else:  # ChandelierExit
        atr_arr = _resolve_atr(atr_series, md, stop.period)
        if s > 0:
            level = md.high[eb] - stop.mult * atr_arr[eb]
            distance = entry - level
        else:
            level = md.low[eb] + stop.mult * atr_arr[eb]
            distance = level - entry

    if not math.isfinite(distance) or distance <= 0.0:
        raise ConfigError(
            "RiskRewardTP initial risk distance must be finite and positive; "
            f"got {distance!r} (entry={entry}, stop={type(stop).__name__})"
        )
    return float(distance)


def risk_reward_take_profit_level(
    cfg: RiskRewardTP,
    market_data: MarketData,
    *,
    side: object,
    entry_price: object,
    entry_bar: int = 0,
    atr_series: object | None = None,
) -> np.ndarray:
    """Per-bar risk/reward target: a constant ``entry ± ratio * initial_risk`` (§8).

    The risk is taken from :func:`initial_risk_distance` at ``entry_bar`` and the target
    is anchored there; it does not move while a linked trailing stop ratchets.
    """
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    risk = initial_risk_distance(
        cfg,
        market_data=md,
        side=s,
        entry_price=entry,
        entry_bar=entry_bar,
        atr_series=atr_series,
    )
    target = entry + cfg.ratio * risk if s > 0 else entry - cfg.ratio * risk
    return np.full(md.n_bars, target, dtype=np.float64)


def stop_loss_level(
    cfg: StopLoss, market_data: MarketData, *, side: object, entry_price: object
) -> np.ndarray:
    """Per-bar static stop level: a constant ``entry * (1 ∓ percent)`` (§8).

    Long: ``entry * (1 - percent)``; short: ``entry * (1 + percent)``. Unlike a trailing
    or ATR stop, the level is anchored to entry and never moves.
    """
    md = _require_market_data(market_data)
    s = _validate_side(side)
    entry = _validate_entry_price(entry_price)
    mult = 1.0 - cfg.percent if s > 0 else 1.0 + cfg.percent
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
    entry_price: object | None = None,
) -> np.ndarray:
    """Per-bar trailing stop as a percentage off the running peak/trough (§8).

    Without ``cfg.activation`` the level trails from the entry bar as before. With it, the
    level is the static entry-anchored stop until the running peak (long) / trough (short)
    has moved ``activation`` in the favourable direction from ``entry_price``, then trails
    off that running extreme. ``entry_price`` is therefore required only for the activation
    form. As with a plain trailing stop, the running extreme includes the current bar, so a
    bar that touches the activation threshold and then reverses is treated as armed on that
    same bar (OHLC has no intra-bar ordering).
    """
    md = _require_market_data(market_data)
    s = _validate_side(side)
    eb = _validate_entry_bar(entry_bar, md.n_bars)
    if cfg.activation is None:
        if s > 0:
            peak = _running_extreme_since(md.high, eb, maximum=True)
            return peak * (1.0 - cfg.percent)
        trough = _running_extreme_since(md.low, eb, maximum=False)
        return trough * (1.0 + cfg.percent)

    entry = _validate_entry_price(entry_price)
    n = md.n_bars
    if s > 0:
        peak = _running_extreme_since(md.high, eb, maximum=True)
        armed = np.zeros(n, dtype=np.bool_)
        armed[eb:] = peak[eb:] >= entry * (1.0 + cfg.activation)
        static = entry * (1.0 - cfg.percent)
        level = np.where(np.maximum.accumulate(armed), peak * (1.0 - cfg.percent), static)
    else:
        trough = _running_extreme_since(md.low, eb, maximum=False)
        armed = np.zeros(n, dtype=np.bool_)
        armed[eb:] = trough[eb:] <= entry * (1.0 - cfg.activation)
        static = entry * (1.0 + cfg.percent)
        level = np.where(np.maximum.accumulate(armed), trough * (1.0 + cfg.percent), static)
    level = np.asarray(level, dtype=np.float64)
    level[:eb] = np.nan
    return level


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


def time_exit_mask_for_entries(entries: object, bars: int) -> np.ndarray:
    """Holding-period exit mask driven by an entry-action column (§8, engine helper).

    Vectorized engines receive signal *actions* rather than a single ``entry_bar``: each
    fresh ``entries`` bar re-arms a countdown and the mask fires at ``entry + bars``,
    then goes idle until the next entry (a stale ``True`` must never close a later
    position). This is the entry-column form of :func:`time_exit_mask`.
    """
    ent = np.asarray(entries)
    if ent.ndim != 1 or ent.dtype != np.bool_:
        raise DataShapeError(
            f"entries must be a 1-D boolean array; got dtype {ent.dtype} "
            f"with shape {ent.shape}"
        )
    held = _validate_period(bars, "TimeExit bars")
    n = ent.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    armed = False
    fire_at: int | None = None
    for i in range(n):
        if bool(ent[i]):
            armed = True
            fire_at = i + held
        if armed and fire_at is not None and i >= fire_at:
            mask[i] = True
            armed = False
            fire_at = None
    return mask


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
    """Apply the §4.7/§9 trigger rule to a level array (pure, vectorized).

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
    if _is_take_profit(cfg):
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
    if isinstance(cfg, ATRTakeProfit):
        return atr_take_profit_level(
            cfg,
            market_data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=atr_series,
        )
    if isinstance(cfg, RiskRewardTP):
        return risk_reward_take_profit_level(
            cfg,
            market_data,
            side=side,
            entry_price=entry_price,
            entry_bar=entry_bar,
            atr_series=atr_series,
        )
    if isinstance(cfg, StopLoss):
        return stop_loss_level(cfg, market_data, side=side, entry_price=entry_price)
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
        return trailing_stop_level(
            cfg,
            market_data,
            side=side,
            entry_bar=entry_bar,
            entry_price=entry_price,
        )
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

    A bool array aligned to the bar index; the deterministic trigger rule of §4.7/§9 is
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


def first_reached_exit(
    levels: Sequence[object],
    *,
    open_price: object,
    triggered: Sequence[bool] | None = None,
) -> int | None:
    """Which of several price levels the bar reached first (§4.7 intra-bar ambiguity).

    A bar that sweeps *both* a stop and a target has no unique intra-bar path: OHLC records
    only that each extreme was printed, never the order they printed in. §8 resolves that
    ambiguity by proximity to the bar's **open** — price starts at ``open_price`` and gets to
    the level nearest it first, so a level the open has already crossed is at distance zero
    and fires immediately. This makes "which exit fired" a function of the bar's own geometry
    rather than of the order the exits happen to sit in the config, so a trade whose bar ran
    through the target before the stop is labelled ``take_profit``.

    ``levels[i]`` is exit ``i``'s level on the bar under test. ``triggered[i]`` optionally
    filters to the exits that actually fired; entries that are ``False``, ``None``,
    non-finite, or non-positive are ignored. ``None``/``NaN`` levels stand for an exit with no
    price level (:class:`TimeExit`, or a ``"close"``-triggered exit) — those fill at the bar
    close, so callers should pass the close as their level. Exact ties (a bar symmetric about
    its open) fall back to the configured order, keeping the result deterministic and
    preserving the stop-first default.

    Returns the index of the first exit reached, or ``None`` when nothing fired.
    """
    op = _validate_positive(open_price, "open_price")
    ranked: list[tuple[float, int]] = []
    for i, raw in enumerate(levels):
        if triggered is not None and not bool(triggered[i]):
            continue
        if raw is None:
            continue
        try:
            f = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f) or f <= 0.0:
            continue
        ranked.append((abs(f - op), i))
    if not ranked:
        return None
    return min(ranked)[1]


# ---------------------------------------------------------------------------
# Multi-level exits with scale-out (§6.4).
# ---------------------------------------------------------------------------


def scale_out_fraction(cfg: Exit) -> float:
    """The scale-out fraction an exit exits when it fires (§6.4).

    ``TakeProfit``, ``ATRTakeProfit``, and ``RiskRewardTP`` exit ``cfg.scale_out`` of
    the position; every stop (``ATRStop``, ``TrailingStop``, ``ChandelierExit``,
    ``TimeExit``) always exits 100% of the remaining position.
    """
    if _is_take_profit(cfg):
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
        total = sum(scale_out_fraction(e) for e in ordered if _is_take_profit(e))
        if total > 1.0 + 1e-12:
            raise ConfigError(
                f"take-profit scale_out fractions sum to {total}; must not exceed 1.0"
            )
        object.__setattr__(self, "exit", ordered)

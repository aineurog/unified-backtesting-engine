"""Tests for the exit subsystem (§6.4, §8)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError
from ube.core.risk import (
    TRIGGERS,
    ATRStop,
    ATRTakeProfit,
    ChandelierExit,
    RiskConfig,
    RiskRewardTP,
    SizeModel,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    atr,
    atr_stop_level,
    atr_take_profit_level,
    chandelier_level,
    exit_atr_config,
    exit_level,
    exit_triggered,
    first_reached_exit,
    initial_risk_distance,
    is_triggered,
    risk_reward_take_profit_level,
    scale_out_fraction,
    scale_out_plan,
    stop_loss_level,
    take_profit_level,
    time_exit_mask,
    trailing_stop_level,
)


def _md(close, *, high=None, low=None, open_=None) -> MarketData:
    """Bars; ``high``/``low`` default to close ± 0.5 (OHLC invariants hold)."""
    close = np.asarray(close, dtype=float)
    n = close.shape[0]
    high = np.asarray(high, dtype=float) if high is not None else close + 0.5
    low = np.asarray(low, dtype=float) if low is not None else close - 0.5
    open_ = np.asarray(open_, dtype=float) if open_ is not None else close.copy()
    return MarketData(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=np.ones(n),
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_triggers_constant():
    assert TRIGGERS == ("touched", "close")


@pytest.mark.parametrize(
    "cfg",
    [
        TakeProfit(0.02),
        ATRTakeProfit(3),
        RiskRewardTP(2.0, StopLoss(0.02)),
        StopLoss(0.03),
        ATRStop(3),
        TrailingStop(0.1),
        TimeExit(5),
        ChandelierExit(3),
    ],
)
def test_exit_configs_are_frozen(cfg):
    with pytest.raises(FrozenInstanceError):
        cfg.trigger = "close"  # type: ignore[misc]


def test_take_profit_defaults():
    tp = TakeProfit(0.02)
    assert tp.scale_out == 1.0
    assert tp.trigger == "touched"


def test_take_profit_validation():
    with pytest.raises(ConfigError):
        TakeProfit(0.0)
    with pytest.raises(ConfigError):
        TakeProfit(0.02, scale_out=1.5)
    with pytest.raises(ConfigError):
        TakeProfit(0.02, scale_out=0.0)
    with pytest.raises(ConfigError):
        TakeProfit(0.02, trigger="intraday")  # type: ignore[arg-type]


def test_atr_take_profit_defaults_and_validation():
    tp = ATRTakeProfit(3)
    assert tp.scale_out == 1.0
    assert tp.trigger == "touched"
    assert tp.period == 14
    assert tp.atr is None
    with pytest.raises(ConfigError):
        ATRTakeProfit(0.0)
    with pytest.raises(ConfigError):
        ATRTakeProfit(-1.0)
    with pytest.raises(ConfigError):
        ATRTakeProfit(3, scale_out=1.5)
    with pytest.raises(ConfigError):
        ATRTakeProfit(3, trigger="weekly")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        ATRTakeProfit(3, period=0)
    with pytest.raises(ConfigError):
        ATRTakeProfit(3, atr="")


def test_risk_reward_tp_defaults_and_validation():
    rr = RiskRewardTP(2.0, StopLoss(0.02))
    assert rr.trigger == "touched"
    assert rr.scale_out == 1.0
    assert isinstance(rr.stop, StopLoss)
    with pytest.raises(ConfigError):
        RiskRewardTP(0.0, StopLoss(0.02))
    with pytest.raises(ConfigError):
        RiskRewardTP(-1.0, StopLoss(0.02))
    with pytest.raises(ConfigError):
        RiskRewardTP(2.0, StopLoss(0.02), scale_out=1.5)
    with pytest.raises(ConfigError):
        RiskRewardTP(2.0, StopLoss(0.02), trigger="weekly")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "stop",
    [TakeProfit(0.02), ATRTakeProfit(3), TimeExit(4), "nope"],
)
def test_risk_reward_tp_rejects_unsupported_stop(stop):
    with pytest.raises(ConfigError):
        RiskRewardTP(2.0, stop)  # type: ignore[arg-type]


def test_stop_loss_validation():
    with pytest.raises(ConfigError):
        StopLoss(0.0)
    with pytest.raises(ConfigError):
        StopLoss(-0.05)
    with pytest.raises(ConfigError):
        StopLoss(0.05, trigger="weekly")  # type: ignore[arg-type]


def test_atr_stop_validation():
    with pytest.raises(ConfigError):
        ATRStop(0.0)
    with pytest.raises(ConfigError):
        ATRStop(3, period=0)
    with pytest.raises(ConfigError):
        ATRStop(3, trigger="weekly")  # type: ignore[arg-type]


def test_trailing_stop_validation():
    with pytest.raises(ConfigError):
        TrailingStop(0.0)


def test_time_exit_validation():
    with pytest.raises(ConfigError):
        TimeExit(0)
    with pytest.raises(ConfigError):
        TimeExit(-2)


def test_chandelier_validation():
    with pytest.raises(ConfigError):
        ChandelierExit(0.0)


# ---------------------------------------------------------------------------
# ATR indicator.
# ---------------------------------------------------------------------------


def test_atr_matches_wilder_on_two_bars():
    md = _md([10, 11], high=[10.5, 11.5], low=[9.5, 10.5])
    result = atr(md, period=2)
    # tr = [1.0, 1.5]; Wilder ewm alpha=0.5 -> [1.0, 1.25]
    np.testing.assert_allclose(result, [1.0, 1.25])


def test_atr_is_nonnegative_and_full_length():
    md = _md([10, 11, 12, 11, 13])
    result = atr(md, period=3)
    assert result.shape == (5,)
    assert (result >= 0).all()


def test_atr_rejects_bad_period():
    with pytest.raises(ConfigError):
        atr(_md([10, 11]), period=0)


# ---------------------------------------------------------------------------
# Level functions.
# ---------------------------------------------------------------------------


def test_take_profit_level_long_and_short():
    md = _md([100, 101, 102])
    long_level = take_profit_level(TakeProfit(0.1), md, side=1, entry_price=100)
    short_level = take_profit_level(TakeProfit(0.1), md, side=-1, entry_price=100)
    np.testing.assert_allclose(long_level, [110.0, 110.0, 110.0])
    np.testing.assert_allclose(short_level, [90.0, 90.0, 90.0])


def test_stop_loss_level_long_and_short():
    md = _md([100, 101, 102])
    long_level = stop_loss_level(StopLoss(0.05), md, side=1, entry_price=100)
    short_level = stop_loss_level(StopLoss(0.05), md, side=-1, entry_price=100)
    np.testing.assert_allclose(long_level, [95.0, 95.0, 95.0])
    np.testing.assert_allclose(short_level, [105.0, 105.0, 105.0])


def test_atr_stop_level_fixed():
    md = _md([100, 101, 102])
    atr_series = np.array([1.0, 1.0, 1.0])
    long_level = atr_stop_level(
        ATRStop(3), md, side=1, entry_price=100, atr_series=atr_series
    )
    short_level = atr_stop_level(
        ATRStop(3), md, side=-1, entry_price=100, atr_series=atr_series
    )
    np.testing.assert_allclose(long_level, [97.0, 97.0, 97.0])
    np.testing.assert_allclose(short_level, [103.0, 103.0, 103.0])


def test_atr_take_profit_level_long_and_short():
    md = _md([100, 101, 102])
    atr_series = np.array([1.0, 1.5, 2.0])
    long_level = atr_take_profit_level(
        ATRTakeProfit(3), md, side=1, entry_price=100, atr_series=atr_series
    )
    short_level = atr_take_profit_level(
        ATRTakeProfit(3), md, side=-1, entry_price=100, atr_series=atr_series
    )
    np.testing.assert_allclose(long_level, [103.0, 104.5, 106.0])
    np.testing.assert_allclose(short_level, [97.0, 95.5, 94.0])


def test_atr_take_profit_level_validates_entry_bar_and_side():
    md = _md([100, 101, 102])
    with pytest.raises(ConfigError):
        atr_take_profit_level(
            ATRTakeProfit(3), md, side=1, entry_price=100, entry_bar=3,
            atr_series=np.ones(3),
        )
    with pytest.raises(ConfigError):
        atr_take_profit_level(
            ATRTakeProfit(3), md, side=0, entry_price=100, atr_series=np.ones(3)
        )


@pytest.mark.parametrize(
    "stop",
    [
        StopLoss(0.02),
        TrailingStop(0.02),
        TrailingStop(0.02, activation=0.01),
    ],
)
def test_initial_risk_distance_percentage_stops(stop):
    md = _md([100, 101, 102])
    rr = RiskRewardTP(2.0, stop)
    assert initial_risk_distance(
        rr, market_data=md, side=1, entry_price=100
    ) == pytest.approx(2.0)
    assert initial_risk_distance(
        rr, market_data=md, side=-1, entry_price=100
    ) == pytest.approx(2.0)


@pytest.mark.parametrize("trailing", [False, True])
def test_initial_risk_distance_atr_stop_uses_entry_bar_atr(trailing):
    md = _md([100, 101, 102])
    atr_series = np.array([1.0, 2.0, 3.0])
    rr = RiskRewardTP(2.0, ATRStop(3, trailing=trailing))
    assert initial_risk_distance(
        rr, market_data=md, side=1, entry_price=100, entry_bar=1, atr_series=atr_series
    ) == pytest.approx(6.0)
    assert initial_risk_distance(
        rr, market_data=md, side=-1, entry_price=100, entry_bar=1, atr_series=atr_series
    ) == pytest.approx(6.0)


def test_initial_risk_distance_chandelier_uses_entry_bar_extreme():
    md = _md([100, 101, 102], high=[101, 105, 103], low=[99, 96, 98])
    atr_series = np.ones(3)
    rr = RiskRewardTP(2.0, ChandelierExit(2))
    # long: entry 104, level = high[1] - 2*1 = 103 -> risk 1
    assert initial_risk_distance(
        rr, market_data=md, side=1, entry_price=104, entry_bar=1, atr_series=atr_series
    ) == pytest.approx(1.0)
    # short: entry 97, level = low[1] + 2*1 = 98 -> risk 1
    assert initial_risk_distance(
        rr, market_data=md, side=-1, entry_price=97, entry_bar=1, atr_series=atr_series
    ) == pytest.approx(1.0)


def test_initial_risk_distance_rejects_non_positive():
    md = _md([100, 100], high=[110, 110], low=[99, 99])
    rr = RiskRewardTP(2.0, ChandelierExit(2))
    # long level = 110 - 2 = 108 is above entry 100 -> negative risk
    with pytest.raises(ConfigError):
        initial_risk_distance(
            rr, market_data=md, side=1, entry_price=100, atr_series=np.ones(2)
        )


def test_initial_risk_distance_validates_entry_bar_and_side():
    md = _md([100, 101, 102])
    rr = RiskRewardTP(2.0, StopLoss(0.02))
    with pytest.raises(ConfigError):
        initial_risk_distance(
            rr, market_data=md, side=1, entry_price=100, entry_bar=3
        )
    with pytest.raises(ConfigError):
        initial_risk_distance(rr, market_data=md, side=0, entry_price=100)


def test_risk_reward_take_profit_level_long_and_short():
    md = _md([100, 101, 102])
    rr = RiskRewardTP(2.0, StopLoss(0.02))
    long_level = risk_reward_take_profit_level(rr, md, side=1, entry_price=100)
    short_level = risk_reward_take_profit_level(rr, md, side=-1, entry_price=100)
    # risk = 100 * 0.02 = 2; ratio 2 -> target entry +/- 4
    np.testing.assert_allclose(long_level, [104.0, 104.0, 104.0])
    np.testing.assert_allclose(short_level, [96.0, 96.0, 96.0])


def test_exit_atr_config_resolves_nested_stop():
    atr_stop = ATRStop(2, atr="atr_1h")
    rr = RiskRewardTP(2.0, atr_stop)
    assert exit_atr_config(rr) is atr_stop
    chandelier = ChandelierExit(3, atr="atr_1h")
    assert exit_atr_config(RiskRewardTP(2.0, chandelier)) is chandelier
    assert exit_atr_config(RiskRewardTP(2.0, StopLoss(0.02))) is None
    assert exit_atr_config(ATRTakeProfit(3, atr="atr_x")) is not None


def test_atr_stop_level_trailing_ratchets_up():
    md = _md([100, 102, 101, 105, 104], high=[100, 102, 101, 105, 104])
    atr_series = np.ones(5)
    level = atr_stop_level(
        ATRStop(3, trailing=True), md, side=1, entry_price=100, atr_series=atr_series
    )
    # base = high - 3 = [97, 99, 98, 102, 101]; running max floored at entry-3=97
    np.testing.assert_allclose(level, [97.0, 99.0, 99.0, 102.0, 102.0])


def test_atr_stop_level_trailing_short_ratchets_down():
    md = _md([100, 98, 99, 95, 96], low=[100, 98, 99, 95, 96])
    atr_series = np.ones(5)
    level = atr_stop_level(
        ATRStop(3, trailing=True), md, side=-1, entry_price=100, atr_series=atr_series
    )
    # base = low + 3 = [103, 101, 102, 98, 99]; running min capped at entry+3=103
    np.testing.assert_allclose(level, [103.0, 101.0, 101.0, 98.0, 98.0])


def test_trailing_stop_level_long():
    md = _md([10, 11, 12, 11, 13], high=[10, 11, 12, 11, 13])
    level = trailing_stop_level(TrailingStop(0.1), md, side=1)
    np.testing.assert_allclose(level, [9.0, 9.9, 10.8, 10.8, 11.7])


def test_trailing_stop_level_short():
    md = _md([10, 9, 9.5, 8, 8.5], low=[10, 9, 9.5, 8, 8.5])
    level = trailing_stop_level(TrailingStop(0.1), md, side=-1)
    np.testing.assert_allclose(level, [11.0, 9.9, 9.9, 8.8, 8.8])


def test_trailing_stop_activation_validation():
    assert TrailingStop(0.1).activation is None
    assert TrailingStop(0.1, activation=0.05).activation == 0.05
    with pytest.raises(ConfigError):
        TrailingStop(0.1, activation=0.0)
    with pytest.raises(ConfigError):
        TrailingStop(0.1, activation=-0.01)


def test_trailing_stop_activation_long_uses_static_stop_until_armed():
    md = _md(
        [100, 101, 104, 102, 105],
        high=[100, 103, 105, 104, 106],
        low=[99, 100, 103, 101, 104],
    )
    level = trailing_stop_level(
        TrailingStop(0.10, activation=0.04), md, side=1, entry_price=100
    )
    # Static 90 until the high reaches 104, then trail off the running peak (105, 105, 106).
    np.testing.assert_allclose(level, [90.0, 90.0, 94.5, 94.5, 95.4])


def test_trailing_stop_activation_short_uses_static_stop_until_armed():
    md = _md(
        [100, 98, 96, 94, 93],
        high=[101, 102, 100, 97, 96],
        low=[100, 97, 95, 94, 92],
    )
    level = trailing_stop_level(
        TrailingStop(0.10, activation=0.04), md, side=-1, entry_price=100
    )
    # Static 110 until the low reaches 96, then trail off the running trough (95, 94, 92).
    np.testing.assert_allclose(level, [110.0, 110.0, 104.5, 103.4, 101.2])


def test_trailing_stop_activation_requires_entry_price():
    md = _md([100, 101, 104])
    with pytest.raises(ConfigError):
        trailing_stop_level(TrailingStop(0.10, activation=0.04), md, side=1)


def test_trailing_stop_activation_respects_entry_bar():
    md = _md(
        [100, 101, 104, 102, 105],
        high=[100, 103, 105, 104, 106],
        low=[99, 100, 103, 101, 104],
    )
    level = trailing_stop_level(
        TrailingStop(0.10, activation=0.04), md, side=1, entry_bar=2, entry_price=100
    )
    assert np.isnan(level[:2]).all()
    np.testing.assert_allclose(level[2:], [94.5, 94.5, 95.4])


def test_exit_level_forwards_entry_price_to_activated_trailing_stop():
    md = _md(
        [100, 101, 104, 102, 105],
        high=[100, 103, 105, 104, 106],
        low=[99, 100, 103, 101, 104],
    )
    level = exit_level(
        TrailingStop(0.10, activation=0.04), market_data=md, side=1, entry_price=100
    )
    np.testing.assert_allclose(level, [90.0, 90.0, 94.5, 94.5, 95.4])


def test_chandelier_level_long_and_short():
    atr_series = np.ones(3)
    long_md = _md([10, 11, 12], high=[10, 11, 12])
    short_md = _md([10, 9, 8], low=[10, 9, 8])
    long_level = chandelier_level(
        ChandelierExit(3), long_md, side=1, atr_series=atr_series
    )
    short_level = chandelier_level(
        ChandelierExit(3), short_md, side=-1, atr_series=atr_series
    )
    np.testing.assert_allclose(long_level, [7.0, 8.0, 9.0])
    np.testing.assert_allclose(short_level, [13.0, 12.0, 11.0])


def test_time_exit_mask():
    md = _md([10, 11, 12, 13, 14])
    mask = time_exit_mask(TimeExit(3), md)
    np.testing.assert_array_equal(mask, [False, False, False, True, True])


def test_time_exit_mask_respects_entry_bar():
    md = _md([10, 11, 12, 13, 14])
    mask = time_exit_mask(TimeExit(2), md, entry_bar=2)
    np.testing.assert_array_equal(mask, [False, False, False, False, True])


def test_exit_level_dispatches():
    md = _md([100, 101, 102])
    level = exit_level(
        TakeProfit(0.1), market_data=md, side=1, entry_price=100
    )
    np.testing.assert_allclose(level, [110.0, 110.0, 110.0])


def test_exit_level_dispatches_atr_take_profit():
    md = _md([100, 101, 102])
    level = exit_level(
        ATRTakeProfit(3),
        market_data=md,
        side=1,
        entry_price=100,
        atr_series=np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(level, [103.0, 103.0, 103.0])


def test_exit_level_dispatches_risk_reward_tp():
    md = _md([100, 101, 102])
    level = exit_level(
        RiskRewardTP(2.0, StopLoss(0.02)),
        market_data=md,
        side=1,
        entry_price=100,
    )
    np.testing.assert_allclose(level, [104.0, 104.0, 104.0])


def test_exit_level_dispatches_stop_loss():
    md = _md([100, 101, 102])
    level = exit_level(StopLoss(0.05), market_data=md, side=1, entry_price=100)
    np.testing.assert_allclose(level, [95.0, 95.0, 95.0])


def test_exit_level_rejects_time_exit():
    with pytest.raises(ConfigError):
        exit_level(TimeExit(5), market_data=_md([10, 11]), side=1, entry_price=10)


def test_exit_level_rejects_bad_side():
    with pytest.raises(ConfigError):
        exit_level(TakeProfit(0.1), market_data=_md([10, 11]), side=0, entry_price=10)


# ---------------------------------------------------------------------------
# Trigger rule.
# ---------------------------------------------------------------------------


def test_is_triggered_touched_vs_close():
    high = np.array([6.0, 4.0, 6.0])
    low = np.array([4.0, 4.0, 4.0])
    close = np.array([6.0, 4.0, 6.0])
    level = np.full(3, 5.0)
    touched_above = is_triggered(
        "touched", level, high=high, low=low, close=close, direction="above"
    )
    touched_below = is_triggered(
        "touched", level, high=high, low=low, close=close, direction="below"
    )
    close_above = is_triggered(
        "close", level, high=high, low=low, close=close, direction="above"
    )
    close_below = is_triggered(
        "close", level, high=high, low=low, close=close, direction="below"
    )
    np.testing.assert_array_equal(touched_above, [True, False, True])
    np.testing.assert_array_equal(touched_below, [True, True, True])
    np.testing.assert_array_equal(close_above, [True, False, True])
    np.testing.assert_array_equal(close_below, [False, True, False])


def test_exit_triggered_take_profit_touched():
    md = _md([108, 110, 111], high=[109, 111, 111], low=[99, 99, 99])
    fired = exit_triggered(TakeProfit(0.1), market_data=md, side=1, entry_price=100)
    np.testing.assert_array_equal(fired, [False, True, True])


def test_exit_triggered_atr_take_profit_uses_high_for_long():
    md = _md([100, 103, 102], high=[101, 104, 102], low=[99, 99, 99])
    fired = exit_triggered(
        ATRTakeProfit(3),
        market_data=md,
        side=1,
        entry_price=100,
        atr_series=np.ones(3),
    )
    # target level = 103 (above); touched means high >= 103 -> [False, True, False]
    np.testing.assert_array_equal(fired, [False, True, False])


def test_exit_triggered_risk_reward_tp_uses_high_for_long():
    md = _md([100, 103, 102], high=[101, 105, 104], low=[99, 100, 101])
    fired = exit_triggered(
        RiskRewardTP(2.0, StopLoss(0.02)),
        market_data=md,
        side=1,
        entry_price=100,
    )
    # target level = 104; touched means high >= 104 -> [False, True, True]
    np.testing.assert_array_equal(fired, [False, True, True])


def test_exit_triggered_stop_uses_low_for_long():
    md = _md([100, 101, 102], high=[100, 101, 102], low=[98, 96, 99])
    atr_series = np.ones(3)
    fired = exit_triggered(
        ATRStop(3), market_data=md, side=1, entry_price=100, atr_series=atr_series
    )
    # stop level = 97 (below); touched means low <= 97 -> [False, True, False]
    np.testing.assert_array_equal(fired, [False, True, False])


def test_exit_triggered_stop_loss_uses_low_for_long():
    md = _md([100, 101, 102], high=[100, 101, 102], low=[98, 94, 99])
    fired = exit_triggered(StopLoss(0.05), market_data=md, side=1, entry_price=100)
    # stop level = 95 (below); touched means low <= 95 -> [False, True, False]
    np.testing.assert_array_equal(fired, [False, True, False])


# ---------------------------------------------------------------------------
# Scale-out.
# ---------------------------------------------------------------------------


def test_scale_out_fraction_target_vs_stop():
    assert scale_out_fraction(TakeProfit(0.05, scale_out=0.5)) == 0.5
    assert scale_out_fraction(TakeProfit(0.05)) == 1.0
    assert scale_out_fraction(ATRTakeProfit(3, scale_out=0.25)) == 0.25
    assert scale_out_fraction(ATRTakeProfit(3)) == 1.0
    assert (
        scale_out_fraction(RiskRewardTP(2.0, StopLoss(0.02), scale_out=0.25)) == 0.25
    )
    assert scale_out_fraction(RiskRewardTP(2.0, StopLoss(0.02))) == 1.0
    assert scale_out_fraction(StopLoss(0.05)) == 1.0
    assert scale_out_fraction(ATRStop(3)) == 1.0
    assert scale_out_fraction(TrailingStop(0.1)) == 1.0
    assert scale_out_fraction(ChandelierExit(3)) == 1.0
    assert scale_out_fraction(TimeExit(5)) == 1.0


def test_scale_out_plan_ordered_fractions_and_triggers():
    md = _md([104, 106, 112], high=[104, 106, 112], low=[99, 99, 99])
    exits = [TakeProfit(0.05, scale_out=0.5), TakeProfit(0.10, scale_out=0.5)]
    plan = scale_out_plan(exits, market_data=md, side=1, entry_price=100)
    assert plan.fractions == (0.5, 0.5)
    tp1, tp2 = plan.triggered
    np.testing.assert_array_equal(tp1, [False, True, True])
    np.testing.assert_array_equal(tp2, [False, False, True])


def test_exit_plan_is_frozen_and_readonly():
    md = _md([104, 106, 112], high=[104, 106, 112], low=[99, 99, 99])
    plan = scale_out_plan(
        [TakeProfit(0.05)], market_data=md, side=1, entry_price=100
    )
    with pytest.raises(FrozenInstanceError):
        plan.fractions = (1.0,)  # type: ignore[misc]
    with pytest.raises(ValueError):
        plan.triggered[0][0] = True  # type: ignore[index]


def test_scale_out_plan_rejects_non_exit():
    with pytest.raises(ConfigError):
        scale_out_plan([object()], market_data=_md([100]), side=1, entry_price=100)  # type: ignore[list-item]


def test_scale_out_plan_rejects_wrong_market_data_type():
    with pytest.raises(DataShapeError):
        scale_out_plan([TakeProfit(0.05)], market_data="not data", side=1, entry_price=100)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Same-bar exit ordering (first_reached_exit, §4.7/§8).
# ---------------------------------------------------------------------------


def test_first_reached_exit_prefers_level_nearest_the_open():
    # A bar that sweeps both levels has no unique intra-bar path; §8 breaks the tie by
    # proximity to the open. Here the open sits just above the target, so the target is
    # reached first even though the stop is configured first.
    assert first_reached_exit([100.0, 90.0], open_price=89.0) == 1
    # Mirrored: the open is just below the stop, so the stop is reached first.
    assert first_reached_exit([100.0, 90.0], open_price=99.0) == 0


def test_first_reached_exit_uses_configured_order_on_a_tie():
    # A bar symmetric about its open reaches both at the same distance: fall back to the
    # configured order, which keeps the result deterministic.
    assert first_reached_exit([100.0, 80.0], open_price=90.0) == 0


def test_first_reached_exit_respects_the_trigger_filter():
    levels = [100.0, 90.0]
    assert first_reached_exit(levels, open_price=89.0, triggered=[False, True]) == 1
    assert first_reached_exit(levels, open_price=89.0, triggered=[True, False]) == 0


def test_first_reached_exit_returns_none_when_nothing_fires():
    assert first_reached_exit([], open_price=100.0) is None
    assert first_reached_exit([100.0], open_price=100.0, triggered=[False]) is None
    assert first_reached_exit([None, float("nan")], open_price=100.0) is None


def test_first_reached_exit_rejects_a_non_positive_open():
    with pytest.raises(ConfigError):
        first_reached_exit([100.0], open_price=0.0)


def test_first_reached_exit_labels_a_swept_bar_take_profit_not_stop_loss():
    # The reported bug: a bar whose range covers both the stop and the target was always
    # labelled stop_loss because StopLoss was configured first. The paper configs run
    # StopLoss(0.0002) then TakeProfit(0.0005), so reproduce that exact shape.
    entry = 100.0
    sl, tp = entry * 1.0002, entry * (1 - 0.0005)  # short: stop above, target below
    open_ = tp + 0.00001  # opened near the target, so the target prints first
    assert first_reached_exit([sl, tp], open_price=open_) == 1


# ---------------------------------------------------------------------------
# RiskConfig.
# ---------------------------------------------------------------------------

def test_risk_config_defaults():
    cfg = RiskConfig()
    assert cfg.sizing == SizeModel()
    assert cfg.sizing.kind == "all_in"
    assert cfg.exit == ()


def test_risk_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        RiskConfig().sizing = SizeModel(kind="fixed_units", value=1)  # type: ignore[misc]


def test_risk_config_accepts_list_and_converts_to_tuple():
    cfg = RiskConfig(exit=[TakeProfit(0.05), ATRStop(3)])  # type: ignore[arg-type]
    assert isinstance(cfg.exit, tuple)
    assert len(cfg.exit) == 2


def test_risk_config_accepts_single_exit_shorthand():
    # §7.2 shows ``RiskConfig(exit=ATRStop(...))`` — a bare exit is wrapped into a 1-tuple.
    cfg = RiskConfig(exit=ATRStop(3))  # type: ignore[arg-type]
    assert isinstance(cfg.exit, tuple)
    assert len(cfg.exit) == 1
    assert isinstance(cfg.exit[0], ATRStop)


def test_risk_config_rejects_bad_sizing():
    with pytest.raises(ConfigError):
        RiskConfig(sizing="all_in")  # type: ignore[arg-type]


def test_risk_config_rejects_bad_exit_type():
    with pytest.raises(ConfigError):
        RiskConfig(exit=("nope",))  # type: ignore[arg-type]


def test_risk_config_rejects_oversubscribed_scale_out():
    with pytest.raises(ConfigError):
        RiskConfig(exit=(TakeProfit(0.05, scale_out=0.6), TakeProfit(0.10, scale_out=0.6)))


def test_risk_config_counts_atr_take_profit_in_scale_out_budget():
    with pytest.raises(ConfigError):
        RiskConfig(exit=(ATRTakeProfit(3, scale_out=0.6), TakeProfit(0.10, scale_out=0.6)))
    cfg = RiskConfig(exit=(ATRTakeProfit(3, scale_out=0.5), TakeProfit(0.10, scale_out=0.5)))
    assert len(cfg.exit) == 2


def test_risk_config_counts_risk_reward_tp_in_scale_out_budget():
    with pytest.raises(ConfigError):
        RiskConfig(
            exit=(
                RiskRewardTP(2.0, StopLoss(0.02), scale_out=0.6),
                TakeProfit(0.10, scale_out=0.6),
            )
        )
    cfg = RiskConfig(
        exit=(
            RiskRewardTP(2.0, StopLoss(0.02), scale_out=0.5),
            TakeProfit(0.10, scale_out=0.5),
        )
    )
    assert len(cfg.exit) == 2


def test_risk_config_allows_stop_plus_partial_targets():
    # Stops are not counted toward the scale-out budget.
    cfg = RiskConfig(exit=(TakeProfit(0.05, scale_out=0.5), ATRStop(3)))
    assert len(cfg.exit) == 2

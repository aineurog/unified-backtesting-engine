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
    ChandelierExit,
    RiskConfig,
    SizeModel,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    atr,
    atr_stop_level,
    chandelier_level,
    exit_level,
    exit_triggered,
    is_triggered,
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


def test_risk_config_allows_stop_plus_partial_targets():
    # Stops are not counted toward the scale-out budget.
    cfg = RiskConfig(exit=(TakeProfit(0.05, scale_out=0.5), ATRStop(3)))
    assert len(cfg.exit) == 2

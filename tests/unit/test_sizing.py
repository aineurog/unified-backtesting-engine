"""Tests for the position-sizing subsystem (§6.3)."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ube.core.cost import CostModel
from ube.core.errors import ConfigError
from ube.core.risk import (
    SIZE_KINDS,
    SizeModel,
    all_in_size,
    equal_weight_size,
    fixed_fraction_size,
    fixed_units_size,
    size_position,
    volatility_target_size,
)
from ube.core.risk.sizing import _check_affordable, floor_to_step


def test_size_kinds_constant():
    assert SIZE_KINDS == (
        "fixed_fraction",
        "fixed_units",
        "volatility_target",
        "all_in",
        "equal_weight",
    )


# ---------------------------------------------------------------------------
# SizeModel validation.
# ---------------------------------------------------------------------------


def test_default_model_is_all_in_without_value():
    model = SizeModel()
    assert model.kind == "all_in"
    assert model.value is None


def test_model_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SizeModel().kind = "all_in"  # type: ignore[misc]


def test_unknown_kind_raises():
    with pytest.raises(ConfigError):
        SizeModel(kind="martingale")  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["fixed_fraction", "fixed_units", "volatility_target"])
def test_value_required_kinds_missing_value_raise(kind):
    with pytest.raises(ConfigError):
        SizeModel(kind=kind)  # type: ignore[arg-type]


def test_value_free_kind_rejects_value():
    with pytest.raises(ConfigError):
        SizeModel(kind="all_in", value=0.5)  # type: ignore[arg-type]


def test_volatility_target_accepts_vol_name():
    model = SizeModel(kind="volatility_target", value=0.01, vol="vol_1d")
    assert model.vol == "vol_1d"


def test_vol_forbidden_on_non_volatility_target_kinds():
    with pytest.raises(ConfigError):
        SizeModel(kind="all_in", vol="vol_1d")
    with pytest.raises(ConfigError):
        SizeModel(kind="fixed_fraction", value=0.1, vol="vol_1d")


def test_vol_name_must_be_non_empty_string():
    with pytest.raises(ConfigError):
        SizeModel(kind="volatility_target", value=0.01, vol="")
    with pytest.raises(ConfigError):
        SizeModel(kind="volatility_target", value=0.01, vol=123)  # type: ignore[arg-type]


def test_non_positive_value_raises():
    with pytest.raises(ConfigError):
        SizeModel(kind="fixed_fraction", value=0.0)
    with pytest.raises(ConfigError):
        SizeModel(kind="fixed_units", value=-3)


# ---------------------------------------------------------------------------
# The five sizers.
# ---------------------------------------------------------------------------


def test_fixed_fraction_size():
    # Allocate 10% of 10_000 at price 50 -> 20 units.
    result = fixed_fraction_size(0.1, capital=10_000, price=50)
    assert float(result) == pytest.approx(20.0)


def test_fixed_fraction_size_is_vectorized():
    result = fixed_fraction_size(0.1, capital=np.array([10_000, 20_000]), price=50)
    np.testing.assert_allclose(result, [20.0, 40.0])


def test_fixed_units_size():
    result = fixed_units_size(5)
    assert float(result) == pytest.approx(5.0)


def test_all_in_size():
    result = all_in_size(10_000, 50)
    assert float(result) == pytest.approx(200.0)


def test_equal_weight_size():
    result = equal_weight_size(10_000, 50, n=4)
    assert float(result) == pytest.approx(50.0)


def test_all_in_size_reserves_entry_fee():
    # §7.1: the entry fee must be reserved up front so the order doesn't push the account
    # negative when the venue charges commission on top. With a 5% commission, the units are
    # sized so that notional + entry fee == capital exactly.
    cost = CostModel(commission=0.05)
    units = float(all_in_size(10_000, 50, cost_model=cost))
    assert units == pytest.approx(10_000 / (50 * 1.05))
    # entry outlay = price * units * (1 + commission) must equal the capital.
    assert 50 * units * 1.05 == pytest.approx(10_000)
    # Without a cost model the legacy fee-less behavior is preserved.
    assert float(all_in_size(10_000, 50)) == pytest.approx(200.0)


def test_equal_weight_size_reserves_entry_fee():
    cost = CostModel(commission=0.05)
    units = float(equal_weight_size(10_000, 50, n=4, cost_model=cost))
    assert units == pytest.approx((10_000 / 4) / (50 * 1.05))
    assert 50 * units * 1.05 == pytest.approx(2_500)


def test_volatility_target_size():
    # units = target * capital / (price * vol)
    result = volatility_target_size(0.01, capital=100_000, price=100, vol=0.02)
    assert float(result) == pytest.approx(500.0)


def test_sizers_reject_bad_inputs():
    with pytest.raises(ConfigError):
        fixed_fraction_size(0.1, capital=10_000, price=0)
    with pytest.raises(ConfigError):
        all_in_size(-100, 50)
    with pytest.raises(ConfigError):
        equal_weight_size(10_000, 50, n=0)
    with pytest.raises(ConfigError):
        volatility_target_size(0.01, capital=100_000, price=100, vol=0.0)


# ---------------------------------------------------------------------------
# size_position dispatcher.
# ---------------------------------------------------------------------------


def test_size_position_accepts_bare_kind_for_value_free_kinds():
    assert float(size_position("all_in", capital=10_000, price=50)) == pytest.approx(200.0)
    assert float(
        size_position("equal_weight", capital=10_000, price=50, n=4)
    ) == pytest.approx(50.0)


def test_size_position_accepts_model():
    model = SizeModel(kind="fixed_fraction", value=0.1)
    assert float(size_position(model, capital=10_000, price=50)) == pytest.approx(20.0)


def test_size_position_fixed_units():
    model = SizeModel(kind="fixed_units", value=7)
    assert float(size_position(model, capital=0, price=0)) == pytest.approx(7.0)


def test_size_position_volatility_target():
    model = SizeModel(kind="volatility_target", value=0.01)
    assert float(
        size_position(model, capital=100_000, price=100, vol=0.02)
    ) == pytest.approx(500.0)


def test_size_position_missing_required_kwarg_raises():
    with pytest.raises(ConfigError):
        size_position("equal_weight", capital=10_000, price=50)  # missing n
    with pytest.raises(ConfigError):
        size_position(SizeModel(kind="volatility_target", value=0.01), capital=1, price=1)


def test_size_position_bare_value_requiring_kind_raises():
    with pytest.raises(ConfigError):
        size_position("fixed_fraction", capital=10_000, price=50)


def test_size_position_affordability_guard_raises_on_shortfall():
    # §7.1: deploying 100% of capital plus a 1% entry fee is a genuine shortfall — the
    # in-core guard must surface it as a clear ConfigError before any order is placed,
    # not let the engine halt internally on a negative balance.
    model = SizeModel(kind="fixed_fraction", value=1.0)
    with pytest.raises(ConfigError, match="exceeds available capital"):
        size_position(model, capital=10_000, price=50, cost_model=CostModel(commission=0.01))


def test_size_position_affordability_guard_passes_for_fee_aware_all_in():
    # The fee-adjusted all_in sizer lands exactly on capital, so the guard must not fire.
    units = float(
        size_position("all_in", capital=10_000, price=50, cost_model=CostModel(commission=0.05))
    )
    assert units == pytest.approx(10_000 / (50 * 1.05))


# ---------------------------------------------------------------------------
# floor_to_step — lot-grid conversion for the adapter's final quantity (§7.1).
# ---------------------------------------------------------------------------


def test_floor_to_step_floors_fractional_lots():
    assert float(floor_to_step(52.37, 1.0)) == 52.0
    assert float(floor_to_step(9.987, 0.01)) == pytest.approx(9.98)
    np.testing.assert_allclose(floor_to_step([52.37, 52.99], 1.0), [52.0, 52.0])


def test_floor_to_step_epsilon_absorbs_float_representation_error():
    # A value intended as an exact multiple of the step but stored just below it must
    # floor UP to that multiple (3.0 -> 2.9999999996 floors to 3, not 2).
    assert float(floor_to_step(3.0 - 4e-10, 1.0)) == 3.0


def test_floor_to_step_never_rounds_up():
    for raw in (52.37, 52.5, 52.99, 0.999):
        assert float(floor_to_step(raw, 1.0)) <= raw


def test_floor_to_step_rejects_bad_step():
    with pytest.raises(ConfigError):
        floor_to_step(1.0, 0.0)
    with pytest.raises(ConfigError):
        floor_to_step(1.0, -0.01)
    with pytest.raises(ConfigError):
        floor_to_step(1.0, "1")  # type: ignore[arg-type]


def test_lot_conversion_regression_floor_passes_where_upround_breaches():
    # §7.1 regression pin: the adapter's sequence is size -> floor to lot -> re-check.
    # With $1,000,000 at price 100 and a 5% commission, all_in yields a fractional
    # 9523.809... units. Rounding UP to the whole-share grid breaches capital by one
    # lot's notional + fee; flooring does not, and the re-check on the FLOORED
    # quantity must pass while the same check on the up-rounded quantity raises.
    capital, price = 1_000_000.0, 100.0
    cost = CostModel(commission=0.05)
    raw = float(
        size_position("all_in", capital=capital, price=price, cost_model=cost).ravel()[0]
    )
    floored = float(floor_to_step(raw, 1.0))
    up_rounded = math.ceil(raw)
    assert floored == 9523 and up_rounded == 9524

    _check_affordable(capital, price, np.array([floored]), cost)  # no raise
    with pytest.raises(ConfigError, match="exceeds available capital"):
        _check_affordable(capital, price, np.array([float(up_rounded)]), cost)

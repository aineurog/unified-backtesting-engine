"""Tests for the cost-model interface and asset-class default resolution (§4.5, §7.1)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ube.core.cost import (
    ZERO_COST,
    CostModel,
    carrying_cost,
    fill_cost,
    resolve_cost_model,
)
from ube.core.errors import ConfigError, InvalidInstrumentError
from ube.core.instrument import Instrument

# ---------------------------------------------------------------------------
# Zero-cost default (§7.1).
# ---------------------------------------------------------------------------


def test_zero_cost_is_truly_zero():
    assert ZERO_COST.commission == 0.0
    assert ZERO_COST.slippage == 0.0
    assert ZERO_COST.funding == 0.0
    assert ZERO_COST.borrow == 0.0


def test_default_cost_model_is_zero_cost():
    assert CostModel() == ZERO_COST


def test_fill_and_carrying_cost_are_zero_for_zero_cost_model():
    assert fill_cost(ZERO_COST, notional=1000.0) == 0.0
    assert carrying_cost(ZERO_COST, notional=1000.0, side=1, bar_span=5) == 0.0


# ---------------------------------------------------------------------------
# CostModel validation (§3 principle 6 — fail fast).
# ---------------------------------------------------------------------------


def test_cost_model_is_frozen():
    with pytest.raises(FrozenInstanceError):
        ZERO_COST.commission = 0.001  # type: ignore[misc]


@pytest.mark.parametrize("field", ["commission", "slippage", "funding", "borrow"])
def test_non_numeric_rate_raises(field):
    with pytest.raises(ConfigError):
        CostModel(**{field: "high"})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["commission", "slippage", "funding", "borrow"])
def test_nan_rate_raises(field):
    with pytest.raises(ConfigError):
        CostModel(**{field: float("nan")})


def test_bool_rate_is_rejected():
    with pytest.raises(ConfigError):
        CostModel(commission=True)  # type: ignore[arg-type]


def test_negative_funding_is_allowed():
    # A negative funding rate models received funding (the trader is paid); it is a
    # legitimate value, not a config error.
    model = CostModel(funding=-0.0001)
    assert model.funding == -0.0001


# ---------------------------------------------------------------------------
# resolve_cost_model (§4.5, §7.2, §24).
# ---------------------------------------------------------------------------


def test_resolve_none_is_zero_cost():
    assert resolve_cost_model(None) is ZERO_COST


@pytest.mark.parametrize("asset_class", ["futures", "stocks", "forex", "commodities"])
def test_resolve_non_perp_asset_class_is_zero_cost(asset_class):
    ins = Instrument("X", asset_class=asset_class)
    assert resolve_cost_model(ins) is ZERO_COST


def test_resolve_crypto_perp_is_nonzero_funding_plus_fee():
    model = resolve_cost_model(Instrument("BTC-USDT", asset_class="crypto_perp"))
    assert model.commission > 0.0
    assert model.funding > 0.0
    assert model.slippage == 0.0
    assert model.borrow == 0.0


def test_resolve_non_instrument_raises():
    with pytest.raises(InvalidInstrumentError):
        resolve_cost_model("BTC-USDT")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fill_cost — pure, vectorized function of notional (§7.2).
# ---------------------------------------------------------------------------


def test_fill_cost_is_proportional_to_notional():
    model = CostModel(commission=0.001, slippage=0.002)
    assert fill_cost(model, notional=1000.0) == pytest.approx(0.003 * 1000.0)


def test_fill_cost_is_vectorized():
    model = CostModel(commission=0.01, slippage=0.0)
    notional = np.array([100.0, 200.0, 300.0])
    np.testing.assert_allclose(fill_cost(model, notional=notional), notional * 0.01)


def test_fill_cost_does_not_mutate_input():
    model = CostModel(commission=0.01)
    notional = np.array([100.0, 200.0])
    original = notional.copy()
    fill_cost(model, notional=notional)
    np.testing.assert_array_equal(notional, original)


# ---------------------------------------------------------------------------
# carrying_cost — pure, vectorized function of notional + side + bar span.
# ---------------------------------------------------------------------------


def test_carrying_cost_funding_applies_to_both_sides():
    model = CostModel(funding=0.001)
    notional = 1000.0
    assert carrying_cost(model, notional=notional, side=1, bar_span=2) == pytest.approx(
        0.001 * 1000.0 * 2
    )
    assert carrying_cost(model, notional=notional, side=-1, bar_span=2) == pytest.approx(
        0.001 * 1000.0 * 2
    )


def test_carrying_cost_borrow_applies_to_short_side_only():
    model = CostModel(borrow=0.0005)
    assert carrying_cost(model, notional=1000.0, side=-1, bar_span=1) == pytest.approx(
        0.0005 * 1000.0
    )
    assert carrying_cost(model, notional=1000.0, side=1, bar_span=1) == 0.0


def test_carrying_cost_scales_linearly_with_bar_span():
    model = CostModel(funding=0.0001)
    one = carrying_cost(model, notional=1000.0, side=1, bar_span=1)
    five = carrying_cost(model, notional=1000.0, side=1, bar_span=5)
    assert five == pytest.approx(5 * one)


def test_carrying_cost_is_vectorized():
    model = CostModel(funding=0.001, borrow=0.0005)
    notional = np.array([1000.0, 2000.0])
    side = np.array([1.0, -1.0])
    span = np.array([3.0, 2.0])
    expected = np.array(
        [
            0.001 * 1000.0 * 3.0,
            (0.001 + 0.0005) * 2000.0 * 2.0,
        ]
    )
    np.testing.assert_allclose(
        carrying_cost(model, notional=notional, side=side, bar_span=span), expected
    )

"""Tests for shared-capital portfolio config and allocation/sizing (§ portfolio plan).

Everything here is engine-free: the config, the allocation policies and the
portfolio-aware sizer are pure ``core`` objects.
"""

import numpy as np
import pytest

from ube.core.config import BacktestConfig
from ube.core.cost import CostModel
from ube.core.errors import ConfigError
from ube.core.instrument import Instrument
from ube.core.portfolio import (
    DEFAULT_PORTFOLIO_BALANCE,
    EqualWeight,
    FixedWeight,
    PortfolioConfig,
    RebalancePolicy,
    allocation_weights,
    portfolio_metadata,
    size_portfolio_entry,
)
from ube.core.risk import SizeModel


def _insts() -> tuple[Instrument, Instrument]:
    return (
        Instrument("BTC-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
        Instrument("ETH-USDT", asset_class="crypto_perp", settlement_currency="USDT"),
    )


def _portfolio(**kwargs) -> PortfolioConfig:
    return PortfolioConfig(
        instruments=_insts(),
        base_currency="USDT",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Allocation policies.
# ---------------------------------------------------------------------------


def test_equal_weight_splits_evenly():
    assert EqualWeight().weights(["A", "B"]) == {"A": 0.5, "B": 0.5}
    assert EqualWeight().weights(["A", "B", "C", "D"])["C"] == 0.25


def test_equal_weight_empty_is_empty():
    assert EqualWeight().weights([]) == {}


def test_fixed_weight_keeps_declared_split():
    policy = FixedWeight(weights={"A": 0.6, "B": 0.4})
    assert policy.weights_for(["B", "A"]) == {"B": 0.4, "A": 0.6}


def test_fixed_weight_rejects_empty_mapping():
    with pytest.raises(ConfigError):
        FixedWeight(weights={})


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), "0.5", True])
def test_fixed_weight_rejects_bad_values(bad):
    with pytest.raises(ConfigError):
        FixedWeight(weights={"A": bad})  # type: ignore[dict-item]


def test_fixed_weight_rejects_over_allocation():
    with pytest.raises(ConfigError):
        FixedWeight(weights={"A": 0.7, "B": 0.5})


def test_fixed_weight_allows_cash_remainder():
    policy = FixedWeight(weights={"A": 0.3, "B": 0.2})
    assert sum(policy.weights.values()) == pytest.approx(0.5)


def test_fixed_weight_requires_every_instrument():
    with pytest.raises(ConfigError):
        _portfolio(allocation=FixedWeight(weights={"BTC-USDT": 0.5}))


def test_fixed_weight_rejects_unknown_instrument():
    with pytest.raises(ConfigError):
        _portfolio(
            allocation=FixedWeight(
                weights={"BTC-USDT": 0.4, "ETH-USDT": 0.4, "SOL-USDT": 0.2}
            )
        )


def test_allocation_weights_dispatches():
    assert allocation_weights(EqualWeight(), ["A", "B"]) == {"A": 0.5, "B": 0.5}


def test_allocation_weights_rejects_unknown_policy():
    with pytest.raises(ConfigError):
        allocation_weights(object(), ["A"])  # type: ignore[arg-type]


def test_allocation_weights_requires_an_instrument():
    with pytest.raises(ConfigError):
        allocation_weights(EqualWeight(), [])


# ---------------------------------------------------------------------------
# PortfolioConfig.
# ---------------------------------------------------------------------------


def test_portfolio_defaults_to_equal_weight():
    portfolio = _portfolio()
    assert portfolio.allocation == EqualWeight()
    assert portfolio.rebalance is None
    assert portfolio.sizing is None
    assert portfolio.weights() == {"BTC-USDT": 0.5, "ETH-USDT": 0.5}
    assert portfolio.weight("BTC-USDT") == 0.5


def test_portfolio_rejects_empty_instruments():
    with pytest.raises(ConfigError):
        PortfolioConfig(instruments=(), base_currency="USDT")


def test_portfolio_rejects_duplicate_symbols():
    inst = Instrument("BTC-USDT", asset_class="crypto_perp")
    with pytest.raises(ConfigError):
        PortfolioConfig(instruments=(inst, inst), base_currency="USDT")


def test_portfolio_rejects_empty_base_currency():
    with pytest.raises(ConfigError):
        PortfolioConfig(instruments=_insts(), base_currency=" ")


def test_portfolio_rejects_non_instrument():
    with pytest.raises(ConfigError):
        PortfolioConfig(instruments=("BTC-USDT",), base_currency="USDT")  # type: ignore[arg-type]


def test_portfolio_rejects_rebalancing_in_this_pass():
    with pytest.raises(ConfigError):
        _portfolio(rebalance=RebalancePolicy(every="1D"))


def test_portfolio_rejects_fixed_units_sizing():
    with pytest.raises(ConfigError):
        _portfolio(sizing=SizeModel(kind="fixed_units", value=1.0))


def test_portfolio_rejects_unknown_sizing_type():
    with pytest.raises(ConfigError):
        _portfolio(sizing="all_in")  # type: ignore[arg-type]


def test_portfolio_instrument_ids_and_map():
    portfolio = _portfolio()
    assert portfolio.instrument_ids == ("BTC-USDT", "ETH-USDT")
    assert set(portfolio.instrument_map) == {"BTC-USDT", "ETH-USDT"}


def test_portfolio_weight_of_unknown_instrument_raises():
    with pytest.raises(ConfigError):
        _portfolio().weight("SOL-USDT")


def test_portfolio_validate_data_exact_match():
    _portfolio().validate_data(["ETH-USDT", "BTC-USDT"])  # must not raise


def test_portfolio_validate_data_missing_and_extra():
    with pytest.raises(ConfigError):
        _portfolio().validate_data(["BTC-USDT"])
    with pytest.raises(ConfigError):
        _portfolio().validate_data(["BTC-USDT", "ETH-USDT", "SOL-USDT"])


def test_portfolio_metadata_is_json_safe():
    meta = portfolio_metadata(_portfolio(allocation=FixedWeight(weights={"BTC-USDT": 1.0, "ETH-USDT": 0.0})))
    assert meta["instruments"] == ["BTC-USDT", "ETH-USDT"]
    assert meta["base_currency"] == "USDT"
    assert meta["allocation"] == {
        "kind": "fixed_weight",
        "weights": {"BTC-USDT": 1.0, "ETH-USDT": 0.0},
    }


# ---------------------------------------------------------------------------
# BacktestConfig wiring.
# ---------------------------------------------------------------------------


def test_backtest_config_accepts_portfolio():
    btc, _ = _insts()
    cfg = BacktestConfig(instrument=btc, portfolio=_portfolio())
    cfg.validate(portfolio=True)  # portfolio.base_currency supplies the base currency


def test_backtest_config_rejects_base_currency_disagreement():
    btc, _ = _insts()
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=btc, base_currency="USD", portfolio=_portfolio())


def test_backtest_config_requires_instrument_in_portfolio():
    _, _ = _insts()
    outsider = Instrument("SOL-USDT", asset_class="crypto_perp")
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=outsider, portfolio=_portfolio())


# ---------------------------------------------------------------------------
# size_portfolio_entry.
# ---------------------------------------------------------------------------


def test_default_portfolio_balance_matches_engine_default():
    assert DEFAULT_PORTFOLIO_BALANCE == 100_000.0


def test_size_first_equal_weight_entry_uses_pool():
    units = size_portfolio_entry(
        EqualWeight(),
        portfolio_equity=100_000.0,
        open_commitments={},
        instrument_id="BTC-USDT",
        price=100.0,
    )
    assert float(units) == pytest.approx(1000.0)


def test_size_second_equal_weight_entry_gets_half_the_pool():
    units = size_portfolio_entry(
        EqualWeight(),
        portfolio_equity=100_000.0,
        open_commitments={"BTC-USDT": 50_000.0},
        instrument_id="ETH-USDT",
        price=100.0,
    )
    assert float(units) == pytest.approx(500.0)


def test_size_respects_fixed_weight():
    policy = FixedWeight(weights={"BTC-USDT": 0.4, "ETH-USDT": 0.6})
    units = size_portfolio_entry(
        policy,
        portfolio_equity=100_000.0,
        open_commitments={},
        instrument_id="ETH-USDT",
        price=100.0,
    )
    assert float(units) == pytest.approx(600.0)


def test_size_never_exceeds_available_cash():
    units = size_portfolio_entry(
        FixedWeight(weights={"BTC-USDT": 1.0, "ETH-USDT": 0.0}),
        portfolio_equity=100_000.0,
        open_commitments={"BTC-USDT": 95_000.0, "ETH-USDT": 0.0},
        instrument_id="BTC-USDT",
        price=100.0,
    )
    assert float(units) == pytest.approx(50.0)


def test_size_iterator_topup_is_capped_at_share():
    # The instrument already holds exactly its target -> nothing more to add.
    units = size_portfolio_entry(
        FixedWeight(weights={"BTC-USDT": 0.5, "ETH-USDT": 0.5}),
        portfolio_equity=100_000.0,
        open_commitments={"BTC-USDT": 50_000.0},
        instrument_id="BTC-USDT",
        price=100.0,
    )
    assert float(units) == pytest.approx(0.0)


def test_size_reserves_entry_commission():
    units = size_portfolio_entry(
        EqualWeight(),
        portfolio_equity=100_000.0,
        open_commitments={},
        instrument_id="BTC-USDT",
        price=100.0,
        cost_model=CostModel(commission=0.01),
    )
    assert float(units) == pytest.approx(100_000.0 / (100.0 * 1.01))


def test_size_target_weight_override():
    units = size_portfolio_entry(
        EqualWeight(),
        portfolio_equity=100_000.0,
        open_commitments={},
        instrument_id="BTC-USDT",
        price=100.0,
        target_weight=0.25,
    )
    assert float(units) == pytest.approx(250.0)


def test_size_long_only_never_returns_negative():
    units = size_portfolio_entry(
        FixedWeight(weights={"BTC-USDT": 0.2, "ETH-USDT": 0.2}),
        portfolio_equity=100_000.0,
        open_commitments={"BTC-USDT": 50_000.0},
        instrument_id="BTC-USDT",
        price=100.0,
    )
    assert float(units) == 0.0
    assert float(units) >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"portfolio_equity": -1.0},
        {"price": 0.0},
        {"instrument_id": ""},
        {"open_commitments": {"BTC-USDT": -1.0}},
    ],
)
def test_size_rejects_invalid_inputs(kwargs):
    base = {
        "portfolio_equity": 100_000.0,
        "open_commitments": {},
        "instrument_id": "BTC-USDT",
        "price": 100.0,
    }
    base.update(kwargs)
    with pytest.raises(ConfigError):
        size_portfolio_entry(EqualWeight(), **base)


def test_size_returns_float64_array():
    units = size_portfolio_entry(
        EqualWeight(),
        portfolio_equity=np.float64(100_000.0),
        open_commitments={},
        instrument_id="BTC-USDT",
        price=100.0,
    )
    assert isinstance(units, np.ndarray)
    assert units.dtype == np.float64

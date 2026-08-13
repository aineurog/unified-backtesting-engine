"""Tests for BacktestConfig and SignalConfig (§7, §4.8, §9.3)."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from ube.core.benchmark import BenchmarkConfig
from ube.core.config import (
    OPPOSITE_SIGNAL_POLICIES,
    BacktestConfig,
    SignalConfig,
)
from ube.core.cost import CostModel
from ube.core.errors import ConfigError, UndeclaredConfigError
from ube.core.instrument import Instrument
from ube.core.risk import RiskConfig, SizeModel


def _inst() -> Instrument:
    return Instrument("BTC-USDT", asset_class="crypto_perp")


# ---------------------------------------------------------------------------
# SignalConfig — position-change policy (§9.3, §4.8).
# ---------------------------------------------------------------------------


def test_signal_policy_constant():
    assert OPPOSITE_SIGNAL_POLICIES == ("reverse", "exit_only", "ignore")


def test_signal_config_defaults_to_undeclared():
    # Explicit-over-default (§4.8): None means "undeclared", not a silent default.
    assert SignalConfig().on_opposite_signal is None


def test_signal_config_accepts_each_policy():
    for policy in OPPOSITE_SIGNAL_POLICIES:
        assert SignalConfig(on_opposite_signal=policy).on_opposite_signal == policy


def test_signal_config_rejects_unknown_policy():
    with pytest.raises(ConfigError):
        SignalConfig(on_opposite_signal="flip")  # type: ignore[arg-type]


def test_signal_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SignalConfig().on_opposite_signal = "reverse"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BacktestConfig — composition and defaults (§7.1, §7.2).
# ---------------------------------------------------------------------------


def test_defaults_match_spec():
    cfg = BacktestConfig(instrument=_inst())
    assert cfg.instrument.symbol == "BTC-USDT"
    assert cfg.cost_model is None  # asset-class default resolved later
    assert cfg.risk == RiskConfig()  # all_in sizing, no exits
    assert cfg.signal == SignalConfig()
    assert cfg.benchmark == BenchmarkConfig()  # buy_and_hold
    assert cfg.engine == "auto"
    assert cfg.engine_overrides is None
    assert cfg.date_range is None
    assert cfg.base_currency is None  # NOT silently defaulted (§4.8)
    assert cfg.warmup_bars == 0


def test_config_is_frozen():
    cfg = BacktestConfig(instrument=_inst())
    with pytest.raises(FrozenInstanceError):
        cfg.base_currency = "USD"  # type: ignore[misc]


def test_accepts_explicit_composition():
    cfg = BacktestConfig(
        instrument=_inst(),
        cost_model=CostModel(commission=0.001),
        risk=RiskConfig(sizing=SizeModel(kind="fixed_fraction", value=0.1)),
        signal=SignalConfig(on_opposite_signal="reverse"),
        benchmark=BenchmarkConfig(kind="equal_weight"),
        engine="vectorbt",
        base_currency="USD",
        warmup_bars=10,
    )
    assert cfg.cost_model is not None
    assert cfg.risk.sizing.kind == "fixed_fraction"
    assert cfg.signal.on_opposite_signal == "reverse"
    assert cfg.benchmark.kind == "equal_weight"
    assert cfg.engine == "vectorbt"
    assert cfg.base_currency == "USD"
    assert cfg.warmup_bars == 10


# ---------------------------------------------------------------------------
# Field-level validation raises ConfigError at construction.
# ---------------------------------------------------------------------------


def test_requires_instrument_instance():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument="BTC-USDT")  # type: ignore[arg-type]


def test_invalid_cost_model_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), cost_model="cheap")  # type: ignore[arg-type]


def test_invalid_risk_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), risk="all_in")  # type: ignore[arg-type]


def test_invalid_signal_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), signal="reverse")  # type: ignore[arg-type]


def test_invalid_benchmark_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), benchmark="buy_and_hold")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", 42])
def test_invalid_engine_raises(bad):
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), engine=bad)  # type: ignore[arg-type]


def test_invalid_engine_overrides_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), engine_overrides="freq=1h")  # type: ignore[arg-type]


def test_engine_overrides_are_copied_not_aliased():
    overrides = {"freq": "1h"}
    cfg = BacktestConfig(instrument=_inst(), engine_overrides=overrides)
    overrides["freq"] = "1d"
    assert cfg.engine_overrides == {"freq": "1h"}


def test_invalid_date_range_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), date_range=(1, 2, 3))  # type: ignore[arg-type]


def test_date_range_start_after_end_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(
            instrument=_inst(),
            date_range=(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")),
        )


def test_date_range_stored_as_tuple():
    rng = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    cfg = BacktestConfig(instrument=_inst(), date_range=rng)
    assert cfg.date_range == rng
    assert isinstance(cfg.date_range, tuple)


def test_empty_base_currency_raises():
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), base_currency="")


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_warmup_bars_raises(bad):
    with pytest.raises(ConfigError):
        BacktestConfig(instrument=_inst(), warmup_bars=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate() — explicit-over-default and cross-field constraints (§4.8).
# ---------------------------------------------------------------------------


def test_validate_passes_when_explicit_fields_declared():
    cfg = BacktestConfig(
        instrument=_inst(),
        base_currency="USD",
        signal=SignalConfig(on_opposite_signal="reverse"),
    )
    cfg.validate(portfolio=True, paper_trading=True)  # must not raise


def test_validate_passes_with_no_mode_flags():
    BacktestConfig(instrument=_inst()).validate()  # must not raise


def test_validate_portfolio_requires_base_currency():
    cfg = BacktestConfig(instrument=_inst())  # base_currency None
    with pytest.raises(UndeclaredConfigError):
        cfg.validate(portfolio=True)


def test_validate_portfolio_ok_with_base_currency():
    cfg = BacktestConfig(instrument=_inst(), base_currency="USD")
    cfg.validate(portfolio=True)  # must not raise


def test_validate_paper_trading_requires_on_opposite_signal():
    cfg = BacktestConfig(instrument=_inst())  # signal default: on_opposite_signal None
    with pytest.raises(UndeclaredConfigError):
        cfg.validate(paper_trading=True)


def test_validate_paper_trading_ok_with_policy():
    cfg = BacktestConfig(instrument=_inst(), signal=SignalConfig(on_opposite_signal="exit_only"))
    cfg.validate(paper_trading=True)  # must not raise


def test_undeclared_is_a_config_error():
    assert issubclass(UndeclaredConfigError, ConfigError)

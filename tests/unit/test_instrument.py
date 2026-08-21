"""Tests for the ``Instrument`` asset-class metadata type (§4.5)."""

from dataclasses import FrozenInstanceError

import pytest

from ube.core.errors import ConfigError, InvalidInstrumentError
from ube.core.instrument import ASSET_CLASSES, LONG_ONLY_ASSET_CLASSES, Instrument, allows_short


def test_default_construction_leaves_asset_class_fields_none():
    ins = Instrument("BTC-USDT", asset_class="crypto_perp")
    assert ins.symbol == "BTC-USDT"
    assert ins.asset_class == "crypto_perp"
    assert ins.tick_size is None
    assert ins.contract_multiplier is None
    assert ins.calendar is None
    assert ins.settlement_currency is None
    assert ins.funding_model is None
    assert ins.borrow_model is None


def test_full_construction_carries_all_fields():
    ins = Instrument(
        "ES",
        asset_class="futures",
        tick_size=0.25,
        contract_multiplier=50.0,
        calendar="CMES",
        settlement_currency="USD",
        funding_model="none",
        borrow_model="none",
    )
    assert ins.tick_size == 0.25
    assert ins.contract_multiplier == 50.0
    assert ins.calendar == "CMES"
    assert ins.settlement_currency == "USD"
    assert ins.funding_model == "none"
    assert ins.borrow_model == "none"


def test_is_frozen_dataclass():
    ins = Instrument("BTC-USDT", asset_class="crypto_perp")
    with pytest.raises(FrozenInstanceError):
        ins.symbol = "ETH-USDT"  # type: ignore[misc]


@pytest.mark.parametrize("symbol", ["", "   "])
def test_blank_symbol_raises(symbol):
    with pytest.raises(InvalidInstrumentError):
        Instrument(symbol, asset_class="crypto_perp")


def test_unknown_asset_class_raises():
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="bonds")


@pytest.mark.parametrize("bad", [0.0, -0.25])
def test_non_positive_tick_size_raises(bad):
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="crypto_perp", tick_size=bad)


@pytest.mark.parametrize("bad", [0.0, -50.0])
def test_non_positive_contract_multiplier_raises(bad):
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="futures", contract_multiplier=bad)


def test_asset_classes_match_fixture_labels():
    assert {
        "crypto_perp",
        "crypto_spot",
        "futures",
        "stocks",
        "forex",
        "commodities",
    } == ASSET_CLASSES


def test_crypto_spot_asset_class_is_supported():
    ins = Instrument("BTC-USDT", asset_class="crypto_spot")
    assert ins.asset_class == "crypto_spot"


def test_long_only_asset_classes_constant():
    assert frozenset({"crypto_spot"}) == LONG_ONLY_ASSET_CLASSES


@pytest.mark.parametrize(
    ("asset_class", "expected"),
    [
        ("crypto_spot", False),
        ("crypto_perp", True),
        ("futures", True),
        ("stocks", True),
        ("forex", True),
        ("commodities", True),
    ],
)
def test_allows_short(asset_class, expected):
    assert allows_short(asset_class) is expected
    assert Instrument("X", asset_class=asset_class).allows_short is expected


@pytest.mark.parametrize("bad", ["foo", object()])
def test_non_numeric_tick_size_raises(bad):
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="crypto_perp", tick_size=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_tick_size_raises(bad):
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="crypto_perp", tick_size=bad)


@pytest.mark.parametrize("bad", ["foo", float("nan"), float("inf")])
def test_bad_contract_multiplier_raises(bad):
    with pytest.raises(InvalidInstrumentError):
        Instrument("X", asset_class="futures", contract_multiplier=bad)  # type: ignore[arg-type]


def test_invalid_instrument_error_is_a_config_error():
    assert issubclass(InvalidInstrumentError, ConfigError)

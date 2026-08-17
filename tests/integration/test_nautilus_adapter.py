"""NautilusTrader adapter tests — one file per adapter, append-only.

This is the single test file for the Nautilus adapter and its shared base-contract
dependencies. Later implementation steps APPEND their tests here; this file is never
recreated or deleted. Step 1 covers the :class:`~ube.adapters.base.EngineAdapter`
contract + the engine registry (§4.1).
"""

import numpy as np
import pytest

import ube
import ube.adapters.base as base_mod
from ube.adapters import EngineAdapter, get_engine, register_engine, registered_engines
from ube.adapters.base import _REGISTRY, AUTO_ENGINE_ORDER
from ube.adapters.nautilus_adapter.overrides import (
    DEFAULT_OMS_TYPE,
    DEFAULT_STARTING_BALANCE,
    DEFAULT_VENUE,
    validate_overrides,
)
from ube.core.errors import ConfigError, DataShapeError, InvalidInstrumentError, InvalidSignalError
from ube.core.signals import from_target
from ube.testing.synthetic import PRESETS, synthetic_bars


class _FakeAdapter(EngineAdapter):
    """A concrete EngineAdapter for registry tests."""

    def run(self, data, signals, config, *, aux_data=None):
        raise NotImplementedError


class _OtherAdapter(EngineAdapter):
    """A second concrete EngineAdapter for duplicate-registration tests."""

    def run(self, data, signals, config, *, aux_data=None):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the global registry around every test."""
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Step 1: EngineAdapter contract.
# ---------------------------------------------------------------------------


def test_run_is_abstract():
    assert EngineAdapter.__abstractmethods__ == frozenset({"run"})


def test_instantiating_abstract_adapter_raises():
    with pytest.raises(TypeError):
        EngineAdapter()


# ---------------------------------------------------------------------------
# Step 1: register_engine.
# ---------------------------------------------------------------------------


def test_register_and_get_roundtrip():
    register_engine("my_engine", _FakeAdapter)
    assert get_engine("my_engine") is _FakeAdapter


def test_register_is_case_and_whitespace_insensitive():
    register_engine("My_Engine", _FakeAdapter)
    assert get_engine("  mY_EnGiNe  ") is _FakeAdapter


def test_register_rejects_empty_name():
    with pytest.raises(ConfigError):
        register_engine("", _FakeAdapter)
    with pytest.raises(ConfigError):
        register_engine("   ", _FakeAdapter)


def test_register_rejects_non_string_name():
    with pytest.raises(ConfigError):
        register_engine(None, _FakeAdapter)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        register_engine(123, _FakeAdapter)  # type: ignore[arg-type]


def test_register_rejects_non_adapter_class():
    class NotAnAdapter:
        pass

    with pytest.raises(ConfigError):
        register_engine("bad", NotAnAdapter)
    with pytest.raises(ConfigError):
        register_engine("bad", 42)


def test_register_rejects_abstract_base_class():
    with pytest.raises(ConfigError):
        register_engine("abstract", EngineAdapter)


def test_register_same_class_twice_is_idempotent():
    register_engine("dup", _FakeAdapter)
    register_engine("dup", _FakeAdapter)
    assert get_engine("dup") is _FakeAdapter


def test_register_different_class_same_name_rejected():
    register_engine("dup", _FakeAdapter)
    with pytest.raises(ConfigError, match="already registered"):
        register_engine("dup", _OtherAdapter)


def test_registered_engines_sorted():
    register_engine("zeta", _FakeAdapter)
    register_engine("alpha", _OtherAdapter)
    assert registered_engines() == ("alpha", "zeta")


# ---------------------------------------------------------------------------
# Step 1: get_engine / "auto".
# ---------------------------------------------------------------------------


def test_auto_with_nothing_registered_raises():
    with pytest.raises(ConfigError, match="auto"):
        get_engine("auto")


def test_auto_returns_first_installed_in_priority_order():
    register_engine("nautilus", _FakeAdapter)
    assert get_engine("auto") is _FakeAdapter


def test_auto_priority_respects_order_vectorbt_backtrader_nautilus():
    register_engine("backtrader", _OtherAdapter)
    register_engine("nautilus", _FakeAdapter)
    # vectorbt not installed → backtrader is the first in priority order.
    assert get_engine("auto") is _OtherAdapter
    register_engine("vectorbt", _FakeAdapter)
    assert get_engine("auto") is _FakeAdapter


def test_auto_is_case_insensitive():
    register_engine("NAUTILUS", _FakeAdapter)
    assert get_engine("auto") is _FakeAdapter


def test_get_engine_unknown_name_raises():
    register_engine("nautilus", _FakeAdapter)
    with pytest.raises(ConfigError, match="unknown engine"):
        get_engine("vectorbt")


def test_auto_order_constant_matches_requirement():
    assert AUTO_ENGINE_ORDER == ("vectorbt", "backtrader", "nautilus")


# ---------------------------------------------------------------------------
# Step 1: package-level exposure (§4.1).
# ---------------------------------------------------------------------------


def test_ube_package_exposes_registry_api():
    assert ube.register_engine is register_engine
    assert ube.get_engine is get_engine
    assert ube.registered_engines is registered_engines
    assert ube.EngineAdapter is EngineAdapter


def test_nautilus_adapter_is_registrable_under_canonical_name():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter

    assert issubclass(NautilusAdapter, EngineAdapter)
    register_engine("nautilus", NautilusAdapter)
    assert get_engine("nautilus") is NautilusAdapter
    # Registration is lazy — nothing is auto-registered on import.
    assert {"nautilus": NautilusAdapter} == base_mod._REGISTRY


# ---------------------------------------------------------------------------
# Step 2: NautilusEngineOverrides (§4.3, §7.2).
# ---------------------------------------------------------------------------


def test_validate_overrides_accepts_none():
    assert validate_overrides(None) == {}


def test_validate_overrides_accepts_empty_mapping():
    assert validate_overrides({}) == {}


def test_validate_overrides_returns_fresh_copy():
    overrides = {"venue": "SIM", "leverage": 5.0}
    out = validate_overrides(overrides)
    assert out == overrides
    assert out is not overrides


def test_validate_overrides_accepts_all_valid_keys():
    overrides = {
        "venue": "SIM",
        "account_type": "margin",
        "leverage": 3.0,
        "starting_balance": 100_000.0,
        "price_precision": 2,
        "size_precision": 3,
        "oms_type": "NETTING",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
    }
    assert validate_overrides(overrides) == overrides


def test_validate_overrides_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown engine override 'bogus'"):
        validate_overrides({"bogus": 1})


def test_validate_overrides_rejects_non_mapping():
    with pytest.raises(ConfigError, match="engine_overrides"):
        validate_overrides("SIM")


def test_validate_overrides_rejects_non_string_venue():
    with pytest.raises(ConfigError, match="venue"):
        validate_overrides({"venue": 42})


def test_validate_overrides_rejects_bad_account_type():
    with pytest.raises(ConfigError, match="account_type"):
        validate_overrides({"account_type": "hedge"})
    with pytest.raises(ConfigError, match="account_type"):
        validate_overrides({"account_type": 1})


def test_validate_overrides_rejects_non_positive_leverage():
    with pytest.raises(ConfigError, match="leverage"):
        validate_overrides({"leverage": 0})
    with pytest.raises(ConfigError, match="leverage"):
        validate_overrides({"leverage": "high"})


def test_validate_overrides_rejects_non_positive_balance():
    with pytest.raises(ConfigError, match="starting_balance"):
        validate_overrides({"starting_balance": 0})
    with pytest.raises(ConfigError, match="starting_balance"):
        validate_overrides({"starting_balance": None})


def test_validate_overrides_rejects_bad_precision():
    with pytest.raises(ConfigError, match="price_precision"):
        validate_overrides({"price_precision": -1})
    with pytest.raises(ConfigError, match="size_precision"):
        validate_overrides({"size_precision": 2.5})


def test_validate_overrides_rejects_bad_oms_type():
    with pytest.raises(ConfigError, match="oms_type"):
        validate_overrides({"oms_type": "NETTINGx"})


def test_validate_overrides_rejects_negative_fees():
    with pytest.raises(ConfigError, match="maker_fee"):
        validate_overrides({"maker_fee": -0.001})
    with pytest.raises(ConfigError, match="taker_fee"):
        validate_overrides({"taker_fee": "0.05"})


def test_defaults_are_exported():
    assert DEFAULT_VENUE == "SIM"
    assert DEFAULT_STARTING_BALANCE == 100_000.0
    assert DEFAULT_OMS_TYPE == "NETTING"


def test_overrides_importable_from_package():
    from ube.adapters.nautilus_adapter import NautilusEngineOverrides, validate_overrides

    assert validate_overrides({"venue": "SIM"}) == {"venue": "SIM"}
    assert NautilusEngineOverrides is not None


# ---------------------------------------------------------------------------
# Step 3: Instrument mapping (canonical Instrument -> Nautilus instrument).
# ---------------------------------------------------------------------------


def test_build_instrument_futures_maps_to_futures_contract():
    from nautilus_trader.model.enums import AssetClass
    from nautilus_trader.model.instruments import FuturesContract

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["futures"].instrument)
    assert isinstance(build.instrument, FuturesContract)
    assert str(build.instrument_id) == "ES.SIM"
    assert build.instrument.asset_class == AssetClass.INDEX
    assert build.instrument.multiplier.as_double() == 50.0
    assert build.instrument.price_increment.as_double() == 0.25
    assert build.instrument.price_precision == 2
    assert build.instrument.underlying == "USD"


def test_build_instrument_commodities_maps_to_futures_contract():
    from nautilus_trader.model.enums import AssetClass
    from nautilus_trader.model.instruments import FuturesContract

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["commodities"].instrument)
    assert isinstance(build.instrument, FuturesContract)
    assert str(build.instrument_id) == "GC.SIM"
    assert build.instrument.asset_class == AssetClass.COMMODITY
    assert build.instrument.multiplier.as_double() == 100.0
    assert build.instrument.price_increment.as_double() == 0.1
    assert build.instrument.price_precision == 1


def test_build_instrument_crypto_perp_maps_to_crypto_perpetual():
    from nautilus_trader.model.instruments import CryptoPerpetual

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["crypto_perp"].instrument)
    assert isinstance(build.instrument, CryptoPerpetual)
    assert str(build.instrument_id) == "BTC-USDT.SIM"
    assert build.instrument.is_inverse is False
    assert str(build.instrument.base_currency) == "BTC"
    assert str(build.instrument.quote_currency) == "USDT"
    assert str(build.instrument.settlement_currency) == "USDT"
    assert build.instrument.price_increment.as_double() == 0.1
    assert build.instrument.price_precision == 1
    assert build.instrument.size_precision == 3


def test_build_instrument_stocks_maps_to_equity():
    from nautilus_trader.model.instruments import Equity

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["stocks"].instrument)
    assert isinstance(build.instrument, Equity)
    assert str(build.instrument_id) == "AAPL.SIM"
    assert str(build.instrument.quote_currency) == "USD"
    assert build.instrument.price_increment.as_double() == 0.01
    assert build.instrument.price_precision == 2


def test_build_instrument_forex_maps_to_currency_pair():
    from nautilus_trader.model.instruments import CurrencyPair

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["forex"].instrument)
    assert isinstance(build.instrument, CurrencyPair)
    assert str(build.instrument_id) == "EURUSD.SIM"
    assert str(build.instrument.base_currency) == "EUR"
    assert str(build.instrument.quote_currency) == "USD"
    assert build.instrument.price_increment.as_double() == 0.0001
    assert build.instrument.price_precision == 4


def test_build_instrument_fees_are_zero():
    from decimal import Decimal

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    for asset_class in ("futures", "commodities", "crypto_perp", "stocks", "forex"):
        instrument = build_instrument(PRESETS[asset_class].instrument).instrument
        assert instrument.maker_fee == Decimal("0")
        assert instrument.taker_fee == Decimal("0")


def test_build_instrument_honors_venue_override():
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["futures"].instrument, {"venue": "TEST"})
    assert str(build.instrument_id) == "ES.TEST"


def test_build_instrument_honors_price_precision_override():
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["stocks"].instrument, {"price_precision": 3})
    assert build.instrument.price_precision == 3
    assert build.instrument.price_increment.precision == 3
    assert str(build.instrument.price_increment) == "0.010"


def test_build_instrument_honors_leverage_override():
    from decimal import Decimal

    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    build = build_instrument(PRESETS["stocks"].instrument, {"leverage": 2.0})
    assert build.instrument.margin_init == Decimal("0.5")
    assert build.instrument.margin_maint == Decimal("0.25")


def test_build_instrument_returns_frozen_dataclass():
    from dataclasses import FrozenInstanceError, is_dataclass

    from ube.adapters.nautilus_adapter.instrument_map import (
        NautilusInstrumentBuild,
        build_instrument,
    )

    build = build_instrument(PRESETS["futures"].instrument)
    assert is_dataclass(build)
    assert isinstance(build, NautilusInstrumentBuild)
    with pytest.raises(FrozenInstanceError):
        build.instrument = None


def test_build_instrument_rejects_unsupported_asset_class():
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument
    from ube.core.instrument import Instrument

    spot = Instrument("BTC-USDT", "crypto_spot", tick_size=0.01, settlement_currency="USDT")
    with pytest.raises(InvalidInstrumentError, match="not supported"):
        build_instrument(spot)


# ---------------------------------------------------------------------------
# Step 4: Data + signal bridge (MarketData/Signals -> Nautilus bars + lookup).
# ---------------------------------------------------------------------------


def test_derive_bar_period_ns_is_median_positive_spacing():
    from ube.adapters.nautilus_adapter.adapt_data import derive_bar_period_ns

    md = synthetic_bars(PRESETS["futures"], n_bars=48)
    assert derive_bar_period_ns(md) == 3_600_000_000_000


def test_derive_bar_period_ns_single_bar_raises():
    from ube.adapters.nautilus_adapter.adapt_data import derive_bar_period_ns

    md = synthetic_bars(PRESETS["futures"], n_bars=1)
    with pytest.raises(DataShapeError, match="at least two bars"):
        derive_bar_period_ns(md)


def test_to_nautilus_bars_builds_hourly_bar_type_and_matches_every_bar():
    from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    md = synthetic_bars(PRESETS["futures"], n_bars=48)
    bars, bar_type = to_nautilus_bars(md, build_instrument(PRESETS["futures"].instrument))
    assert str(bar_type) == "ES.SIM-1-HOUR-LAST-EXTERNAL"
    assert len(bars) == 48
    for bar in bars:
        assert bar.bar_type == bar_type


def test_to_nautilus_bars_formats_prices_to_instrument_precision():
    from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    preset = PRESETS["forex"]
    build = build_instrument(preset.instrument)
    md = synthetic_bars(preset, n_bars=24)
    bars, _ = to_nautilus_bars(md, build)
    precision = int(build.instrument.price_precision)
    for i in range(len(bars)):
        assert str(bars[i].open) == f"{md.open[i]:.{precision}f}"
        assert str(bars[i].high) == f"{md.high[i]:.{precision}f}"
        assert str(bars[i].low) == f"{md.low[i]:.{precision}f}"
        assert str(bars[i].close) == f"{md.close[i]:.{precision}f}"


def test_to_nautilus_bars_formats_volume_to_size_precision():
    from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    futures = PRESETS["futures"]
    md = synthetic_bars(futures, n_bars=24)
    bars, _ = to_nautilus_bars(md, build_instrument(futures.instrument))
    for i, bar in enumerate(bars):
        assert bar.volume.as_double() == float(int(round(md.volume[i])))

    crypto = PRESETS["crypto_perp"]
    md = synthetic_bars(crypto, n_bars=24)
    bars, _ = to_nautilus_bars(md, build_instrument(crypto.instrument))
    for i, bar in enumerate(bars):
        assert abs(bar.volume.as_double() - round(float(md.volume[i]), 3)) < 1e-9


def test_to_nautilus_bars_ts_event_equals_ts_init_and_matches_index():
    from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    preset = PRESETS["stocks"]
    md = synthetic_bars(preset, n_bars=24)
    bars, _ = to_nautilus_bars(md, build_instrument(preset.instrument))
    for i, bar in enumerate(bars):
        assert bar.ts_event == int(md.timestamps.asi8[i])
        assert bar.ts_init == bar.ts_event


def test_to_nautilus_bars_treats_missing_volume_as_zero():
    from ube.adapters.nautilus_adapter.adapt_data import to_nautilus_bars
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument
    from ube.core.data import MarketData

    preset = PRESETS["futures"]
    md = synthetic_bars(preset, n_bars=24)
    no_volume = MarketData(
        md.open,
        md.high,
        md.low,
        md.close,
        np.full(md.n_bars, np.nan),
        md.index,
    )
    bars, _ = to_nautilus_bars(no_volume, build_instrument(preset.instrument))
    assert all(bar.volume.as_double() == 0.0 for bar in bars)


def test_build_bar_type_prefers_largest_exact_unit():
    from ube.adapters.nautilus_adapter.adapt_data import build_bar_type
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    iid = build_instrument(PRESETS["futures"].instrument).instrument_id
    hour = 3_600_000_000_000
    assert str(build_bar_type(iid, hour)) == "ES.SIM-1-HOUR-LAST-EXTERNAL"
    assert str(build_bar_type(iid, 30 * 60_000_000_000)) == "ES.SIM-30-MINUTE-LAST-EXTERNAL"
    assert str(build_bar_type(iid, 2 * hour)) == "ES.SIM-2-HOUR-LAST-EXTERNAL"
    assert str(build_bar_type(iid, 7 * 86_400_000_000_000)) == "ES.SIM-1-WEEK-LAST-EXTERNAL"


def test_build_bar_type_falls_back_to_seconds_for_non_divisible_period():
    from ube.adapters.nautilus_adapter.adapt_data import build_bar_type
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument

    iid = build_instrument(PRESETS["futures"].instrument).instrument_id
    # 2000 s is not a valid Nautilus time step (must divide 60/24/...); the
    # deterministic irregular-bar fallback is the 1-SECOND routing label.
    assert str(build_bar_type(iid, 2000 * 1_000_000_000)) == "ES.SIM-1-SECOND-LAST-EXTERNAL"


def test_to_signal_map_aligns_to_bar_timestamps():
    from ube.adapters.nautilus_adapter.adapt_data import to_signal_map

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    sig = from_target([0, 1, 1, 0, -1, -1])
    mapping = to_signal_map(md, sig)
    assert len(mapping) == 6
    ts = md.timestamps.asi8
    assert mapping[int(ts[0])] == (False, False, False, False)
    assert mapping[int(ts[1])] == (True, False, False, False)
    assert mapping[int(ts[3])] == (False, True, False, False)
    assert mapping[int(ts[4])] == (False, False, True, False)


def test_to_signal_map_length_mismatch_raises():
    from ube.adapters.nautilus_adapter.adapt_data import to_signal_map

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    sig = from_target([0, 1, 1])
    with pytest.raises(InvalidSignalError, match="row-aligned"):
        to_signal_map(md, sig)


def main() -> int:
    """Run every ``test_*`` function in this module and report a summary.

    Returns:
        ``0`` when all tests pass, ``1`` otherwise.
    """
    tests = [
        (n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)
    ]

    passed: list[str] = []
    failed: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        snapshot = dict(_REGISTRY)
        _REGISTRY.clear()
        try:
            fn()
            passed.append(name)
        except BaseException as exc:  # noqa: BLE001 — direct-run harness, pytest replaces this.
            failed.append((name, exc))
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(snapshot)

    for name in passed:
        print(f"PASS {name}")
    for name, exc in failed:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
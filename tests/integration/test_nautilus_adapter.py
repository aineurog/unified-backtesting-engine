"""NautilusTrader adapter tests — one file per adapter, append-only.

This is the single test file for the Nautilus adapter and its shared base-contract
dependencies. Later implementation steps APPEND their tests here; this file is never
recreated or deleted. Step 1 covers the :class:`~ube.adapters.base.EngineAdapter`
contract + the engine registry (§4.1).
"""

from dataclasses import replace

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
from ube.core.data import MarketData
from ube.core.errors import ConfigError, InvalidInstrumentError, InvalidSignalError
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
    import ube.core.instrument as instrument_mod
    from ube.adapters.nautilus_adapter.instrument_map import build_instrument
    from ube.core.instrument import Instrument

    # Every core asset class is now mapped by the adapter, so fabricate one more
    # to exercise the rejection path (manual patch so the module-level ``main()``
    # direct-run harness can execute this test too; restore in ``finally``).
    original = instrument_mod.ASSET_CLASSES
    instrument_mod.ASSET_CLASSES = frozenset({*original, "bogus"})
    try:
        spot = Instrument("BTC-USDT", "bogus", tick_size=0.01, settlement_currency="USDT")
        with pytest.raises(InvalidInstrumentError, match="not supported"):
            build_instrument(spot)
    finally:
        instrument_mod.ASSET_CLASSES = original


# ---------------------------------------------------------------------------
# Step 4: Data + signal bridge (MarketData/Signals -> Nautilus bars + lookup).
# ---------------------------------------------------------------------------


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
        assert bar.ts_event == int(md.timestamps.as_unit("ns").asi8[i])  # type: ignore[attr-defined]
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
    ts = md.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# Step 6: end-to-end Nautilus runs — ledger fold + BacktestResult (§4.5, §4.6, §7.3).
# ---------------------------------------------------------------------------


def _bars_from_rows(rows, *, columns=("open", "high", "low", "close"), volume=100.0):
    """Build a MarketData from ``"ts,open,high,low,close[,volume]"`` CSV-like rows.

    The adaptive bar model splits a market fill into sub-fills whose quantity depends
    on the bar's range and volume, so tests must not pin individual sub-fill sizes —
    aggregate quantities are the stable contract. ``volume`` defaults to ``100.0``
    (the synthetic-fixture convention) and is overridable per row via a 6th field.
    """
    import pandas as pd

    arrays = {col: [] for col in columns}
    timestamps = []
    for row in rows:
        parts = row.split(",")
        timestamps.append(parts[0])
        for i, col in enumerate(columns):
            arrays[col].append(float(parts[i + 1]))
    selected = (
        np.full(len(rows), volume)
        if len(rows[0].split(",")) <= 5
        else np.asarray([float(r.split(",")[5]) for r in rows])
    )
    return MarketData(
        open=np.asarray(arrays["open"]),
        high=np.asarray(arrays["high"]),
        low=np.asarray(arrays["low"]),
        close=np.asarray(arrays["close"]),
        volume=selected,
        index=pd.DatetimeIndex(timestamps).as_unit("ns"),
    )


def _fills(result):
    from ube.core.ledger import EventType

    return [e for e in result.ledger if e.event_type is EventType.FILL]


def _ledger_events(result, event_type):
    return [e for e in result.ledger if e.event_type is event_type]


def test_nautilus_full_loop_futures_signal_roundtrip():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType

    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    result = NautilusAdapter().run(
        md,
        signals,
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )

    fills = _fills(result)
    assert len(fills) == 2
    # Entry fills at the signal bar's close; the signal exit at the next close.
    assert [(e.side, e.quantity, e.price, e.exit_reason) for e in fills] == [
        (1, 20.0, 5002.5, None),
        (-1, 20.0, 4989.5, "signal"),
    ]
    assert sum(e.quantity for e in fills[:-1]) == 20.0  # no scale-out partials
    (trade,) = result.trades
    assert trade.quantity == 20.0
    assert trade.entry_price == 5002.5
    assert trade.exit_price == 4989.5
    assert trade.exit_reason == "signal"
    assert trade.net_pnl == -13000.0  # 20 * multiplier 50 * (4989.5 - 5002.5)
    # Self-financing equity (zero-cost model): last == starting + realized.
    assert float(result.equity_curve.equity[0]) == 100000.0
    assert float(result.equity_curve.equity[-1]) == 87000.0
    assert not _ledger_events(result, EventType.COMMISSION)
    assert not _ledger_events(result, EventType.FUNDING_PAYMENT)


def test_nautilus_full_loop_flat_run_stays_flat():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType

    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    result = NautilusAdapter().run(
        md,
        from_target([0] * 12),
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )

    assert _fills(result) == []
    assert result.trades == ()
    assert not _ledger_events(result, EventType.FILL)
    assert (result.equity_curve.equity == 100000.0).all()


def test_nautilus_full_loop_crypto_perp_books_fees_and_funding():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.cost import CostModel
    from ube.core.ledger import EventType

    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    instrument = replace(PRESETS["crypto_perp"].instrument, funding_interval_hours=1.0)
    result = NautilusAdapter().run(
        md,
        from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0002, funding=0.0001),
            engine_overrides={"starting_balance": 100000.0},
        ),
    )

    fills = _fills(result)
    assert len(fills) == 2
    assert [(e.side, round(e.quantity, 3), e.price, e.exit_reason) for e in fills] == [
        (1, 1.65, 60535.9, None),
        (-1, 1.65, 60693.9, "signal"),
    ]
    commissions = _ledger_events(result, EventType.COMMISSION)
    fundings = _ledger_events(result, EventType.FUNDING_PAYMENT)
    assert len(commissions) == 2
    assert all(e.amount > 0.0 for e in commissions)  # commission + slippage, per fill
    assert len(fundings) == 3  # three held bars between the entry and exit bars
    assert all(e.amount > 0.0 for e in fundings)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"
    # all_in now reserves the entry fee in the sized quantity (§7.1), so the entry fills
    # slightly fewer units and the net PnL/equity reflect the correctly-reserved fee.
    assert trade.net_pnl == pytest.approx(90.5904, rel=1e-4)  # gross less fees + funding
    assert float(result.equity_curve.equity[-1]) == pytest.approx(100090.6453, rel=1e-6)


def test_nautilus_full_loop_crypto_perp_short_pays_loss():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.cost import CostModel
    from ube.core.ledger import EventType

    md = synthetic_bars(PRESETS["crypto_perp"], seed=11, n_bars=10)
    instrument = replace(PRESETS["crypto_perp"].instrument, funding_interval_hours=1.0)
    result = NautilusAdapter().run(
        md,
        from_target([0, -1, -1, -1, 0, 0, 0, 0, 0, 0]),
        BacktestConfig(
            instrument=instrument,
            cost_model=CostModel(commission=0.0005, slippage=0.0002, funding=0.0001),
            engine_overrides={"starting_balance": 100000.0},
        ),
    )

    fills = _fills(result)
    assert [(e.side, round(e.quantity, 3), e.price, e.exit_reason) for e in fills] == [
        (-1, 1.65, 60535.9, None),
        (1, 1.65, 60693.9, "signal"),
    ]
    (trade,) = result.trades
    assert trade.side == -1
    # all_in now reserves the entry fee in the sized quantity (§7.1), so the entry fills
    # slightly fewer units and the net PnL/equity reflect the correctly-reserved fee.
    assert trade.net_pnl == pytest.approx(-430.8096, rel=1e-4)
    assert len(_ledger_events(result, EventType.COMMISSION)) == 2
    assert len(_ledger_events(result, EventType.FUNDING_PAYMENT)) == 3
    assert float(result.equity_curve.equity[-1]) == pytest.approx(99569.1904, rel=1e-6)


def test_nautilus_cash_account_short_rejection_raises_engine_error():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import EngineError

    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    with pytest.raises(EngineError, match="market order rejected by the venue"):
        NautilusAdapter().run(
            md,
            from_target([0, -1, -1, -1, -1, -1]),
            BacktestConfig(
                instrument=PRESETS["stocks"].instrument,
                engine_overrides={"account_type": "cash"},
            ),
        )


def test_nautilus_last_bar_rejection_surfaces_after_run():
    # Regression for failure mode B: a market-order rejection on the *final* bar has no
    # subsequent on_bar to trip the per-bar market_rejection check, so the adapter must
    # re-check it after engine.run() returns. The short signal is placed only on the last
    # bar; on a cash account the order is rejected, and the run must raise rather than
    # report success with a silently missing trade.
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import EngineError

    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    with pytest.raises(EngineError, match="market order rejected by the venue"):
        NautilusAdapter().run(
            md,
            from_target([0, 0, 0, 0, 0, -1]),
            BacktestConfig(
                instrument=PRESETS["stocks"].instrument,
                engine_overrides={"account_type": "cash"},
            ),
        )


def test_actor_index_for_raises_on_unknown_timestamp():
    # Regression for failure mode A: a bar whose timestamp does not match any input bar is
    # a genuine data/logic error and must raise, not silently fall back to "the bar after
    # the last one".
    from types import SimpleNamespace

    from ube.adapters.nautilus_adapter.actor import UbeActor, UbeActorConfig
    from ube.adapters.nautilus_adapter.adapter import (
        build_instrument,
        to_nautilus_bars,
        to_signal_map,
    )
    from ube.core.config import BacktestConfig
    from ube.core.errors import EngineError
    from ube.core.risk.sizing import SizeModel

    md = synthetic_bars(PRESETS["stocks"], seed=3, n_bars=6)
    config = BacktestConfig(instrument=PRESETS["stocks"].instrument)
    build = build_instrument(config.instrument, {})
    _, bar_type = to_nautilus_bars(md, build)
    signal_map = to_signal_map(md, from_target([0, 0, 0, 0, 0, 0]))

    actor = UbeActor(
        UbeActorConfig(
            instrument_id=build.instrument_id,
            bar_type=bar_type,
            signal_map=signal_map,
            asset_class=config.instrument.asset_class,
        ),
        market_data=md,
        sizing=SizeModel(),
        exits=(),
        aux_atr=None,
        leverage=1.0,
        cost_model=None,
    )

    ts = int(md.timestamps.as_unit("ns").asi8[3])
    assert actor._index_for(SimpleNamespace(ts_event=ts)) == 3

    with pytest.raises(EngineError, match="not found in the known bar index"):
        actor._index_for(SimpleNamespace(ts_event=999_999_999_999_999_999))


# Reference mirror: rows 2026-08-03 12:00 -> 23:00 of the nautilus_trader
# TRAILING_stop_test.csv feedstock with a 0.1% trailing stop.
_TRAILING_REFERENCE_ROWS = [
    "2026-08-03 12:00:00+00:00,65000.0,65005.0,64995.0,65000.0,1.0",
    "2026-08-03 13:00:00+00:00,65064.99999999999,65078.01299999999,65051.986999999994,65064.99999999999,1.0",  # noqa: E501
    "2026-08-03 14:00:00+00:00,65130.06499999999,65143.09101299998,65117.03898699999,65130.06499999999,1.0",  # noqa: E501
    "2026-08-03 15:00:00+00:00,65195.19506499998,65208.234104012976,65182.15602598698,65195.19506499998,1.0",  # noqa: E501
    "2026-08-03 16:00:00+00:00,65260.390260064974,65273.44233811698,65247.338182012965,65260.390260064974,1.0",  # noqa: E501
    "2026-08-03 17:00:00+00:00,65325.650650325035,65338.715780455095,65312.585520194974,65325.650650325035,1.0",  # noqa: E501
    "2026-08-03 18:00:00+00:00,65390.97630097535,65404.05449623554,65377.89810571516,65390.97630097535,1.0",  # noqa: E501
    "2026-08-03 19:00:00+00:00,65456.36727727632,65469.45855073177,65443.276003820865,65456.36727727632,1.0",  # noqa: E501
    "2026-08-03 20:00:00+00:00,65521.82364455359,65534.9280092825,65508.719279824676,65521.82364455359,1.0",  # noqa: E501
    "2026-08-03 21:00:00+00:00,65587.34546819814,65600.46293729177,65574.2279991045,65587.34546819814,1.0",  # noqa: E501
    "2026-08-03 22:00:00+00:00,65652.93281366632,65666.06340022906,65639.80222710359,65652.93281366632,1.0",  # noqa: E501
    "2026-08-03 23:00:00+00:00,65455.97401522532,65469.06521002836,65442.88282042228,65455.97401522532,1.0",  # noqa: E501
]


def test_nautilus_reference_trailing_stop_mirror():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import RiskConfig, TrailingStop

    data = _bars_from_rows(_TRAILING_REFERENCE_ROWS)
    instrument = replace(PRESETS["crypto_perp"].instrument, funding_interval_hours=1.0)
    result = NautilusAdapter().run(
        data,
        from_target([1] * 12),
        BacktestConfig(
            instrument=instrument,
            risk=RiskConfig(exit=(TrailingStop(0.001),)),
        ),
    )

    # Entry splits: all-in sizing on 100000 at ~65000 -> 1.537 BTC in two fills (the
    # fee-aware all_in size is floored to the lot grid, §7.1); the trailing stop is
    # armed at the running peak * 0.999 and fires on the last (drop) bar — the
    # reference journals its exit on this same 23:00 bar.
    fills = _fills(result)
    assert len(fills) == 4
    assert [(e.side, round(e.quantity, 3), e.price, e.exit_reason) for e in fills] == [
        (1, 0.25, 65000.0, None),
        (1, 1.287, 65000.1, None),
        (-1, 0.25, 65600.4, "trailing_stop"),
        (-1, 1.287, 65600.3, "trailing_stop"),
    ]
    (trade,) = result.trades
    assert trade.exit_reason == "trailing_stop"
    assert trade.entry_price == pytest.approx(65000.0837, abs=1e-3)
    assert trade.exit_price == pytest.approx(65600.4, abs=1e-1)
    assert trade.net_pnl == pytest.approx(711.7444, rel=1e-4)
    assert float(result.equity_curve.equity[0]) == pytest.approx(99939.9282, abs=1e-3)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(100711.7444, rel=1e-6)


def test_nautilus_touched_take_profit_stamps_exit_reason():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import RiskConfig, TakeProfit

    # Short TP: the 97.50 low on the third bar crosses entry*(1-0.02) = 98.0.
    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 02:00:00+00:00,100.00,100.50,97.50,98.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([-1, -1, -1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,  # 1000-share lot via integer size
            risk=RiskConfig(exit=(TakeProfit(0.02),)),
        ),
    )

    fills = _fills(result)
    assert [(e.side, e.quantity, e.price, e.exit_reason) for e in fills] == [
        (-1, 25.0, 100.0, None),
        (-1, 975.0, 99.99, None),
        (1, 25.0, 98.0, "take_profit"),
        (1, 975.0, 98.0, "take_profit"),
    ]
    (trade,) = result.trades
    assert (trade.side, trade.exit_reason) == (-1, "take_profit")
    assert trade.net_pnl == pytest.approx(1990.25, rel=1e-4)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(101990.25, rel=1e-6)


def test_nautilus_stop_loss_stamps_exit_reason():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import RiskConfig, StopLoss

    # Long StopLoss(0.02): the 97.50 low on the third bar crosses entry*(1-0.02) = 98.0.
    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 02:00:00+00:00,100.00,100.50,97.50,98.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(StopLoss(0.02),)),
        ),
    )

    fills = _fills(result)
    assert [(e.side, e.quantity, e.price, e.exit_reason) for e in fills] == [
        (1, 25.0, 100.0, None),
        (1, 975.0, 100.01, None),
        (-1, 25.0, 98.0, "stop_loss"),
        (-1, 975.0, 97.99, "stop_loss"),
    ]
    (trade,) = result.trades
    assert (trade.side, trade.exit_reason) == (1, "stop_loss")
    assert trade.net_pnl == pytest.approx(-2019.5, rel=1e-4)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(97980.5, rel=1e-6)


def test_nautilus_trailing_and_chandelier_ratchet_without_lookahead():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import (
        ChandelierExit,
        RiskConfig,
        TrailingStop,
    )
    from ube.core.risk.exits import atr, chandelier_level, trailing_stop_level

    rows = [
        "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
        "2024-01-01 01:00:00+00:00,100.00,104.00,100.00,103.00",
        "2024-01-01 02:00:00+00:00,103.00,103.00,101.00,102.00",
    ]
    data = _bars_from_rows(rows)

    trailing = NautilusAdapter().run(
        data,
        from_target([1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(TrailingStop(0.02),)),
        ),
    )
    assert [(e.quantity, e.price, e.exit_reason) for e in _fills(trailing)[2:]] == [
        (25.0, 101.92, "trailing_stop"),
        (975.0, 101.91, "trailing_stop"),
    ]
    assert trailing_stop_level(TrailingStop(0.02), data, side=1, entry_bar=0)[2] == pytest.approx(
        101.92, abs=1e-9
    )

    chandelier = NautilusAdapter().run(
        data,
        from_target([1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(ChandelierExit(2, atr="atr_14"),)),
        ),
        aux_data={"atr_14": atr(data, 14)},
    )
    # Ratchet uses only connected bars: the level for bar 1 (104 - 2*ATR[1]) triggers
    # on bar 2. The pre-fix lookahead used bar 2's own ATR (104 - 2*ATR[2] = 101.46).
    assert [(e.quantity, e.price, e.exit_reason) for e in _fills(chandelier)[2:]] == [
        (25.0, 101.57, "chandelier"),
        (975.0, 101.56, "chandelier"),
    ]
    assert chandelier_level(ChandelierExit(2), data, side=1, entry_bar=0, atr_series=atr(data))[
        1
    ] == pytest.approx(101.5714, abs=1e-3)
    assert (
        chandelier_level(ChandelierExit(2), data, side=1, entry_bar=0, atr_series=atr(data))[2]
        < 101.5
    )  # lookahead would be here


# ---------------------------------------------------------------------------
# Step 5: Actor — entries, flips, exits, trigger modes, scale-out, precedence.
# ---------------------------------------------------------------------------


def _fills_by_reason(result, reason):
    return [e for e in _fills(result) if e.exit_reason == reason]


def test_nautilus_actor_entry_sized_in_integer_contracts():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType

    # 100000 / 5000.0 close = 20 whole ES contracts (size precision 0).
    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,5000.00,5008.00,4992.00,5000.00",
            "2024-01-01 01:00:00+00:00,5000.00,5012.00,4990.00,5004.00",
            "2024-01-01 02:00:00+00:00,5004.00,5015.00,4998.00,5010.00",
        ],
        volume=100.0,
    )
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1]),
        BacktestConfig(instrument=PRESETS["futures"].instrument),
    )
    fills = _fills(result)
    assert all(e.side == 1 and e.exit_reason is None for e in fills)
    assert sum(e.quantity for e in fills) == pytest.approx(20.0, abs=1e-9)
    assert result.trades == ()
    (position_value,) = [
        e.position_after for e in result.ledger if e.event_type is EventType.POSITION_CHANGE
    ]
    assert position_value == 20.0  # remains flat after entry, one contract block


def test_nautilus_actor_same_bar_flip_closes_and_opens():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig

    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,102.00,99.90,101.00",
            "2024-01-01 02:00:00+00:00,101.00,103.00,100.50,102.50",
            "2024-01-01 03:00:00+00:00,102.50,103.50,100.00,101.00",
            "2024-01-01 04:00:00+00:00,101.00,101.50,98.50,99.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([0, 1, 1, -1, 0]),
        BacktestConfig(instrument=PRESETS["stocks"].instrument),
    )

    fills = _fills(result)
    assert len(fills) == 8
    entries = [e for e in fills if e.exit_reason is None]
    signals = _fills_by_reason(result, "signal")
    # Two entries (long + short) and two signal closes, each as a pair of fills.
    assert len(entries) == 4 and len(signals) == 4
    long_entry = sum(e.quantity for e in entries if e.side > 0)
    short_entry = sum(e.quantity for e in entries if e.side < 0)
    long_close = sum(e.quantity for e in signals if e.side < 0)
    short_close = sum(e.quantity for e in signals if e.side > 0)
    assert long_entry == pytest.approx(long_close, rel=1e-9)
    assert short_entry == pytest.approx(short_close, rel=1e-9)
    assert len(result.trades) == 2
    assert all(t.exit_reason == "signal" for t in result.trades)
    last_position = result.positions.position[-1]
    assert last_position == 0.0  # flat again after the second signal close


def test_nautilus_actor_atr_stop_touched_fills_at_trigger():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import ATRStop, RiskConfig
    from ube.core.risk.exits import atr, exit_level

    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,101.00,99.50,100.50",
            "2024-01-01 01:00:00+00:00,100.50,102.00,99.00,99.50",
            "2024-01-01 02:00:00+00:00,99.50,100.00,95.00,95.50",
            "2024-01-01 03:00:00+00:00,95.50,96.00,94.00,94.50",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(ATRStop(2, atr="atr_14"),)),
        ),
        aux_data={"atr_14": atr(data, 14)},
    )

    fills = _fills(result)
    assert len(fills) == 4
    entries = [e for e in fills if e.exit_reason is None]
    stops = _fills_by_reason(result, "atr_stop")
    assert len(entries) == 2 and len(stops) == 2
    assert all(e.side == 1 for e in entries)
    assert all(e.side == -1 and round(e.price, 2) in (97.29, 97.28) for e in stops)  # trigger
    (trade,) = result.trades
    assert trade.exit_reason == "atr_stop"
    assert trade.entry_price == pytest.approx(100.51, abs=1e-2)
    assert trade.exit_price == pytest.approx(97.28, abs=1e-2)
    assert trade.net_pnl == pytest.approx(-3213.35, rel=1e-3)
    # The ratcheted trigger is the level on the last completed bar (no lookahead).
    assert exit_level(ATRStop(2), market_data=data, side=1, entry_price=100.5, entry_bar=0)[
        1
    ] == pytest.approx(97.2857, abs=1e-3)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(96786.65, rel=1e-6)


def test_nautilus_actor_time_exit_closes_at_hold_bar():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType
    from ube.core.risk import RiskConfig, TimeExit

    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,101.00,99.50,101.00",
            "2024-01-01 02:00:00+00:00,101.00,102.00,100.50,102.00",
            "2024-01-01 03:00:00+00:00,102.00,103.00,101.50,103.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(TimeExit(2),)),
        ),
    )

    fills = _fills(result)
    assert len(fills) == 4
    times = _fills_by_reason(result, "time_exit")
    assert len(times) == 2
    assert all(round(e.price, 2) in (102.0, 101.99) for e in times)  # hold bar's close
    assert not _fills_by_reason(result, "signal")
    (trade,) = result.trades
    assert trade.exit_reason == "time_exit"
    assert trade.net_pnl == pytest.approx(1980.5, rel=1e-3)
    evaluated = {
        int(e.timestamp): e.action
        for e in result.ledger
        if e.event_type is EventType.SIGNAL_EVALUATED
    }
    assert evaluated[int(data.timestamps.asi8[2])] == "exit_time_exit"
    assert float(result.equity_curve.equity[-1]) == pytest.approx(101980.5, rel=1e-6)


def test_nautilus_actor_close_trigger_mode_ignores_intra_bar_touch():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType
    from ube.core.risk import RiskConfig, TakeProfit

    # Bar 2's high (103.0) touches the target but its close (99.0) does not; only bar 3's
    # close (103.0, above entry*1.02=102.51) may trigger a close-mode exit.
    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,101.00,99.00,100.50",
            "2024-01-01 02:00:00+00:00,100.50,103.00,99.00,99.00",
            "2024-01-01 03:00:00+00:00,99.00,103.50,98.50,103.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([0, 1, 1, 1]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(TakeProfit(0.02, trigger="close"),)),
        ),
    )

    fills = _fills(result)
    assert len(fills) == 4
    targets = _fills_by_reason(result, "take_profit")
    assert len(targets) == 2
    assert all(e.timestamp == int(data.timestamps.asi8[3]) for e in targets)  # bar 3, not 2
    assert all(round(e.price, 2) in (103.0, 102.99) for e in targets)
    (trade,) = result.trades
    assert trade.exit_reason == "take_profit"
    assert trade.net_pnl == pytest.approx(2468.1, rel=1e-3)
    evaluated = {
        int(e.timestamp): e.action
        for e in result.ledger
        if e.event_type is EventType.SIGNAL_EVALUATED
    }
    assert evaluated[int(data.timestamps.asi8[3])] == "exit_take_profit"


def test_nautilus_actor_scale_out_take_profit_then_signal():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.risk import RiskConfig, TakeProfit

    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,101.00,99.00,100.50",
            "2024-01-01 02:00:00+00:00,100.50,103.00,100.00,102.00",
            "2024-01-01 03:00:00+00:00,102.00,103.00,101.50,102.50",
            "2024-01-01 04:00:00+00:00,102.50,103.00,100.00,101.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([0, 1, 1, 1, 0]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(TakeProfit(0.02, scale_out=0.4),)),
        ),
    )

    fills = _fills(result)
    entry_total = sum(e.quantity for e in fills if e.exit_reason is None)
    partials = [e for e in fills if e.exit_reason == "take_profit"]
    signal_close = [e for e in fills if e.exit_reason == "signal"]
    assert all(round(e.price, 2) == 102.51 for e in partials)  # target = last close * 1.02
    assert sum(e.quantity for e in partials) == pytest.approx(entry_total * 0.4, rel=1e-9)
    assert sum(e.quantity for e in signal_close) == pytest.approx(entry_total * 0.6, rel=1e-9)
    (trade,) = result.trades
    assert trade.exit_reason == "signal"  # the final (larger) close wins the round trip
    assert trade.net_pnl == pytest.approx(1083.06, rel=1e-3)
    assert float(result.equity_curve.equity[-1]) == pytest.approx(101083.06, rel=1e-6)


def test_nautilus_actor_close_time_exit_precedes_close_time_signal():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType
    from ube.core.risk import RiskConfig, TimeExit

    # Bar 2 both reaches the 2-bar time exit AND a long_exit signal. Close-time risk
    # exits are evaluated before close-time signal actions (§8), so a single close with
    # reason "time_exit" results, never a duplicate signal close.
    data = _bars_from_rows(
        [
            "2024-01-01 00:00:00+00:00,100.00,100.50,99.50,100.00",
            "2024-01-01 01:00:00+00:00,100.00,101.00,99.50,101.00",
            "2024-01-01 02:00:00+00:00,101.00,102.00,100.50,102.00",
        ]
    )
    result = NautilusAdapter().run(
        data,
        from_target([1, 1, 0]),
        BacktestConfig(
            instrument=PRESETS["stocks"].instrument,
            risk=RiskConfig(exit=(TimeExit(2),)),
        ),
    )

    fills = _fills(result)
    assert len(fills) == 4  # one entry pair + exactly one closing pair
    assert not _fills_by_reason(result, "signal")
    assert len(_fills_by_reason(result, "time_exit")) == 2
    (trade,) = result.trades
    assert trade.exit_reason == "time_exit"
    assert trade.net_pnl == pytest.approx(1980.5, rel=1e-3)
    evaluated = {
        int(e.timestamp): e.action
        for e in result.ledger
        if e.event_type is EventType.SIGNAL_EVALUATED
    }
    assert evaluated[int(data.timestamps.asi8[2])] == "exit_time_exit"


# ---------------------------------------------------------------------------
# Step 7: Error hardening — fail-fast inputs, engine exceptions wrapped (§15).
# ---------------------------------------------------------------------------


def test_nautilus_run_rejects_non_market_data_input():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import DataShapeError

    cfg = BacktestConfig(instrument=PRESETS["futures"].instrument)
    with pytest.raises(DataShapeError, match="MarketData"):
        NautilusAdapter().run(object(), from_target([0] * 6), cfg)


def test_nautilus_run_rejects_non_signals_input():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import InvalidSignalError

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    cfg = BacktestConfig(instrument=PRESETS["futures"].instrument)
    with pytest.raises(InvalidSignalError, match="Signals"):
        NautilusAdapter().run(md, object(), cfg)


def test_nautilus_run_rejects_non_backtest_config():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.errors import ConfigError

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    with pytest.raises(ConfigError, match="BacktestConfig"):
        NautilusAdapter().run(md, from_target([0] * 6), "not-a-config")  # type: ignore[arg-type]


def test_nautilus_run_rejects_row_misaligned_signals():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import InvalidSignalError

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    cfg = BacktestConfig(instrument=PRESETS["futures"].instrument)
    with pytest.raises(InvalidSignalError, match="row-aligned"):
        NautilusAdapter().run(md, from_target([0, 1, 1]), cfg)


def test_nautilus_run_rejects_single_bar_data():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import DataShapeError

    md = synthetic_bars(PRESETS["futures"], n_bars=1)
    cfg = BacktestConfig(instrument=PRESETS["futures"].instrument)
    with pytest.raises(DataShapeError, match="at least two bars"):
        NautilusAdapter().run(md, from_target([0]), cfg)


def test_nautilus_run_rejects_contradictory_signal_entries():
    from ube.core.errors import InvalidSignalError
    from ube.core.signals import Signals

    # Bar 2: both long_entry and short_entry True — rejected in the §6.1 validate
    # phase at construction (before any engine work), naming the offending bar.
    with pytest.raises(InvalidSignalError, match="bar 2"):
        Signals.from_array(
            np.array(
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [True, False, True, False],
                    [False, False, False, False],
                ]
            )
        )


def test_nautilus_run_skips_short_on_crypto_spot():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.instrument import Instrument

    md = synthetic_bars(PRESETS["crypto_perp"], seed=7, n_bars=6)
    instr = Instrument(
        "BTC-USDT", "crypto_spot", tick_size=0.1, calendar="24/7", settlement_currency="USDT"
    )
    cfg = BacktestConfig(
        instrument=instr, engine_overrides={"starting_balance": 100000.0}
    )
    # from_target([0, -1, ...]) emits short_entry at bar 1; spot is long-only, so the
    # shorting gate at the strategy level (§4.5/§6.1) skips it — never a short, and
    # never an error (mirrors the nautilus reference signals.py gate).
    result = NautilusAdapter().run(md, from_target([0, -1, -1, -1, -1, -1]), cfg)
    assert all(e.side >= 0 for e in _fills(result))


def test_nautilus_engine_exception_wrapped_preserves_cause():
    import ube.adapters.nautilus_adapter.adapter as adapter_mod
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.errors import EngineError

    md = synthetic_bars(PRESETS["futures"], n_bars=6)
    cfg = BacktestConfig(instrument=PRESETS["futures"].instrument)

    class _BoomEngine:
        def __init__(self, config=None):
            pass

        def add_venue(self, *args, **kwargs):
            pass

        def add_instrument(self, *args, **kwargs):
            pass

        def add_data(self, *args, **kwargs):
            pass

        def add_strategy(self, *args, **kwargs):
            pass

        def run(self):
            raise RuntimeError("engine exploded")

        def dispose(self):
            pass

    # Patch manually (no monkeypatch) so the module-level ``main()`` direct-run
    # harness can execute this test too; restore in ``finally``.
    original = adapter_mod.BacktestEngine
    adapter_mod.BacktestEngine = _BoomEngine
    try:
        with pytest.raises(EngineError, match="nautilus backtest failed") as excinfo:
            NautilusAdapter().run(md, from_target([0, 1, 1, 0, 0, 0]), cfg)
        assert isinstance(excinfo.value.__cause__, RuntimeError)  # from original preserved
    finally:
        adapter_mod.BacktestEngine = original


def test_nautilus_emits_order_submitted_events():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter
    from ube.core.config import BacktestConfig
    from ube.core.ledger import EventType

    md = synthetic_bars(PRESETS["futures"], seed=7, n_bars=12)
    signals = from_target([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    result = NautilusAdapter().run(
        md,
        signals,
        BacktestConfig(
            instrument=PRESETS["futures"].instrument,
            engine_overrides={"starting_balance": 100000.0},
        ),
    )

    submitted = _ledger_events(result, EventType.ORDER_SUBMITTED)
    # One entry order (long) + one signal-exit order (short); no risk exits in this run.
    assert [(e.side, e.quantity) for e in submitted] == [(1, 20.0), (-1, 20.0)]
    assert all(e.order_id for e in submitted)
    # Submissions land on the two acting bars, in order.
    assert submitted[0].timestamp < submitted[1].timestamp
    # Every fill links back to its submitted order via order_id (§4.6).
    fills = _ledger_events(result, EventType.FILL)
    submitted_ids = {e.order_id for e in submitted}
    assert fills and all(e.order_id in submitted_ids for e in fills)


# ---------------------------------------------------------------------------
# Step 8: engine-name resolution + lazy built-in registration (§7.1).
# ---------------------------------------------------------------------------


def test_resolve_engine_name_auto_with_nothing_registered_raises():
    with pytest.raises(ConfigError, match="auto"):
        ube.resolve_engine_name("auto")


def test_resolve_engine_name_auto_returns_first_in_priority_order():
    register_engine("nautilus", _FakeAdapter)
    assert ube.resolve_engine_name("auto") == "nautilus"
    register_engine("backtrader", _OtherAdapter)
    assert ube.resolve_engine_name("auto") == "backtrader"


def test_resolve_engine_name_normalizes_specific_name():
    assert ube.resolve_engine_name("  NAUTILUS  ") == "nautilus"


def test_resolve_engine_name_unknown_name_returns_normalized():
    # Resolution only normalizes; whether the name is *registered* is get_engine's job.
    assert ube.resolve_engine_name("  VeCtOrBt  ") == "vectorbt"


def test_ensure_builtin_engines_registered_registers_nautilus():
    from ube.adapters.nautilus_adapter.adapter import NautilusAdapter

    ube.ensure_builtin_engines_registered()
    assert ube.get_engine("nautilus") is NautilusAdapter
    assert ube.resolve_engine_name("auto") == "nautilus"


def test_ensure_builtin_engines_registered_is_idempotent():
    ube.ensure_builtin_engines_registered()
    ube.ensure_builtin_engines_registered()
    assert registered_engines() == ("nautilus",)


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

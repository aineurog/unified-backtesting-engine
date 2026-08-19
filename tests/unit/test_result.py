"""Tests for BacktestResult (§7.3, §4.6)."""

import uuid
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.benchmark import buy_and_hold_curve
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import DataShapeError, UndeclaredConfigError
from ube.core.instrument import Instrument
from ube.core.ledger import EventLedger, EventType, LedgerEvent, Trade, equity_curve
from ube.core.result import BacktestResult, result_hash

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _ns(ts):
    """Normalize a timestamp to int64 ns (int bar index or pandas Timestamp)."""
    if isinstance(ts, pd.Timestamp):
        return int(ts.value)
    return int(ts)


def _fill(ts, iid, side, qty, price):
    return LedgerEvent(EventType.FILL, _ns(ts), iid, side=side, quantity=qty, price=price)


def _position(ts, iid, position_after):
    return LedgerEvent(
        EventType.POSITION_CHANGE, _ns(ts), iid, position_after=position_after
    )


def _cash(ts, amount, currency="USD"):
    return LedgerEvent(
        EventType.CASH_MOVEMENT, _ns(ts), "portfolio", amount=amount, currency=currency
    )


def _time_bars(timestamps, closes):
    close = np.asarray(closes, dtype=float)
    n = close.shape[0]
    return MarketData(
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.ones(n),
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
    )


_T0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
_T1 = pd.Timestamp("2024-01-01T11:00", tz="UTC")
_T2 = pd.Timestamp("2024-01-01T12:00", tz="UTC")
_T3 = pd.Timestamp("2024-01-01T13:00", tz="UTC")


def _single_instrument():
    """A single-instrument USD run with one completed round-trip trade."""
    market_data = {"A": _time_bars([_T0, _T1, _T2], [100.0, 110.0, 120.0])}
    instruments = {"A": Instrument("A", asset_class="stocks", settlement_currency="USD")}
    ledger = EventLedger(
        [
            _cash(_T0, 1000.0),
            _fill(_T0, "A", 1, 10.0, 100.0),
            _position(_T0, "A", 10.0),
            _fill(_T1, "A", -1, 10.0, 110.0),
            _position(_T1, "A", 0.0),
        ]
    )
    config = BacktestConfig(instrument=instruments["A"], base_currency="USD")
    return ledger, market_data, instruments, config


def _two_trades(warmup_bars):
    """Two round-trip trades over four bars (trade entries at t0 and t2)."""
    market_data = {"A": _time_bars([_T0, _T1, _T2, _T3], [100.0, 110.0, 115.0, 120.0])}
    instruments = {"A": Instrument("A", asset_class="stocks", settlement_currency="USD")}
    ledger = EventLedger(
        [
            _cash(_T0, 1000.0),
            _fill(_T0, "A", 1, 10.0, 100.0),
            _position(_T0, "A", 10.0),
            _fill(_T1, "A", -1, 10.0, 110.0),
            _position(_T1, "A", 0.0),
            _fill(_T2, "A", 1, 5.0, 115.0),
            _position(_T2, "A", 5.0),
            _fill(_T3, "A", -1, 5.0, 120.0),
            _position(_T3, "A", 0.0),
        ]
    )
    config = BacktestConfig(
        instrument=instruments["A"], base_currency="USD", warmup_bars=warmup_bars
    )
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments, run_id="run-1"
    )
    return result, ledger, market_data, instruments


# ---------------------------------------------------------------------------
# Construction and derived views (§4.6, §7.3).
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    with pytest.raises(FrozenInstanceError):
        result.equity_curve = result.equity_curve  # type: ignore[misc]


def test_trades_are_derived_and_correct():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert isinstance(t, Trade)
    assert t.side == 1
    assert t.entry_price == 100.0
    assert t.exit_price == 110.0
    assert t.gross_pnl == pytest.approx(100.0)


def test_trades_view_applies_contract_multiplier():
    # from_ledger must derive .trades with the same instruments map as the equity
    # curves — otherwise a futures trade's notional/gross_pnl would use multiplier 1.0
    # while the equity curve used the real multiplier (§4.6, §7.3).
    instruments = {
        "ES": Instrument(
            "ES", asset_class="futures", contract_multiplier=50.0, settlement_currency="USD"
        )
    }
    market_data = {"ES": _time_bars([_T0, _T1], [5000.0, 5100.0])}
    ledger = EventLedger(
        [
            _fill(_T0, "ES", 1, 1.0, 5000.0),
            _position(_T0, "ES", 1.0),
            _fill(_T1, "ES", -1, 1.0, 5100.0),
            _position(_T1, "ES", 0.0),
        ]
    )
    config = BacktestConfig(instrument=instruments["ES"], base_currency="USD")
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    (t,) = result.trades
    assert t.entry_price == pytest.approx(5000.0)
    assert t.exit_price == pytest.approx(5100.0)
    assert t.entry_notional == pytest.approx(1.0 * 5000.0 * 50.0)
    assert t.gross_pnl == pytest.approx(1.0 * (5100.0 - 5000.0) * 50.0)
    # The equity curve marks to the same multiplier: 1 contract × 5000 × 50.
    assert result.equity_curve.equity.tolist() == pytest.approx([250000.0, 0.0])


def test_positions_are_derived_for_single_instrument():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    assert result.positions is not None
    assert result.positions.position.tolist() == [10.0, 0.0]
    assert result.positions.timestamps.tolist() == [_ns(_T0), _ns(_T1)]


def test_equity_curve_is_derived_and_correct():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    # cash 1000 + 10 * [100, 110, 120], flat at t1/t2.
    assert result.equity_curve.equity.tolist() == pytest.approx([2000.0, 1000.0, 1000.0])
    assert result.equity_curve_by_instrument["A"].equity.tolist() == pytest.approx(
        [1000.0, 0.0, 0.0]
    )


def test_metrics_defaults_to_none_and_benchmark_optional():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    assert result.metrics is None
    assert result.benchmark is None


def test_run_id_defaults_to_uuid_and_can_be_set():
    ledger, market_data, instruments, config = _single_instrument()
    defaulted = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    # A fresh UUID4 string is generated when not supplied.
    assert uuid.UUID(defaulted.run_id)

    explicit = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments, run_id="abc"
    )
    assert explicit.run_id == "abc"


def test_benchmark_is_stored_when_provided():
    ledger, market_data, instruments, config = _single_instrument()
    curve = buy_and_hold_curve(market_data["A"])
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments, benchmark=curve
    )
    assert result.benchmark is curve


# ---------------------------------------------------------------------------
# Cached once: derived views are computed at construction, not on access.
# ---------------------------------------------------------------------------


def test_derived_views_are_cached_not_recomputed():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    n_trades = len(result.trades)
    equity_before = result.equity_curve.equity.copy()

    # Append a new event to the (append-only) ledger after construction: the result's
    # cached views must NOT change — they were derived once (§4.6).
    ledger.append(_fill(_T2, "A", 1, 1.0, 120.0))

    assert len(result.trades) == n_trades
    assert result.equity_curve.equity.tolist() == equity_before.tolist()


def test_derived_views_do_not_alias_the_ledger_after_the_fact():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    # The stored equity array is independent of any recomputation from the mutated ledger.
    fresh = equity_curve(ledger, market_data, instruments, base_currency="USD")
    assert result.equity_curve.equity is not fresh.equity


# ---------------------------------------------------------------------------
# Single vs portfolio: positions and base_currency.
# ---------------------------------------------------------------------------


def test_positions_none_for_multi_instrument():
    market_data = {"A": _time_bars([_T0], [100.0]), "B": _time_bars([_T0], [50.0])}
    instruments = {
        "A": Instrument("A", asset_class="stocks", settlement_currency="USD"),
        "B": Instrument("B", asset_class="stocks", settlement_currency="USD"),
    }
    ledger = EventLedger([_position(_T0, "A", 1.0), _position(_T0, "B", 2.0)])
    config = BacktestConfig(instrument=instruments["A"], base_currency="USD")
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    assert result.positions is None


def test_portfolio_without_base_currency_raises_undeclared():
    market_data = {"A": _time_bars([_T0], [100.0]), "B": _time_bars([_T0], [50.0])}
    instruments = {
        "A": Instrument("A", asset_class="stocks", settlement_currency="USD"),
        "B": Instrument("B", asset_class="stocks", settlement_currency="USD"),
    }
    ledger = EventLedger([_position(_T0, "A", 1.0), _position(_T0, "B", 2.0)])
    config = BacktestConfig(instrument=instruments["A"])  # base_currency None
    with pytest.raises(UndeclaredConfigError):
        BacktestResult.from_ledger(
            ledger, config, market_data=market_data, instruments=instruments
        )


def test_single_instrument_without_base_currency_uses_settlement():
    market_data = {"A": _time_bars([_T0, _T1, _T2], [100.0, 110.0, 120.0])}
    instruments = {"A": Instrument("A", asset_class="stocks", settlement_currency="USD")}
    ledger = EventLedger([_cash(_T0, 1000.0), _position(_T0, "A", 10.0)])
    config = BacktestConfig(instrument=instruments["A"])  # base_currency None
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    assert result.equity_curve.equity.tolist() == pytest.approx([2000.0, 2100.0, 2200.0])


def test_from_ledger_requires_ledger_and_config_types():
    ledger, market_data, instruments, config = _single_instrument()
    with pytest.raises(DataShapeError):
        BacktestResult.from_ledger(
            "ledger", config, market_data=market_data, instruments=instruments  # type: ignore[arg-type]
        )
    with pytest.raises(DataShapeError):
        BacktestResult.from_ledger(
            ledger, "config", market_data=market_data, instruments=instruments  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Warm-up exclusion (§7.2).
# ---------------------------------------------------------------------------


def test_warmup_slices_equity_and_drops_warmup_trades():
    result, _, _, _ = _two_trades(warmup_bars=1)
    # Trade 1 (entry at t0, bar 0) is dropped; trade 2 (entry at t2, bar 2) is kept.
    assert len(result.trades) == 1
    assert result.trades[0].entry_timestamp == _ns(_T2)
    # Equity is sliced to bars [t1, t2, t3].
    assert len(result.equity_curve.index) == 3
    # mark: t1=0, t2=5*115=575, t3=0 -> cash 1000 => [1000, 1575, 1000].
    assert result.equity_curve.equity.tolist() == pytest.approx([1000.0, 1575.0, 1000.0])
    # Per-instrument breakdown (marks only) is sliced the same way.
    assert result.equity_curve_by_instrument["A"].equity.tolist() == pytest.approx(
        [0.0, 575.0, 0.0]
    )


def test_warmup_zero_keeps_everything():
    result, _, _, _ = _two_trades(warmup_bars=0)
    assert len(result.trades) == 2
    assert len(result.equity_curve.index) == 4


def test_warmup_greater_than_length_drops_everything():
    result, _, _, _ = _two_trades(warmup_bars=4)
    assert result.trades == ()
    assert len(result.equity_curve.index) == 0


# ---------------------------------------------------------------------------
# Persistence: save / load round-trip (§7.3).
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments, run_id="run-1"
    )
    path = tmp_path / "result.pkl"
    result.save(path)

    loaded = BacktestResult.load(path)
    assert loaded.run_id == "run-1"
    assert len(loaded.trades) == 1
    assert loaded.equity_curve.equity.tolist() == result.equity_curve.equity.tolist()
    assert set(loaded.equity_curve_by_instrument) == set(
        result.equity_curve_by_instrument
    )
    assert loaded.config.base_currency == "USD"
    assert loaded.metrics is None
    # The loaded result is still frozen.
    with pytest.raises(FrozenInstanceError):
        loaded.metrics = 1.0  # type: ignore[misc]


def test_load_rejects_non_result(tmp_path):
    path = tmp_path / "not_a_result.pkl"
    with open(path, "wb") as fh:
        import pickle

        pickle.dump({"not": "a result"}, fh)
    with pytest.raises(DataShapeError):
        BacktestResult.load(path)


def test_load_preserves_read_only_arrays(tmp_path):
    # Pickle drops ndarray.flags.writeable; load() must re-apply it (§3 principle 5).
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    path = tmp_path / "result.pkl"
    result.save(path)
    loaded = BacktestResult.load(path)

    assert not loaded.equity_curve.equity.flags.writeable
    assert not loaded.equity_curve_by_instrument["A"].equity.flags.writeable
    with pytest.raises(ValueError):
        loaded.equity_curve.equity[0] = 999.0


# ---------------------------------------------------------------------------
# result_hash (§4.8, §16) — deterministic fingerprint of trades + equity.
# ---------------------------------------------------------------------------


def test_result_hash_is_stable_hex_digest():
    ledger, market_data, instruments, config = _single_instrument()
    result = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments, run_id="r"
    )
    digest = result_hash(result)
    assert isinstance(digest, str)
    assert len(digest) == 64
    int(digest, 16)  # valid hex
    assert result_hash(result) == digest


def test_result_hash_changes_when_equity_changes():
    ledger, market_data, instruments, config = _single_instrument()
    a = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    # Same ledger (same trades), different mark-to-market closes -> different equity.
    shifted = {
        k: MarketData(
            open=v.open + 1.0,
            high=v.high + 1.0,
            low=v.low + 1.0,
            close=v.close + 1.0,
            volume=v.volume,
            index=v.index,
        )
        for k, v in market_data.items()
    }
    b = BacktestResult.from_ledger(
        ledger, config, market_data=shifted, instruments=instruments
    )
    assert a.trades == b.trades  # trades derive from the ledger, not the marks
    assert result_hash(a) != result_hash(b)


def test_result_hash_differs_across_different_runs():
    ledger, market_data, instruments, config = _single_instrument()
    one = BacktestResult.from_ledger(
        ledger, config, market_data=market_data, instruments=instruments
    )
    result_two, _, _, _ = _two_trades(0)
    assert result_hash(one) != result_hash(result_two)


def test_result_hash_rejects_non_result():
    with pytest.raises(DataShapeError):
        result_hash("not a result")  # type: ignore[arg-type]

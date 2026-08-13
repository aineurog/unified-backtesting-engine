"""Tests for the append-only event ledger and its derived views (§4.6)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, FXRateUnavailableError
from ube.core.instrument import Instrument
from ube.core.ledger import (
    EVENT_TYPES,
    EquityCurve,
    EventLedger,
    EventType,
    FXSeries,
    LedgerEvent,
    Positions,
    Trade,
    equity_curve,
    equity_curve_by_instrument,
    positions,
    trades,
)

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _ns(ts):
    """Normalize an event timestamp (int bar index or pandas Timestamp) to int64 ns."""
    if isinstance(ts, pd.Timestamp):
        return int(ts.value)
    return int(ts)


def _fill(ts, iid, side, qty, price, *, gap=False):
    return LedgerEvent(
        EventType.FILL, _ns(ts), iid, side=side, quantity=qty, price=price, gap_fill=gap
    )


def _commission(ts, iid, amount, currency="USD"):
    return LedgerEvent(EventType.COMMISSION, _ns(ts), iid, amount=amount, currency=currency)


def _funding(ts, iid, amount, currency="USD"):
    return LedgerEvent(
        EventType.FUNDING_PAYMENT, _ns(ts), iid, amount=amount, currency=currency
    )


def _position(ts, iid, position_after):
    return LedgerEvent(EventType.POSITION_CHANGE, _ns(ts), iid, position_after=position_after)


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
        bar_type="time",
    )


# ---------------------------------------------------------------------------
# Event schema: sealed type set + per-type validation.
# ---------------------------------------------------------------------------


def test_event_types_are_sealed_and_match_spec():
    assert tuple(e.value for e in EVENT_TYPES) == (
        "signal_evaluated",
        "order_submitted",
        "fill",
        "funding_payment",
        "futures_rollover",
        "commission",
        "cash_movement",
        "position_change",
    )
    # StrEnum: the string name is the discriminator value.
    assert str(EventType.FILL) == "fill"
    assert EventType("fill") is EventType.FILL


def test_event_is_frozen():
    e = _fill(0, "A", 1, 10.0, 100.0)
    with pytest.raises(FrozenInstanceError):
        e.price = 99.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _fill(0, "A", 2, 10.0, 100.0),  # side not +/-1
        lambda: _fill(0, "A", 1, -10.0, 100.0),  # quantity sign != side
        lambda: _fill(0, "A", 1, 0.0, 100.0),  # zero quantity
        lambda: _fill(0, "A", 1, 10.0, -5.0),  # non-positive price
        lambda: LedgerEvent(EventType.FILL, 0, "A"),  # missing required fields
        lambda: LedgerEvent(EventType.COMMISSION, 0, "A", amount=-1.0, currency="USD"),
        lambda: LedgerEvent(EventType.COMMISSION, 0, "A", amount=1.0),  # missing currency
        lambda: LedgerEvent(EventType.CASH_MOVEMENT, 0, "A", amount=1.0, currency=""),
        lambda: LedgerEvent(EventType.POSITION_CHANGE, 0, "A"),  # missing position_after
    ],
)
def test_malformed_event_raises_datashape(builder):
    with pytest.raises(DataShapeError):
        builder()


def test_timestamp_must_be_integer():
    with pytest.raises(DataShapeError):
        LedgerEvent(EventType.FILL, 1.5, "A", side=1, quantity=1.0, price=1.0)  # type: ignore[arg-type]
    with pytest.raises(DataShapeError):
        LedgerEvent(EventType.FILL, True, "A", side=1, quantity=1.0, price=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Append-only enforcement + instrument tagging.
# ---------------------------------------------------------------------------


def test_ledger_is_append_only_and_entries_cannot_be_replaced():
    ledger = EventLedger()
    e = _fill(0, "A", 1, 10.0, 100.0)
    ledger.append(e)
    ledger.append(_fill(1, "A", -1, 10.0, 110.0))
    assert len(ledger) == 2
    assert ledger[0] is e
    assert ledger.events == (ledger[0], ledger[1])

    # The snapshot is a tuple: it cannot alias the internal buffer.
    snapshot = ledger.events
    ledger.append(_fill(2, "A", 1, 5.0, 105.0))
    assert len(snapshot) == 2  # snapshot did not grow

    # Appending a non-LedgerEvent is rejected.
    with pytest.raises(DataShapeError):
        ledger.append("nope")  # type: ignore[arg-type]


def test_ledger_filters_by_instrument_id():
    ledger = EventLedger(
        [
            _fill(0, "A", 1, 10.0, 100.0),
            _fill(1, "B", 1, 2.0, 50.0),
            _fill(2, "A", -1, 10.0, 110.0),
        ]
    )
    assert [e.instrument_id for e in ledger.by_instrument("A")] == ["A", "A"]
    assert [e.instrument_id for e in ledger.by_instrument("B")] == ["B"]
    assert ledger.by_instrument("C") == ()
    assert ledger.instrument_ids() == ("A", "B")


# ---------------------------------------------------------------------------
# trades view (round-trip fold).
# ---------------------------------------------------------------------------


def test_trades_long_round_trip():
    ledger = EventLedger(
        [
            _fill(0, "A", 1, 10.0, 100.0),
            _commission(0, "A", 5.0),
            _fill(1, "A", -1, 10.0, 110.0),
        ]
    )
    result = trades(ledger)
    assert len(result) == 1
    t = result[0]
    assert isinstance(t, Trade)
    assert t.instrument_id == "A"
    assert t.side == 1
    assert t.quantity == 10.0
    assert t.entry_price == 100.0
    assert t.exit_price == 110.0
    assert t.gross_pnl == pytest.approx(100.0)
    assert t.commission == pytest.approx(5.0)
    assert t.net_pnl == pytest.approx(95.0)


def test_trades_short_round_trip_with_funding_received():
    ledger = EventLedger(
        [
            _fill(0, "B", -1, 10.0, 100.0),
            _funding(1, "B", -2.0),  # received funding -> reduces cost
            _fill(2, "B", 1, 10.0, 90.0),
        ]
    )
    (t,) = trades(ledger)
    assert t.side == -1
    assert t.entry_price == 100.0
    assert t.exit_price == 90.0
    assert t.gross_pnl == pytest.approx(100.0)  # sold 1000, bought back 900
    assert t.funding == pytest.approx(-2.0)
    assert t.net_pnl == pytest.approx(102.0)


def test_trades_partial_exit_and_flip():
    ledger = EventLedger(
        [
            _fill(0, "A", 1, 10.0, 100.0),   # long 10 @ 100
            _fill(1, "A", -1, 4.0, 110.0),   # exit 4 @ 110
            _fill(2, "A", -1, 6.0, 120.0),   # exit 6 @ 120
            _fill(3, "A", -1, 5.0, 130.0),   # flip: open short 5 @ 130
            _fill(4, "A", 1, 5.0, 125.0),    # cover short @ 125
        ]
    )
    result = trades(ledger)
    assert len(result) == 2
    long_t, short_t = result
    # Long round trip: bought 10@100 = -1000, sold 4@110 + 6@120 = +440 +720 = +1160.
    assert long_t.side == 1
    assert long_t.gross_pnl == pytest.approx(160.0)
    # Short round trip: sold 5@130 = +650, covered 5@125 = -625.
    assert short_t.side == -1
    assert short_t.gross_pnl == pytest.approx(25.0)


def test_trades_ignores_open_position():
    ledger = EventLedger([_fill(0, "A", 1, 10.0, 100.0)])
    assert trades(ledger) == ()


def test_trades_commission_after_close_attaches_to_last_trade():
    ledger = EventLedger(
        [
            _fill(0, "A", 1, 10.0, 100.0),
            _fill(1, "A", -1, 10.0, 110.0),  # closes; position flat
            _commission(1, "A", 7.0),        # trailing commission on the closed trade
            _fill(2, "A", 1, 5.0, 105.0),    # opens a new trade
        ]
    )
    result = trades(ledger)
    assert len(result) == 1  # second trade is still open
    assert result[0].commission == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# positions view.
# ---------------------------------------------------------------------------


def test_positions_step_series():
    ledger = EventLedger(
        [
            _position(0, "A", 10.0),
            _position(1, "A", 10.0),  # same value, later timestamp
            _position(2, "A", -5.0),
            _position(3, "A", 0.0),
        ]
    )
    pos = positions(ledger)
    assert isinstance(pos, Positions)
    assert pos.timestamps.tolist() == [0, 2, 3]
    assert pos.position.tolist() == [10.0, -5.0, 0.0]


def test_positions_keeps_last_change_per_timestamp():
    ledger = EventLedger(
        [
            _position(0, "A", 5.0),
            _position(0, "A", 10.0),  # last one wins at the same boundary
        ]
    )
    pos = positions(ledger)
    assert pos.timestamps.tolist() == [0]
    assert pos.position.tolist() == [10.0]


def test_positions_requires_instrument_id_for_multi_instrument():
    ledger = EventLedger([_position(0, "A", 1.0), _position(1, "B", 2.0)])
    with pytest.raises(DataShapeError):
        positions(ledger)
    assert positions(ledger, instrument_id="A").position.tolist() == [1.0]
    assert positions(ledger, instrument_id="B").position.tolist() == [2.0]


# ---------------------------------------------------------------------------
# equity curve (combined union / mark-to-market / base currency).
# ---------------------------------------------------------------------------


def _equity_fixture():
    """A two-instrument USD portfolio with an uneven time-bar grid."""
    t0, t1, t2 = (
        pd.Timestamp("2024-01-01T10:00", tz="UTC"),
        pd.Timestamp("2024-01-01T11:00", tz="UTC"),
        pd.Timestamp("2024-01-01T12:00", tz="UTC"),
    )
    market_data = {
        "A": _time_bars([t0, t1, t2], [100.0, 110.0, 120.0]),
        "B": _time_bars([t0, t1], [50.0, 55.0]),
    }
    instruments = {
        "A": Instrument("A", asset_class="stocks", settlement_currency="USD"),
        "B": Instrument("B", asset_class="stocks", settlement_currency="USD"),
    }
    ledger = EventLedger(
        [
            _cash(t0, 1000.0),       # +1000 USD cash
            _position(t0, "A", 10.0),
            _position(t1, "B", 5.0),
        ]
    )
    return ledger, market_data, instruments


def test_equity_curve_combined_union_and_mark_to_market():
    ledger, market_data, instruments = _equity_fixture()
    curve = equity_curve(
        ledger, market_data, instruments, base_currency="USD"
    )
    assert isinstance(curve, EquityCurve)
    assert isinstance(curve.index, pd.DatetimeIndex)
    assert curve.index.tz is not None
    # Grid = union of A (3 bars) and B (2 bars) timestamps.
    assert len(curve.index) == 3
    # t0: cash 1000 + A(10 * 100) + B(flat) = 2000
    # t1: cash 1000 + A(10 * 110) + B(5 * 55) = 1000 + 1100 + 275 = 2375
    # t2: cash 1000 + A(10 * 120) + B(5 * 55 ffill) = 1000 + 1200 + 275 = 2475
    assert curve.equity.tolist() == pytest.approx([2000.0, 2375.0, 2475.0])


def test_equity_curve_by_instrument_sums_to_combined_minus_cash():
    ledger, market_data, instruments = _equity_fixture()
    by_instrument = equity_curve_by_instrument(
        ledger, market_data, instruments, base_currency="USD"
    )
    assert set(by_instrument) == {"A", "B"}
    # A: 10 * [100, 110, 120] = [1000, 1100, 1200]
    assert by_instrument["A"].equity.tolist() == pytest.approx([1000.0, 1100.0, 1200.0])
    # B: flat, then 5 * [55, 55] = [0, 275, 275]
    assert by_instrument["B"].equity.tolist() == pytest.approx([0.0, 275.0, 275.0])

    combined = equity_curve(ledger, market_data, instruments, base_currency="USD")
    cash = np.array([1000.0, 1000.0, 1000.0])
    expected = cash + by_instrument["A"].equity + by_instrument["B"].equity
    assert combined.equity.tolist() == pytest.approx(expected.tolist())


def test_equity_curve_cash_movement_converted_and_cumulated():
    t0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    t1 = pd.Timestamp("2024-01-01T11:00", tz="UTC")
    market_data = {"A": _time_bars([t0, t1], [100.0, 100.0])}
    instruments = {"A": Instrument("A", asset_class="stocks", settlement_currency="USD")}
    ledger = EventLedger(
        [_cash(t0, 500.0), _cash(t1, -200.0)]  # inflow then outflow
    )
    curve = equity_curve(ledger, market_data, instruments, base_currency="USD")
    assert curve.equity.tolist() == pytest.approx([500.0, 300.0])


# ---------------------------------------------------------------------------
# Multi-currency normalization (§4.6, §4.8).
# ---------------------------------------------------------------------------


def _fx_fixture():
    t0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    t1 = pd.Timestamp("2024-01-01T11:00", tz="UTC")
    market_data = {
        "USD": _time_bars([t0, t1], [100.0, 110.0]),
        "EUR": _time_bars([t0, t1], [200.0, 200.0]),
    }
    instruments = {
        "USD": Instrument("USD", asset_class="stocks", settlement_currency="USD"),
        "EUR": Instrument("EUR", asset_class="stocks", settlement_currency="EUR"),
    }
    return t0, t1, market_data, instruments


def test_equity_curve_converts_settlement_to_base_currency():
    t0, t1, market_data, instruments = _fx_fixture()
    # EURUSD = 1.1 USD per EUR, constant across the grid.
    fx = FXSeries(
        index=pd.DatetimeIndex([t0], tz="UTC").as_unit("ns").asi8, rate=np.array([1.1])
    )
    ledger = EventLedger([_position(t0, "USD", 10.0), _position(t0, "EUR", 5.0)])
    curve = equity_curve(
        ledger, market_data, instruments, base_currency="USD", fx_rates={"EURUSD": fx}
    )
    # t0: USD 10*100 = 1000, EUR 5*200 EUR * 1.1 = 1100 USD -> 2100.
    # t1: USD 10*110 = 1100, EUR 5*200 EUR * 1.1 = 1100 -> 2200.
    assert curve.equity.tolist() == pytest.approx([2100.0, 2200.0])


def test_equity_curve_missing_fx_rate_raises():
    t0, t1, market_data, instruments = _fx_fixture()
    ledger = EventLedger([_position(t0, "EUR", 5.0)])
    with pytest.raises(FXRateUnavailableError):
        equity_curve(ledger, market_data, instruments, base_currency="USD")


def test_equity_curve_missing_settlement_currency_raises():
    t0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    market_data = {"A": _time_bars([t0], [100.0])}
    instruments = {"A": Instrument("A", asset_class="stocks")}  # no settlement_currency
    ledger = EventLedger([_position(t0, "A", 1.0)])
    with pytest.raises(FXRateUnavailableError):
        equity_curve(ledger, market_data, instruments, base_currency="USD")


def test_equity_curve_empty_base_currency_raises_config():
    t0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    market_data = {"A": _time_bars([t0], [100.0])}
    instruments = {"A": Instrument("A", asset_class="stocks", settlement_currency="USD")}
    ledger = EventLedger([_position(t0, "A", 1.0)])
    with pytest.raises(ConfigError):
        equity_curve(ledger, market_data, instruments, base_currency="")


def test_equity_curve_position_without_market_data_raises():
    # A held position with no market_data cannot be marked to market (§4.6 step 2) —
    # it must surface as DataShapeError, not silently drop from the combined equity.
    t0 = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    market_data = {"A": _time_bars([t0], [100.0])}
    instruments = {
        "A": Instrument("A", asset_class="stocks", settlement_currency="USD"),
        "C": Instrument("C", asset_class="stocks", settlement_currency="USD"),
    }
    ledger = EventLedger([_position(t0, "A", 1.0), _position(t0, "C", 100.0)])
    with pytest.raises(DataShapeError):
        equity_curve(ledger, market_data, instruments, base_currency="USD")


def test_fx_series_from_series_and_validation():
    s = pd.Series(
        [1.0, 1.1],
        index=pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02"], tz="UTC"
        ),
    )
    fx = FXSeries.from_series(s)
    assert fx.rate.tolist() == [1.0, 1.1]
    with pytest.raises(DataShapeError):
        FXSeries(index=np.array([0, 0]), rate=np.array([1.0, 1.0]))  # not ascending


# ---------------------------------------------------------------------------
# Immutability: frozen containers copy arrays; views are read-only.
# ---------------------------------------------------------------------------


def test_positions_copies_input_arrays():
    ts = np.array([0, 1, 2], dtype=np.int64)
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    container = Positions(timestamps=ts, position=pos)
    ts[0] = 99
    pos[0] = 99.0
    assert container.timestamps[0] == 0
    assert container.position[0] == 1.0


def test_equity_curve_copies_input_and_is_read_only():
    eq = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    curve = EquityCurve(index=pd.RangeIndex(3), equity=eq)
    eq[0] = 99.0
    assert curve.equity[0] == 1.0
    with pytest.raises(ValueError):
        curve.equity[0] = 5.0


def test_derived_view_arrays_do_not_alias_ledger():
    ledger = EventLedger([_position(0, "A", 1.0), _position(1, "A", 2.0)])
    pos = positions(ledger)
    assert not pos.position.flags.writeable
    # The derived array is independent of any internal buffer: mutating the source
    # position array (rebuilt from the ledger) must not affect the view.
    assert pos.position.tolist() == [1.0, 2.0]

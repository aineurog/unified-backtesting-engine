"""Tests for the canonical 4-column signal format and its converters (§6.1, §6.2, §9.2)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ube.core.data import MarketData
from ube.core.errors import ConfigError, DataShapeError, InvalidSignalError
from ube.core.signals import SIGNAL_COLUMNS, Signals, from_callable, from_target


def _signals(**overrides: object) -> Signals:
    """A length-3 all-False signal frame with optional per-column overrides."""
    cols = {c: np.zeros(3, dtype=bool) for c in SIGNAL_COLUMNS}
    cols.update(overrides)
    return Signals(**cols)


def _assert_signals_equal(got: Signals, expected: Signals) -> None:
    for col in SIGNAL_COLUMNS:
        assert np.array_equal(getattr(got, col), getattr(expected, col)), col
    assert got.n_bars == expected.n_bars


def _bars(closes: list[float]) -> MarketData:
    """Bars with the given closes, for exercising ``from_callable``."""
    close = np.asarray(closes, dtype=float)
    n = close.shape[0]
    return MarketData(
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.ones(n),
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


# ---------------------------------------------------------------------------
# Canonical format — construction and shape/dtype validation (§6.1).
# ---------------------------------------------------------------------------


def test_signal_columns_constant():
    assert SIGNAL_COLUMNS == ("long_entry", "long_exit", "short_entry", "short_exit")


def test_from_dataframe_round_trips():
    df = pd.DataFrame(
        {
            "long_entry": [True, False, False],
            "long_exit": [False, True, False],
            "short_entry": [False, False, True],
            "short_exit": [False, False, False],
        }
    )
    sig = Signals.from_dataframe(df)
    assert sig.n_bars == 3
    pd.testing.assert_frame_equal(sig.to_dataframe(), df)


def test_from_dataframe_ignores_extra_columns():
    df = pd.DataFrame(
        {
            "long_entry": [True, False],
            "long_exit": [False, False],
            "short_entry": [False, False],
            "short_exit": [False, False],
            "note": ["a", "b"],
        }
    )
    assert Signals.from_dataframe(df).n_bars == 2


def test_from_dataframe_missing_column_raises():
    df = pd.DataFrame({"long_entry": [True, False]})
    with pytest.raises(InvalidSignalError):
        Signals.from_dataframe(df)


def test_from_dataframe_non_boolean_column_raises():
    df = pd.DataFrame(
        {
            "long_entry": [1, 0],
            "long_exit": [0, 0],
            "short_entry": [0, 0],
            "short_exit": [0, 0],
        }
    )
    with pytest.raises(InvalidSignalError):
        Signals.from_dataframe(df)


def test_non_boolean_column_raises():
    with pytest.raises(InvalidSignalError):
        Signals(long_entry=np.array([1, 0]), long_exit=np.zeros(2, bool),
                short_entry=np.zeros(2, bool), short_exit=np.zeros(2, bool))


def test_length_mismatch_raises():
    with pytest.raises(InvalidSignalError):
        Signals(long_entry=np.zeros(3, bool), long_exit=np.zeros(2, bool),
                short_entry=np.zeros(3, bool), short_exit=np.zeros(3, bool))


def test_from_array_wrong_shape_raises():
    with pytest.raises(InvalidSignalError):
        Signals.from_array(np.zeros((3, 3), dtype=bool))


def test_from_array_non_bool_dtype_raises():
    with pytest.raises(InvalidSignalError):
        Signals.from_array(np.zeros((3, 4), dtype=int))


def test_from_array_round_trips():
    arr = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ],
        dtype=bool,
    )
    sig = Signals.from_array(arr)
    assert sig.long_entry.tolist() == [True, False, False]
    assert sig.short_entry.tolist() == [False, False, True]


def test_signals_is_frozen():
    sig = _signals()
    with pytest.raises(FrozenInstanceError):
        sig.long_entry = np.zeros(3, bool)  # type: ignore[misc]


def test_arrays_are_read_only():
    sig = _signals()
    with pytest.raises(ValueError):
        sig.long_entry[0] = True


def test_constructor_does_not_alias_caller_buffer():
    # A frozen container must own its data: mutating the caller's buffer afterwards
    # must not change the container.
    src = np.zeros(3, dtype=bool)
    sig = Signals(
        long_entry=src,
        long_exit=np.zeros(3, dtype=bool),
        short_entry=np.zeros(3, dtype=bool),
        short_exit=np.zeros(3, dtype=bool),
    )
    src[0] = True
    assert not sig.long_entry[0]


def test_constructor_does_not_freeze_caller_array():
    # Construction must not silently make the caller's own array read-only.
    src = np.zeros(3, dtype=bool)
    Signals(
        long_entry=src,
        long_exit=np.zeros(3, dtype=bool),
        short_entry=np.zeros(3, dtype=bool),
        short_exit=np.zeros(3, dtype=bool),
    )
    src[0] = True  # still writable — no hidden side effect
    assert src[0]


def test_from_array_does_not_alias_input():
    src = np.zeros((3, 4), dtype=bool)
    sig = Signals.from_array(src)
    src[0, 0] = True
    assert not sig.long_entry[0]


# ---------------------------------------------------------------------------
# Conflict rule (§6.1) — the only two contradictory pairings, naming the bar.
# ---------------------------------------------------------------------------


def test_entry_conflict_names_the_bar():
    with pytest.raises(InvalidSignalError) as excinfo:
        _signals(
            long_entry=np.array([False, True, False]),
            short_entry=np.array([False, True, False]),
        )
    message = str(excinfo.value)
    assert "bar 1" in message
    assert "long_entry" in message and "short_entry" in message


def test_exit_conflict_names_the_bar():
    with pytest.raises(InvalidSignalError) as excinfo:
        _signals(
            long_exit=np.array([False, False, True]),
            short_exit=np.array([False, False, True]),
        )
    message = str(excinfo.value)
    assert "bar 2" in message
    assert "long_exit" in message and "short_exit" in message


def test_invalid_signal_error_is_a_config_error():
    assert issubclass(InvalidSignalError, ConfigError)


# ---------------------------------------------------------------------------
# Flip encodings are legal, not conflicts (§6.1 note, §6.2).
# ---------------------------------------------------------------------------


def test_long_exit_plus_short_entry_is_legal_flip():
    # Bar 1 flips long -> short (two simultaneous events), which is the correct
    # encoding and must not be flagged as a contradiction.
    sig = _signals(
        long_exit=np.array([False, True, False]),
        short_entry=np.array([False, True, False]),
    )
    assert sig.long_exit[1] and sig.short_entry[1]


def test_short_exit_plus_long_entry_is_legal_flip():
    # Bar 1 flips short -> long.
    sig = _signals(
        short_exit=np.array([False, True, False]),
        long_entry=np.array([False, True, False]),
    )
    assert sig.short_exit[1] and sig.long_entry[1]


# ---------------------------------------------------------------------------
# from_target — the §6.2 transition table (each row, both flips, no-change).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, long_entry, long_exit, short_entry, short_exit",
    [
        # 0 -> 1
        ([0, 1], [False, True], [False, False], [False, False], [False, False]),
        # 0 -> -1
        ([0, -1], [False, False], [False, False], [False, True], [False, False]),
        # 1 -> 0
        ([1, 0], [True, False], [False, True], [False, False], [False, False]),
        # -1 -> 0
        ([-1, 0], [False, False], [False, False], [True, False], [False, True]),
        # 1 -> -1  (flip: long_exit + short_entry)
        ([1, -1], [True, False], [False, True], [False, True], [False, False]),
        # -1 -> 1  (flip: short_exit + long_entry)
        ([-1, 1], [False, True], [False, False], [True, False], [False, True]),
        # no change: 1 -> 1
        ([1, 1], [True, False], [False, False], [False, False], [False, False]),
        # no change: -1 -> -1
        ([-1, -1], [False, False], [False, False], [True, False], [False, False]),
        # no change: 0 -> 0
        ([0, 0], [False, False], [False, False], [False, False], [False, False]),
    ],
)
def test_from_target_transition_table(
    target, long_entry, long_exit, short_entry, short_exit
):
    sig = from_target(np.array(target))
    assert sig.long_entry.tolist() == long_entry
    assert sig.long_exit.tolist() == long_exit
    assert sig.short_entry.tolist() == short_entry
    assert sig.short_exit.tolist() == short_exit


def test_from_target_hold_is_repeated_value():
    # [1, 0, 0] = long, then flat, then flat -> exit at bar 1 (§6.2).
    sig = from_target(np.array([1, 0, 0]))
    assert sig.long_entry.tolist() == [True, False, False]
    assert sig.long_exit.tolist() == [False, True, False]
    assert not sig.short_entry.any() and not sig.short_exit.any()

    # [1, 1, 1] = long, long, long -> stay long (a hold); only the initial entry.
    sig = from_target(np.array([1, 1, 1]))
    assert sig.long_entry.tolist() == [True, False, False]
    assert not sig.long_exit.any() and not sig.short_entry.any() and not sig.short_exit.any()


def test_from_target_full_sequence():
    target = np.array([0, 0, 1, 1, 0, -1, -1, 0])
    sig = from_target(target)
    assert sig.long_entry.tolist() == [False, False, True, False, False, False, False, False]
    assert sig.long_exit.tolist() == [False, False, False, False, True, False, False, False]
    assert sig.short_entry.tolist() == [False, False, False, False, False, True, False, False]
    assert sig.short_exit.tolist() == [False, False, False, False, False, False, False, True]


def test_from_target_empty():
    sig = from_target(np.array([], dtype=int))
    assert sig.n_bars == 0
    assert not sig.long_entry.any() and not sig.short_exit.any()


@pytest.mark.parametrize("bad", [2, 5, -2, 7])
def test_from_target_rejects_out_of_range_value(bad):
    with pytest.raises(InvalidSignalError):
        from_target(np.array([0, bad, 1]))


def test_from_target_rejects_nan():
    with pytest.raises(InvalidSignalError):
        from_target(np.array([0.0, np.nan]))


def test_from_target_rejects_non_numeric():
    with pytest.raises(InvalidSignalError):
        from_target(np.array(["long", "flat"]))


def test_from_target_rejects_2d():
    with pytest.raises(InvalidSignalError):
        from_target(np.array([[0, 1], [1, 0]]))


def test_from_target_accepts_list_and_float_integral_values():
    # A python list of ints, and a float array whose values are exactly -1/0/1.
    _assert_signals_equal(from_target([0, 1, 0]), from_target(np.array([0.0, 1.0, 0.0])))


# ---------------------------------------------------------------------------
# from_callable (§9.2) — per-bar window, target collection, from_target path.
# ---------------------------------------------------------------------------


def test_from_callable_agrees_with_from_target():
    closes = [10.0, 11.0, 9.0, 12.0, 8.0, 10.0]
    threshold = 10.0
    bars = _bars(closes)

    def strat(window: MarketData) -> int:
        last = float(window.close[-1])
        if last > threshold:
            return 1
        if last < threshold:
            return -1
        return 0

    expected_targets = [1 if c > threshold else (-1 if c < threshold else 0) for c in closes]
    _assert_signals_equal(from_callable(strat, bars), from_target(expected_targets))


def test_from_callable_receives_growing_window_once_per_bar():
    seen: list[int] = []
    bars = _bars([10.0, 11.0, 12.0, 13.0])

    def strat(window: MarketData) -> int:
        seen.append(window.n_bars)
        return 0

    sig = from_callable(strat, bars)
    # Called once per bar, each with an ever-growing window (1..n bars).
    assert seen == [1, 2, 3, 4]
    assert sig.n_bars == 4
    assert not sig.long_entry.any() and not sig.short_entry.any()


def test_from_callable_round_trips_a_flip():
    # Closes alternate above/below the threshold, producing a long -> short flip then
    # a short -> long flip. Both are legal two-event encodings (§6.2) and must emerge
    # end-to-end without being flagged as contradictions.
    closes = [11.0, 9.0, 11.0]
    bars = _bars(closes)

    def strat(window: MarketData) -> int:
        return 1 if window.close[-1] > 10.0 else (-1 if window.close[-1] < 10.0 else 0)

    sig = from_callable(strat, bars)
    assert sig.long_entry.tolist() == [True, False, True]
    assert sig.long_exit.tolist() == [False, True, False]
    assert sig.short_entry.tolist() == [False, True, False]
    assert sig.short_exit.tolist() == [False, False, True]


def test_from_callable_invalid_return_raises():
    bars = _bars([10.0, 11.0])

    def bad_strat(window: MarketData) -> int:
        return 2  # type: ignore[return-value]

    with pytest.raises(InvalidSignalError):
        from_callable(bad_strat, bars)


def test_from_callable_empty_bars():
    # Empty input is rejected up-front by MarketData (fail loudly).
    with pytest.raises(DataShapeError):
        from_callable(lambda window: 0, _bars([]))


def test_from_callable_requires_market_data():
    with pytest.raises(InvalidSignalError):
        from_callable(lambda window: 0, pd.DataFrame({"close": [1.0, 2.0]}))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MarketData.head — the zero-copy window used by from_callable.
# ---------------------------------------------------------------------------


def test_market_data_head_returns_zero_copy_prefix():
    bars = _bars([10.0, 11.0, 12.0, 13.0])
    window = bars.head(2)
    assert window.n_bars == 2
    assert window.close.tolist() == [10.0, 11.0]
    # Zero-copy: the view shares storage with the source.
    assert np.shares_memory(window.close, bars.close)


def test_market_data_head_out_of_range_raises():
    bars = _bars([10.0, 11.0])
    from ube.core.errors import DataShapeError

    with pytest.raises(DataShapeError):
        bars.head(-1)
    with pytest.raises(DataShapeError):
        bars.head(3)

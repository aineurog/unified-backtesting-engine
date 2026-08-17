"""Canonical data + signals → Nautilus bars and per-bar lookup (§4.4).

Two pure translations, one direction (canonical in → Nautilus out):

- :func:`to_nautilus_bars` — canonical :class:`~ube.core.data.MarketData` columns
  become native Nautilus :class:`~nautilus_trader.model.data.Bar` objects. Every bar
  carries ``ts_event = ts_init`` (canonical int-ns UTC); prices are formatted to the
  instrument's ``price_precision`` and volumes to its ``size_precision``.
- :func:`to_signal_map` — the canonical 4-column :class:`~ube.core.signals.Signals`
  (row-aligned with the ``MarketData`` bars, §6.1) becomes a per-bar lookup dict
  keyed by the bar's ``int(ts_ns)`` (mirrors the reference ``signal_map`` approach).

The ``BarType`` is derived from the **median positive inter-bar spacing** (requirements
§4.9) — never a hard-coded minute label — so hourly/volume/event-driven bars feed
through with the resolution the data actually carries (plan §5.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from ube.adapters.nautilus_adapter.instrument_map import NautilusInstrumentBuild
from ube.core.data import MarketData
from ube.core.errors import DataShapeError, InvalidSignalError
from ube.core.signals import Signals

__all__ = [
    "build_bar_type",
    "derive_bar_period_ns",
    "to_nautilus_bars",
    "to_signal_map",
]

#: Resolvable time aggregations, largest first. ``MONTH``/``YEAR`` are excluded
#: because they are not a fixed number of nanoseconds (and are not meaningful for
#: a single backtest bar series).
_AGG_BY_NS: Sequence[tuple[BarAggregation, int]] = (
    (BarAggregation.WEEK, 604_800_000_000_000),
    (BarAggregation.DAY, 86_400_000_000_000),
    (BarAggregation.HOUR, 3_600_000_000_000),
    (BarAggregation.MINUTE, 60_000_000_000),
    (BarAggregation.SECOND, 1_000_000_000),
    (BarAggregation.MILLISECOND, 1_000_000),
)


def derive_bar_period_ns(market_data: MarketData) -> int:
    """Median positive inter-bar spacing in nanoseconds (§4.9, plan §5.1).

    ``MarketData`` guarantees a sorted, unique, tz-aware UTC index, so every
    consecutive delta is strictly positive.

    Raises:
        DataShapeError: fewer than two bars (no spacing to infer a period from).
    """
    if market_data.n_bars < 2:
        raise DataShapeError(
            f"cannot derive a bar period from {market_data.n_bars} bar(s); "
            "require at least two bars"
        )
    deltas = np.diff(
        market_data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
    )
    return int(np.median(deltas[deltas > 0]))


def _is_valid_step(step: int, aggregation: BarAggregation) -> bool:
    """Nautilus time-step rules (``data.pyx`` ``BarSpecification``).

    Sub-day steps must evenly divide their parent unit (and not *be* it — use the
    higher aggregation instead); ``DAY``/``WEEK`` accept only ``step == 1``.
    """
    if aggregation is BarAggregation.MILLISECOND:
        return 1000 % step == 0 and step != 1000
    if aggregation in (BarAggregation.SECOND, BarAggregation.MINUTE):
        return 60 % step == 0 and step != 60
    if aggregation is BarAggregation.HOUR:
        return 24 % step == 0 and step != 24
    if aggregation in (BarAggregation.DAY, BarAggregation.WEEK):
        return step == 1
    return False


def _aggregation_for(period_ns: int) -> tuple[int, BarAggregation]:
    """Map a period in nanoseconds to a ``(step, aggregation)`` pair.

    Prefers the largest aggregation that expresses the period *exactly* with a valid
    step — a 3 600 s period → ``1-HOUR``, not ``3600-SECOND``. When no aggregation does
    (irregular/event-driven bars have no stable time period, §4.9), falls back to the
    deterministic ``1-SECOND`` routing label: external bars carry their own timestamps,
    and annualization is inferred by core from elapsed time — never from this label.
    """
    for aggregation, unit_ns in _AGG_BY_NS:
        if period_ns % unit_ns == 0 and _is_valid_step(period_ns // unit_ns, aggregation):
            return period_ns // unit_ns, aggregation
    return 1, BarAggregation.SECOND


def build_bar_type(instrument_id: InstrumentId, period_ns: int) -> BarType:
    """Build the ``LAST/EXTERNAL`` Nautilus ``BarType`` for a period (§4.4)."""
    step, aggregation = _aggregation_for(period_ns)
    return BarType(instrument_id, BarSpecification(step, aggregation, PriceType.LAST))


def to_nautilus_bars(
    market_data: MarketData,
    build: NautilusInstrumentBuild,
) -> tuple[Sequence[Bar], BarType]:
    """Convert canonical bars to native Nautilus :class:`Bar` objects (§4.4).

    Prices are formatted to the instrument's ``price_precision`` and volumes to its
    ``size_precision`` (integer quantities when the size precision is 0, e.g. ES/GC
    futures). ``ts_event = ts_init`` is the canonical timestamp's int-ns UTC value.

    Returns:
        The bars (chronological order — ``MarketData`` is already sorted) plus the
        ``BarType`` they were built with (the actor subscribes to the same type).
    """
    bar_type = build_bar_type(build.instrument_id, derive_bar_period_ns(market_data))
    price_precision = int(build.instrument.price_precision)
    size_precision = int(build.instrument.size_precision)
    ts_ns = market_data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]

    bars: list[Bar] = []
    for i in range(market_data.n_bars):
        # `MarketData` fills NaN where volume was not provided (§5.1); Nautilus requires
        # a scalar volume, so NaN is represented as 0.0.
        volume_value = float(market_data.volume[i])
        if math.isnan(volume_value):
            volume_value = 0.0
        volume: Quantity
        if size_precision > 0:
            volume = Quantity.from_str(f"{volume_value:.{size_precision}f}")
        else:
            volume = Quantity.from_str(str(int(round(volume_value))))
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{market_data.open[i]:.{price_precision}f}"),
                high=Price.from_str(f"{market_data.high[i]:.{price_precision}f}"),
                low=Price.from_str(f"{market_data.low[i]:.{price_precision}f}"),
                close=Price.from_str(f"{market_data.close[i]:.{price_precision}f}"),
                volume=volume,
                ts_event=int(ts_ns[i]),
                ts_init=int(ts_ns[i]),
            )
        )
    return bars, bar_type


def to_signal_map(
    market_data: MarketData,
    signals: Signals,
) -> dict[int, tuple[bool, bool, bool, bool]]:
    """Build the per-bar signal lookup: ``int(ts_ns) -> (le, lx, se, sx)``.

    Signals are row-aligned with the canonical bars (§6.1), so bar ``i`` is keyed
    by the ``int`` nanosecond timestamp of bar ``i`` of ``market_data``.

    Raises:
        InvalidSignalError: signal rows do not match the bar count.
    """
    if signals.n_bars != market_data.n_bars:
        raise InvalidSignalError(
            f"signals cover {signals.n_bars} bars but market data has "
            f"{market_data.n_bars}; signal rows must be row-aligned with bars"
        )
    ts_ns = market_data.timestamps.as_unit("ns").asi8  # type: ignore[attr-defined]
    return {
        int(ts_ns[i]): (
            bool(signals.long_entry[i]),
            bool(signals.long_exit[i]),
            bool(signals.short_entry[i]),
            bool(signals.short_exit[i]),
        )
        for i in range(market_data.n_bars)
    }
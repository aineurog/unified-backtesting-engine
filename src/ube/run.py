"""The single public entry point: :func:`ube.run` (§7, §5.2).

``ube.run`` is the one call a user makes. It resolves the configured engine
(``config.engine``; ``"auto"`` -> the first registered adapter, lazily
registering the bundled Nautilus adapter when available), forwards the canonical
inputs — ``data``, ``signals``, ``config`` and the optional ``aux_data``
derived-series map (§5.2) — to the adapter, and returns the canonical
:class:`~ube.core.result.BacktestResult`.

Single-asset form::

    result = ube.run(bars, signals, config, aux_data={"atr_12h": atr_series})

``aux_data`` is a flat ``name -> value`` mapping referenced by name from exit
configs (e.g. ``ATRStop(atr="atr_12h")``). A value may be a precomputed 1-D array
(length must match ``data``'s bar count), or a :class:`~ube.core.data.MarketData` of
the *signal-timeframe OHLCV* — in which case the adapter computes the ATR internally
(with no look-ahead bias) and aligns it to ``data``'s bar grid. When a referenced name
is absent the library computes the series from ``data``. Portfolio (dict-keyed) inputs
are accepted by the signature but delegate entirely to the adapter, which decides
support.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from ube.adapters import EngineAdapter, get_engine, register_engine
from ube.core.config import BacktestConfig
from ube.core.data import MarketData
from ube.core.errors import ConfigError
from ube.core.result import BacktestResult
from ube.core.signals import Signals

__all__ = ["run"]


def _ensure_engine(name: str) -> type[EngineAdapter]:
    """Resolve an adapter class for ``name``, lazily registering the bundled
    Nautilus adapter when ``name`` is ``"auto"``/``"nautilus"`` and nothing is
    registered yet (§7.1)."""
    try:
        return get_engine(name)
    except ConfigError:
        pass
    if name in ("auto", "nautilus"):
        adapter_cls: type[EngineAdapter] | None = None
        try:
            from ube.adapters.nautilus_adapter import NautilusAdapter

            adapter_cls = NautilusAdapter
        except ImportError:  # pragma: no cover - nautilus_trader missing
            adapter_cls = None
        if adapter_cls is not None:
            register_engine("nautilus", adapter_cls)
            return get_engine(name)
    raise ConfigError(
        f"engine {name!r} is not available; register an adapter via "
        "ube.register_engine(...) or install ube[nautilus]"
    )


def run(
    data: MarketData | Mapping[str, MarketData],
    signals: Signals | Mapping[str, Signals],
    config: BacktestConfig,
    *,
    aux_data: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Run one backtest and return the canonical result (§7, §5.2).

    Args:
        data: Single-instrument OHLCV bars (a :class:`~ube.core.data.MarketData`),
            or a ``{instrument_id: MarketData}`` mapping for a portfolio run.
        signals: The 4-column entry/exit signals aligned to ``data``'s bar grid (a
            :class:`~ube.core.signals.Signals`, or a per-instrument mapping).
        config: The full :class:`~ube.core.config.BacktestConfig` (instrument,
            cost, risk/exits, benchmark, engine selection, overrides).
        aux_data: Optional derived-series map referenced by name from exit configs
            (§5.2) — e.g. ``{"atr_12h": <array or MarketData>}`` for
            ``ATRStop(atr="atr_12h")``. A value may be a precomputed 1-D array (length
            must match ``data``'s bar count) or a :class:`~ube.core.data.MarketData` of
            the signal-timeframe OHLCV, which the adapter turns into ATR internally with
            no look-ahead bias. When a referenced name is absent the library computes
            the series from ``data``. A flat ``name -> value`` map for single-asset
            runs, or a ``{instrument_id: {name: value}}`` map for portfolio runs.

    Returns:
        The canonical :class:`~ube.core.result.BacktestResult` (``trades``,
        ``positions``, ``equity_curve``, ``ledger``, ``trade_table``, ``metrics``).

    Raises:
        ConfigError: ``config`` is not a :class:`~ube.core.config.BacktestConfig`,
            or the configured engine cannot be resolved.
    """
    if not isinstance(config, BacktestConfig):
        raise ConfigError(
            "ube.run expects a BacktestConfig; got "
            f"{type(config).__name__}"
        )
    # The minimal ``ube.Instrument(symbol, asset_class=...)`` usage declares no
    # settlement currency; the downstream ledger normalization (core) requires one.
    # Default it per asset class here (the canonical entry point) so the simple
    # construction path runs without forcing a settlement currency on the caller.
    config = _default_settlement_currency(config)
    adapter = _ensure_engine(config.engine)
    # Portfolio (dict-keyed) inputs are forwarded as-is; the adapter decides support.
    return adapter().run(
        cast(MarketData, data),
        cast(Signals, signals),
        config,
        aux_data=aux_data,
    )


#: Per-asset-class default settlement currency, applied when an instrument declares
#: none (mirrors :data:`ube.core.result._DEFAULT_SETTLEMENT`).
_DEFAULT_SETTLEMENT: dict[str, str] = {
    "crypto_spot": "USDT",
    "crypto_perp": "USDT",
    "forex": "USD",
    "futures": "USD",
    "stocks": "USD",
    "commodities": "USD",
}


def _default_settlement_currency(config: BacktestConfig) -> BacktestConfig:
    """Return a config whose instrument has a settlement currency when omitted."""
    instrument = config.instrument
    if instrument.settlement_currency is not None:
        return config
    settled = _DEFAULT_SETTLEMENT.get(instrument.asset_class, "USD")
    return replace(
        config,
        instrument=replace(instrument, settlement_currency=settled),
    )

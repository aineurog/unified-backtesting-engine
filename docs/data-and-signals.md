# Data and Signals

## Market Data

`ube.MarketData` is the canonical single-instrument bar container. It has aligned
`open`, `high`, `low`, `close`, `volume`, and timestamp/index values. Timestamps
must be timezone-aware and strictly increasing; price arrays must satisfy the
data validation rules. The library standardizes in-memory input; it does not
fetch, resample, or align external provider data for you.

DataFrames can be standardized at the public run boundary with `ube.run()` or
constructed explicitly with `MarketData.from_dataframe()`:

```python
market_data = ube.MarketData.from_dataframe(
    frame,
    timestamp_col="timestamp",
)
```

The DataFrame must contain canonical OHLC columns and a timestamp column or
compatible index. See `MarketData.standardize()` and its constructors in
`src/ube/core/data.py` for accepted array, mapping, and record forms.

## Signal Formats

The canonical `Signals` fields are boolean arrays with one element per bar:

| Field | Action |
|---|---|
| `long_entry` | Open long |
| `long_exit` | Close long |
| `short_entry` | Open short |
| `short_exit` | Close short |

`Signals.from_dataframe()` accepts those four exact column names. Contradictory
long and short entries, or contradictory long and short exits, are rejected.
A flip is encoded as `long_exit + short_entry` or `short_exit + long_entry` on
the same bar.

`ube.from_target()` converts a one-dimensional `-1/0/1` target array to actions.
In this representation `0` means flat, so a transition from `1` to `0` emits a
long exit. `ube.from_callable()` evaluates a callable over growing data prefixes
and converts its `-1/0/1` results in the same way.

## Alignment

Signals and bars are positional and must have equal lengths. UBE does not infer
timestamp alignment from a separate signal DataFrame: align rows to bars before
constructing `Signals`. For derived data, use the named `aux_data` mapping
accepted by `ube.run()`; adapter-specific requirements are described on the
[engine pages](engines/overview.md).
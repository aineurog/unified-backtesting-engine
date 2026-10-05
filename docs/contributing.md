# Contributing

UBE depends on a small, stable core contract. Keep cross-engine behavior in
`src/ube/core/` and engine translation in the owning adapter package. Do not let
core modules import optional engine packages.

## Changes and Tests

- Add unit tests for core validation, sizing, exits, and ledger behavior.
- Add adapter integration tests for native translation and failure handling.
- When changing shared semantics, run the parity tests and explain any deliberate
  engine-specific difference.
- Keep optional engine imports lazy so importing `ube` does not require every
  backend.
- Add a `CHANGELOG.md` entry under `Unreleased` for user-visible changes.
- Run `pytest` and relevant lint/type checks before opening a pull request.

## Adding an Adapter

Implement `EngineAdapter.run()` for canonical `MarketData`, `Signals`, and
`BacktestConfig`; return `BacktestResult` built from the canonical event ledger.
Validate engine-specific overrides locally. Add the optional dependency as an
extra rather than a required core dependency, register the adapter only when
its engine can be imported, and include an integration test plus parity coverage
for semantics the adapter claims to share.

Start with the existing [API map](api-reference.md), adapter implementations in
`src/ube/adapters/`, and [testing guidance](testing.md).
"""Thin CLI for paper trading — `ube paper step` / `ube paper daemon` (§20, T9).

A thin-but-real entry point: loads a YAML ``PaperConfig`` (or bare ``BacktestConfig``),
bootstraps the engine-agnostic paper session (``init``), prints a one-line summary, and
exits non-zero on any error. It deliberately adds no trading logic and no new flags
beyond the ``--state``/``--run-id`` already stubbed. Daemon mode remains a stub (the plan
defers it to a "later task"; synchronous ``run_auto`` in ``core.py`` is production Mode B).

The YAML shape mirrors what :func:`ube.papertrading.core.init` serializes
(``yaml.safe_dump(asdict(config.base))``), with a few conveniences:
  * ``symbol`` / ``asset_class`` are the minimal instrument fields (the rest default).
  * nested configs (``risk``, ``signal``, ``benchmark``, ``cost_model``) are rebuilt from
    their dataclass dicts.
  * exits **must** carry a ``_type`` discriminator for ``StopLoss`` vs ``TrailingStop``
    (both are ``percent``-only and indistinguishable without it); other exits may omit
    ``_type`` and are inferred from keys. Supported ``_type`` values: ``TakeProfit``,
    ``StopLoss``, ``ATRStop``, ``TrailingStop``, ``TimeExit``, ``ChandelierExit``.
    Without ``_type`` for the ambiguous pair the loader raises ``ConfigError``.
  * an optional ``paper`` mapping on the top level carries ``PaperConfig``-only fields
    (``state_path``, ``starting_balance``, ``engine``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ube.core.benchmark import BenchmarkConfig
from ube.core.config import BacktestConfig, SignalConfig
from ube.core.cost import CostModel
from ube.core.errors import ConfigError
from ube.core.instrument import Instrument
from ube.core.risk import RiskConfig
from ube.core.risk.exits import (
    ATRStop,
    ChandelierExit,
    Exit,
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
)
from ube.core.risk.sizing import SizeModel
from ube.papertrading.config import PaperConfig


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _rebuild_exit(data: dict[str, Any]) -> Exit:
    """Rebuild one exit config, using a ``_type`` discriminator if present (§20)."""
    data = dict(data)
    etype = data.pop("_type", None)
    if etype is not None:
        cls: type[Exit] | None = {
            "TakeProfit": TakeProfit,
            "StopLoss": StopLoss,
            "ATRStop": ATRStop,
            "TrailingStop": TrailingStop,
            "TimeExit": TimeExit,
            "ChandelierExit": ChandelierExit,
        }.get(str(etype))
        if cls is None:
            raise ConfigError(f"unknown exit _type={etype!r}")
        exit_obj = cls(**data)
        assert isinstance(exit_obj, Exit)
        return exit_obj
    # Bare asdict form (no discriminator). Disambiguate by fields; StopLoss vs TrailingStop
    # are indistinguishable from keys, so require _type for those.
    if "bars" in data:
        return TimeExit(**data)
    if "scale_out" in data:
        return TakeProfit(**data)
    if "mult" in data:
        if "trailing" in data:
            return ATRStop(**data)
        return ChandelierExit(**data)
    if "percent" in data:
        raise ConfigError(
            "cannot disambiguate StopLoss vs TrailingStop from a bare asdict YAML; "
            "add `_type: StopLoss` or `_type: TrailingStop`"
        )
    raise ConfigError(f"cannot identify exit from keys {sorted(data)}")


def _rebuild_risk(data: Any) -> RiskConfig:
    data = _require_mapping(data, "risk")
    sizing: SizeModel = SizeModel()
    if data.get("sizing"):
        sizing = SizeModel(**dict(data["sizing"]))
    exits: list[Exit] = []
    raw_exit = data.get("exit", ())
    if isinstance(raw_exit, list):
        exits = [
            _rebuild_exit(_require_mapping(x, f"exit[{i}]")) for i, x in enumerate(raw_exit)
        ]
    elif isinstance(raw_exit, dict):
        exits = [_rebuild_exit(raw_exit)]
    elif raw_exit:  # pragma: no cover - defensive
        raise ConfigError(
            f"risk.exit must be a dict or list of dicts, got {type(raw_exit).__name__}"
        )
    return RiskConfig(sizing=sizing, exit=tuple(exits))


def _rebuild_signal(data: Any) -> SignalConfig:
    data = _require_mapping(data, "signal")
    return SignalConfig(**data)


def _rebuild_benchmark(data: Any) -> BenchmarkConfig:
    data = _require_mapping(data, "benchmark")
    weights = data.get("weights")
    if weights is not None and not isinstance(weights, tuple):
        data["weights"] = tuple(weights)
    return BenchmarkConfig(**data)


def _rebuild_cost(data: Any) -> CostModel | None:
    if data is None:
        return None
    return CostModel(**dict(_require_mapping(data, "cost_model")))


def _rebuild_instrument(data: Any) -> Instrument:
    data = _require_mapping(data, "instrument")
    symbol = data.get("symbol")
    asset_class = data.get("asset_class")
    if not symbol or not asset_class:
        raise ConfigError("instrument requires 'symbol' and 'asset_class'")
    return Instrument(**dict(data))


def _load_base(data: dict[str, Any]) -> BacktestConfig:
    """Rebuild a ``BacktestConfig`` from its ``asdict``-style YAML mapping."""
    # Derive allowed top-level keys from the dataclass (fix 9: no hard-coded allowlist).
    allowed = set(BacktestConfig.__dataclass_fields__.keys()) | {"paper"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unrecognized config key(s): {sorted(unknown)}")
    instrument = _rebuild_instrument(data.get("instrument"))
    kwargs: dict[str, Any] = {"instrument": instrument}
    if data.get("cost_model") is not None:
        kwargs["cost_model"] = _rebuild_cost(data["cost_model"])
    if data.get("risk"):
        kwargs["risk"] = _rebuild_risk(data["risk"])
    if data.get("signal"):
        kwargs["signal"] = _rebuild_signal(data["signal"])
    if data.get("benchmark") and data["benchmark"] is not None:
        kwargs["benchmark"] = _rebuild_benchmark(data["benchmark"])
    if data.get("engine") is not None:
        kwargs["engine"] = data["engine"]
    if data.get("engine_overrides") is not None:
        kwargs["engine_overrides"] = dict(data["engine_overrides"])
    if data.get("date_range") is not None:
        kwargs["date_range"] = tuple(data["date_range"])
    if data.get("base_currency") is not None:
        kwargs["base_currency"] = data["base_currency"]
    if data.get("warmup_bars") is not None:
        kwargs["warmup_bars"] = data["warmup_bars"]
    return BacktestConfig(**kwargs)


def _load_config(path: str) -> PaperConfig:
    """Parse a YAML config file and return a :class:`PaperConfig` (§20)."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    data = _require_mapping(data, "config")

    paper = data.get("paper") or {}
    paper = _require_mapping(paper, "paper")
    base = _load_base(data)

    kwargs: dict[str, Any] = {"base": base}
    if paper.get("state_path") is not None:
        kwargs["state_path"] = paper["state_path"]
    if paper.get("starting_balance") is not None:
        kwargs["starting_balance"] = float(paper["starting_balance"])
    if paper.get("engine") is not None:
        kwargs["engine"] = paper["engine"]
    return PaperConfig(**kwargs)


def cmd_step(args: argparse.Namespace) -> int:
    """Load the config, bootstrap the session, and print a one-line summary (§20)."""
    from ube.papertrading.core import init
    from ube.papertrading.state import PaperState

    try:
        config = _load_config(args.config)
        config.base.validate(paper_trading=True)
        run_id = args.run_id
        state_path = args.state

        # Resume if the state already exists, otherwise create a fresh session.
        try:
            state = PaperState.load(state_path, run_id=run_id)
            resuming = True
        except Exception:
            state = init(config, run_id=run_id)
            resuming = False
            state.save(state_path, run_id=run_id)

        instr = config.base.instrument
        print(
            f"[ube paper step] ok instrument={instr.symbol} "
            f"asset_class={instr.asset_class} engine={config.engine} "
            f"state={state_path} run_id={run_id} "
            f"session={'resumed' if resuming else 'initialized'} "
            f"bars_processed={state.last_processed_ns is not None} "
            f"open_position={'open' if state.open_position is not None else 'flat'}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI surface; report and fail loudly.
        print(f"[ube paper step] error: {exc}")
        return 1


def cmd_daemon(args: argparse.Namespace) -> int:
    print(f"[ube paper daemon] config={args.config} state={args.state} run_id={args.run_id}")
    print("Daemon mode is a thin T9 stub — synchronous `run_auto` is the production Mode B.")
    print("See `src/ube/papertrading/core.py:run_auto` for the streaming API.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ube paper", description="Paper trading CLI (T9)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("step", help="Run one paper slice (Mode A)")
    s.add_argument("config", help="Path to BacktestConfig YAML (or Python import)")
    s.add_argument("--state", default="paper.db", help="sqlite state path")
    s.add_argument("--run-id", default="default", help="run_id for multi-tenant DB")
    s.set_defaults(func=cmd_step)

    d = sub.add_parser("daemon", help="Run paper daemon (Mode B, stub)")
    d.add_argument("config", help="Path to BacktestConfig YAML")
    d.add_argument("--state", default="paper.db", help="sqlite state path")
    d.add_argument("--run-id", default="default", help="run_id")
    d.set_defaults(func=cmd_daemon)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

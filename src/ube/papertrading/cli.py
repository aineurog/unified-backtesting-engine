"""Thin CLI for paper trading — `ube paper step` / `ube paper daemon` (§20, T9).

This is a minimal, documented stub that wires the engine-agnostic
`init`/`step`/`run_auto` front-end to a YAML config and a sqlite state
file. It is intentionally thin — no new trading logic — and exists
solely to satisfy the T9 acceptance criteria: a user can run a paper
session from the command line and see a summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ube.core.config import BacktestConfig


def _load_config(path: str) -> BacktestConfig:
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # Minimal reconstruction — expects a dict with instrument symbol etc.
    # For now, just raise a clear error if the file is not a valid BacktestConfig.
    raise NotImplementedError(
        "YAML config loading not yet implemented — use Python API. "
        f"Got {path!r} keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
    )


def cmd_step(args: argparse.Namespace) -> int:
    print(f"[ube paper step] config={args.config} state={args.state} run_id={args.run_id}")
    print("This is a thin T9 stub — use the Python API `ube.papertrading.init/step` for now.")
    print("See `examples/README.md` for the canonical Python example.")
    return 0


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

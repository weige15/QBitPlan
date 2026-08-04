"""Scientific CLI entry point for the issue-25 real smoke path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .execution import execute_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qbitplan")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1")
    stage1_commands = stage1.add_subparsers(dest="stage1_command", required=True)
    run = stage1_commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--mode", choices=("smoke",), required=True)
    run.add_argument("--gpu-uuid", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with args.plan.open("r", encoding="utf-8") as handle:
            plan: dict[str, Any] = json.load(handle)
        if not isinstance(plan, dict):
            raise ValueError("experiment plan must be a JSON object")
        if plan.get("mode") != args.mode:
            raise ValueError("CLI mode does not match the explicit plan mode")
        if "gpu_uuid" in plan and plan["gpu_uuid"] != args.gpu_uuid:
            raise ValueError("CLI GPU UUID does not match the explicit plan GPU UUID")
        plan["gpu_uuid"] = args.gpu_uuid
        bundle = execute_plan(plan)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"qbitplan: rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"bundle_path": str(bundle.path), "bundle_id": bundle.bundle_id, "run_id": bundle.run_id}))
    return 0

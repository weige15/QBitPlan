"""Scientific CLI entry point for the Stage-1 execution seam."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from .stage1 import ExperimentPlan, execute_plan
from .stage1.executor import RuntimeDependencyError, TorchAOProfileExecutor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qbitplan")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1")
    stage1_commands = stage1.add_subparsers(dest="stage1_command", required=True)
    run = stage1_commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--mode", required=True, choices=("smoke",))
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--mode", choices=("smoke", "functional-quality"), required=True)
    run.add_argument("--gpu-uuid", required=True)
    estimate = stage1_commands.add_parser("estimate-cost")
    estimate.add_argument("--plan", type=Path, required=True)
    estimate.add_argument("--lookup", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "stage1" or args.stage1_command != "run":
        raise AssertionError("unreachable command parser state")
    try:
        plan = ExperimentPlan.from_json_file(args.plan)
        if plan.mode != args.mode:
            raise ValueError("CLI mode does not match the plan mode")
        if plan.data["hardware"]["gpu_uuid"] != args.gpu_uuid:
            raise ValueError("CLI GPU UUID does not match the plan GPU UUID")
        executor = TorchAOProfileExecutor(plan)
        bundle = execute_plan(plan, executor)
    except (FileExistsError, OSError, RuntimeDependencyError, ValueError, RuntimeError) as exc:
        print(f"qbitplan: run rejected: {exc}", file=sys.stderr)
        with args.plan.open("r", encoding="utf-8") as handle:
            plan: dict[str, Any] = json.load(handle)
        if not isinstance(plan, dict):
            raise ValueError("experiment plan must be a JSON object")
        if args.stage1_command == "estimate-cost":
            if plan.get("mode") != "estimate-cost":
                raise ValueError("estimate-cost plan must declare mode estimate-cost")
            from .stage1.lookup import LookupCostEstimateAdapter
            bundle = execute_plan(
                plan,
                executor=LookupCostEstimateAdapter.from_path(args.lookup),
            )
        else:
            if plan.get("mode") != args.mode:
                raise ValueError("CLI mode does not match the explicit plan mode")
            if "gpu_uuid" in plan and plan["gpu_uuid"] != args.gpu_uuid:
                raise ValueError("CLI GPU UUID does not match the explicit plan GPU UUID")
            plan["gpu_uuid"] = args.gpu_uuid
            bundle = execute_plan(plan)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"qbitplan: rejected: {exc}", file=sys.stderr)

        return 2

    print(
        json.dumps(
            {
                "artifact_id": bundle.artifact_id,
                "artifact_root": str(bundle.root),
                "manifest_id": bundle.manifest_id,
                "run_id": bundle.run_id,
                "claim_boundary": bundle.payload["claim_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

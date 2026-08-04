"""Scientific CLI entry point for the Stage-1 execution seam."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .execution import ProfileExecutor, execute_plan
from .plan import ExperimentPlan
from .stage1.direct_cost import DirectCostRunner
from .stage1.executor import TorchAOProfileExecutor
from .stage1.lookup import LookupCostEstimateAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qbitplan")
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1")
    stage1_commands = stage1.add_subparsers(dest="stage1_command", required=True)
    run = stage1_commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument(
        "--mode", choices=("smoke", "functional-quality", "direct-cost"), required=True
    )
    run.add_argument("--gpu-uuid", required=True)
    estimate = stage1_commands.add_parser("estimate-cost")
    estimate.add_argument("--plan", type=Path, required=True)
    estimate.add_argument("--lookup", type=Path, required=True)
    return parser


def _load_plan(path: Path) -> ExperimentPlan:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("experiment plan must be a JSON object")
    return ExperimentPlan.from_mapping(value)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = _load_plan(args.plan)
        if args.stage1_command == "estimate-cost":
            if plan.data["mode"] != "estimate-cost":
                raise ValueError("estimate-cost plan must declare mode estimate-cost")
            bundle = execute_plan(
                plan,
                executor=LookupCostEstimateAdapter.from_path(args.lookup),
            )
        else:
            if plan.data["mode"] != args.mode:
                raise ValueError("CLI mode does not match the plan mode")
            if plan.data["gpu_uuid"] != args.gpu_uuid:
                raise ValueError("CLI GPU UUID does not match the plan GPU UUID")
            if args.mode == "direct-cost":
                if not args.smoke:
                    raise ValueError("direct-cost mode requires --smoke")
                executor: ProfileExecutor = DirectCostRunner(
                    plan, TorchAOProfileExecutor(plan)
                )
            else:
                if args.smoke:
                    raise ValueError("--smoke is only valid for direct-cost mode")
                executor = TorchAOProfileExecutor(plan)
            bundle = execute_plan(plan, executor=executor)
    except (FileExistsError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"qbitplan: rejected: {exc}", file=sys.stderr)
        return 2

    metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact_id": bundle.files["bundle.json"],
                "artifact_root": str(plan.artifact_root),
                "bundle_path": str(bundle.path),
                "bundle_id": bundle.bundle_id,
                "manifest_id": metadata["source_manifest_id"],
                "run_id": bundle.run_id,
                "claim_boundary": metadata["claim_scope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

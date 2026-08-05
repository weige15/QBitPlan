"""Build a final-only functional-quality plan from the pinned MMLU-Pro source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from qbitplan.config import accepted_smoke_configuration
from qbitplan.identity import canonical_json_bytes, sha256_canonical
from qbitplan.plan import (
    MMLU_PRO_SOURCE_REVISION,
    MMLU_PRO_TEST_COUNT,
    ExperimentPlan,
    mmlu_prompt,
)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    content = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def build_plan(
    *,
    source_parquet: Path,
    manifest: dict[str, Any],
    artifact_root: str,
    attempt_id: str,
    gpu_uuid: str,
    profile_inventory: dict[str, Any],
) -> dict[str, Any]:
    from pyarrow import parquet

    parquet_hash = hashlib.sha256(source_parquet.read_bytes()).hexdigest()
    if parquet_hash != manifest["artifact_id"]:
        raise ValueError("source parquet does not match manifest artifact_id")
    rows = parquet.read_table(source_parquet).to_pylist()
    if len(rows) != MMLU_PRO_TEST_COUNT:
        raise ValueError("source parquet does not contain the pinned test count")
    base = accepted_smoke_configuration(
        artifact_root=artifact_root, attempt_id=attempt_id, gpu_uuid=gpu_uuid
    )
    base["mode"] = "functional-quality"
    base["profiles"] = list(profile_inventory["p_exec"])
    base["profile_inventory"] = profile_inventory
    base["source_manifest"] = manifest
    queries = []
    for row in sorted(rows, key=lambda item: int(item["question_id"])):
        source_id = str(row["question_id"])
        record = dict(row)
        queries.append(
            {
                "query_id": source_id,
                "source_id": source_id,
                "source_artifact_id": manifest["artifact_id"],
                "record": record,
                "record_hash": sha256_canonical(record),
                "dataset": "MMLU-Pro",
                "phase": "final",
                "source_revision": MMLU_PRO_SOURCE_REVISION,
                "prompt": mmlu_prompt(record["question"], record["options"]),
                "permitted": True,
            }
        )
    base["queries"] = queries
    ExperimentPlan.from_mapping(base)
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-inventory", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inventory_bytes = args.profile_inventory.read_bytes()
    inventory_payload = json.loads(inventory_bytes)
    source_artifact_ids = inventory_payload.get("source_artifact_ids")
    if not isinstance(source_artifact_ids, list) or len(source_artifact_ids) != 1:
        raise ValueError("profile inventory must identify one source artifact")
    profile_inventory = {
        "artifact_id": hashlib.sha256(inventory_bytes).hexdigest(),
        "artifact_type": inventory_payload.get("artifact_type"),
        "schema_version": inventory_payload.get("schema_version"),
        "source_manifest_id": inventory_payload.get("source_manifest_id"),
        "source_artifact_id": source_artifact_ids[0],
        "profile_ids": inventory_payload.get("profile_ids"),
        "record_count": inventory_payload.get("record_count"),
        "p_exec": inventory_payload.get("p_exec"),
        "outcome_record_count": inventory_payload.get("outcome_record_count"),
        "outcome_file": inventory_payload.get("outcome_file"),
        "claim_scope": inventory_payload.get("claim_scope"),
    }
    plan = build_plan(
        source_parquet=args.source_parquet,
        manifest=manifest,
        artifact_root=args.artifact_root,
        attempt_id=args.attempt_id,
        gpu_uuid=args.gpu_uuid,
        profile_inventory=profile_inventory,
    )
    _write_once(args.output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

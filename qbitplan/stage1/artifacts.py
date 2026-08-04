"""Immutable artifact materialization for the Stage-1 smoke seam."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contract import ExperimentPlan, ProfileExecutionResult, canonical_json_bytes


def _write_once(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class ArtifactBundle:
    """A content-addressed, write-once Stage-1 smoke artifact directory."""

    root: Path
    bundle_path: Path
    artifact_id: str
    manifest_id: str
    run_id: str
    payload: dict[str, Any]

    @classmethod
    def write(
        cls,
        plan: ExperimentPlan,
        results: Iterable[ProfileExecutionResult],
    ) -> ArtifactBundle:
        rows = list(results)
        expected_count = len(plan.profile_ids) * len(plan.queries)
        if len(rows) != expected_count:
            raise ValueError(f"expected {expected_count} execution results; got {len(rows)}")
        expected_keys = {
            (profile_id, query["query_id"])
            for profile_id in plan.profile_ids
            for query in plan.queries
        }
        actual_keys = {(row.profile_id, row.query_id) for row in rows}
        if actual_keys != expected_keys:
            raise ValueError("execution results do not account for every profile/query pair")

        created_at = datetime.now(UTC).isoformat()
        execution_result_mappings = [row.to_mapping() for row in rows]
        execution_results_content = b"".join(
            canonical_json_bytes(row) + b"\n" for row in execution_result_mappings
        )
        manifest_content = canonical_json_bytes(plan.manifest_payload)
        manifest_artifact_id = hashlib.sha256(manifest_content).hexdigest()

        payload = {
            "artifact_type": "stage1-smoke-artifact-bundle",
            "schema_version": "1.0",
            "evidence_class": "analytical",
            "evidence_class_scope": "artifact-metadata-only",
            "execution_evidence_classes": sorted({row.evidence_class for row in rows}),
            "claim_boundary": "executable-path-only/non-evidentiary-smoke",
            "mode": plan.mode,
            "plan_id": plan.plan_id,
            "profile_ids": list(plan.profile_ids),
            "execution_results": execution_result_mappings,
            "execution_results_artifact_id": hashlib.sha256(execution_results_content).hexdigest(),
            "execution_results_file": "execution-results.ndjson",
            "lineage": {
                "producer_git_sha": plan.data["producer_git_sha"],
                "source_artifact_ids": [manifest_artifact_id],
                "manifest_id": plan.manifest_id,
                "run_id": plan.run_id,
                "configuration_hash": plan.configuration_hash,
                "record_count": len(rows),
                "created_at": created_at,
            },
            "run": {
                "run_id": plan.run_id,
                "attempt_id": plan.data["attempt_id"],
            },
            "run_configuration": plan.data,
        }
        content = canonical_json_bytes(payload) + b"\n"
        artifact_id = hashlib.sha256(content).hexdigest()

        artifact_root = plan.artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        bundle_root = artifact_root / plan.run_id
        bundle_root.mkdir(exist_ok=False)
        bundle_path = bundle_root / "bundle.json"
        execution_results_path = bundle_root / "execution-results.ndjson"
        manifest_path = bundle_root / "phase-manifest.json"
        run_manifest_path = bundle_root / "run-manifest.json"
        index_path = bundle_root / "artifact-index.json"

        _write_once(bundle_path, content)
        _write_once(
            execution_results_path,
            execution_results_content,
        )
        _write_once(
            manifest_path,
            manifest_content,
        )
        _write_once(
            run_manifest_path,
            canonical_json_bytes(
                {
                    "run_id": plan.run_id,
                    "artifact_type": "stage1-run-manifest",
                    "schema_version": "1.0",
                    "evidence_class": "analytical",
                    "evidence_class_scope": "run-metadata-only",
                    "plan_id": plan.plan_id,
                    "producer_git_sha": plan.data["producer_git_sha"],
                    "source_artifact_ids": [manifest_artifact_id],
                    "source_manifest_id": plan.manifest_id,
                    "record_count": len(rows),
                    "created_at": created_at,
                    "run_configuration_hash": plan.configuration_hash,
                    "attempt_id": plan.data["attempt_id"],
                    "configuration_hash": plan.configuration_hash,
                }
            )
            + b"\n",
        )
        _write_once(
            index_path,
            canonical_json_bytes(
                {
                    "artifact_id": artifact_id,
                    "evidence_class": "analytical",
                    "evidence_class_scope": "index-metadata-only",
                    "artifact_type": payload["artifact_type"],
                    "schema_version": "1.0",
                    "producer_git_sha": plan.data["producer_git_sha"],
                    "source_artifact_ids": [manifest_artifact_id],
                    "source_manifest_id": plan.manifest_id,
                    "configuration_hash": plan.configuration_hash,
                    "record_count": len(rows),
                    "created_at": created_at,
                    "bundle_file": "bundle.json",
                    "execution_results_file": "execution-results.ndjson",
                    "execution_results_artifact_id": payload["execution_results_artifact_id"],
                    "manifest_id": plan.manifest_id,
                    "run_id": plan.run_id,
                }
            )
            + b"\n",
        )
        return cls(
            root=bundle_root,
            bundle_path=bundle_path,
            artifact_id=artifact_id,
            manifest_id=plan.manifest_id,
            run_id=plan.run_id,
            payload=payload,
        )

    @classmethod
    def load(cls, root: str | Path) -> ArtifactBundle:
        bundle_root = Path(root)
        index = _read_json(bundle_root / "artifact-index.json")
        bundle_path = bundle_root / index["bundle_file"]
        content = bundle_path.read_bytes()
        artifact_id = hashlib.sha256(content).hexdigest()
        if artifact_id != index["artifact_id"]:
            raise ValueError("artifact content hash does not match artifact index")
        execution_results = (bundle_root / index["execution_results_file"]).read_bytes()
        if hashlib.sha256(execution_results).hexdigest() != index["execution_results_artifact_id"]:
            raise ValueError("execution-results content hash does not match artifact index")
        payload = _read_json(bundle_path)
        return cls(
            root=bundle_root,
            bundle_path=bundle_path,
            artifact_id=artifact_id,
            manifest_id=index["manifest_id"],
            run_id=index["run_id"],
            payload=payload,
        )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact file is not a JSON object: {path}")  # noqa: TRY004
    return value

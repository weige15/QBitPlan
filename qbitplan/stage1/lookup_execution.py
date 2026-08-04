"""Artifact materialization for the Stage-1 lookup-cost adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from ..identity import sha256_canonical
from ..plan import COST_DIMENSIONS, ExperimentPlan


def _cost_observation_fields(observation: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "status",
        "reason_code",
        "cost_vector",
        "method",
        "lookup_artifact_id",
        "lookup_manifest_id",
        "source_artifact_id",
        "coverage",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"cost adapter observation is missing fields: {missing}")
    if observation["status"] not in {"complete", "incomplete"}:
        raise ValueError("cost adapter observation has an unsupported terminal status")
    for label in (
        "reason_code",
        "method",
        "lookup_artifact_id",
        "lookup_manifest_id",
        "source_artifact_id",
    ):
        if not isinstance(observation[label], str) or not observation[label]:
            raise ValueError(f"cost adapter observation requires a non-empty {label}")
    vector = observation["cost_vector"]
    if not isinstance(vector, Mapping) or set(vector) != set(COST_DIMENSIONS):
        raise ValueError("cost adapter observation must contain the complete six-dimension vector")
    omitted = 0
    for dimension in COST_DIMENSIONS:
        value = vector[dimension]
        if not isinstance(value, Mapping):
            raise ValueError(f"cost vector dimension {dimension} must be an object")
        if value.get("evidence_class") != "lookup-table estimated":
            raise ValueError(f"cost vector dimension {dimension} is not lookup-table estimated")
        if value.get("lookup_artifact_id") != observation["lookup_artifact_id"]:
            raise ValueError(f"cost vector dimension {dimension} has mismatched lookup lineage")
        if value.get("source_artifact_id") != observation["source_artifact_id"]:
            raise ValueError(f"cost vector dimension {dimension} has mismatched source lineage")
        status = value.get("status")
        if status == "estimated":
            raw_value = value.get("value")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"cost vector dimension {dimension} has a non-numeric estimate")
            if not math.isfinite(raw_value) or raw_value < 0:
                raise ValueError(f"cost vector dimension {dimension} has an invalid estimate")
            if "reason" in value:
                raise ValueError(f"estimated cost dimension {dimension} has an omission reason")
        elif status == "omitted/unavailable":
            omitted += 1
            if not isinstance(value.get("reason"), str) or not value["reason"]:
                raise ValueError(f"omitted cost dimension {dimension} requires a reason")
            if "value" in value:
                raise ValueError(f"omitted cost dimension {dimension} must not contain a value")
        else:
            raise ValueError(f"cost vector dimension {dimension} has an unsupported status")
    if (observation["status"] == "complete") != (omitted == 0):
        raise ValueError("cost adapter terminal status does not match dimension coverage")
    if not isinstance(observation["coverage"], Mapping):
        raise ValueError("cost adapter observation coverage must be an object")
    return dict(observation)


def execute_lookup_plan(experiment_plan: ExperimentPlan, executor: Any) -> Any:
    """Run exact lookup reads through the shared immutable bundle seam."""

    from .. import execution as seam

    if getattr(executor, "evidence_class", None) != "lookup-table estimated":
        raise ValueError("estimate-cost mode requires the lookup-table estimated adapter")
    metadata_factory = getattr(executor, "lookup_metadata", None)
    if not callable(metadata_factory):
        raise ValueError("estimate-cost adapter must expose lookup metadata")
    metadata = dict(cast(Mapping[str, Any], metadata_factory()))
    required = {
        "lookup_artifact_id",
        "lookup_manifest_id",
        "source_artifact",
        "source_artifact_id",
        "method",
        "coverage_manifest",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"lookup metadata is missing fields: {missing}")
    if metadata["source_artifact_id"] != metadata["source_artifact"].get("artifact_id"):
        raise ValueError("lookup metadata source artifact identity is inconsistent")
    if metadata["lookup_manifest_id"] != metadata["coverage_manifest"].get("manifest_id"):
        raise ValueError("lookup metadata coverage identity is inconsistent")

    producer_git_sha = seam._producer_git_sha()
    configuration_hash = seam._configuration_hash(experiment_plan)
    plan_id = experiment_plan.plan_id()
    run_identity = {
        "schema_version": "qbitplan.stage1.run-identity.v1",
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "mode": experiment_plan.data["mode"],
        "configuration_hash": configuration_hash,
        "lookup_artifact_id": metadata["lookup_artifact_id"],
        "lookup_manifest_id": metadata["lookup_manifest_id"],
    }
    run_id = sha256_canonical(run_identity)
    bundle_path = experiment_plan.artifact_root / run_id
    experiment_plan.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        bundle_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"immutable run already exists for this attempt: {bundle_path}; use a new attempt_id"
        ) from exc

    try:
        hardware_identity = dict(executor.hardware_identity())
    except (RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, MemoryError) as exc:
        hardware_identity = {
            "identity_status": "not_applicable",
            "reason_code": f"LOOKUP_HARDWARE_IDENTITY_EXCEPTION:{type(exc).__name__}",
        }
    created_at = seam._now()
    source_ids = [metadata["source_artifact_id"], metadata["lookup_artifact_id"]]
    estimates: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    for query in experiment_plan.data["queries"]:
        for profile_id in experiment_plan.data["profiles"]:
            observation = _cost_observation_fields(executor.execute(query, profile_id))
            common = {
                "phase": query["phase"],
                "query_id": query["query_id"],
                "profile_id": profile_id,
                "method": observation["method"],
                "lookup_artifact_id": observation["lookup_artifact_id"],
                "lookup_manifest_id": observation["lookup_manifest_id"],
                "source_artifact_id": observation["source_artifact_id"],
            }
            lineage = {
                "producer_git_sha": producer_git_sha,
                "source_manifest_id": experiment_plan.source_manifest_id,
                "source_artifact_id": experiment_plan.data["source_manifest"]["artifact_id"],
                "additional_source_artifact_ids": source_ids,
                "configuration_hash": configuration_hash,
                "record_count": 1,
                "created_at": created_at,
                "evidence_class": "lookup-table estimated",
            }
            estimates.append(
                {
                    **seam._lineage(
                        artifact_type="cost-estimate",
                        schema_version="qbitplan.stage1.cost-estimate.v1",
                        **lineage,
                    ),
                    **common,
                    "status": observation["status"],
                    "terminal_status": observation["status"],
                    "reason_code": observation["reason_code"],
                    "cost_vector": observation["cost_vector"],
                    "coverage": observation["coverage"],
                    "claim_scope": "lookup-table estimated cost comparisons only",
                }
            )
            coverage_records.append(
                {
                    **seam._lineage(
                        artifact_type="cost-coverage",
                        schema_version="qbitplan.stage1.cost-coverage.v1",
                        **lineage,
                    ),
                    **common,
                    "status": observation["status"],
                    "coverage": observation["coverage"],
                    "omitted_dimensions": observation["coverage"]["omitted_dimensions"],
                    "claim_scope": "lookup-table coverage only",
                }
            )

    shared_lineage = {
        "producer_git_sha": producer_git_sha,
        "source_manifest_id": experiment_plan.source_manifest_id,
        "source_artifact_id": experiment_plan.data["source_manifest"]["artifact_id"],
        "additional_source_artifact_ids": source_ids,
        "configuration_hash": configuration_hash,
        "created_at": created_at,
        "evidence_class": "lookup-table estimated",
    }
    lookup_manifest = {
        **seam._lineage(
            artifact_type="lookup-manifest",
            schema_version="qbitplan.stage1.lookup-manifest.v1",
            record_count=len(estimates),
            **shared_lineage,
        ),
        "lookup_artifact_id": metadata["lookup_artifact_id"],
        "lookup_manifest_id": metadata["lookup_manifest_id"],
        "method": metadata["method"],
        "source_artifact": metadata["source_artifact"],
        "coverage_manifest": metadata["coverage_manifest"],
        "claim_scope": "lookup-table estimated cost comparisons only",
    }
    run_manifest = {
        **seam._lineage(
            artifact_type="run-manifest",
            schema_version="qbitplan.stage1.run-manifest.v1",
            record_count=len(estimates),
            **shared_lineage,
        ),
        "run_id": run_id,
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "run_identity": run_identity,
        "file_hash_scope": "payload files; artifact-index.json records bundle.json and payload hashes",
        "hardware_identity": hardware_identity,
        "mode": experiment_plan.data["mode"],
        "profile_count": len(experiment_plan.data["profiles"]),
        "query_count": len(experiment_plan.data["queries"]),
    }
    file_contents = {
        "plan.json": seam._canonical_json_file(experiment_plan.to_mapping()),
        "run-manifest.json": seam._canonical_json_file(run_manifest),
        "lookup-manifest.json": seam._canonical_json_file(lookup_manifest),
        "cost-estimates.ndjson": seam._canonical_ndjson(estimates),
        "coverage.ndjson": seam._canonical_ndjson(coverage_records),
    }
    file_hashes: dict[str, str] = {}
    for filename, content in file_contents.items():
        file_hashes[filename] = seam._write_once(bundle_path / filename, content)
    bundle_identity = {
        "schema_version": "qbitplan.stage1.artifact-bundle.v1",
        "run_id": run_id,
        "plan_id": plan_id,
        "source_manifest_id": experiment_plan.source_manifest_id,
        "configuration_hash": configuration_hash,
        "lookup_artifact_id": metadata["lookup_artifact_id"],
        "lookup_manifest_id": metadata["lookup_manifest_id"],
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    bundle_payload = {
        **seam._lineage(
            artifact_type="artifact-bundle",
            schema_version="qbitplan.stage1.artifact-bundle.v1",
            record_count=len(estimates),
            **shared_lineage,
        ),
        "bundle_id": sha256_canonical(bundle_identity),
        "run_id": run_id,
        "plan_id": plan_id,
        "lookup_artifact_id": metadata["lookup_artifact_id"],
        "lookup_manifest_id": metadata["lookup_manifest_id"],
        "claim_scope": "lookup-table estimated cost comparisons only",
        "non_evidentiary": False,
        "files": [*file_contents, "bundle.json"],
        "artifact_index_file": "artifact-index.json",
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
        "file_hash_scope": "payload files; artifact-index.json records bundle.json and payload hashes",
        "hardware_identity": hardware_identity,
        "evidence_boundary": {
            "quality": "omitted",
            "cost": "lookup-table estimated; bounded to declared table, source artifact, covered profiles, and covered dimensions",
            "systems_benefit": "omitted/unavailable/lookup-table-not-direct-measurement",
            "generalization": "omitted",
        },
    }
    file_hashes["bundle.json"] = seam._write_once(
        bundle_path / "bundle.json", seam._canonical_json_file(bundle_payload)
    )
    index_payload = {
        **seam._lineage(
            artifact_type="artifact-index",
            schema_version="qbitplan.stage1.artifact-index.v1",
            record_count=len(file_hashes),
            **shared_lineage,
        ),
        "bundle_file": "bundle.json",
        "bundle_artifact_id": file_hashes["bundle.json"],
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    file_hashes["artifact-index.json"] = seam._write_once(
        bundle_path / "artifact-index.json", seam._canonical_json_file(index_payload)
    )
    return seam.ArtifactBundle(
        path=bundle_path,
        run_id=run_id,
        bundle_id=bundle_payload["bundle_id"],
        files=file_hashes,
    )

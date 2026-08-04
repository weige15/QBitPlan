"""The single Stage-1 smoke ExperimentPlan-to-ArtifactBundle seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Protocol

from .identity import canonical_json_bytes, sha256_bytes, sha256_canonical
from .plan import ExperimentPlan


class ProfileExecutor(Protocol):
    """The adapter boundary used by the public seam and test-scoped fakes."""

    evidence_class: str

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        """Transform and execute one query under one declared smoke variant."""

    def hardware_identity(self) -> Mapping[str, Any]:
        """Return the observed hardware identity for the run."""


@dataclass(frozen=True)
class ArtifactBundle:
    """A write-once materialized artifact bundle."""

    path: Path
    run_id: str
    bundle_id: str
    files: Mapping[str, str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _producer_git_sha() -> str:
    configured = os.environ.get("QBITPLAN_GIT_SHA")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    sha = result.stdout.strip()
    return sha if sha else "unavailable"


def _write_once(path: Path, content: bytes) -> str:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from exc
    return sha256_bytes(content)


def _canonical_json_file(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _canonical_ndjson(records: list[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _invalid_observation(reason_code: str) -> dict[str, Any]:
    return {
        "status": "invalid",
        "transform_status": "invalid",
        "forward_status": "not_attempted",
        "reason_code": reason_code,
        "transform_reason_code": reason_code,
        "forward_reason_code": "TRANSFORM_FAILED",
        "observed_group_prefix": [],
    }


def _observation_fields(observation: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "status",
        "transform_status",
        "forward_status",
        "reason_code",
        "transform_reason_code",
        "forward_reason_code",
        "observed_group_prefix",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"executor observation is missing fields: {missing}")
    if observation["status"] not in {"complete", "invalid"}:
        raise ValueError("executor observation has an unsupported terminal status")
    if observation["transform_status"] not in {"complete", "invalid", "not_applicable"}:
        raise ValueError("executor observation has an unsupported transform status")
    if observation["forward_status"] not in {"complete", "invalid", "not_attempted"}:
        raise ValueError("executor observation has an unsupported forward status")
    if observation["status"] == "invalid" and (not isinstance(observation["reason_code"], str) or not observation["reason_code"]):
        raise ValueError("executor observation requires a non-empty reason code")
    prefix = observation["observed_group_prefix"]
    if not isinstance(prefix, list):
        raise ValueError("executor observation group prefix must be a list")
    for group in prefix:
        if not isinstance(group, Mapping):
            raise ValueError("executor group prefix entries must be objects")
        if set(group) != {"group_index", "prefix_bits"}:
            raise ValueError("executor group prefix entries have unspecified fields")
        if not isinstance(group["group_index"], int) or not isinstance(group["prefix_bits"], str):
            raise ValueError("executor group prefix entries have invalid types")
    if observation["status"] == "complete":
        if observation["transform_status"] not in {"complete", "not_applicable"}:
            raise ValueError("complete observation has an incomplete transform")
        if observation["forward_status"] != "complete":
            raise ValueError("complete observation has an incomplete forward")
        if [entry["group_index"] for entry in prefix] != list(range(8)):
            raise ValueError("complete observation has an incomplete group order")
    elif observation["forward_status"] == "complete":
        raise ValueError("invalid observation has a complete forward")
    return dict(observation)


def _configuration_hash(plan: ExperimentPlan) -> str:
    data = plan.to_mapping()
    return sha256_canonical(
        {
            "model": data["model"],
            "tokenizer": data["tokenizer"],
            "software": data["software"],
            "decoder": data["decoder"],
            "runtime": data["runtime"],
            "seed": data["seed"],
            "determinism": data["determinism"],
            "gpu_uuid": data["gpu_uuid"],
            "mode": data["mode"],
        }
    )


def _lineage(
    *,
    artifact_type: str,
    schema_version: str,
    source_artifact_id: str,
    producer_git_sha: str,
    source_manifest_id: str,
    configuration_hash: str,
    record_count: int,
    created_at: str,
    evidence_class: str,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "producer_git_sha": producer_git_sha,
        "source_artifact_ids": [source_artifact_id],
        "source_manifest_id": source_manifest_id,
        "configuration_hash": configuration_hash,
        "record_count": record_count,
        "created_at": created_at,
        "evidence_class": evidence_class,
    }


def _profile_bits(variant_id: str) -> str:
    return "BF16" if variant_id == "BF16" else variant_id


def execute_plan(
    plan: ExperimentPlan | Mapping[str, Any],
    *,
    executor: ProfileExecutor | None = None,
) -> ArtifactBundle:
    """Execute one validated smoke plan into one immutable artifact bundle.

    ``executor`` is intentionally a Python seam for deterministic contract
    tests. The CLI never exposes a selector for it and always constructs the
    real :class:`TorchAOProfileExecutor`.
    """

    experiment_plan = plan if isinstance(plan, ExperimentPlan) else ExperimentPlan.from_mapping(plan)
    if executor is None:
        from .stage1.executor import TorchAOProfileExecutor

        executor = TorchAOProfileExecutor(experiment_plan)

    producer_git_sha = _producer_git_sha()
    configuration_hash = _configuration_hash(experiment_plan)
    plan_id = experiment_plan.plan_id()
    run_identity = {
        "schema_version": "qbitplan.stage1.run-identity.v1",
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "mode": experiment_plan.data["mode"],
        "gpu_uuid": experiment_plan.data["gpu_uuid"],
        "configuration_hash": configuration_hash,
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

    created_at = _now()
    evidence_class = getattr(executor, "evidence_class", "simulated")
    if evidence_class not in {"simulated", "directly measured"}:
        raise ValueError(f"unsupported executor evidence class: {evidence_class!r}")
    try:
        hardware_identity = dict(executor.hardware_identity())
    except (RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, MemoryError) as exc:
        hardware_identity = {
            "gpu_uuid": experiment_plan.data["gpu_uuid"],
            "identity_status": "unavailable",
            "identity_reason_code": f"HARDWARE_IDENTITY_EXCEPTION:{type(exc).__name__}",
        }

    outcomes: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    queries = experiment_plan.data["queries"]
    for query in queries:
        for variant_id in experiment_plan.data["profiles"]:
            try:
                observation = _observation_fields(executor.execute(query, variant_id))
            except (RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, MemoryError) as exc:
                observation = _invalid_observation(f"EXECUTOR_EXCEPTION:{type(exc).__name__}")
            outcome = {
                **_lineage(
                    artifact_type="profile-outcome",
                    schema_version="qbitplan.stage1.profile-outcome.v1",
                    producer_git_sha=producer_git_sha,
                    source_manifest_id=experiment_plan.source_manifest_id,
                    source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
                    configuration_hash=configuration_hash,
                    record_count=1,
                    created_at=created_at,
                    evidence_class=evidence_class,
                ),
                "phase": query["phase"],
                "query_id": query["query_id"],
                "variant_id": variant_id,
                "profile_bits": _profile_bits(variant_id),
                "status": observation["status"],
                "transform_status": observation["transform_status"],
                "terminal_status": observation["status"],
                "executable": observation["status"] == "complete"
                and observation["forward_status"] == "complete"
                and observation["transform_status"] in {"complete", "not_applicable"},
                "forward_status": observation["forward_status"],
                "reason_code": observation["reason_code"],
                "transform_reason_code": observation["transform_reason_code"],
                "forward_reason_code": observation["forward_reason_code"],
            }
            for field in ("token_count", "output_hash"):
                if field in observation:
                    outcome[field] = observation[field]
            outcomes.append(outcome)
            for boundary in observation["observed_group_prefix"]:
                boundaries.append(
                    {
                        **_lineage(
                            artifact_type="group-boundary",
                            schema_version="qbitplan.stage1.group-boundary.v1",
                            producer_git_sha=producer_git_sha,
                            source_manifest_id=experiment_plan.source_manifest_id,
                            source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
                            configuration_hash=configuration_hash,
                            record_count=1,
                            created_at=created_at,
                            evidence_class=evidence_class,
                        ),
                        "phase": query["phase"],
                        "query_id": query["query_id"],
                        "variant_id": variant_id,
                        "group_index": boundary["group_index"],
                        "prefix_bits": boundary["prefix_bits"],
                    }
                )

    plan_payload = experiment_plan.to_mapping()
    run_manifest = {
        **_lineage(
            artifact_type="run-manifest",
            schema_version="qbitplan.stage1.run-manifest.v1",
            producer_git_sha=producer_git_sha,
            source_manifest_id=experiment_plan.source_manifest_id,
            source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
            configuration_hash=configuration_hash,
            record_count=len(outcomes),
            created_at=created_at,
            evidence_class="analytical",
        ),
        "run_id": run_id,
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "run_identity": run_identity,
        "file_hash_scope": "payload files; artifact-index.json records bundle.json and payload hashes",
        "hardware_identity": hardware_identity,
        "mode": experiment_plan.data["mode"],
        "profile_count": len(experiment_plan.data["profiles"]),
        "query_count": len(queries),
    }
    file_contents = {
        "plan.json": _canonical_json_file(plan_payload),
        "run-manifest.json": _canonical_json_file(run_manifest),
        "profile-outcomes.ndjson": _canonical_ndjson(outcomes),
        "group-boundaries.ndjson": _canonical_ndjson(boundaries),
    }
    file_hashes: dict[str, str] = {}
    for filename, content in file_contents.items():
        file_hashes[filename] = _write_once(bundle_path / filename, content)

    bundle_identity = {
        "schema_version": "qbitplan.stage1.artifact-bundle.v1",
        "run_id": run_id,
        "plan_id": plan_id,
        "source_manifest_id": experiment_plan.source_manifest_id,
        "configuration_hash": configuration_hash,
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    bundle_payload = {
        **_lineage(
            artifact_type="artifact-bundle",
            schema_version="qbitplan.stage1.artifact-bundle.v1",
            producer_git_sha=producer_git_sha,
            source_manifest_id=experiment_plan.source_manifest_id,
            source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
            configuration_hash=configuration_hash,
            record_count=len(outcomes),
            created_at=created_at,
            evidence_class=evidence_class,
        ),
        "bundle_id": sha256_canonical(bundle_identity),
        "run_id": run_id,
        "plan_id": plan_id,
        "claim_scope": "executable-path smoke only",
        "non_evidentiary": True,
        "files": [*file_contents, "bundle.json"],
        "artifact_index_file": "artifact-index.json",
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
        "file_hash_scope": "payload files; artifact-index.json records bundle.json and payload hashes",
        "hardware_identity": hardware_identity,
        "evidence_boundary": {
            "quality": "omitted",
            "cost": "omitted/unavailable/smoke-mode-no-cost-adapter",
            "systems_benefit": "omitted",
            "generalization": "omitted",
        },
    }
    bundle_content = _canonical_json_file(bundle_payload)
    file_hashes["bundle.json"] = _write_once(bundle_path / "bundle.json", bundle_content)
    index_payload = {
        **_lineage(
            artifact_type="artifact-index",
            schema_version="qbitplan.stage1.artifact-index.v1",
            producer_git_sha=producer_git_sha,
            source_manifest_id=experiment_plan.source_manifest_id,
            source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
            configuration_hash=configuration_hash,
            record_count=len(file_hashes),
            created_at=created_at,
            evidence_class="analytical",
        ),
        "bundle_file": "bundle.json",
        "bundle_artifact_id": file_hashes["bundle.json"],
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    file_hashes["artifact-index.json"] = _write_once(
        bundle_path / "artifact-index.json", _canonical_json_file(index_payload)
    )
    return ArtifactBundle(
        path=bundle_path,
        run_id=run_id,
        bundle_id=bundle_payload["bundle_id"],
        files=file_hashes,
    )

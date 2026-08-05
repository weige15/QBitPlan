"""The single Stage-1 ExperimentPlan-to-ArtifactBundle seam."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .identity import canonical_json_bytes, sha256_bytes, sha256_canonical
from .plan import ExperimentPlan


class ProfileExecutor(Protocol):
    """The adapter boundary used by the public seam and test-scoped fakes."""

    evidence_class: str

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        ...

    def hardware_identity(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ArtifactBundle:
    """A write-once materialized artifact bundle."""

    path: Path
    run_id: str
    bundle_id: str
    files: Mapping[str, str]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


def _canonical_ndjson(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


_FAILURE_METADATA_FIELDS = frozenset(
    {
        "failure_phase",
        "group_index",
        "bit_width",
        "failing_fqn",
        "exception_type",
        "exception_message",
    }
)
_EMPTY_FAILURE_METADATA = {field: None for field in _FAILURE_METADATA_FIELDS}


def _invalid_observation(reason_code: str) -> dict[str, Any]:
    return {
        "status": "invalid",
        "transform_status": "invalid",
        "cuda_transfer_status": "not_attempted",
        "forward_status": "not_attempted",
        "reason_code": reason_code,
        "transform_reason_code": reason_code,
        "forward_reason_code": "TRANSFORM_FAILED",
        "failure_metadata": dict(_EMPTY_FAILURE_METADATA),
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
    observation = dict(observation)
    if "cuda_transfer_status" not in observation:
        observation["cuda_transfer_status"] = (
            "complete"
            if observation["forward_status"] == "complete"
            or (
                observation["transform_status"] == "complete"
                and observation["forward_status"] == "invalid"
            )
            else "not_attempted"
        )
    if "failure_metadata" not in observation:
        observation["failure_metadata"] = dict(_EMPTY_FAILURE_METADATA)
    if observation["cuda_transfer_status"] not in {"complete", "invalid", "not_attempted"}:
        raise ValueError("executor observation has an unsupported CUDA transfer status")
    metadata = observation["failure_metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("executor failure metadata must be an object")
    if set(metadata) != _FAILURE_METADATA_FIELDS:
        raise ValueError("executor failure metadata has unspecified fields")
    if metadata["failure_phase"] is not None and not isinstance(metadata["failure_phase"], str):
        raise TypeError("executor failure phase must be a string or null")
    if metadata["group_index"] is not None and (
        isinstance(metadata["group_index"], bool) or not isinstance(metadata["group_index"], int)
    ):
        raise TypeError("executor failure group index must be an integer or null")
    if metadata["bit_width"] is not None and metadata["bit_width"] not in {4, 8}:
        raise ValueError("executor failure bit width must be 4, 8, or null")
    for field in ("failing_fqn", "exception_type", "exception_message"):
        value = metadata[field]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"executor failure {field} must be a string or null")
        if field == "exception_message" and value is not None and len(value) > 512:
            raise ValueError("executor failure exception message exceeds its bound")
    observation["failure_metadata"] = dict(metadata)
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
        raise TypeError("executor observation group prefix must be a list")
    for group in prefix:
        if not isinstance(group, Mapping):
            raise TypeError("executor group prefix entries must be objects")
        if set(group) != {"group_index", "prefix_bits"}:
            raise ValueError("executor group prefix entries have unspecified fields")
        if not isinstance(group["group_index"], int) or not isinstance(group["prefix_bits"], str):
            raise TypeError("executor group prefix entries have invalid types")
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
    if data["mode"] == "estimate-cost":
        return sha256_canonical({
            "mode": data["mode"],
            "source_manifest": data["source_manifest"],
            "queries": data["queries"],
            "profiles": data["profiles"],
        })
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
    additional_source_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "producer_git_sha": producer_git_sha,
        "source_artifact_ids": [source_artifact_id, *(additional_source_artifact_ids or [])],
        "source_manifest_id": source_manifest_id,
        "configuration_hash": configuration_hash,
        "record_count": record_count,
        "created_at": created_at,
        "evidence_class": evidence_class,
    }


def _profile_bits(variant_id: str) -> str:
    return "BF16" if variant_id == "BF16" else variant_id


def _build_profile_inventory(
    *,
    profile_ids: list[str],
    outcomes: Sequence[Mapping[str, Any]],
    source_manifest_id: str,
    source_artifact_id: str,
    producer_git_sha: str,
    configuration_hash: str,
    created_at: str,
    outcome_evidence_classes: set[str],
) -> dict[str, Any]:
    """Summarize profile-level executable status without replacing outcomes."""

    records_by_profile: dict[str, list[Mapping[str, Any]]] = {
        profile_id: [] for profile_id in profile_ids
    }
    for outcome in outcomes:
        records_by_profile[outcome["variant_id"]].append(outcome)

    p_exec: list[str] = []
    excluded_profiles: list[dict[str, str]] = []
    profile_statuses: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        records = records_by_profile[profile_id]
        transform_failures = sorted(
            {
                str(outcome["transform_reason_code"] or outcome["reason_code"])
                for outcome in records
                if outcome["transform_status"] == "invalid"
            }
        )
        if transform_failures:
            reason = transform_failures[0]
            excluded_profiles.append(
                {"profile_id": profile_id, "reason_code": reason}
            )
            profile_statuses.append(
                {"profile_id": profile_id, "executable": False, "exclusion_reason_code": reason}
            )
            continue
        if any(
            outcome["status"] == "complete"
            and outcome["forward_status"] == "complete"
            and outcome["transform_status"] in {"complete", "not_applicable"}
            for outcome in records
        ):
            p_exec.append(profile_id)
            profile_statuses.append(
                {"profile_id": profile_id, "executable": True, "exclusion_reason_code": None}
            )
            continue
        forward_failures = sorted(
            {
                str(outcome["forward_reason_code"] or outcome["reason_code"])
                for outcome in records
                if outcome["forward_status"] == "invalid"
            }
        )
        reason = forward_failures[0] if forward_failures else "NO_COMPLETE_FORWARD"
        excluded_profiles.append(
            {
                "profile_id": profile_id,
                "reason_code": reason,
            }
        )
        profile_statuses.append(
            {"profile_id": profile_id, "executable": False, "exclusion_reason_code": reason}
        )

    return {
        **_lineage(
            artifact_type="profile-execution-inventory",
            schema_version="qbitplan.stage1.profile-inventory.v1",
            producer_git_sha=producer_git_sha,
            source_manifest_id=source_manifest_id,
            source_artifact_id=source_artifact_id,
            configuration_hash=configuration_hash,
            record_count=len(profile_ids),
            created_at=created_at,
            evidence_class="analytical",
        ),
        "profile_ids": profile_ids,
        "p_exec": p_exec,
        "profile_statuses": profile_statuses,
        "excluded_profiles": excluded_profiles,
        "enumeration_evidence_class": "analytical",
        "outcome_evidence_classes": sorted(outcome_evidence_classes),
        "outcome_file": "profile-outcomes.ndjson",
        "outcome_record_count": len(outcomes),
        "claim_scope": "executable profile feasibility only",
    }


def execute_plan(
    plan: ExperimentPlan | Mapping[str, Any],
    *,
    executor: ProfileExecutor | None = None,
) -> ArtifactBundle:
    """Execute one validated Stage-1 plan into one immutable artifact bundle.

    ``executor`` is intentionally a Python seam for deterministic contract
    tests. The CLI never exposes a selector for it and always constructs the
    real :class:`TorchAOProfileExecutor`.
    """

    experiment_plan = plan if isinstance(plan, ExperimentPlan) else ExperimentPlan.from_mapping(plan)
    if experiment_plan.data["mode"] == "estimate-cost":
        if executor is None:
            raise ValueError("estimate-cost mode requires an explicit lookup adapter")
        from .stage1.lookup_execution import execute_lookup_plan

        return execute_lookup_plan(experiment_plan, executor)
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
    if evidence_class not in {"simulated", "lookup-table estimated", "directly measured"}:
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
                "cuda_transfer_status": observation["cuda_transfer_status"],
                "terminal_status": observation["status"],
                "executable": observation["status"] == "complete"
                and observation["forward_status"] == "complete"
                and observation["transform_status"] in {"complete", "not_applicable"},
                "forward_status": observation["forward_status"],
                "reason_code": observation["reason_code"],
                "transform_reason_code": observation["transform_reason_code"],
                "forward_reason_code": observation["forward_reason_code"],
                "failure_metadata": observation["failure_metadata"],
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

    profile_inventory = None
    if experiment_plan.data["mode"] == "functional-quality":
        profile_inventory = _build_profile_inventory(
            profile_ids=list(experiment_plan.data["profiles"]),
            outcomes=outcomes,
            source_manifest_id=experiment_plan.source_manifest_id,
            source_artifact_id=experiment_plan.data["source_manifest"]["artifact_id"],
            producer_git_sha=producer_git_sha,
            configuration_hash=configuration_hash,
            created_at=created_at,
            outcome_evidence_classes={str(row["evidence_class"]) for row in outcomes},
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
    if profile_inventory is not None:
        file_contents["profile-inventory.json"] = _canonical_json_file(profile_inventory)
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
    is_smoke = experiment_plan.data["mode"] == "smoke"
    bundle_claim_scope = "executable-path smoke only" if is_smoke else "executable profile feasibility inventory only"
    cost_boundary = (
        "omitted/unavailable/smoke-mode-no-cost-adapter"
        if is_smoke
        else "omitted/unavailable/no-cost-adapter"
    )
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
        "claim_scope": bundle_claim_scope,
        "non_evidentiary": is_smoke,
        "files": [*file_contents, "bundle.json"],
        "artifact_index_file": "artifact-index.json",
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
        "file_hash_scope": "payload files; artifact-index.json records bundle.json and payload hashes",
        "hardware_identity": hardware_identity,
        "evidence_boundary": {
            "quality": "omitted",
            "cost": cost_boundary,
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

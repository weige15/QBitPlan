"""Validated public records for the first Stage-1 execution slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EVIDENCE_CLASSES = frozenset(
    {
        "analytical",
        "simulated",
        "lookup-table estimated",
        "directly measured",
    }
)
SMOKE_PROFILE_IDS = ("bf16", "00000000", "11111111", "01010101")
TERMINAL_STATUSES = frozenset({"complete", "invalid", "incomplete", "aborted"})
TRANSFORM_STATUSES = frozenset({"complete", "failed", "not_applicable"})
FORWARD_STATUSES = frozenset({"complete", "failed", "not_run"})


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize this contract's JSON identity domain deterministically.

    Issue #25 accepts only strings, booleans, integers, arrays, and objects.
    Unsupported non-integral floats are rejected before hashing so identities
    cannot depend on a non-canonical JSON number spelling.
    """

    return json.dumps(
        _normalize_canonical_value(value),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalize_canonical_value(value: Any) -> Any:
    """Normalize the JSON values used by this contract before hashing."""

    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("non-integral floats are not supported in canonical identity JSON")
        return int(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical identity objects require string keys")
        normalized = {
            key: _normalize_canonical_value(item) for key, item in value.items()
        }
        return dict(sorted(normalized.items(), key=lambda item: item[0].encode("utf-16-be")))
    if isinstance(value, list):
        return [_normalize_canonical_value(item) for item in value]
    return value


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"plan is missing required field: {key}")
    return mapping[key]


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _require(mapping, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"plan field {key} must be an object")
    return value


def _require_exact(mapping: Mapping[str, Any], key: str, expected: Any) -> Any:
    value = _require(mapping, key)
    if value != expected:
        raise ValueError(f"plan field {key} must be {expected!r}; got {value!r}")
    return value


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


@dataclass(frozen=True)
class ExperimentPlan:
    """An accepted, fully explicit Stage-1 smoke plan.

    The plan deliberately has no scientific defaults.  Values accepted by the
    protocol must be present in the serialized input so an omitted control
    cannot become a library default.
    """

    data: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentPlan":
        data = _copy_json(raw)
        if not isinstance(data, dict):
            raise ValueError("experiment plan must be a JSON object")

        _require_exact(data, "mode", "smoke")
        _require(data, "artifact_root")
        _require(data, "attempt_id")
        _require(data, "producer_git_sha")

        model = _require_mapping(data, "model")
        _require_exact(model, "id", "meta-llama/Llama-3.1-8B")
        _require_exact(model, "revision", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
        _require_exact(model, "architecture", "LlamaForCausalLM")
        _require_exact(model, "layers", 32)
        _require_exact(model, "dtype", "bfloat16")

        tokenizer = _require_mapping(data, "tokenizer")
        _require_exact(tokenizer, "revision", model["revision"])
        _require_exact(tokenizer, "id", model["id"])
        file_hashes = _require(tokenizer, "file_hashes")
        if not isinstance(file_hashes, dict) or not file_hashes:
            raise ValueError("tokenizer file_hashes must be a non-empty object")
        if any(not isinstance(name, str) or not isinstance(digest, str) or not digest for name, digest in file_hashes.items()):
            raise ValueError("tokenizer file_hashes must map filenames to non-empty hashes")
        _require_exact(tokenizer, "use_fast", True)
        _require_exact(tokenizer, "trust_remote_code", False)

        backend = _require_mapping(data, "backend")
        _require_exact(backend, "name", "torchao")
        _require_exact(backend, "version", "0.5.0")
        _require_exact(backend, "cuda", "12.4")
        _require_exact(backend, "pytorch", "2.4.0+cu124")
        _require_exact(backend, "transformers", "5.12.1")
        _require_exact(backend, "python", "3.12.3")
        _require_exact(backend, "numpy", "2.1.0")
        _require_exact(backend, "datasets", "5.0.0")
        _require_exact(backend, "accelerate", "1.14.0")
        _require_exact(backend, "safetensors", "0.8.0")

        hardware = _require_mapping(data, "hardware")
        _require(hardware, "gpu_uuid")
        _require_exact(hardware, "gpu_name", "NVIDIA GeForce RTX 3090")
        _require_exact(hardware, "driver", "580.159.03")

        groups = _require_mapping(data, "groups")
        _require_exact(groups, "count", 8)
        _require_exact(groups, "layers_per_group", 4)
        _require_exact(
            groups,
            "target_projections",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        _require_exact(groups, "int4_group_size", 128)

        profiles = _require(data, "profiles")
        if profiles != list(SMOKE_PROFILE_IDS):
            raise ValueError(f"smoke plan profiles must be {list(SMOKE_PROFILE_IDS)!r}")

        decoder = _require_mapping(data, "decoder")
        expected_decoder = {
            "max_new_tokens": 1024,
            "num_beams": 4,
            "num_return_sequences": 1,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
        }
        if dict(decoder) != expected_decoder:
            raise ValueError("decoder controls must exactly match the accepted smoke controls")

        runtime = _require_mapping(data, "runtime")
        expected_runtime = {
            "batch_size": 1,
            "model_eval": True,
            "inference_mode": True,
            "use_cache": True,
            "compile": False,
            "graph_capture": False,
            "cpu_offload": False,
            "dynamic_batching": False,
            "padding": False,
            "add_special_tokens": True,
            "truncation": False,
            "cublas_workspace_config": ":4096:8",
            "float32_matmul_precision": "highest",
            "tf32": False,
            "cudnn_benchmark": False,
            "deterministic_cudnn": True,
            "deterministic_algorithms": True,
        }
        if dict(runtime) != expected_runtime:
            raise ValueError("runtime controls must exactly match the accepted smoke controls")

        seeds = _require_mapping(data, "seeds")
        if dict(seeds) != {
            "python": 20260807,
            "numpy": 20260807,
            "torch_cpu": 20260807,
            "torch_cuda": 20260807,
        }:
            raise ValueError("seed controls must exactly match the accepted smoke controls")

        source_manifest = _require_mapping(data, "source_manifest")
        _require_exact(source_manifest, "dataset", "MATH")
        _require_exact(source_manifest, "source_revision", "985bdc1696e88e8643f081a0ff4719da39f2ae2a")
        permitted_source_ids = _require_mapping(source_manifest, "permitted_source_ids")
        training_source_ids = _require(permitted_source_ids, "training")
        validation_source_ids = _require(permitted_source_ids, "validation")
        final_source_ids = _require(source_manifest, "final_source_ids")
        source_id_lists = (training_source_ids, validation_source_ids, final_source_ids)
        if any(
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            for ids in source_id_lists
        ):
            raise ValueError("source manifest IDs must be non-empty string lists")
        if set(training_source_ids) & set(validation_source_ids):
            raise ValueError("training and validation source IDs must be disjoint")
        if set(training_source_ids) & set(final_source_ids) or set(validation_source_ids) & set(final_source_ids):
            raise ValueError("final source IDs must be disjoint from smoke source IDs")
        manifest_identity = {
            "dataset": source_manifest["dataset"],
            "source_revision": source_manifest["source_revision"],
            "permitted_source_ids": permitted_source_ids,
            "final_source_ids": final_source_ids,
        }
        _require_exact(source_manifest, "artifact_id", content_hash(manifest_identity))
        queries = _require(data, "queries")
        if not isinstance(queries, list) or len(queries) != 2:
            raise ValueError("smoke plan must contain exactly one training and one validation query")
        phases = set()
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError("each smoke query must be an object")
            for key in (
                "query_id",
                "source_id",
                "source_artifact_id",
                "record",
                "record_hash",
                "dataset",
                "phase",
                "source_revision",
                "prompt",
                "permitted",
            ):
                _require(query, key)
            _require_exact(query, "dataset", "MATH")
            _require_exact(query, "source_revision", "985bdc1696e88e8643f081a0ff4719da39f2ae2a")
            if not isinstance(query["source_id"], str) or not query["source_id"]:
                raise ValueError("smoke queries require a source-defined query ID")
            if not isinstance(query["source_artifact_id"], str) or not query["source_artifact_id"]:
                raise ValueError("smoke queries require a source artifact ID")
            record = query["record"]
            if not isinstance(record, dict):
                raise ValueError("each smoke query record must be an object")
            for key in ("problem", "solution", "level", "type"):
                _require(record, key)
            if any(not isinstance(record[key], str) for key in ("problem", "solution", "level", "type")):
                raise ValueError("MATH source record fields must be strings")
            if not isinstance(query["query_id"], str) or not query["query_id"]:
                raise ValueError("smoke queries require a non-empty query ID")
            if query["record_hash"] != content_hash(record):
                raise ValueError(f"record_hash does not match source record {query['query_id']}")
            expected_prompt = (
                "Problem:\n" + record["problem"]
                + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
                + "\\boxed{...}.\nSolution:"
            )
            if query["prompt"] != expected_prompt:
                raise ValueError(f"prompt does not match the accepted MATH format for {query['query_id']}")
            if query["phase"] not in {"training", "validation"}:
                raise ValueError("smoke queries may only be training or validation records")
            allowed_source_ids = (
                training_source_ids if query["phase"] == "training" else validation_source_ids
            )
            if query["source_id"] not in allowed_source_ids:
                raise ValueError("query source ID is not permitted for its phase")
            _require_exact(query, "source_artifact_id", source_manifest["artifact_id"])
            _require_exact(query, "permitted", True)
            phases.add(query["phase"])
        if phases != {"training", "validation"}:
            raise ValueError("smoke plan must contain one training and one validation query")
        if len({query["query_id"] for query in queries}) != len(queries):
            raise ValueError("query IDs must be unique")

        return cls(data=data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExperimentPlan":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    @property
    def mode(self) -> str:
        return self.data["mode"]

    @property
    def artifact_root(self) -> Path:
        return Path(self.data["artifact_root"])

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self.data["profiles"])

    @property
    def queries(self) -> tuple[dict[str, Any], ...]:
        return tuple(_copy_json(query) for query in self.data["queries"])

    @property
    def plan_id(self) -> str:
        return content_hash(self.data)

    @property
    def manifest_payload(self) -> dict[str, Any]:
        return {
            "artifact_type": "stage1-phase-manifest",
            "schema_version": "1.0",
            "evidence_class": "analytical",
            "evidence_class_scope": "source-manifest-metadata-only",
            "artifact_role": "source-manifest",
            "producer_git_sha": self.data["producer_git_sha"],
            "source_artifact_ids": sorted(
                {query["source_artifact_id"] for query in self.data["queries"]}
            ),
            "record_count": len(self.data["queries"]),
            "configuration_hash": self.configuration_hash,
            "source_manifest": self.data["source_manifest"],
            "queries": self.data["queries"],
        }

    @property
    def manifest_id(self) -> str:
        return content_hash(self.manifest_payload)

    @property
    def configuration_hash(self) -> str:
        configuration = {
            key: value
            for key, value in self.data.items()
            if key not in {"artifact_root", "attempt_id", "producer_git_sha", "queries"}
        }
        return content_hash(configuration)

    @property
    def run_id(self) -> str:
        return content_hash(
            {
                "plan_id": self.plan_id,
                "manifest_id": self.manifest_id,
                "configuration_hash": self.configuration_hash,
                "attempt_id": self.data["attempt_id"],
            }
        )


@dataclass(frozen=True)
class ProfileExecutionResult:
    """Externally visible outcome for one profile/query execution."""

    profile_id: str
    query_id: str
    transform_status: str
    forward_status: str
    terminal_status: str
    executable: bool
    reason_code: str
    evidence_class: str
    output_digest: str = ""
    detail: str = ""
    group_boundaries: tuple[str, ...] = ()
    upstream_state: str = ""

    def __post_init__(self) -> None:
        if self.transform_status not in TRANSFORM_STATUSES:
            raise ValueError(f"unsupported transform status: {self.transform_status}")
        if self.forward_status not in FORWARD_STATUSES:
            raise ValueError(f"unsupported forward status: {self.forward_status}")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal status: {self.terminal_status}")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"unsupported evidence class: {self.evidence_class}")
        if not self.reason_code:
            raise ValueError("profile execution results require a reason code")
        if self.executable and (
            self.transform_status not in {"complete", "not_applicable"}
            or self.forward_status != "complete"
            or self.terminal_status != "complete"
        ):
            raise ValueError("an executable result must have complete transform and forward statuses")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "query_id": self.query_id,
            "transform_status": self.transform_status,
            "forward_status": self.forward_status,
            "terminal_status": self.terminal_status,
            "executable": self.executable,
            "reason_code": self.reason_code,
            "evidence_class": self.evidence_class,
            "output_digest": self.output_digest,
            "detail": self.detail,
            "group_boundaries": list(self.group_boundaries),
            "upstream_state": self.upstream_state,
        }

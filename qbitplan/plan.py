"""Validation for the explicit Stage-1 smoke experiment plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import canonical_json_bytes, sha256_canonical

MODEL_IDENTIFIER = "meta-llama/Llama-3.1-8B"
MODEL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MATH_SOURCE_REVISION = "985bdc1696e88e8643f081a0ff4719da39f2ae2a"
EXPECTED_PROFILES = ("BF16", "00000000", "11111111", "01010101")
EXPECTED_SOFTWARE = {
    "python": "3.12.3",
    "pytorch": "2.4.0+cu124",
    "transformers": "5.12.1",
    "torchao": "0.5.0",
    "numpy": "2.1.0",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "cuda": "12.4",
    "nvidia_driver": "580.159.03",
}
EXPECTED_DECODER = {
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
EXPECTED_RUNTIME = {
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
    "pad_token_id_explicit": True,
    "eos_token_id_explicit": True,
}
EXPECTED_DETERMINISM = {
    "tf32": False,
    "float32_matmul_precision": "highest",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{label} has unspecified fields: {', '.join(details)}")


def _require_equal(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _require_exact_keys(value, set(expected), label)
    for key, expected_value in expected.items():
        numeric_decimal = (
            isinstance(expected_value, float)
            and isinstance(value[key], (int, float))
            and not isinstance(value[key], bool)
        )
        if value[key] != expected_value or (type(value[key]) is not type(expected_value) and not numeric_decimal):
            raise ValueError(
                f"{label}.{key} must be the accepted value {expected_value!r}; "
                f"got {value[key]!r}"
            )


def math_prompt(problem: str) -> str:
    return (
        "Problem:\n"
        + problem
        + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
        + r"\boxed{...}."
        + "\nSolution:"
    )


def _validate_source_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "dataset",
        "source_revision",
        "permitted_source_ids",
        "final_source_ids",
        "record_hash_algorithm",
        "record_id_format",
        "record_hashes",
        "counts",
        "artifact_id",
    }
    _require_exact_keys(value, required, "source_manifest")
    if value["dataset"] != "MATH" or value["source_revision"] != MATH_SOURCE_REVISION:
        raise ValueError("source_manifest must identify the accepted MATH revision")
    permitted = _require_mapping(value["permitted_source_ids"], "source_manifest.permitted_source_ids")
    _require_exact_keys(permitted, {"training", "validation"}, "source_manifest.permitted_source_ids")
    final_ids = value["final_source_ids"]
    if not isinstance(final_ids, list) or not all(isinstance(item, str) for item in final_ids):
        raise ValueError("source_manifest.final_source_ids must be a string array")
    permitted_ids: set[str] = set()
    for phase in ("training", "validation"):
        ids = permitted[phase]
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"source_manifest.permitted_source_ids.{phase} must be a string array")
        if len(ids) != len(set(ids)):
            raise ValueError(f"source_manifest.permitted_source_ids.{phase} must be unique")
        permitted_ids.update(ids)
    if permitted_ids & set(final_ids):
        raise ValueError("permitted source IDs overlap final source IDs")
    hashes = _require_mapping(value["record_hashes"], "source_manifest.record_hashes")
    _require_exact_keys(hashes, {"training", "validation", "final"}, "source_manifest.record_hashes")
    for phase, ids in (("training", permitted["training"]), ("validation", permitted["validation"]), ("final", final_ids)):
        phase_hashes = _require_mapping(hashes[phase], f"source_manifest.record_hashes.{phase}")
        if set(phase_hashes) != set(ids):
            raise ValueError(f"source_manifest.record_hashes.{phase} does not match its source IDs")
        if not all(isinstance(item, str) and len(item) == 64 for item in phase_hashes.values()):
            raise ValueError(f"source_manifest.record_hashes.{phase} contains invalid hashes")
    counts = _require_mapping(value["counts"], "source_manifest.counts")
    _require_exact_keys(counts, {"training", "validation", "final"}, "source_manifest.counts")
    if counts != {phase: len(ids) for phase, ids in (("training", permitted["training"]), ("validation", permitted["validation"]), ("final", final_ids))}:
        raise ValueError("source_manifest.counts does not match source ID manifests")
    identity = {
        "dataset": value["dataset"],
        "source_revision": value["source_revision"],
        "permitted_source_ids": permitted,
        "final_source_ids": final_ids,
    }
    if value["artifact_id"] != sha256_canonical(identity):
        raise ValueError("source_manifest.artifact_id does not match its canonical identity")


def _validate_queries(value: Any, source_manifest: Mapping[str, Any]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("queries must contain exactly one training and one validation record")
    permitted = source_manifest["permitted_source_ids"]
    seen_phases: set[str] = set()
    for index, query in enumerate(value):
        query_map = _require_mapping(query, f"queries[{index}]")
        required = {
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
        }
        _require_exact_keys(query_map, required, f"queries[{index}]")
        phase = query_map["phase"]
        if phase not in ("training", "validation") or phase in seen_phases:
            raise ValueError("queries must contain one distinct training and validation record")
        seen_phases.add(phase)
        source_id = query_map["source_id"]
        if query_map["query_id"] != source_id or source_id not in permitted[phase]:
            raise ValueError(f"queries[{index}] is outside its permitted source manifest")
        if query_map["source_artifact_id"] != source_manifest["artifact_id"]:
            raise ValueError(f"queries[{index}] has the wrong source artifact ID")
        if query_map["dataset"] != "MATH" or query_map["source_revision"] != MATH_SOURCE_REVISION:
            raise ValueError(f"queries[{index}] does not identify the accepted MATH source")
        if query_map["permitted"] is not True:
            raise ValueError(f"queries[{index}] must be explicitly permitted")
        record = _require_mapping(query_map["record"], f"queries[{index}].record")
        if not isinstance(record.get("problem"), str):
            raise ValueError(f"queries[{index}].record.problem must be a string")
        if query_map["record_hash"] != sha256_canonical(record):
            raise ValueError(f"queries[{index}] record hash does not match its record")
        if query_map["record_hash"] != source_manifest["record_hashes"][phase][source_id]:
            raise ValueError(f"queries[{index}] record hash does not match its source manifest")
        if query_map["prompt"] != math_prompt(record["problem"]):
            raise ValueError(f"queries[{index}] does not use the canonical zero-shot MATH prompt")
    if seen_phases != {"training", "validation"}:
        raise ValueError("queries must contain one training and one validation record")


@dataclass(frozen=True)
class ExperimentPlan:
    """An already validated, immutable view of an explicit experiment plan."""

    data: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentPlan":
        raw = dict(value)
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "artifact_root",
                "attempt_id",
                "mode",
                "model",
                "tokenizer",
                "software",
                "decoder",
                "runtime",
                "seed",
                "determinism",
                "gpu_uuid",
                "source_manifest",
                "queries",
                "profiles",
            },
            "experiment plan",
        )
        if raw["schema_version"] != "qbitplan.stage1.experiment-plan.v1":
            raise ValueError("experiment plan has an unsupported schema_version")
        if not isinstance(raw["artifact_root"], str) or not raw["artifact_root"]:
            raise ValueError("artifact_root must be an explicit non-empty path")
        if not isinstance(raw["attempt_id"], str) or not raw["attempt_id"]:
            raise ValueError("attempt_id must be explicit and non-empty")
        if raw["mode"] != "smoke":
            raise ValueError("issue #25 supports only explicit smoke mode")
        model = _require_mapping(raw["model"], "model")
        _require_exact_keys(model, {"identifier", "revision", "architecture", "layers", "dtype"}, "model")
        if model != {"identifier": MODEL_IDENTIFIER, "revision": MODEL_REVISION, "architecture": "LlamaForCausalLM", "layers": 32, "dtype": "bfloat16"}:
            raise ValueError("model must match the accepted pinned BF16 pilot")
        tokenizer = _require_mapping(raw["tokenizer"], "tokenizer")
        _require_exact_keys(tokenizer, {"identifier", "revision", "use_fast", "trust_remote_code"}, "tokenizer")
        if tokenizer != {"identifier": MODEL_IDENTIFIER, "revision": MODEL_REVISION, "use_fast": True, "trust_remote_code": False}:
            raise ValueError("tokenizer must match the accepted pinned fast tokenizer")
        software = _require_mapping(raw["software"], "software")
        _require_equal(software, EXPECTED_SOFTWARE, "software")
        _require_equal(_require_mapping(raw["decoder"], "decoder"), EXPECTED_DECODER, "decoder")
        _require_equal(_require_mapping(raw["runtime"], "runtime"), EXPECTED_RUNTIME, "runtime")
        if raw["seed"] != 20260807 or type(raw["seed"]) is not int:
            raise ValueError("seed must be the accepted deterministic seed 20260807")
        _require_equal(_require_mapping(raw["determinism"], "determinism"), EXPECTED_DETERMINISM, "determinism")
        if not isinstance(raw["gpu_uuid"], str) or not raw["gpu_uuid"]:
            raise ValueError("gpu_uuid must be explicit and non-empty")
        source_manifest = _require_mapping(raw["source_manifest"], "source_manifest")
        _validate_source_manifest(source_manifest)
        _validate_queries(raw["queries"], source_manifest)
        if raw["profiles"] != list(EXPECTED_PROFILES):
            raise ValueError("smoke profiles must be BF16, all-4, all-8, and mixed 01010101 in order")
        return cls(copy.deepcopy(raw))

    @property
    def artifact_root(self) -> Path:
        return Path(self.data["artifact_root"]).expanduser().resolve()

    @property
    def attempt_id(self) -> str:
        return self.data["attempt_id"]

    @property
    def source_manifest_id(self) -> str:
        return sha256_canonical(self.data["source_manifest"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def plan_id(self) -> str:
        return sha256_canonical(self.data)

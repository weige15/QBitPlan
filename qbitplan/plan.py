"""Validation for the explicit Stage-1 smoke experiment plan."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import canonical_json_bytes, sha256_canonical  # noqa: F401

MODEL_IDENTIFIER = "meta-llama/Llama-3.1-8B"
MODEL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MATH_SOURCE_REVISION = "985bdc1696e88e8643f081a0ff4719da39f2ae2a"
MMLU_PRO_SOURCE_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_SOURCE_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_SOURCE_SPLIT = "test"
MMLU_PRO_TEST_PARQUET = "data/test-00000-of-00001.parquet"
MMLU_PRO_TEST_PARQUET_SHA256 = (
    "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
)
MMLU_PRO_TEST_COUNT = 12_032
MATH_SOURCE_ARCHIVE_URL = "https://web.archive.org/web/20240101000000id_/https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
MATH_SOURCE_ARCHIVE_SHA256 = "0fbe4fad0df66942db6c221cdcc95b298cc7f4595a2f0f518360cce84e90d9ac"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
MATH_SOURCE_TRAINING_COUNT = 7_500
MATH_SOURCE_VALIDATION_COUNT = 4_500
MATH_FINAL_COUNT = 500
MATH_SOURCE_FILE_COUNT = 12_500
MATH_PROFILE_INVENTORY_SOURCE_MANIFEST_ID = (
    "e6b158a296905e843984b19059226629059fd660a8cf073fcf7ed51daabadf15"
)
MATH_PROFILE_INVENTORY_SOURCE_ARTIFACT_ID = MATH_SOURCE_ARCHIVE_SHA256
MATH_SOURCE_TREE_ARTIFACT_ID = "d6d24801c6380e8f325c6fa7b226807a81c8f3bcc5f753f31c7adf7d523860ba"
TOKENIZER_FILE_HASHES = {
    "tokenizer.json": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
    "tokenizer_config.json": "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
    "original/tokenizer.model": "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
}
EXPECTED_PROFILES = ("BF16", "00000000", "11111111", "01010101")
ANALYTICAL_PROFILE_IDS = tuple(f"{profile_id:08b}" for profile_id in range(256))
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
COST_DIMENSIONS = (
    "resident_accelerator_bytes",
    "host_to_device_bytes",
    "latency",
    "prefetch_stall_time",
    "kernel_switch_count",
    "controller_probe_feedback_overhead",
)


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


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def math_prompt(problem: str) -> str:
    return (
        "Problem:\n"
        + problem
        + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
        + r"\boxed{...}."
        + "\nSolution:"
    )


def mmlu_prompt(question: str, options: list[str]) -> str:
    option_lines = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        "Question:\n"
        + question
        + "\n\nOptions:\n"
        + option_lines
        + "\n\nAnswer with the single letter of the correct option.\nAnswer:"
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
        "raw_artifact",
        "manifest_id",
        "counts",
        "artifact_id",
    }
    _require_exact_keys(value, required, "source_manifest")
    if value["dataset"] == "MMLU-Pro":
        _validate_mmlu_source_manifest(value)
        return
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
        if not all(_is_sha256_hex(item) for item in phase_hashes.values()):
            raise ValueError(f"source_manifest.record_hashes.{phase} contains invalid hashes")
    counts = _require_mapping(value["counts"], "source_manifest.counts")
    _require_exact_keys(counts, {"training", "validation", "final"}, "source_manifest.counts")
    if counts != {phase: len(ids) for phase, ids in (("training", permitted["training"]), ("validation", permitted["validation"]), ("final", final_ids))}:
        raise ValueError("source_manifest.counts does not match source ID manifests")
    if counts != {
        "training": MATH_SOURCE_TRAINING_COUNT,
        "validation": MATH_SOURCE_VALIDATION_COUNT,
        "final": MATH_FINAL_COUNT,
    }:
        raise ValueError("source_manifest.counts must match the pinned MATH split")
    raw_artifact = _require_mapping(value["raw_artifact"], "source_manifest.raw_artifact")
    _require_exact_keys(
        raw_artifact,
        {
            "artifact_id",
            "source_tree_artifact_id",
            "source_archive_url",
            "source_archive_sha256",
            "source_layout",
            "source_file_count",
            "math500_file",
            "math500_file_sha256",
        },
        "source_manifest.raw_artifact",
    )
    if raw_artifact["artifact_id"] != value["artifact_id"]:
        raise ValueError("source_manifest.raw_artifact.artifact_id does not match artifact_id")
    if raw_artifact["artifact_id"] != MATH_SOURCE_ARCHIVE_SHA256:
        raise ValueError("source_manifest.artifact_id must match the pinned Berkeley source archive")
    if raw_artifact["source_archive_url"] != MATH_SOURCE_ARCHIVE_URL:
        raise ValueError("source_manifest.raw_artifact.source_archive_url must match the pinned acquisition")
    if raw_artifact["source_archive_sha256"] != MATH_SOURCE_ARCHIVE_SHA256:
        raise ValueError("source_manifest.raw_artifact.source_archive_sha256 must match the pinned acquisition")
    if raw_artifact["source_tree_artifact_id"] != MATH_SOURCE_TREE_ARTIFACT_ID:
        raise ValueError("source_manifest.raw_artifact.source_tree_artifact_id must match the pinned source tree")
    if raw_artifact["source_file_count"] != MATH_SOURCE_FILE_COUNT:
        raise ValueError("source_manifest.raw_artifact.source_file_count must match the pinned source tree")
    if raw_artifact["math500_file_sha256"] != MATH500_SHA256:
        raise ValueError("source_manifest.raw_artifact.math500_file_sha256 must match the pinned MATH-500 input")
    if not _is_sha256_hex(value["artifact_id"]):
        raise ValueError("source_manifest.artifact_id must be a SHA-256 hex digest")
    if not _is_sha256_hex(raw_artifact["source_tree_artifact_id"]):
        raise ValueError("source_manifest.raw_artifact.source_tree_artifact_id must be a SHA-256 hex digest")
    if not isinstance(raw_artifact["source_archive_url"], str) or not raw_artifact["source_archive_url"]:
        raise ValueError("source_manifest.raw_artifact.source_archive_url must be a non-empty string")
    if not _is_sha256_hex(raw_artifact["source_archive_sha256"]):
        raise ValueError("source_manifest.raw_artifact.source_archive_sha256 must be a SHA-256 hex digest")
    if not isinstance(raw_artifact["source_file_count"], int) or raw_artifact["source_file_count"] <= 0:
        raise ValueError("source_manifest.raw_artifact.source_file_count must be positive")
    if not isinstance(raw_artifact["source_layout"], str) or not raw_artifact["source_layout"]:
        raise ValueError("source_manifest.raw_artifact.source_layout must be a non-empty string")
    if not isinstance(raw_artifact["math500_file"], str) or not raw_artifact["math500_file"]:
        raise ValueError("source_manifest.raw_artifact.math500_file must be a non-empty string")
    if not _is_sha256_hex(raw_artifact["math500_file_sha256"]):
        raise ValueError("source_manifest.raw_artifact.math500_file_sha256 must be a SHA-256 hex digest")
    if not _is_sha256_hex(value["manifest_id"]):
        raise ValueError("source_manifest.manifest_id must be a SHA-256 hex digest")
    manifest_payload = dict(value)
    del manifest_payload["manifest_id"]
    if value["manifest_id"] != sha256_canonical(manifest_payload):
        raise ValueError("source_manifest.manifest_id does not match its canonical manifest payload")


def _validate_queries(
    value: Any, source_manifest: Mapping[str, Any], mode: str = "smoke"
) -> None:
    if source_manifest["dataset"] == "MMLU-Pro":
        _validate_mmlu_queries(value, source_manifest, mode)
        return
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


def _validate_mmlu_queries(
    value: Any, source_manifest: Mapping[str, Any], mode: str
) -> None:
    if mode != "functional-quality":
        raise ValueError("MMLU-Pro final-only plans require functional-quality mode")
    if not isinstance(value, list) or len(value) != MMLU_PRO_TEST_COUNT:
        raise ValueError("MMLU-Pro plans must contain all pinned final test records")
    final_ids = set(source_manifest["final_source_ids"])
    seen: set[str] = set()
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
        source_id = query_map["source_id"]
        if (
            query_map["query_id"] != source_id
            or not isinstance(source_id, str)
            or source_id not in final_ids
            or source_id in seen
        ):
            raise ValueError(f"queries[{index}] is outside the MMLU-Pro final manifest")
        seen.add(source_id)
        if query_map["phase"] != "final":
            raise ValueError("MMLU-Pro queries must be final-phase records")
        if query_map["source_artifact_id"] != source_manifest["artifact_id"]:
            raise ValueError(f"queries[{index}] has the wrong source artifact ID")
        if (
            query_map["dataset"] != "MMLU-Pro"
            or query_map["source_revision"] != MMLU_PRO_SOURCE_REVISION
        ):
            raise ValueError(
                f"queries[{index}] does not identify the accepted MMLU-Pro source"
            )
        if query_map["permitted"] is not True:
            raise ValueError(f"queries[{index}] must be explicitly permitted")
        record = _require_mapping(query_map["record"], f"queries[{index}].record")
        question = record.get("question")
        options = record.get("options")
        answer = record.get("answer")
        answer_index = record.get("answer_index")
        if (
            not isinstance(question, str)
            or not isinstance(options, list)
            or not 3 <= len(options) <= 10
            or not all(isinstance(option, str) for option in options)
            or not isinstance(answer, str)
            or len(answer) != 1
            or answer not in "ABCDEFGHIJ"
            or not isinstance(answer_index, int)
            or isinstance(answer_index, bool)
            or not 0 <= answer_index < len(options)
            or answer != chr(ord("A") + answer_index)
        ):
            raise ValueError(f"queries[{index}] has an invalid MMLU-Pro record")
        if query_map["record_hash"] != sha256_canonical(record):
            raise ValueError(f"queries[{index}] record hash does not match its record")
        if (
            query_map["record_hash"]
            != source_manifest["record_hashes"]["final"][source_id]
        ):
            raise ValueError(
                f"queries[{index}] record hash does not match its source manifest"
            )
        if query_map["prompt"] != mmlu_prompt(question, options):
            raise ValueError(
                f"queries[{index}] does not use the canonical MMLU-Pro prompt"
            )
    if seen != final_ids:
        raise ValueError("MMLU-Pro queries must cover every pinned final question_id")


def _validate_profile_inventory(value: Mapping[str, Any]) -> None:
    _require_exact_keys(
        value,
        {"artifact_id", "artifact_type", "schema_version", "source_manifest_id", "source_artifact_id", "profile_ids", "record_count", "p_exec", "outcome_record_count", "outcome_file", "claim_scope"},
        "profile_inventory",
    )
    if value["artifact_type"] != "profile-execution-inventory":
        raise ValueError("profile_inventory has the wrong artifact type")
    if value["schema_version"] != "qbitplan.stage1.profile-inventory.v1":
        raise ValueError("profile_inventory has the wrong schema version")
    if not _is_sha256_hex(value["artifact_id"]):
        raise ValueError("profile_inventory.artifact_id must be a SHA-256 hex digest")
    if value["source_manifest_id"] != MATH_PROFILE_INVENTORY_SOURCE_MANIFEST_ID:
        raise ValueError("profile_inventory must come from the accepted non-final MATH manifest")
    if value["source_artifact_id"] != MATH_PROFILE_INVENTORY_SOURCE_ARTIFACT_ID:
        raise ValueError("profile_inventory must come from the accepted MATH source artifact")
    if value["profile_ids"] != list(ANALYTICAL_PROFILE_IDS):
        raise ValueError("profile_inventory must enumerate all canonical profiles")
    if value["record_count"] != len(ANALYTICAL_PROFILE_IDS):
        raise ValueError("profile_inventory.record_count must be 256")
    if (
        not isinstance(value["outcome_record_count"], int)
        or isinstance(value["outcome_record_count"], bool)
        or value["outcome_record_count"] < len(ANALYTICAL_PROFILE_IDS)
    ):
        raise ValueError("profile_inventory must retain per-profile outcomes")
    if value["outcome_file"] != "profile-outcomes.ndjson":
        raise ValueError("profile_inventory must reference profile-outcomes.ndjson")
    if value["claim_scope"] != "executable profile feasibility only":
        raise ValueError("profile_inventory has an unsupported claim scope")
    p_exec = value["p_exec"]
    if (
        not isinstance(p_exec, list)
        or not p_exec
        or not all(isinstance(profile, str) for profile in p_exec)
        or len(p_exec) != len(set(p_exec))
        or any(profile not in ANALYTICAL_PROFILE_IDS for profile in p_exec)
        or p_exec != sorted(p_exec)
    ):
        raise ValueError("profile_inventory.p_exec must be a non-empty sorted canonical profile list")

@dataclass(frozen=True)
class ExperimentPlan:
    """An already validated, immutable view of an explicit experiment plan."""

    data: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentPlan:
        raw = dict(value)
        if raw.get("mode") == "estimate-cost":
            return cls._from_cost_estimate_mapping(raw)
        required_plan_keys = {
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
        }
        source_dataset = (
            raw.get("source_manifest", {}).get("dataset")
            if isinstance(raw.get("source_manifest"), Mapping)
            else None
        )
        if source_dataset == "MMLU-Pro":
            required_plan_keys.add("profile_inventory")
        _require_exact_keys(raw, required_plan_keys, "experiment plan")
        if raw["schema_version"] != "qbitplan.stage1.experiment-plan.v1":
            raise ValueError("experiment plan has an unsupported schema_version")
        if not isinstance(raw["artifact_root"], str) or not raw["artifact_root"]:
            raise ValueError("artifact_root must be an explicit non-empty path")
        if not isinstance(raw["attempt_id"], str) or not raw["attempt_id"]:
            raise ValueError("attempt_id must be explicit and non-empty")
        if raw["mode"] not in {"smoke", "functional-quality"}:
            raise ValueError("mode must be smoke or functional-quality")
        model = _require_mapping(raw["model"], "model")
        _require_exact_keys(model, {"identifier", "revision", "architecture", "layers", "dtype"}, "model")
        if model != {"identifier": MODEL_IDENTIFIER, "revision": MODEL_REVISION, "architecture": "LlamaForCausalLM", "layers": 32, "dtype": "bfloat16"}:
            raise ValueError("model must match the accepted pinned BF16 pilot")
        tokenizer = _require_mapping(raw["tokenizer"], "tokenizer")
        tokenizer_keys = {"identifier", "revision", "use_fast", "trust_remote_code"}
        if "file_hashes" in tokenizer:
            tokenizer_keys.add("file_hashes")
        _require_exact_keys(tokenizer, tokenizer_keys, "tokenizer")
        expected_tokenizer = {
            "identifier": MODEL_IDENTIFIER,
            "revision": MODEL_REVISION,
            "use_fast": True,
            "trust_remote_code": False,
        }
        if tokenizer != expected_tokenizer and (
            set(tokenizer) != {*expected_tokenizer, "file_hashes"}
            or any(
                tokenizer.get(key) != value
                for key, value in expected_tokenizer.items()
            )
            or tokenizer.get("file_hashes") != TOKENIZER_FILE_HASHES
        ):
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
        if (
            source_manifest["dataset"] == "MMLU-Pro"
            and tokenizer.get("file_hashes") != TOKENIZER_FILE_HASHES
        ):
            raise ValueError(
                "MMLU-Pro plans must declare the accepted tokenizer file hashes"
            )
        _validate_queries(raw["queries"], source_manifest, raw["mode"])
        if source_manifest["dataset"] == "MMLU-Pro":
            profile_inventory = _require_mapping(
                raw["profile_inventory"], "profile_inventory"
            )
            _validate_profile_inventory(profile_inventory)
            expected_profiles = list(profile_inventory["p_exec"])
        else:
            expected_profiles = (
                list(EXPECTED_PROFILES)
                if raw["mode"] == "smoke"
                else list(ANALYTICAL_PROFILE_IDS)
            )
        if raw["profiles"] != expected_profiles:
            raise ValueError("profiles must match the declared mode's canonical profile scope")
        return cls(copy.deepcopy(raw))

    @classmethod
    def _from_cost_estimate_mapping(cls, raw: dict[str, Any]) -> ExperimentPlan:
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "artifact_root",
                "attempt_id",
                "mode",
                "source_manifest",
                "queries",
                "profiles",
            },
            "cost estimate experiment plan",
        )
        if raw["schema_version"] != "qbitplan.stage1.experiment-plan.v1":
            raise ValueError("cost estimate plan has an unsupported schema_version")
        if not isinstance(raw["artifact_root"], str) or not raw["artifact_root"]:
            raise ValueError("artifact_root must be an explicit non-empty path")
        if not isinstance(raw["attempt_id"], str) or not raw["attempt_id"]:
            raise ValueError("attempt_id must be explicit and non-empty")
        source_manifest = _require_mapping(raw["source_manifest"], "source_manifest")
        _validate_source_manifest(source_manifest)
        _validate_queries(raw["queries"], source_manifest)
        profiles = raw["profiles"]
        if not isinstance(profiles, list) or not profiles or len(profiles) != len(set(profiles)):
            raise ValueError("cost estimate profiles must be a non-empty unique string array")
        for profile in profiles:
            if profile == "BF16":
                continue
            if not isinstance(profile, str) or len(profile) != 8 or set(profile) - {"0", "1"}:
                raise ValueError(f"cost estimate profile is not a canonical eight-bit ID: {profile!r}")
        return cls(copy.deepcopy(raw))

    @property
    def artifact_root(self) -> Path:
        return Path(self.data["artifact_root"]).expanduser().resolve()

    @property
    def attempt_id(self) -> str:
        return self.data["attempt_id"]

    @property
    def source_manifest_id(self) -> str:
        return self.data["source_manifest"]["manifest_id"]

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def plan_id(self) -> str:
        return sha256_canonical(self.data)

def _validate_mmlu_source_manifest(value: Mapping[str, Any]) -> None:
    if value["source_revision"] != MMLU_PRO_SOURCE_REVISION:
        raise ValueError("source_manifest must identify the accepted MMLU-Pro revision")
    permitted = _require_mapping(
        value["permitted_source_ids"], "source_manifest.permitted_source_ids"
    )
    _require_exact_keys(
        permitted, {"training", "validation"}, "source_manifest.permitted_source_ids"
    )
    if permitted != {"training": [], "validation": []}:
        raise ValueError("MMLU-Pro is final-only and cannot populate non-final phases")
    final_ids = value["final_source_ids"]
    if not isinstance(final_ids, list) or not all(
        isinstance(item, str) for item in final_ids
    ):
        raise ValueError("MMLU-Pro final_source_ids must be a string array")
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("MMLU-Pro final_source_ids must be unique")
    if len(final_ids) != MMLU_PRO_TEST_COUNT:
        raise ValueError("MMLU-Pro final_source_ids must contain all pinned test rows")
    hashes = _require_mapping(value["record_hashes"], "source_manifest.record_hashes")
    _require_exact_keys(
        hashes, {"training", "validation", "final"}, "source_manifest.record_hashes"
    )
    if hashes["training"] != {} or hashes["validation"] != {}:
        raise ValueError("MMLU-Pro has no non-final source hashes")
    final_hashes = _require_mapping(
        hashes["final"], "source_manifest.record_hashes.final"
    )
    if set(final_hashes) != set(final_ids) or not all(
        _is_sha256_hex(item) for item in final_hashes.values()
    ):
        raise ValueError("MMLU-Pro final record hashes do not match source IDs")
    counts = _require_mapping(value["counts"], "source_manifest.counts")
    _require_exact_keys(
        counts, {"training", "validation", "final"}, "source_manifest.counts"
    )
    if counts != {"training": 0, "validation": 0, "final": MMLU_PRO_TEST_COUNT}:
        raise ValueError("MMLU-Pro counts do not match the pinned test split")
    raw_artifact = _require_mapping(
        value["raw_artifact"], "source_manifest.raw_artifact"
    )
    _require_exact_keys(
        raw_artifact,
        {
            "artifact_id",
            "source_repo",
            "source_path",
            "source_sha256",
            "source_split",
            "source_rows",
        },
        "source_manifest.raw_artifact",
    )
    if raw_artifact != {
        "artifact_id": MMLU_PRO_TEST_PARQUET_SHA256,
        "source_repo": MMLU_PRO_SOURCE_REPO,
        "source_path": MMLU_PRO_TEST_PARQUET,
        "source_sha256": MMLU_PRO_TEST_PARQUET_SHA256,
        "source_split": MMLU_PRO_SOURCE_SPLIT,
        "source_rows": MMLU_PRO_TEST_COUNT,
    }:
        raise ValueError("MMLU-Pro raw artifact does not match the accepted pin")
    if value["artifact_id"] != MMLU_PRO_TEST_PARQUET_SHA256:
        raise ValueError("MMLU-Pro artifact_id must match the pinned parquet")
    if not _is_sha256_hex(value["manifest_id"]):
        raise ValueError("source_manifest.manifest_id must be a SHA-256 hex digest")
    manifest_payload = dict(value)
    del manifest_payload["manifest_id"]
    if value["manifest_id"] != sha256_canonical(manifest_payload):
        raise ValueError(
            "source_manifest.manifest_id does not match its canonical manifest payload"
        )

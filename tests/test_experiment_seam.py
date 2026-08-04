from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import qbitplan.plan as plan_module
from qbitplan import execute_plan
from qbitplan.plan import TOKENIZER_FILE_HASHES, canonical_json_bytes

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


def _record(problem: str) -> dict[str, Any]:
    return {"problem": problem, "solution": "hidden", "level": 1, "type": "Algebra"}


def _prompt(problem: str) -> str:
    return (
        "Problem:\n"
        + problem
        + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
        + r"\boxed{...}."
        + "\nSolution:"
    )


def _with_manifest_id(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["manifest_id"] = _sha256(value)
    return value


def _source_manifest() -> dict[str, Any]:
    records = {
        "train/0.json": _record("training problem"),
        "test/1.json": _record("validation problem"),
    }
    identity = {
        "dataset": "MATH",
        "source_revision": "985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        "permitted_source_ids": {
            "training": ["train/0.json"],
            "validation": ["test/1.json"],
        },
        "final_source_ids": ["test/final.json"],
    }
    return _with_manifest_id({
        **identity,
        "record_hash_algorithm": "SHA-256(canonical JSON record)",
        "record_id_format": "source-relative POSIX JSON path",
        "record_hashes": {
            "training": {"train/0.json": _sha256(records["train/0.json"])},
            "validation": {"test/1.json": _sha256(records["test/1.json"])},
            "final": {"test/final.json": "0000000000000000000000000000000000000000000000000000000000000000"},
        },
        "counts": {"training": 1, "validation": 1, "final": 1},
        "artifact_id": "1" * 64,
        "raw_artifact": {
            "artifact_id": "1" * 64,
            "source_tree_artifact_id": "2" * 64,
            "source_archive_url": "https://web.archive.org/web/20240101000000id_/https://people.eecs.berkeley.edu/~hendrycks/MATH.tar",
            "source_archive_sha256": "1" * 64,
            "source_layout": "train/**/*.json + test/**/*.json",
            "source_file_count": 2,
            "math500_file": "test.jsonl",
            "math500_file_sha256": "0" * 64,
        },


    })
def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plan(root: Path, *, attempt_id: str = "attempt-0001") -> dict[str, Any]:
    source_manifest = _source_manifest()
    queries = []
    for phase, source_id, problem in (
        ("training", "train/0.json", "training problem"),
        ("validation", "test/1.json", "validation problem"),
    ):
        record = _record(problem)
        queries.append(
            {
                "query_id": source_id,
                "source_id": source_id,
                "source_artifact_id": source_manifest["artifact_id"],
                "record": record,
                "record_hash": _sha256(record),
                "dataset": "MATH",
                "phase": phase,
                "source_revision": source_manifest["source_revision"],
                "prompt": _prompt(problem),
                "permitted": True,
            }
        )
    return {
        "schema_version": "qbitplan.stage1.experiment-plan.v1",
        "artifact_root": str(root),
        "attempt_id": attempt_id,
        "mode": "smoke",
        "model": {
            "identifier": "meta-llama/Llama-3.1-8B",
            "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
            "architecture": "LlamaForCausalLM",
            "layers": 32,
            "dtype": "bfloat16",
        },
        "tokenizer": {
            "identifier": "meta-llama/Llama-3.1-8B",
            "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
            "file_hashes": dict(TOKENIZER_FILE_HASHES),
            "use_fast": True,
            "trust_remote_code": False,
        },
        "software": EXPECTED_SOFTWARE,
        "decoder": {
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
        },
        "runtime": {
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
        },
        "seed": 20260807,
        "determinism": {
            "tf32": False,
            "float32_matmul_precision": "highest",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "gpu_uuid": "test-gpu-uuid",
        "source_manifest": source_manifest,
        "queries": queries,
        "profiles": ["BF16", "00000000", "11111111", "01010101"],
    }


class FakeExecutor:
    evidence_class = "simulated"

    def __init__(
        self,
        failing_profile: str | None = None,
        failing_forward_profile: str | None = None,
    ) -> None:
        self.failing_profile = failing_profile
        self.failing_forward_profile = failing_forward_profile

    def hardware_identity(self) -> dict[str, str]:
        return {"gpu_uuid": "test-gpu-uuid", "gpu_name": "test adapter"}

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        if variant_id == self.failing_profile:
            return {
                "status": "invalid",
                "transform_status": "invalid",
                "forward_status": "not_attempted",
                "reason_code": "TRANSFORM_UNSUPPORTED",
                "transform_reason_code": "TRANSFORM_UNSUPPORTED",
                "forward_reason_code": "TRANSFORM_FAILED",
                "observed_group_prefix": [],
            }
        if variant_id == self.failing_forward_profile:
            return {
                "status": "invalid",
                "transform_status": "complete",
                "forward_status": "invalid",
                "reason_code": "FORWARD_NON_FINITE",
                "transform_reason_code": None,
                "forward_reason_code": "FORWARD_NON_FINITE",
                "observed_group_prefix": [],
            }
        profile_bits = "00000000" if variant_id == "BF16" else variant_id
        return {
            "status": "complete",
            "transform_status": "not_applicable" if variant_id == "BF16" else "complete",
            "forward_status": "complete",
            "reason_code": None,
            "transform_reason_code": "REFERENCE_UNQUANTIZED" if variant_id == "BF16" else None,
            "forward_reason_code": None,
            "observed_group_prefix": [
                {
                    "group_index": group_index,
                    "prefix_bits": "BF16" if variant_id == "BF16" else profile_bits[: group_index + 1],
                }
                for group_index in range(8)
            ],
            "token_count": 1,
        }




def _patch_synthetic_manifest_constants(monkeypatch) -> None:
    monkeypatch.setattr(plan_module, "MATH_SOURCE_ARCHIVE_SHA256", "1" * 64)
    monkeypatch.setattr(plan_module, "MATH500_SHA256", "0" * 64)
    monkeypatch.setattr(plan_module, "MATH_SOURCE_TRAINING_COUNT", 1)
    monkeypatch.setattr(plan_module, "MATH_SOURCE_VALIDATION_COUNT", 1)
    monkeypatch.setattr(plan_module, "MATH_FINAL_COUNT", 1)
    monkeypatch.setattr(plan_module, "MATH_SOURCE_FILE_COUNT", 2)
    monkeypatch.setattr(plan_module, "MATH_SOURCE_TREE_ARTIFACT_ID", "2" * 64)


def test_public_seam_writes_immutable_bundle_with_lineage(tmp_path: Path, monkeypatch) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    bundle = execute_plan(_plan(tmp_path), executor=FakeExecutor())

    assert bundle.path.is_dir()
    assert bundle.run_id
    metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest_id"] == _source_manifest()["manifest_id"]
    assert metadata["evidence_class"] == "simulated"
    assert metadata["claim_scope"] == "executable-path smoke only"
    assert set(metadata["files"]) == {
        "plan.json",
        "run-manifest.json",
        "profile-outcomes.ndjson",
        "group-boundaries.ndjson",
        "bundle.json",
    }
    outcomes = [
        json.loads(line)
        for line in (bundle.path / "profile-outcomes.ndjson").read_text(encoding="utf-8").splitlines()
    ]
    assert len(outcomes) == 8
    assert {row["status"] for row in outcomes} == {"complete"}
    assert all(row["source_manifest_id"] == _source_manifest()["manifest_id"] for row in outcomes)
    assert all(row["evidence_class"] == "simulated" for row in outcomes)
    assert len((bundle.path / "group-boundaries.ndjson").read_text(encoding="utf-8").splitlines()) == 64


def test_profile_transform_failure_is_immutable_and_has_no_substitute(tmp_path: Path, monkeypatch) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    bundle = execute_plan(_plan(tmp_path), executor=FakeExecutor("00000000"))
    outcomes = [
        json.loads(line)
        for line in (bundle.path / "profile-outcomes.ndjson").read_text(encoding="utf-8").splitlines()
    ]

    failed = [row for row in outcomes if row["variant_id"] == "00000000"]
    assert len(failed) == 2
    assert all(row["status"] == "invalid" for row in failed)
    assert all(row["reason_code"] == "TRANSFORM_UNSUPPORTED" for row in failed)
    assert {row["variant_id"] for row in outcomes} == {
        "BF16",
        "00000000",
        "11111111",
        "01010101",
    }


def test_functional_quality_inventory_attempts_all_canonical_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _plan(tmp_path)
    plan["mode"] = "functional-quality"
    plan["profiles"] = [f"{profile_id:08b}" for profile_id in range(256)]

    bundle = execute_plan(plan, executor=FakeExecutor())

    inventory = json.loads((bundle.path / "profile-inventory.json").read_text(encoding="utf-8"))
    expected_profiles = [f"{profile_id:08b}" for profile_id in range(256)]
    assert inventory["profile_ids"] == expected_profiles
    assert inventory["p_exec"] == expected_profiles
    assert inventory["enumeration_evidence_class"] == "analytical"
    assert inventory["outcome_evidence_classes"] == ["simulated"]
    assert inventory["source_manifest_id"] == _source_manifest()["manifest_id"]
    assert inventory["outcome_file"] == "profile-outcomes.ndjson"
    assert inventory["outcome_record_count"] == 2 * 256

    bundle_metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    assert bundle_metadata["claim_scope"] == "executable profile feasibility inventory only"
    assert bundle_metadata["non_evidentiary"] is False
    assert "profile-inventory.json" in bundle_metadata["files"]
    artifact_index = json.loads((bundle.path / "artifact-index.json").read_text(encoding="utf-8"))
    assert "profile-inventory.json" in artifact_index["file_hashes"]

    outcomes = [
        json.loads(line)
        for line in (bundle.path / "profile-outcomes.ndjson").read_text(encoding="utf-8").splitlines()
    ]
    assert len(outcomes) == 2 * 256
    assert {row["variant_id"] for row in outcomes} == set(expected_profiles)
    assert len((bundle.path / "group-boundaries.ndjson").read_text(encoding="utf-8").splitlines()) == 2 * 256 * 8


def test_functional_quality_inventory_excludes_only_failed_profile(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _plan(tmp_path)
    plan["mode"] = "functional-quality"
    plan["profiles"] = [f"{profile_id:08b}" for profile_id in range(256)]

    bundle = execute_plan(plan, executor=FakeExecutor("00000000"))

    inventory = json.loads((bundle.path / "profile-inventory.json").read_text(encoding="utf-8"))
    assert "00000000" not in inventory["p_exec"]
    assert len(inventory["p_exec"]) == 255
    assert inventory["excluded_profiles"] == [
        {
            "profile_id": "00000000",
            "reason_code": "TRANSFORM_UNSUPPORTED",
        }
    ]


def test_functional_quality_inventory_excludes_forward_failure_without_substitute(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _plan(tmp_path)
    plan["mode"] = "functional-quality"
    plan["profiles"] = [f"{profile_id:08b}" for profile_id in range(256)]

    bundle = execute_plan(plan, executor=FakeExecutor(failing_forward_profile="00000000"))

    inventory = json.loads((bundle.path / "profile-inventory.json").read_text(encoding="utf-8"))
    assert "00000000" not in inventory["p_exec"]
    assert {
        tuple(sorted(item.items()))
        for item in inventory["excluded_profiles"]
    } == {
        (("profile_id", "00000000"), ("reason_code", "FORWARD_NON_FINITE")),
    }
    failed_outcomes = [
        json.loads(line)
        for line in (bundle.path / "profile-outcomes.ndjson").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["variant_id"] == "00000000"
    ]
    assert len(failed_outcomes) == 2
    assert all(row["forward_status"] == "invalid" for row in failed_outcomes)


def test_plan_rejects_unavailable_defaults_before_adapter_execution(tmp_path: Path, monkeypatch) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _plan(tmp_path)
    del plan["runtime"]["padding"]
    executor = FakeExecutor()

    try:
        execute_plan(plan, executor=executor)
    except ValueError as exc:
        assert "runtime" in str(exc) and "padding" in str(exc)
    else:
        raise AssertionError("missing runtime control was accepted")

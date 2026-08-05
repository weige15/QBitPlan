import json
import subprocess
import sys
from pathlib import Path

import pytest

from qbitplan.stage1 import (
    ArtifactBundle,
    ExperimentPlan,
    ProfileExecutionResult,
    execute_plan,
)
from qbitplan.stage1.contract import content_hash


class FakeSmokeExecutor:
    evidence_class = "simulated"

    def execute(self, profile_id: str, query: dict[str, str]) -> ProfileExecutionResult:
        return ProfileExecutionResult(
            profile_id=profile_id,
            query_id=query["query_id"],
            transform_status="complete",
            forward_status="complete",
            terminal_status="complete",
            executable=True,
            reason_code="completed",
            evidence_class=self.evidence_class,
            output_digest=f"{profile_id}:{query['query_id']}",
        )


class FailingSmokeExecutor(FakeSmokeExecutor):
    def execute(self, profile_id: str, query: dict[str, str]) -> ProfileExecutionResult:
        if profile_id == "00000000":
            return ProfileExecutionResult(
                profile_id=profile_id,
                query_id=query["query_id"],
                transform_status="failed",
                forward_status="not_run",
                terminal_status="invalid",
                executable=False,
                reason_code="unsupported_transform",
                evidence_class=self.evidence_class,
                detail="test adapter rejected the profile transform",
            )
        return super().execute(profile_id, query)


def source_manifest() -> dict[str, object]:
    identity = {
        "dataset": "MATH",
        "source_revision": "985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        "permitted_source_ids": {
            "training": ["math-train-0001"],
            "validation": ["math-val-0001"],
        },
        "final_source_ids": ["math-final-0001"],
    }
    return {**identity, "artifact_id": content_hash(identity)}


def math_query(query_id: str, phase: str, problem: str) -> dict[str, object]:
    record = {
        "problem": problem,
        "solution": "The answer is \\boxed{0}.",
        "level": "Level 1",
        "type": "Algebra",
    }
    prompt = (
        "Problem:\n"
        + problem
        + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
        + "\\boxed{...}.\nSolution:"
    )
    return {
        "query_id": query_id,
        "source_id": query_id,
        "source_artifact_id": source_manifest()["artifact_id"],
        "record": record,
        "record_hash": content_hash(record),
        "dataset": "MATH",
        "phase": phase,
        "source_revision": "985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        "prompt": prompt,
        "permitted": True,
    }


def smoke_plan(artifact_root: Path) -> ExperimentPlan:

    return ExperimentPlan.from_mapping(
        {
            "mode": "smoke",
            "artifact_root": str(artifact_root),
            "attempt_id": "attempt-0001",
            "producer_git_sha": "test-git-sha",
            "model": {
                "id": "meta-llama/Llama-3.1-8B",
                "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "architecture": "LlamaForCausalLM",
                "layers": 32,
                "dtype": "bfloat16",
            },
            "tokenizer": {
                "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "id": "meta-llama/Llama-3.1-8B",
                "file_hashes": {"tokenizer.json": "test-tokenizer-hash"},
                "use_fast": True,
                "trust_remote_code": False,
            },
            "backend": {
                "name": "torchao",
                "version": "0.5.0",
                "cuda": "12.4",
                "pytorch": "2.4.0+cu124",
                "transformers": "5.12.1",
                "python": "3.12.3",
                "numpy": "2.1.0",
                "datasets": "5.0.0",
                "accelerate": "1.14.0",
                "safetensors": "0.8.0",
            },
            "hardware": {
                "gpu_uuid": "GPU-test",
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "driver": "580.159.03",
            },
            "groups": {
                "count": 8,
                "layers_per_group": 4,
                "target_projections": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "int4_group_size": 128,
            },
            "profiles": ["bf16", "00000000", "11111111", "01010101"],
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
                "cublas_workspace_config": ":4096:8",
                "float32_matmul_precision": "highest",
                "tf32": False,
                "cudnn_benchmark": False,
                "deterministic_cudnn": True,
                "deterministic_algorithms": True,
            },
            "seeds": {
                "python": 20260807,
                "numpy": 20260807,
                "torch_cpu": 20260807,
                "torch_cuda": 20260807,
            },
            "source_manifest": source_manifest(),
            "queries": [
                math_query("math-train-0001", "training", "1+1=2"),
                math_query("math-val-0001", "validation", "2+2=4"),
            ],
        }
    )


def test_smoke_seam_writes_immutable_bundle_with_all_profile_outcomes(
    tmp_path: Path,
) -> None:
    plan = smoke_plan(tmp_path / "artifacts")

    bundle = execute_plan(plan, FakeSmokeExecutor())
    assert isinstance(bundle, ArtifactBundle)

    assert bundle.bundle_path.exists()
    assert bundle.artifact_id
    assert bundle.manifest_id
    assert bundle.run_id

    payload = json.loads(bundle.bundle_path.read_text(encoding="utf-8"))
    assert payload["claim_boundary"] == "executable-path-only/non-evidentiary-smoke"
    assert payload["evidence_class"] == "analytical"
    assert payload["profile_ids"] == ["bf16", "00000000", "11111111", "01010101"]
    assert len(payload["execution_results"]) == 8
    assert {row["evidence_class"] for row in payload["execution_results"]} == {
        "simulated"
    }
    assert all(
        row["terminal_status"] == "complete" for row in payload["execution_results"]
    )
    assert payload["lineage"]["manifest_id"] == bundle.manifest_id
    assert payload["lineage"]["run_id"] == bundle.run_id
    execution_results_path = bundle.root / "execution-results.ndjson"
    assert len(execution_results_path.read_text(encoding="utf-8").splitlines()) == 8
    loaded = ArtifactBundle.load(bundle.root)
    assert loaded.artifact_id == bundle.artifact_id

    with pytest.raises(FileExistsError):
        execute_plan(plan, FakeSmokeExecutor())


def test_transform_failure_is_recorded_and_never_substituted(tmp_path: Path) -> None:
    bundle = execute_plan(smoke_plan(tmp_path / "artifacts"), FailingSmokeExecutor())

    failed = [
        row
        for row in bundle.payload["execution_results"]
        if row["profile_id"] == "00000000"
    ]
    assert len(failed) == 2
    assert all(row["transform_status"] == "failed" for row in failed)
    assert all(row["forward_status"] == "not_run" for row in failed)
    assert all(row["terminal_status"] == "invalid" for row in failed)
    assert all(row["executable"] is False for row in failed)
    assert all(row["reason_code"] == "unsupported_transform" for row in failed)
    assert {row["profile_id"] for row in bundle.payload["execution_results"]} == {
        "bf16",
        "00000000",
        "11111111",
        "01010101",
    }


def test_plan_rejects_an_omitted_scientific_control(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(smoke_plan(tmp_path / "artifacts").data))
    del raw["decoder"]["num_beams"]

    with pytest.raises(ValueError, match="decoder controls"):
        ExperimentPlan.from_mapping(raw)


def test_plan_rejects_a_query_not_explicitly_marked_permitted(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(smoke_plan(tmp_path / "artifacts").data))
    raw["queries"][0]["permitted"] = False

    with pytest.raises(ValueError, match="permitted"):
        ExperimentPlan.from_mapping(raw)


def test_plan_rejects_a_query_source_outside_the_manifest(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(smoke_plan(tmp_path / "artifacts").data))
    raw["queries"][0]["source_id"] = "not-permitted"

    with pytest.raises(ValueError, match="not permitted"):
        ExperimentPlan.from_mapping(raw)


def test_scientific_cli_has_no_fake_adapter_selector(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qbitplan",
            "stage1",
            "run",
            "--plan",
            str(tmp_path / "unused-plan.json"),
            "--mode",
            "smoke",
            "--gpu-uuid",
            "GPU-test",
            "--executor",
            "fake",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --executor fake" in result.stderr

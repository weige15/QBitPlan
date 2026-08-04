from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from test_experiment_seam import _patch_synthetic_manifest_constants, _plan, _sha256

from qbitplan import execute_plan
from qbitplan.stage1.quality import FunctionalQualityRunner


class FakeQualityExecutor:
    evidence_class = "simulated"

    def __init__(self, invalid_profile: str | None = None) -> None:
        self.invalid_profile = invalid_profile
        self.diagnostic_calls: list[tuple[str, tuple[int, ...]]] = []

    def hardware_identity(self) -> dict[str, str]:
        return {"gpu_uuid": "test-gpu-uuid", "identity_status": "test-only"}

    def execute(self, query: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
        if variant_id == self.invalid_profile:
            return {
                "status": "invalid",
                "transform_status": "invalid",
                "forward_status": "not_attempted",
                "reason_code": "TRANSFORM_UNSUPPORTED",
                "transform_reason_code": "TRANSFORM_UNSUPPORTED",
                "forward_reason_code": "TRANSFORM_FAILED",
                "observed_group_prefix": [],
            }
        correct = variant_id == "BF16" or query["query_id"].startswith("train/")
        tokens = [11, 22] if variant_id == "BF16" else [33, 44]
        return {
            "status": "complete",
            "transform_status": "not_applicable"
            if variant_id == "BF16"
            else "complete",
            "forward_status": "complete",
            "reason_code": "completed",
            "transform_reason_code": "REFERENCE_UNQUANTIZED"
            if variant_id == "BF16"
            else None,
            "forward_reason_code": None,
            "observed_group_prefix": [
                {
                    "group_index": index,
                    "prefix_bits": "BF16"
                    if variant_id == "BF16"
                    else variant_id[: index + 1],
                }
                for index in range(8)
            ],
            "generated_tokens": tokens,
            "output_text": "\\boxed{hidden}" if correct else "\\boxed{wrong}",
            "output_hash": f"{variant_id}:{query['query_id']}",
        }

    def teacher_forced_diagnostic(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        reference_tokens: list[int],
    ) -> dict[str, Any]:
        del query
        self.diagnostic_calls.append((variant_id, tuple(reference_tokens)))
        return {
            "status": "complete",
            "reason_code": "completed",
            "kl_mean": 0.0 if variant_id == "BF16" else 0.25,
            "evidence_class": self.evidence_class,
        }



@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("A.", "A"),
        ("A)", "A"),
        ("Answer: A", "A"),
        ("Final answer: J", "J"),
        ("Answer:A", None),
        ("Answer:\nA", None),
    ],
)
def test_mmlu_answer_parser_uses_exact_accepted_forms(
    text: str, expected: str | None
) -> None:
    assert FunctionalQualityRunner.parse_mmlu_answer(text) == expected


def _quality_plan(root: Path, monkeypatch) -> Any:
    _patch_synthetic_manifest_constants(monkeypatch)
    raw = _plan(root)
    raw["mode"] = "functional-quality"
    raw["profiles"] = [f"{profile_id:08b}" for profile_id in range(256)]
    for query in raw["queries"]:
        query["record"]["solution"] = "\\boxed{hidden}"
        query["record_hash"] = _sha256(query["record"])
        raw["source_manifest"]["record_hashes"][query["phase"]][query["source_id"]] = (
            query["record_hash"]
        )
    manifest = dict(raw["source_manifest"])
    del manifest["manifest_id"]
    raw["source_manifest"]["manifest_id"] = _sha256(manifest)
    return raw


def test_quality_runner_reports_paired_correctness_degradation_and_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _quality_plan(tmp_path / "artifacts", monkeypatch)
    executor = FakeQualityExecutor(invalid_profile="00000000")

    result = FunctionalQualityRunner(plan, executor).run()

    assert len(result.quality_records) == 2 * 257
    bf16_training = next(
        row
        for row in result.quality_records
        if row["query_id"] == "train/0.json" and row["profile_id"] == "BF16"
    )
    assert bf16_training["correct"] is True
    assert bf16_training["degradation"] == 0
    failed = next(
        row
        for row in result.quality_records
        if row["query_id"] == "train/0.json" and row["profile_id"] == "00000000"
    )
    assert failed["status"] == "invalid"
    assert "correct" not in failed
    assert failed["reason_code"] == "TRANSFORM_UNSUPPORTED"

    profile_training = next(
        row
        for row in result.quality_records
        if row["query_id"] == "train/0.json" and row["profile_id"] == "00000001"
    )
    assert profile_training["correct"] is True
    assert profile_training["degradation"] == 0
    profile_validation = next(
        row
        for row in result.quality_records
        if row["query_id"] == "test/1.json" and row["profile_id"] == "00000001"
    )
    assert profile_validation["correct"] is False
    assert profile_validation["degradation"] == 1

    diagnostics = [
        row for row in result.diagnostic_records if row["query_id"] == "train/0.json"
    ]
    assert len(diagnostics) == 257
    assert sum(row["status"] == "complete" for row in diagnostics) == 256
    assert (
        next(row for row in diagnostics if row["profile_id"] == "00000000")[
            "reason_code"
        ]
        == "PROFILE_EXECUTION_INVALID"
    )
    assert all(
        row["reference_token_count"] == 2
        for row in diagnostics
        if row["status"] == "complete"
    )
    assert all(call[1] == (11, 22) for call in executor.diagnostic_calls)
    assert "00000000" not in {call[0] for call in executor.diagnostic_calls}


def test_quality_runner_rejects_final_only_query_before_execution(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _quality_plan(tmp_path / "artifacts", monkeypatch)
    plan["queries"][0]["source_id"] = "test/final.json"
    plan["queries"][0]["query_id"] = "test/final.json"
    executor = FakeQualityExecutor()

    with pytest.raises(ValueError, match="final-only"):
        FunctionalQualityRunner(plan, executor)


def test_quality_runner_supports_strict_mmlu_option_parsing() -> None:
    assert FunctionalQualityRunner.parse_mmlu_answer("A") == "A"
    assert FunctionalQualityRunner.parse_mmlu_answer("Answer: J") == "J"
    assert FunctionalQualityRunner.parse_mmlu_answer("A.\n") == "A"
    assert FunctionalQualityRunner.parse_mmlu_answer("A and explanation") is None


def test_public_seam_writes_quality_artifacts(tmp_path: Path, monkeypatch) -> None:
    plan = _quality_plan(tmp_path / "artifacts", monkeypatch)
    bundle = execute_plan(plan, executor=FakeQualityExecutor())

    metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    assert metadata["quality_claim_scope"].startswith(
        "pinned functional-quality evidence"
    )
    assert {
        "quality-outcomes.ndjson",
        "quality-diagnostics.ndjson",
        "quality-summary.json",
    }.issubset(metadata["files"])
    assert (
        len(
            (bundle.path / "quality-outcomes.ndjson")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2 * 257
    )
    assert (
        len(
            (bundle.path / "quality-diagnostics.ndjson")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2 * 257
    )
    assert (
        "quality-summary.json"
        in json.loads(
            (bundle.path / "artifact-index.json").read_text(encoding="utf-8")
        )["file_hashes"]
    )
    summary = json.loads(
        (bundle.path / "quality-summary.json").read_text(encoding="utf-8")
    )
    assert summary["source_artifact_ids"] == [plan["source_manifest"]["artifact_id"]]
    assert summary["record_count"] == 2 * 257
    assert isinstance(summary["configuration_hash"], str)


def test_public_seam_preserves_incomplete_and_aborted_statuses(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _quality_plan(tmp_path / "artifacts", monkeypatch)
    executor = FakeQualityExecutor()
    original_execute = executor.execute

    def terminal_execute(query: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
        if variant_id == "00000000":
            observation = original_execute(query, variant_id)
            observation.update(
                status="incomplete",
                reason_code="FORWARD_TIMEOUT",
                forward_reason_code="FORWARD_TIMEOUT",
            )
            return observation
        if variant_id == "00000001":
            observation = original_execute(query, variant_id)
            observation.update(
                status="aborted",
                reason_code="RUN_ABORTED",
                forward_status="not_attempted",
                forward_reason_code="RUN_ABORTED",
            )
            return observation
        return original_execute(query, variant_id)

    monkeypatch.setattr(executor, "execute", terminal_execute)
    bundle = execute_plan(plan, executor=executor)
    outcomes = [
        json.loads(line)
        for line in (bundle.path / "profile-outcomes.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    statuses = {
        row["variant_id"]: row["status"]
        for row in outcomes
        if row["query_id"] == "train/0.json"
    }
    assert statuses["00000000"] == "incomplete"
    assert statuses["00000001"] == "aborted"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import qbitplan.plan as plan_module
from qbitplan.config import accepted_smoke_configuration
from qbitplan.identity import sha256_canonical
from qbitplan.plan import (
    MATH_PROFILE_INVENTORY_SOURCE_ARTIFACT_ID,
    MATH_PROFILE_INVENTORY_SOURCE_MANIFEST_ID,
    TOKENIZER_FILE_HASHES,
    ExperimentPlan,
    mmlu_prompt,
)


def _manifest(monkeypatch) -> tuple[dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(plan_module, "MMLU_PRO_TEST_COUNT", 1)
    monkeypatch.setattr(plan_module, "MMLU_PRO_TEST_PARQUET_SHA256", "1" * 64)
    record = {
        "question_id": 70,
        "question": "Which option is correct?",
        "options": ["wrong", "right", "other"],
        "answer": "B",
        "answer_index": 1,
        "cot_content": "",
        "category": "test",
        "src": "synthetic",
    }
    record_hash = sha256_canonical(record)
    manifest = {
        "dataset": "MMLU-Pro",
        "source_revision": plan_module.MMLU_PRO_SOURCE_REVISION,
        "permitted_source_ids": {"training": [], "validation": []},
        "final_source_ids": ["70"],
        "record_hash_algorithm": "SHA-256(canonical JSON record)",
        "record_id_format": "MMLU-Pro question_id decimal string",
        "record_hashes": {
            "training": {},
            "validation": {},
            "final": {"70": record_hash},
        },
        "counts": {"training": 0, "validation": 0, "final": 1},
        "raw_artifact": {
            "artifact_id": "1" * 64,
            "source_repo": plan_module.MMLU_PRO_SOURCE_REPO,
            "source_path": plan_module.MMLU_PRO_TEST_PARQUET,
            "source_sha256": "1" * 64,
            "source_split": plan_module.MMLU_PRO_SOURCE_SPLIT,
            "source_rows": 1,
        },
        "artifact_id": "1" * 64,
    }
    manifest["manifest_id"] = sha256_canonical(manifest)
    return manifest, record


def test_pinned_manifest_is_present() -> None:
    manifest = json.loads(
        Path("data/manifests/mmlu-pro-test.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"] == "MMLU-Pro"
    assert manifest["source_revision"] == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert manifest["counts"] == {"training": 0, "validation": 0, "final": 12032}
    assert len(manifest["final_source_ids"]) == 12032
    assert manifest["final_source_ids"] == sorted(manifest["final_source_ids"], key=int)


def test_final_only_mmlu_plan_validates_with_source_faithful_options(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, record = _manifest(monkeypatch)
    plan = accepted_smoke_configuration(
        artifact_root=str(tmp_path), attempt_id="mmlu-test", gpu_uuid="test-gpu"
    )
    plan.update(
        {
            "mode": "functional-quality",
            "profiles": ["00000000"],
            "profile_inventory": {
                "artifact_id": "2" * 64,
                "artifact_type": "profile-execution-inventory",
                "schema_version": "qbitplan.stage1.profile-inventory.v1",
                "source_manifest_id": MATH_PROFILE_INVENTORY_SOURCE_MANIFEST_ID,
                "source_artifact_id": MATH_PROFILE_INVENTORY_SOURCE_ARTIFACT_ID,
                "profile_ids": [f"{profile_id:08b}" for profile_id in range(256)],
                "record_count": 256,
                "p_exec": ["00000000"],
                "outcome_record_count": 512,
                "outcome_file": "profile-outcomes.ndjson",
                "claim_scope": "executable profile feasibility only",
            },
            "source_manifest": manifest,
            "queries": [
                {
                    "query_id": "70",
                    "source_id": "70",
                    "source_artifact_id": manifest["artifact_id"],
                    "record": record,
                    "record_hash": sha256_canonical(record),
                    "dataset": "MMLU-Pro",
                    "phase": "final",
                    "source_revision": manifest["source_revision"],
                    "prompt": mmlu_prompt(record["question"], record["options"]),
                    "permitted": True,
                }
            ],
        }
    )
    validated = ExperimentPlan.from_mapping(plan)
    assert validated.data["queries"][0]["phase"] == "final"
    assert validated.data["tokenizer"]["file_hashes"] == TOKENIZER_FILE_HASHES

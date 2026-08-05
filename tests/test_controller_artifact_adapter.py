from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qbitplan.stage1.controller import (
    Stage1ControllerArtifactAdapter,
    profile_id,
)

_SOURCE_MANIFEST_ID = "a" * 64
_ARTIFACT_ID = "b" * 64


def _artifact_payloads() -> tuple[dict, dict, dict, dict]:
    profile = {
        "artifact_type": "profile-execution-inventory",
        "schema_version": "qbitplan.stage1.profile-inventory.v1",
        "artifact_id": _ARTIFACT_ID,
        "source_manifest_id": _SOURCE_MANIFEST_ID,
        "profile_ids": ["00", "01", "10"],
        "p_exec": ["00", "01"],
        "record_count": 3,
        "outcome_record_count": 6,
        "outcome_file": "profile-outcomes.ndjson",
        "claim_scope": "executable profile feasibility only",
    }
    features = {
        "artifact_type": "stage1-query-features",
        "schema_version": "qbitplan.stage1.query-features.v1",
        "artifact_id": "c" * 64,
        "source_manifest_id": _SOURCE_MANIFEST_ID,
        "records": [
            {
                "query_id": "train/q0",
                "dataset": "MATH",
                "phase": "training",
                "structural": [1.0, 2.0],
                "embedding": [3.0, 4.0],
            },
            {
                "query_id": "validation/q1",
                "dataset": "MATH",
                "phase": "validation",
                "structural": [5.0, 6.0],
                "embedding": [0.0, 0.0],
            },
        ],
    }
    targets = {
        "artifact_type": "stage1-target-sets",
        "schema_version": "qbitplan.stage1.target-sets.v1",
        "artifact_id": "d" * 64,
        "source_manifest_id": _SOURCE_MANIFEST_ID,
        "records": [
            {
                "query_id": "train/q0",
                "dataset": "MATH",
                "phase": "training",
                "target_profile_ids": ["00"],
            },
            {
                "query_id": "validation/q1",
                "dataset": "MATH",
                "phase": "validation",
                "target_profile_ids": [],
            },
        ],
    }
    prefixes = {
        "artifact_type": "stage1-causal-prefixes",
        "schema_version": "qbitplan.stage1.causal-prefixes.v1",
        "artifact_id": "e" * 64,
        "source_manifest_id": _SOURCE_MANIFEST_ID,
        "records": [
            {
                "query_id": "train/q0",
                "dataset": "MATH",
                "phase": "training",
                "prefix_profile_id": "0",
                "hidden_summary": [9.0, 12.0],
            }
        ],
    }
    return profile, features, targets, prefixes


def test_adapter_maps_stage1_artifacts_to_controller_types() -> None:
    adapter = Stage1ControllerArtifactAdapter.from_mappings(*_artifact_payloads())

    assert adapter.executable_profiles == ((4, 4), (4, 8))
    examples = adapter.training_queries
    assert len(examples) == 1
    assert examples[0].query_id == "train/q0"
    assert examples[0].target_profiles == ((4, 4),)
    assert examples[0].features.structural.tolist() == [1.0, 2.0]
    assert np.allclose(examples[0].features.embedding, [0.6, 0.8])

    context = adapter.context_provider
    assert np.allclose(context("train/q0", (4,)), [9.0, 12.0])


def test_adapter_preserves_empty_targets_and_rejects_context_fallback() -> None:
    adapter = Stage1ControllerArtifactAdapter.from_mappings(*_artifact_payloads())

    validation = adapter.queries(phase="validation")
    assert len(validation) == 1
    assert validation[0].target_profiles == ()
    with pytest.raises(KeyError, match="causal prefix artifact"):
        adapter.context_provider("validation/q1", (4,))


def test_adapter_rejects_final_queries_and_unknown_target_profiles() -> None:
    profile, features, targets, prefixes = _artifact_payloads()
    features["records"][0]["phase"] = "final"
    with pytest.raises(ValueError, match="final"):
        Stage1ControllerArtifactAdapter.from_mappings(
            profile, features, targets, prefixes
        )

    profile, features, targets, prefixes = _artifact_payloads()
    targets["records"][0]["target_profile_ids"] = ["10"]
    with pytest.raises(ValueError, match="executable"):
        Stage1ControllerArtifactAdapter.from_mappings(
            profile, features, targets, prefixes
        )


def test_adapter_rejects_mismatched_lineage_and_duplicate_records() -> None:
    profile, features, targets, prefixes = _artifact_payloads()
    features["source_manifest_id"] = "f" * 64
    with pytest.raises(ValueError, match="source_manifest_id"):
        Stage1ControllerArtifactAdapter.from_mappings(
            profile, features, targets, prefixes
        )

    profile, features, targets, prefixes = _artifact_payloads()
    targets["records"].append(dict(targets["records"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        Stage1ControllerArtifactAdapter.from_mappings(
            profile, features, targets, prefixes
        )


def test_adapter_loads_canonical_json_artifact_files(tmp_path: Path) -> None:
    payloads = _artifact_payloads()
    paths = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"artifact-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    adapter = Stage1ControllerArtifactAdapter.from_paths(*paths)

    assert profile_id(adapter.executable_profiles[1]) == "01"
    assert tuple(query.query_id for query in adapter.queries()) == (
        "train/q0",
        "validation/q1",
    )

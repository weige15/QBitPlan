from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_experiment_seam import _patch_synthetic_manifest_constants, _plan

from qbitplan import execute_plan
from qbitplan.identity import sha256_canonical
from qbitplan.stage1.lookup import LookupCostEstimateAdapter

DIMENSIONS = (
    "resident_accelerator_bytes",
    "host_to_device_bytes",
    "latency",
    "prefetch_stall_time",
    "kernel_switch_count",
    "controller_probe_feedback_overhead",
)


def _estimate_plan(root: Path) -> dict:
    smoke_plan = _plan(root)
    return {
        "schema_version": smoke_plan["schema_version"],
        "artifact_root": smoke_plan["artifact_root"],
        "attempt_id": smoke_plan["attempt_id"],
        "mode": "estimate-cost",
        "source_manifest": smoke_plan["source_manifest"],
        "queries": smoke_plan["queries"],
        "profiles": ["00000000", "11111111"],
    }


def _coverage_manifest() -> dict:
    payload = {
        "schema_version": "qbitplan.stage1.lookup-cost-coverage.v1",
        "query_ids": ["train/0.json", "test/1.json"],
        "profile_ids": ["00000000"],
        "dimensions": ["latency", "resident_accelerator_bytes"],
    }
    return {**payload, "manifest_id": sha256_canonical(payload)}


def _lookup_table() -> dict:
    return {
        "schema_version": "qbitplan.stage1.lookup-cost-table.v1",
        "method": "declared hardware-calibrated lookup",
        "evidence_class": "lookup-table estimated",
        "source_artifact": {
            "artifact_id": "source-calibration-artifact",
            "location": "calibration/cost-run.json",
        },
        "coverage_manifest": _coverage_manifest(),
        "entries": [
            {
                "query_id": "train/0.json",
                "profile_id": "00000000",
                "costs": {"latency": 17, "resident_accelerator_bytes": 4096},
            }
        ],
    }


def test_lookup_estimate_bundle_preserves_coverage_and_omissions(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    table_path = tmp_path / "lookup-table.json"
    table_path.write_text(json.dumps(_lookup_table()), encoding="utf-8")

    plan = _estimate_plan(tmp_path / "artifacts")
    adapter = LookupCostEstimateAdapter.from_path(table_path)
    bundle = execute_plan(plan, executor=adapter)

    metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    assert metadata["evidence_class"] == "lookup-table estimated"
    assert metadata["claim_scope"] == "lookup-table estimated cost comparisons only"
    assert metadata["evidence_boundary"]["systems_benefit"] == (
        "omitted/unavailable/lookup-table-not-direct-measurement"
    )

    records = [
        json.loads(line)
        for line in (bundle.path / "cost-estimates.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 4
    covered = next(row for row in records if row["query_id"] == "train/0.json")
    assert covered["profile_id"] == "00000000"
    assert covered["cost_vector"]["latency"]["status"] == "estimated"
    assert covered["cost_vector"]["latency"]["value"] == 17
    assert (
        covered["cost_vector"]["host_to_device_bytes"]["status"]
        == "omitted/unavailable"
    )
    assert covered["cost_vector"]["host_to_device_bytes"]["reason"]
    assert covered["evidence_class"] == "lookup-table estimated"

    uncovered = next(row for row in records if row["profile_id"] == "11111111")
    assert all(
        dimension["status"] == "omitted/unavailable"
        for dimension in uncovered["cost_vector"].values()
    )
    assert all(
        dimension.get("value") != 0
        for row in records
        for dimension in row["cost_vector"].values()
        if dimension["status"] == "omitted/unavailable"
    )


def test_lookup_adapter_rejects_entries_outside_declared_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    table = _lookup_table()
    table["entries"].append(
        {
            "query_id": "test/1.json",
            "profile_id": "11111111",
            "costs": {"latency": 1},
        }
    )
    table_path = tmp_path / "out-of-coverage.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the declared coverage manifest"):
        LookupCostEstimateAdapter.from_path(table_path)


def test_lookup_adapter_rejects_a_directly_measured_declaration(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    table = _lookup_table()
    table["evidence_class"] = "directly measured"
    table_path = tmp_path / "direct-table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")

    try:
        LookupCostEstimateAdapter.from_path(table_path)
    except ValueError as exc:
        assert "lookup-table estimated" in str(exc)
    else:
        raise AssertionError("directly measured lookup table was accepted")

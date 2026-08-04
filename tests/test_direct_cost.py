from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import pytest
from test_experiment_seam import _patch_synthetic_manifest_constants, _plan

from qbitplan import execute_plan
from qbitplan.stage1.direct_cost import DirectCostRunner

DIMENSIONS = (
    "resident_accelerator_bytes",
    "host_to_device_bytes",
    "latency",
    "prefetch_stall_time",
    "kernel_switch_count",
    "controller_probe_feedback_overhead",
)
T = TypeVar("T")


def _direct_cost_plan(root: Path) -> dict[str, Any]:
    plan = _plan(root)
    plan["mode"] = "direct-cost"
    return plan


class FakeProfileExecutor:
    evidence_class = "directly measured"

    def __init__(self, failing_profile: str | None = None) -> None:
        self.failing_profile = failing_profile
        self.prepared: list[str] = []
        self.forward_calls: list[tuple[str, str]] = []

    def hardware_identity(self) -> Mapping[str, Any]:
        return {
            "identity_status": "complete",
            "gpu_uuid": "test-gpu-uuid",
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "driver_version": "580.159.03",
        }

    def prepare_variant(self, variant_id: str) -> Mapping[str, Any]:
        self.prepared.append(variant_id)
        if variant_id == self.failing_profile:
            return {
                "status": "invalid",
                "reason_code": "TRANSFORM_UNSUPPORTED",
                "model_load_ns": 11,
                "quantization_ns": 13,
            }
        return {
            "status": "complete",
            "reason_code": "PREPARED",
            "model_load_ns": 11,
            "quantization_ns": 13 if variant_id != "BF16" else None,
        }

    def execute_prepared(
        self, query: Mapping[str, Any], variant_id: str
    ) -> Mapping[str, Any]:
        self.forward_calls.append((query["query_id"], variant_id))
        return {
            "status": "complete",
            "transform_status": "not_applicable"
            if variant_id == "BF16"
            else "complete",
            "forward_status": "complete",
            "reason_code": None,
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
        }


class FakeMemoryProbe:
    def measure(
        self, operation: Callable[[], T], **_: Any
    ) -> tuple[T, Mapping[str, Any]]:
        result = operation()
        return result, {
            "status": "measured",
            "peak_device_used_bytes": 123456,
            "baseline_device_used_bytes": 120000,
            "sample_count": 2,
        }


class FakeTraceProbe:
    def trace(
        self, operation: Callable[[], T], **_: Any
    ) -> tuple[T, Mapping[str, Any]]:
        result = operation()
        return result, {
            "status": "complete",
            "host_to_device_bytes": 4096,
            "prefetch_stall_time_ns": 17,
            "kernel_switch_count": 3,
            "events": [{"name": "test.cuda.copy", "bytes": 4096}],
        }


def test_direct_cost_runner_uses_shared_executor_and_writes_measured_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")
    executor = FakeProfileExecutor()
    runner = DirectCostRunner(
        plan,
        executor,
        memory_probe=FakeMemoryProbe(),
        trace_probe=FakeTraceProbe(),
    )

    bundle = execute_plan(plan, executor=runner)

    metadata = json.loads((bundle.path / "bundle.json").read_text(encoding="utf-8"))
    assert (
        metadata["claim_scope"] == "directly measured cost evidence; smoke frame only"
    )
    assert metadata["non_evidentiary"] is True
    assert metadata["evidence_boundary"]["quality"] == "omitted"
    assert metadata["evidence_boundary"]["systems_benefit"] == (
        "omitted/unavailable/direct-cost-smoke"
    )
    assert set(metadata["files"]) == {
        "plan.json",
        "run-manifest.json",
        "setup-observations.ndjson",
        "cost-observations.ndjson",
        "trace-observations.ndjson",
        "cost-coverage.ndjson",
        "bundle.json",
        "artifact-index.json",
    }

    observations = [
        json.loads(line)
        for line in (bundle.path / "cost-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(observations) == 8
    assert all(row["evidence_class"] == "directly measured" for row in observations)
    assert all(row["repetitions"]["warmup_count"] == 5 for row in observations)
    assert all(row["repetitions"]["measured_count"] == 10 for row in observations)
    assert all(
        row["cost_vector"][dimension]["evidence_class"] == "directly measured"
        for row in observations
        for dimension in DIMENSIONS[:5]
    )
    assert all(
        row["cost_vector"]["controller_probe_feedback_overhead"]["status"]
        == "omitted/unavailable"
        for row in observations
    )

    traces = [
        json.loads(line)
        for line in (bundle.path / "trace-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(traces) == 8
    assert all(row["evidence_class"] == "directly measured" for row in traces)
    assert all(row["raw_trace"]["events"] for row in traces)
    assert len(executor.prepared) == 4
    assert len(executor.forward_calls) == 4 * 2 * (5 + 10 + 1)


def test_direct_cost_omits_uncovered_dimensions_without_zero_or_lookup_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")

    class IncompleteTraceProbe(FakeTraceProbe):
        def trace(
            self, operation: Callable[[], T], **_: Any
        ) -> tuple[T, Mapping[str, Any]]:
            result = operation()
            return result, {"status": "invalid", "reason_code": "TRACE_UNAVAILABLE"}

    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            FakeProfileExecutor(),
            memory_probe=FakeMemoryProbe(),
            trace_probe=IncompleteTraceProbe(),
        ),
    )

    observations = [
        json.loads(line)
        for line in (bundle.path / "cost-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in observations:
        assert row["cost_vector"]["latency"]["status"] == "measured"
        for dimension in (
            "host_to_device_bytes",
            "prefetch_stall_time",
            "kernel_switch_count",
            "controller_probe_feedback_overhead",
        ):
            assert row["cost_vector"][dimension]["status"] == "omitted/unavailable"
            assert row["cost_vector"][dimension].get("value") != 0
            assert (
                row["cost_vector"][dimension]["evidence_class"] == "directly measured"
            )


def test_direct_cost_profile_failure_is_recorded_without_substitution(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")
    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            FakeProfileExecutor(failing_profile="00000000"),
            memory_probe=FakeMemoryProbe(),
            trace_probe=FakeTraceProbe(),
        ),
    )

    rows = [
        json.loads(line)
        for line in (bundle.path / "cost-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failed = [row for row in rows if row["profile_id"] == "00000000"]
    assert len(failed) == 2
    assert all(row["terminal_status"] == "invalid" for row in failed)
    assert all(row["reason_code"] == "TRANSFORM_UNSUPPORTED" for row in failed)
    assert {row["profile_id"] for row in rows} == {
        "BF16",
        "00000000",
        "11111111",
        "01010101",
    }


def test_direct_cost_plan_and_cli_require_real_path_without_fake_selector(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    from qbitplan.plan import ExperimentPlan

    plan = ExperimentPlan.from_mapping(_direct_cost_plan(tmp_path / "artifacts"))
    assert plan.data["mode"] == "direct-cost"

    from qbitplan.cli import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "stage1",
                "run",
                "--plan",
                "plan.json",
                "--mode",
                "direct-cost",
                "--smoke",
                "--gpu-uuid",
                "GPU-test",
                "--executor",
                "fake",
            ]
        )
    assert error.value.code == 2

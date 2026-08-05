from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, TypeVar

import pytest
from test_experiment_seam import _patch_synthetic_manifest_constants, _plan

from qbitplan import execute_plan
from qbitplan.stage1.direct_cost import (
    CudaTraceProbe,
    DirectCostRunner,
    MemoryProbeOperationError,
)
from qbitplan.stage1.executor import TorchAOProfileExecutor

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
        self.trace_calls: list[bool] = []

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
        self,
        query: Mapping[str, Any],
        variant_id: str,
        *,
        trace: bool = False,
    ) -> Mapping[str, Any]:
        self.forward_calls.append((query["query_id"], variant_id))
        self.trace_calls.append(trace)
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
    run_manifest = json.loads(
        (bundle.path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["measurement_scope"]["host_to_device_bytes"] == (
        "trace-derived direct measurement"
    )
    assert run_manifest["measurement_scope"]["prefetch_stall_time"] == (
        "trace-derived direct measurement"
    )
    assert run_manifest["measurement_scope"]["kernel_switch_count"] == (
        "trace-derived direct measurement"
    )
    assert run_manifest["trace_attempted_count"] == 8
    assert run_manifest["trace_completed_count"] == 8
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
    assert executor.trace_calls.count(False) == 4 * 2 * (5 + 10)
    assert executor.trace_calls.count(True) == 4 * 2


def test_cuda_trace_probe_reports_unavailable_without_untraced_operation(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    calls: list[str] = []

    def operation() -> dict[str, str]:
        calls.append("operation")
        return {"status": "complete"}

    result, trace = CudaTraceProbe().trace(operation)

    assert calls == []
    assert result["status"] == "invalid"
    assert trace == {
        "status": "invalid",
        "reason_code": "CUDA_TRACE_UNAVAILABLE",
    }


def test_cuda_trace_probe_extracts_annotated_copy_and_group_transitions(
    monkeypatch,
) -> None:
    class FakeEvent:
        def __init__(
            self,
            name: str,
            start: float,
            end: float,
            device_memory_usage: int = 0,
        ) -> None:
            self.name = name
            self.cpu_time_total = end - start
            self.self_cpu_time_total = end - start
            self.device_time_total = 1.5
            self.self_device_time_total = 1.5
            self.device_memory_usage = device_memory_usage
            self.time_range = SimpleNamespace(start=start, end=end)

    class FakeProfiler:
        def __init__(self, **_: Any) -> None:
            self._events = [
                FakeEvent("Memcpy HtoD (Pageable -> Device)", 10.0, 20.0),
                FakeEvent("qbitplan.group.0.0", 30.0, 40.0),
                FakeEvent("qbitplan.group.1.1", 50.0, 60.0),
                FakeEvent("qbitplan.group.2.0", 70.0, 80.0),
            ]

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def step(self) -> None:
            return None

        def events(self) -> list[FakeEvent]:
            return self._events

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
        ),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=FakeProfiler,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result, trace = CudaTraceProbe().trace(lambda: {"status": "complete"})

    assert result == {"status": "complete"}
    assert trace["status"] == "complete"
    assert trace["trace_representation"] == "profiler_events"
    assert trace["kernel_switch_count"] == 2
    assert trace["dimension_coverage"]["host_to_device_bytes"] == {
        "status": "omitted/unavailable",
        "reason": "TRACE_H2D_COPY_BYTES_UNAVAILABLE",
        "evidence_class": "analytical",
    }
    assert trace["dimension_coverage"]["prefetch_stall_time"] == {
        "status": "omitted/unavailable",
        "reason": "no-prefetch-path",
        "evidence_class": "analytical",
    }


def test_cuda_trace_probe_omits_malformed_group_timestamps_without_attribute_error(
    monkeypatch,
) -> None:
    class BrokenEvent:
        name = "qbitplan.group.0.0"
        cpu_time_total = 10.0
        device_time_total = 1.0
        device_memory_usage = 0

        @property
        def time_range(self) -> Any:
            raise AttributeError("profiler interval unavailable")

    class ReversedEvent:
        name = "qbitplan.group.1.1"
        cpu_time_total = 10.0
        device_time_total = 1.0
        device_memory_usage = 0
        time_range = SimpleNamespace(start=20.0, end=10.0)

    class FakeProfiler:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def step(self) -> None:
            return None

        def events(self) -> list[Any]:
            return [BrokenEvent(), ReversedEvent()]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
        ),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=lambda **_: FakeProfiler(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result, trace = CudaTraceProbe().trace(lambda: {"status": "complete"})

    assert result == {"status": "complete"}
    assert trace["status"] == "complete"
    assert "CUDA_TRACE_FAILED:AttributeError" not in str(trace)
    assert trace["dimension_coverage"]["kernel_switch_count"] == {
        "status": "omitted/unavailable",
        "reason": "TRACE_GROUP_TIMESTAMP_UNAVAILABLE",
        "evidence_class": "analytical",
    }


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
    manifest = json.loads(
        (bundle.path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["trace_attempted_count"] == 8
    assert manifest["trace_completed_count"] == 0
    for row in observations:
        assert row["repetitions"]["trace_pass_count"] == 0
        assert row["repetitions"]["trace_pass_attempted"] == 1
        assert row["repetitions"]["trace_status"] == "invalid"
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
                row["cost_vector"][dimension]["evidence_class"] == "analytical"
            )


def test_direct_cost_manifest_labels_mixed_trace_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")

    class OneMeasuredTraceProbe(FakeTraceProbe):
        calls = 0

        def trace(
            self, operation: Callable[[], T], **_: Any
        ) -> tuple[T, Mapping[str, Any]]:
            result = operation()
            self.calls += 1
            if self.calls == 1:
                return result, {
                    "status": "complete",
                    "host_to_device_bytes": 4096,
                    "prefetch_stall_time_ns": 17,
                    "kernel_switch_count": 3,
                    "events": [{"name": "test.cuda.copy", "bytes": 4096}],
                }
            return result, {"status": "invalid", "reason_code": "TRACE_UNAVAILABLE"}

    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            FakeProfileExecutor(),
            memory_probe=FakeMemoryProbe(),
            trace_probe=OneMeasuredTraceProbe(),
        ),
    )
    manifest = json.loads(
        (bundle.path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["measurement_scope"]["host_to_device_bytes"].startswith(
        "partially measured (1/8); omitted/unavailable/"
    )
    assert manifest["trace_attempted_count"] == 8
    assert manifest["trace_completed_count"] == 1


def test_direct_cost_preserves_partial_timing_on_measured_forward_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")

    class FailingMeasuredForwardExecutor(FakeProfileExecutor):
        def execute_prepared(
            self,
            query: Mapping[str, Any],
            variant_id: str,
            *,
            trace: bool = False,
        ) -> Mapping[str, Any]:
            result = super().execute_prepared(query, variant_id, trace=trace)
            if variant_id == "BF16" and len(self.forward_calls) >= 6:
                return {
                    **result,
                    "status": "invalid",
                    "forward_status": "invalid",
                    "reason_code": "FORWARD_RUNTIME_FAILURE",
                    "forward_reason_code": "FORWARD_RUNTIME_FAILURE",
                    "observed_group_prefix": [],
                }
            return result

    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            FailingMeasuredForwardExecutor(),
            memory_probe=FakeMemoryProbe(),
            trace_probe=FakeTraceProbe(),
        ),
    )

    rows = [
        json.loads(line)
        for line in (bundle.path / "cost-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["profile_id"] == "BF16"
    ]
    assert len(rows) == 2
    assert all(row["terminal_status"] == "invalid" for row in rows)
    assert sorted(row["repetitions"]["measured_count"] for row in rows) == [0, 1]
    assert max(len(row["repetitions"]["host_timestamps"]) for row in rows) == 1
    assert all(
        len(row["memory_observations"]) == row["repetitions"]["measured_count"]
        for row in rows
    )


def test_direct_cost_preserves_failed_memory_probe_observation(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")

    class FailingMemoryProbe(FakeMemoryProbe):
        def measure(
            self, operation: Callable[[], T], **context: Any
        ) -> tuple[T, Mapping[str, Any]]:
            if context.get("phase") == "setup":
                return super().measure(operation, **context)
            operation()
            raise MemoryProbeOperationError(
                RuntimeError("forward failed"),
                {
                    "status": "measured",
                    "peak_device_used_bytes": 456789,
                    "baseline_device_used_bytes": 450000,
                    "sample_count": 3,
                },
            )

    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            FakeProfileExecutor(),
            memory_probe=FailingMemoryProbe(),
            trace_probe=FakeTraceProbe(),
        ),
    )

    rows = [
        json.loads(line)
        for line in (bundle.path / "cost-observations.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    assert all(row["terminal_status"] == "invalid" for row in rows)
    manifest = json.loads(
        (bundle.path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["trace_attempted_count"] == 0
    assert manifest["trace_completed_count"] == 0
    assert all(row["memory_observations"][0]["peak_device_used_bytes"] == 456789 for row in rows)
    assert all(len(row["repetitions"]["host_timestamps"]) == 1 for row in rows)


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


def test_direct_cost_bf16_oom_is_recorded_without_profile_substitution(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan = _direct_cost_plan(tmp_path / "artifacts")

    class Bf16OomExecutor(FakeProfileExecutor):
        def prepare_variant(self, variant_id: str) -> Mapping[str, Any]:
            if variant_id == "BF16":
                self.prepared.append(variant_id)
                return {
                    "status": "invalid",
                    "reason_code": "OOM",
                    "transform_status": "not_applicable",
                    "forward_status": "not_attempted",
                }
            return super().prepare_variant(variant_id)

    bundle = execute_plan(
        plan,
        executor=DirectCostRunner(
            plan,
            Bf16OomExecutor(),
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
    bf16_rows = [row for row in rows if row["profile_id"] == "BF16"]
    executable_rows = [row for row in rows if row["profile_id"] != "BF16"]
    assert len(bf16_rows) == 2
    assert all(row["terminal_status"] == "invalid" for row in bf16_rows)
    assert all(row["reason_code"] == "OOM" for row in bf16_rows)
    assert all(row["terminal_status"] == "complete" for row in executable_rows)
    assert {row["profile_id"] for row in rows} == {
        "BF16",
        "00000000",
        "11111111",
        "01010101",
    }


def test_real_executor_bf16_oom_reports_failed_setup_without_quantization(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    from qbitplan.plan import ExperimentPlan

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_: Any, **__: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeAutoModel),
    )

    class TestExecutor(TorchAOProfileExecutor):
        def _resolve_device(self) -> dict[str, Any]:
            return {"device_index": 0}

    executor = TestExecutor(ExperimentPlan.from_mapping(_direct_cost_plan(tmp_path)))
    preparation = executor.prepare_variant("BF16")

    assert preparation["status"] == "invalid"
    assert preparation["reason_code"] == "OOM"
    assert preparation["transform_status"] == "not_applicable"
    assert preparation["forward_status"] == "not_attempted"
    assert preparation["failure_phase"] == "model_load"
    assert preparation["preparation_phases"]["phase_status"] == "failed"
    assert preparation["preparation_phases"]["failure_phase"] == "model_load"
    assert "active_phase" not in preparation["preparation_phases"]


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

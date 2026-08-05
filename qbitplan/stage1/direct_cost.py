"""Direct hardware-cost measurement around the shared profile executor."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Any, Protocol, TypeVar, cast

from ..execution import (
    ArtifactBundle,
    _canonical_json_file,
    _canonical_ndjson,
    _configuration_hash,
    _lineage,
    _now,
    _producer_git_sha,
    _write_once,
)
from ..identity import sha256_canonical
from ..plan import COST_DIMENSIONS, ExperimentPlan

DIRECT_EVIDENCE_CLASS = "directly measured"
OMISSION_EVIDENCE_CLASS = "analytical"
WARMUP_COUNT = 5
MEASURED_COUNT = 10
TRACE_PASS_COUNT = 1
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.001
T = TypeVar("T")


class PreparedProfileExecutor(Protocol):
    """Public lifecycle required by direct measurement."""

    evidence_class: str

    def hardware_identity(self) -> Mapping[str, Any]: ...

    def prepare_variant(self, variant_id: str) -> Mapping[str, Any]: ...

    def execute_prepared(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        *,
        trace: bool = False,
    ) -> Mapping[str, Any]: ...


class MemoryProbeOperationError(RuntimeError):
    """Preserve probe metadata when the measured operation itself fails."""

    def __init__(
        self, cause: Exception, observation: Mapping[str, Any]
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.observation = dict(observation)


class MemoryProbe(Protocol):
    def measure(
        self, operation: Callable[[], T], **context: Any
    ) -> tuple[T, Mapping[str, Any]]: ...


class TraceProbe(Protocol):
    def trace(
        self, operation: Callable[[], T], **context: Any
    ) -> tuple[T, Mapping[str, Any]]: ...


def _omitted(reason: str) -> dict[str, Any]:
    return {
        "status": "omitted/unavailable",
        "reason": reason,
        "evidence_class": OMISSION_EVIDENCE_CLASS,
    }


def _measured(value: float, *, unit: str, method: str) -> dict[str, Any]:
    return {
        "status": "measured",
        "value": value,
        "unit": unit,
        "method": method,
        "evidence_class": DIRECT_EVIDENCE_CLASS,
    }


def _valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _execution_status(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "status",
        "transform_status",
        "forward_status",
        "reason_code",
        "transform_reason_code",
        "forward_reason_code",
        "observed_group_prefix",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"prepared executor observation is missing fields: {missing}")
    if value["status"] not in {"complete", "invalid"}:
        raise ValueError("prepared executor returned an unsupported terminal status")
    if value["transform_status"] not in {"complete", "invalid", "not_applicable"}:
        raise ValueError("prepared executor returned an unsupported transform status")
    if value["forward_status"] not in {"complete", "invalid", "not_attempted"}:
        raise ValueError("prepared executor returned an unsupported forward status")
    prefix = value["observed_group_prefix"]
    if not isinstance(prefix, list):
        raise TypeError("prepared executor group prefix must be a list")
    if value["status"] == "complete":
        if value["transform_status"] not in {"complete", "not_applicable"}:
            raise ValueError("complete observation has an incomplete transform")
        if value["forward_status"] != "complete":
            raise ValueError("complete observation has an incomplete forward")
        if [entry["group_index"] for entry in prefix] != list(range(8)):
            raise ValueError("complete observation has an incomplete group order")
    elif value["forward_status"] == "complete":
        raise ValueError("invalid observation has a complete forward")
    return dict(value)


def _invalid_execution(reason: str) -> dict[str, Any]:
    return {
        "status": "invalid",
        "transform_status": "complete",
        "forward_status": "invalid",
        "reason_code": reason,
        "transform_reason_code": None,
        "forward_reason_code": reason,
        "observed_group_prefix": [],
    }


class NvmlMemoryProbe:
    """Sample absolute NVML device-used bytes around one operation."""

    def __init__(
        self,
        gpu_uuid: str,
        *,
        sample_interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self.gpu_uuid = gpu_uuid
        self.sample_interval_seconds = sample_interval_seconds

    def measure(
        self, operation: Callable[[], T], **_: Any
    ) -> tuple[T, Mapping[str, Any]]:
        try:
            import pynvml  # type: ignore[import-untyped]
        except ImportError:
            return operation(), {
                "status": "invalid",
                "reason_code": "NVML_UNAVAILABLE",
            }

        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            handle = pynvml.nvmlDeviceGetHandleByUUID(self.gpu_uuid.encode())
            first = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception as exc:  # noqa: BLE001
            if initialized:
                pynvml.nvmlShutdown()
            return operation(), {
                "status": "invalid",
                "reason_code": f"NVML_QUERY_FAILED:{type(exc).__name__}",
            }

        stop = threading.Event()
        peak = [int(first.used)]
        samples = [1]
        sample_error: list[str] = []
        operation_result: T | None = None
        operation_error: Exception | None = None

        def sample() -> None:
            while not stop.wait(self.sample_interval_seconds):
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    peak[0] = max(peak[0], int(info.used))
                    samples[0] += 1
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - hardware dependent
                    sample_error.append(type(exc).__name__)
                    break

        worker = threading.Thread(
            target=sample, name="qbitplan-nvml-sampler", daemon=True
        )
        worker.start()
        try:
            try:
                operation_result = operation()
            except Exception as exc:  # noqa: BLE001
                operation_error = exc
        finally:
            stop.set()
            worker.join()
            try:
                final = pynvml.nvmlDeviceGetMemoryInfo(handle)
                peak[0] = max(peak[0], int(final.used))
                samples[0] += 1
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - hardware dependent
                sample_error.append(type(exc).__name__)
            pynvml.nvmlShutdown()
        observation: dict[str, Any] = {
            "status": "measured",
            "peak_device_used_bytes": peak[0],
            "baseline_device_used_bytes": int(first.used),
            "sample_count": samples[0],
            "sample_interval_seconds": self.sample_interval_seconds,
            "method": "NVML device-used absolute peak",
        }
        if sample_error:
            observation = {
                "status": "invalid",
                "reason_code": f"NVML_SAMPLE_FAILED:{sample_error[0]}",
                "peak_device_used_bytes": peak[0],
                "baseline_device_used_bytes": int(first.used),
                "sample_count": samples[0],
                "sample_interval_seconds": self.sample_interval_seconds,
            }
        if operation_error is not None:
            raise MemoryProbeOperationError(operation_error, observation) from operation_error
        assert operation_result is not None
        return operation_result, observation


class CudaTraceProbe:
    """Capture one separate annotated CUDA trace without mixing primary timing."""

    def trace(
        self, operation: Callable[[], T], **_: Any
    ) -> tuple[T, Mapping[str, Any]]:
        try:
            import torch
        except ImportError:
            return cast(T, _invalid_execution("CUDA_TRACE_UNAVAILABLE")), {
                "status": "invalid",
                "reason_code": "CUDA_TRACE_UNAVAILABLE",
            }
        if not torch.cuda.is_available():
            return cast(T, _invalid_execution("CUDA_TRACE_UNAVAILABLE")), {
                "status": "invalid",
                "reason_code": "CUDA_TRACE_UNAVAILABLE",
            }
        operation_started = False
        operation_completed = False
        result: T | None = None
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                operation_started = True
                result = operation()
                operation_completed = True
                _synchronize_cuda()
                profiler.step()

            events = [
                event
                for event in (
                    _profiler_event_record(raw_event)
                    for raw_event in (profiler.events() or [])
                )
                if event is not None
            ]
            group_events = [
                event
                for event in events
                if _group_trace_bit(event["name"]) is not None
            ]
            timestamped_group_events = all(
                _valid_number(event.get("start_time_us"))
                and _valid_number(event.get("end_time_us"))
                and event["end_time_us"] > event["start_time_us"]
                for event in group_events
            )
            kernel_switch_count: int | None = None
            if group_events and timestamped_group_events:
                group_events.sort(key=lambda event: event["start_time_us"])
                group_bits = [_group_trace_bit(event["name"]) for event in group_events]
                kernel_switch_count = sum(
                    left != right
                    for left, right in pairwise(group_bits)
                    if left is not None and right is not None
                )
            omitted_dimensions = {
                dimension: {
                    "status": "omitted/unavailable",
                    "reason": "no-prefetch-path",
                    "evidence_class": OMISSION_EVIDENCE_CLASS,
                }
                for dimension in (
                    "prefetch_stall_time",
                )
            }
            host_to_device_coverage = _omitted(
                "TRACE_H2D_COPY_BYTES_UNAVAILABLE"
            )
            if group_events and timestamped_group_events:
                kernel_switch_coverage = _measured_coverage(
                    "annotated qbitplan.group profiler transitions"
                )
            elif group_events:
                kernel_switch_coverage = _omitted(
                    "TRACE_GROUP_TIMESTAMP_UNAVAILABLE"
                )
            else:
                kernel_switch_coverage = _omitted(
                    "TRACE_GROUP_ANNOTATION_UNAVAILABLE"
                )
            omitted_dimensions["host_to_device_bytes"] = host_to_device_coverage
            omitted_dimensions["kernel_switch_count"] = kernel_switch_coverage
            trace: dict[str, Any] = {
                "status": "complete",
                "events": events,
                "method": "PyTorch CUDA profiler annotated ranges",
                "trace_representation": "profiler_events",
                "dimension_coverage": omitted_dimensions,
                "tracer_overhead": _omitted("PROFILER_OVERHEAD_NOT_SEPARABLE"),
            }
            if kernel_switch_count is not None:
                trace["kernel_switch_count"] = kernel_switch_count
            return result, trace
        except Exception as exc:
            if not operation_started:
                return cast(T, _invalid_execution(
                    f"CUDA_TRACE_FAILED:{type(exc).__name__}"
                )), {
                    "status": "invalid",
                    "reason_code": f"CUDA_TRACE_FAILED:{type(exc).__name__}",
                }
            if not operation_completed:
                raise
            return cast(T, result), {
                "status": "invalid",
                "reason_code": f"CUDA_TRACE_FAILED:{type(exc).__name__}",
            }


def _profiler_event_record(event: Any) -> dict[str, Any] | None:
    """Extract only stable, project-annotated fields from one profiler event."""

    name = _safe_event_attribute(event, "name")
    is_project_event = isinstance(name, str) and name.startswith("qbitplan.")
    is_h2d_copy_event = isinstance(name, str) and name.startswith("Memcpy HtoD")
    if not (is_project_event or is_h2d_copy_event):
        return None
    cpu_time_us = _safe_event_attribute(event, "cpu_time_total")
    if not _valid_number(cpu_time_us) or (
        cpu_time_us <= 0 and not is_h2d_copy_event
    ):
        return None
    record: dict[str, Any] = {
        "name": name,
        "cpu_time_ns": int(float(cpu_time_us) * 1_000),
    }
    device_time_us = _safe_event_attribute(event, "device_time_total")
    if _valid_number(device_time_us):
        record["device_time_ns"] = int(float(device_time_us) * 1_000)
    device_memory_bytes = _safe_event_attribute(event, "device_memory_usage")
    if _valid_number(device_memory_bytes):
        record["device_memory_bytes"] = int(device_memory_bytes)
    interval = _safe_event_attribute(event, "time_range")
    start_us = _safe_event_attribute(interval, "start")
    end_us = _safe_event_attribute(interval, "end")
    if _valid_number(start_us):
        record["start_time_us"] = int(float(start_us))
    if _valid_number(end_us):
        record["end_time_us"] = int(float(end_us))
    return record


def _safe_event_attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:  # noqa: BLE001  # profiler event API varies by version
        return None


def _group_trace_bit(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) != 4 or parts[:2] != ["qbitplan", "group"]:
        return None
    if not parts[2].isdigit() or not (0 <= int(parts[2]) < 8):
        return None
    bit = parts[3]
    return bit if bit in {"0", "1", "BF16"} else None


def _measured_coverage(method: str) -> dict[str, Any]:
    return {
        "status": "measured",
        "method": method,
        "evidence_class": DIRECT_EVIDENCE_CLASS,
    }


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_events() -> tuple[Any | None, Any | None]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
    except (ImportError, RuntimeError):
        return None, None


class DirectCostRunner:
    """Measure the declared cost vector around a shared executor lifecycle."""

    evidence_class = DIRECT_EVIDENCE_CLASS

    def __init__(
        self,
        plan: ExperimentPlan | Mapping[str, Any],
        executor: PreparedProfileExecutor,
        *,
        memory_probe: MemoryProbe | None = None,
        trace_probe: TraceProbe | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        normalized_plan = (
            plan
            if isinstance(plan, ExperimentPlan)
            else ExperimentPlan.from_mapping(plan)
        )
        if normalized_plan.data["mode"] != "direct-cost":
            raise ValueError("DirectCostRunner requires a direct-cost plan")
        if getattr(executor, "evidence_class", None) != DIRECT_EVIDENCE_CLASS:
            raise ValueError("DirectCostRunner requires a directly measured executor")
        self.plan = normalized_plan
        self.executor = executor
        self.memory_probe = memory_probe or NvmlMemoryProbe(
            normalized_plan.data["gpu_uuid"]
        )
        self.trace_probe = trace_probe or CudaTraceProbe()
        self.monotonic_ns = monotonic_ns
        self._preparations: dict[str, dict[str, Any]] = {}

    def hardware_identity(self) -> Mapping[str, Any]:
        return self.executor.hardware_identity()

    def prepare_variant(self, variant_id: str) -> Mapping[str, Any]:
        if variant_id in self._preparations:
            return self._preparations[variant_id]
        started = self.monotonic_ns()
        setup_memory: Mapping[str, Any] = {
            "status": "not_attempted",
            "reason_code": "SETUP_MEMORY_NOT_MEASURED",
        }
        preparation: dict[str, Any]
        try:
            prepared, setup_memory = self.memory_probe.measure(
                lambda: self.executor.prepare_variant(variant_id),
                phase="setup",
                variant_id=variant_id,
            )
            preparation = dict(prepared)
        except MemoryProbeOperationError as exc:
            preparation = {
                "status": "invalid",
                "reason_code": f"PREPARATION_EXCEPTION:{type(exc.cause).__name__}",
                "transform_status": "invalid"
                if variant_id != "BF16"
                else "not_applicable",
                "forward_status": "not_attempted",
                "failure_phase": "setup",
            }
            setup_memory = exc.observation
        except (
            RuntimeError,
            ValueError,
            OSError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            MemoryError,
        ) as exc:
            preparation = {
                "status": "invalid",
                "reason_code": f"PREPARATION_EXCEPTION:{type(exc).__name__}",
                "transform_status": "invalid"
                if variant_id != "BF16"
                else "not_applicable",
                "forward_status": "not_attempted",
                "failure_phase": "setup",
            }
        preparation.setdefault("status", "invalid")
        preparation.setdefault("reason_code", "PREPARATION_FAILED")
        preparation["evidence_class"] = DIRECT_EVIDENCE_CLASS
        preparation["preparation_elapsed_ns"] = self.monotonic_ns() - started
        preparation["setup_memory_observation"] = dict(setup_memory)
        self._preparations[variant_id] = preparation
        return preparation

    def measure(self, query: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
        preparation = self.prepare_variant(variant_id)
        if preparation["status"] != "complete":
            return self._invalid_observation(
                query, variant_id, preparation["reason_code"]
            )

        latencies: list[int] = []
        device_latencies: list[int] = []
        memory_observations: list[Mapping[str, Any]] = []
        last_result: Mapping[str, Any] = _invalid_execution("NO_EXECUTION")
        for _ in range(WARMUP_COUNT):
            try:
                last_result = _execution_status(
                    self.executor.execute_prepared(query, variant_id)
                )
            except Exception as exc:  # noqa: BLE001
                return self._invalid_observation(
                    query, variant_id, f"WARMUP_EXCEPTION:{type(exc).__name__}"
                )
            if last_result["status"] != "complete":
                return self._invalid_observation(
                    query,
                    variant_id,
                    str(last_result["reason_code"] or "WARMUP_FAILED"),
                )

        host_timing_observations: list[dict[str, int]] = []
        for repetition in range(MEASURED_COUNT):
            latency_observation: list[int] = []
            device_latency_observation: list[int] = []
            timing_observation: list[dict[str, int]] = []

            def timed_operation(
                record: list[int] = latency_observation,
                device_record: list[int] = device_latency_observation,
                timing_record: list[dict[str, int]] = timing_observation,
            ) -> Mapping[str, Any]:
                _synchronize_cuda()
                started = self.monotonic_ns()
                cuda_start, cuda_end = _cuda_events()
                if cuda_start is not None:
                    cuda_start.record()
                try:
                    return self.executor.execute_prepared(query, variant_id)
                finally:
                    if cuda_end is not None:
                        cuda_end.record()
                    _synchronize_cuda()
                    finished = self.monotonic_ns()
                    elapsed = finished - started
                    record.append(elapsed)
                    timing_record.append(
                        {
                            "host_start_ns": started,
                            "host_end_ns": finished,
                            "host_elapsed_ns": elapsed,
                        }
                    )
                    if cuda_start is not None and cuda_end is not None:
                        device_record.append(
                            int(cuda_start.elapsed_time(cuda_end) * 1_000_000)
                        )

            try:
                last_result, memory = self.memory_probe.measure(
                    timed_operation,
                    query_id=query["query_id"],
                    variant_id=variant_id,
                    repetition=repetition,
                )
                last_result = _execution_status(last_result)
            except MemoryProbeOperationError as exc:
                return self._invalid_observation(
                    query,
                    variant_id,
                    f"MEASUREMENT_EXCEPTION:{type(exc.cause).__name__}",
                    measured_count=len(host_timing_observations),
                    host_timestamps=[*host_timing_observations, *timing_observation],
                    memory_observations=[
                        *memory_observations,
                        exc.observation,
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                return self._invalid_observation(
                    query,
                    variant_id,
                    f"MEASUREMENT_EXCEPTION:{type(exc).__name__}",
                    measured_count=len(host_timing_observations),
                    host_timestamps=[*host_timing_observations, *timing_observation],
                    memory_observations=memory_observations,
                )
            if len(latency_observation) != 1 or len(timing_observation) != 1:
                return self._invalid_observation(
                    query,
                    variant_id,
                    "MEASUREMENT_ADAPTER_EXECUTION_COUNT_INVALID",
                    measured_count=len(host_timing_observations),
                    host_timestamps=[*host_timing_observations, *timing_observation],
                    memory_observations=memory_observations,
                )
            latencies.append(latency_observation[0])
            host_timing_observations.append(timing_observation[0])
            if len(device_latency_observation) == 1:
                device_latencies.append(device_latency_observation[0])
            memory_observations.append(memory)
            if last_result["status"] != "complete":
                return self._invalid_observation(
                    query,
                    variant_id,
                    str(last_result["reason_code"] or "MEASURED_FORWARD_FAILED"),
                    measured_count=len(host_timing_observations),
                    host_timestamps=host_timing_observations,
                    memory_observations=memory_observations,
                )

        try:
            trace_result, trace_observation = self.trace_probe.trace(
                lambda: self.executor.execute_prepared(
                    query, variant_id, trace=True
                ),
                query_id=query["query_id"],
                variant_id=variant_id,
            )
            trace_result = _execution_status(trace_result)
        except Exception as exc:  # noqa: BLE001
            trace_result = _invalid_execution(f"TRACE_EXCEPTION:{type(exc).__name__}")
            trace_observation = {
                "status": "invalid",
                "reason_code": f"TRACE_EXCEPTION:{type(exc).__name__}",
            }
        trace_observation = dict(trace_observation)
        trace_observation.setdefault("status", "invalid")
        if trace_observation["status"] != "complete":
            trace_observation.setdefault("reason_code", "TRACE_UNAVAILABLE")
        trace_observation["evidence_class"] = DIRECT_EVIDENCE_CLASS
        if (
            trace_result["status"] != "complete"
            or trace_observation["status"] != "complete"
        ):
            reason_code = str(
                trace_observation.get("reason_code")
                or trace_result["reason_code"]
                or "TRACE_UNAVAILABLE"
            )
            trace_result = _invalid_execution(reason_code)
            trace_observation["status"] = "invalid"
            trace_observation["reason_code"] = reason_code

        cost_vector = self._cost_vector(
            latencies, memory_observations, trace_observation
        )
        return {
            "evidence_class": DIRECT_EVIDENCE_CLASS,
            "phase": query["phase"],
            "query_id": query["query_id"],
            "profile_id": variant_id,
            "profile_bits": "BF16" if variant_id == "BF16" else variant_id,
            "status": "complete",
            "terminal_status": "complete",
            "reason_code": "MEASURED",
            "repetitions": {
                "warmup_count": WARMUP_COUNT,
                "measured_count": MEASURED_COUNT,
                "trace_pass_count": int(trace_result["status"] == "complete"),
                "trace_pass_attempted": TRACE_PASS_COUNT,
                "trace_status": trace_observation["status"],
                "timing_method": "host monotonic plus synchronized CUDA event",
                "timer_overhead": _omitted("CUDA_EVENT_TIMER_OVERHEAD_NOT_SEPARABLE"),
                "latency_ns": latencies,
                "host_timestamps": host_timing_observations,
                "device_latency_ns": device_latencies,
                "median_latency_ns": int(statistics.median(latencies)),
                "median_device_latency_ns": (
                    int(statistics.median(device_latencies))
                    if device_latencies
                    else None
                ),
            },
            "preparation": preparation,
            "memory_observations": [dict(observation) for observation in memory_observations],
            "cost_vector": cost_vector,
            "trace_observation": trace_observation,
        }

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        """Expose the adapter protocol while keeping measurement explicit."""

        return self.measure(query, variant_id)

    def _invalid_observation(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        reason_code: str,
        *,
        measured_count: int = 0,
        host_timestamps: Sequence[Mapping[str, Any]] = (),
        memory_observations: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "evidence_class": DIRECT_EVIDENCE_CLASS,
            "phase": query["phase"],
            "query_id": query["query_id"],
            "profile_id": variant_id,
            "profile_bits": "BF16" if variant_id == "BF16" else variant_id,
            "status": "invalid",
            "terminal_status": "invalid",
            "reason_code": reason_code,
            "repetitions": {
                "warmup_count": WARMUP_COUNT,
                "measured_count": measured_count,
                "trace_pass_count": 0,
                "trace_pass_attempted": 0,
                "trace_status": "not_attempted",
                "host_timestamps": [dict(value) for value in host_timestamps],
            },
            "preparation": self._preparations.get(variant_id, {}),
            "memory_observations": [
                dict(value) for value in memory_observations
            ],
            "cost_vector": {
                dimension: _omitted(f"measurement-invalid/{reason_code}")
                for dimension in COST_DIMENSIONS
            },
            "trace_observation": {
                "status": "not_attempted",
                "reason_code": reason_code,
                "evidence_class": DIRECT_EVIDENCE_CLASS,
            },
        }

    @staticmethod
    def _cost_vector(
        latencies: Sequence[int],
        memory_observations: Sequence[Mapping[str, Any]],
        trace_observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        memory_values = [
            observation["peak_device_used_bytes"]
            for observation in memory_observations
            if observation.get("status") == "measured"
            and _valid_number(observation.get("peak_device_used_bytes"))
        ]
        vector: dict[str, Any] = {
            "resident_accelerator_bytes": (
                _measured(
                    max(memory_values),
                    unit="bytes",
                    method="NVML device-used absolute peak",
                )
                if memory_values
                else _omitted("NVML_PEAK_UNAVAILABLE")
            ),
            "host_to_device_bytes": _trace_dimension(
                trace_observation,
                "host_to_device_bytes",
                "bytes",
                "CUDA-copy trace records",
            ),
            "latency": (
                _measured(
                    int(statistics.median(latencies)),
                    unit="nanoseconds",
                    method="host monotonic dispatch-to-output",
                )
                if latencies
                else _omitted(
                    str(trace_observation.get("reason_code") or "LATENCY_UNAVAILABLE")
                )
            ),
            "prefetch_stall_time": _trace_dimension(
                trace_observation,
                "prefetch_stall_time_ns",
                "nanoseconds",
                "trace-derived prefetch dependency gaps",
            ),
            "kernel_switch_count": _trace_dimension(
                trace_observation,
                "kernel_switch_count",
                "count",
                "trace-derived annotated group transitions",
            ),
            "controller_probe_feedback_overhead": _omitted("no-controller-path"),
        }
        return vector


def _trace_dimension(
    trace: Mapping[str, Any], key: str, unit: str, method: str
) -> dict[str, Any]:
    if trace.get("status") != "complete":
        return _omitted(str(trace.get("reason_code") or "TRACE_UNAVAILABLE"))
    coverage_key = key.removesuffix("_ns")
    coverage = trace.get("dimension_coverage", {}).get(coverage_key)
    if isinstance(coverage, Mapping) and coverage.get("status") != "measured":
        return _omitted(str(coverage.get("reason") or "TRACE_DIMENSION_UNAVAILABLE"))
    value = trace.get(key)
    if not _valid_number(value):
        return _omitted(f"{key.upper()}_UNAVAILABLE")
    return _measured(cast(float, value), unit=unit, method=method)


def _coverage_records(
    observation: Mapping[str, Any],
    *,
    producer_git_sha: str,
    source_manifest_id: str,
    source_artifact_id: str,
    configuration_hash: str,
    created_at: str,
) -> list[dict[str, Any]]:
    records = []
    for dimension, value in observation["cost_vector"].items():
        records.append(
            {
                **_lineage(
                    artifact_type="cost-coverage",
                    schema_version="qbitplan.stage1.cost-coverage.v1",
                    producer_git_sha=producer_git_sha,
                    source_manifest_id=source_manifest_id,
                    source_artifact_id=source_artifact_id,
                    configuration_hash=configuration_hash,
                    record_count=1,
                    created_at=created_at,
                    evidence_class="analytical",
                ),
                "phase": observation["phase"],
                "query_id": observation["query_id"],
                "profile_id": observation["profile_id"],
                "dimension": dimension,
                "status": value["status"],
                "reason": value.get("reason"),
                "measurement_evidence_class": value["evidence_class"],
                "claim_scope": "direct measurement coverage only",
            }
        )
    return records


def _measurement_scope(observations: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    scope = {
        "latency": "host monotonic dispatch through synchronized output readiness",
        "resident_accelerator_bytes": "absolute peak NVML device-used bytes",
        "controller_probe_feedback_overhead": "omitted for offline profile execution",
    }
    for dimension in (
        "host_to_device_bytes",
        "prefetch_stall_time",
        "kernel_switch_count",
    ):
        values = [
            observation["cost_vector"][dimension]
            for observation in observations
        ]
        measured_count = sum(value["status"] == "measured" for value in values)
        if values and measured_count == len(values):
            scope[dimension] = "trace-derived direct measurement"
            continue
        reasons = sorted(
            {
                str(value.get("reason") or "TRACE_DIMENSION_UNAVAILABLE")
                for value in values
                if value["status"] != "measured"
            }
        )
        if measured_count:
            scope[dimension] = (
                f"partially measured ({measured_count}/{len(values)}); "
                "omitted/unavailable/"
                + (reasons[0] if reasons else "MIXED_COVERAGE")
            )
        else:
            scope[dimension] = "omitted/unavailable/" + (
                reasons[0] if reasons else "MIXED_COVERAGE"
            )
    return scope


def execute_direct_cost_plan(
    experiment_plan: ExperimentPlan, runner: DirectCostRunner
) -> ArtifactBundle:
    """Execute the fixed direct-cost frame into one immutable bundle."""

    if experiment_plan.data["mode"] != "direct-cost":
        raise ValueError("direct-cost execution requires a direct-cost plan")
    producer_git_sha = _producer_git_sha()
    configuration_hash = _configuration_hash(experiment_plan)
    plan_id = experiment_plan.plan_id()
    run_identity = {
        "schema_version": "qbitplan.stage1.run-identity.v1",
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "mode": experiment_plan.data["mode"],
        "gpu_uuid": experiment_plan.data["gpu_uuid"],
        "configuration_hash": configuration_hash,
    }
    run_id = sha256_canonical(run_identity)
    bundle_path = experiment_plan.artifact_root / run_id
    experiment_plan.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        bundle_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"immutable run already exists for this attempt: {bundle_path}; use a new attempt_id"
        ) from exc

    created_at = _now()
    try:
        hardware_identity = dict(runner.hardware_identity())
    except (
        RuntimeError,
        ValueError,
        OSError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        MemoryError,
    ) as exc:
        hardware_identity = {
            "identity_status": "unavailable",
            "reason_code": f"HARDWARE_IDENTITY_EXCEPTION:{type(exc).__name__}",
            "gpu_uuid": experiment_plan.data["gpu_uuid"],
        }

    setups: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for profile_id in experiment_plan.data["profiles"]:
        preparation = runner.prepare_variant(profile_id)
        setups.append(
            {
                **_lineage(
                    artifact_type="direct-cost-setup",
                    schema_version="qbitplan.stage1.direct-cost-setup.v1",
                    producer_git_sha=producer_git_sha,
                    source_manifest_id=experiment_plan.source_manifest_id,
                    source_artifact_id=experiment_plan.data["source_manifest"][
                        "artifact_id"
                    ],
                    configuration_hash=configuration_hash,
                    record_count=1,
                    created_at=created_at,
                    evidence_class=DIRECT_EVIDENCE_CLASS,
                ),
                "profile_id": profile_id,
                **preparation,
                "hardware_identity": hardware_identity,
                "claim_scope": "model loading and quantization setup timing only",
            }
        )
        for query in experiment_plan.data["queries"]:
            observation = runner.measure(query, profile_id)
            observation = {
                **_lineage(
                    artifact_type="direct-cost-observation",
                    schema_version="qbitplan.stage1.direct-cost-observation.v1",
                    producer_git_sha=producer_git_sha,
                    source_manifest_id=experiment_plan.source_manifest_id,
                    source_artifact_id=experiment_plan.data["source_manifest"][
                        "artifact_id"
                    ],
                    configuration_hash=configuration_hash,
                    record_count=1,
                    created_at=created_at,
                    evidence_class=DIRECT_EVIDENCE_CLASS,
                ),
                **observation,
            }
            observations.append(observation)
            trace = observation["trace_observation"]
            traces.append(
                {
                    **_lineage(
                        artifact_type="direct-cost-trace",
                        schema_version="qbitplan.stage1.direct-cost-trace.v1",
                        producer_git_sha=producer_git_sha,
                        source_manifest_id=experiment_plan.source_manifest_id,
                        source_artifact_id=experiment_plan.data["source_manifest"][
                            "artifact_id"
                        ],
                        configuration_hash=configuration_hash,
                        record_count=1,
                        created_at=created_at,
                        evidence_class=DIRECT_EVIDENCE_CLASS,
                    ),
                    "phase": observation["phase"],
                    "query_id": observation["query_id"],
                    "profile_id": observation["profile_id"],
                    "status": trace["status"],
                    "reason_code": trace.get("reason_code"),
                    "raw_trace": trace,
                    "claim_scope": "trace-derived direct cost evidence only",
                }
            )
            coverage.extend(
                _coverage_records(
                    observation,
                    producer_git_sha=producer_git_sha,
                    source_manifest_id=experiment_plan.source_manifest_id,
                    source_artifact_id=experiment_plan.data["source_manifest"][
                        "artifact_id"
                    ],
                    configuration_hash=configuration_hash,
                    created_at=created_at,
                )
            )
        release = getattr(runner.executor, "release_variant", None)
        if callable(release):
            release(profile_id)

    shared_lineage = {
        "producer_git_sha": producer_git_sha,
        "source_manifest_id": experiment_plan.source_manifest_id,
        "source_artifact_id": experiment_plan.data["source_manifest"]["artifact_id"],
        "configuration_hash": configuration_hash,
        "created_at": created_at,
    }
    run_manifest = {
        **_lineage(
            artifact_type="run-manifest",
            schema_version="qbitplan.stage1.direct-cost-run-manifest.v1",
            record_count=len(observations),
            evidence_class="analytical",
            **shared_lineage,
        ),
        "run_id": run_id,
        "plan_id": plan_id,
        "attempt_id": experiment_plan.attempt_id,
        "run_identity": run_identity,
        "mode": experiment_plan.data["mode"],
        "profile_count": len(experiment_plan.data["profiles"]),
        "query_count": len(experiment_plan.data["queries"]),
        "warmup_count": WARMUP_COUNT,
        "measured_count": MEASURED_COUNT,
        "trace_pass_count": TRACE_PASS_COUNT,
        "trace_attempted_count": sum(
            observation["repetitions"].get("trace_pass_attempted", 0)
            for observation in observations
        ),
        "trace_completed_count": sum(
            observation["repetitions"].get("trace_pass_count", 0)
            for observation in observations
        ),
        "cost_dimensions": list(COST_DIMENSIONS),
        "measurement_scope": _measurement_scope(observations),
        "hardware_identity": hardware_identity,
    }
    file_contents = {
        "plan.json": _canonical_json_file(experiment_plan.to_mapping()),
        "run-manifest.json": _canonical_json_file(run_manifest),
        "setup-observations.ndjson": _canonical_ndjson(setups),
        "cost-observations.ndjson": _canonical_ndjson(observations),
        "trace-observations.ndjson": _canonical_ndjson(traces),
        "cost-coverage.ndjson": _canonical_ndjson(coverage),
    }
    file_hashes = {
        filename: _write_once(bundle_path / filename, content)
        for filename, content in file_contents.items()
    }
    bundle_identity = {
        "schema_version": "qbitplan.stage1.direct-cost-bundle.v1",
        "run_id": run_id,
        "plan_id": plan_id,
        "source_manifest_id": experiment_plan.source_manifest_id,
        "configuration_hash": configuration_hash,
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    bundle_payload = {
        **_lineage(
            artifact_type="direct-cost-bundle",
            schema_version="qbitplan.stage1.direct-cost-bundle.v1",
            record_count=len(observations),
            evidence_class=DIRECT_EVIDENCE_CLASS,
            **shared_lineage,
        ),
        "bundle_id": sha256_canonical(bundle_identity),
        "run_id": run_id,
        "plan_id": plan_id,
        "claim_scope": "directly measured cost evidence; smoke frame only",
        "non_evidentiary": True,
        "evidence_class_scope": "valid cost dimensions only",
        "files": [*file_contents, "bundle.json", "artifact-index.json"],
        "artifact_index_file": "artifact-index.json",
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
        "evidence_boundary": {
            "quality": "omitted",
            "cost": "directly measured; bounded to valid dimensions and declared smoke frame",
            "systems_benefit": "omitted/unavailable/direct-cost-smoke",
            "generalization": "omitted",
        },
        "hardware_identity": hardware_identity,
    }
    file_hashes["bundle.json"] = _write_once(
        bundle_path / "bundle.json", _canonical_json_file(bundle_payload)
    )
    index_payload = {
        **_lineage(
            artifact_type="artifact-index",
            schema_version="qbitplan.stage1.artifact-index.v1",
            record_count=len(file_hashes),
            evidence_class="analytical",
            **shared_lineage,
        ),
        "bundle_file": "bundle.json",
        "bundle_artifact_id": file_hashes["bundle.json"],
        "file_hashes": file_hashes,
        "indexed_files": list(file_hashes),
    }
    file_hashes["artifact-index.json"] = _write_once(
        bundle_path / "artifact-index.json", _canonical_json_file(index_payload)
    )
    return ArtifactBundle(
        path=bundle_path,
        run_id=run_id,
        bundle_id=bundle_payload["bundle_id"],
        files=file_hashes,
    )

"""Direct hardware-cost measurement around the shared profile executor."""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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
        self, query: Mapping[str, Any], variant_id: str
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
        "evidence_class": DIRECT_EVIDENCE_CLASS,
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
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
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
    """Capture one separate PyTorch CUDA trace without making estimates."""

    def trace(
        self, operation: Callable[[], T], **_: Any
    ) -> tuple[T, Mapping[str, Any]]:
        try:
            import torch
        except ImportError:
            return operation(), {
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
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                operation_started = True
                result = operation()
                operation_completed = True
                profiler.step()
            events = []
            for event in profiler.key_averages() or []:
                device_time_us = getattr(event, "device_time_total", None)
                if device_time_us is None:
                    device_time_us = getattr(event, "cuda_time_total", None)
                event_record: dict[str, Any] = {
                    "name": event.name,
                    "cpu_time_ns": int(event.self_cpu_time_total * 1000),
                }
                if device_time_us is not None:
                    event_record["device_time_ns"] = int(device_time_us * 1000)
                events.append(event_record)
            omitted_dimensions = {
                dimension: {
                    "status": "omitted/unavailable",
                    "reason": "TRACE_ANNOTATION_UNAVAILABLE",
                    "evidence_class": DIRECT_EVIDENCE_CLASS,
                }
                for dimension in (
                    "host_to_device_bytes",
                    "prefetch_stall_time",
                    "kernel_switch_count",
                )
            }
            return result, {
                "status": "complete",
                "events": events,
                "method": "PyTorch CUDA profiler key-average trace",
                "trace_representation": "bounded_key_averages",
                "dimension_coverage": omitted_dimensions,
                "tracer_overhead": _omitted("PROFILER_OVERHEAD_NOT_SEPARABLE"),
            }
        except Exception as exc:
            if not operation_started:
                return operation(), {
                    "status": "invalid",
                    "reason_code": f"CUDA_TRACE_FAILED:{type(exc).__name__}",
                }
            if not operation_completed:
                raise
            return cast(T, result), {
                "status": "invalid",
                "reason_code": f"CUDA_TRACE_FAILED:{type(exc).__name__}",
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

        for repetition in range(MEASURED_COUNT):
            latency_observation: list[int] = []

            device_latency_observation: list[int] = []

            def timed_operation(
                record: list[int] = latency_observation,
                device_record: list[int] = device_latency_observation,
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
                    record.append(self.monotonic_ns() - started)
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
            except Exception as exc:  # noqa: BLE001
                return self._invalid_observation(
                    query, variant_id, f"MEASUREMENT_EXCEPTION:{type(exc).__name__}"
                )
            if len(latency_observation) != 1:
                return self._invalid_observation(
                    query, variant_id, "MEASUREMENT_ADAPTER_EXECUTION_COUNT_INVALID"
                )
            latencies.append(latency_observation[0])
            if len(device_latency_observation) == 1:
                device_latencies.append(device_latency_observation[0])
            memory_observations.append(memory)
            if last_result["status"] != "complete":
                return self._invalid_observation(
                    query,
                    variant_id,
                    str(last_result["reason_code"] or "MEASURED_FORWARD_FAILED"),
                )

        try:
            trace_result, trace_observation = self.trace_probe.trace(
                lambda: self.executor.execute_prepared(query, variant_id),
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
        trace_observation.setdefault("reason_code", "TRACE_UNAVAILABLE")
        trace_observation["evidence_class"] = DIRECT_EVIDENCE_CLASS
        if trace_result["status"] != "complete":
            trace_observation["status"] = "invalid"
            trace_observation["reason_code"] = trace_result["reason_code"]

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
                "trace_pass_count": TRACE_PASS_COUNT,
                "latency_ns": latencies,
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
        self, query: Mapping[str, Any], variant_id: str, reason_code: str
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
                "measured_count": 0,
                "trace_pass_count": 0,
            },
            "preparation": self._preparations.get(variant_id, {}),
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
        "cost_dimensions": list(COST_DIMENSIONS),
        "measurement_scope": {
            "latency": "host monotonic dispatch through synchronized output readiness",
            "resident_accelerator_bytes": "absolute peak NVML device-used bytes",
            "host_to_device_bytes": "annotated CUDA-copy trace records",
            "prefetch_stall_time": "trace-derived dependent-copy gaps",
            "kernel_switch_count": "trace-derived annotated group transitions",
            "controller_probe_feedback_overhead": "omitted for offline profile execution",
        },
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

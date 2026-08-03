# Issue 5: Stage-1 hardware, proxy, and observable cost dimensions

Research date: 2026-08-03
Ticket: [Verify Stage-1 hardware or proxy and observable cost dimensions](https://github.com/weige15/QBitPlan/issues/5)

## Scope and rule

This note applies the ticket's predeclared rule:

> Retain only options with primary-source hardware/backend documentation,
> reproducible identity, a clear measurement or lookup path, and enough
> observability to distinguish resident accelerator bytes, host-to-device
> bytes, latency, prefetch stalls, kernel switches, and controller overhead.

The note compares measurement and proxy paths. It does not select the project
hardware, backend, or cost coefficients. The exact target remains the scope of
[Select target hardware or hardware proxy for the Stage-1 cost contract](https://github.com/weige15/QBitPlan/issues/12).

## Source facts

### Execution and identity

- The related model/backend feasibility record retains four pinned 32-layer
  model candidates for later selection. It identifies TorchAO's documented
  `Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`, and per-FQN configuration
  mechanism as a conditional mixed 4/8 path; it does not directly verify a
  named model on a named device. See [Issue 3 feasibility](issue-3-model-backend-feasibility.md)
  and [Issue 7 mixed-execution verification](issue-7-mixed-execution-verification.md).
- NVIDIA's NVML API exposes the device name, an immutable device UUID, driver
  and CUDA-driver version queries, and used/free/reserved/total device memory
  queries. NVML documents that memory accounting is dynamic and can depend on
  operating-system accounting; under Linux/TCC, used memory is the sum of
  allocations by active channels. See the [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html),
  [NVML system queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlSystemQueries.html),
  and [NVML UUID documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html).
- NVIDIA Nsight Systems documents CUDA API tracing and CUDA workload tracing,
  including host-to-device memory operations and kernel executions. It also
  supports CUDA memory-usage tracking and NVTX annotations, while warning that
  tracing options can add significant runtime overhead. See the [Nsight Systems
  CUDA trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and
  [NVTX trace](https://docs.nvidia.com/nsight-systems/2024.1/UserGuide/index.html)
  documentation.
- CUPTI documents activity records for CUDA runtime/driver calls, concurrent
  kernels, memory copies, allocation/free activity, NVTX markers, and
  timestamps. CUPTI also documents that activity buffers can consume device
  memory and that dropped records or tracing overhead are possible. See the
  [CUPTI overview](https://docs.nvidia.com/cupti/index.html) and [CUPTI Activity
  API](https://docs.nvidia.com/cupti/13.3.0/api/group__CUPTI__ACTIVITY__API.html).
- PyTorch's profiler can collect CPU and CUDA activity, CUDA memory-copy
  activity, operator memory allocation/deallocation, and exported traces. The
  PyTorch memory profiler only sees memory managed by the PyTorch allocator;
  allocations made directly through CUDA APIs are invisible to it. See the
  [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html)
  and [PyTorch CUDA memory documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html).

### Observable-dimension interpretation

The following classifications distinguish source capability from the metric
calculation QBitPlan would perform over the trace:

- A CUDA activity, memory-copy, allocation, or timestamp record is direct
  execution evidence.
- A stall duration, kernel-switch count, or controller range total computed by
  correlating records is a derived measurement, not a vendor-provided scalar.
- A value obtained from a fixed-profile table or an analytical byte formula is
  estimated or simulated and cannot be reported as direct systems evidence.

## Comparison matrix

| Option | Evidence class | Identity and path | Resident accelerator bytes | Host-to-device bytes | Latency | Prefetch stalls | Kernel switches | Controller overhead | Rule result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Named NVIDIA CUDA device + TorchAO + Nsight Systems/CUPTI, with NVML and NVTX/PyTorch annotations | Direct measurement, with derived summaries | Conditional: record GPU UUID/name, driver/CUDA, PyTorch/TorchAO versions, model/revision, and trace configuration; collect a fixed-profile run on that device | Measurable with NVML plus allocator/allocation traces, but define process/peak/reserved scope; PyTorch-only accounting is incomplete | Measurable from CUDA workload/CUPTI memory-copy records | Measurable from CUDA/API/kernel timestamps | Derivable from copy completion, dependent-kernel start, stream, and NVTX group ranges; not a native single metric | Derivable by comparing ordered kernel launches/configurations within annotated group ranges | Measurable from controller CPU/NVTX/CUDA ranges, with profiler overhead recorded separately | **Conditional pass** for the complete contract; exact device and backend remain OPEN |
| CUDA execution + PyTorch/Kineto profiler without Nsight/CUPTI/NVML augmentation | Direct partial measurement | Conditional: record CUDA/PyTorch identity and profiler configuration; `GPU_MEMCPY` and CUDA activity paths are documented | Partial only: PyTorch allocator visibility does not include direct CUDA allocations or all device-resident memory | Available for collected CUDA memory-copy activity | Available for collected CPU/CUDA activity | Derivable only where the trace contains the needed stream and range events | Limited; external CUDA trace may be needed for a reliable launch/configuration sequence | CPU/operator ranges can be annotated, but external device/API correlation may be missing | **Not sufficient as the sole Stage-1 systems path**; useful supplementary instrumentation |
| Hardware-calibrated per-group lookup table | Lookup-table estimated | Conditional: build from direct measurements on one pinned device/backend and retain the profile, revision, and calibration manifest | Estimated for unmeasured profiles from calibrated entries; not a direct resident-byte observation | Estimated from calibrated transfer entries; unseen prefetch behavior is not established | Estimated from replayed profile measurements/interpolation; must remain lookup evidence | Not observable for an unseen profile unless its relevant transfer/stream schedule was measured | Not observable for an unseen profile unless its launch sequence was measured | Not observable for an unseen controller path unless that path was traced | **Retain only as a secondary proxy**; cannot support a measured systems claim by itself |
| Parameter-count/average-bit analytical cost or fake-quantized execution | Analytical/simulated | No hardware identity or runtime trace; clear arithmetic path exists | Estimated parameter storage only; resident slices, allocator, and runtime buffers are omitted | Omitted or assumed | Omitted or assumed | Omitted | Omitted | Omitted | **Reject as a standalone cost contract**; average bit-width is not a systems result |

## Inference

The documentation supports one complete evidence path in principle: a named
CUDA device running the declared TorchAO/PyTorch execution path, traced with
CUDA activities and annotated with group/profile/controller ranges, plus NVML
and allocator/allocation measurements. The support is conditional because no
QBitPlan target device, software tuple, model-specific mixed execution, or
prefetch schedule has been selected or run.

The complete path still contains derived metrics. In particular, prefetch
stalls and kernel switches must be defined as calculations over timestamped
events and annotations. They must not be presented as directly measured
vendor counters. A proxy table can estimate costs for profiles that were not
replayed, but it cannot establish those dynamic dimensions without the
corresponding direct trace evidence.

## Project preference and uncertainty

- The scientific standards require each cost dimension to be marked measured,
  estimated, or omitted, and prohibit systems claims from average bit-width or
  simulated/fake-quantized execution alone. This is a standing project
  invariant, not a hardware selection.
- No target hardware, exact TorchAO/PyTorch/CUDA version tuple, group execution
  semantics, host-memory/prefetch policy, or cost coefficients are specified
  here. Filling any of them by convention would answer the downstream human
  selection ticket.
- No model inference or hardware experiment was run for this note. The
  primary sources establish observability capabilities and limitations, not
  candidate-specific execution success or a measured systems benefit.
- Profiling can perturb the run: Nsight Systems warns about tracing overhead,
  and CUPTI documents profiling-buffer memory and dropped-record risks. The
  eventual protocol must declare the profiling mode and report its overhead or
  use an unprofiled timing pass alongside the trace pass.

## Conclusion

The evidence-supported comparison is: retain a pinned NVIDIA CUDA execution
path with Nsight Systems/CUPTI plus explicit memory and NVTX instrumentation as
the conditional direct-measurement option; retain a hardware-calibrated lookup
table only as a labeled secondary proxy; and reject average-bit or
fake-quantized cost as a standalone systems contract. This is a feasibility
result, not a selection of the project's target hardware or backend.

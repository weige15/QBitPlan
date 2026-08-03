# ADR-0003: Stage-1 cost vector and profile-comparison policy

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Define cost dimensions used by profile
selection](https://github.com/weige15/QBitPlan/issues/16). The accepted target
from [Select target hardware or hardware proxy for the Stage-1 cost
contract](https://github.com/weige15/QBitPlan/issues/12) is direct measurement on
one UUID-pinned NVIDIA GeForce RTX 3090 per run, using the declared
CUDA/PyTorch/TorchAO path. The hardware-feasibility evidence is recorded in
[the issue-5 cost-observability note](../research/issue-5-hardware-cost-feasibility.md).

Stage-1 profile comparison must not substitute average bit-width for execution
cost or combine unlike dimensions with unrecorded weights.

## Decision

Represent the cost of a query/profile execution as this vector:

    C(query, profile) = (
      resident_accelerator_bytes,
      host_to_device_bytes,
      latency,
      prefetch_stall_time,
      kernel_switch_count,
      controller_probe_feedback_overhead
    )

The inclusion rules are:

- Resident accelerator bytes, host-to-device bytes, and latency are included
  for every hardware-executable profile.
- Prefetch stalls and kernel switches are included when the accepted execution
  semantics make them observable and able to vary profile ranking.
- Controller, probe, and feedback overhead is included for every variant that
  executes that path; it is not silently excluded from the variant cost.
- A dimension that is unavailable, unobserved, invalidated by profiling, or
  not covered by a declared estimation procedure is recorded as
  omitted/unavailable/<reason>. It is not recorded as zero, imputed, or
  replaced by average bit-width. A declared lookup table may instead mark an
  unobserved dimension estimated, never measured.

Use a vector comparison rather than an arbitrary scalar coefficient. A profile
is cheaper than another profile only when it is no greater in every included
dimension and strictly lower in at least one. Incomparable profiles remain on
the reported quality-cost Pareto frontier. Any per-dimension budget ceilings
used by a later protocol must be predeclared there; this ADR does not invent
their values.

## Evidence and calculation status

- Resident accelerator bytes, host-to-device bytes, and latency are directly
  measured on the declared target with the relevant NVML, allocation, CUDA
  activity, and timestamp evidence.
- Prefetch-stall time and kernel-switch count are derived calculations over
  timestamped CUDA/CUPTI records and annotations. They must be labeled as
  trace-derived measurements, not vendor-provided scalar counters.
- Controller/probe/feedback overhead is measured from annotated controller and
  execution ranges, with profiling overhead recorded separately.
- A declared hardware-calibrated lookup table may provide estimated costs
  for profiles not replayed, but it is secondary evidence and cannot establish
  a measured systems claim for an unobserved dimension.

Every reported component records its evidence class, value or result,
calculation/measurement method, profile and run coverage, and a pointer to the
immutable raw artifact. The evidence and claim boundaries in
[ADR-0001](0001-v0.1-research-boundary.md) and
[SCIENTIFIC_STANDARDS.md](../../SCIENTIFIC_STANDARDS.md) continue to apply.

## Consequences

- Profile comparisons preserve the separate resource dimensions and expose
  trade-offs instead of hiding them in a weighted sum.
- Quality thresholds, the primary quality metric, exact execution semantics,
  prefetch schedule, measurement scope, and any per-dimension ceilings remain
  downstream protocol decisions.
- Average-bit arithmetic, fake-quantized execution, and an uncalibrated lookup
  table do not establish a measured systems benefit.
- Stage-1 claims remain bounded to the evaluated model, dataset, software and
  backend tuple, UUID-pinned target environment, and declared execution
  configuration.

## Uncertainty retained

This decision does not select the model or revision, the exact software tuple,
4/8 group execution semantics, prefetch/cache policy, profiling mode, resident
memory accounting scope, or the values of later budget ceilings. Those fields
must be declared when the dependent protocol decisions are accepted.

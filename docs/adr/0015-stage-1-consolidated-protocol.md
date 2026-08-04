# ADR-0015: Accepted Stage-1 consolidated protocol

- Status: ACCEPTED
- Date: 2026-08-04
- Decision owners: project maintainers

## Context

The originating decision ticket is [Accept the consolidated Stage-1 protocol
and ADR set](https://github.com/weige15/QBitPlan/issues/20), a child of
[Wayfinder: Lock the QBitPlan Stage-1 falsification
protocol](https://github.com/weige15/QBitPlan/issues/2). Its native blockers,
[Reconcile the Stage-1 metric ADR with the accepted task
suite](https://github.com/weige15/QBitPlan/issues/22) and [Set Stage-1
stop/continue rules](https://github.com/weige15/QBitPlan/issues/19), are closed.

The ticket's predeclared rule requires mutual consistency and implementation
readiness for the pilot model/revision, reference and ground-truth roles, 4/8
semantics, partition/exclusions, task and immutable splits, metrics,
degradation/epsilon, evidence classes, hardware/cost contract, oracle
procedure, signal/interaction thresholds, oracle-headroom criterion, and
stop/continue rules. The durable consolidated record is
[QBitPlan Stage-1 falsification protocol](../protocol/stage-1-falsification-protocol.md).

## Decision

Accept the consolidated Stage-1 scientific contract in the protocol document
and the accepted ADR set ADR-0003 through ADR-0014. The set is coherent on the
following shared boundaries:

- one pinned `meta-llama/Llama-3.1-8B` revision;
- one unquantized BF16 reference kept distinct from external ground truth;
- eight ordered four-layer groups with explicit TorchAO weight-only 4/8
  semantics and a fixed BF16 complement;
- the corrected MMLU-Pro plus MATH/MATH-500 suite and immutable phase manifests;
- task correctness as primary, reference-relative diagnostics separate, and
  `epsilon = 0.01` as the per-dataset degradation gate;
- claim-specific evidence classes and the six-dimensional componentwise/Pareto
  cost contract on the UUID-pinned RTX 3090 path;
- exhaustive 256-profile oracle enumeration with the fixed non-final cost
  frames; and
- ordered signal → interaction → oracle-headroom gates with the accepted
  five-percentage-point paired-uncertainty thresholds and stop rules.

The metric record is reconciled with the accepted task suite as documented by
[Reconcile the Stage-1 metric ADR with the accepted task
suite](https://github.com/weige15/QBitPlan/issues/22): GPQA is excluded, each
accepted final dataset has one-half macro weight, and BF16-relative measures
remain diagnostics.

This acceptance is for the pre-`$to-spec` scientific contract. It does not
authorize implementation, an experiment run, a controller architecture,
production infrastructure, or a serving claim.

## Source facts

- `CONTEXT.md` defines the reference/ground-truth distinction, profile and
  causal terminology, and accepted/proposed/open language.
- `SCIENTIFIC_STANDARDS.md` requires evidence-class separation, paired query
  evaluation, causal boundaries, explicit cost dimensions, reproducibility,
  and honest null reporting.
- The closed decision tickets and ADRs linked in the consolidated protocol
  provide the selected values and their source evidence.
- Issue 22 supplies the latest correction to the metric/task-suite boundary;
  its blocker and issue 19 are both closed before this acceptance.

## Inference

The accepted records are mutually consistent because the model, reference,
task/split, quality, cost/evidence, oracle, gate, and stop/continue contracts
refer to the same bounded Stage-1 study and do not introduce incompatible
targets or hidden scalar objectives. The protocol document makes remaining
implementation/evaluation fields visible without treating them as selected
values.

## Project preference

The project prefers a minimal task suite, conservative five-percentage-point
practical margins, paired uncertainty, explicit practical-null/inconclusive/
negative outcomes, Pareto cost comparison, and no controller authorization from
gross oracle headroom alone. These are preferences, not source facts.

## Retained uncertainty and handoff boundary

At acceptance, the exact tokenizer and software tuple, inference controls,
prompt and answer normalization, diagnostic positions, bootstrap seed/replicate
count, static-profile and feature tie handling, upstream-context
representation, raw-artifact schema, profiling/prefetch procedures,
measurement scopes, and per-dimension budget ceilings were retained as OPEN
handoff fields. No value was inferred by convention.

## Handoff resolution

The [Stage-1 execution contract](../protocol/stage-1-execution-contract.md)
fixes or explicitly excludes every retained handoff field before `$to-spec`.
It does not amend the scientific decisions in this ADR. Actual GPU UUIDs,
`P_exec`, artifact hashes, invalid-run outcomes, and observed cost-dimension
coverage remain execution records rather than preselected values.

## Related records

- [QBitPlan Stage-1 falsification protocol](../protocol/stage-1-falsification-protocol.md)
- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0008: Stage-1 primary and diagnostic metrics](0008-stage-1-primary-and-diagnostic-metrics.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [ADR-0010: Stage-1 oracle profile enumeration and sampling](0010-stage-1-oracle-profile-enumeration-and-sampling.md)
- [ADR-0011: Stage-1 signal test and success threshold](0011-stage-1-signal-test-and-success-threshold.md)
- [ADR-0012: Stage-1 interaction test and success threshold](0012-stage-1-interaction-test-and-success-threshold.md)
- [ADR-0013: Stage-1 oracle-headroom test and stop criterion](0013-stage-1-oracle-headroom-and-stop-criterion.md)
- [ADR-0014: Stage-1 stop/continue rules](0014-stage-1-stop-continue-rules.md)

# ADR-0012: Stage-1 interaction test and success threshold

- Status: ACCEPTED
- Date: 2026-08-04
- Decision owners: project maintainers

## Context

The originating decision ticket is [Design the interaction test and
predeclare its success threshold](https://github.com/weige15/QBitPlan/issues/17).
The Stage-1 profile universe and causal upstream-execution boundary are fixed
by [ADR-0010](0010-stage-1-oracle-profile-enumeration-and-sampling.md). The
signal-test baseline and feasible-profile endpoint are fixed by
[ADR-0011](0011-stage-1-signal-test-and-success-threshold.md). The quality
feasibility rule is fixed by [ADR-0009](0009-stage-1-degradation-and-epsilon.md),
and the separate cost vector and evidence policy are fixed by
[ADR-0003](0003-stage-1-cost-contract.md).

The interaction gate asks whether the value of a later group changes with
causally available upstream precision context, beyond the value predicted by
an additive independent planner. It is a falsification test for interaction,
not an implementation decision or a measured serving claim.

## Decision

### Estimand and evaluation boundary

The primary estimand is the improvement in feasible-profile hit rate from an
interaction-aware planner over the additive independent baseline. Evaluate the
paired difference on the 4,500 held-out non-final MATH validation records.
Construct targets and fit or select planner behavior using the 7,500 non-final
MATH source-training records. MATH-500 and MMLU-Pro remain final-only and must
not be used for profile construction, planner fitting, threshold selection, or
declaring the interaction gate.

For each validation query, the target set is the set of executable profiles
that satisfy the accepted Stage-1 quality-feasibility rule. A planner records a
hit when its one selected executable profile belongs to that query's target
set. The same query IDs, profile universe, BF16 reference, quality rule, and
answer-evaluation contract are used for both planners.

### Planner comparison and causal boundary

- The independent baseline is the query-conditioned additive per-group scorer
  or MCKP planner from ADR-0011. It uses the same causally available query
  features and has no upstream context or interaction terms.
- The interaction-aware planner uses those same query features and may
  condition a later group decision on upstream precision choices and the
  resulting execution context available after earlier groups have run.
- Information from group `g` may influence groups `g+1` through `g7` only. No
  later-group outcome, final correctness result, ground-truth answer, future
  hidden state, or other post-decision information may affect an earlier
  decision.

Both planners are evaluated under the same declared quality and cost contract.
The cost vector includes resident accelerator bytes, host-to-device bytes,
latency, observable prefetch stalls and kernel switches, and all applicable
probe, controller, and feedback overhead. Unavailable dimensions remain
`omitted/unavailable` with a reason. No additional scalar cost weighting or
per-dimension ceiling is selected by this ADR; any later budget ceiling must be
predeclared by the dependent protocol.

### Predeclared success rule

Let `HR_interaction` and `HR_independent` be the feasible-profile hit rates on
the same validation queries, and let

`Delta = HR_interaction - HR_independent`.

Declare the interaction gate successful only when both conditions hold:

1. `Delta >= 0.05` (an absolute improvement of at least five percentage
   points); and
2. the paired 95% bootstrap lower bound for `Delta` is greater than zero.

Bootstrap resampling uses validation queries as the paired unit. A positive
improvement below five percentage points is a practical null. An interval that
includes zero is inconclusive. A worse interaction-aware result is negative.
All outcomes, paired query results, uncertainty intervals, precision-shape
diagnostics, and quality-cost/Pareto results are reported.

## Source facts

- `SCIENTIFIC_STANDARDS.md` requires causal information boundaries, paired
  query evaluation, inclusion of probe/controller/feedback overhead, and
  explicit reporting of negative and null findings.
- ADR-0003 fixes the six-dimensional cost vector, componentwise/Pareto
  comparison, and omission rules for unavailable dimensions.
- ADR-0005 fixes the eight ordered binary group choices and the hardware-
  executable 4/8 group semantics.
- ADR-0006 fixes the non-final MATH training/validation boundary and keeps
  MATH-500 and MMLU-Pro final-only.
- ADR-0009 fixes paired BF16-relative degradation and `epsilon = 0.01` for
  quality feasibility.
- ADR-0010 fixes exhaustive enumeration of the 256 analytical profiles and
  complete ordered execution with profile-specific upstream contexts.
- ADR-0011 fixes feasible-profile hit rate, the held-out validation boundary,
  and the additive independent baseline.

## Inference

A paired feasible-profile hit-rate difference tests whether conditioning later
decisions on upstream context adds useful selection value beyond additive
query-only scoring. Holding the query features, executable profiles, quality
rule, query IDs, and cost contract constant isolates the interaction-context
condition as the intended comparison.

## Project preference

The maintainer accepted a five-percentage-point practical margin and a paired
95% bootstrap lower-bound rule, matching the project's signal-test standard.
The project prefers a set-valued quality-feasible target and conservative
reporting of practical null, inconclusive, and negative interaction results.

## Consequences and uncertainty

- This ADR defines a research gate and does not authorize an interaction-aware
  controller, scorer, evaluator implementation, or production infrastructure.
- The result is bounded to the pinned model, backend, execution configuration,
  manifests, and non-final MATH validation phase. It does not establish
  generalization to MATH-500, MMLU-Pro, another model, or another hardware
  environment.
- The exact upstream-context representation, planner/scorer architecture,
  feature encoding, tie handling, bootstrap seed, prompt formatting, answer
  normalization, and raw-artifact schema remain downstream protocol details.
- This ADR does not select numerical per-dimension cost ceilings or a scalar
  cost coefficient. Such choices must be resolved before an implementation
  applies a budgeted interaction test.

## Related records

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0005: Stage-1 4/8 group execution semantics](0005-stage-1-group-execution-semantics.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [ADR-0010: Stage-1 oracle profile enumeration and sampling](0010-stage-1-oracle-profile-enumeration-and-sampling.md)
- [ADR-0011: Stage-1 signal test and success threshold](0011-stage-1-signal-test-and-success-threshold.md)
- [Design the interaction test and predeclare its success threshold](https://github.com/weige15/QBitPlan/issues/17)
- [Set Stage-1 stop/continue rules](https://github.com/weige15/QBitPlan/issues/19)

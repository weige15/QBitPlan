# ADR-0009: Stage-1 degradation and epsilon selection

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Define degradation and the epsilon-selection rule](https://github.com/weige15/QBitPlan/issues/13).
ADR-0006 selects MATH/MATH-500 and MMLU-Pro for Stage 1. MATH supplies the
permitted non-final calibration/training and validation phases; MATH-500 and
MMLU-Pro are final paired evaluation artifacts. ADR-0007 selects unquantized
BF16 execution as the high-precision reference and keeps it distinct from
external task ground truth. ADR-0008 selects task correctness as primary and
keeps reference-relative diagnostics separate.

The ticket's predeclared rule requires degradation to be computable from the
primary metric and declared diagnostics, to distinguish reference comparison
from ground truth, to use paired queries consistently, and to pre-register
epsilon without using final evaluation IDs.

## Decision

### Degradation

For a paired query `q` in dataset `d` and executable profile `p`, let
`c(d,q,p)` be the externally judged task-correctness result, with values in
`{0, 1}`. Define the paired per-query degradation relative to the BF16
reference as:

`d(d,q,p) = c(d,q,BF16) - c(d,q,p)`

Thus, `1` is a reference-correct/profile-incorrect loss, `0` is no change,
and `-1` is a profile correction of a BF16 error. For each evaluated dataset,
aggregate the paired values as:

`D_d(p) = (1 / n_d) * sum_q d(d,q,p)`

The equal-weight macro summary is `D_macro(p) = (1 / |S|) * sum_d D_d(p)`
over the datasets in the evaluated phase. Per-dataset values remain the
decision-facing quality results; the macro value is a summary and cannot hide
a dataset-level failure.

For every paired query, report the declared BF16-relative diagnostics:
output-distribution divergence using the declared logits or log-probabilities,
answer-flip indicators, and any later-approved hidden-state distance. These
diagnostics explain model-relative behavior but do not replace task
correctness or independently impose a quality gate.

### Epsilon and feasibility rule

Pre-register one absolute epsilon before accessing any final evaluation
manifest:

`epsilon = 0.01`

This is one percentage point in task-correctness units. The value is fixed by
the protocol and is not tuned after seeing final results. Any empirical
feasibility check used while selecting or enumerating oracle profiles may use
only the permitted non-final MATH calibration/training and validation
records; final query IDs are never used to select epsilon, profiles, or
thresholds.

A profile or oracle selection is within the accepted quality threshold only
when every evaluated dataset satisfies `D_d(p) <= 0.01`. A profile is flagged
as under-precision when any evaluated dataset has `D_d(p) > 0.01`. The same
per-dataset gate and epsilon apply to oracle feasibility; the equal-weight
macro result is reported secondarily.

## Rule application

- **Primary metric — pass.** Degradation uses external task correctness, not
  agreement with BF16.
- **Reference separation — pass.** BF16 supplies the comparison outcome;
  external targets supply correctness.
- **Paired reporting — pass.** Quantized and BF16 outcomes are compared for
  the same immutable query records before dataset aggregation.
- **Threshold boundary — pass.** The fixed 0.01 threshold is declared before
  final evaluation and is applied identically to under-precision and oracle
  feasibility.
- **Final-data isolation — pass.** Threshold and profile selection use no
  final query IDs; MMLU-Pro remains final-only under ADR-0006.
- **Diagnostic separation — pass.** Diagnostics are retained and reported,
  but cannot silently replace the primary quality metric.

## Source facts, inference, and project preference

**Source facts.** `CONTEXT.md` defines degradation relative to the
high-precision reference and distinguishes that reference from ground truth.
`SCIENTIFIC_STANDARDS.md` requires paired query sets, makes task-appropriate
correctness primary for externally judged answers, keeps KL/answer-flip/
hidden-state measures diagnostic unless promoted, and forbids final IDs from
threshold selection. ADR-0006 records the accepted phase boundaries; ADR-0007
records the BF16 reference; ADR-0008 records the primary/diagnostic metric
roles.

**Inference.** The paired correctness difference is computable for every
profile and preserves improvements over BF16 as negative degradation. A
per-dataset gate is necessary because an equal-weight macro value can conceal
a failure on one dataset.

**Project preference.** The maintainer selected a fixed one-percentage-point
absolute tolerance and chose not to turn model-relative diagnostics into
additional hard quality constraints.

## Consequences and retained uncertainty

- Oracle profile enumeration and sampling remain governed by [Define oracle profile enumeration and sampling](https://github.com/weige15/QBitPlan/issues/15).
- Confidence intervals, prompt formatting, answer normalization, diagnostic
  positions, and any hidden-state measurement details remain downstream
  protocol decisions.
- [Reconcile the Stage-1 metric ADR with the accepted task suite](https://github.com/weige15/QBitPlan/issues/22)
  corrected the historical GPQA references in ADR-0008. The current metric
  ADR and ADR-0006 now agree on the accepted MMLU-Pro plus MATH/MATH-500
  suite; this ADR follows that corrected boundary.
- This ADR defines the research decision only; it does not implement an
  evaluator, router, oracle, or experiment runner.

## Related records

- [Select the primary quality metric and diagnostic metrics](https://github.com/weige15/QBitPlan/issues/9)
- [Adopt the high-precision reference and its role relative to ground truth](https://github.com/weige15/QBitPlan/issues/6)
- [Select the task suite, dataset revisions, and immutable query-ID splits](https://github.com/weige15/QBitPlan/issues/11)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0007: Stage-1 high-precision reference and ground-truth role](0007-stage-1-high-precision-reference.md)
- [ADR-0008: Stage-1 primary and diagnostic metrics](0008-stage-1-primary-and-diagnostic-metrics.md)

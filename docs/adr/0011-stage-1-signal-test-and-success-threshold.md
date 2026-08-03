# ADR-0011: Stage-1 signal test and success threshold

- Status: ACCEPTED
- Date: 2026-08-04
- Decision owners: project maintainers

## Context

The originating decision ticket is [Design the signal test and predeclare its
success threshold](https://github.com/weige15/QBitPlan/issues/18). The Stage-1
pilot has eight binary, ordered layer-group decisions under
[ADR-0005](0005-stage-1-group-execution-semantics.md). The accepted data
boundary in [ADR-0006](0006-stage-1-task-suite-and-query-splits.md) provides
7,500 non-final MATH source-training records and 4,500 non-final MATH
validation records; MATH-500 and MMLU-Pro remain final-only. The paired
BF16-relative degradation definition and `epsilon = 0.01` are fixed by
[ADR-0009](0009-stage-1-degradation-and-epsilon.md), and the executable profile
universe is fixed by [ADR-0010](0010-stage-1-oracle-profile-enumeration-and-sampling.md).

The signal gate asks whether a query-conditioned predictor can select a
query-specific precision profile from information available before that
profile is executed. It must be compared with a static profile and the
mandatory additive independent baseline on held-out paired queries. This gate
is a falsification test for predictive precision-shape signal, not an
implementation or a claim of measured serving benefit.

## Decision

### Primary estimand and target

Precision-profile prediction is the primary signal estimand. Quantization
difficulty and quality-cost behavior are secondary practical outcomes.

The target is set-valued rather than a forced single oracle winner. For each
query, the target set consists of executable profiles that satisfy the
accepted quality-feasibility rule under the declared Stage-1 protocol. The
predictor is evaluated as one top-1 executable profile per query. A prediction
is a hit when it belongs to that query's quality-feasible target set.

The signal test does not use human or published difficulty labels as routing
targets. It does not turn the high-precision reference into ground truth.

### Data and evaluation boundary

- Fit profile targets and the predictor using the 7,500 non-final MATH
  source-training records.
- Evaluate the signal gate on the 4,500 held-out non-final MATH validation
  records, using paired query IDs and the same profile universe and BF16
  reference for every method.
- Do not use MATH-500 or MMLU-Pro query IDs for profile construction,
  predictor fitting, threshold selection, or declaring the signal gate. Those
  final-only artifacts remain downstream confirmation evidence.

### Causal feature boundary

The primary predictor may use only query content and structure available before
profile selection, including deterministic length/format features and a fixed
query embedding computed before execution. It must exclude:

- human or published difficulty labels as routing targets;
- ground-truth answers and reference or quantized outcomes;
- profile outcomes, hidden states, logits, generated tokens, and any other
  post-decision or post-execution information.

Task or dataset labels may be reported as ablations, but are not part of the
primary feature set.

### Baselines

- **Static baseline:** one query-independent executable profile selected and
  frozen using only non-final MATH training data.
- **Independent baseline:** a query-conditioned additive per-group scorer or
  MCKP planner using the same causal query features, with no upstream context
  or interaction terms.

Both baselines use the same profile universe, BF16 reference, quality rule,
held-out queries, and reporting contract as the signal predictor.

### Primary metric and predeclared success rule

The primary metric is feasible-profile hit rate on the held-out validation
queries. Report the paired difference between the signal predictor and each
baseline over the same query IDs.

Declare the signal gate successful only when both conditions hold against both
the static and independent baselines:

1. the absolute hit-rate improvement is at least 5 percentage points; and
2. the paired 95% bootstrap lower bound for the improvement is above zero.

The bootstrap resamples validation queries as the paired unit. The 5-point
margin is a practical project threshold, not a source-derived scientific
constant.

### Outcome taxonomy

- **Success:** both baseline comparisons meet the 5-point margin and the
  lower-bound rule.
- **Practical null:** the observed improvement is below 5 points, even if it
  is positive.
- **Inconclusive:** the uncertainty interval includes zero.
- **Negative:** the signal predictor is worse than a baseline.

Report all four possible outcomes honestly, together with per-query paired
results and secondary precision-shape and quality-cost/Pareto diagnostics.

## Source facts

- `SCIENTIFIC_STANDARDS.md` requires causal information boundaries, paired
  query evaluation, primary task-appropriate quality metrics, and explicit
  reporting of negative and null findings.
- ADR-0005 fixes the eight ordered binary group choices and hardware-executable
  profile semantics.
- ADR-0006 fixes the non-final MATH training/validation boundary and keeps
  MATH-500 and MMLU-Pro final-only.
- ADR-0009 fixes paired BF16-relative degradation and `epsilon = 0.01` for
  quality feasibility.
- ADR-0010 fixes exhaustive enumeration of the 256 analytical profiles and
  the separation of quality evaluation from sampled hardware-cost evidence.
- The initial research brief is unverified research input. Its relevant
  hypothesis is that query and early-execution signals may predict precision
  shape; it is not used as independent evidence for this decision.

## Inference

A set-valued target avoids treating equivalent quality-feasible executable
profiles as different scientific answers. A top-1 hit-rate endpoint tests
whether the predictor can select a useful profile, while the static and
independent comparisons establish whether the result contains query-specific
signal beyond a fixed allocation or additive group scoring.

The paired validation design makes each baseline comparison query-matched and
keeps the signal claim separate from any later systems claim.

## Project preference

The project prioritizes precision-profile prediction as the main Stage-1
signal question. It prefers a practical absolute margin of 5 percentage
points, a paired bootstrap uncertainty rule, set-valued feasible targets, and
an outcome taxonomy that distinguishes practical null, inconclusive, and
negative results.

## Consequences and uncertainty

- This ADR defines a research gate and does not authorize a router, scorer,
  evaluator implementation, or production infrastructure.
- The gate is bounded to the pinned model, backend, manifests, and non-final
  MATH validation phase. It does not establish generalization to MATH-500,
  MMLU-Pro, another model, or another execution environment.
- The exact feature encoding, query embedding implementation, profile tie
  handling, bootstrap seed, and raw-artifact schema remain protocol-document
  details to be consolidated before execution.
- The accepted quality ADR defines the formal `epsilon` gate at dataset
  aggregation level. The exact construction of per-query set-valued profile
  targets from that gate must be made explicit in the consolidated protocol;
  this ADR does not silently introduce a different aggregation rule.
- Secondary cost results must retain the separate cost-vector and evidence
  rules in ADR-0003. A hit-rate result alone is not a measured systems claim.

## Related records

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0005: Stage-1 4/8 group execution semantics](0005-stage-1-group-execution-semantics.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [ADR-0010: Stage-1 oracle profile enumeration and sampling](0010-stage-1-oracle-profile-enumeration-and-sampling.md)
- [Design the interaction test and predeclare its success threshold](https://github.com/weige15/QBitPlan/issues/17)
- [Set Stage-1 stop/continue rules](https://github.com/weige15/QBitPlan/issues/19)

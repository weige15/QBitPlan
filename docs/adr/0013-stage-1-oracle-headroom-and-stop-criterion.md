# ADR-0013: Stage-1 oracle-headroom test and stop criterion

- Status: ACCEPTED
- Date: 2026-08-04
- Decision owners: project maintainers

## Context

The originating decision ticket is [Design the oracle-headroom test and its
predeclared stop criterion](https://github.com/weige15/QBitPlan/issues/21).
The executable profile universe and query-phase boundaries are fixed by
[ADR-0010](0010-stage-1-oracle-profile-enumeration-and-sampling.md). The
primary task-correctness metric is fixed by
[ADR-0008](0008-stage-1-primary-and-diagnostic-metrics.md), the paired
reference-relative degradation and `epsilon = 0.01` by
[ADR-0009](0009-stage-1-degradation-and-epsilon.md), and the separate
componentwise cost vector by
[ADR-0003](0003-stage-1-cost-contract.md).

The test estimates attainable per-query quality-cost headroom over a frozen
static profile. It is an offline oracle test, not an online controller,
implementation authorization, or serving result.

## Decision

### Data and profile roles

- Use all 7,500 non-final MATH source-training records for oracle-outcome
  construction and for selecting the single static profile under the accepted
  static-baseline procedure in [ADR-0011](0011-stage-1-signal-test-and-success-threshold.md).
- Use all 4,500 non-final MATH validation records for quality feasibility and
  the headroom assessment. MATH-500 and MMLU-Pro query IDs remain final-only
  and are not used for profile construction, threshold selection, or this
  gate.
- Let `P_exec` be the hardware-executable profiles from ADR-0010. For the
  validation phase, define:

  `F_val = { p in P_exec : D_d,val(p) <= 0.01 for every evaluated dataset d }`.

  In the non-final headroom gate the evaluated dataset is MATH. The same
  per-dataset gate, not the equal-weight macro summary alone, controls
  feasibility.
- The static profile is frozen before validation outcomes are inspected. If
  it fails the validation quality gate, the headroom result is marked
  invalid/inconclusive; it is not reselected on validation.

### Oracle headroom estimand

For each query `q` in the fixed hardware-cost frame, evaluate the complete
ordered execution of every profile in `F_val` and retain the query's
non-dominated cost frontier under the included cost dimensions `J`:

`O_q = Pareto-min({ C_J(q, p) : p in F_val })`.

Let `p_static` be the frozen static profile. Define the primary opportunity
indicator:

`H_q = 1` when there exists `p in O_q` such that
`C_J(q, p) <= C_J(q, p_static)` componentwise and the inequality is strict in
at least one included dimension; otherwise `H_q = 0`.

The primary headroom estimand is the Pareto-dominance opportunity rate:

`H = mean_q H_q`.

This strict-dominance endpoint counts only unambiguous componentwise
improvement. Incomparable trade-offs do not count as a primary win, but the
complete per-query Pareto frontiers and per-dimension attainable deltas are
reported as secondary evidence. No arbitrary scalar cost coefficient or
average-bit substitute is introduced.

### Cost evidence and evaluation frames

- Quality feasibility is evaluated exhaustively on all 4,500 non-final MATH
  validation records.
- Direct or accepted-estimated cost headroom is evaluated on the fixed 256-ID
  per-phase frame frozen by ADR-0010. The same query IDs and complete
  static/oracle comparisons are used for paired cost evidence.
- An optional quick-look preview may use 64 of those frozen IDs, selected
  deterministically as every fourth query in canonical query-ID order. The
  preview is exploratory only; it cannot satisfy the stop criterion, alter
  the frame, or justify controller work.
- Freeze one common set `J` of cost dimensions before evaluation. Include only
  dimensions covered by an accepted measurement or estimation procedure for
  both compared variants. Unavailable dimensions are marked
  `omitted/unavailable` with a reason; they are not recorded as zero, imputed,
  or replaced by average bit-width. Results are bounded to `J` and its
  evidence classes. If no common dimension is available, the result is not
  interpretable.
- This is gross execution headroom. Offline oracle selection has no online
  controller/probe/feedback path; any later controller must account for its
  own overhead in the downstream amortization and systems decisions.

### Uncertainty and predeclared stop criterion

Resample complete query records as the paired unit in a predeclared 95%
bootstrap over the 256-query cost frame. Each replicate carries the complete
static/oracle Pareto comparison for the sampled query; profiles within a
query are not independent bootstrap units.

Declare the oracle-headroom gate **useful** only when both conditions hold:

1. `H >= 0.05`, an absolute opportunity rate of at least five percentage
   points; and
2. the paired 95% bootstrap lower bound for `H` is greater than zero.

Classify outcomes with this precedence:

- **Useful headroom:** `H >= 0.05` and lower bound greater than zero.
- **Practical null:** lower bound greater than zero but `H < 0.05`.
- **Inconclusive:** the uncertainty interval includes zero, regardless of the
  point estimate.

The five-point margin is a project practical threshold aligned with the
accepted signal and interaction gates. The opportunity-rate endpoint is
non-negative; signed per-dimension deltas may additionally show negative
directional results but do not override the primary classification.

## Source facts, inference, and project preference

**Source facts.** `SCIENTIFIC_STANDARDS.md` requires task-appropriate primary
correctness, paired query evaluation, explicit reporting of null findings,
and quality-cost Pareto reporting. ADR-0003 fixes the six-dimensional cost
vector, componentwise comparison, and omission rules. ADR-0006 fixes the
non-final MATH training/validation boundary and final-only MATH-500/MMLU-Pro.
ADR-0009 fixes paired BF16-relative degradation and `epsilon = 0.01` for
per-dataset feasibility. ADR-0010 fixes exhaustive `P_exec` quality
evaluation, complete ordered executions, and the fixed 256-query hardware
cost frame. ADR-0011 and ADR-0012 use paired validation uncertainty and a
five-point practical margin for earlier gates.

**Inference.** Allowing the offline oracle to inspect validation outcomes
estimates attainable headroom unavailable to an online controller. A complete
query-level cost vector and strict Pareto dominance preserve the accepted
cost-comparison policy without inventing a scalar objective. The resulting
test is an upper-bound feasibility gate, not evidence that a controller will
generalize or amortize its runtime overhead.

**Project preference.** The project prefers full non-final training data, a
single frozen static comparator, conservative strict dominance, paired query
uncertainty, a five-percentage-point practical margin, explicit null and
inconclusive outcomes, and no controller authorization from gross oracle
headroom alone.

## Consequences and retained uncertainty

- A useful result is necessary evidence for considering later controller work,
  not sufficient evidence for implementation, amortization, or serving
  benefit. The ordering and final stop/continue policy remain with
  [Set Stage-1 stop/continue rules](https://github.com/weige15/QBitPlan/issues/19).
- The result is bounded to the pinned model, backend, execution configuration,
  non-final MATH records, fixed cost frame, and included cost dimensions. It
  does not establish generalization to MATH-500, MMLU-Pro, another model,
  another backend, or another hardware environment.
- Bootstrap replicate count/seed, prompt formatting, answer normalization,
  detailed measurement procedures, and the exact static-profile tie procedure
  remain consolidated-protocol fields and must be fixed before evaluation.
- The 64-query preview is not completion evidence and cannot replace the
  required 256-query gate.
- This ADR does not implement an evaluator, oracle, controller, or experiment
  runner.

## Related records

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0008: Stage-1 primary and diagnostic metrics](0008-stage-1-primary-and-diagnostic-metrics.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [ADR-0010: Stage-1 oracle profile enumeration and sampling](0010-stage-1-oracle-profile-enumeration-and-sampling.md)
- [ADR-0011: Stage-1 signal test and success threshold](0011-stage-1-signal-test-and-success-threshold.md)
- [ADR-0012: Stage-1 interaction test and success threshold](0012-stage-1-interaction-test-and-success-threshold.md)
- [Design the oracle-headroom test and its predeclared stop criterion](https://github.com/weige15/QBitPlan/issues/21)

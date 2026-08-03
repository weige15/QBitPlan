# ADR-0006: Stage-1 task suite and immutable query-ID splits

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Select the task suite, dataset revisions,
and immutable query-ID splits](https://github.com/weige15/QBitPlan/issues/11).
The feasibility boundary is recorded in
[ADR-0002](0002-task-suite-dataset-feasibility.md) and the detailed source
comparison is in the [Issue 4 research note](../research/issue-4-task-suite-feasibility.md).
The pilot model is fixed by
[ADR-0004](0004-stage-1-pilot-model.md).

The decision must apply the ticket's predeclared rule: choose the smallest
suite that tests broad task behavior and interaction-sensitive behavior while
providing external ground truth, reproducible revisions, sufficient held-out
queries, and no reuse of final query IDs for calibration, threshold selection,
or profile-library construction.

## Decision

Stage 1 uses the following task suite:

- **MMLU-Pro** for broad domain and task coverage; and
- **MATH/MATH-500** for structured reasoning with exact answer evaluation.

The suite is intentionally the constrained cross-dataset form: MATH supplies
the calibration/training and validation phases, while MMLU-Pro is a locked
final-only breadth evaluation. GPQA Diamond, LiveCodeBench, and WikiText-103
are excluded from the Stage-1 suite to keep the selection minimal. They remain
possible later or diagnostic candidates, not part of this accepted protocol.

### Pinned artifacts and phase roles

| Phase | Dataset and artifact | Query records |
| --- | --- | --- |
| Calibration/training | MATH source Git revision `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | All 7,500 source training problems |
| Validation | The same MATH revision | The 4,500 original test records whose source IDs are not in the final MATH-500 manifest |
| Final paired evaluation | MATH-500 Hub revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | All 500 pinned `unique_id` records |
| Final paired evaluation | MMLU-Pro Hub revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` | All 12,032 pinned test `question_id` records |

The MATH validation count follows the source's 5,000-record test set after
excluding the 500 final MATH-500 records. MMLU-Pro is a final-only
component; its calibration/training and validation are supplied by the pinned MATH records under the accepted cross-dataset
policy. GPQA Diamond is not part of this suite.

### Immutable manifests and query identity

Each phase has an immutable manifest created before calibration, threshold
selection, or profile-library construction. Every manifest row records:

- dataset and configuration;
- exact source revision and source split;
- phase;
- query ID; and
- a hash of the canonical query record.

Source-defined IDs are used wherever available: MMLU-Pro `question_id` and
MATH/MATH-500 `unique_id` (or the corresponding source-relative record path
where the raw MATH source exposes that identity). The exact pinned source row
remains the query record. A manifest integrity hash is computed over the
UTF-8 RFC-8785-canonicalized query record; it is not a substitute for a
source-defined query ID.

Final query IDs are never used for calibration/training, validation,
threshold selection, or profile-library construction. All methods are
evaluated on the same immutable final manifests for paired comparisons. Raw
manifests and source records are immutable; derived summaries must point back
to them.

## Rule application

- **Broad behavior — pass by accepted coverage policy.** MMLU-Pro supplies
  broad domain/task coverage.
- **Structured reasoning — pass.** MATH/MATH-500 provides open-ended
  reasoning with answer-equivalence evaluation.
- **External ground truth — pass.** MMLU-Pro provides keyed answers and MATH
  provides answer-equivalence evaluation.
- **Pinned revisions — pass.** Every selected artifact has an exact revision
  recorded above.
- **Held-out phase separation — pass under the accepted cross-dataset policy.**
  MATH supplies calibration/training and validation; MATH-500 and MMLU-Pro
  remain final evaluation artifacts, with MMLU-Pro final-only.
- **Leakage control — pass as a protocol boundary, not a claim of no prior
  model exposure.** Final manifests are fixed before profile work and are not
  reused in earlier phases. Public benchmark contamination remains a reported
  limitation.

## Source facts, inference, and project preference

**Source facts.** The Issue 4 research note verifies the listed source
revisions, task formats, target fields, record counts, MATH/MATH-500 split
structure, and MMLU-Pro `question_id`. It also records that MMLU-Pro has no
standalone calibration/training split. GPQA Diamond was considered but is not
selected here.

**Inference.** The non-final MATH test records form a disjoint validation set,
and MMLU-Pro can be a final-only component relying on MATH for
calibration/training and validation. This makes MMLU-Pro evaluation a
cross-dataset transfer test rather than an in-domain estimate. These are
protocol inferences, not source claims.

**Project preference.** The maintainer accepted the minimal suite matching the
intended broad and structured-reasoning coverage. MMLU-Pro is retained as a
final-only breadth check; GPQA Diamond is excluded to avoid adding a small,
gated, derived-ID-only final component to Stage 1. The initial research brief
recommendation to use non-test MMLU-Pro for calibration was not adopted
because the verified pinned MMLU-Pro artifact has no standalone
calibration/training split.

## Consequences and retained uncertainty

- Stage 1 does not claim generalization beyond these pinned artifacts, query
  manifests, the selected model revision, and the later accepted execution,
  metric, and cost protocol.
- MMLU-Pro final-only results must be reported separately as cross-dataset
  transfer evidence, not pooled with MATH as if they were in-domain.
- Exact prompt formatting, primary and diagnostic metrics, high-precision
  reference, degradation threshold, and execution/evaluation commands remain
  governed by their downstream decisions.
- A future change to the suite, artifact revisions, phase boundaries, or ID
  construction requires a superseding ADR and new immutable manifests.

## Evidence

- [Issue 4 feasibility resolution](https://github.com/weige15/QBitPlan/issues/4)
- [Issue 4 research note](../research/issue-4-task-suite-feasibility.md)
- [ADR-0002: Task-suite dataset feasibility boundary](0002-task-suite-dataset-feasibility.md)
- [ADR-0004: Stage-1 pilot model and immutable revision](0004-stage-1-pilot-model.md)

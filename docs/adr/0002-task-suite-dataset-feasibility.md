# ADR-0002: Task-suite dataset feasibility boundary

- Status: PROPOSED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

Stage-1 task-suite selection requires pinned datasets, externally defined
targets, and immutable query-ID separation before calibration/training,
validation, and final evaluation. The originating research ticket is
[Verify task-suite datasets, revisions, and immutable split feasibility](https://github.com/weige15/QBitPlan/issues/4).
Its cited evidence is recorded in
[the issue-4 research note](../research/issue-4-task-suite-feasibility.md).

The initial research brief is background input and remains unverified. This
ADR records feasibility evidence only; it does not select the project suite,
metric protocol, or cross-dataset split policy.

## Predeclared feasibility rule

Retain a task/dataset candidate for policy consideration only when it has:

1. a citable primary source;
2. a stable revision or snapshot;
3. externally defined ground truth;
4. a reproducible query-ID manifest; and
5. a feasible separation of calibration/training, validation, and final
   evaluation queries.

## Evidence summary

| Candidate | Verified feasibility boundary | Rule result |
| --- | --- | --- |
| MMLU-Pro | Pinned Hub data has explicit `question_id`, answer keys, validation, and test, but no standalone calibration/training split. | Conditional final-evaluation component; not a standalone pass. |
| MATH/MATH-500 | Pinned source artifacts provide train/test material and MATH-500 exposes stable `unique_id` values, answer/solution targets, and a held-out evaluation artifact. | Pass for a separately declared three-way split protocol. |
| GPQA Diamond | Pinned gated artifact and external expert-validated answers are available, but the source does not expose a source-defined query-ID field or three-way split. | Conditional final-only component; derived IDs and cross-dataset calibration require an accepted protocol. |
| LiveCodeBench | Pinned repository/Hub snapshots expose `question_id`, contest dates, hidden tests, and versioned releases. | Conditional pass by inference; temporal boundaries must be accepted and cumulative releases must be de-duplicated. |
| WikiText-103 raw | Pinned artifact provides train, validation, and test text splits. A manifest can be derived from pinned row identity and content hash; perplexity must be predeclared as the primary metric. | Pass for a separately declared language-modeling protocol. |

The classifications above distinguish source facts from inference. Public
benchmark text and common pretraining corpora create leakage risk even when a
revision is pinned; pinning prevents drift, not prior model exposure.

## Consequences and downstream record

- No project suite was selected by this feasibility ADR or by ticket #4. The
  accepted Stage-1 selection is recorded in
  [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md).
- ADR-0006 permits MMLU-Pro as a final-only component supplied by MATH
  calibration/training and validation under an explicit cross-dataset policy.
- ADR-0006 fixes the selected revisions, phase boundaries, query-ID manifest
  rule, and canonical query-record integrity hashing required before profile
  work or threshold selection.
- GPQA access restrictions and non-disclosure requirements remain relevant if
  that candidate is considered in a future superseding decision.
- LiveCodeBench release dates provide a contamination mitigation, not a
  guarantee of no leakage.

## Research sources

The detailed comparison matrix and source/inference/preference separation are
in [docs/research/issue-4-task-suite-feasibility.md](../research/issue-4-task-suite-feasibility.md).
The source citations there include the official dataset repositories/cards and
primary papers for MMLU-Pro, MATH/MATH-500, GPQA, LiveCodeBench, and
WikiText-103.

# ADR-0010: Stage-1 oracle profile enumeration and sampling

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Define oracle profile enumeration and
sampling](https://github.com/weige15/QBitPlan/issues/15). ADR-0005 fixes eight
ordered contiguous layer groups, with one weight-only choice from `{4, 8}` per
group. ADR-0006 fixes the Stage-1 task phases and immutable query manifests;
ADR-0009 fixes the paired degradation definition and `epsilon = 0.01`; and
ADR-0003 fixes the separate cost vector and Pareto comparison policy.

The ticket's predeclared rule requires exhaustive enumeration of the 256
profiles when feasible for this eight-group pilot. If any measurement is
sampled, the sampling frame, seed, coverage rule, and stopping rule must be
declared before measurement. Final evaluation IDs must not influence profile
construction or threshold selection.

## Decision

### Profile universe and canonical identity

Enumerate the complete analytical universe

`P = {4, 8}^8`,

which contains `2^8 = 256` ordered profiles. The profile order is `g0` through
`g7`. Encode `4-bit` as `0` and `8-bit` as `1`; enumerate profile IDs
lexicographically from `00000000` through `11111111`. Profile IDs are assigned
before quality or cost outcomes are observed.

Enumeration is separate from execution feasibility. The oracle execution set
`P_exec` contains only profiles for which all eight group transforms and the
complete ordered forward path execute successfully on the pinned model,
revision, backend, and configuration. An infeasible profile is excluded; it
is not replaced by BF16, a nearby profile, or a silently altered profile.

### Phase roles and query boundaries

- **Calibration/training:** use all 7,500 non-final MATH source training
  records to construct per-query oracle outcomes from `P_exec`.
- **Validation:** use all 4,500 non-final MATH validation records as a held-out
  assessment of retained profiles and oracle headroom. Validation does not
  change the profile universe or the pre-registered threshold.
- **Final evaluation:** MATH-500 and MMLU-Pro remain untouched until final
  paired evaluation. Their immutable query IDs are never used for profile
  construction, profile-library construction, or threshold selection.

Quality and degradation measurements are exhaustive over every profile in
`P_exec` and every query in the two permitted non-final MATH phases. The same
BF16 reference and the same query record are used for each paired comparison.

### Conditional upstream contexts

Each profile is evaluated as one ordered eight-group execution. A later group
receives the upstream hidden state produced by the earlier choices in that same
profile execution. Groups are not scored in isolation, and upstream contexts
are never mixed across profiles or queries. This preserves the interaction and
causality boundary in `SCIENTIFIC_STANDARDS.md`.

### Hardware-cost sampling

Hardware-cost observations may use a separate fixed frame while quality
measurements remain exhaustive. For each non-final MATH phase:

1. sort the immutable manifest by its canonical source-defined query ID;
2. partition the ordered rows into 256 contiguous, as-even-as-possible
   strata;
3. within each stratum, select the row with the smallest SHA-256 digest of
   the UTF-8 string `20260803|<phase>|<query_id>`.

This selects exactly 256 query IDs per phase. The frame is frozen before any
profile execution. Every profile in `P_exec` and the BF16 reference is run on
the same selected IDs for paired cost comparison. Measurement stops after the
fixed 256 IDs per phase; there is no adaptive addition, early stopping, or
result-dependent resampling.

Cost results follow ADR-0003. A dimension that is not measured or otherwise
covered by an accepted estimation procedure is reported as
`omitted/unavailable` with its reason. It is not reported as zero, imputed, or
replaced by average bit-width. Sampled cost evidence is bounded to the declared
frame and does not establish an unmeasured systems claim outside it.

## Source facts, inference, and project preference

**Source facts.** ADR-0005 fixes the eight binary group choices and the
hardware-executable TorchAO semantics. ADR-0006 fixes the non-final MATH
calibration/validation records and final-only MATH-500/MMLU-Pro manifests.
ADR-0009 fixes paired reference-relative degradation and `epsilon = 0.01`.
ADR-0003 fixes the six-dimensional cost vector, evidence labels, and
componentwise/Pareto comparison. `SCIENTIFIC_STANDARDS.md` requires causal
upstream information, immutable query manifests, paired evaluation, and
recorded seeds.

**Inference.** Eight binary choices yield 256 analytical profiles. Evaluating a
profile as a complete ordered execution is required to preserve upstream
interaction effects. A shared fixed cost frame makes profile cost observations
paired and reproducible.

**Project preference.** The project prefers exhaustive non-final quality
measurement, a canonical profile identity, deterministic fixed-size cost
sampling, and conservative omission of unavailable cost dimensions. The
sampling rule is a protocol convenience for hardware cost, not a claim that
256 queries are statistically sufficient for every future hardware question.

## Consequences and retained uncertainty

- The oracle set is bounded to the selected model, revisions, backend,
  manifests, and execution configuration.
- No final result, threshold, or profile-library record may be constructed
  from MATH-500 or MMLU-Pro IDs before final evaluation.
- The sampled cost frame supports paired cost comparison only on its declared
  IDs. Confidence intervals, prompt formatting, answer normalization, and
  downstream oracle-headroom success criteria remain governed by later
  protocol decisions.
- Profile clustering, medoid count, router architecture, and implementation
  remain outside this decision.

## Related records

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0005: Stage-1 4/8 group execution semantics](0005-stage-1-group-execution-semantics.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [Design the oracle-headroom test and its predeclared stop criterion](https://github.com/weige15/QBitPlan/issues/21)

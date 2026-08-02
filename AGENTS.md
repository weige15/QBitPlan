# QBitPlan repository instructions

## Read before working

Read these sources in order:

1. `README.md`
2. `CONTEXT.md`
3. `SCIENTIFIC_STANDARDS.md`
4. Relevant ADRs under `docs/adr/`
5. The originating GitHub issue and its comments

Read `docs/research/initial-research-brief.md` when research rationale is
needed. It is background input, not an accepted specification.

## Source-of-truth hierarchy

- Accepted technical or research decision: relevant ADR
- Project terminology: `CONTEXT.md`
- Scientific and reporting invariants: `SCIENTIFIC_STANDARDS.md`
- Task scope and acceptance criteria: originating GitHub issue
- Research background: files under `docs/research/`
- Public orientation: `README.md`

When sources conflict, report the conflict. Do not silently reconcile them.

## Current project phase

The project is defining its v0.1 experimental contract.

Do not implement production infrastructure, choose unresolved dependencies,
or run expensive experiments unless the originating issue explicitly
authorizes it.

## Work boundaries

For requests to inspect, explain, research, review, or plan:

- Inspect the relevant repository material.
- Report findings without modifying files unless edits were requested.

For requests to implement an approved issue:

- Make only the requested in-scope changes.
- Run relevant non-destructive validation.
- Stop before destructive actions, external publication, costly full-scale
  experiments, or material scope expansion.

## Research boundaries

- Do not describe layer precision decisions as independent.
- Do not equate human task difficulty with quantization difficulty.
- Do not use final-evaluation queries for training, threshold selection,
  profile clustering, or hyperparameter tuning.
- Do not claim memory or latency improvement from average bit-width.
- Do not use a later hidden state to justify an earlier precision decision.
- Do not fabricate experimental measurements when the required hardware,
  model, dataset, or backend is unavailable.

## Engineering workflow

- Every implementation change begins from a GitHub issue or accepted spec.
- Use one branch per issue.
- Prefer small vertical slices that are independently verifiable.
- Use TDD for deterministic behavior at agreed public seams.
- Run a smoke configuration before any authorized expensive experiment.
- Keep raw experiment artifacts immutable.
- Record decisions that outlive one issue in an ADR.
- Do not create abstractions for deferred features.

## Completion requirements

Before considering an implementation issue complete:

- Run targeted tests and repository-wide checks that exist.
- Record the exact validation commands and outcomes.
- Confirm the implementation matches the originating issue.
- Confirm it complies with `SCIENTIFIC_STANDARDS.md`.
- Document unavailable validations honestly.

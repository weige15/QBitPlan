# QBitPlan scientific standards

These are non-negotiable standards for QBitPlan experiments and claims. They
define how evidence is classified, how comparisons are made, and what may be
claimed. They do not select a model, dataset, environment, backend, hardware,
or controller architecture.

## 1. Evidence classes

Every reported result must identify itself as one of:

- analytical;
- simulated;
- lookup-table estimated;
- directly measured.

These categories must not be blurred. In particular, simulated or
fake-quantized execution is not direct evidence of memory or latency
improvement.

## 2. Reference, ground truth, and metrics

- Every experiment declares its high-precision reference and its numeric
  format. The choice is OPEN until an experiment protocol accepts it.
- Ground truth is the external task target. A reference execution is a model
  comparison point; it is not ground truth.
- Every experiment declares a task-appropriate primary quality metric before
  evaluation.
- For tasks with externally judged answers, task correctness is the final
  quality metric.
- For language-modeling tasks without task labels, perplexity may be the
  primary quality metric when declared by the protocol.
- KL divergence, answer flips, hidden-state distances, and related measures
  are diagnostic unless a later accepted protocol explicitly promotes one.
- A diagnostic metric must not silently replace the primary quality metric.
- Degradation is reported relative to the declared high-precision reference.
  The degradation threshold is OPEN until selected by an accepted protocol.

## 3. Data and evaluation boundaries

- Query IDs in final evaluation sets must not be used for training,
  calibration, threshold selection, or profile-library construction.
- Every run records the query-ID manifest used for each phase.
- Human difficulty categories and published difficulty labels may be used as
  features or analysis axes, but are not ground-truth routing labels.
- Methods are evaluated on paired query sets.

## 4. Interaction and causality

- Precision decisions must not be described as independent by default. An
  independent planner is a required additive baseline only.
- Query text and structure may control the initial bit profile.
- A fixed probe may control only groups that have not yet run.
- Output from group g may control groups g+1 through G only.
- A later hidden state may not retroactively justify an earlier decision.
- Probe computation, duplicate computation, controller work, and feedback
  overhead must be included in reported cost.

## 5. Hardware cost

Average bit-width is not sufficient evidence of a systems benefit. A cost
report must state which of the following were measured, estimated, or omitted:

- resident accelerator bytes;
- host-to-device bytes;
- latency;
- prefetch stalls;
- kernel switches;
- controller overhead.

If unused higher-precision slices remain resident, selecting four-bit
execution must not be reported as reducing peak storage. Hardware and backend
choices, along with cost coefficients, are OPEN until accepted separately.

## 6. Reproducibility

Every run records:

- git commit SHA;
- full configuration;
- random seeds;
- model identifier and revision;
- dataset identifier and revision;
- query-ID manifest;
- quantization and backend mode;
- software environment;
- hardware identity;
- raw artifact location.

Raw run artifacts are immutable. Derived summaries point back to their source
runs. If a required field is unavailable, the report says so explicitly.

## 7. Comparisons and reporting

- Report quality-cost Pareto behavior, not only a selected operating point.
- Include all baselines required by the accepted experiment protocol.
- Report negative and null findings.
- Do not claim superiority when required baselines were not run.
- Do not claim generalization beyond the evaluated model, datasets, hardware,
  and backend.

## 8. Falsification gates

Proceed in this order:

1. signal test;
2. interaction test;
3. oracle-headroom test;
4. amortization test;
5. systems test.

If the oracle offers little benefit over a static profile, stop before
developing a sophisticated controller unless a later ADR changes the premise.
Stopping is a valid scientific outcome.

## 9. Claims policy

Do not claim:

- elastic memory from fake quantization alone;
- latency benefit from average bits alone;
- online causality when later information affected earlier decisions;
- generalization beyond the evaluated model, datasets, hardware, and backend;
- superiority when required baselines were not run;
- a measured systems benefit from simulated or lookup-table evidence alone.

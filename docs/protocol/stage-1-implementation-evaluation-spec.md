# QBitPlan Stage-1 implementation and evaluation specification

Status: ready for agent handoff
Published handoff: https://github.com/weige15/QBitPlan/issues/24

## Problem statement

QBitPlan has an accepted Stage-1 falsification protocol, but the protocol
cannot be implemented reproducibly until each retained execution uncertainty
is made explicit. The implementation must evaluate whether query-conditioned
selection can preserve correctness while reducing a declared common execution
cost envelope. It must distinguish the independent and interaction-aware
offline baselines, prevent final-data and answer leakage, preserve paired
comparisons, and make invalid or incomplete runs auditable.

The desired result is an executable research plan, not a production router, a
new serving architecture, or a paper claim. The accepted scientific decisions
remain authoritative. Reversible implementation details are fixed by the
Stage-1 execution contract, and this specification translates those decisions
into a coherent implementation seam and testable deliverables.

## Solution

Implement one end-to-end Stage-1 execution-plan boundary. The boundary takes
immutable model, dataset, tokenizer, profile, hardware, and run manifests;
constructs the permitted training and validation records; executes the pinned
BF16 reference and hardware-executable profiles; evaluates correctness and
diagnostics; measures the common cost dimensions; and emits immutable,
lineage-linked artifacts. Separate analysis jobs consume only sealed outputs
from their predecessors.

The plan has four ordered stages: establish executable profiles and training
targets, fit and freeze the static and independent baselines, evaluate the
interaction-aware planner and its gate, and evaluate oracle headroom only when
the earlier gates succeed. Final records are released only for confirmatory
evaluation of already-frozen methods after the gate sequence reaches a
terminal status.

The implementation must fail closed when a required input, deterministic
control, paired record, schema, provenance link, cost-dimension measurement,
or artifact is missing. It must never fill a missing value from a library
default or silently substitute a profile, query result, tokenizer, parser,
hardware measurement, or control.

## User stories

### Inputs, manifests, and run identity

1. As an experiment operator, I want every run to declare immutable model,
   dataset, tokenizer, profile, hardware, software, seed, decoder, and
   measurement manifests so that the run can be reproduced from its identity.

2. As an evaluator, I want artifact and manifest hashes to be deterministic
   and separately defined so that byte identity, canonical metadata identity,
   and run identity cannot be confused.

3. As a provenance auditor, I want every derived artifact to name its schema,
   producer revision, source artifacts, source manifest, configuration hash,
   record count, and creation time so that lineage can be checked without
   trusting directory names.

4. As an experiment operator, I want raw artifacts to be write-once and
   retries to use new attempt identities so that recovery cannot overwrite
   evidence.

5. As a scheduler operator, I want deterministic work-unit keys and shard
   assignment so that parallel execution and later merging produce the same
   logical workload.

### Profile execution and feasibility

6. As a profile evaluator, I want every declared profile transformed and
   completely forwarded on the target hardware before it enters the
   executable set so that unsupported profiles cannot enter comparisons.

7. As a feasibility analyst, I want the phase- and dataset-specific feasible
   set constructed only from permitted complete records and the accepted
   degradation threshold so that feasibility is not tuned on validation or
   final evidence.

8. As a researcher, I want empty feasible or target sets represented as
   explicit records so that a missing opportunity is a scientific result rather
   than a dropped row or an implicit substitute.

9. As a reproducibility auditor, I want canonical profile IDs and deterministic
   tie handling so that repeated runs select the same static profile and
   resolve the same executable-profile choices.

### Query construction and leakage control

10. As a query-feature evaluator, I want prompts constructed from only the
    permitted problem/question/options fields in a fixed raw-text format so
    that chat templates, demonstrations, labels, and answer metadata cannot
    change the estimand.

11. As a leakage auditor, I want feature construction to happen before any
    transformer group runs and to exclude answers, correctness, logits,
    generated tokens, profile IDs, and future hidden states so that selection
    uses only information available at its decision point.

12. As a feature evaluator, I want the structural feature list, tokenization,
    embedding pooling, normalization, and zero-vector behavior fixed so that
    feature values do not depend on library conventions.

13. As a data split auditor, I want scalar normalization statistics and all
    learned target models fit from permitted MATH training records only so
    that validation and final data cannot influence selection.

### Independent baseline

14. As a baseline evaluator, I want one marginal scorer for each group and bit
    trained from target-support labels, including empty-target all-zero
    examples, so that the independent baseline has a declared estimand.

15. As a baseline evaluator, I want the independent scorer to select only
    executable profiles and to resolve equal scores by canonical ID so that
    its behavior is total and deterministic.

16. As a scientific reviewer, I want the independent baseline to exclude
    upstream context, interaction features, cost scalars, and unaccepted
    budgets so that the comparison isolates the intended independent model.

### Interaction-aware planner

17. As a planner evaluator, I want the planner’s upstream context to contain
    only the actually executed prefix, its causal hidden-state summary, and
    selected prefix bits so that future information cannot leak into an
    earlier decision.

18. As a planner evaluator, I want training examples to use each target
    profile’s own ordered prefix context and to support only observed target
    continuations so that the planner does not train on impossible contexts.

19. As a planner evaluator, I want runtime decisions restricted to executable
    continuations, deterministic tie handling, and explicit invalidation when
    no continuation exists so that planner failures are auditable.

20. As a research-scope reviewer, I want the interaction-aware planner
    described as an offline baseline with measured overhead so that it cannot
    be mistaken for a production router, controller, cache, or serving layout.

### Inference, correctness, and diagnostics

21. As an inference operator, I want the exact tokenizer revision, fast-path
    requirement, model-loading mode, beam-search controls, one-process
    execution, GPU pinning, and deterministic runtime controls recorded so
    that repeated outcomes have the same declared inference regime.

22. As a correctness evaluator, I want the same normalization, parser,
    evaluator, and external ground truth applied to BF16 and every profile so
    that correctness differences are not parser artifacts.

23. As a MATH evaluator, I want the pinned original answer-equivalence
    evaluator used without a custom fallback so that mathematical correctness
    remains comparable to the source task.

24. As a diagnostics analyst, I want the exact BF16 token sequence teacher
    forced through every variant and full-vocabulary forward KL reported per
    query, while keeping diagnostics outside selection and primary cost, so
    that output drift is measured without changing the gate estimand.

25. As a scientific reviewer, I want hidden-state distance diagnostics omitted
    while allowing the planner’s causal context summary, so that a planning
    input is not mislabeled as an independent diagnostic metric.

### Cost measurement and comparability

26. As a cost analyst, I want the common static-versus-oracle envelope to be
    resident accelerator bytes, host-to-device bytes, latency, prefetch stall
    time, and kernel switch count, with no invented scalar budget, so that
    gross headroom remains a componentwise comparison.

27. As a measurement operator, I want model loading and quantization recorded
    separately from query execution, fixed warm-up and repetition counts,
    synchronized timing, and a separate traced pass so that primary latency is
    not mixed with setup or tracer overhead.

28. As a measurement auditor, I want resident memory measured as absolute
    peak device use, transfers measured from annotated CUDA-copy records, and
    trace-derived dimensions marked unavailable when evidence is absent so
    that omitted measurements are never converted into optimistic zeros.

29. As a comparison analyst, I want only the common subset of cost dimensions
    with valid coverage compared, with controller/probe/feedback overhead
    recorded whenever applicable, so that methods are not compared on unequal
    evidence.

### Gates, uncertainty, and sealing

30. As a gate evaluator, I want paired complete-query bootstrap samples,
    fixed replicate counts, independent declared seeds, and explicit percentile
    interpolation so that uncertainty intervals are reproducible.

31. As a gate evaluator, I want the validation signal sealed before the
    interaction gate and the interaction gate sealed before oracle headroom so
    that downstream analyses cannot influence upstream decisions.

32. As a final-data auditor, I want final manifests frozen, hashed, read-only,
    and unavailable to training, fitting, threshold selection, and gate jobs
    so that confirmatory results cannot become exploratory tuning.

33. As a research lead, I want valid null, inconclusive, or negative results
    to stop the sequence while invalid results block interpretation, so that
    the project does not manufacture a positive claim from a failed gate.

### Operations and failure recovery

34. As an operator, I want smoke mode to use the real pinned loader and real
    execution path while remaining unable to satisfy a gate, so that health
    checks do not become scientific evidence.

35. As an operator, I want query failures, measurement failures, deterministic
    failures, overflow, OOM, provenance mismatch, and incomplete traces to
    have distinct statuses and reason codes so that recovery does not blur
    invalidity with incorrectness.

36. As a scheduler operator, I want only allowlisted transient failures
    retried, with a bounded attempt count and no control relaxation, so that
    resume cannot turn operational instability into selection bias.

37. As a release reviewer, I want the complete artifact, schema, lineage,
    gate, and reproducibility checks to run before handoff so that an agent can
    implement the plan without reopening accepted protocol decisions.

## Implementation decisions

1. The feasible set uses the declared degradation threshold of `0.01` for the
   phase and dataset. The per-query target set is the feasible subset whose
   external correctness function is one; empty targets are recorded misses.

2. Static selection uses permitted MATH training records, a frozen 256-query
   training cost frame, Pareto-minimal mean vectors over the common five
   dimensions, and the smallest canonical eight-bit profile ID. Validation is
   never used to choose the static profile.

3. Query features are the seven structural counts plus a fixed BF16 pilot
   embedding formed by mean-pooling non-padding embedding rows, casting to
   float32, L2-normalizing, and concatenating after training-only scalar
   normalization. Zero vectors remain zero and zero-variance scales are one.

4. The independent baseline uses minimum-norm ordinary least squares for
   per-group, per-bit target-support labels and chooses the best executable
   profile by summed bit scores, with canonical-ID tie handling.

5. The interaction-aware planner adds the causal pooled hidden-state summary,
   query/context products, and one-hot executed-prefix bits. It trains on
   observed target-profile prefixes, scores executable continuations, favors
   bit `4` on ties, and invalidates an unavailable continuation.

6. Prompts are zero-shot raw text. MATH requests a boxed final answer and
   MMLU-Pro requests one option letter. MMLU-Pro parsing is strict; MATH uses
   the pinned source evaluator. Both use Unicode NFKC, normalized line endings,
   trimmed text, and external ground truth.

7. Diagnostics use one BF16 completion, teacher forcing, and full-vocabulary
   forward KL in float32 at generated positions. They run after the primary
   pass, carry separate overhead, and never affect fitting, selection, costs,
   or gates. Hidden-state distance diagnostics are excluded.

8. Bootstrap is paired over complete query records with 10,000 NumPy PCG64
   replicates, fixed phase-specific seeds, percentile 95% intervals, and

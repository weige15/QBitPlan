# Implement the QBitPlan Stage-1 falsification harness

Status: ready for agent handoff
Published handoff: https://github.com/weige15/QBitPlan/issues/24

## Problem Statement

QBitPlan has an accepted Stage-1 falsification contract but does not yet have a
reproducible, manifest-driven harness that can generate the required profile
outcomes, evaluate the static/query-only/independent/interaction comparisons,
collect cost evidence, and classify the ordered gates without leakage or
unsupported claims.

The harness must make the accepted estimands executable, preserve the complete
256-profile and required non-final data contract, and make every result
auditable by artifact lineage and evidence class. It must report failed or
inconclusive science honestly rather than turning operational gaps into claims.

## Solution

Implement one high-level seam: an accepted Stage-1 experiment plan is executed
into an immutable artifact bundle and gate report. The seam validates the
plan, freezes query and final manifests, executes the BF16 reference and all
declared profiles, constructs target sets, evaluates the static, direct,
query-only, additive-independent, MCKP, and interaction-aware comparisons,
collects lookup or direct cost evidence, and produces ordered gate reports.

The bundle is immutable, content-addressed, schema-validated, and linked to
the exact plan and source artifacts. The execution plan has four ordered
stages: establish executable profiles and oracle outcomes, fit and freeze
training-only selectors and baselines, evaluate the validation signal and
interaction gates, and evaluate oracle headroom only after earlier gates
succeed. Final data are released only for confirmatory evaluation of already
frozen methods after the ordered gate sequence reaches a terminal status.

The implementation must fail closed when a required input, deterministic
control, paired record, schema, provenance link, cost-dimension measurement,
or artifact is missing. It must never fill a missing value from a library
default or silently substitute a profile, query result, tokenizer, parser,
hardware measurement, or control.

## User Stories

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

### Selection, evidence, modes, and report outputs

38. As a protocol operator, I want plan validation to reject missing or
    conflicting accepted controls before execution, so that a run cannot
    silently acquire library defaults.

39. As a manifest auditor, I want immutable query manifests to bind canonical
    query IDs, split membership, prompt inputs, dataset revisions, and hashes,
    so that the same query population can be reconstructed and paired.

40. As a final-data steward, I want final manifests sealed before any fitting,
    target construction, threshold selection, or gate job, so that final IDs
    cannot leak into exploratory decisions.

41. As a model executor, I want the BF16 reference, transformed profile, and
    complete ordered eight-group execution paths to share declared tokenizer,
    decoder, seed, and synchronization controls, so that quality comparisons
    have a common execution boundary.

42. As a profile auditor, I want all 256 canonical profiles to have stable
    identities, attempted outcomes, and transform/forward status, so that the
    executable universe is exhaustive and failure exclusions are explicit.

43. As an oracle analyst, I want exhaustive per-query outcomes for every
    executable profile on every required non-final record, so that the oracle
    and target sets are lookupable rather than inferred from average bits.

44. As a target-set analyst, I want feasibility and external correctness
    joined by query and profile identity under the declared epsilon rule, so
    that an empty target is a recorded miss and never a substitute profile.

45. As a static-baseline analyst, I want Pareto-minimal mean costs from the
    frozen training cost frame and canonical-ID tie handling, so that the
    static profile is frozen without validation selection.

46. As a direct-scoring analyst, I want a query-only score for each executable
    profile’s training target-membership outcome, so that direct profile
    scoring is a separately auditable comparison rather than being conflated
    with additive group scoring.

47. As an independent-baseline analyst, I want additive per-group/per-bit
    scores and an executable-profile-constrained MCKP selection path, so that
    the mandatory independent baseline remains query-only, deterministic, and
    free of an invented cost scalar.

48. As an interaction analyst, I want later decisions to use only the actually
    executed prefix context and causal hidden-state summary, so that the
    interaction comparison cannot use future or post-decision information.

49. As a cost analyst, I want lookup-table estimates and direct hardware
    measurements represented as distinct evidence classes with method and
    coverage, so that estimated cost never masquerades as measured benefit.

50. As an operator, I want smoke, functional-quality, and direct-cost modes
    to share the same plan and adapters while declaring their evidence limits,
    so that health checks and quality checks cannot be mistaken for systems
    evidence.

51. As a scheduler, I want deterministic sharding, idempotent work-unit
    completion, matching-plan resume, bounded transient retries, and immutable
    attempts, so that recovery cannot duplicate or overwrite scientific data.

52. As a schema auditor, I want every raw artifact to validate its schema,
    content hash, source manifest, producer revision, and lineage before merge,
    so that an artifact bundle is reviewable independently of its storage path.

53. As an uncertainty analyst, I want paired bootstrap resampling to preserve
    complete query records and method correlations, so that gate intervals
    quantify the declared query-level estimands.

54. As a gate controller, I want signal, interaction, and oracle gates to run
    only in order and to stop on valid practical-null, inconclusive, or
    negative evidence, so that downstream controller work cannot be authorized
    by an upstream failure.

55. As a report reviewer, I want practical-null, inconclusive, negative,
    invalid, and successful reports to be mutually distinguishable, so that
    operational invalidity is not presented as a scientific negative.

56. As a final evaluator, I want sealed final evaluation to run only frozen
    methods after gate termination and to be visibly confirmatory, so that final
    results cannot change the accepted gate estimands.

57. As a reproducibility reviewer, I want a complete manifest, artifact,
    lineage, evidence-class, paired-analysis, and report replay check, so that
    another agent can review the result without trusting process memory.

## Implementation Decisions

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
   explicit linear interpolation. Use seeds `20260804`, `20260805`, and
   `20260806` for signal, interaction, and oracle headroom respectively; use
   complete queries as the bootstrap unit and do not use adaptive resampling.

9. Tokenization and software are immutable inputs. Require the fast tokenizer
   for the pinned model revision, reject fallback tokenizers and overflow, set
   `trust_remote_code=False`, and record tokenizer configuration and file
   hashes. Record the verified Python, PyTorch, Transformers, TorchAO, NumPy,
   Datasets, Accelerate, Safetensors, CUDA, and driver tuple plus each run’s
   UUID-pinned RTX 3090 identity.

10. Inference uses one process, batch size one, one UUID-pinned GPU, BF16 model
    loading, evaluation mode, inference mode, cache enabled, deterministic
    four-beam search with `max_new_tokens=1024`, and all accepted decoder and
    synchronization controls. Disable compilation, graph capture, CPU offload,
    dynamic batching, padding, and control relaxation after deterministic-op
    failure.

11. Prompt construction is zero-shot raw text with the accepted MATH and
    MMLU-Pro forms. Feature extraction uses only canonical pre-decision query
    text and structure: the seven structural counts and the fixed BF16
    embedding. Exclude labels, answers, correctness, profile outcomes, logits,
    generated tokens, future hidden states, and profile IDs.

12. Normalize generated text with Unicode NFKC, normalized line endings, and
    trimmed whitespace. Use strict MMLU-Pro parsing and the pinned original
    MATH equivalence evaluator. Parse failures are incorrect and separately
    recorded; external ground truth is never replaced by BF16 agreement.

13. Preserve the four evidence classes: analytical, simulated, lookup-table
    estimated, and directly measured. Attach evidence class, method, coverage,
    and source artifact to every result and cost dimension. Simulated quality
    may support declared quality evidence, lookup tables support only labeled
    estimates, and direct systems claims require direct execution and
    measurement on the UUID-pinned hardware path.

14. The common static-versus-oracle cost envelope is resident accelerator
    bytes, host-to-device bytes, latency, prefetch stall time, and kernel
    switch count. The full six-dimensional reporting vector retains applicable
    controller/probe/feedback overhead. Compare componentwise and by Pareto
    dominance; do not invent a scalar budget, per-dimension ceiling, or
    average-bit cost proxy.

15. Attempt every one of the 256 canonical eight-bit profiles. A profile enters
    `P_exec` only after successful transformation and complete finite forward
    execution under the accepted group semantics. Failed profiles are retained
    as explicit non-executable outcomes; no nearby, BF16, fake-quantized, or
    library-default substitute is introduced.

16. Generate exhaustive oracle outcomes for all required non-final records and
    all executable profiles. Each profile executes as a complete ordered
    eight-group path with its own upstream context. The 7,500 non-final MATH
    training records construct targets and selectors; the 4,500 held-out
    validation records assess signal and interaction; the fixed 256-query
    non-final cost frame supplies paired oracle headroom evidence.

17. Construct the phase- and dataset-specific feasible set with the accepted
    epsilon threshold and construct each query target set by joining feasibility
    with external correctness. Empty target sets are retained and count as
    misses. Final IDs are not available to profile construction, target
    construction, training, tuning, threshold selection, or gate declaration.

18. Select the static profile from permitted training evidence only. Compute
    mean costs on the frozen 256-query training cost frame, retain Pareto-
    minimal profiles over the common dimensions, and choose the smallest
    canonical profile ID. Freeze it before validation outcomes are read; an
    empty training feasible set invalidates the static baseline.

19. Direct profile scoring is a query-only comparator. For every executable
    profile, fit a minimum-norm scorer for training target membership using the
    shared intercept-augmented query features, select the highest-scoring
    executable profile, and resolve equal scores by canonical profile ID. It
    has no cost scalar, outcome leakage, or upstream context.

20. The mandatory independent baseline has both equivalent declared execution
    forms available: additive per-group/per-bit OLS scores from target-support
    labels and an executable-profile-constrained multiple-choice knapsack
    selection over those additive scores. MCKP selects one bit per group while
    respecting `P_exec`; it does not introduce a cost objective or unaccepted
    budget. Include empty-target all-zero training examples.

21. The interaction-aware planner uses the query-only features plus the causal
    hidden-state summary after the actually executed prefix and one-hot prefix
    bits. Train each supported continuation from its own ordered target-profile
    prefix. At runtime score only executable continuations, favor bit `4` on
    ties, execute the selected group, and build the next context from that
    execution. Future states, correctness, answers, and later outcomes are
    unavailable to earlier decisions.

22. The execution-plan boundary supports three operational modes. Smoke mode
    uses one real pinned training query and one validation query with BF16,
    all-4, all-8, and one mixed profile and is never evidentiary. Functional-
    quality mode generates the required profile outcomes and correctness,
    target, baseline, and gate inputs without claiming direct systems benefit.
    Direct-cost mode executes the fixed cost frame with the accepted warm-up,
    repetition, synchronization, NVML, CUDA-trace, and annotation scopes.

23. A lookup-table adapter may provide explicitly labeled estimated costs only
    for covered profiles and dimensions. The direct-cost adapter must execute
    and measure the profile on the UUID-pinned target, exclude model loading
    and quantization from query timing, separate traced latency from primary
    latency, and mark missing dimensions as `omitted/unavailable/<reason>`.

24. Work units include phase, query, profile, execution mode, pass, and
    repetition. Assign shards by the declared hash modulo rule. A completed
    unit is idempotent only when its plan identity, configuration hash, schema,
    and source manifests match. Resume reuses matching complete units; retries
    create new attempts, never overwrite artifacts, stop at three attempts, and
    apply only the accepted transient-failure allowlist.

25. Raw per-query, per-profile, trace, manifest, tensor, bootstrap, and gate
    artifacts use the declared canonical JSON, strict NDJSON, or Safetensors
    formats. Every derived artifact records type, schema version, producer
    revision, source artifact IDs, source manifest ID, configuration hash,
    count, and creation timestamp. Content, manifest, and run identities use
    their distinct SHA-256 definitions.

26. Terminal run states are complete, invalid, incomplete, and aborted. A
    query runtime failure is not an incorrect outcome or a dropped row. Missing
    paired records, unsupported deterministic operations, provenance mismatch,
    non-finite output, overflow, OOM, and insufficient trace evidence receive
    explicit reason codes. Invalid evidence remains in lineage but cannot enter
    aggregates or satisfy a gate.

27. Bootstrap analysis is paired by complete query record. Signal and
    interaction resample the 4,500 validation IDs; oracle headroom resamples
    the 256 cost-frame IDs while keeping profiles within a query dependent.
    Report percentile 95% intervals and use the lower 2.5th percentile for
    gates. No bootstrap result tunes a model, threshold, profile, or prompt.

28. Gate enforcement is a state machine with no manual override. The signal
    gate compares the query-conditioned selector with static and additive
    independent baselines and requires a five-point improvement plus paired
    lower bounds above zero against both. Only signal success permits the
    interaction gate; interaction requires the same five-point improvement and
    lower bound against independent. Only interaction success permits oracle
    headroom; useful headroom requires a five-point Pareto-opportunity rate and
    lower bound above zero.

29. A valid positive result below the five-point margin is `practical-null`; an
    interval including zero is `inconclusive`; a worse result is `negative`;
    missing or invalid required evidence is `invalid`; and a result meeting
    the predeclared endpoint and interval conditions is `successful`. These
    classifications are endpoint-specific and preserve evidence-class and
    coverage bounds.

30. Final manifests are frozen, hashed, and read-only before exploratory
    calibration. After the ordered gate sequence reaches a terminal status,
    final evaluation may run only for methods, parameters, profile libraries,
    and thresholds already frozen before final release. Final results cannot
    change gate statuses or authorize a controller architecture.

31. Reproducibility review replays plan validation, manifest identity, schema
    validation, lineage closure, leakage checks, paired aggregation, bootstrap
    seeds, gate state transitions, and report classification. Runtime facts
    such as GPU UUID, `P_exec`, trace coverage, artifact hashes, and invalid
    attempts are recorded observations, not guessed configuration values.

## Testing Decisions

Tests exercise the high-level experiment-plan-to-artifact-bundle seam as the
primary contract. They assert externally visible artifacts, statuses, hashes,
lineage, reports, and gate transitions rather than private implementation
details. The repository has no existing harness test suite; the new tests use
the accepted ADR and execution-contract invariants as their prior art.

1. Use deterministic fake adapters for model loading, tokenization, profile
   transformation, forward execution, dataset access, cost lookup, NVML, and
   trace collection in most contract tests. Fakes must expose the same failure
   and evidence states as real adapters and must not become a scientific data
   path.

2. Include a tiny non-evidentiary integration fixture using the real plan
   format, a reduced query manifest, the same artifact schemas, and the same
   high-level seam. The fixture proves orchestration and serialization only;
   it cannot satisfy a Stage-1 gate or replace the full 256-profile contract.

3. Validate protocol and configuration invariants: required fields, immutable
   revisions, exact tokenizer and software tuple, decoder controls, seed
   controls, hardware identity, execution mode, evidence class, cost scopes,
   and rejection of unspecified defaults.

4. Validate query-manifest and final-sealing invariants: canonical ordering,
   stable IDs, split disjointness, source hashes, read-only final state, and
   rejection of any final record entering training, tuning, target construction,
   profile selection, or threshold fitting.

5. Validate profile invariants: all 256 identities are attempted, bit order is
   stable, transform and complete-forward failures exclude only that profile,
   ordered group execution preserves causal prefixes, tie behavior is stable,
   and unsupported profiles cannot be substituted.

6. Validate oracle and target invariants: every required query/profile pair is
   accounted for, external correctness is used consistently, empty targets are
   retained as misses, feasibility uses the declared epsilon, and final-only
   records never enter non-final artifacts.

7. Validate feature and scoring leakage invariants: pre-decision query-only
   features exclude answers, labels, correctness, profile IDs, outputs, and
   future hidden states; direct scoring is profile-level; additive scoring is
   group/bit-level; MCKP respects `P_exec`; and interaction context is causal.

8. Validate static and planner invariants: training-only fitting, frozen
   256-query cost frame, Pareto static selection, canonical ties, supported
   continuations, executable-only runtime choices, and no budget scalar or
   average-bit systems proxy.

9. Validate evidence and cost invariants: analytical, simulated, lookup-table
   estimated, and directly measured labels remain distinct; cost lookup cannot
   produce a measured claim; direct-cost records have the required scopes and
   raw sources; and unavailable dimensions are never zero or imputed.

10. Validate smoke, functional-quality, and direct-cost mode invariants,
    including non-evidentiary smoke reports, complete quality accounting,
    direct measurement boundaries, mode-specific artifact identities, and
    refusal to mix trace overhead into primary latency.

11. Validate artifact invariants: canonical identity hashes, strict schemas,
    source closure, write-once raw records, matching-plan idempotence, shard
    merge rejection, new attempt IDs, bounded retries, and explicit terminal
    invalid/incomplete/aborted states.

12. Validate bootstrap and gate invariants with deterministic fixtures for
    every required outcome: practical-null, inconclusive, negative, invalid,
    and successful. Tests must verify paired units, fixed seeds, percentile
    interpolation, lower-bound extraction, ordered stop enforcement, and no
    manual override.

13. Validate sealed-final invariants by proving that a final record can be
    consumed only by confirmatory evaluation after gate termination and cannot
    alter parameters, thresholds, profile libraries, gate reports, or prior
    artifacts.

14. Run the reproducibility test twice over the same permitted fixture and
    compare canonical manifests, configuration hashes, deterministic derived
    outputs, lineage, and report classifications. Timestamps and hardware
    observations remain explicitly variable run metadata.

15. No test may require a signal, interaction, or oracle hypothesis to succeed.
    A correct implementation must be able to produce each valid outcome class
    and must treat invalidity as a blocked interpretation rather than as a
    negative scientific finding.

## Out of Scope

- Changing accepted Stage-1 ADRs.
- Production router architecture.
- A causal GRU or attention controller.
- Profile medoids and profile-library compression.
- Activation or KV-cache quantization.
- Production residual-slice prefetch/cache design.
- Custom CUDA kernels.
- Serving deployment claims.
- Paper drafting.
- Result-dependent protocol revision.

## Further Notes

Smoke results are non-evidentiary. Final IDs cannot influence training,
tuning, target construction, profile selection, or thresholds. Average bits
cannot establish a systems benefit. Every result is bounded to its evidence
class, cost coverage, execution mode, and declared estimand. All 256 profiles
and all required non-final records remain the full scientific contract even
when development uses smaller fixtures.

The high-level seam is intentionally singular: an accepted experiment plan
becomes an immutable artifact bundle and gate report. Functional-quality,
direct-cost, lookup, smoke, sharded, resumed, and sealed-final workflows are
adapters or modes of that seam, not parallel scientific pipelines.

Completion of this specification means the harness is ready for agent
implementation. It does not mean that an experiment has run, a gate has
passed, a systems benefit has been measured, or a paper claim is warranted.

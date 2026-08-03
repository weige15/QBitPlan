# ADR-0007: Stage-1 high-precision reference and ground-truth role

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Adopt the high-precision reference and
its role relative to ground truth](https://github.com/weige15/QBitPlan/issues/6).
Stage 1 needs a reproducible comparison execution for the selected pilot while
preserving the distinction between a model reference and an external task
target.

ADR-0004 selects `meta-llama/Llama-3.1-8B` at immutable revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. The pinned configuration records
`torch_dtype: bfloat16`. The scientific standards require every experiment to
declare its reference and numeric format, report degradation relative to that
reference, and keep ground truth separate from reference execution.

## Decision

Use the unquantized BF16 execution of the exact ADR-0004 model revision as the
Stage-1 high-precision reference. The reference run must use the same declared
model revision, tokenizer and input manifest, backend, software tuple, and
inference controls as the corresponding quantized run, with quantization
disabled. The final execution manifest remains responsible for pinning those
values.

For each paired query:

- compare quantized per-position logits or log-probabilities with the BF16
  reference for model-relative diagnostics;
- allow hidden-state distances only as optional diagnostics, and only when a
  later accepted metric protocol justifies their layer, position,
  representation, and aggregation choices; and
- judge externally evaluated task correctness against the external ground-truth
  target, not against agreement with the reference. Perplexity may be primary
  only for a declared unlabeled language-modeling task.

The reference is a comparison point. It is not ground truth, an oracle profile,
or evidence of task quality or systems benefit by itself.

## Rule application

- **Reproducibility — conditional pass.** The exact model revision and BF16
  format are fixed. Reproducibility is completed only when the downstream
  tokenizer, software, backend, inference-control, and input-manifest fields
  are pinned in the final protocol.
- **Distinct from ground truth — pass.** Ground truth remains the externally
  defined task target. Reference comparisons describe model-relative behavior;
  task outcomes are evaluated against the task target.
- **Declared measurements — conditional pass.** BF16 logits and
  log-probabilities support the required output-distribution comparisons.
  Hidden-state distances are not silently promoted to required metrics and
  need explicit downstream justification before use.

## Evidence classification

### Source facts

- `CONTEXT.md` defines ground truth as the external task target and the
  high-precision reference as the declared model execution used for comparison.
- `SCIENTIFIC_STANDARDS.md` requires a declared reference and numeric format,
  and states that KL divergence, answer flips, hidden-state distances, and
  related measures are diagnostic unless an accepted protocol promotes one.
- ADR-0004 records the selected model identifier, immutable revision,
  architecture, and 32-layer structure; its cited pinned configuration records
  BF16 as the model dtype.
- The Issue 7 research record directly verified a bounded mixed TorchAO 4/8
  forward path for the selected revision, but did not establish task quality,
  systems benefit, or generalization.

### Inference

An unquantized BF16 execution of the pinned revision on the same declared
execution stack is a reproducible comparison baseline for model-relative output
diagnostics. Paired identical inputs allow those comparisons without treating
the reference as the external task target.

### Project preference

The maintainers selected BF16 as the Stage-1 high-precision format and chose
output-distribution comparisons as the required model-relative diagnostic.
Keeping hidden-state distances optional bounds the initial protocol while
preserving them as a possible later diagnostic.

## Consequences

- Quantized and reference evaluations must use paired query inputs and retain
  the query-ID manifest for each evaluation phase.
- Reference-relative degradation and ground-truth task correctness must be
  reported as separate outcomes.
- A later metric decision must define aggregation and any use of optional
  hidden-state distances before evaluation.
- BF16 reference execution does not establish latency, memory, kernel, or
  serving benefits.

## Uncertainty retained

- The final tokenizer, Python/PyTorch/Transformers/TorchAO tuple, quantization
  implementation, group execution semantics, inference controls, task suite,
  query-ID manifests, metric aggregation, and degradation threshold remain
  downstream protocol decisions.
- Hidden-state distance details remain OPEN unless the metric decision accepts
  and defines them.

## Related records

- [ADR-0004: Stage-1 pilot model and immutable revision](0004-stage-1-pilot-model.md)
- [Select the primary quality metric and diagnostic metrics](https://github.com/weige15/QBitPlan/issues/9)
- [Define degradation and the epsilon-selection rule](https://github.com/weige15/QBitPlan/issues/13)

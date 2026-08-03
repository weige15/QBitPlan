# ADR-0004: Stage-1 pilot model and immutable revision

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Select the pilot model identifier and exact
revision](https://github.com/weige15/QBitPlan/issues/7). The feasibility
matrix is recorded in [Issue 3: Pilot model, revision, and
quantization/backend feasibility](https://github.com/weige15/QBitPlan/issues/3),
and the candidate-specific mixed-path checks are recorded in
[the Issue 7 research note](../research/issue-7-mixed-execution-verification.md).

The Stage-1 rule requires one public model identifier and immutable revision
that supports the selected task suite and eight-group weight-only 4/8 study,
is feasible on the accepted target hardware or proxy, and has no unresolved
capability or backend fact that would make the protocol non-falsifiable.

## Decision

Use the following pilot model:

| Field | Accepted value |
| --- | --- |
| Model identifier | `meta-llama/Llama-3.1-8B` |
| Immutable revision | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| Architecture | `LlamaForCausalLM` |
| Transformer layers | 32 |

The revision is the exact pinned Hugging Face commit used for the
candidate-specific check. The model is manually gated; access approval and
the authenticated environment are therefore part of reproducibility metadata,
not a reason to substitute a floating revision.

## Rule application

- **Public exact revision — pass.** The pinned Hugging Face configuration and
  model metadata identify the revision and architecture.
- **Eight-group regularity — pass by verified structure.** The configuration
  has 32 decoder layers, matching eight contiguous four-layer groups.
- **Mixed 4/8 execution — pass on the tested stack.** The direct check used
  TorchAO's real `quantize_` path with explicit per-group filters. Groups 0,
  2, 4, and 6 used int4 weight-only transforms with group size 128; groups 1,
  3, 5, and 7 used int8 weight-only transforms. All 28 target linears per
  group transformed to the expected layouts, and the mixed forward was finite.
- **Accepted target hardware — pass for the declared environment.** The
  check ran on an available NVIDIA GeForce RTX 3090 in the accepted direct-
  measurement environment.
- **Backend falsifiability — pass with a bounded implementation path.** The
  tested TorchAO path executed real mixed representations. The newer
  `FqnToConfig` API and final group-execution semantics remain downstream
  decisions; they are not silently assumed by this ADR.
- **Task-suite fit — accepted downstream.**
  [ADR-0006](0006-stage-1-task-suite-and-query-splits.md) fixes the
  MMLU-Pro plus MATH/MATH-500 suite and immutable phase boundaries. This ADR
  still records no task-quality evidence for the model choice.
- **Project preference — recorded separately.** The maintainer's prior Llama
  experimentation and authenticated Hugging Face access informed the choice.
  They are preference/access facts, not evidence of task quality or systems
  benefit.

## Consequences

- Downstream Stage-1 protocol decisions use this identifier and exact revision;
  a different pilot requires a superseding ADR.
- Tokenizer revision, final Python/PyTorch/Transformers/TorchAO tuple, group
  execution semantics, high-precision exclusions, task suite, metrics, and
  cost measurements remain governed by their downstream tickets.
- The selection does not establish task quality, latency, memory benefit,
  kernel behavior beyond the bounded forward check, or generalization.
- Manual gating and local authentication must be recorded in the final
  reproducibility manifest; credentials must never be committed.

## Evidence

- [Pinned Llama configuration](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json)
- [Issue 7 mixed-execution research note](../research/issue-7-mixed-execution-verification.md)
- [TorchAO inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)

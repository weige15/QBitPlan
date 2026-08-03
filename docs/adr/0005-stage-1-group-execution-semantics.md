# ADR-0005: Stage-1 4/8 group execution semantics

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision tickets are [Define 4-bit and 8-bit group execution
semantics](https://github.com/weige15/QBitPlan/issues/8) and [Fix the eight-group
partition and high-precision exclusions](https://github.com/weige15/QBitPlan/issues/10).
The Stage-1 pilot is
`meta-llama/Llama-3.1-8B` at immutable revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, as recorded in
[ADR-0004](0004-stage-1-pilot-model.md). The model has 32 Transformer layers,
so the Stage-1 boundary needs one unambiguous bit choice for each of eight
contiguous four-layer groups.

The candidate-specific check recorded in the [Issue 7 mixed-execution
research note](../research/issue-7-mixed-execution-verification.md) used real
TorchAO weight-only transforms and a mixed forward pass. That establishes a
bounded executable path, not a quality or systems result.

## Decision

### Group and bit semantics

- A group is four contiguous entries of `model.layers`.
- A bit profile has eight entries, one choice from `{4, 8}` per group.
- The group choice applies to the seven projection linears in each member
  layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`. Thus, one group covers 28 target linear modules in this pilot.
- `4` means TorchAO `int4_weight_only(group_size=128)` applied to those target
  weights.
- `8` means TorchAO `int8_weight_only()` applied to those target weights.

Both choices are weight-only. The residual stream and residual additions,
activations, normalization modules, embeddings, and output head are not
quantized by a group choice. They remain in the high-precision reference
representation selected by the dependent reference decision. This decision
does not introduce residual quantization or a nested/residual-slice
representation.

### Precision-decision boundary and explicit exclusions

For signal measurement, the group bit decisions apply to exactly 224 target
projection weights: seven projection linears in each of the 32 Transformer
layers. The eight groups are:

| Profile entry | `model.layers` slice | Layer indices |
| --- | --- | --- |
| `g0` | `[0:4]` | 0–3 |
| `g1` | `[4:8]` | 4–7 |
| `g2` | `[8:12]` | 8–11 |
| `g3` | `[12:16]` | 12–15 |
| `g4` | `[16:20]` | 16–19 |
| `g5` | `[20:24]` | 20–23 |
| `g6` | `[24:28]` | 24–27 |
| `g7` | `[28:32]` | 28–31 |

The explicit complement of that target set is excluded from group precision
decisions and remains in the unquantized BF16 reference execution selected by
[ADR-0007](0007-stage-1-high-precision-reference.md):

- `model.embed_tokens`;
- `model.layers.{i}.input_layernorm` and
  `model.layers.{i}.post_attention_layernorm` for every layer `i`;
- `model.norm`;
- `model.layers.{i}.self_attn.rotary_emb` and positional-encoding
  calculations for every layer `i`;
- `model.layers.{i}.mlp.act_fn` for every layer `i`;
- residual streams and additions, activations, attention score/softmax
  calculations, and other non-quantized floating-point operations; and
- `lm_head`.

Input IDs and position indices retain their required integer types; they are
not precision-decision targets. The reproducibility manifest must record the
actual pinned-config/checkpoint relationship between `lm_head.weight` and
`model.embed_tokens.weight`; no tying or untied relationship is inferred by
convention.

### Representation and dequantization

The packed representation and dequantization behavior are those of the
declared, supported TorchAO backend configuration for the selected model and
software tuple. The execution path must consume the weight-only representation
through the backend's supported linear operation; a fake-quantized floating
weight path is not equivalent. A profile is hardware-executable only when the
declared model, revision, backend, and configuration can apply this mapping
and run the resulting model path.

### Evidence and claims

Evidence is classified per claim and cost dimension rather than assigning one
class to the entire protocol:

- Profile enumeration and bit/parameter arithmetic are analytical evidence.
- Quality or degradation evaluation may be simulated or directly measured,
  but must follow the declared reference, real targets, paired queries, and
  primary metric.
- A lookup table supports only explicitly labeled estimated cost comparisons
  for covered profiles.
- A measured hardware-cost claim requires direct execution and measurement on
  the accepted UUID-pinned RTX 3090/CUDA/PyTorch/TorchAO path. Unavailable
  dimensions remain `omitted/unavailable` with their reason.

No simulated, lookup-table, average-bit, or fake-quantized result may be
reported as a measured memory or latency benefit.

## Rule application

- **Implementability — pass with bounded scope.** The Issue 7 check applied
  real TorchAO int4 and int8 weight-only transforms to alternating four-layer
  groups of the selected revision and produced finite logits on the tested
  RTX 3090 stack.
- **Unambiguous bit choices — pass.** Each group has exactly one of the two
  declared TorchAO weight-only configurations, with the int4 group size
  explicit.
- **Single comparable revision — pass.** The semantics are bound to the
  immutable model revision selected by ADR-0004.
- **Coverage and exclusion boundary — pass.** The eight non-overlapping
  four-layer slices cover all 32 Transformer layers exactly once, yielding 224
  target projection weights. The strict complement is explicitly retained in
  the BF16 reference path, including embeddings, normalization, positional
  encoding, non-quantized computation, and the output head.
- **No fake systems benefit — pass.** Evidence classes and claim boundaries
  preserve the distinction between executable path checks, simulated quality,
  estimated cost, and directly measured hardware cost.

## Decision provenance

### Source facts

- ADR-0004 fixes the pilot at 32 Transformer layers and the immutable
  `meta-llama/Llama-3.1-8B` revision.
- The pinned Llama implementation exposes the seven target projection linears
  within each decoder layer, plus embeddings, RMSNorms, rotary position
  computation, MLP activation, and the output head.
- ADR-0007 fixes the comparison reference as unquantized BF16 execution.

### Inference

The listed slices are disjoint and their union is all layer indices 0–31, so
each target projection receives exactly one group bit decision and no target
layer is omitted. The listed non-target modules and operations form the
explicit complement of the group-controlled projection set.

### Project preference

The project retains every excluded module and operation in the BF16 reference
representation and does not introduce residual or nested quantization.

## Consequences and retained uncertainty

This ADR fixes the meaning of the two group choices for Stage 1. The exact
high-precision format is BF16 under ADR-0007. The final package/software
tuple, backend version, kernel support, profiling scope, and cost-accounting
details remain governed by their dependent protocol decisions. This ADR does
not select a cache or prefetch policy, optimized kernel, or production serving
layout.

The Issue 7 path check does not establish task quality, latency improvement,
resident-memory reduction, or generalization. Those claims require the
evidence and evaluation procedures accepted elsewhere in the Stage-1 map.

## Evidence

- [Define 4-bit and 8-bit group execution semantics](https://github.com/weige15/QBitPlan/issues/8)
- [ADR-0004: Stage-1 pilot model and immutable revision](0004-stage-1-pilot-model.md)
- [Fix the eight-group partition and high-precision exclusions](https://github.com/weige15/QBitPlan/issues/10)
- [ADR-0007: Stage-1 high-precision reference and ground-truth role](0007-stage-1-high-precision-reference.md)
- [Issue 7 mixed-execution research note](../research/issue-7-mixed-execution-verification.md)
- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)

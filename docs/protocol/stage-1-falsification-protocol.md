# QBitPlan Stage-1 falsification protocol

- Status: ACCEPTED scientific contract; implementation/evaluation details
  fixed before `$to-spec`
- Date: 2026-08-04
- Durable acceptance record: [ADR-0015](../adr/0015-stage-1-consolidated-protocol.md)
- Execution contract: [Stage-1 execution contract](stage-1-execution-contract.md)
- Map: [Wayfinder: Lock the QBitPlan Stage-1 falsification protocol](https://github.com/weige15/QBitPlan/issues/2)

## Purpose and boundary

This document consolidates the accepted Stage-1 scientific decisions before
invoking `$to-spec`. It defines the falsification contract for one pilot model;
it does not implement an evaluator, oracle, planner, controller, profiler, or
serving system.

The contract is bounded to the selected model, pinned dataset artifacts,
declared CUDA/PyTorch/TorchAO execution path, one UUID-pinned NVIDIA GeForce
RTX 3090 per run, and the cost dimensions covered by the accepted evidence
rules. It does not claim generalization beyond those boundaries.

## Fixed scientific contract

### Pilot and execution boundary

- Model: `meta-llama/Llama-3.1-8B`.
- Immutable revision: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- Architecture: `LlamaForCausalLM` with 32 Transformer layers.
- Hardware: direct measurement on one available NVIDIA GeForce RTX 3090 per
  run, selected with `nvidia-smi` and pinned by GPU UUID in the run record.
- Backend: CUDA/PyTorch/TorchAO weight-only 4/8 execution.

The model choice is bounded execution evidence and maintainer preference; it is
not evidence of task quality, systems benefit, or generalization.

### Reference and ground truth

- The high-precision reference is unquantized BF16 execution of the exact
  pinned model revision.
- Reference and quantized runs share the declared model revision, tokenizer and
  input manifest, backend, software tuple, and inference controls.
- Reference-relative logits/log-probabilities, answer flips, and any accepted
  hidden-state distances are diagnostics.
- Task correctness is judged against the external target, never by agreement
  with BF16. The reference is not ground truth, an oracle profile, or a
  systems result.

### Groups and precision semantics

- The profile has eight ordered entries in `{4, 8}`; each controls one
  contiguous four-layer group:

  | Group | Layers |
  | --- | --- |
  | `g0` | 0–3 |
  | `g1` | 4–7 |
  | `g2` | 8–11 |
  | `g3` | 12–15 |
  | `g4` | 16–19 |
  | `g5` | 20–23 |
  | `g6` | 24–27 |
  | `g7` | 28–31 |

- The choice applies only to `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`: 28 target linears per group,
  224 total.
- `4` means TorchAO `int4_weight_only(group_size=128)`; `8` means
  TorchAO `int8_weight_only()`.
- Embeddings, normalization, rotary/positional calculations, activations,
  attention score/softmax calculations, residual streams and additions, other
  non-quantized floating-point operations, and `lm_head` remain BF16. Input
  IDs and position indices retain their required integer types.
- Fake-quantized floating weights are not equivalent to executable weight-only
  representation.

### Tasks, targets, and immutable phases

The accepted Stage-1 suite is MMLU-Pro plus MATH/MATH-500. GPQA Diamond,
LiveCodeBench, and WikiText-103 are excluded.

| Phase | Artifact and revision | Records and role |
| --- | --- | --- |
| Calibration/training | MATH source Git `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | All 7,500 source training problems |
| Validation | Same MATH revision | 4,500 original test records not in final MATH-500 |
| Final paired evaluation | MATH-500 Hub `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | All 500 pinned `unique_id` records |
| Final paired evaluation | MMLU-Pro Hub `b189ec765aa7ed75c8acfea42df31fdae71f97be` | All 12,032 pinned test `question_id` records |

Every phase has an immutable manifest with dataset/configuration, exact
revision and split, phase, query identity, and canonical query-record hash.
Final IDs are never used for calibration/training, validation, threshold
selection, or profile-library construction. MMLU-Pro is final-only
cross-dataset transfer evidence, not in-domain MATH data.

### Metrics and degradation

- MMLU-Pro uses exact-option correctness against its external answer key.
- MATH/MATH-500 uses answer-equivalence correctness against its external target.
- Per-dataset correctness is decision-facing. The equal-weight macro accuracy
  across MMLU-Pro and MATH-500 is a secondary summary with one-half weight per
  dataset; pooled query accuracy is not primary.
- BF16-relative output-distribution divergence and answer flips are diagnostics.
  Hidden-state distances are optional only with justified measurement choices.
  No Stage-1 perplexity branch is used.
- For paired query `q`, dataset `d`, and executable profile `p`,
  `d(d,q,p) = c(d,q,BF16) - c(d,q,p)` and
  `D_d(p) = mean_q d(d,q,p)`.
- A profile is feasible only when every evaluated dataset has
  `D_d(p) <= epsilon`; `epsilon = 0.01`. Diagnostics do not silently
  become quality gates.

### Evidence and cost

Every claim and cost dimension carries one class: analytical, simulated,
lookup-table estimated, or directly measured. Analytical evidence supports
enumeration/arithmetic only; simulated or directly measured execution supports
declared paired quality/degradation tests; lookup tables support only labeled
estimated cost comparisons. Measured systems claims require direct execution on
the UUID-pinned RTX 3090 path. Average bits, fake quantization, simulation, and
lookup tables alone cannot establish measured systems benefit.

The cost vector is
`(resident accelerator bytes, host-to-device bytes, latency, prefetch stall
time, kernel switch count, controller/probe/feedback overhead)`. Resident
bytes, host-to-device bytes, and latency are direct measurements; stalls and
switches are trace-derived; controller overhead is measured when that path
executes. Unavailable dimensions are `omitted/unavailable/<reason>`, never
zero or imputed. Compare profiles componentwise and by Pareto dominance, not
with an invented scalar coefficient.

### Oracle and sampling

- Enumerate all 256 ordered profiles in `{4, 8}^8`, group order `g0`–`g7`,
  `4-bit = 0`, `8-bit = 1`, IDs `00000000`–`11111111`.
- Retain only profiles whose complete ordered execution succeeds on the pinned
  model, revision, backend, and configuration. Later groups use the same
  profile's upstream state; contexts are not mixed.
- Evaluate quality exhaustively over permitted non-final MATH records.
- Freeze exactly 256 IDs per non-final phase by the deterministic stratified
  SHA-256 rule and seed `20260803` in
  [ADR-0010](../adr/0010-stage-1-oracle-profile-enumeration-and-sampling.md).
  Run feasible profiles and BF16 on the same frame; do not adaptively resample
  or use final IDs.

### Ordered falsification gates

1. **Signal:** feasible-profile hit rate improves over both the frozen static
   and additive independent baselines by at least 0.05, with a paired 95%
   bootstrap lower bound above zero for both comparisons.
2. **Interaction:** the interaction-aware hit rate improves over the additive
   independent baseline by at least 0.05, with a paired 95% bootstrap lower
   bound above zero.
3. **Oracle headroom:** the Pareto-dominance opportunity rate over the frozen
   static profile is at least 0.05 on the fixed cost frame, with a paired 95%
   bootstrap lower bound above zero.

Only a successful signal gate permits interaction; only a successful interaction
gate permits oracle headroom. Useful headroom is necessary before considering
later controller decisions and does not authorize implementation, architecture,
amortization, or serving claims. Valid below-margin positives are practical
nulls, intervals including zero are inconclusive, and worse results are
negative. Protocol revision is reserved for validity failures.

## Reproducibility and retained uncertainty

Every run must record model/dataset revisions, manifests, configuration, seeds,
backend mode, software environment, hardware identity, and immutable raw
artifacts. The reversible implementation and evaluation details formerly
retained as handoff uncertainty are fixed in the
[Stage-1 execution contract](stage-1-execution-contract.md). This includes
the tokenizer/software tuple, inference controls, prompt and answer
evaluation, output diagnostics, bootstrap procedure, target sets, static and
planner contracts, artifact schemas and lineage, profiling, measurement
scopes, gate ordering, final-data sealing, smoke mode, sharding, resume, and
failure recovery.

The following remain execution observations rather than choices: the actual
GPU UUID selected per run, the subset `P_exec` that passes complete execution,
the artifact hashes and locations, invalid-run outcomes, and which cost
dimensions have valid common coverage. Hidden-state distance diagnostics and
numeric per-dimension budget ceilings are explicitly excluded; an unavailable
prefetch path is recorded as omitted/unavailable rather than zero.

A missing field blocks or invalidates the corresponding evaluation path; it does
not permit silent default selection.

## Provenance and claim boundaries

The values above are transcribed from the closed decision tickets and ADRs.
Their source facts, inferences, and project preferences remain distinguished
in the individual records. The acceptance inference is that the contracts
share one model, one reference/ground-truth distinction, one task/split
boundary, one quality/degradation rule, one evidence/cost policy, and one
ordered gate sequence after issue 22's metric correction.

The project preference is a conservative five-percentage-point margin, paired
uncertainty, explicit null/inconclusive/negative reporting, minimal task
coverage, and no controller authorization from gross oracle headroom. No
implementation or experiment is authorized here.

## Authoritative ADRs

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](../adr/0003-stage-1-cost-contract.md)
- [ADR-0004: Stage-1 pilot model and immutable revision](../adr/0004-stage-1-pilot-model.md)
- [ADR-0005: Stage-1 4/8 group execution semantics](../adr/0005-stage-1-group-execution-semantics.md)
- [ADR-0006: Stage-1 task suite and immutable query-ID splits](../adr/0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0007: Stage-1 high-precision reference and ground-truth role](../adr/0007-stage-1-high-precision-reference.md)
- [ADR-0008: Stage-1 primary and diagnostic metrics](../adr/0008-stage-1-primary-and-diagnostic-metrics.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](../adr/0009-stage-1-degradation-and-epsilon.md)
- [ADR-0010: Stage-1 oracle profile enumeration and sampling](../adr/0010-stage-1-oracle-profile-enumeration-and-sampling.md)
- [ADR-0011: Stage-1 signal test and success threshold](../adr/0011-stage-1-signal-test-and-success-threshold.md)
- [ADR-0012: Stage-1 interaction test and success threshold](../adr/0012-stage-1-interaction-test-and-success-threshold.md)
- [ADR-0013: Stage-1 oracle-headroom test and stop criterion](../adr/0013-stage-1-oracle-headroom-and-stop-criterion.md)
- [ADR-0014: Stage-1 stop/continue rules](../adr/0014-stage-1-stop-continue-rules.md)

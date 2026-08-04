# QBitPlan Stage-1 execution contract

- Status: ACCEPTED implementation and evaluation contract for the `$to-spec` handoff
- Scientific source: [Stage-1 falsification protocol](stage-1-falsification-protocol.md)
- Implementation specification: [Stage-1 implementation and evaluation specification](stage-1-implementation-evaluation-spec.md)
- Durable decisions: [ADR-0003](../adr/0003-stage-1-cost-contract.md) through
  [ADR-0015](../adr/0015-stage-1-consolidated-protocol.md)

This document records reversible implementation and evaluation details needed
to execute and reproduce the accepted Stage-1 gates. It does not amend the
accepted ADRs, authorize a production controller, or select an out-of-scope
serving architecture.

## 1. Per-query target-set construction

### Accepted

For each phase `s` and evaluated dataset `d`, the feasible profile set is:

`F_{s,d} = { p in P_exec : D_{d,s}(p) <= 0.01 }`.

The set uses only the records permitted for that phase and dataset. Final
query IDs are not used for training, profile construction, threshold
selection, or gate declaration.

For each query `q`, the target set is:

`T_{s,d}(q) = { p in F_{s,d} : c(d,q,p) = 1 }`.

The hit endpoint counts a selected profile as a hit exactly when it belongs
to this set. Correctness is judged against the external task target. An empty
target set is a recorded miss, not a reason to substitute a profile or alter
the feasibility rule.

## 2. Common cost envelope

For the static-versus-oracle gross headroom comparison, the common included
dimension set is:

`J = { resident accelerator bytes, host-to-device bytes, latency,
prefetch stall time, kernel switch count }`.

The full six-dimensional cost vector from ADR-0003 remains the reporting
contract. Controller/probe/feedback overhead is recorded whenever a compared
variant executes that path; it is not part of this offline static-versus-
oracle `J`.

Stage 1 declares no numeric per-dimension budget ceilings. Cost comparison
uses the accepted componentwise/Pareto rule only.

## 3. Static-profile selection

Construct `F_train` from the 7,500 permitted MATH training records. For each
profile in `F_train`, compute its mean cost vector over the frozen 256-query
training cost frame and the common dimensions `J`. Retain the profiles that
are Pareto-minimal under those mean vectors, then select the smallest
canonical profile ID among the retained profiles.

Profile IDs are the lexicographic binary IDs `00000000` through `11111111`
defined by ADR-0010. The profile ID is the deterministic tie-break for
identical cost vectors and for multiple incomparable profiles on the training
cost frontier. The selected static profile is frozen before validation
outcomes are read. An empty `F_train` invalidates the static baseline; no
infeasible or validation-selected substitute is allowed.

## 4. Query-only feature and embedding contract

The primary feature vector uses only the canonical query text and structure
available before profile selection. Structural features are token count,
character count, line count, digit count, whitespace count, punctuation count,
and option count. The dataset label is not a primary feature.

Compute a fixed embedding by mean-pooling the BF16 pilot model's
`model.embed_tokens` rows over non-padding input IDs. Cast the pooled vector to
`float32` and L2-normalize it; retain an all-zero vector as all-zero. Fit any
scalar-feature normalization statistics on MATH training records only, with a
unit scale for a zero-variance feature. Concatenate the normalized structural
features and normalized embedding.

Compute features before any Transformer group runs. Exclude positions,
hidden states, logits, generated tokens, external answers, reference or
quantized outcomes, and profile IDs. The exact input serialization is
inherited from the accepted prompt-formatting contract.

## 5. Independent baseline

For training query `q`, group `g`, and bit `b`, define the marginal target
support label:

`y(q,g,b) = 1` iff there exists `p` in `T_train(q)` with `p_g = b`.

Fit one minimum-norm ordinary-least-squares scorer per `(g,b)` using the
shared query-only feature vector with an intercept:

`score(g,b,q) = w_{g,b} dot [1, x(q)]`.

Include training queries with empty target sets as all-zero examples. Select
the independent profile as the executable-profile maximizer:

`p_ind(q) = argmax_{p in P_exec} sum_g score(g, p_g, q)`.

Resolve equal scores by canonical profile ID. The scorer has no upstream
context or interaction term, and no cost scalar or unaccepted budget is used.

## 6. Interaction-aware representation and planner

At group `g`, the planner receives the query-only features plus causal
upstream context. For `g=0`, `u_g` is a zero vector. For `g>0`, `u_g` is
the mean-pooled hidden state after the actually executed prefix through
group `g-1`, over non-padding positions; it is cast to `float32` and
L2-normalized. The already selected prefix bits are represented one-hot.

Use the deterministic interaction feature map:

`z_g = [1, structural_features, query_embedding, u_g,`
`query_embedding * u_g, prefix_bits]`.

Training examples use each training target profile and its own complete
ordered prefix context. A bit is supported for a prefix when any training
target profile with that prefix selects the bit. Fit minimum-norm ordinary
least-squares scorers for the two bit choices at each group.

At runtime, score only executable continuations, select the higher-scoring
bit, resolve equal scores in favor of `4`, execute that group, and then
construct the next context from the resulting hidden state. No future
hidden state, correctness result, answer, or later-group outcome may affect
an earlier decision. A prefix with no executable continuation invalidates
the run; no substitute profile is allowed.

The planner is an offline research baseline, not a production router or
controller architecture. Its feature extraction, feedback, and context
construction are included in the applicable controller/probe/feedback
overhead record.

## 7. Prompt formatting

Use zero-shot raw-text prompts with no chat template, system message, or
few-shot examples.

MATH prompt:

    Problem:
    {problem}

    Solve the problem. Show your reasoning and put the final answer in
    \\boxed{...}.
    Solution:

MMLU-Pro prompt:

    Question:
    {question}

    Options:
    A. {options[0]}
    B. {options[1]}
    ...
    J. {options[9]}

    Answer with the single letter of the correct option.
    Answer:

Use only the problem/question/options fields. Exclude solutions, answer
keys, answer indices, subject/category labels, and difficulty labels. Preserve
option order. Line-ending and Unicode normalization, answer extraction, and
grading remain defined by the answer-evaluation section.

## 8. Answer evaluation

Normalize generated text with Unicode NFKC, convert CRLF and CR to LF, and
trim surrounding whitespace.

For MMLU-Pro, accept only a single option letter `A` through `J`, optionally
followed by `.` or `)`, or the exact forms `Answer: <letter>` and
`Final answer: <letter>`. Explanations, multiple letters, and parse failures
are incorrect.

For MATH, use the original MATH answer-extraction and mathematical-
equivalence evaluator from source revision
`985bdc1696e88e8643f081a0ff4719da39f2ae2a`; no custom fallback parser is
added. Ground truth is the external task target, never BF16 agreement.

Parse failures are incorrect and are recorded separately. BF16 and every
profile use the same prompt, decoder controls, normalization, parser, and
ground-truth comparison.

## 9. Output diagnostics

Generate one BF16 completion under the decoding controls in section 12. Use
that exact BF16 token sequence as a teacher-forced diagnostic continuation
for BF16 and every executable profile.

At every generated-token position, excluding prompt positions, compare the
full-vocabulary distributions with forward KL in `float32`:

`KL(P_BF16 || P_profile)`.

Report the per-query mean over generated positions. MMLU-Pro has one answer
position; MATH covers the BF16 continuation through EOS or the fixed
generation limit. Run this as a separate post-run diagnostic pass, record
its overhead separately, and exclude it from the primary execution-cost
frame. Diagnostics do not influence selection, training, targets, or gates.

Hidden-state distance diagnostics are explicitly omitted from Stage 1. The
causal hidden-state summaries used by the interaction-aware planner are not
diagnostic distance metrics.

## 10. Bootstrap uncertainty

Use a nonparametric paired bootstrap over complete query records. Resample
the 4,500 validation query IDs for signal and interaction with replacement;
preserve every method outcome for each sampled query. For oracle headroom,
resample the 256 fixed cost-frame query IDs; profiles within a query are not
independent bootstrap units.

Use 10,000 replicates with NumPy `PCG64` and independent seeds: signal
`20260804`, interaction `20260805`, and oracle headroom `20260806`.
Compute percentile 95% intervals using explicit linear interpolation at the
2.5th and 97.5th percentiles. The gate lower bound is the 2.5th percentile.
Do not use BCa correction, adaptive resampling, or bootstrap-based tuning.

## 11. Tokenizer and software tuple

Use the fast tokenizer for `meta-llama/Llama-3.1-8B` at immutable revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. Fail closed if the fast tokenizer
files are unavailable; do not fall back to another tokenizer implementation.
Use `trust_remote_code=False` and record tokenizer configuration and file
content hashes.

The verified tuple is Python `3.12.3`, PyTorch `2.4.0+cu124`, Transformers
`5.12.1`, TorchAO `0.5.0`, NumPy `2.1.0`, Datasets `5.0.0`, Accelerate
`1.14.0`, Safetensors `0.8.0`, CUDA `12.4`, and NVIDIA driver `580.159.03`.
Record the per-run GPU UUID separately.

## 12. Inference and decoding controls

The accepted generation regime uses deterministic beam search with
`max_new_tokens=1024`, `num_beams=4`, and `num_return_sequences=1`.
`do_sample=False`, `early_stopping=True`, `length_penalty=1.0`,
`repetition_penalty=1.0`, `no_repeat_ngram_size=0`, one beam group, and zero
diversity penalty are explicit controls.

Run one process on one UUID-pinned GPU with batch size `1`, `model.eval()`,
`torch.inference_mode()`, BF16 model loading, and `use_cache=True`. Disable
`torch.compile`, graph capture, CPU offload, dynamic batching, and padding.
Tokenize with `add_special_tokens=True`, `padding=False`, and
`truncation=False`; set `pad_token_id=eos_token_id` and `eos_token_id`
explicitly. Reject prompt-plus-generation overflow; never truncate.

Set Python, NumPy, Torch CPU, and Torch CUDA seeds to `20260807`. Disable
TF32, set float32 matmul precision to `highest`, disable cuDNN benchmarking,
enable deterministic cuDNN behavior, require
`torch.use_deterministic_algorithms(True)`, and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Unsupported deterministic operations make
the run invalid; controls are not relaxed.

## 13. Artifact identity, schema, and lineage

Use immutable content hashes: `artifact_id` is the SHA-256 of file bytes,
`manifest_id` is the SHA-256 of RFC-8785 canonical manifest JSON, and
`run_id` is the SHA-256 of canonical run-identity JSON including a
deterministic `attempt_id` but excluding volatile timestamps and outcomes.

Manifests and metadata use canonical JSON. Per-query/profile records use
strict-schema UTF-8 NDJSON, one record per query/profile execution. Tensor
arrays and token sequences use immutable Safetensors artifacts.

Every derived artifact records its artifact type, schema version, producer
git SHA, source artifact IDs, source manifest ID, configuration hash, record
count, and creation timestamp. Required artifact types are phase manifest,
run manifest, profile-feasibility record, per-query outcome, raw cost/trace
artifact, bootstrap result, and gate report. Raw artifacts are write-once;
derived summaries point to their sources.

## 14. Invalid-run semantics

Terminal statuses are `complete`, `invalid`, `incomplete`, and `aborted`.
A profile-level transform or complete-forward failure excludes the profile
from `P_exec`; no BF16, nearby-profile, or fake-quantized substitute is
allowed.

A query-level runtime failure is not scored as incorrect or silently dropped.
Missing required paired records invalidate the corresponding gate. A
measurement failure invalidates only affected cost dimensions when quality
remains valid; those dimensions are recorded as
`omitted/unavailable/<reason>`.

Dropped traces, unsupported deterministic operations, schema or provenance
mismatches, non-finite outputs, OOMs, and context overflow receive explicit
reason codes and immutable logs. Invalid or incomplete attempts are excluded
from aggregates but retained in lineage; recovery uses a new `attempt_id`.

## 15. Profiling and repetition

Exclude model loading and quantization from query execution timing; record
them separately. For each query/profile/reference, run five unprofiled warm
ups, then ten unprofiled measured repetitions and report the median
synchronized latency. Run a separate traced pass with the same query and
controls for trace-derived stalls and kernel switches; traced latency is not
the latency metric and tracer overhead is recorded separately.

Reset input state between repetitions and do not reuse a prior query's KV
cache. Synchronize before and after each timed interval. Use CUDA events for
device elapsed time and retain host timestamps for lineage.

## 16. Cost measurement scopes

Resident accelerator bytes are peak absolute NVML device-used bytes during
the loaded query interval, including resident profile weights and runtime
allocations; the pre-query baseline is recorded separately and not
subtracted. Host-to-device bytes are the sum of H2D CUDA-copy records from
query dispatch through completion, including input and runtime transfers but
excluding model loading and quantization.

Latency is host monotonic time from dispatch before input transfer through
final synchronized output readiness, including H2D, generation, and planner
overhead. Prefetch stall time is trace-derived from positive gaps between
required copy completion and first dependent annotated kernel; an absent
accepted prefetch path is `omitted/unavailable/no-prefetch-path`, never zero.
Kernel switches are trace-derived precision/configuration transitions between
consecutive annotated group launches; incomplete traces omit the dimension.

Controller/probe/feedback overhead is annotated CPU and GPU time for feature
extraction, scoring, hidden-state pooling, feedback, and synchronization. It
is not applicable to offline static/oracle selection. The effective cost set
is the common subset with valid coverage; no comparison is made outside it.

## 17. Analysis ordering and final-data sealing

Freeze and hash final manifests before calibration; final records remain
read-only and unavailable to training, profile construction, threshold
selection, and gate jobs.

Build training artifacts, `F_train`, target sets, the static profile, and
both planners from non-final MATH. Seal the validation signal report, run
interaction only after signal success, and run oracle headroom only after
interaction success. Valid null, inconclusive, or negative results stop the
research sequence; invalid results block interpretation.

After the gate sequence reaches a terminal status, release final manifests
read-only for confirmatory evaluation of already-frozen methods only. Final
data cannot alter gate statuses, fit parameters, thresholds, or profile
libraries. Each downstream job requires the predecessor's sealed, matching
report and has no manual override.

## 18. Smoke, sharding, resume, and recovery

Smoke mode uses the real pinned MATH loader and execution path on one
training and one validation record with BF16, all-4, all-8, and one mixed
profile. It never uses final IDs and cannot satisfy a gate.

A work unit is `(phase, query_id, profile_id, execution_mode, pass,
repetition)`. Assign it to a shard with
`int(SHA256(work_key), 16) mod shard_count`; shards write separate immutable
artifacts and merge only after hash and lineage validation.

Resume reuses only a matching complete work unit from the same run plan.
Retries create a new `attempt_id` and never overwrite artifacts. Allow at
most three attempts, retrying only allowlisted preemption, I/O, or
CUDA-context failures. Do not retry schema, provenance, non-finite-output,
unsupported-determinism, or OOM failures by changing controls. A required
work unit without one accepted attempt invalidates its dependent gate.

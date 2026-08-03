# Issue 7: Mixed 4/8 execution verification for the pilot candidates

Research date: 2026-08-03

Question: whether the four Issue 3 candidates were directly verified for
mixed per-group 4/8 execution, and whether `meta-llama/Llama-3.1-8B` fits the
same TorchAO path.

This note is research only. It does not select a pilot, change an ADR, update
an issue, or implement code. It records bounded candidate-specific model
loads and mixed-path forward checks. It supplements [Issue 3: Pilot model,
revision, and quantization/backend feasibility](issue-3-model-backend-feasibility.md).

## Decision rule and evidence boundary

Issue 3 predeclared this rule: retain only model/revision pairs with an
immutable public revision, enough regularity for eight contiguous groups,
primary-source model/backend documentation, and a reproducible path for both
4-bit and 8-bit group choices. The existing Issue 3 note labels all four
original candidates **conditional passes by inference** and explicitly says a
model-specific run is still required.

The phrase “directly verified” below means a primary-source or QBitPlan
artifact that actually applies the 4/8 mapping to the named model and runs the
resulting model path. A generic API unit test, a configuration example, or a
uniform-precision benchmark is not direct evidence for a candidate.

## Findings

### 1. The four Issue 3 candidates were directly verified

#### Source facts

The pinned Hugging Face configurations confirm the architectural regularity
used by Issue 3:

| Candidate | Pinned revision | Architecture | `num_hidden_layers` |
| --- | --- | --- | ---: |
| `microsoft/phi-2` | `810d367871c1d460086d9f82db8696f2e0a0fcd0` | `PhiForCausalLM` | 32 |
| `EleutherAI/pythia-2.8b` | `2a259cdd96a4beb1cdf467512e3904197345f6a9` | `GPTNeoXForCausalLM` | 32 |
| `facebook/opt-2.7b` | `905a4b602cda5c501f1b3a2650a4152680238254` | `OPTForCausalLM` | 32 |
| `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `MistralForCausalLM` | 32 |

Primary sources: [Phi-2 pinned
config](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/config.json),
[Pythia pinned
config](https://huggingface.co/EleutherAI/pythia-2.8b/blob/2a259cdd96a4beb1cdf467512e3904197345f6a9/config.json),
[OPT pinned
config](https://huggingface.co/facebook/opt-2.7b/blob/905a4b602cda5c501f1b3a2650a4152680238254/config.json),
and [Mistral pinned
config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json).

#### Candidate-specific mixed-path check

On 2026-08-03, each pinned checkpoint was loaded on an NVIDIA RTX 3090 and
run through the same bounded profile: groups 0, 2, 4, and 6 received
TorchAO `int4_weight_only(group_size=128)`, while groups 1, 3, 5, and 7
received `int8_weight_only()`. Each group contains four contiguous Transformer
layers. The check asserted the linear-module count before and after each
transform, inspected the resulting weight layouts, and ran a fixed forward
pass both before and after mixed quantization.

| Candidate | GPU | Linear modules/group | Mixed logits shape | Reference / mixed finite | Peak allocated |
| --- | ---: | ---: | --- | --- | ---: |
| `microsoft/phi-2` | 0 | 24 | `(1, 5, 51200)` | yes / yes | 5.39 GiB |
| `EleutherAI/pythia-2.8b` | 1 | 16 | `(1, 5, 50304)` | yes / yes | 5.38 GiB |
| `facebook/opt-2.7b` | 4 | 24 | `(1, 6, 50272)` | yes / yes | 5.30 GiB |
| `mistralai/Mistral-7B-v0.1` | 3 | 28 | `(1, 6, 32000)` | yes / yes | 13.88 GiB |

For every row, all eight groups retained the expected count. The int4 groups
reported TorchAO `TensorCoreTiledAQTLayout` with `torch.int32` packed layout
data, and the int8 groups reported `PlainAQTLayout` with `torch.int8` layout
data. These are direct candidate-specific execution facts for the tested
stack, not a task-quality or performance result.

The Phi-2, Pythia, and Mistral checks used Python 3.12, PyTorch 2.4.0+cu124,
Transformers 5.12.1, and TorchAO 0.5.0. The pinned OPT revision exposes a
legacy PyTorch `.bin` checkpoint; Transformers correctly refused to load it
with PyTorch 2.4 because of the `torch.load` safety requirement. OPT was then
rerun successfully in an isolated temporary environment with PyTorch
2.6.0+cu124, TorchAO 0.6.1, matching torchvision/torchaudio, and the same
Transformers 5.12.1 model path. The project environment and dependency files
were not changed by that retry.

TorchAO’s official stable documentation states that `Int4WeightOnlyConfig`
and `Int8WeightOnlyConfig` are weight-only workflows, and that
`FqnToConfig` applies different quantization configurations by fully
qualified module or parameter name. Its Llama example places int4 and int8
configs on different Llama module FQNs. These are API capabilities and a
configuration example, not a run of any Issue 3 candidate. See
[TorchAO inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html),
[TorchAO quantization API reference](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html),
and [TorchAO’s FQN configuration example](https://docs.pytorch.org/ao/stable/eager_tutorials/torchao_vllm_integration.html#3-fqn-configuration).

TorchAO’s official tests do directly exercise mixed configuration dispatch on
a CUDA `ToyLinearModel`: one FQN receives `Int4WeightOnlyConfig`, while a
regex-matched second linear receives the default `IntxWeightOnlyConfig`,
whose source default is `torch.int8`. The test is evidence for generic
TorchAO dispatch, not for Phi-2, Pythia, OPT, or Mistral. See [the mixed FQN
test](https://github.com/pytorch/ao/blob/main/test/quantization/test_quant_api.py#L482-L516)
and [the `IntxWeightOnlyConfig` default](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py#L1391-L1429).

TorchAO’s published inference accuracy table includes `meta-llama/Llama-3.1-8B`,
but the listed rows are uniform modes such as bfloat16, float8, and int8; it
does not report a per-group mixed 4/8 run for that model or for the four Issue
3 candidates. See [the TorchAO inference benchmark
table](https://docs.pytorch.org/ao/stable/workflows/inference.html#accuracy-benchmarks).

#### Inference

The completed runs establish candidate-specific mixed-path execution for all
four pinned rows on the tested RTX 3090/software stacks. They do not establish
that the newer `FqnToConfig` API is required or preferred, that an untested
final environment will behave identically, or that mixed execution improves
task quality or systems performance. The result is eligibility evidence, not
a project selection.

### 2. Llama 3.1 8B is structurally compatible, conditionally

#### Source facts

The official Hugging Face model API identifies
`meta-llama/Llama-3.1-8B` as a public, manually gated Transformers repository
at revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. See the [official model API
metadata](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B).

The exact pinned `config.json` at that revision records:

| Key | Exact value |
| --- | --- |
| `architectures` | `["LlamaForCausalLM"]` |
| `model_type` | `"llama"` |
| `hidden_size` | `4096` |
| `intermediate_size` | `14336` |
| `num_hidden_layers` | `32` |
| `num_attention_heads` | `32` |
| `num_key_value_heads` | `8` |
| `max_position_embeddings` | `131072` |
| `rope_theta` | `500000.0` |
| `rope_scaling` | `factor: 8.0`, `low_freq_factor: 1.0`, `high_freq_factor: 4.0`, `original_max_position_embeddings: 8192`, `rope_type: "llama3"` |
| `torch_dtype` | `"bfloat16"` |
| `vocab_size` | `128256` |
| `transformers_version` | `"4.43.0.dev0"` |

Primary source: [Llama 3.1 pinned
config](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json).
Meta’s [official Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)
also identifies the 8B text-only model, 128K context, and grouped-query
attention.

The official Hugging Face Llama implementation constructs a `model.layers`
`ModuleList` of decoder layers. Each decoder layer contains
`self_attn.q_proj`, `k_proj`, `v_proj`, and `o_proj` linear modules plus MLP
`gate_proj`, `up_proj`, and `down_proj` linear modules. See the [official
Llama Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).
TorchAO’s official FQN example uses the same Llama namespace, including
`model.layers.0.self_attn.q_proj` and `model.layers.0.mlp.gate_proj`, with
different int4 and int8 configs. See the [TorchAO Llama FQN
example](https://docs.pytorch.org/ao/stable/eager_tutorials/torchao_vllm_integration.html#3-fqn-configuration).

#### Inference

Llama 3.1 8B is **structurally eligible as an additional candidate** for the
same FqnToConfig/TorchAO path:

- 32 layers imply the same eight contiguous groups of four under the project
  preference;
- its standard Transformers Llama module hierarchy exposes the linear FQNs
  used by TorchAO’s example; and
- the backend documentation includes both 4-bit and 8-bit weight-only configs
  and per-FQN selection.

This is a structural inference, not direct mixed-execution evidence. GQA,
RoPE scaling, the 128K context setting, and the manually gated checkpoint do
not by themselves disqualify the model, but they make the model-specific load,
FQN coverage, kernel, and environment checks necessary.

#### Candidate-specific mixed-path check

On 2026-08-03, the pinned checkpoint was loaded and checked on one available
NVIDIA RTX 3090 (24 GiB) with Python 3.12, PyTorch 2.4.0+cu124, Transformers
5.12.1, and TorchAO 0.5.0. TorchAO 0.5.0 predates the newer stable
`FqnToConfig` API, so this check used its real `quantize_` path with explicit
per-group filters and `int4_weight_only`/`int8_weight_only` transforms. It did
not change the repository dependency set.

The profile assigned int4 to groups 0, 2, 4, and 6 and int8 to groups 1, 3,
5, and 7, with group size 128 for int4. Each four-layer group matched 28
linear modules (seven projections per layer). The int4 groups reported
TorchAO `TensorCoreTiledAQTLayout` weights with `(1, 128)` blocks; the int8
groups reported `PlainAQTLayout` weights with int8 layout data and `(1, 4096)`
blocks. A fixed forward pass produced logits of shape `(1, 6, 128256)` with
all values finite; peak allocated GPU memory was approximately 15.34 GiB.

This is now direct candidate-specific mixed-path evidence for Llama 3.1 8B on
the tested RTX 3090/software stack. It is not a quality, latency, memory
benefit, or cross-hardware result, and it does not verify the newer
`FqnToConfig` API path.

#### Project preference

The user’s prior experience with Llama and authenticated access in this
environment are reasonable project preferences for retaining it in the
selection comparison. They do not establish cross-environment reproducibility,
license/access availability for collaborators, task-suite suitability, or
mixed execution.

### 3. What remains unverified

#### Uncertainty

The following claims remain open for all five candidates outside the bounded
checks above:

- a candidate-specific `FqnToConfig` mapping, as distinct from the legacy
  explicit-filter path used here, transforms every intended linear in each
  four-layer group and leaves embeddings, norms, and heads governed by an
  explicit policy;
- the selected final TorchAO/PyTorch/Transformers versions accept each pinned
  config, especially Llama 3.1’s RoPE schema and OPT’s legacy checkpoint
  format;
- the chosen group size, packing format, dtype, and kernel remain supported on
  the declared final hardware and driver;
- the four original models and Llama fit the intended task suite and quality
  threshold; and
- memory, latency, kernel switches, prefetch behavior, and controller overhead
  satisfy the project’s scientific reporting requirements. No direct systems
  benefit follows from the structural inference.

HF manual gating is an additional reproducibility condition for Llama: the
repository metadata is public but access to the files is approval-dependent.
The local login proves access for this environment only.

## Verification scope and remaining checks

The bounded check above is sufficient to establish candidate-specific mixed
execution on the tested environments. Before a final protocol locks a
candidate, repeat or promote the check on the declared final environment and
retain the following evidence:

1. Pin the model identifier and revision, tokenizer revision, Python,
   PyTorch, Transformers, TorchAO, CUDA/driver, device, dtype, group size,
   packing format, and quantization parameters.
2. Load the pinned model without changing its config, enumerate its named
   modules, and assert the expected 32-layer hierarchy and the exact linear
   FQN inventory.
3. Define eight contiguous four-layer groups and build a mapping containing at
   least one 4-bit group and one 8-bit group. Explicitly account for every
   non-group module; do not rely on an accidental default.
4. Apply the selected documented `FqnToConfig`/`quantize_` path and assert, by FQN and
   tensor representation, that every intended group received its declared
   bit width and that no intended module was skipped or quantized with the
   wrong config.
5. Run one fixed, short forward pass on the declared target hardware, checking
   for finite outputs and recording any fallback, unsupported-kernel, packing,
   or shape error. Compare against an unquantized reference sufficiently to
   detect a broken path; this is a path check, not a quality claim.
6. Repeat with all-4 and all-8 controls if needed to distinguish mixed-dispatch
   failures from general model or kernel failures. Save the module inventory,
   profile, logs, environment, hardware identity, and raw artifacts.

Passing this check establishes candidate-specific mixed-path execution only.
It does not select the pilot or establish task quality, measured systems
benefit, or generalization. Any failure leaves the candidate conditional or
removes it under Issue 3’s predeclared rule; it must not be replaced with
fake-quantized or uniform-precision evidence.

## Conclusion

All four Issue 3 candidates and Llama 3.1 8B now have successful
candidate-specific mixed-path checks on the tested RTX 3090 stacks. The
checks used TorchAO’s real legacy `quantize_` path with explicit per-group
filters; OPT required an isolated PyTorch 2.6 loader environment because its
pinned checkpoint is a legacy `.bin` file. These results satisfy the
candidate-specific execution portion of the predeclared eligibility rule, but
do not resolve task-suite fit, final environment choice, quality, performance,
or the human pilot selection. No candidate is selected by this note.

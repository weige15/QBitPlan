# Issue 3: Pilot model, revision, and quantization/backend feasibility

Research date: 2026-08-03  
Ticket: [Verify pilot model, revision, and quantization/backend feasibility](https://github.com/weige15/QBitPlan/issues/3)

## Scope and rule

This note applies the ticket's predeclared rule:

> Retain only model/revision pairs with an immutable public revision, enough
> architectural regularity for an eight-contiguous-group pilot,
> primary-source documentation of the model and backend, and a reproducible
> path to execute both 4-bit and 8-bit group choices.

The comparison set is a bounded research sample of public 32-layer,
decoder-only causal language models. The 32-layer filter follows the
project's research-input description of eight groups of four layers; it is a
project preference for this pilot, not a claim that these are the only viable
models. No model or backend is selected here. Selection remains the scope of
[Select the pilot model identifier and exact revision](https://github.com/weige15/QBitPlan/issues/7).

## Source facts

### Model and revision metadata

The Hugging Face model API exposes a repository commit SHA for each public
model. The pinned `config.json` at that SHA supplies the architecture and
layer count. The following values were read on 2026-08-03:

| Model/repository | Public revision SHA | Architecture | Layers | Hidden size | Max positions | Source task/use signals |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `microsoft/phi-2` | `810d367871c1d460086d9f82db8696f2e0a0fcd0` | `PhiForCausalLM` | 32 | 2560 | 2048 | Model card describes QA, chat, and code prompting; English-focused. |
| `EleutherAI/pythia-2.8b` | `2a259cdd96a4beb1cdf467512e3904197345f6a9` | `GPTNeoXForCausalLM` | 32 | 2560 | 2048 | Pythia is an English research language-model suite trained on the Pile; its card documents evaluation-harness results and intended research use. |
| `facebook/opt-2.7b` | `905a4b602cda5c501f1b3a2650a4152680238254` | `OPTForCausalLM` | 32 | 2560 | 2048 | OPT is predominantly English, pretrained with a causal-LM objective, and its card says the pretrained model can be prompted for downstream-task evaluation and text generation. |
| `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `MistralForCausalLM` | 32 | 4096 | 32768 | Model card describes a pretrained generative text model with grouped-query and sliding-window attention; it is a base model, not an instruction-tuned checkpoint. |

Primary sources for the table:

- [Phi-2 API metadata](https://huggingface.co/api/models/microsoft/phi-2), [Phi-2 pinned config](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/config.json), and [Phi-2 pinned model card](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/README.md).
- [Pythia API metadata](https://huggingface.co/api/models/EleutherAI/pythia-2.8b), [Pythia pinned config](https://huggingface.co/EleutherAI/pythia-2.8b/blob/2a259cdd96a4beb1cdf467512e3904197345f6a9/config.json), and [Pythia pinned model card](https://huggingface.co/EleutherAI/pythia-2.8b/blob/2a259cdd96a4beb1cdf467512e3904197345f6a9/README.md).
- [OPT API metadata](https://huggingface.co/api/models/facebook/opt-2.7b), [OPT pinned config](https://huggingface.co/facebook/opt-2.7b/blob/905a4b602cda5c501f1b3a2650a4152680238254/config.json), and [OPT pinned model card](https://huggingface.co/facebook/opt-2.7b/blob/905a4b602cda5c501f1b3a2650a4152680238254/README.md).
- [Mistral API metadata](https://huggingface.co/api/models/mistralai/Mistral-7B-v0.1), [Mistral pinned config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json), and [Mistral pinned model card](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/README.md).

All four configurations therefore have an exact 32/8 partition of four
Transformer layers per contiguous group. That arithmetic is an inference
from the pinned configuration plus the project's eight-group preference; it
does not establish that every submodule in a group can already be quantized
by a selected runtime.

### Backend facts

| Backend | Documented weight-only capability | Documented module selection | 4/8 group-choice assessment |
| --- | --- | --- | --- |
| [TorchAO 0.17 inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html) | Stable `Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`, and `IntxWeightOnlyConfig` workflows; the latter covers 1–8-bit integer weights. | [`FqnToConfig`](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.FqnToConfig.html) applies different quantization configs by fully qualified module or parameter name; [`quantize_`](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html) transforms selected linear modules. | **Conditional survivor.** The official API directly describes the needed 4/8 per-module mapping, so eight contiguous groups can be represented by FQN patterns. Model-specific module coverage and hardware kernels still need a real verification run. |
| [Transformers + bitsandbytes](https://huggingface.co/docs/transformers/v4.46.0/en/quantization/bitsandbytes) | Documents 4-bit and 8-bit loading for models containing `torch.nn.Linear` layers. | The documented `BitsAndBytesConfig` examples select a model-wide 4-bit or 8-bit load; the reviewed documentation does not provide a per-FQN mixed 4/8 recipe. | **Not retained under the declared rule as documented.** Uniform 4 and uniform 8 are documented; group-level mixed 4/8 would require an additional module-replacement design and verification. |
| [Optimum Quanto](https://huggingface.co/docs/transformers/quantization/quanto) | Documents weight quantization at `int4` and `int8`, and says the Transformers integration works for models containing `torch.nn.Linear` layers. | The reviewed integration exposes a model-level `QuantoConfig(weights=...)`; no official per-FQN mixed 4/8 recipe was found. | **Not retained under the declared rule as documented.** It is evidence for uniform 4/8 execution, not yet for the required group-level mixed choices. |

TorchAO is the only reviewed backend with primary documentation that combines
both required bit widths with an explicit per-module configuration mechanism.
Calling it a feasibility survivor is an inference from the documented APIs,
not direct execution evidence. The backend version, PyTorch version, GPU/CPU
kernel availability, packing format, and whether every target model's linear
module names map cleanly to eight groups remain OPEN.

## Comparison against the rule

| Candidate | Immutable revision | Eight-group regularity | Model/backend primary sources | Reproducible 4/8 group path | Status for ticket 7 |
| --- | --- | --- | --- | --- | --- |
| `microsoft/phi-2` + TorchAO | Pass: public Hub SHA above | Pass by 32 layers → 8 × 4; Phi architecture and any remote-code requirements must be pinned | Pass: pinned model config/card plus TorchAO docs | **Conditional pass by inference** through TorchAO FQN mapping; requires model-specific run | Retain for selection comparison; no selection made |
| `EleutherAI/pythia-2.8b` + TorchAO | Pass: public Hub SHA above | Pass by 32 layers → 8 × 4; GPT-NeoX block structure is documented | Pass: pinned model config/card plus TorchAO docs | **Conditional pass by inference** through TorchAO FQN mapping; requires model-specific run | Retain for selection comparison; no selection made |
| `facebook/opt-2.7b` + TorchAO | Pass: public Hub SHA above | Pass by 32 layers → 8 × 4; OPT block structure is documented | Pass: pinned model config/card plus TorchAO docs | **Conditional pass by inference** through TorchAO FQN mapping; requires model-specific run | Retain for selection comparison; license and hardware checks remain |
| `mistralai/Mistral-7B-v0.1` + TorchAO | Pass: public Hub SHA above | Pass by 32 layers → 8 × 4; GQA/SWA are architecture details to preserve | Pass: pinned model config/card plus TorchAO docs | **Conditional pass by inference** through TorchAO FQN mapping; requires model-specific run | Retain for selection comparison; hardware feasibility remains especially material |

The four rows are an eligibility set, not a ranking. The task-suite column
cannot be resolved here because the task-suite and immutable splits are the
scope of later map tickets. The source cards establish causal-LM loading and
different intended-use/evaluation facts, but they do not prove compatibility
with the eventual QBitPlan suite or quality metric.

## Uncertainty and handoff

- The exact task suite, dataset revisions, query-ID splits, reference format,
  target hardware, and cost dimensions are OPEN in the map. No candidate is
  preferred on those unresolved human/project choices.
- The model SHA is immutable, but the full reproducibility tuple still needs
  the tokenizer SHA/config, Python/PyTorch/Transformers/TorchAO versions,
  quantization parameters, packing format, seed, hardware identity, and raw
  artifact location.
- “Conditional pass” is not direct evidence that a model can execute a
  mixed profile on the target hardware. The next selection/definition work
  must run one real 4/8 group-path check for the surviving candidate(s), then
  record failures rather than silently falling back to fake quantization.
- Bitsandbytes and Quanto remain possible uniform-precision references, but
  their reviewed official integrations do not establish the ticket's
  group-level mixed-precision requirement.

## Conclusion

The rule does not select a pilot. It narrows the evidence-supported path to a
32-layer candidate set paired with TorchAO's documented per-FQN 4/8
weight-only APIs, subject to model-specific and hardware-specific verification
in the dependent selection and execution-semantics tickets.

---
title: Initial QBitPlan Research Brief
status: research-input
date: 2026-08-02
verification: primary-source verification pending
---

This document is the initial research synthesis that motivated QBitPlan.
It is background input, not an accepted specification or verified
bibliography. Claims intended for publication must be checked against
their primary sources.

## 1. The key research verdict

QAQ already establishes the feasibility of decomposing weights into bit-planes, using a query-conditioned router to select precision, and loading selected planes on demand. DP-LLM goes further temporally: it observes that layer sensitivity changes between decoding steps and uses the current layer input to estimate whether high or low precision is needed. MoBiQuant supplies a recursively quantized, token-adaptive representation in which additional residual slices reconstruct higher-bit weights.

Most importantly for your question about whether blocks are independent, the very recent MixQuant paper directly reports that a layer’s sensitivity changes substantially depending on the bit-widths of upstream layers; its measured sensitivity metrics can vary by orders of magnitude across upstream contexts, changing the resulting allocation. It then marginalizes sensitivity over sampled upstream quantization contexts and solves a multiple-choice knapsack problem. IMPQ independently models inter-layer interactions with Shapley-based estimates and a binary quadratic optimization problem.

Therefore:

* “Are layer decisions independent?” is no longer the novel question. The answer is generally no.
* “Can we formulate bit assignment as knapsack?” is a useful baseline, not the main novelty.
* A query-conditioned MLP that outputs a profile is also insufficient as the central novelty.

A stronger, defensible research direction is:

**Query-conditioned, interaction-aware bit planning with causal hidden-state correction and hardware-executable profile compression.**

A working name could be QBitPlan: Query-Conditioned Causal Bit Planning for Budgeted LLM Inference.

## 2. The precise research question

A clean formulation would be:

*Given a query q, a runtime resource budget B, and a progressively quantized Transformer, can a lightweight controller predict a hardware-executable mixed-precision profile that minimizes quality degradation, while accounting for both upstream quantization choices and causally available hidden-state signals?*

This gives four concrete research questions:

1. **Predictability:** Are query features predictive of which layer groups need extra precision?
2. **Interaction:** How much does upstream precision change the marginal value of later precision?
3. **Feedback:** Does observing actual hidden-state behavior improve future bit decisions enough to justify its runtime overhead?
4. **Execution:** Can many per-query optimal profiles be compressed into a small library of hardware-friendly profiles without losing most of the benefit?

The paper’s main hypothesis should not be “hard questions need more bits.” That is too coarse. A better hypothesis is:

*Two queries with similar total precision requirements can require their bits in different parts of the model, and this “precision shape” can be predicted from query and early-execution signals.*

## 3. Define “query difficulty” operationally

Do not initially label queries as easy, medium, or hard using human categories. Human difficulty, model difficulty, and quantization difficulty are different concepts.

Define the quantization difficulty of query $q$ as the minimum hardware cost required to preserve a reference quality level:

$$d_{\text{quant}}(q) = \min_{p \in P} C(p) \text{ subject to } D(q,p) \leq \epsilon,$$

where:

* $p = (b_1, \dots, b_G)$ is a bit profile over $G$ layer groups;
* $C(p)$ is measured cost;
* $D(q,p)$ is degradation relative to the high-precision reference;
* $\epsilon$ is the allowed degradation.

A single scalar $d_{\text{quant}}(q)$ is still incomplete. Your profiler should ideally predict three objects:

$$\psi(q) = (\hat{B}_q, \hat{s}_q, \hat{u}_q),$$

where:

* $\hat{B}_q$ is the estimated total precision budget;
* $\hat{s}_q \in \mathbb{R}^G$ is a precision-shape vector, estimating where extra bits are valuable;
* $\hat{u}_q$ is uncertainty or risk of under-precision.

This distinction solves a central problem in your example. Two profiles such as

$[4,4,8,4] \text{ and } [4,8,8,4]$

may have similar total budgets but different shapes. Whether they should belong to the same class depends on the behavior of the differing block, not merely their Hamming distance.

## 4. Recommended architecture

The scientifically clean architecture is not “MLP outputs all bits.” It is:

```text
                         ┌────────────────────────────┐
Query ──► Query/probe ──►│ Conditional distortion     │
          encoder        │ and interaction predictor  │
                         └─────────────┬──────────────┘
                                       │
                                       ▼
                         Budget-constrained optimizer
                         MCKP / quadratic / beam search
                                       │
                                       ▼
                            Initial precision profile
                                       │
                    Profile medoid / cache / prefetch
                                       │
                                       ▼
        Group 1 ─► monitor ─► Group 2 ─► monitor ─► ... ─► output
                               future-only precision promotions

```

### 4.1 Progressive quantized backbone

For the first implementation:

* Quantize weights only.
* Use bits {4,8}.
* Divide a 32-layer model into eight contiguous groups of four layers.
* Keep embeddings, normalization parameters, and the final language-model head at high precision.
* Initially assign one bit-width to the entire group.

In the second version, split each group into:

* attention projections: Q/K/V/O;
* MLP projections: gate/up/down.

Use nested residual slices:

$$W_g^{(2k)} = \sum_{r=1}^k S_{g,r},$$

where each $S_{g,r}$ is a two-bit slice. Four-bit execution uses two slices; eight-bit execution uses four. This is conceptually aligned with recursive residual slicing and single-checkpoint multi-precision methods such as MoBiQuant, Any-Precision LLM, and MatGPTQ.

A crucial systems detail is that selecting four bits does not automatically reduce peak GPU memory. It reduces GPU memory only when the unused residual slices are not resident. A realistic initial layout is:

* base four-bit slices permanently resident on the GPU;
* extra four-bit residual slices in pinned CPU memory;
* predicted residual slices prefetched before their layer group;
* a small GPU cache for frequently used residual slices.

If all slices remain resident, you have elastic computation but essentially eight-bit storage.

### 4.2 Query and probe encoder

Use three signal families.

**Static structural features, available before execution:**

* token count and context length;
* numeric and symbolic token ratios;
* number of answer choices;
* equation, table, and code-block indicators;
* question/instruction/continuation format;
* lexical or embedding-based task representation.

**Query embedding features:**

* mean- or attention-pooled input embeddings;
* a small frozen text encoder;
* alternatively, the target model’s embedding output with a small pooling head.

**Probe features, available after a fixed high-precision prefix:**

* hidden-state RMS and maximum magnitude;
* kurtosis or outlier fraction;
* LayerNorm statistics;
* attention entropy;
* MLP gate saturation;
* representation anisotropy or low-rank summary.

A sensible initial configuration is:

* structural vector: 32–64 dimensions;
* pooled query representation: 256 dimensions;
* fused query latent $z_q \in \mathbb{R}^{256}$;
* controller hidden dimension: 128;
* eight layer groups;
* 8–32 compiled profile medoids.

Treat these as starting values, not claims of optimality.

### 4.3 Query-conditioned distortion predictor

Instead of predicting the final bits directly, predict how damaging each choice is:

$$\hat{d}_{g,b}(q) = f_\theta(z_q, e_g, e_b),$$

where $e_g$ is a learned layer-group embedding and $e_b$ is a bit-width embedding.

The output is a table:

$$\hat{D}(q) \in \mathbb{R}^{G \times \vert{}B\vert{}}.$$

This has several advantages over direct classification:

* the decision is interpretable;
* the budget can change without retraining;
* the optimizer guarantees the budget;
* you can inspect the estimated gain per extra byte for every group.

For interactions, predict low-rank group factors:

$$u_g(q) = r_\theta(z_q, e_g) \in \mathbb{R}^r,$$

and approximate the interaction between groups $g$ and $h$ as

$$\hat{I}_{gh}(q) = u_g(q)^\top u_h(q).$$

Start with $r=4$ or 8, and restrict interactions to nearby groups or a sparse learned graph. A full $G^2$ interaction matrix becomes expensive once you move to sub-block granularity.

### 4.4 Budgeted optimizer

The independent baseline is a query-conditioned multiple-choice knapsack problem:

$$\min_{x} \sum_{g=1}^G \sum_{b \in B} \hat{d}_{g,b}(q)x_{g,b}$$

subject to

$$\sum_b x_{g,b} = 1, \quad \sum_{g,b} c_{g,b} x_{g,b} \leq B, \quad x_{g,b} \in \{0,1\}.$$

This follows the general MCKP structure used by mixed-precision methods, including MixQuant.

With interactions:

$$\min_{x} \sum_{g,b} \hat{d}_{g,b}(q) x_{g,b} + \sum_{g<h} \hat{I}_{gh}(q, x_g, x_h)$$

under the same budget. This becomes a quadratic knapsack or MIQP/MILP-style problem, similar in spirit to IMPQ’s interaction-aware formulation.

For runtime:

* use exact dynamic programming for the independent MCKP;
* use beam search or a precomputed profile library for the interaction-aware version;
* reserve MILP for offline oracle generation and evaluation.

The cost $c_{g,b}$ should not merely equal parameter count times bit-width. Use measured cost:
$c_{g,b} = \alpha \text{ resident bytes} + \beta \text{ H2D bytes} + \gamma \text{ latency} + \eta \text{ kernel switching}.$

Otherwise, the optimizer may produce profiles that look efficient by average bits but are slower in practice.

### 4.5 Causal online guard

After running group $g$, summarize its output:

$$\phi(h_g) = [\text{RMS}(h_g), \text{kurt}(h_g), \text{outlier\_ratio}(h_g), \text{attention\_entropy}(h_g), \dots].$$

Maintain a causal state:

$$s_g = \text{GRU}(s_{g-1}, [e_g, b_g, B_{\text{remaining}}, \phi(h_g)]).$$

The guard predicts whether selected future groups should be promoted. It must not use $h_g$ to retroactively choose the precision of group $g$.

This is where your RNN-like memory idea fits naturally. It carries:

* previous precision decisions;
* accumulated risk;
* remaining budget;
* observed hidden-state behavior.

A linear-attention controller is unlikely to be the best first choice. The sequence length is only the number of groups or modules, usually 8–100, so the asymptotic advantage of linear attention is negligible. A GRU, small state-space model, or tiny causal Transformer is simpler to train and audit. Linear attention becomes more justified if the controller processes hundreds of token-by-module observations.

The recommended comparison is:

* query-only MLP;
* layer-embedding scorer plus MCKP;
* causal GRU;
* causal attention or linear-attention controller.

Do not make the most complex controller the only model.

## 5. How one block affects later blocks

For a simplified residual block,

$$h_{g+1} = h_g + F_g^{(b_g)}(h_g).$$

Let $h_g^\star$ be the high-precision state and $e_g = h_g - h_g^\star$ the accumulated error. Locally,

$$e_{g+1} \approx J_g e_g + \eta_g(b_g, h_g),$$

where:

* $J_g$ propagates prior errors;
* $\eta_g$ is the new error introduced by quantizing group $g$.

Therefore, assigning four bits to an early group changes the input distribution seen by all later groups. That can change:

* activation scales;
* outlier locations;
* attention logits;
* MLP gating;
* the sensitivity of later quantized weights.

Residual connections and normalization may damp some errors, but they do not make the groups independent. MixQuant’s upstream-context experiments and IMPQ’s interaction modeling provide direct empirical support for this conclusion.

There are three progressively stronger models of this dependence:

| Model | Assumption | Use |
| --- | --- | --- |
| **Independent MCKP** | Each group has additive value | Mandatory baseline |
| **Mean-field/conditional scoring** | Average over sampled upstream profiles | Strong offline method |
| **Sequential causal controller** | Condition on actual previous choices and hidden signals | Most adaptive, highest overhead |

A valuable research result would be determining how much the third method improves over the second after counting its real overhead.

**Signal availability must remain causal**

| Signal | When it becomes available | What it may control |
| --- | --- | --- |
| **Query text and structure** | Before inference | Entire initial profile |
| **Hidden state after a fixed probe prefix** | After the prefix | Remaining groups |
| **Output of group $g$** | After group $g$ | Groups $g+1, \dots, G$ |
| **Final token logits** | After a complete forward pass | Next decoding step |
| **Generated-token entropy** | During decoding | Later tokens, not the current completed token |

Using a later hidden state to choose an earlier precision without explicitly paying for a probe or duplicate computation is information leakage.

## 6. Profile clustering and your two-vector example

Do not cluster prompts only by semantic embedding. Cluster by precision behavior.

For every calibration query, obtain one or more near-optimal profiles. Define an equivalence set:

$$E_\epsilon(q) = \{p : D(q,p) \leq \epsilon \land C(p) \leq C^\star(q) + \delta\}.$$

This accounts for the fact that several profiles may be essentially equally good.

Define a distance between profiles:

$$d(p, p') = \alpha \sum_g \omega_g \vert{}b_g - b_g'\vert{} + \beta \vert{}C(p) - C(p')\vert{} + \gamma \mathbb{E}_q [JS(P_p(\cdot \mid q), P_{p'}(\cdot \mid q))] + \eta d_{\text{layout}}(p, p').$$

The terms represent:

* sensitivity-weighted bit differences;
* hardware-cost difference;
* behavioral output difference;
* kernel/layout compatibility.

Use k-medoids, not k-means, because every medoid is an executable discrete profile.

Profiles $[4,4,8,4]$ and $[4,8,8,4]$ can share a class when:

* both preserve quality for nearly the same queries;
* their cost difference is negligible;
* the second block’s precision rarely changes the output;
* they can share the same compiled execution path.

They should be different classes when that second block is interaction-critical for a subset of queries.

Train the router with set-valued or soft targets rather than arbitrarily selecting one profile:

$$L_{\text{profile}} = -\log \sum_{p \in E_\epsilon(q)} \pi_\theta(p \mid q, B).$$

This is better than one-hot classification because it does not punish the controller for selecting a different but equivalent profile.

## 7. Generating the oracle training data

The controller needs labels that reflect actual quantization behavior.

**Step 1: Reference execution**
For each query, run the highest-precision reference and save:

* option logits for multiple-choice tasks;
* generated answer;
* teacher-forced token distributions;
* selected hidden-state summaries;
* task correctness.

**Step 2: Candidate profile generation**
Exhaustive evaluation is infeasible at fine granularity. Use a combination of:

* static sensitivity profiles;
* MCKP profiles at several budgets;
* random structured profiles;
* conditional greedy search;
* beam search through layer groups;
* uncertainty-driven active sampling.

With eight binary layer groups, all 256 profiles are technically enumerable for a small pilot. At larger granularity, use beam search with width 4–16.

**Step 3: Quality targets**
For multiple choice:

$$D_{\text{MC}}(q, p) = KL(P_{\text{high}}^{\text{options}} \parallel P_p^{\text{options}}) + \kappa \mathbf{1}[\text{argmax} P_{\text{high}} \neq \text{argmax} P_p].$$

For generative reasoning:

$$D_{\text{gen}}(q, p) = \frac{1}{T} \sum_{t=1}^T KL(P_{\text{high},t} \parallel P_{p,t}) + \kappa \mathbf{1}[\text{final answer changes}].$$

Teacher-forced KL provides a dense signal. Exact correctness remains the final metric.

**Step 4: Conditional marginal labels**
Sample an upstream profile context $c$. Measure the benefit of promoting group $g$:

$$\Delta_g(q, c) = D(q, c) - D(q, c[g \leftarrow \text{high}]).$$

This is the actual query- and context-dependent marginal value of extra precision. It is a better target than isolated layer MSE.

**Step 5: Router losses**
A practical combination is:

$$L = L_{\text{distortion}} + \lambda_1 L_{\text{ranking}} + \lambda_2 L_{\text{profile}} + \lambda_3 L_{\text{risk}}.$$

Use:

* Huber loss for predicted degradation;
* pairwise ranking loss for competing profiles;
* set-valued profile loss;
* quantile loss for a conservative upper-bound prediction.

The risk head should predict an upper quantile of degradation. At runtime, select the cheapest profile whose estimated upper bound is below $\epsilon$:

$$p^\star = \arg\min_p C(p) \text{ subject to } D_{1-\delta}(q,p) \leq \epsilon.$$

This directly controls the under-precision rate.

## 8. Observing how quantization propagates

Build an instrumentation layer before training the controller.

**Influence matrix**
For each query and source group $i$, switch only $i$ between low and high precision while holding the preceding context fixed. Measure the effect at every later group $j$:

$$A_{i \to j}(q) = D_h(h_j^{i=\text{low}}, h_j^{i=\text{high}}).$$

Useful $D_h$ choices are:

* normalized MSE;
* cosine distance;
* centered-kernel alignment;
* projected representation distance;
* final-logit KL;
* answer-flip indicator.

Plot A as an upper-triangular heatmap. Aggregate it by:

* dataset;
* task family;
* query difficulty;
* profile class;
* correct versus incorrect low-bit inference.

**Pairwise interaction**
For two groups:

$$I_{ij}(q) = L_{ij}(q) - L_i(q) - L_j(q) + L_0(q),$$

where $L_{ij}$ is the loss when both are low precision, $L_i$ and $L_j$ when only one is low, and $L_0$ when both are high.

* $I_{ij} > 0$: errors reinforce one another;
* $I_{ij} < 0$: one perturbation partially masks the other;
* $I_{ij} \approx 0$: additive approximation is reasonable.

**Per-query trace schema**
Log at minimum:

* query_id
* dataset / task / difficulty metadata
* input length and structural features
* query latent
* budget
* initial profile
* final profile after promotions
* per-group action probabilities
* remaining budget at each group
* hidden-state summary at each boundary
* predicted degradation
* actual KL / task correctness
* effective bits
* GPU resident bytes
* host-to-device bytes
* prefetch stalls
* kernel switches
* latency
* under-precision flag
* over-precision estimate

The two most useful deployment metrics will be:

* **under-precision:** the selected profile violates the quality threshold;
* **over-precision:** a cheaper profile would also have passed.

## 9. Dataset recommendation

Use datasets for complementary roles rather than simply combining everything.

| Dataset | Role |
| --- | --- |
| **MMLU-Pro** | Broad task and domain diversity; multiple-choice logits make intervention evaluation relatively cheap |
| **MATH training set / MATH-500 evaluation** | Structured reasoning, exact answers, and difficulty-level metadata |
| **GPQA Diamond** | Hard held-out scientific reasoning and out-of-distribution stress testing |
| **LiveCodeBench** | Optional late-stage code evaluation with executable correctness |
| **A generic language-modeling corpus** | Calibration of hidden/output distortion independently of benchmark answers |

MMLU-Pro was designed to be more reasoning-intensive and more stable to prompt variation than the original MMLU, making it preferable for your primary broad-domain study. GPQA contains a small set of very difficult expert-written science questions, so it is better reserved for evaluation than router training. MATH provides 12,500 competition problems with detailed solutions, while MATH-500 is a convenient held-out subset. LiveCodeBench is useful for a later code-generalization experiment because it continually sources newer contest problems and supports execution-based evaluation.

A good first split is:

* router training/calibration: non-test MMLU-Pro and MATH samples;
* validation: held-out domains and MATH levels;
* final evaluation: untouched MMLU-Pro test, MATH-500, and GPQA;
* optional OOD: LiveCodeBench.

Dataset category and published difficulty labels should be features or analysis axes, not the ground-truth routing target.

## 10. Minimum publishable experiment

Keep the first paper deliberately constrained.

| Include now | Defer |
| --- | --- |
| Weight-only quantization | Activation and KV-cache quantization |
| Bits {4,8} | 2-, 3-, 6-bit kernels |
| Eight contiguous layer groups | Per-head or per-channel routing |
| Single-query execution | Dynamic batching |
| Query-conditioned distortion prediction | Full token-level adaptation |
| MCKP plus one interaction-aware method | Large custom serving engine |
| Query-only and probe variants | Linear-attention controller |
| Real latency lookup table | Fully optimized custom CUDA kernels |

Required baselines:

* uniform four-bit;
* uniform eight-bit;
* static layerwise mixed precision;
* static interaction-aware allocation;
* QAQ-style direct MLP/profile classifier;
* query-conditioned scorer plus MCKP;
* causal scorer/GRU with online feedback;
* offline oracle.

Required ablations:

* task label only;
* query length only;
* structural features;
* embedding only;
* probe hidden state only;
* all features;
* independent versus mean-field versus pairwise interactions;
* query-only versus online feedback;
* raw profile prediction versus profile medoids;
* bit-count cost versus measured hardware cost;
* different group sizes;
* different profile-library sizes.

Evaluation should report paired quality–cost Pareto curves rather than a single operating point. Use paired bootstrap intervals over queries, because every method is tested on the same query set.

A sensible falsification-first sequence is:

1. **Signal test:** determine whether low-bit failure is predictable from query features.
2. **Interaction test:** determine whether conditional or pairwise scores materially outperform isolated sensitivity.
3. **Oracle test:** estimate the maximum possible gain of per-query profiles.
4. **Amortization test:** determine how much of the oracle gain the learned router recovers.
5. **Systems test:** determine whether the gain survives controller, transfer, and kernel overhead.

If the oracle itself barely improves over one static profile, stop before building a sophisticated router.

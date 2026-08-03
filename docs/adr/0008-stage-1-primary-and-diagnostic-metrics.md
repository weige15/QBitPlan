# ADR-0008: Stage-1 primary and diagnostic metrics

- Status: ACCEPTED
- Date: 2026-08-03
- Decision owners: project maintainers

## Context

The originating decision ticket is [Select the primary quality metric and
diagnostic metrics](https://github.com/weige15/QBitPlan/issues/9). ADR-0006
selects MMLU-Pro and MATH/MATH-500 for Stage-1 final paired
evaluation. ADR-0007 selects the unquantized BF16 execution as the
high-precision reference while keeping it distinct from external ground truth.

The ticket's predeclared rule requires task-appropriate external correctness
where answers are judged, permits perplexity only for a declared unlabeled
language-modeling task, and keeps model-relative measures diagnostic.

## Decision

### Primary quality metric

Task correctness is the primary quality metric for every selected Stage-1
dataset:

- MMLU-Pro uses exact-option correctness against its external answer key.
- MATH-500 uses answer-equivalence correctness against the external target.

No unlabeled language-modeling task is in the accepted Stage-1 suite, so
perplexity is not a Stage-1 primary metric. Agreement with the BF16 reference
is not task correctness.

### Equal-weight macro-average

Let (D = \{\text{MMLU-Pro}, \text{MATH-500}\}). For a
profile (p), let (c_{d,q}(p) \in \{0,1\}) be the correctness result for
query (q) in dataset (d), and let (n_d) be that dataset's final query
count. First compute one accuracy per dataset:

\[
A_d(p) = \frac{1}{n_d} \sum_{q \in d} c_{d,q}(p).
\]

The equal-weight macro-average is:

\[
A_{\mathrm{macro}}(p) = \frac{1}{2} \sum_{d \in D} A_d(p).
\]

Each dataset therefore contributes one-half of the summary, regardless of
its number of queries. The per-dataset accuracies remain the primary reported
results; the macro-average is a cross-dataset summary and must not replace
them. A pooled query-level accuracy is not the Stage-1 primary aggregate
because it would give MMLU-Pro greater weight solely due to its sample count.

For reference-relative reporting, compute the paired difference for each
dataset, (\Delta_d(p) = A_d(p) - A_d(\mathrm{BF16})), and summarize it with
the same equal-weight rule:

\[
\Delta_{\mathrm{macro}}(p) = \frac{1}{2} \sum_{d \in D} \Delta_d(p).
\]

### Diagnostic metrics and paired reporting

Quantized and BF16 runs use the same immutable final query manifests. For each
paired query, report:

- reference-relative output-distribution divergence, using the declared
  per-position logits or log-probabilities; KL divergence is diagnostic, not a
  quality replacement;
- answer-flip indicators relative to the BF16 reference, also diagnostic and
  distinct from external task correctness; and
- hidden-state distances only if a later protocol justifies the layer,
  position, representation, and aggregation. They are not required in the
  initial Stage-1 metric set.

Report diagnostic and correctness values per dataset, together with the
paired per-query quantized-versus-BF16 differences. Any cross-dataset
diagnostic summary must use the same equal-weight dataset aggregation and must
retain the per-dataset values.

## Rule application

- **Externally judged answers — pass.** All accepted Stage-1 tasks expose an
  external correctness target, so task correctness is primary.
- **Unlabeled language modeling — not applicable.** WikiText-103 is excluded
  by ADR-0006, so the conditional perplexity branch is unused.
- **Reference separation — pass.** BF16-relative logits, log-probabilities,
  answer flips, and any hidden-state distances remain diagnostics; correctness
  is judged against external targets.
- **Paired reporting — pass.** All methods use the same immutable final query
  manifests, and differences are computed within query pairs before any
  dataset-level or macro aggregation.

## Source facts, inference, and project preference

**Source facts.** `SCIENTIFIC_STANDARDS.md` requires a task-appropriate primary
metric, makes task correctness primary for externally judged answers, allows
perplexity for a declared unlabeled language-modeling task, keeps KL and
hidden-state measures diagnostic, and requires paired query sets. ADR-0006
records MMLU-Pro and MATH/MATH-500 as the accepted Stage-1 datasets and their
external targets. ADR-0007 records the BF16 reference and its separation from
ground truth.

**Inference.** Because the two selected datasets have unequal query counts, an
equal-weight macro-average with denominator two is the direct aggregation that
gives each accepted dataset equal influence while preserving paired per-query
comparisons.

**Project preference.** The project prefers per-dataset correctness as the
decision-facing result, with the equal-weight macro-average as a compact
cross-dataset summary. Output-distribution diagnostics are required; hidden
states are optional to keep Stage 1 bounded.

## Consequences and retained uncertainty

- A profile can have a favorable macro-average while underperforming on one
  dataset; per-dataset results must therefore accompany every macro value.
- The degradation threshold and epsilon-selection rule remain governed by
  [Define degradation and the epsilon-selection rule](https://github.com/weige15/QBitPlan/issues/13).
- Exact prompt formatting, answer normalization, valid-position selection for
  output-distribution diagnostics, and any confidence-interval procedure must
  be declared by the downstream evaluation protocol before evaluation.
- No decision here establishes generalization beyond the pinned model,
  datasets, manifests, and execution protocol.

## Related records

- [ADR-0006: Stage-1 task suite and immutable query-ID splits](0006-stage-1-task-suite-and-query-splits.md)
- [ADR-0007: Stage-1 high-precision reference and ground-truth role](0007-stage-1-high-precision-reference.md)
- [Define degradation and the epsilon-selection rule](https://github.com/weige15/QBitPlan/issues/13)

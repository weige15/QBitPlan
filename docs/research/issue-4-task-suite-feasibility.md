# Issue 4: Task-suite datasets, revisions, and immutable split feasibility

Research date: 2026-08-03  
Ticket: [Verify task-suite datasets, revisions, and immutable split feasibility](https://github.com/weige15/QBitPlan/issues/4)

## Scope and rule

This note checks only the candidates named in the initial research brief:
MMLU-Pro, MATH/MATH-500, GPQA Diamond, LiveCodeBench, and one concrete generic
language-modeling candidate, WikiText-103. It applies the ticket's rule
exactly: retain only candidates with a citable primary source, a stable
revision or snapshot, externally defined ground truth, a reproducible query-ID
manifest, and a feasible separation of calibration/training, validation, and
final-evaluation queries.

This is feasibility evidence, not a task-suite selection. The project suite,
metric protocol, and allowed cross-dataset split policy remain OPEN.

## Source facts

### Comparison matrix

| Candidate and pinned source | Coverage and task format | Correctness / ground truth | Revision and query IDs | Split feasibility and leakage risk | Rule result |
| --- | --- | --- | --- | --- | --- |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be) at Hub revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`; [official repository](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/f418b116db00b065c2aea046518d8fcf74d39872) | 12,032 test questions and 70 validation questions in the pinned Hub card; 14 broad academic/domain categories; ten-option multiple choice. | The pinned data exposes `answer` and `answer_index`; exact-option accuracy is externally keyed. The authors describe the benchmark and its reasoning-focused curation in the [paper](https://arxiv.org/abs/2406.01574). | `question_id` is an explicit integer field. The Hub commit is immutable for reproducibility, but the authors document later answer/format corrections, so an unpinned `main`/latest load is not reproducible. | The pinned artifact has validation and test but no calibration/training split. It can be a final-only component of a cross-dataset protocol, but is not a standalone three-way split. Public benchmark text and answer keys create ordinary pretraining and evaluation leakage risk. | **Conditional only; not a standalone pass.** |
| [MATH](https://github.com/hendrycks/math/tree/985bdc1696e88e8643f081a0ff4719da39f2ae2a) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) | The original MATH paper defines 12,500 competition problems: 7,500 train and 5,000 test, across seven subjects and five difficulty levels. MATH-500 is a 500-problem held-out subset represented by the pinned `test` artifact. Format is open-ended mathematical problem → solution/answer. | Each problem has a step-by-step solution and final answer; the original evaluator extracts the boxed answer and checks mathematical equivalence, with accuracy as the primary metric. | MATH-500 rows expose `unique_id` values such as source split/path identifiers. The MATH-500 Hub revision and the [OpenAI PRM800K pinned split](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163) are immutable references; PRM800K documents its nonstandard 4,500-train-plus-500-held-out split. | Feasible: use source training problems for calibration/training, a deterministic held-out subset for validation, and the pinned MATH-500 IDs for final evaluation. The main leakage risk is that MATH and derived MATH-500 are widely public and may occur in model training or fine-tuning; the final manifest must therefore be fixed before any profile work. | **Pass for a declared split protocol; no selection made.** |
| [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa/tree/633f5ee89ab8ad4522a9f850766b73f62147ffdd) at gated Hub revision `633f5ee89ab8ad4522a9f850766b73f62147ffdd`; [official repository](https://github.com/idavidrein/gpqa/tree/56686c06f5e19865c153de0fdb11be3890014df7) | The primary paper defines 198 Diamond questions, the highest-quality subset of a 546-question extended set, covering biology, physics, and chemistry. Format is four-option expert-written multiple choice. | Questions were expert-written and expert-validated; Diamond requires both expert validators to agree under the paper's stated rule and a majority of non-experts to be incorrect. The correct option is external ground truth. | The gated dataset has a pinned revision and a named `gpqa_diamond.csv` artifact. The public card/repository do not document a source-defined query-ID field; a manifest therefore needs a deterministic derived key from the pinned canonical row, with no examples copied into the note because the source explicitly requests non-disclosure. | Diamond is an evaluation subset, not a three-way train/validation/final resource. It is feasible only as a final-only component when calibration/training and validation come from other pinned data; access gating and public benchmark use reduce but do not eliminate leakage risk. | **Conditional only; not a standalone pass.** |
| [LiveCodeBench code-generation-lite](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505); [official repository](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24) | Competitive-programming code generation from LeetCode, AtCoder, and CodeForces. The pinned card defines `question_id`, platform, contest ID/date, difficulty, prompt, starter code, public tests, and private tests. It provides cumulative `release_v1`–`release_v6` configurations and incremental `v1`–`v6` files. | Correctness is externally defined by hidden/public tests and the official execution-based evaluator; code generation is scored by Pass@1 when a program passes all tests. | Release tags plus the Hub commit pin both the snapshot and the temporal boundary; `question_id` is an explicit stable key. The official paper uses release dates to evaluate post-cutoff problems and documents contamination detection. | Feasible by a declared temporal split, for example earlier incremental releases for calibration/training, a later release for validation, and a still later release for final evaluation. This is an inference from the documented release layout, not an official train/validation recommendation. Public prompts and contest solutions remain leakage risks; source release dates provide a measurable mitigation, not a guarantee. | **Conditional pass by inference; requires an accepted temporal split.** |
| [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), a concrete generic language-modeling candidate | English text from verified Good and Featured Wikipedia articles; the pinned card defines `text` rows and train/validation/test splits with 1,801,350 / 3,760 / 4,358 rows for `wikitext-103-raw-v1`. | There are no task labels. The externally supplied continuation text is the target and perplexity is the appropriate language-modeling correctness measure under the project's scientific standards; this is not an answer-accuracy benchmark. | The Hub revision is immutable, but rows have no source-defined query IDs. A reproducible manifest can be made from `(config, split, row index, content hash)` after pinning the artifact. | Feasible: the source provides all three splits, with train for calibration/training, validation for threshold/selection checks, and test for final evaluation. Leakage risk is high because Wikipedia is public and common in pretraining; this is a calibration/diagnostic candidate rather than evidence of benchmark generalization. | **Pass for a declared LM protocol; no selection made.** |

## Candidate-specific observations

### MMLU-Pro

**Source facts.** The official repository describes more than 12,000 curated
questions across 14 domains, with ten answer options and evaluation scripts.
The pinned Hub metadata provides explicit `question_id`, `answer`,
`answer_index`, `category`, and `src` fields, with 12,032 test rows and 70
validation rows. The Hub card records historical answer and formatting
corrections, including corrections keyed by question ID.

**Inference.** Pinning the Hub commit is necessary but not sufficient: the
project must also record the exact split and prompt/evaluation code. The
available artifact supports final evaluation and a small validation set, but
not standalone calibration/training.

**Project preference.** Using MMLU-Pro as the broad-domain final component is
consistent with the initial brief, but this note does not accept that
preference as a selection.

### MATH and MATH-500

**Source facts.** The MATH paper defines the original train/test counts,
subjects, difficulty levels, step-by-step solutions, final boxed answers, and
equivalence-based grading. OpenAI's PRM800K repository explicitly documents a
nonstandard split that combines 4,500 MATH test problems with the 7,500
training problems and holds out 500 test problems selected uniformly from the
remaining test set. The MATH-500 Hub artifact exposes `problem`, `solution`,
`answer`, `subject`, `level`, and `unique_id`.

**Inference.** This is the clearest named candidate for a reproducible
three-way protocol, provided the exact MATH source artifact, validation rule,
and MATH-500 manifest are all pinned. The PRM800K split is a source-owned
precedent, not an automatic requirement for QBitPlan.

### GPQA Diamond

**Source facts.** The paper defines Diamond as 198 questions selected by
stricter expert-agreement and non-expert-difficulty criteria than the main
set. The official Hub is gated, names `gpqa_diamond.csv`, records a revision,
and asks users not to publish examples to reduce leakage into training
corpora.

**Inference.** A pinned CSV plus a deterministic canonical-row hash can make a
query manifest reproducible, but that ID scheme is not source-defined and
must be documented if the candidate is used. Diamond should therefore remain
final-only unless a later accepted protocol explicitly allows cross-dataset
calibration and validation.

### LiveCodeBench

**Source facts.** The paper and repository define code-generation inputs and
execution-based correctness, and the current dataset card exposes explicit
question IDs, contest dates, hidden/private tests, and versioned releases.
The paper reports that release-date windows can expose likely contamination
and that generated programs must pass all tests.

**Inference.** Incremental release IDs make a non-overlapping temporal
manifest feasible, but this is not the same as the benchmark's documented
evaluation-only use. A future protocol must choose the release boundaries
before calibration and must not use a cumulative release accidentally.

### WikiText-103

**Source facts.** The pinned card provides train, validation, and test splits
and identifies the corpus as Wikipedia text. The source paper describes the
dataset as a language-modeling corpus designed to preserve long-range
dependencies.

**Inference.** A content-hash-plus-row-index ID is reproducible for a pinned
artifact, but it is a project-derived manifest rather than a source field.
Perplexity can serve as the declared primary quality metric under
`SCIENTIFIC_STANDARDS.md`; it must not be silently compared as answer accuracy
with the other candidates.

## Gate summary

- **Citable primary source:** available for all five candidates through the
  source paper and/or source-owned repository/card.
- **Stable revision/snapshot:** available for all five when the listed Hub or
  Git commit is recorded; floating `main`, `latest`, and cumulative LCB
  releases are not sufficient.
- **Externally defined ground truth:** clear for MMLU-Pro, MATH/MATH-500,
  GPQA Diamond, and LiveCodeBench; for WikiText it is the held-out text
  continuation and requires a predeclared perplexity protocol.
- **Reproducible query-ID manifest:** source-defined for MMLU-Pro, MATH-500,
  and LiveCodeBench; derived-but-feasible for GPQA Diamond and WikiText if the
  canonical row serialization and hash algorithm are recorded.
- **Immutable phase separation:** directly supported by MATH/MATH-500 and
  WikiText; supported by inference for LCB temporal releases; not supported
  internally by the MMLU-Pro or GPQA Diamond evaluation artifacts.

## Uncertainty and handoff

- No project suite is selected here. The next ticket must decide whether
  final-only benchmarks may rely on calibration/validation queries from other
  pinned datasets.
- No exact calibration/validation manifest is created by this research note;
  creating one would be protocol design and implementation work.
- MMLU-Pro's maintained corrections mean that published benchmark numbers
  without a dataset commit are not reproducible.
- GPQA access is gated and its source requests that examples not be
  republished; the final artifact must preserve that restriction.
- LCB's release versions are cumulative in `release_v*` form, so a future
  temporal manifest must use explicit incremental versions or de-duplicate by
  `question_id`.
- WikiText is a viable generic language-modeling source, but its public
  Wikipedia content makes contamination risk materially higher than the
  expert-authored evaluation candidates.

## Conclusion

The evidence supports MATH/MATH-500 and WikiText-103 as candidates with
source-provided three-way split structure. LiveCodeBench has a reproducible
temporal-snapshot path, while MMLU-Pro and GPQA Diamond are reproducible
final-evaluation artifacts but lack standalone calibration/training splits.
Whether those final-only candidates may participate in a cross-dataset suite
is an unresolved project decision; this note does not make it.

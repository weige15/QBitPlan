# MATH source manifest preparation

`scripts/build_math_manifest.py` is a standalone data-preparation command for
issue 25. It is not part of the scientific CLI and does not load the combined
MATH cache.

The accepted source revision is `985bdc1696e88e8643f081a0ff4719da39f2ae2a`.
The Git checkout contains loaders and evaluation code, but not the raw 12,500
problem JSON files. The materialized pinned inputs are:

The pinned raw tree is materialized at `data/pinned/math-raw/MATH/` from the
content-addressed archive `https://gitee.com/hf-datasets/competition_math/raw/main/data/MATH.zip`, with archive SHA-256
`d9b88da85e6ffa3e1057ae675238d6e192574243bdc45ca7d00a1339fc4d0874`. The
pinned MATH-500 `test.jsonl` has SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.

```text
<source-root>/
  train/**/*.json
  test/**/*.json
```

Each source JSON object must contain the original MATH fields `problem`,
`solution`, string `level` (for example, `Level 5`), and string `type`. The
source-relative POSIX path (for example `test/algebra/807.json`) is the source ID.

Run:

```bash
python3 scripts/build_math_manifest.py \
  --source-root data/pinned/math-raw/MATH \
  --math500-root data/pinned/math-500 \
  --manifest-output data/manifests/math-source.json \
  --smoke-plan-output data/manifests/math-smoke-plan.json
```

The command fails closed unless it finds exactly 7,500 training records and
5,000 source-test records, reads exactly 500 unique MATH-500 `unique_id`
values, confirms every final ID is in source test, and derives exactly 4,500
validation IDs by set subtraction. It hashes every source record with compact
canonical UTF-8 JSON and SHA-256, writes the manifest once, and selects the
lexicographically first training and validation IDs for the source-bound
smoke plan. Both outputs are write-once; remove or choose a new output path
only after deliberately reviewing an existing artifact.

The existing combined 12,500-row Parquet cache is intentionally not accepted:
its provenance does not verify the original split or source-relative IDs.

## 256-profile inventory

Issue #26 reuses the same `ExperimentPlan` → `ArtifactBundle` seam with
`mode` set to `functional-quality` and `profiles` set, explicitly and before
execution, to the canonical IDs `00000000` through `11111111` in lexicographic
order. The scientific CLI accepts this mode:

```bash
qbitplan stage1 run \
  --plan /path/to/functional-plan.json \
  --mode functional-quality \
  --gpu-uuid GPU-UUID
```

The plan must still provide the accepted model, tokenizer, software, runtime,
source-manifest, and permitted query fields; missing defaults are rejected.
The resulting write-once bundle contains `profile-inventory.json` with all
256 attempts, sorted `P_exec`, exclusion reason codes, evidence classes, and
lineage references to `profile-outcomes.ndjson`. Transform or complete-forward
failure excludes only that profile and is retained as an explicit invalid
outcome. Issue #27 additionally writes paired functional-quality outcomes and
separately labeled teacher-forced diagnostic records for BF16 and every profile.
Those records provide external correctness and BF16-relative degradation only;
they do not produce cost, latency, memory, systems-benefit, or generalization
claims.

## Final MMLU-Pro quality campaign

The accepted final-only MMLU-Pro artifact is Hub revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`, split `test`, with 12,032
source-defined `question_id` records. Build its manifest from the exact pinned
parquet; the command verifies the parquet SHA-256 and writes the manifest once:

```bash
python scripts/build_mmlu_manifest.py \
  --download-dir /path/to/mmlu-cache \
  --manifest-output data/manifests/mmlu-pro-test.json
```

The source-faithful option list is retained per question (the pinned source has
some reviewed rows with fewer than ten options); options are never padded or
fabricated. The final plan embeds the canonical prompts and records, and pins
the tokenizer file hashes declared by the execution contract:

```bash
python scripts/build_mmlu_quality_plan.py \
  --source-parquet /path/to/mmlu-cache/data/test-00000-of-00001.parquet \
  --manifest data/manifests/mmlu-pro-test.json \
  --profile-inventory /path/to/profile-inventory.json \
  --artifact-root /path/to/artifacts \
  --attempt-id mmlu-pro-final-quality-0001 \
  --gpu-uuid GPU-UUID \
  --output /path/to/mmlu-pro-final-quality-plan.json
```

Run the full paired quality campaign on one UUID-pinned RTX 3090:

```bash
qbitplan stage1 run \
  --plan /path/to/mmlu-pro-final-quality-plan.json \
  --mode functional-quality \
  --gpu-uuid GPU-UUID
```

MMLU-Pro is final-only cross-dataset evidence. Its outcomes are reported
separately from non-final MATH calibration and validation; the bundle claims
quality and diagnostics only, not cost or systems benefit.

## Direct-cost smoke frame

Issue #30 uses a validated plan declaring `mode` as `direct-cost` and the same
four smoke profiles. Run the fixed direct-cost frame on the selected UUID-pinned
RTX 3090 with:

```bash
qbitplan stage1 run \
  --plan /path/to/cost-smoke-plan.json \
  --mode direct-cost \
  --smoke \
  --gpu-uuid GPU-UUID
```

The run separates model loading and profile transformation from query
execution, performs five unprofiled warmups and ten synchronized measured
repetitions, and writes one separate trace pass. Its immutable bundle contains
`setup-observations.ndjson`, `cost-observations.ndjson`,
`trace-observations.ndjson`, and `cost-coverage.ndjson` alongside the plan and
run manifests. Only valid dimensions are directly measured; unavailable
dimensions carry explicit omission reasons.

## Issue #32 OOM diagnosis and direct-cost smoke

Issue #32 adds a direct hardware-cost smoke mode over the same explicit
`ExperimentPlan` and the same two permitted non-final MATH records. Run it with
a new immutable `attempt_id` and plan-declared artifact root:

```bash
qbitplan stage1 run \
  --plan /path/to/direct-cost-plan.json \
  --mode direct-cost \
  --smoke \
  --gpu-uuid GPU-UUID
```

The direct-cost bundle records setup/quantization observations separately from
query execution, five warmups, ten unprofiled measurements, one traced pass,
and per-dimension coverage. Setup, memory, latency, and valid trace
observations are directly measured; dimensions without accepted coverage are
`omitted/unavailable/<reason>`, never zero or a lookup fallback. The bundle is
non-evidentiary for task quality and systems benefit.

The real executor retains only one prepared profile at a time and releases it
before the next profile. Transform or forward failures remain immutable invalid
records and never substitute BF16, a nearby profile, or fake quantization.

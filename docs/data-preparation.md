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
outcome. This inventory does not produce quality, cost, latency, memory,
systems-benefit, or generalization claims.

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

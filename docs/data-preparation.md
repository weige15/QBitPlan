# MATH source manifest preparation

`scripts/build_math_manifest.py` is a standalone data-preparation command for
issue 25. It is not part of the scientific CLI and does not load the combined
MATH cache.

The pinned Git checkout of `hendrycks/math` contains loaders and evaluation
code, but not the raw 12,500 problem JSON files. Supply an explicit extracted
source tree with this layout:

```text
<source-root>/
  train/**/*.json
  test/**/*.json
```

Each source JSON object must contain the original MATH fields `problem`,
`solution`, integer `level`, and string `type`. The source-relative POSIX path
(for example `test/algebra/807.json`) is the source ID.

Run:

```bash
python3 scripts/build_math_manifest.py \
  --source-root /path/to/MATH \
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

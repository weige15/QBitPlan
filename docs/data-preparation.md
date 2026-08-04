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
`solution`, `level` (`"Level 1"` through `"Level 5"` or `"Level ?"`, or an integer 1 through 5 in an equivalent canonical export), and string `type`. The source-relative POSIX path
(for example `test/algebra/807.json`) is the source ID.

## Pinned inputs

The accepted issue-25 acquisition uses the Berkeley source archive snapshot and the pinned MATH-500 JSONL below. Verify the bytes before building the manifest:

```bash
mkdir -p data/pinned/math-500
curl -L --fail --retry 3 \
  'https://web.archive.org/web/20240101000000id_/https://people.eecs.berkeley.edu/~hendrycks/MATH.tar' \
  -o /tmp/MATH.tar
printf '0fbe4fad0df66942db6c221cdcc95b298cc7f4595a2f0f518360cce84e90d9ac  /tmp/MATH.tar\n' | sha256sum -c -
tar -xf /tmp/MATH.tar -C data/pinned
git clone https://github.com/hendrycks/math.git data/pinned/hendrycks-math
git -C data/pinned/hendrycks-math checkout --detach 985bdc1696e88e8643f081a0ff4719da39f2ae2a
curl -L --fail --retry 3 \
  'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl?download=true' \
  -o data/pinned/math-500/test.jsonl
printf '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132  data/pinned/math-500/test.jsonl\n' | sha256sum -c -
```

The raw source contains exactly 7,500 training and 5,000 test records. The provenance checkout used for the source revision is `hendrycks/math` at commit `985bdc1696e88e8643f081a0ff4719da39f2ae2a`; it is not a substitute for the raw JSON tree.
The builder records the downloaded Berkeley archive SHA-256 as `artifact_id`, a digest of only the extracted `train/**/*.json` and `test/**/*.json` bytes as `source_tree_artifact_id`, and the MATH-500 file SHA-256 separately. It records a separate `manifest_id` from the canonical manifest payload. The identities must not be conflated. The builder rejects a tree digest mismatch and validates the complete public plan before writing either output.


Run:

```bash
python3 scripts/build_math_manifest.py \
  --source-root /path/to/MATH \
  --math500-root data/pinned/math-500 \
  --source-archive /tmp/MATH.tar \
  --manifest-output data/manifests/math-source.json \
  --smoke-plan-output data/manifests/math-smoke-plan.json \
  --artifact-root /path/to/artifacts \
  --attempt-id attempt-0001 \
  --gpu-uuid GPU-UUID
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

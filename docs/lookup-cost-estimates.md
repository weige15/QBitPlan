# Lookup-cost estimate adapter

Issue #29 adds the offline `estimate-cost` path to the shared
`ExperimentPlan` → `ArtifactBundle` seam. It reads a declared JSON lookup
table and never loads a model or queries a GPU:

```bash
qbitplan stage1 estimate-cost \
  --plan <estimate-plan.json> \
  --lookup <lookup-table.json>
```

The estimate plan uses `qbitplan.stage1.experiment-plan.v1` with
`"mode": "estimate-cost"`, an explicit `artifact_root`, `attempt_id`, source
manifest, permitted queries, and canonical profile IDs. It has no GPU UUID or
model-execution fields.

The lookup table uses
`qbitplan.stage1.lookup-cost-table.v1` and declares `method`, the exact
evidence class `lookup-table estimated`, a `source_artifact`, a coverage
manifest, and exact `(query_id, profile_id)` entries. The coverage manifest
uses `qbitplan.stage1.lookup-cost-coverage.v1` and carries its SHA-256
canonical-payload `manifest_id`.

The adapter emits the complete six-dimension cost vector for every planned
query/profile pair. Exact entries are labeled `lookup-table estimated`.
Uncovered profiles, queries, dimensions, and missing entries are emitted as
`omitted/unavailable` with a reason; no interpolation, nearby-profile
substitution, average-bit conversion, or zero filling is performed.

The bundle contains `cost-estimates.ndjson`, `coverage.ndjson`,
`lookup-manifest.json`, the run manifest, and immutable bundle/index hashes.
Its claim scope is estimated cost comparison bounded to the declared table and
coverage. Lookup output cannot establish directly measured latency, memory,
transfer, kernel, or serving benefit.

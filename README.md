# QBitPlan

Query-conditioned causal bit planning for budgeted LLM inference.

## What this project is

QBitPlan is a research project about selecting executable mixed-precision
profiles for model inference. The repository is currently a documentation
and research-definition workspace; this initialization does not include an
implementation or experiment runner.

## Start here

Read these documents in order:

1. [Project context and terminology](CONTEXT.md)
2. [Scientific standards](SCIENTIFIC_STANDARDS.md)
3. [v0.1 research-boundary ADR](docs/adr/0001-v0.1-research-boundary.md)
4. [Initial research brief](docs/research/initial-research-brief.md)

Contributors and coding agents should also read [AGENTS.md](AGENTS.md) and the
guides under [docs/agents](docs/agents/).

## Repository map

- [AGENTS.md](AGENTS.md) — operational instructions for repository work
- [CONTEXT.md](CONTEXT.md) — canonical project terminology
- [SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md) — scientific and claims
  invariants
- [docs/adr](docs/adr/) — accepted or proposed decisions
- [docs/research](docs/research/) — research inputs and synthesis
- [docs/agents](docs/agents/) — issue and domain workflow guides
- [.github/pull_request_template.md](.github/pull_request_template.md) — pull
  request checklist

## Current state

The accepted Stage-1 execution contract now has its first implementation slice:
issue #25 exposes one `ExperimentPlan` → immutable `ArtifactBundle` seam for
the non-evidentiary smoke mode. The real CLI command is:

```text
qbitplan stage1 run --plan <smoke-plan> --mode smoke --gpu-uuid <uuid>
```

The plan must explicitly declare the accepted model, tokenizer, software,
hardware, decoder, deterministic controls, source-manifest-bound two permitted non-final MATH query
records, and artifact output root. The smoke bundle runs BF16, all-4, all-8,
and the recorded mixed profile, and reports executable-path evidence only; it
does not establish task quality, cost, systems benefit, or generalization.

The bundle directory contains the canonical phase manifest, run manifest,
artifact index, immutable bundle JSON, plan/run identities, source lineage,
strict per-profile/query NDJSON records,
and per-profile/query transform and forward statuses. Unsupported transforms,
incomplete forwards, context overflow, non-finite output, OOM, and other
runtime failures are recorded with explicit invalid statuses and reason codes;
no fallback profile is substituted.

The accepted Stage-1 contract and its execution seam are implemented for issues
#25 and #26. Build an explicit smoke plan with
[the data-preparation command](docs/data-preparation.md), then run
`qbitplan stage1 run --plan <smoke-plan> --mode smoke --gpu-uuid <uuid>`.
For the issue-26 profile inventory, use a validated functional-quality plan
with the complete canonical profile list and run
`qbitplan stage1 run --plan <functional-plan> --mode functional-quality --gpu-uuid <uuid>`.
The inventory is immutable executable-profile feasibility evidence only; it
does not establish task quality, cost, latency, memory, systems benefit, or
generalization.

The same seam now supports the typed functional-quality runner for the pinned
MATH and final-only MMLU-Pro orchestration. It writes paired quality outcomes,
diagnostics, and summaries while preserving profile-major preparation and
canonical query/profile artifact order. These artifacts are evidence records,
not a quality claim; a research quality result requires a real directly
measured GPU artifact.

The historical all-profile attempt from issue #32 recorded an empty `P_exec`
after OOM-class exclusions; it is not the current state wording for the
repository. A newer CPU-first smoke run completed BF16, all-4, all-8, and one
mixed profile for the two permitted non-final MATH queries and directly
measured absolute NVML resident bytes. Smoke evidence remains
non-evidentiary and does not establish the full 256-profile `P_exec`.

The local-only progress report and separated figure are in
[docs/progress/stage1-progress-20260805.md](docs/progress/stage1-progress-20260805.md).

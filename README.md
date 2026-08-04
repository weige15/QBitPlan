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
The bundle also contains paired functional-quality outcome and diagnostic artifacts
for BF16 plus every executable profile. These records provide pinned external
correctness, BF16-relative degradation, and separately labeled output diagnostics;
they do not establish cost, latency, memory, systems benefit, or generalization.

Issue #30 adds the fixed direct-cost smoke frame around the same real executor:

```bash
qbitplan stage1 run \
  --plan /path/to/cost-smoke-plan.json \
  --mode direct-cost \
  --smoke \
  --gpu-uuid GPU-UUID
```

The direct-cost bundle writes separate setup, cost-observation, trace, and
cost-coverage artifacts. Valid cost dimensions are directly measured; missing
NVML or trace coverage is recorded as `omitted/unavailable` and is never
replaced with zero or lookup evidence. The bundle remains bounded to the
pinned model, two-query smoke frame, four declared profiles, and selected GPU.


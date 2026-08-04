# QBitPlan

Query-conditioned causal bit planning for budgeted LLM inference.

## What this project is

QBitPlan is a research project about selecting executable mixed-precision
profiles for model inference. The first runtime slice is the explicit Stage-1
ExperimentPlan-to-ArtifactBundle smoke seam; it is executable-path evidence
only and does not establish quality or systems claims.

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

The accepted Stage-1 contract and its execution seam are implemented for issue
#25. Build an explicit smoke plan with
[the data-preparation command](docs/data-preparation.md), then run
`qbitplan stage1 run --plan <smoke-plan> --mode smoke --gpu-uuid <uuid>`.
Smoke output is immutable and non-evidentiary; see the issue and execution
contract for the required data and hardware boundary.

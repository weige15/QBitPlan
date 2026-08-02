# QBitPlan

Query-conditioned causal bit planning for budgeted LLM inference.

## Status

QBitPlan is in the research-definition phase. No implementation or
experimental result should currently be treated as established.

## Research objective

Given a query, a runtime resource budget, and a progressively quantized
Transformer, determine whether a lightweight controller can select a
hardware-executable mixed-precision profile that preserves reference
quality while accounting for upstream precision interactions and causally
available execution signals.

## Research questions

1. Predictability — can query features predict where precision is valuable?
2. Interaction — how much do upstream choices alter later sensitivity?
3. Feedback — do causal hidden-state signals improve future decisions enough
   to justify their overhead?
4. Execution — can per-query profiles be compressed into a small executable
   library without losing most of their benefit?

## Working hypothesis

Queries requiring similar total precision may require that precision in
different parts of the model. This precision shape may be predictable from
query structure and early execution signals.

## Initial v0.1 boundary

- Weight-only quantization
- Bit widths `{4, 8}`
- Eight contiguous layer groups
- Single-query execution
- Independent MCKP as a mandatory baseline
- One interaction-aware planning method
- Query-only and causally valid probe variants
- No custom serving engine or custom CUDA requirement

See `docs/adr/0001-v0.1-research-boundary.md`.

## Non-claims

QBitPlan does not assume:

- that layer precision decisions are independent;
- that average bit-width is a sufficient hardware-cost metric;
- that human-perceived question difficulty equals quantization difficulty;
- that simulated mixed precision implies real memory or latency savings.

## Repository map

- `AGENTS.md` — repository instructions for coding agents
- `CONTEXT.md` — project terminology
- `SCIENTIFIC_STANDARDS.md` — experiment and claims policy
- `docs/research/` — research inputs and verified research notes
- `docs/adr/` — accepted and proposed decisions
- `docs/agents/` — agent workflow configuration

## Current milestone

Resolve the decisions required to define a reproducible v0.1 oracle
experiment. Implementation begins only after that experimental contract is
accepted.

## License

Not yet selected.

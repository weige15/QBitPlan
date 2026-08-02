# QBitPlan repository instructions

## Before editing

Read, in order:

1. [README.md](README.md)
2. [CONTEXT.md](CONTEXT.md)
3. [SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md)
4. Relevant [ADRs](docs/adr/)
5. The originating GitHub issue and its comments

Use the [issue-tracker guide](docs/agents/issue-tracker.md) for GitHub work
and the [domain guide](docs/agents/domain.md) for terminology and ADRs.

## Source of truth

When documents conflict, report the conflict explicitly. Resolve it in this
order:

1. Accepted technical or research decision: relevant ADR.
2. Project terminology: [CONTEXT.md](CONTEXT.md).
3. Scientific and reporting invariants:
   [SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md).
4. Task scope: originating GitHub issue.
5. Research background: documents under `docs/research/`.
6. Public orientation: [README.md](README.md).

## Document ownership

- Keep shared terminology in [CONTEXT.md](CONTEXT.md).
- Keep scientific and reporting invariants in
  [SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md).
- Keep proposed, accepted, and OPEN technical or research decisions in the
  relevant ADR.
- Keep [README.md](README.md) as orientation, not specification.
- Treat the research brief body as immutable research input; metadata may
  identify its verification status, but its claims must not be presented as
  independently verified.

## Scope and implementation

- Work only from an originating issue or accepted specification.
- Keep changes within the requested scope and preserve unrelated user work.
- Do not choose OPEN dependencies, environments, hardware, datasets, or
  backends by convention.
- Do not add production infrastructure, code, dependencies, CI, configuration
  frameworks, or experiment implementations to a documentation-only task.
- Do not run costly experiments or publish externally without explicit scope.

## Verification and handoff

- Run the smallest relevant checks, then broader repository checks that exist.
- Check every relative Markdown link after documentation changes.
- Record exact commands, results, unavailable checks, remaining OPEN decisions,
  and known limitations in the handoff.

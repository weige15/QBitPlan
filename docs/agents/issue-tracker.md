# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues. Use the `gh` CLI
for issue operations; it infers the repository from the Git remote.

## Required workflow

- Identify the originating issue before implementation.
- Read its body and comments with `gh issue view <number> --comments`.
- Treat the issue as the task contract; record durable technical or research
  decisions in an ADR.
- Do not substitute an unverified research brief for an accepted issue or ADR.

## Common commands

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open`
- Comment: `gh issue comment <number> --body "..."`
- Edit labels: `gh issue edit <number> --add-label "..."`
- Close: `gh issue close <number> --comment "..."`

Pull requests are not a substitute for an originating issue. For repository
scope and document ownership, see [AGENTS.md](../../AGENTS.md).

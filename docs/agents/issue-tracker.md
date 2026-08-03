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

## Wayfinding operations

A Wayfinder map is a GitHub issue carrying the `wayfinder:map` label.

Every decision ticket:

- is a sub-issue of the map;
- carries exactly one `wayfinder:<type>` label;
- uses native GitHub blocking relationships;
- is claimed by assigning it before work begins;
- records its resolution in a closing comment;
- stores durable accepted decisions in an ADR.

### Create a map

```bash
gh issue create \
  --title "Wayfinder: <destination>" \
  --label "wayfinder:map" \
  --body-file /tmp/wayfinder-map.md

### Create a child ticket
gh issue create \
  --title "<decision title>" \
  --label "wayfinder:research" \
  --parent <MAP_NUMBER> \
  --body-file /tmp/decision-ticket.md

Replace the label with wayfinder:grilling, wayfinder:prototype, or
wayfinder:task as appropriate.

### Add a blocking relationship
gh issue edit <TICKET_NUMBER> \
  --add-blocked-by <BLOCKER_NUMBER>
### Claim a ticket
gh issue edit <TICKET_NUMBER> --add-assignee "@me"
### Inspect relationships
gh issue view <TICKET_NUMBER> \
  --json parent,subIssues,assignees,blockedBy,blocking
### Resolve a ticket

Post the resolution as a comment, close the ticket, and append a one-line
context pointer to the map's Decisions so far.


Current GitHub CLI supports parent/sub-issue relationships, native blocked-by relationships, self-assignment, and inspection of `parent`, `subIssues`, `blockedBy`, and `blocking` fields. :contentReference

Create these labels before invoking Wayfinder:

```text
wayfinder:map
wayfinder:research
wayfinder:grilling
wayfinder:prototype
wayfinder:task

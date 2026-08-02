# Domain Docs

## Before exploring, read these

- `CONTEXT.md` at the repo root
- `docs/adr/` — read ADRs that touch the area being explored

If these files do not exist, proceed silently. The `/domain-modeling` skill creates them lazily when concepts or decisions are resolved.

## File structure

This is a single-context repo:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Use the glossary's vocabulary

When naming domain concepts, use the terms defined in `CONTEXT.md`. If a needed concept is missing, note the gap for `/domain-modeling`.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding it.

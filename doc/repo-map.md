# Repository Map

## Repository Summary

QBitPlan is currently a documentation and research-definition workspace for
query-conditioned causal bit planning. The current main branch contains no
scientific runtime or experiment runner.

## Directory Structure

- `docs/adr/` contains accepted and proposed technical/research decisions.
- `docs/protocol/` contains the accepted Stage-1 protocol and handoff.
- `docs/research/` contains research inputs and feasibility notes.
- `docs/agents/` contains repository workflow and domain guidance.
- `scripts/` contains standalone repository/data-preparation commands.
- `tests/` contains focused tests for standalone commands.
- `data/pinned/` contains local pinned source checkouts requested for data
  preparation; these are not implementation modules.

## Main Source Files

The current main branch has no application package. The data-preparation
entry point is `scripts/build_math_manifest.py`.

## Existing Tests

Before this change, no test suite or test directory existed on main. The
focused manifest-builder tests are in `tests/test_build_math_manifest.py`.

## Build System

No build system or package metadata is present on main. The builder uses only
the Python standard library.

## Runtime or CLI Entry Points

The standalone command is invoked with `python3 scripts/build_math_manifest.py`.
It is deliberately separate from the future scientific CLI.

## Data and Assets

The requested local checkouts are `data/pinned/math` at MATH revision
`985bdc1696e88e8643f081a0ff4719da39f2ae2a` and `data/pinned/math-500` at
MATH-500 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. The MATH Git
checkout contains loader code but not the raw 12,500 problem JSON files. The
builder therefore requires an explicit raw source root with `train/` and
`test/` directories.

## Existing Documentation

Project orientation is in `README.md`; terminology and scientific invariants
are in `CONTEXT.md` and `SCIENTIFIC_STANDARDS.md`; accepted task splits are in
`docs/adr/0006-stage-1-task-suite-and-query-splits.md`.

## Detected Dependencies

The builder has no production dependency and imports only Python standard
library modules. The repository does not declare a Python package on main.

## Important Scripts

- `scripts/build_math_manifest.py` discovers raw source records, validates the
  split, hashes records, writes the immutable manifest, and writes the two
  source-bound smoke queries.

## Current Git State

The working branch is `main` at the documentation-only baseline. The local
requested source checkouts are present under `data/pinned/` and are currently
untracked until their storage policy is finalized.

## Missing or Ambiguous Areas

The raw MATH source-record archive/path is not fixed by the repository and is
not included in the MATH Git checkout. The combined issue-8 Parquet cache is
not accepted as a source-ID authority because its provenance marks split
identity as unverified.

## Notes for Future Skills

The source manifest's `artifact_id` is the SHA-256 of the canonical identity
object containing dataset, source revision, permitted training/validation
IDs, and final IDs. The scientific CLI should consume the generated source
manifest and smoke query records rather than rediscovering data.

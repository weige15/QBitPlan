# Repository Map

## Repository Summary

QBitPlan is a research project for query-conditioned causal bit planning. This
branch contains the Issue 25 execution seam and its standalone MATH
data-preparation command.

## Directory Structure

- `docs/adr/` contains accepted and proposed technical/research decisions.
- `docs/protocol/` contains the accepted Stage-1 protocol and handoff.
- `docs/research/` contains research inputs and feasibility notes.
- `docs/agents/` contains repository workflow and domain guidance.
- `scripts/` contains standalone repository/data-preparation commands.
- `tests/` contains focused tests for standalone commands.
- `data/pinned/` contains local pinned source checkouts requested for data
  preparation; raw records are ignored runtime inputs, not implementation modules.

## Main Source Files

The data-preparation entry point is `scripts/build_math_manifest.py`; the
Issue 25 scientific entry point is `qbitplan`.

## Existing Tests

Focused manifest-builder tests are in `tests/test_build_math_manifest.py` and
Issue 25 seam tests are in `tests/test_stage1_smoke_seam.py`.

## Build System

`pyproject.toml` defines the `qbitplan` package and console script. The
manifest builder uses only the Python standard library.

## Runtime or CLI Entry Points

The standalone preparation command is `python3 scripts/build_math_manifest.py`;
the scientific smoke command is `qbitplan stage1 run --plan <smoke-plan> --mode smoke --gpu-uuid <uuid>`.

## Data and Assets

The requested local checkout is `data/pinned/math` at MATH revision
`985bdc1696e88e8643f081a0ff4719da39f2ae2a`; raw records are under
`data/pinned/math-raw/MATH`; `data/pinned/math-500` is at MATH-500
revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. The MATH Git checkout
contains loader code but not the raw 12,500 problem JSON files. The builder
requires an explicit raw source root with `train/` and `test/` directories.

## Existing Documentation

Project orientation is in `README.md`; terminology and scientific invariants
are in `CONTEXT.md` and `SCIENTIFIC_STANDARDS.md`; accepted task splits are in
`docs/adr/0006-stage-1-task-suite-and-query-splits.md`.

## Detected Dependencies

The builder has no production dependency and imports only Python standard
library modules. The `qbitplan` package is stdlib-only at this slice.

## Important Scripts

- `scripts/build_math_manifest.py` discovers raw source records, validates the
  split, hashes records, writes the immutable manifest, and writes the two
  source-bound smoke queries.

## Current Git State

Pinned source data is materialized under `data/pinned/` and ignored by Git;
generated manifests are retained under `data/manifests/`.


## Missing or Ambiguous Areas

The raw MATH source archive is not included in the MATH Git checkout; its
content hash is recorded in `docs/data-preparation.md`. The combined issue-8
Parquet cache is not accepted as a source-ID authority because its provenance
marks split identity as unverified.

## Notes for Future Skills

The source manifest's `artifact_id` is the SHA-256 of the canonical identity
object containing dataset, source revision, permitted training/validation
IDs, and final IDs. The scientific CLI should consume the generated source
manifest and smoke query records rather than rediscovering data.

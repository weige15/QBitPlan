#!/usr/bin/env python3
"""Build the pinned, non-final MATH manifest used by the Stage-1 smoke plan.

The MATH Git repository contains loaders, not the 12,500 raw problem files.
This command therefore requires an explicit extracted source tree containing
``train/**/*.json`` and ``test/**/*.json``. It intentionally does not read
Hugging Face caches or the combined Parquet cache used by earlier work.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


DATASET = "MATH"
SOURCE_REVISION = "985bdc1696e88e8643f081a0ff4719da39f2ae2a"
EXPECTED_TRAINING_COUNT = 7_500
EXPECTED_SOURCE_TEST_COUNT = 5_000
EXPECTED_FINAL_COUNT = 500
EXPECTED_VALIDATION_COUNT = 4_500
MATH500_JSONL = "test.jsonl"


class ManifestBuildError(ValueError):
    """Raised when the pinned source cannot satisfy the manifest contract."""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ManifestBuildError("canonical JSON does not accept floating-point values")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ManifestBuildError("canonical JSON object keys must be strings")
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-16-be"))
        return {key: _canonical_value(item) for key, item in items}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise ManifestBuildError(f"unsupported value in canonical JSON: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact UTF-8 canonical JSON for the manifest identity domain."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestBuildError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ManifestBuildError(f"non-standard JSON constant is not allowed: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_json_constant,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"cannot read JSON record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestBuildError(f"source record must be a JSON object: {path}")
    return value


def _validate_source_record(record: Mapping[str, Any], path: Path) -> None:
    required = ("problem", "solution", "level", "type")
    missing = [key for key in required if key not in record]
    if missing:
        raise ManifestBuildError(f"source record {path} is missing fields: {missing}")
    if not isinstance(record["problem"], str) or not isinstance(record["solution"], str):
        raise ManifestBuildError(f"source record text fields must be strings: {path}")
    if not isinstance(record["level"], str) or not record["level"]:
        raise ManifestBuildError(f"source record level must be a non-empty string: {path}")
    if not isinstance(record["type"], str):
        raise ManifestBuildError(f"source record type must be a string: {path}")


def _source_id(source_root: Path, path: Path) -> str:
    source_id = path.relative_to(source_root).as_posix()
    if source_id.startswith("./") or "\\" in source_id:
        raise ManifestBuildError(f"source path is not a normalized POSIX ID: {source_id!r}")
    return source_id


def _load_split(source_root: Path, split: str) -> dict[str, dict[str, Any]]:
    split_root = source_root / split
    if not split_root.is_dir():
        raise ManifestBuildError(
            f"source root must contain {split}/ JSON records; missing directory: {split_root}"
        )

    records: dict[str, dict[str, Any]] = {}
    paths = sorted(path for path in split_root.rglob("*.json") if path.is_file())
    if not paths:
        raise ManifestBuildError(f"source split contains no JSON records: {split_root}")
    for path in paths:
        if path.is_symlink():
            raise ManifestBuildError(f"source record must not be a symlink: {path}")
        source_id = _source_id(source_root, path)
        if source_id in records:
            raise ManifestBuildError(f"duplicate source ID: {source_id}")
        record = _load_json_object(path)
        _validate_source_record(record, path)
        records[source_id] = record
    return records


def _load_math500_ids(math500_root: Path) -> list[str]:
    path = math500_root / MATH500_JSONL
    if not path.is_file():
        raise ManifestBuildError(f"MATH-500 checkout is missing {MATH500_JSONL}: {path}")

    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_non_json_constant,
                    )
                except (json.JSONDecodeError, ManifestBuildError) as exc:
                    raise ManifestBuildError(
                        f"invalid MATH-500 JSONL at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict) or not isinstance(row.get("unique_id"), str):
                    raise ManifestBuildError(
                        f"MATH-500 line {line_number} must contain a string unique_id"
                    )
                unique_id = row["unique_id"]
                if unique_id.startswith("./") or "\\" in unique_id:
                    raise ManifestBuildError(f"MATH-500 unique_id is not normalized: {unique_id!r}")
                if not unique_id.startswith("test/") or not unique_id.endswith(".json"):
                    raise ManifestBuildError(
                        f"MATH-500 unique_id must be a source test path: {unique_id!r}"
                    )
                ids.append(unique_id)
    except OSError as exc:
        raise ManifestBuildError(f"cannot read MATH-500 IDs: {exc}") from exc

    if len(ids) != EXPECTED_FINAL_COUNT:
        raise ManifestBuildError(
            f"MATH-500 must contain exactly {EXPECTED_FINAL_COUNT} IDs; found {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ManifestBuildError("MATH-500 unique_id values are not unique")
    return sorted(ids)


def _record_hashes(records: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, str]:
    return {source_id: sha256_canonical(records[source_id]) for source_id in ids}


def _math_prompt(record: Mapping[str, Any]) -> str:
    return (
        "Problem:\n"
        + str(record["problem"])
        + "\n\nSolve the problem. Show your reasoning and put the final answer in\n"
        + r"\boxed{...}."
        + "\nSolution:"
    )


def build_artifacts(source_root: Path, math500_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root.resolve()
    math500_root = math500_root.resolve()
    if not source_root.exists():
        raise ManifestBuildError(f"source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise ManifestBuildError(f"source root is not a directory: {source_root}")
    if not math500_root.is_dir():
        raise ManifestBuildError(f"MATH-500 root is not a directory: {math500_root}")

    training_records = _load_split(source_root, "train")
    source_test_records = _load_split(source_root, "test")
    final_ids = _load_math500_ids(math500_root)

    if len(training_records) != EXPECTED_TRAINING_COUNT:
        raise ManifestBuildError(
            f"source training must contain exactly {EXPECTED_TRAINING_COUNT} records; "
            f"found {len(training_records)}"
        )
    if len(source_test_records) != EXPECTED_SOURCE_TEST_COUNT:
        raise ManifestBuildError(
            f"source test must contain exactly {EXPECTED_SOURCE_TEST_COUNT} records; "
            f"found {len(source_test_records)}"
        )
    training_ids = sorted(training_records)
    source_test_ids = sorted(source_test_records)
    if set(training_ids) & set(source_test_ids):
        raise ManifestBuildError("source training and test IDs overlap")

    missing_final_ids = sorted(set(final_ids) - set(source_test_ids))
    if missing_final_ids:
        raise ManifestBuildError(
            "MATH-500 IDs missing from source test set: " + ", ".join(missing_final_ids[:5])
        )
    validation_ids = sorted(set(source_test_ids) - set(final_ids))
    if len(validation_ids) != EXPECTED_VALIDATION_COUNT:
        raise ManifestBuildError(
            f"validation must contain exactly {EXPECTED_VALIDATION_COUNT} records; "
            f"found {len(validation_ids)}"
        )

    identity = {
        "dataset": DATASET,
        "source_revision": SOURCE_REVISION,
        "permitted_source_ids": {
            "training": training_ids,
            "validation": validation_ids,
        },
        "final_source_ids": final_ids,
    }
    artifact_id = sha256_canonical(identity)
    record_hashes = {
        "training": _record_hashes(training_records, training_ids),
        "validation": _record_hashes(source_test_records, validation_ids),
        "final": _record_hashes(source_test_records, final_ids),
    }
    manifest = {
        **identity,
        "record_hash_algorithm": "SHA-256(canonical JSON record)",
        "record_id_format": "source-relative POSIX JSON path",
        "record_hashes": record_hashes,
        "counts": {
            "training": len(training_ids),
            "validation": len(validation_ids),
            "final": len(final_ids),
        },
        "artifact_id": artifact_id,
    }

    smoke_ids = (("training", training_ids[0]), ("validation", validation_ids[0]))
    source_records = {**training_records, **source_test_records}
    queries = []
    for phase, source_id in smoke_ids:
        record = source_records[source_id]
        queries.append(
            {
                "query_id": source_id,
                "source_id": source_id,
                "source_artifact_id": artifact_id,
                "record": record,
                "record_hash": record_hashes[phase][source_id],
                "dataset": DATASET,
                "phase": phase,
                "source_revision": SOURCE_REVISION,
                "prompt": _math_prompt(record),
                "permitted": True,
            }
        )
    smoke_plan = {
        "dataset": DATASET,
        "source_revision": SOURCE_REVISION,
        "manifest_artifact_id": artifact_id,
        "source_manifest": manifest,
        "queries": queries,
    }
    return manifest, smoke_plan


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ManifestBuildError(f"refusing to overwrite immutable output: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
    except FileExistsError as exc:
        raise ManifestBuildError(f"refusing to overwrite immutable output: {path}") from exc
    except OSError as exc:
        raise ManifestBuildError(f"cannot write output {path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the separated MATH source manifest and two-record smoke plan."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="explicit raw MATH root containing train/ and test/ JSON records",
    )
    parser.add_argument(
        "--math500-root",
        type=Path,
        default=Path("data/pinned/math-500"),
        help="pinned MATH-500 checkout (default: data/pinned/math-500)",
    )
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--smoke-plan-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.manifest_output.resolve() == args.smoke_plan_output.resolve():
            raise ManifestBuildError("manifest and smoke-plan outputs must be different files")
        manifest, smoke_plan = build_artifacts(args.source_root, args.math500_root)
        if args.manifest_output.exists() or args.smoke_plan_output.exists():
            raise ManifestBuildError("refusing to overwrite an existing immutable output")
        _write_once(args.manifest_output, manifest)
        _write_once(args.smoke_plan_output, smoke_plan)
    except (ManifestBuildError, OSError) as exc:
        print(f"build_math_manifest: rejected: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "training_count": manifest["counts"]["training"],
                "validation_count": manifest["counts"]["validation"],
                "final_count": manifest["counts"]["final"],
                "manifest_output": str(args.manifest_output),
                "smoke_plan_output": str(args.smoke_plan_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

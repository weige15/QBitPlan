"""Build the immutable final-only MMLU-Pro source manifest."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qbitplan.identity import canonical_json_bytes, sha256_canonical
from qbitplan.plan import (
    MMLU_PRO_SOURCE_REPO,
    MMLU_PRO_SOURCE_REVISION,
    MMLU_PRO_SOURCE_SPLIT,
    MMLU_PRO_TEST_COUNT,
    MMLU_PRO_TEST_PARQUET,
    MMLU_PRO_TEST_PARQUET_SHA256,
)


class ManifestBuildError(ValueError):
    """Raised when the pinned source cannot satisfy the manifest contract."""


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise ManifestBuildError(f"refusing to overwrite existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise ManifestBuildError(
            "pyarrow is required to read the pinned parquet"
        ) from exc
    try:
        rows = parquet.read_table(path).to_pylist()
    except (OSError, ValueError) as exc:
        raise ManifestBuildError(f"cannot read MMLU-Pro parquet {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ManifestBuildError("MMLU-Pro parquet rows must be objects")
    return rows


def _validate_row(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    required = {
        "question_id",
        "question",
        "options",
        "answer",
        "answer_index",
        "cot_content",
        "category",
        "src",
    }
    if set(row) != required:
        raise ManifestBuildError(
            f"MMLU-Pro row has unexpected fields: {sorted(set(row) ^ required)}"
        )
    question_id = row["question_id"]
    options = row["options"]
    answer = row["answer"]
    answer_index = row["answer_index"]
    if (
        not isinstance(question_id, int)
        or isinstance(question_id, bool)
        or not isinstance(row["question"], str)
        or not isinstance(options, list)
        or not 3 <= len(options) <= 10
        or not all(isinstance(option, str) for option in options)
        or not isinstance(answer, str)
        or len(answer) != 1
        or answer not in "ABCDEFGHIJ"
        or not isinstance(answer_index, int)
        or isinstance(answer_index, bool)
        or not 0 <= answer_index < len(options)
        or answer != chr(ord("A") + answer_index)
        or not isinstance(row["cot_content"], str)
        or not isinstance(row["category"], str)
        or not isinstance(row["src"], str)
    ):
        raise ManifestBuildError(f"invalid MMLU-Pro row {question_id!r}")
    return str(question_id), dict(row)


def build_manifest(source_parquet: Path) -> dict[str, Any]:
    source_parquet = source_parquet.resolve()
    if not source_parquet.is_file():
        raise ManifestBuildError(f"missing source parquet: {source_parquet}")
    source_sha256 = hashlib.sha256(source_parquet.read_bytes()).hexdigest()
    if source_sha256 != MMLU_PRO_TEST_PARQUET_SHA256:
        raise ManifestBuildError(
            f"source parquet hash mismatch: expected {MMLU_PRO_TEST_PARQUET_SHA256}, got {source_sha256}"
        )
    rows = _load_rows(source_parquet)
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id, record = _validate_row(row)
        if source_id in records:
            raise ManifestBuildError(f"duplicate MMLU-Pro question_id: {source_id}")
        records[source_id] = record
    final_ids = sorted(records, key=lambda value: int(value))
    if len(final_ids) != MMLU_PRO_TEST_COUNT:
        raise ManifestBuildError(
            f"MMLU-Pro test must contain exactly {MMLU_PRO_TEST_COUNT} rows; found {len(final_ids)}"
        )
    manifest = {
        "dataset": "MMLU-Pro",
        "source_revision": MMLU_PRO_SOURCE_REVISION,
        "permitted_source_ids": {"training": [], "validation": []},
        "final_source_ids": final_ids,
        "record_hash_algorithm": "SHA-256(canonical JSON record)",
        "record_id_format": "MMLU-Pro question_id decimal string",
        "record_hashes": {
            "training": {},
            "validation": {},
            "final": {
                source_id: sha256_canonical(records[source_id])
                for source_id in final_ids
            },
        },
        "counts": {"training": 0, "validation": 0, "final": len(final_ids)},
        "raw_artifact": {
            "artifact_id": MMLU_PRO_TEST_PARQUET_SHA256,
            "source_repo": MMLU_PRO_SOURCE_REPO,
            "source_path": MMLU_PRO_TEST_PARQUET,
            "source_sha256": MMLU_PRO_TEST_PARQUET_SHA256,
            "source_split": MMLU_PRO_SOURCE_SPLIT,
            "source_rows": len(final_ids),
        },
        "artifact_id": MMLU_PRO_TEST_PARQUET_SHA256,
    }
    manifest["manifest_id"] = sha256_canonical(manifest)
    return manifest


def _download_pinned(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ManifestBuildError(
            "huggingface_hub is required for download mode"
        ) from exc
    return Path(
        hf_hub_download(
            MMLU_PRO_SOURCE_REPO,
            MMLU_PRO_TEST_PARQUET,
            revision=MMLU_PRO_SOURCE_REVISION,
            repo_type="dataset",
            local_dir=cache_dir,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-parquet", type=Path)
    source.add_argument("--download-dir", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        source_parquet = args.source_parquet or _download_pinned(args.download_dir)
        _write_once(args.manifest_output, build_manifest(source_parquet))
    except (ManifestBuildError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

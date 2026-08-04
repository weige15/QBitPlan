"""Declared lookup-table cost estimates for the Stage-1 artifact seam."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..identity import sha256_bytes, sha256_canonical
from ..plan import COST_DIMENSIONS

LOOKUP_EVIDENCE_CLASS = "lookup-table estimated"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")  # noqa: TRY004
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _profile_id(value: Any, label: str) -> str:
    if value == "BF16":
        return value
    if not isinstance(value, str) or len(value) != 8 or set(value) - {"0", "1"}:
        raise ValueError(f"{label} is not a canonical profile ID")
    return value


def _unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must be unique")
    return list(value)


def _validate_cost(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")  # noqa: TRY004
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


class LookupCostEstimateAdapter:
    """Serve exact declared lookup values without observing hardware."""

    evidence_class = LOOKUP_EVIDENCE_CLASS

    def __init__(self, table: Mapping[str, Any], *, lookup_artifact_id: str) -> None:
        self._lookup_artifact_id = lookup_artifact_id
        self._table = self._validate_table(table)
        coverage = self._table["coverage_manifest"]
        self._coverage = coverage
        self._entries = {
            (entry["query_id"], entry["profile_id"]): entry["costs"]
            for entry in self._table["entries"]
        }

    @classmethod
    def from_path(cls, path: Path) -> LookupCostEstimateAdapter:
        try:
            raw_bytes = path.read_bytes()
            table = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read declared lookup table: {path}") from exc
        if not isinstance(table, Mapping):
            raise ValueError("declared lookup table must be a JSON object")  # noqa: TRY004
        return cls(table, lookup_artifact_id=sha256_bytes(raw_bytes))

    @classmethod
    def from_mapping(
        cls, table: Mapping[str, Any], *, lookup_artifact_id: str | None = None
    ) -> LookupCostEstimateAdapter:
        artifact_id = lookup_artifact_id or sha256_canonical(table)
        return cls(table, lookup_artifact_id=artifact_id)

    @staticmethod
    def _validate_table(table: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "schema_version",
            "method",
            "evidence_class",
            "source_artifact",
            "coverage_manifest",
            "entries",
        }
        missing = sorted(required - set(table))
        unexpected = sorted(set(table) - required)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValueError(f"lookup table has unspecified fields: {', '.join(details)}")
        if table["schema_version"] != "qbitplan.stage1.lookup-cost-table.v1":
            raise ValueError("lookup table has an unsupported schema_version")
        method = _require_nonempty_string(table["method"], "lookup table method")
        if table["evidence_class"] != LOOKUP_EVIDENCE_CLASS:
            raise ValueError(
                f"lookup table evidence_class must be {LOOKUP_EVIDENCE_CLASS!r}"
            )
        source = _require_mapping(table["source_artifact"], "lookup table source_artifact")
        source_artifact_id = _require_nonempty_string(
            source.get("artifact_id"), "lookup table source_artifact.artifact_id"
        )
        location = _require_nonempty_string(
            source.get("location"), "lookup table source_artifact.location"
        )
        coverage = _require_mapping(table["coverage_manifest"], "coverage_manifest")
        coverage_required = {
            "schema_version",
            "query_ids",
            "profile_ids",
            "dimensions",
            "manifest_id",
        }
        if set(coverage) != coverage_required:
            raise ValueError("coverage_manifest has unspecified fields")
        if coverage["schema_version"] != "qbitplan.stage1.lookup-cost-coverage.v1":
            raise ValueError("coverage_manifest has an unsupported schema_version")
        query_ids = _unique_strings(coverage["query_ids"], "coverage_manifest.query_ids")
        profile_ids = [
            _profile_id(item, "coverage_manifest.profile_ids entry")
            for item in _unique_strings(coverage["profile_ids"], "coverage_manifest.profile_ids")
        ]
        dimensions = _unique_strings(coverage["dimensions"], "coverage_manifest.dimensions")
        if not dimensions or set(dimensions) - set(COST_DIMENSIONS):
            raise ValueError("coverage_manifest.dimensions must be accepted cost dimensions")
        coverage_payload = {
            key: coverage[key]
            for key in ("schema_version", "query_ids", "profile_ids", "dimensions")
        }
        if coverage["manifest_id"] != sha256_canonical(coverage_payload):
            raise ValueError("coverage_manifest.manifest_id does not match its canonical payload")
        entries = table["entries"]
        if not isinstance(entries, list):
            raise ValueError("lookup table entries must be an array")  # noqa: TRY004
        normalized_entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw_entry in enumerate(entries):
            entry = _require_mapping(raw_entry, f"lookup table entries[{index}]")
            if set(entry) != {"query_id", "profile_id", "costs"}:
                raise ValueError(f"lookup table entries[{index}] has unspecified fields")
            query_id = _require_nonempty_string(entry["query_id"], "lookup entry query_id")
            profile_id = _profile_id(entry["profile_id"], "lookup entry profile_id")
            key = (query_id, profile_id)
            if query_id not in query_ids or profile_id not in profile_ids:
                raise ValueError(
                    f"lookup entry {key!r} is outside the declared coverage manifest"
                )
            if key in seen:
                raise ValueError(f"lookup table contains duplicate entry {key!r}")
            seen.add(key)
            costs = _require_mapping(entry["costs"], f"lookup table entries[{index}].costs")
            if set(costs) - set(dimensions):
                raise ValueError(f"lookup entry {key!r} contains an uncovered dimension")
            normalized_entries.append(
                {
                    "query_id": query_id,
                    "profile_id": profile_id,
                    "costs": {
                        dimension: _validate_cost(value, f"lookup entry {key!r}.{dimension}")
                        for dimension, value in costs.items()
                    },
                }
            )
        return {
            "schema_version": table["schema_version"],
            "method": method,
            "evidence_class": LOOKUP_EVIDENCE_CLASS,
            "source_artifact": {
                **dict(source),
                "artifact_id": source_artifact_id,
                "location": location,
            },
            "coverage_manifest": {
                "schema_version": coverage["schema_version"],
                "query_ids": query_ids,
                "profile_ids": profile_ids,
                "dimensions": dimensions,
                "manifest_id": coverage["manifest_id"],
            },
            "entries": normalized_entries,
        }

    def lookup_metadata(self) -> Mapping[str, Any]:
        return {
            "lookup_artifact_id": self._lookup_artifact_id,
            "lookup_manifest_id": self._coverage["manifest_id"],
            "source_artifact": copy.deepcopy(self._table["source_artifact"]),
            "source_artifact_id": self._table["source_artifact"]["artifact_id"],
            "method": self._table["method"],
            "coverage_manifest": copy.deepcopy(self._coverage),
        }

    def hardware_identity(self) -> Mapping[str, Any]:
        return {
            "identity_status": "not_applicable",
            "reason_code": "LOOKUP_TABLE_NO_HARDWARE_EXECUTION",
        }

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        query_id = _require_nonempty_string(query.get("query_id"), "query.query_id")
        profile_id = _profile_id(variant_id, "variant_id")
        declared_queries = set(self._coverage["query_ids"])
        declared_profiles = set(self._coverage["profile_ids"])
        declared_dimensions = set(self._coverage["dimensions"])
        entry = self._entries.get((query_id, profile_id), {})
        cost_vector: dict[str, dict[str, Any]] = {}
        estimated_dimensions: list[str] = []
        omitted_dimensions: list[dict[str, str]] = []
        for dimension in COST_DIMENSIONS:
            reason: str | None = None
            if query_id not in declared_queries:
                reason = "LOOKUP_QUERY_UNCOVERED"
            elif profile_id not in declared_profiles:
                reason = "LOOKUP_PROFILE_UNCOVERED"
            elif dimension not in declared_dimensions:
                reason = "LOOKUP_DIMENSION_UNCOVERED"
            elif dimension not in entry:
                reason = "LOOKUP_ENTRY_DIMENSION_MISSING"
            if reason is not None:
                omitted_dimensions.append({"dimension": dimension, "reason": reason})
                cost_vector[dimension] = {
                    "status": "omitted/unavailable",
                    "reason": reason,
                    "evidence_class": LOOKUP_EVIDENCE_CLASS,
                    "method": self._table["method"],
                    "coverage": self._coverage["manifest_id"],
                    "source_artifact_id": self._table["source_artifact"]["artifact_id"],
                    "lookup_artifact_id": self._lookup_artifact_id,
                }
                continue
            estimated_dimensions.append(dimension)
            cost_vector[dimension] = {
                "status": "estimated",
                "value": entry[dimension],
                "evidence_class": LOOKUP_EVIDENCE_CLASS,
                "method": self._table["method"],
                "coverage": self._coverage["manifest_id"],
                "source_artifact_id": self._table["source_artifact"]["artifact_id"],
                "lookup_artifact_id": self._lookup_artifact_id,
            }
        status = "complete" if not omitted_dimensions else "incomplete"
        return {
            "status": status,
            "reason_code": "LOOKUP_COVERAGE_COMPLETE" if status == "complete" else "LOOKUP_COVERAGE_INCOMPLETE",
            "cost_vector": cost_vector,
            "method": self._table["method"],
            "lookup_artifact_id": self._lookup_artifact_id,
            "lookup_manifest_id": self._coverage["manifest_id"],
            "source_artifact_id": self._table["source_artifact"]["artifact_id"],
            "coverage": {
                "query_id": query_id,
                "profile_id": profile_id,
                "query_covered": query_id in declared_queries,
                "profile_covered": profile_id in declared_profiles,
                "declared_dimensions": list(self._coverage["dimensions"]),
                "estimated_dimensions": estimated_dimensions,
                "omitted_dimensions": omitted_dimensions,
                "coverage_manifest_id": self._coverage["manifest_id"],
            },
        }

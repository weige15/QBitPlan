"""Typed, leakage-checked inputs for the Stage-1 controller interfaces.

This module only adapts immutable Stage-1 records.  It does not construct
profiles, derive target sets, fit a model, or synthesize causal context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._controller_common import (
    PrefixContextProvider,
    Profile,
    QueryFeatures,
    TrainingQuery,
    profile_from_id,
    profile_id,
    validate_profile,
)

_PROFILE_ARTIFACT_TYPE = "profile-execution-inventory"
_PROFILE_SCHEMA = "qbitplan.stage1.profile-inventory.v1"
_FEATURE_ARTIFACT_TYPE = "stage1-query-features"
_FEATURE_SCHEMA = "qbitplan.stage1.query-features.v1"
_TARGET_ARTIFACT_TYPE = "stage1-target-sets"
_TARGET_SCHEMA = "qbitplan.stage1.target-sets.v1"
_PREFIX_ARTIFACT_TYPE = "stage1-causal-prefixes"
_PREFIX_SCHEMA = "qbitplan.stage1.causal-prefixes.v1"
_NON_FINAL_PHASES = frozenset({"training", "validation"})
_PHASE_ORDER = {"training": 0, "validation": 1}


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _require_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _required(mapping, key, label)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _require_sha256(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _require_string(mapping, key, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label}.{key} must be a lowercase SHA-256 digest")
    return value


def _copy_records(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping):
        records = _required(value, "records", label)
    else:
        records = value
    if not isinstance(records, list):
        raise ValueError(f"{label}.records must be an array")  # noqa: TRY004
    copied: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(  # noqa: TRY004
                f"{label}.records[{index}] must be an object"
            )
        copied.append(dict(record))
    return tuple(copied)


def _validate_artifact_header(
    value: Mapping[str, Any],
    *,
    artifact_type: str,
    schema_version: str,
    label: str,
) -> tuple[str, str]:
    if _require_string(value, "artifact_type", label) != artifact_type:
        raise ValueError(f"{label}.artifact_type is not {artifact_type!r}")
    if _require_string(value, "schema_version", label) != schema_version:
        raise ValueError(f"{label}.schema_version is not {schema_version!r}")
    artifact_id = _require_sha256(value, "artifact_id", label)
    source_manifest_id = _require_sha256(value, "source_manifest_id", label)
    final_ids = value.get("final_source_ids", [])
    if not isinstance(final_ids, list) or any(
        not isinstance(item, str) or not item for item in final_ids
    ):
        raise ValueError(f"{label}.final_source_ids must be a string array")
    return artifact_id, source_manifest_id


def _validate_non_final_record(
    record: Mapping[str, Any],
    *,
    label: str,
    final_source_ids: frozenset[str],
) -> tuple[str, str, str]:
    query_id = _require_string(record, "query_id", label)
    dataset = _require_string(record, "dataset", label)
    phase = _require_string(record, "phase", label)
    if phase not in _NON_FINAL_PHASES:
        raise ValueError(
            f"{label} contains final or unsupported phase {phase!r}; "
            "controller fitting accepts non-final training/validation only"
        )
    source_id = record.get("source_id", query_id)
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"{label}.source_id must be a non-empty string")
    if query_id in final_source_ids or source_id in final_source_ids:
        raise ValueError(f"{label} contains final query ID {query_id!r}")
    return query_id, dataset, phase


def _profile_tuple(identifier: str, *, label: str) -> Profile:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{label} must be a non-empty canonical profile ID")
    try:
        profile = profile_from_id(identifier)
    except ValueError as exc:
        raise ValueError(f"{label} is not a canonical binary profile ID") from exc
    if profile_id(profile) != identifier:
        raise ValueError(f"{label} is not a canonical binary profile ID")
    return profile


def _readonly_float_tuple(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty numeric array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric")  # noqa: TRY004
        numeric = float(item)
        if not np.isfinite(numeric):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(numeric)
    return tuple(result)


@dataclass(frozen=True)
class Stage1ProfileArtifact:
    """Validated executable-profile identity from one immutable inventory."""

    artifact_id: str
    source_manifest_id: str
    profile_ids: tuple[str, ...]
    executable_profiles: tuple[Profile, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Stage1ProfileArtifact:
        artifact_id, source_manifest_id = _validate_artifact_header(
            value,
            artifact_type=_PROFILE_ARTIFACT_TYPE,
            schema_version=_PROFILE_SCHEMA,
            label="profile artifact",
        )
        profile_values = _required(value, "profile_ids", "profile artifact")
        executable_values = _required(value, "p_exec", "profile artifact")
        if not isinstance(profile_values, list) or not profile_values:
            raise ValueError("profile artifact.profile_ids must be a non-empty array")
        parsed_profile_ids = tuple(
            profile_id(
                _profile_tuple(identifier, label=f"profile_ids[{index}]")
            )
            for index, identifier in enumerate(profile_values)
        )
        if len(parsed_profile_ids) != len(set(parsed_profile_ids)):
            raise ValueError("profile artifact.profile_ids must be unique")
        profile_ids = parsed_profile_ids
        if profile_ids != tuple(sorted(profile_ids)):
            raise ValueError("profile artifact.profile_ids must be canonical and sorted")
        if not isinstance(executable_values, list) or not executable_values:
            raise ValueError(
                "profile artifact.p_exec must be a non-empty sorted executable set"
            )
        parsed_executable_ids = tuple(
            profile_id(
                _profile_tuple(identifier, label=f"p_exec[{index}]")
            )
            for index, identifier in enumerate(executable_values)
        )
        if len(parsed_executable_ids) != len(set(parsed_executable_ids)):
            raise ValueError("profile artifact.p_exec must be unique")
        executable_ids = parsed_executable_ids
        if executable_ids != tuple(sorted(executable_ids)):
            raise ValueError("profile artifact.p_exec must be canonical and sorted")
        if not set(executable_ids).issubset(profile_ids):
            raise ValueError("profile artifact.p_exec must be a subset of profile_ids")
        if len({len(profile) for profile in map(profile_from_id, profile_ids)}) != 1:
            raise ValueError("profile artifact profiles must share one group count")
        return cls(
            artifact_id=artifact_id,
            source_manifest_id=source_manifest_id,
            profile_ids=profile_ids,
            executable_profiles=tuple(profile_from_id(identifier) for identifier in executable_ids),
        )


@dataclass(frozen=True)
class _FeatureRecord:
    query_id: str
    dataset: str
    phase: str
    features: QueryFeatures


@dataclass(frozen=True)
class _TargetRecord:
    query_id: str
    dataset: str
    phase: str
    target_profiles: tuple[Profile, ...]


@dataclass(frozen=True)
class _PrefixRecord:
    query_id: str
    dataset: str
    phase: str
    prefix: Profile
    hidden_summary: tuple[float, ...]


@dataclass(frozen=True)
class Stage1ControllerArtifactAdapter:
    """Adapt four typed Stage-1 artifact families to controller interfaces.

    The adapter accepts only non-final records, requires one common source
    manifest, preserves empty target sets, and raises when a causal context is
    absent.  It never substitutes a profile, a zero context, or a future
    hidden state.
    """

    profile_artifact: Stage1ProfileArtifact
    _features: tuple[_FeatureRecord, ...]
    _targets: tuple[_TargetRecord, ...]
    _prefixes: tuple[_PrefixRecord, ...]
    _context: Mapping[tuple[str, Profile], tuple[float, ...]]

    @classmethod
    def from_mappings(
        cls,
        profile_artifact: Mapping[str, Any],
        query_features_artifact: Mapping[str, Any],
        target_sets_artifact: Mapping[str, Any],
        causal_prefix_artifact: Mapping[str, Any],
    ) -> Stage1ControllerArtifactAdapter:
        profiles = Stage1ProfileArtifact.from_mapping(profile_artifact)
        features, feature_manifest, feature_final_ids = cls._parse_features(
            query_features_artifact
        )
        targets, target_manifest, target_final_ids = cls._parse_targets(
            target_sets_artifact,
            profiles,
        )
        prefixes, prefix_manifest, prefix_final_ids = cls._parse_prefixes(
            causal_prefix_artifact,
            features,
            len(profiles.executable_profiles[0]),
        )
        manifests = {
            profiles.source_manifest_id,
            feature_manifest,
            target_manifest,
            prefix_manifest,
        }
        if len(manifests) != 1:
            raise ValueError("controller artifacts must share one source_manifest_id")
        final_ids = feature_final_ids | target_final_ids | prefix_final_ids
        if final_ids:
            raise ValueError("controller artifacts must not declare final query IDs")
        feature_by_id = {record.query_id: record for record in features}
        target_by_id = {record.query_id: record for record in targets}
        if set(feature_by_id) != set(target_by_id):
            raise ValueError("query-feature and target-set artifacts must cover the same queries")
        for query_id, feature in feature_by_id.items():
            target = target_by_id[query_id]
            if (feature.dataset, feature.phase) != (target.dataset, target.phase):
                raise ValueError(f"query {query_id!r} has mismatched dataset/phase lineage")
        context: dict[tuple[str, Profile], tuple[float, ...]] = {}
        for record in prefixes:
            key = (record.query_id, record.prefix)
            if key in context:
                raise ValueError("causal prefix artifact contains duplicate records")
            context[key] = record.hidden_summary
        for target in targets:
            for profile in target.target_profiles:
                for group_index in range(1, len(profile)):
                    prefix = profile[:group_index]
                    if (target.query_id, prefix) not in context:
                        raise ValueError(
                            "causal prefix artifact is missing the observed target "
                            f"prefix for query {target.query_id!r}"
                        )
        return cls(
            profile_artifact=profiles,
            _features=tuple(sorted(features, key=cls._record_sort_key)),
            _targets=tuple(sorted(targets, key=cls._record_sort_key)),
            _prefixes=tuple(sorted(prefixes, key=cls._prefix_sort_key)),
            _context=context,
        )

    @classmethod
    def from_paths(
        cls,
        profile_artifact_path: str | Path,
        query_features_artifact_path: str | Path,
        target_sets_artifact_path: str | Path,
        causal_prefix_artifact_path: str | Path,
    ) -> Stage1ControllerArtifactAdapter:
        return cls.from_mappings(
            cls._load_json_artifact(profile_artifact_path, "profile artifact"),
            cls._load_json_artifact(query_features_artifact_path, "query-feature artifact"),
            cls._load_json_artifact(target_sets_artifact_path, "target-set artifact"),
            cls._load_json_artifact(causal_prefix_artifact_path, "causal-prefix artifact"),
        )

    @classmethod
    def from_bundle_root(
        cls,
        profile_bundle_root: str | Path,
        query_features_artifact_path: str | Path,
        target_sets_artifact_path: str | Path,
        causal_prefix_artifact_path: str | Path,
    ) -> Stage1ControllerArtifactAdapter:
        """Load a profile inventory from an immutable execution bundle root."""

        root = Path(profile_bundle_root)
        inventory_path = root / "profile-inventory.json"
        index_path = root / "artifact-index.json"
        profile_payload = cls._load_json_artifact(inventory_path, "profile artifact")
        if "artifact_id" not in profile_payload and index_path.is_file():
            index = cls._load_json_artifact(index_path, "artifact index")
            file_hashes = index.get("file_hashes")
            if isinstance(file_hashes, Mapping) and isinstance(
                file_hashes.get("profile-inventory.json"), str
            ):
                profile_payload["artifact_id"] = file_hashes["profile-inventory.json"]
        return cls.from_mappings(
            profile_payload,
            cls._load_json_artifact(query_features_artifact_path, "query-feature artifact"),
            cls._load_json_artifact(target_sets_artifact_path, "target-set artifact"),
            cls._load_json_artifact(causal_prefix_artifact_path, "causal-prefix artifact"),
        )

    @staticmethod
    def _load_json_artifact(path: str | Path, label: str) -> dict[str, Any]:
        artifact_path = Path(path)
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {label}: {artifact_path}") from exc
        if not isinstance(value, dict):
            raise ValueError(  # noqa: TRY004
                f"{label} must be a JSON object: {artifact_path}"
            )
        return value

    @staticmethod
    def _record_sort_key(record: _FeatureRecord | _TargetRecord) -> tuple[int, str]:
        return (_PHASE_ORDER[record.phase], record.query_id)

    @staticmethod
    def _prefix_sort_key(record: _PrefixRecord) -> tuple[int, str, str]:
        return (_PHASE_ORDER[record.phase], record.query_id, profile_id(record.prefix))

    @staticmethod
    def _parse_features(
        value: Mapping[str, Any],
    ) -> tuple[tuple[_FeatureRecord, ...], str, frozenset[str]]:
        _, manifest = _validate_artifact_header(
            value,
            artifact_type=_FEATURE_ARTIFACT_TYPE,
            schema_version=_FEATURE_SCHEMA,
            label="query-feature artifact",
        )
        final_ids = frozenset(value.get("final_source_ids", []))
        records = _copy_records(value, "query-feature artifact")
        parsed: list[_FeatureRecord] = []
        seen: set[str] = set()
        for index, record in enumerate(records):
            label = f"query-feature artifact.records[{index}]"
            query_id, dataset, phase = _validate_non_final_record(
                record,
                label=label,
                final_source_ids=final_ids,
            )
            if query_id in seen:
                raise ValueError("query-feature artifact contains duplicate query IDs")
            seen.add(query_id)
            structural = _readonly_float_tuple(record.get("structural"), label=f"{label}.structural")
            embedding = _readonly_float_tuple(record.get("embedding"), label=f"{label}.embedding")
            parsed.append(
                _FeatureRecord(
                    query_id=query_id,
                    dataset=dataset,
                    phase=phase,
                    features=QueryFeatures(structural, embedding),
                )
            )
        if not parsed:
            raise ValueError("query-feature artifact must contain at least one record")
        return tuple(parsed), manifest, final_ids

    @staticmethod
    def _parse_targets(
        value: Mapping[str, Any],
        profiles: Stage1ProfileArtifact,
    ) -> tuple[tuple[_TargetRecord, ...], str, frozenset[str]]:
        _, manifest = _validate_artifact_header(
            value,
            artifact_type=_TARGET_ARTIFACT_TYPE,
            schema_version=_TARGET_SCHEMA,
            label="target-set artifact",
        )
        final_ids = frozenset(value.get("final_source_ids", []))
        records = _copy_records(value, "target-set artifact")
        executable = set(profiles.executable_profiles)
        parsed: list[_TargetRecord] = []
        seen: set[str] = set()
        for index, record in enumerate(records):
            label = f"target-set artifact.records[{index}]"
            query_id, dataset, phase = _validate_non_final_record(
                record,
                label=label,
                final_source_ids=final_ids,
            )
            if query_id in seen:
                raise ValueError("target-set artifact contains duplicate query IDs")
            seen.add(query_id)
            raw_targets = _required(record, "target_profile_ids", label)
            if not isinstance(raw_targets, list):
                raise ValueError(  # noqa: TRY004
                    f"{label}.target_profile_ids must be an array"
                )
            target_profiles = tuple(
                sorted(
                    {
                        _profile_tuple(identifier, label=f"{label}.target_profile_ids[{item_index}]")
                        for item_index, identifier in enumerate(raw_targets)
                    },
                    key=profile_id,
                )
            )
            if not set(target_profiles).issubset(executable):
                raise ValueError(
                    f"target set for query {query_id!r} contains a non-executable profile"
                )
            parsed.append(_TargetRecord(query_id, dataset, phase, target_profiles))
        if not parsed:
            raise ValueError("target-set artifact must contain at least one record")
        return tuple(parsed), manifest, final_ids

    @staticmethod
    def _parse_prefixes(
        value: Mapping[str, Any],
        features: Sequence[_FeatureRecord],
        num_groups: int,
    ) -> tuple[tuple[_PrefixRecord, ...], str, frozenset[str]]:
        _, manifest = _validate_artifact_header(
            value,
            artifact_type=_PREFIX_ARTIFACT_TYPE,
            schema_version=_PREFIX_SCHEMA,
            label="causal-prefix artifact",
        )
        final_ids = frozenset(value.get("final_source_ids", []))
        feature_by_id = {record.query_id: record for record in features}
        records = _copy_records(value, "causal-prefix artifact")
        parsed: list[_PrefixRecord] = []
        seen: set[tuple[str, Profile]] = set()
        for index, record in enumerate(records):
            label = f"causal-prefix artifact.records[{index}]"
            query_id, dataset, phase = _validate_non_final_record(
                record,
                label=label,
                final_source_ids=final_ids,
            )
            if query_id not in feature_by_id:
                raise ValueError(f"{label} references an unknown query ID")
            feature = feature_by_id[query_id]
            if (dataset, phase) != (feature.dataset, feature.phase):
                raise ValueError(f"{label} has mismatched dataset/phase lineage")
            prefix_identifier = _required(record, "prefix_profile_id", label)
            if prefix_identifier == "":
                prefix: Profile = ()
            else:
                prefix = _profile_tuple(prefix_identifier, label=f"{label}.prefix_profile_id")
            validate_profile(prefix, allow_empty=True)
            if len(prefix) >= num_groups:
                raise ValueError(
                    f"{label} contains a post-final prefix; causal summaries may only follow an executed proper prefix"
                )
            key = (query_id, prefix)
            if key in seen:
                raise ValueError("causal prefix artifact contains duplicate records")
            seen.add(key)
            hidden = _readonly_float_tuple(
                record.get("hidden_summary"), label=f"{label}.hidden_summary"
            )
            if len(hidden) != feature.features.embedding_dim:
                raise ValueError(
                    f"{label}.hidden_summary dimension does not match query embedding"
                )
            parsed.append(_PrefixRecord(query_id, dataset, phase, prefix, hidden))
        return tuple(parsed), manifest, final_ids

    @property
    def executable_profiles(self) -> tuple[Profile, ...]:
        return self.profile_artifact.executable_profiles

    @property
    def training_queries(self) -> tuple[TrainingQuery, ...]:
        return self.queries(phase="training")

    @property
    def context_provider(self) -> PrefixContextProvider:
        return self.causal_context

    def queries(
        self,
        *,
        phase: str | None = None,
        dataset: str | None = None,
    ) -> tuple[TrainingQuery, ...]:
        feature_by_id = {record.query_id: record for record in self._features}
        target_by_id = {record.query_id: record for record in self._targets}
        selected: list[TrainingQuery] = []
        for query_id in sorted(feature_by_id, key=lambda item: (_PHASE_ORDER[feature_by_id[item].phase], item)):
            feature = feature_by_id[query_id]
            target = target_by_id[query_id]
            if phase is not None and feature.phase != phase:
                continue
            if dataset is not None and feature.dataset != dataset:
                continue
            selected.append(
                TrainingQuery(
                    query_id=query_id,
                    features=feature.features,
                    target_profiles=target.target_profiles,
                )
            )
        if not selected:
            raise ValueError("no controller queries match the requested phase/dataset")
        return tuple(selected)

    def query_features(self, query_id: str) -> QueryFeatures:
        for record in self._features:
            if record.query_id == query_id:
                return record.features
        raise KeyError(f"unknown query-feature artifact query ID: {query_id}")

    def causal_context(self, query_id: str, prefix: Profile) -> Sequence[float]:
        validate_profile(prefix, allow_empty=True)
        if len(prefix) >= len(self.executable_profiles[0]):
            raise ValueError("causal context prefix must be a proper executed prefix")
        if not prefix:
            raise KeyError(
                "causal prefix artifact is not queried for group 0; no synthetic empty context is provided"
            )
        try:
            return self._context[(query_id, prefix)]
        except KeyError as exc:
            raise KeyError(
                f"causal prefix artifact has no context for query {query_id!r} "
                f"and prefix {profile_id(prefix)!r}"
            ) from exc

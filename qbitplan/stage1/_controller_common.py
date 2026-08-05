"""Shared types and deterministic numerical helpers for controller baselines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Bit: TypeAlias = int
Profile: TypeAlias = tuple[Bit, ...]
FeatureArray: TypeAlias = NDArray[np.float32]
FloatArray: TypeAlias = NDArray[np.float64]
PrefixContextProvider: TypeAlias = Callable[[str, Profile], Sequence[float]]

_ALLOWED_BITS = (4, 8)


def _readonly_vector(
    values: Sequence[float],
    *,
    dtype: type[np.floating[Any]],
    name: str,
) -> NDArray[Any]:
    vector = np.asarray(values, dtype=dtype)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    vector = vector.copy()
    vector.flags.writeable = False
    return vector


def readonly_feature_vector(
    values: Sequence[float], *, name: str
) -> FeatureArray:
    return _readonly_vector(values, dtype=np.float32, name=name)


def readonly_float_vector(values: Sequence[float], *, name: str) -> FloatArray:
    return _readonly_vector(values, dtype=np.float64, name=name)


def l2_normalize_feature(
    values: Sequence[float], *, name: str
) -> FeatureArray:
    vector = readonly_feature_vector(values, name=name)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    normalized = vector / norm
    normalized.flags.writeable = False
    return normalized


def validate_profile(
    profile: Profile,
    *,
    num_groups: int | None = None,
    allow_empty: bool = False,
) -> None:
    if not profile and not allow_empty:
        raise ValueError("profiles must contain at least one group")
    if num_groups is not None and len(profile) != num_groups:
        raise ValueError(
            f"profile has {len(profile)} groups; expected {num_groups}"
        )
    if any(bit not in _ALLOWED_BITS for bit in profile):
        raise ValueError("profiles may contain only 4-bit and 8-bit decisions")


def profile_id(profile: Profile) -> str:
    """Return the canonical binary ID, mapping 4 -> 0 and 8 -> 1."""

    validate_profile(profile)
    return "".join("0" if bit == 4 else "1" for bit in profile)


def prefix_id(prefix: Profile) -> str:
    return "" if not prefix else profile_id(prefix)


def profile_from_id(identifier: str) -> Profile:
    """Decode a canonical binary profile ID."""

    if not identifier or any(character not in "01" for character in identifier):
        raise ValueError("profile ID must be a non-empty binary string")
    return tuple(4 if character == "0" else 8 for character in identifier)


def enumerate_profiles(num_groups: int = 8) -> tuple[Profile, ...]:
    """Enumerate the canonical {4, 8}^G profile universe."""

    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    return tuple(product(_ALLOWED_BITS, repeat=num_groups))


def canonical_profiles(profiles: Iterable[Profile]) -> tuple[Profile, ...]:
    materialized = tuple(profiles)
    if not materialized:
        raise ValueError("at least one executable profile is required")
    num_groups = len(materialized[0])
    for profile in materialized:
        validate_profile(profile, num_groups=num_groups)
    if len(set(materialized)) != len(materialized):
        raise ValueError("executable profiles must be unique")
    return tuple(sorted(materialized, key=profile_id))


@dataclass(frozen=True)
class QueryFeatures:
    """Causally available query features with shapes structural[S], embedding[E]."""

    structural: FeatureArray
    embedding: FeatureArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "structural",
            readonly_feature_vector(
                self.structural, name="structural features"
            ),
        )
        object.__setattr__(
            self,
            "embedding",
            l2_normalize_feature(self.embedding, name="query embedding"),
        )

    @property
    def structural_dim(self) -> int:
        return int(self.structural.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[0])

    def vector(self) -> FeatureArray:
        result = np.concatenate((self.structural, self.embedding))
        result.flags.writeable = False
        return result


@dataclass(frozen=True)
class TrainingQuery:
    """One training query and its set-valued executable profile target."""

    query_id: str
    features: QueryFeatures
    target_profiles: tuple[Profile, ...]

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if len(set(self.target_profiles)) != len(self.target_profiles):
            raise ValueError("target_profiles must be unique")
        for profile in self.target_profiles:
            validate_profile(profile)


@dataclass(frozen=True)
class StructuralNormalizer:
    """Training-only mean and scale for structural query features."""

    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]]) -> StructuralNormalizer:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("structural training values must have shape [N, S]")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("structural training values must be finite")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(
            mean=readonly_float_vector(mean, name="normalizer mean"),
            scale=readonly_float_vector(scale, name="normalizer scale"),
        )

    def transform(self, values: Sequence[float]) -> FeatureArray:
        vector = readonly_float_vector(values, name="raw structural features")
        if vector.shape != self.mean.shape:
            raise ValueError(
                f"structural feature shape {vector.shape} does not match "
                f"normalizer shape {self.mean.shape}"
            )
        transformed = np.asarray(
            (vector - self.mean) / self.scale, dtype=np.float32
        )
        transformed.flags.writeable = False
        return transformed


@dataclass(frozen=True)
class LinearLeastSquares:
    """Minimum-norm multi-output ordinary least squares."""

    weights: FloatArray
    training_mse: float

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]] | FeatureArray | FloatArray,
        targets: Sequence[Sequence[float]] | FeatureArray | FloatArray,
    ) -> LinearLeastSquares:
        feature_matrix = np.asarray(features, dtype=np.float64)
        target_matrix = np.asarray(targets, dtype=np.float64)
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
            raise ValueError("features must have shape [N, D] with N > 0")
        if target_matrix.ndim == 1:
            target_matrix = target_matrix[:, None]
        if target_matrix.ndim != 2:
            raise ValueError("targets must have shape [N, O]")
        if feature_matrix.shape[0] != target_matrix.shape[0]:
            raise ValueError("features and targets must have the same row count")
        if not np.all(np.isfinite(feature_matrix)):
            raise ValueError("features must be finite")
        if not np.all(np.isfinite(target_matrix)):
            raise ValueError("targets must be finite")
        design = np.concatenate(
            (np.ones((feature_matrix.shape[0], 1)), feature_matrix), axis=1
        )
        weights, _, _, _ = np.linalg.lstsq(design, target_matrix, rcond=None)
        mse = float(np.mean((design @ weights - target_matrix) ** 2))
        weights.flags.writeable = False
        return cls(weights=weights, training_mse=mse)

    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[0] - 1)

    def predict(self, features: Sequence[float]) -> FloatArray:
        vector = readonly_float_vector(features, name="linear model input")
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"linear model input has {vector.shape[0]} values; "
                f"expected {self.input_dim}"
            )
        result = np.concatenate((np.ones(1), vector)) @ self.weights
        if not np.all(np.isfinite(result)):
            raise ValueError("linear model produced a non-finite score")
        result.flags.writeable = False
        return result


def query_feature_matrix(examples: Sequence[TrainingQuery]) -> FeatureArray:
    if not examples:
        raise ValueError("at least one training query is required")
    structural_dim = examples[0].features.structural_dim
    embedding_dim = examples[0].features.embedding_dim
    rows: list[FeatureArray] = []
    seen_query_ids: set[str] = set()
    for example in examples:
        if example.query_id in seen_query_ids:
            raise ValueError("training query IDs must be unique")
        seen_query_ids.add(example.query_id)
        if example.features.structural_dim != structural_dim:
            raise ValueError("all structural feature vectors must share one shape")
        if example.features.embedding_dim != embedding_dim:
            raise ValueError("all query embeddings must share one shape")
        rows.append(example.features.vector())
    matrix = np.stack(rows)
    matrix.flags.writeable = False
    return matrix

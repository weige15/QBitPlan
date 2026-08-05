"""Query-only and static Stage-1 selector baselines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._controller_common import (
    FloatArray,
    LinearLeastSquares,
    Profile,
    QueryFeatures,
    TrainingQuery,
    canonical_profiles,
    profile_id,
    query_feature_matrix,
    readonly_float_vector,
)


@dataclass(frozen=True)
class DirectProfileScorer:
    """One query-only OLS target-membership scorer per executable profile."""

    profiles: tuple[Profile, ...]
    model: LinearLeastSquares

    @classmethod
    def fit(
        cls,
        examples: Sequence[TrainingQuery],
        executable_profiles: Iterable[Profile],
    ) -> DirectProfileScorer:
        profiles = canonical_profiles(executable_profiles)
        matrix = query_feature_matrix(examples)
        profile_to_column = {
            profile: index for index, profile in enumerate(profiles)
        }
        targets = np.zeros((len(examples), len(profiles)), dtype=np.float64)
        for row, example in enumerate(examples):
            for profile in example.target_profiles:
                try:
                    targets[row, profile_to_column[profile]] = 1.0
                except KeyError as exc:
                    raise ValueError(
                        "target profiles must be a subset of executable profiles"
                    ) from exc
        return cls(profiles, LinearLeastSquares.fit(matrix, targets))

    def scores(self, features: QueryFeatures) -> FloatArray:
        return self.model.predict(features.vector())

    def select(self, features: QueryFeatures) -> Profile:
        scores = self.scores(features)
        best_index = 0
        for index in range(1, len(self.profiles)):
            if scores[index] > scores[best_index]:
                best_index = index
        return self.profiles[best_index]


@dataclass(frozen=True)
class IndependentGroupScorer:
    """Additive query-only per-group/per-bit OLS baseline."""

    profiles: tuple[Profile, ...]
    model: LinearLeastSquares
    num_groups: int

    @classmethod
    def fit(
        cls,
        examples: Sequence[TrainingQuery],
        executable_profiles: Iterable[Profile],
    ) -> IndependentGroupScorer:
        profiles = canonical_profiles(executable_profiles)
        num_groups = len(profiles[0])
        matrix = query_feature_matrix(examples)
        executable_set = set(profiles)
        targets = np.zeros((len(examples), num_groups * 2), dtype=np.float64)
        for row, example in enumerate(examples):
            for target_profile in example.target_profiles:
                if target_profile not in executable_set:
                    raise ValueError(
                        "target profiles must be a subset of executable profiles"
                    )
                for group_index, bit in enumerate(target_profile):
                    targets[row, group_index * 2 + (0 if bit == 4 else 1)] = 1.0
        return cls(
            profiles,
            LinearLeastSquares.fit(matrix, targets),
            num_groups,
        )

    def bit_scores(self, features: QueryFeatures) -> FloatArray:
        result = self.model.predict(features.vector()).reshape(self.num_groups, 2)
        result.flags.writeable = False
        return result

    def select(self, features: QueryFeatures) -> Profile:
        bit_scores = self.bit_scores(features)
        best_profile = self.profiles[0]
        best_score = _additive_profile_score(best_profile, bit_scores)
        for candidate in self.profiles[1:]:
            candidate_score = _additive_profile_score(candidate, bit_scores)
            if candidate_score > best_score:
                best_profile = candidate
                best_score = candidate_score
        return best_profile


def _additive_profile_score(profile: Profile, bit_scores: FloatArray) -> float:
    return float(
        sum(
            bit_scores[group_index, 0 if bit == 4 else 1]
            for group_index, bit in enumerate(profile)
        )
    )


def select_pareto_static_profile(
    feasible_profiles: Iterable[Profile],
    mean_costs: Mapping[Profile, Sequence[float]],
) -> Profile:
    """Select the canonical minimum from the Pareto-minimal cost frontier."""

    profiles = canonical_profiles(feasible_profiles)
    vectors: dict[Profile, FloatArray] = {}
    dimension: int | None = None
    for profile in profiles:
        try:
            vector = readonly_float_vector(
                mean_costs[profile], name="mean cost vector"
            )
        except KeyError as exc:
            identifier = profile_id(profile)
            raise ValueError(f"missing mean cost for profile {identifier}") from exc
        if vector.shape[0] == 0:
            raise ValueError("mean cost vectors must contain at least one dimension")
        if dimension is None:
            dimension = int(vector.shape[0])
        elif vector.shape[0] != dimension:
            raise ValueError("all mean cost vectors must share one shape")
        vectors[profile] = vector

    frontier: list[Profile] = []
    for candidate in profiles:
        dominated = any(
            bool(
                np.all(vectors[other] <= vectors[candidate])
                and np.any(vectors[other] < vectors[candidate])
            )
            for other in profiles
            if other != candidate
        )
        if not dominated:
            frontier.append(candidate)
    if not frontier:
        raise ValueError("Pareto frontier is unexpectedly empty")
    return min(frontier, key=profile_id)

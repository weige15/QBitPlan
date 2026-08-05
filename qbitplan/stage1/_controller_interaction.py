"""Causally valid sequential interaction-aware planner."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._controller_common import (
    Bit,
    FeatureArray,
    LinearLeastSquares,
    PrefixContextProvider,
    Profile,
    QueryFeatures,
    TrainingQuery,
    canonical_profiles,
    l2_normalize_feature,
    prefix_id,
    query_feature_matrix,
    validate_profile,
)


@dataclass(frozen=True)
class BitDecision:
    group_index: int
    prefix_before: Profile
    selected_bit: Bit
    score_4: float | None
    score_8: float | None
    executable_continuations: int


@dataclass(frozen=True)
class InteractionPlanResult:
    valid: bool
    profile: Profile | None
    decisions: tuple[BitDecision, ...]
    reason_code: str | None


@dataclass(frozen=True)
class CausalInteractionPlanner:
    """Sequential OLS planner using query, prefix bits, and causal context."""

    profiles: tuple[Profile, ...]
    models: tuple[LinearLeastSquares, ...]
    structural_dim: int
    embedding_dim: int
    num_groups: int

    @classmethod
    def fit(
        cls,
        examples: Sequence[TrainingQuery],
        executable_profiles: Iterable[Profile],
        context_provider: PrefixContextProvider,
    ) -> CausalInteractionPlanner:
        profiles = canonical_profiles(executable_profiles)
        query_feature_matrix(examples)
        structural_dim = examples[0].features.structural_dim
        embedding_dim = examples[0].features.embedding_dim
        num_groups = len(profiles[0])
        executable_set = set(profiles)
        models: list[LinearLeastSquares] = []

        for group_index in range(num_groups):
            rows: list[FeatureArray] = []
            targets: list[tuple[float, float]] = []
            for example in examples:
                grouped: dict[Profile, set[Bit]] = {}
                for target_profile in example.target_profiles:
                    if target_profile not in executable_set:
                        raise ValueError(
                            "target profiles must be a subset of executable profiles"
                        )
                    prefix = target_profile[:group_index]
                    grouped.setdefault(prefix, set()).add(
                        target_profile[group_index]
                    )
                for prefix, supported_bits in sorted(
                    grouped.items(), key=lambda item: prefix_id(item[0])
                ):
                    rows.append(
                        interaction_features(
                            example.query_id,
                            example.features,
                            prefix,
                            group_index,
                            context_provider,
                        )
                    )
                    targets.append(
                        (
                            1.0 if 4 in supported_bits else 0.0,
                            1.0 if 8 in supported_bits else 0.0,
                        )
                    )
            if not rows:
                raise ValueError(
                    f"group {group_index} has no supported interaction prefixes"
                )
            models.append(LinearLeastSquares.fit(np.stack(rows), targets))
        return cls(
            profiles,
            tuple(models),
            structural_dim,
            embedding_dim,
            num_groups,
        )

    def plan(
        self,
        query_id: str,
        features: QueryFeatures,
        context_provider: PrefixContextProvider,
    ) -> InteractionPlanResult:
        if features.structural_dim != self.structural_dim:
            raise ValueError("structural feature shape does not match planner")
        if features.embedding_dim != self.embedding_dim:
            raise ValueError("query embedding shape does not match planner")

        prefix: Profile = ()
        decisions: list[BitDecision] = []
        for group_index, model in enumerate(self.models):
            continuations = tuple(
                profile
                for profile in self.profiles
                if profile[:group_index] == prefix
            )
            if not continuations:
                return InteractionPlanResult(
                    False,
                    None,
                    tuple(decisions),
                    f"NO_EXECUTABLE_CONTINUATION_GROUP_{group_index}",
                )
            valid_bits = {profile[group_index] for profile in continuations}
            scores = model.predict(
                interaction_features(
                    query_id,
                    features,
                    prefix,
                    group_index,
                    context_provider,
                )
            )
            score_4 = float(scores[0]) if 4 in valid_bits else None
            score_8 = float(scores[1]) if 8 in valid_bits else None
            if valid_bits == {4, 8}:
                selected_bit = 8 if scores[1] > scores[0] else 4
            elif 4 in valid_bits:
                selected_bit = 4
            elif 8 in valid_bits:
                selected_bit = 8
            else:
                return InteractionPlanResult(
                    False,
                    None,
                    tuple(decisions),
                    f"NO_EXECUTABLE_BIT_GROUP_{group_index}",
                )
            decisions.append(
                BitDecision(
                    group_index,
                    prefix,
                    selected_bit,
                    score_4,
                    score_8,
                    len(continuations),
                )
            )
            prefix = (*prefix, selected_bit)

        if prefix not in set(self.profiles):
            return InteractionPlanResult(
                False,
                None,
                tuple(decisions),
                "SELECTED_PROFILE_NOT_EXECUTABLE",
            )
        return InteractionPlanResult(True, prefix, tuple(decisions), None)


def interaction_features(
    query_id: str,
    features: QueryFeatures,
    prefix: Profile,
    group_index: int,
    context_provider: PrefixContextProvider,
) -> FeatureArray:
    if len(prefix) != group_index:
        raise ValueError("interaction prefix length must equal group index")
    validate_profile(prefix, num_groups=group_index, allow_empty=True)
    if group_index == 0:
        hidden = np.zeros(features.embedding_dim, dtype=np.float32)
    else:
        hidden = l2_normalize_feature(
            context_provider(query_id, prefix),
            name=f"causal hidden context for {query_id}/{prefix_id(prefix)}",
        )
        if hidden.shape[0] != features.embedding_dim:
            raise ValueError(
                "causal hidden context and query embedding must have equal size"
            )
    prefix_one_hot = np.zeros(group_index * 2, dtype=np.float32)
    for prefix_index, bit in enumerate(prefix):
        prefix_one_hot[prefix_index * 2 + (0 if bit == 4 else 1)] = 1.0
    result = np.concatenate(
        (
            features.structural,
            features.embedding,
            hidden,
            features.embedding * hidden,
            prefix_one_hot,
        )
    )
    result.flags.writeable = False
    return result

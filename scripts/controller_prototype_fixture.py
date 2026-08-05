"""Deterministic synthetic fixture for the controller prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from qbitplan.stage1.controller import (
    Profile,
    QueryFeatures,
    StructuralNormalizer,
    TrainingQuery,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _RawQuery:
    query_id: str
    structural: FloatArray
    embedding: FloatArray
    target_profile: Profile


class _SyntheticPrefixContext:
    """Causal context depending only on the actually executed prefix."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def __call__(self, query_id: str, prefix: Profile) -> FloatArray:
        del query_id
        if not prefix:
            return np.zeros(self.embedding_dim, dtype=np.float64)
        sign = 1.0 if prefix[-1] == 8 else -1.0
        return np.full(self.embedding_dim, sign, dtype=np.float64)


def _target_profile(embedding_signs: tuple[int, ...]) -> Profile:
    bits: list[int] = []
    for group_index, query_sign in enumerate(embedding_signs):
        if group_index == 0:
            decision_sign = query_sign
        else:
            prefix_sign = 1 if bits[-1] == 8 else -1
            decision_sign = query_sign * prefix_sign
        bits.append(8 if decision_sign > 0 else 4)
    return tuple(bits)


def _raw_structural_features(
    generator: np.random.Generator,
    embedding_signs: tuple[int, ...],
) -> FloatArray:
    token_count = int(generator.integers(32, 513))
    character_count = token_count * int(generator.integers(3, 8))
    line_count = int(generator.integers(1, 17))
    digit_count = int(generator.integers(0, max(2, token_count // 4)))
    whitespace_count = max(1, token_count - 1 + int(generator.integers(-4, 5)))
    punctuation_count = int(generator.integers(0, max(2, token_count // 5)))
    option_count = 0 if embedding_signs[0] < 0 else int(generator.integers(0, 11))
    return np.asarray(
        (
            token_count,
            character_count,
            line_count,
            digit_count,
            whitespace_count,
            punctuation_count,
            option_count,
        ),
        dtype=np.float64,
    )


def _generate_raw_queries(
    *,
    generator: np.random.Generator,
    repeats: int,
    split: str,
    embedding_dim: int,
) -> list[_RawQuery]:
    queries: list[_RawQuery] = []
    patterns = tuple(product((-1, 1), repeat=embedding_dim))
    for repeat_index in range(repeats):
        shuffled = list(patterns)
        generator.shuffle(shuffled)
        for pattern_index, signs in enumerate(shuffled):
            embedding = np.asarray(signs, dtype=np.float64)
            queries.append(
                _RawQuery(
                    query_id=(
                        f"synthetic-{split}-{repeat_index:02d}-{pattern_index:03d}"
                    ),
                    structural=_raw_structural_features(generator, signs),
                    embedding=embedding,
                    target_profile=_target_profile(signs),
                )
            )
    return queries


def _normalize_queries(
    raw_queries: list[_RawQuery],
    normalizer: StructuralNormalizer,
) -> list[TrainingQuery]:
    return [
        TrainingQuery(
            query_id=query.query_id,
            features=QueryFeatures(
                structural=normalizer.transform(query.structural),
                embedding=query.embedding,
            ),
            target_profiles=(query.target_profile,),
        )
        for query in raw_queries
    ]


def _analytical_costs(profiles: tuple[Profile, ...]) -> dict[Profile, FloatArray]:
    costs: dict[Profile, FloatArray] = {}
    for profile in profiles:
        promoted_groups = sum(bit == 8 for bit in profile)
        costs[profile] = np.asarray(
            (
                1.0 + promoted_groups,
                2.0 + promoted_groups * 1.5,
                3.0 + promoted_groups * 0.75,
                4.0 + promoted_groups * 0.5,
                5.0 + promoted_groups * 0.25,
            ),
            dtype=np.float64,
        )
    return costs

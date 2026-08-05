"""Paired query-level bootstrap for controller comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    point_estimate: float
    lower_95: float
    upper_95: float
    replicates: int
    seed: int


def paired_bootstrap_difference(
    method_hits: Sequence[bool | int | float],
    baseline_hits: Sequence[bool | int | float],
    *,
    seed: int,
    replicates: int = 10_000,
    batch_size: int = 256,
) -> PairedBootstrapResult:
    """Paired percentile bootstrap for a hit-rate difference."""

    method = np.asarray(method_hits, dtype=np.float64)
    baseline = np.asarray(baseline_hits, dtype=np.float64)
    if method.ndim != 1 or baseline.ndim != 1:
        raise ValueError("paired hit arrays must be one-dimensional")
    if method.shape != baseline.shape or method.shape[0] == 0:
        raise ValueError("paired hit arrays must have equal non-zero length")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    if not np.all(np.isfinite(method)) or not np.all(np.isfinite(baseline)):
        raise ValueError("paired hit arrays must be finite")
    differences = method - baseline
    generator = np.random.Generator(np.random.PCG64(seed))
    replicate_means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = generator.integers(
            0,
            differences.shape[0],
            size=(stop - start, differences.shape[0]),
        )
        replicate_means[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(
        replicate_means, [0.025, 0.975], method="linear"
    )
    return PairedBootstrapResult(
        point_estimate=float(differences.mean()),
        lower_95=float(lower),
        upper_95=float(upper),
        replicates=replicates,
        seed=seed,
    )

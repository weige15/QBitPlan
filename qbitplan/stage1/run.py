"""High-level Stage-1 plan-to-artifact execution seam."""

from __future__ import annotations

from typing import Protocol

from .artifacts import ArtifactBundle
from .contract import ExperimentPlan, ProfileExecutionResult


class ProfileExecutor(Protocol):
    evidence_class: str

    def execute(self, profile_id: str, query: dict[str, str]) -> ProfileExecutionResult:  # pyright: ignore[reportReturnType]
        """Execute one complete profile/query path and return its status."""


def execute_plan(plan: ExperimentPlan, executor: ProfileExecutor) -> ArtifactBundle:
    """Execute the explicit smoke plan once through the supplied public adapter."""

    if plan.mode != "smoke":
        raise ValueError("issue #25 only supports smoke mode")
    results: list[ProfileExecutionResult] = []
    for profile_id in plan.profile_ids:
        for query in plan.queries:
            result = executor.execute(profile_id, query)
            if not isinstance(result, ProfileExecutionResult):
                raise TypeError("profile executors must return ProfileExecutionResult")
            if result.profile_id != profile_id or result.query_id != query["query_id"]:
                raise ValueError(
                    "profile executor returned a result for the wrong profile/query"
                )
            results.append(result)
    return ArtifactBundle.write(plan, results)

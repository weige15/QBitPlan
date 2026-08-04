from __future__ import annotations

from pathlib import Path
from typing import Any

from test_functional_quality import FakeQualityExecutor, _quality_plan

from qbitplan import execute_plan


class PreparedFakeExecutor(FakeQualityExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.prepared: list[str] = []
        self.released: list[str] = []
        self.prepared_calls: list[str] = []

    def prepare_variant(self, variant_id: str) -> dict[str, Any]:
        self.prepared.append(variant_id)
        return {
            "status": "complete",
            "reason_code": "PREPARED",
            "transform_status": "not_applicable"
            if variant_id == "BF16"
            else "complete",
            "forward_status": "not_attempted",
        }

    def execute_prepared(
        self, query: dict[str, Any], variant_id: str
    ) -> dict[str, Any]:
        self.prepared_calls.append(variant_id)
        return super().execute(query, variant_id)

    def release_variant(self, variant_id: str) -> None:
        self.released.append(variant_id)


def test_public_seam_reuses_prepared_profiles_and_diagnostic_order(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _quality_plan(tmp_path / "artifacts", monkeypatch)
    executor = PreparedFakeExecutor()

    execute_plan(plan, executor=executor)

    profiles = [f"{profile_id:08b}" for profile_id in range(256)]
    assert executor.prepared == profiles
    assert executor.released == profiles
    assert len(executor.prepared_calls) == 2 * 256
    assert len(executor.diagnostic_calls) == 2 * 257
    diagnostic_variants = [variant for variant, _ in executor.diagnostic_calls]
    assert diagnostic_variants[:2] == ["BF16", "BF16"]
    assert diagnostic_variants[2:4] == [profiles[0], profiles[0]]
    assert diagnostic_variants[-2:] == [profiles[-1], profiles[-1]]

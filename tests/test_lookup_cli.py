from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_experiment_seam import _patch_synthetic_manifest_constants
from test_lookup_cost import _estimate_plan, _lookup_table

from qbitplan.cli import main


def test_estimate_cost_cli_uses_lookup_adapter_without_gpu(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_synthetic_manifest_constants(monkeypatch)
    plan_path = tmp_path / "estimate-plan.json"
    plan_path.write_text(
        json.dumps(_estimate_plan(tmp_path / "artifacts")), encoding="utf-8"
    )
    table_path = tmp_path / "lookup-table.json"
    table_path.write_text(json.dumps(_lookup_table()), encoding="utf-8")

    result = main(
        [
            "stage1",
            "estimate-cost",
            "--plan",
            str(plan_path),
            "--lookup",
            str(table_path),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    bundle = json.loads(
        (Path(output["bundle_path"]) / "bundle.json").read_text(encoding="utf-8")
    )
    assert bundle["evidence_class"] == "lookup-table estimated"
    assert bundle["hardware_identity"]["identity_status"] == "not_applicable"

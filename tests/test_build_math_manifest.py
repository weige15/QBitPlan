from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_math_manifest import canonical_json_bytes, main


def _write_source_record(root: Path, source_id: str, problem: str) -> None:
    path = root / source_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "problem": problem,
                "solution": f"solution for {problem}",
                "level": "Level 1",
                "type": "Algebra",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_canonical_json_uses_sorted_utf8_json_identity() -> None:
    assert canonical_json_bytes({"b": 2, "a": "é"}) == '{"a":"é","b":2}'.encode()


def test_build_command_writes_separated_manifest_and_smoke_plan(tmp_path: Path) -> None:
    source_root = tmp_path / "MATH"
    math500_root = tmp_path / "math-500"
    math500_root.mkdir()

    for index in range(7500):
        source_id = f"train/algebra/{index}.json"
        _write_source_record(source_root, source_id, f"train-{index}")

    final_rows = []
    for index in range(5000):
        source_id = f"test/algebra/{index}.json"
        problem = f"test-{index}"
        _write_source_record(source_root, source_id, problem)
        if index < 500:
            final_rows.append(
                {
                    "problem": problem,
                    "solution": f"solution for {problem}",
                    "answer": "1",
                    "subject": "Algebra",
                    "level": 1,
                    "unique_id": source_id,
                }
            )

    (math500_root / "test.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.json"
    smoke_path = tmp_path / "smoke-plan.json"
    assert (
        main(
            [
                "--source-root",
                str(source_root),
                "--math500-root",
                str(math500_root),
                "--manifest-output",
                str(manifest_path),
                "--smoke-plan-output",
                str(smoke_path),
            ]
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["permitted_source_ids"]["training"]) == 7500
    assert len(manifest["permitted_source_ids"]["validation"]) == 4500
    assert len(manifest["final_source_ids"]) == 500
    assert not (
        set(manifest["permitted_source_ids"]["validation"])
        & set(manifest["final_source_ids"])
    )

    expected_train_hash = hashlib.sha256(
        b'{"level":"Level 1","problem":"train-0","solution":"solution for train-0","type":"Algebra"}'
    ).hexdigest()
    assert (
        manifest["record_hashes"]["training"]["train/algebra/0.json"]
        == expected_train_hash
    )

    smoke_plan = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert smoke_plan["manifest_artifact_id"] == manifest["artifact_id"]
    assert {query["phase"] for query in smoke_plan["queries"]} == {
        "training",
        "validation",
    }
    assert all(
        query["source_artifact_id"] == manifest["artifact_id"]
        for query in smoke_plan["queries"]
    )
    assert all(query["permitted"] is True for query in smoke_plan["queries"])
    assert all("Problem:\n" in query["prompt"] for query in smoke_plan["queries"])


def test_build_command_rejects_missing_raw_source_tree(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "--source-root",
            str(tmp_path / "missing"),
            "--math500-root",
            str(tmp_path / "math-500"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
            "--smoke-plan-output",
            str(tmp_path / "smoke-plan.json"),
        ]
    )

    assert result == 2
    assert "source root does not exist" in capsys.readouterr().err

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from qbitplan.stage1.controller import (
    CausalInteractionPlanner,
    DirectProfileScorer,
    IndependentGroupScorer,
    QueryFeatures,
    TrainingQuery,
    enumerate_profiles,
    paired_bootstrap_difference,
    profile_from_id,
    profile_id,
    select_pareto_static_profile,
)
from scripts.run_controller_prototype import run


def _features(value: float) -> QueryFeatures:
    return QueryFeatures(
        structural=np.asarray([0.0, 0.0], dtype=np.float32),
        embedding=np.asarray([value, 0.0], dtype=np.float32),
    )


def test_profile_ids_and_pareto_static_selection_are_canonical() -> None:
    profiles = enumerate_profiles(2)

    assert tuple(profile_id(profile) for profile in profiles) == (
        "00",
        "01",
        "10",
        "11",
    )
    assert profile_from_id("10") == (8, 4)

    costs = {
        (4, 4): (2.0, 4.0),
        (4, 8): (1.0, 3.0),
        (8, 4): (3.0, 1.0),
        (8, 8): (4.0, 4.0),
    }
    assert select_pareto_static_profile(profiles, costs) == (4, 8)


def test_independent_scorer_respects_executable_profiles_and_canonical_ties() -> None:
    executable = ((8, 8), (4, 8))
    examples = (
        TrainingQuery("empty-0", _features(-1.0), ()),
        TrainingQuery("empty-1", _features(1.0), ()),
    )

    scorer = IndependentGroupScorer.fit(examples, executable)

    assert scorer.select(_features(0.0)) == (4, 8)
    assert scorer.select(_features(0.0)) in executable


def test_direct_scorer_uses_canonical_profile_for_equal_scores() -> None:
    scorer = DirectProfileScorer.fit(
        (
            TrainingQuery(
                "direct-tie",
                _features(0.0),
                ((4, 4), (4, 8)),
            ),
        ),
        ((4, 4), (4, 8)),
    )

    assert scorer.select(_features(0.0)) == (4, 4)


def test_interaction_score_equality_selects_4_bit() -> None:
    planner = CausalInteractionPlanner.fit(
        (
            TrainingQuery(
                "interaction-tie",
                _features(0.0),
                ((4,), (8,)),
            ),
        ),
        enumerate_profiles(1),
        lambda _query_id, _prefix: np.zeros(2, dtype=np.float64),
    )

    result = planner.plan(
        "interaction-tie-runtime",
        _features(0.0),
        lambda _query_id, _prefix: np.zeros(2, dtype=np.float64),
    )

    assert result.valid is True
    assert result.profile == (4,)
    assert result.decisions[0].score_4 == result.decisions[0].score_8


def test_interaction_planner_returns_explicit_invalid_continuation_result() -> None:
    planner = CausalInteractionPlanner.fit(
        (TrainingQuery("invalid", _features(0.0), ((4,),)),),
        enumerate_profiles(1),
        lambda _query_id, _prefix: np.zeros(2, dtype=np.float64),
    )
    invalid_planner = replace(planner, profiles=())

    result = invalid_planner.plan(
        "invalid-runtime",
        _features(0.0),
        lambda _query_id, _prefix: np.zeros(2, dtype=np.float64),
    )

    assert result.valid is False
    assert result.profile is None
    assert result.reason_code == "NO_EXECUTABLE_CONTINUATION_GROUP_0"


def test_interaction_planner_uses_only_the_actually_selected_prefix() -> None:
    executable = enumerate_profiles(2)
    training = (
        TrainingQuery("train-neg", _features(-1.0), ((4, 8),)),
        TrainingQuery("train-pos", _features(1.0), ((8, 4),)),
    )

    def training_context(query_id: str, prefix: tuple[int, ...]) -> np.ndarray:
        del query_id
        return np.asarray([-1.0 if prefix[-1] == 4 else 1.0, 0.0])

    planner = CausalInteractionPlanner.fit(
        training,
        executable,
        training_context,
    )
    observed_prefixes: list[tuple[int, ...]] = []

    def runtime_context(query_id: str, prefix: tuple[int, ...]) -> np.ndarray:
        assert query_id == "runtime"
        observed_prefixes.append(prefix)
        return np.asarray([-1.0 if prefix[-1] == 4 else 1.0, 0.0])

    result = planner.plan("runtime", _features(1.0), runtime_context)

    assert result.valid is True
    assert result.profile == (8, 4)
    assert observed_prefixes == [(8,)]
    assert tuple(decision.prefix_before for decision in result.decisions) == (
        (),
        (8,),
    )


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    method = [1, 1, 0, 1, 0, 1]
    baseline = [0, 1, 0, 0, 0, 1]

    first = paired_bootstrap_difference(
        method,
        baseline,
        seed=20260805,
        replicates=500,
    )
    second = paired_bootstrap_difference(
        method,
        baseline,
        seed=20260805,
        replicates=500,
    )

    assert first == second
    assert first.point_estimate == 2 / 6
    assert first.lower_95 <= first.point_estimate <= first.upper_95


def test_synthetic_prototype_emits_labeled_plot_and_metrics(tmp_path: Path) -> None:
    metrics = run(output_dir=tmp_path, seed=20260805, source_git_sha="a" * 40)

    assert metrics["evidence_class"] == "simulated"
    assert metrics["evidentiary_status"] == "non-evidentiary"
    assert (
        metrics["methods"]["interaction"]["synthetic_target_profile_match_rate"]
        == 1.0
    )
    assert (
        metrics["methods"]["interaction"]["synthetic_target_profile_match_rate"]
        > metrics["methods"]["independent"]["synthetic_target_profile_match_rate"]
    )
    assert (
        metrics["methods"]["direct"]["synthetic_target_profile_match_rate"]
        > metrics["methods"]["static"]["synthetic_target_profile_match_rate"]
    )
    assert metrics["schema_version"] == "qbitplan.controller-prototype.v2"
    assert metrics["provenance"]["source_git_sha"] == "a" * 40
    assert metrics["provenance"]["hardware"]["identity_status"] == "unavailable"
    assert metrics["provenance"]["query_id_manifest"]["path"] == (
        "query-manifest.json"
    )
    assert metrics["provenance"]["query_id_manifest"]["sha256"]
    assert "feasible_profile_hit_rate" not in json.dumps(metrics)
    predictions_path = tmp_path / "predictions.csv"
    assert predictions_path.is_file()
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows
    assert {row["evidence_class"] for row in prediction_rows} == {"simulated"}
    assert {row["evidentiary_status"] for row in prediction_rows} == {
        "non-evidentiary"
    }
    assert {row["schema_version"] for row in prediction_rows} == {
        "qbitplan.controller-prototype.prediction.v2"
    }
    assert "direct_target_profile_match" in prediction_rows[0]
    assert all(not key.endswith("_hit") for key in prediction_rows[0])
    assert all(
        "not a Stage-1" in row["claim_boundary"] for row in prediction_rows
    )
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "query-manifest.json").is_file()
    svg = (tmp_path / "controller-prototype.svg").read_text(encoding="utf-8")
    assert "simulated, non-evidentiary" in svg
    assert "Synthetic target-profile match rate" in svg
    assert "No LLM, GPU, quantized weight" in svg


def test_query_manifest_hash_and_outputs_are_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run(output_dir=first_dir, seed=20260805, source_git_sha="a" * 40)
    second = run(output_dir=second_dir, seed=20260805, source_git_sha="a" * 40)

    first_manifest = (first_dir / "query-manifest.json").read_bytes()
    second_manifest = (second_dir / "query-manifest.json").read_bytes()
    assert first_manifest == second_manifest
    assert first["provenance"]["query_id_manifest"] == second["provenance"][
        "query_id_manifest"
    ]
    assert first["provenance"]["query_id_manifest"]["sha256"]
    assert (first_dir / "metrics.json").read_bytes() == (
        second_dir / "metrics.json"
    ).read_bytes()
    assert (first_dir / "predictions.csv").read_bytes() == (
        second_dir / "predictions.csv"
    ).read_bytes()
    assert (first_dir / "controller-prototype.svg").read_bytes() == (
        second_dir / "controller-prototype.svg"
    ).read_bytes()


def test_cli_emits_simulated_non_evidentiary_labels(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_controller_prototype",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "20260805",
            "--source-git-sha",
            "a" * 40,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["evidence_class"] == "simulated"
    assert output["evidentiary_status"] == "non-evidentiary"
    assert "not a Stage-1" in output["claim_boundary"]


def test_every_generated_profile_is_executable(tmp_path: Path) -> None:
    run(output_dir=tmp_path, seed=20260805, source_git_sha="a" * 40)
    profiles = set(enumerate_profiles(8))
    with (tmp_path / "predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for method in ("static", "direct", "independent", "interaction"):
            assert tuple(
                4 if bit == "0" else 8
                for bit in row[f"{method}_profile_id"]
            ) in profiles

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qbitplan.stage1.controller import (
    CausalInteractionPlanner,
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
    metrics = run(output_dir=tmp_path, seed=20260805)

    assert metrics["evidence_class"] == "simulated"
    assert metrics["methods"]["interaction"]["feasible_profile_hit_rate"] == 1.0
    assert (
        metrics["methods"]["interaction"]["feasible_profile_hit_rate"]
        > metrics["methods"]["independent"]["feasible_profile_hit_rate"]
    )
    assert (
        metrics["methods"]["direct"]["feasible_profile_hit_rate"]
        > metrics["methods"]["static"]["feasible_profile_hit_rate"]
    )
    predictions_path = tmp_path / "predictions.csv"
    assert predictions_path.is_file()
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows
    assert {row["evidence_class"] for row in prediction_rows} == {"simulated"}
    assert {row["schema_version"] for row in prediction_rows} == {
        "qbitplan.controller-prototype.prediction.v1"
    }
    assert all(
        "not a Stage-1" in row["claim_boundary"] for row in prediction_rows
    )
    assert (tmp_path / "metrics.json").is_file()
    svg = (tmp_path / "controller-prototype.svg").read_text(encoding="utf-8")
    assert "simulated, non-evidentiary" in svg
    assert "No LLM, GPU, quantized weight" in svg

"""Run the deterministic, non-evidentiary QBitPlan controller prototype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from qbitplan.stage1.controller import (
    CausalInteractionPlanner,
    DirectProfileScorer,
    IndependentGroupScorer,
    Profile,
    StructuralNormalizer,
    TrainingQuery,
    enumerate_profiles,
    paired_bootstrap_difference,
    profile_id,
    select_pareto_static_profile,
)
from scripts.controller_prototype_fixture import (
    _SyntheticPrefixContext,
    _analytical_costs,
    _generate_raw_queries,
    _normalize_queries,
)
from scripts.controller_prototype_plot import _write_svg


_EVIDENCE_CLASS = "simulated"
_CLAIM_BOUNDARY = (
    "Deterministic synthetic engineering fixture only. This is not a Stage-1 "
    "signal, interaction, oracle, quality, cost, memory, latency, or serving "
    "result."
)
_PREDICTION_SCHEMA_VERSION = "qbitplan.controller-prototype.prediction.v1"


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values))


def _evaluate(
    *,
    validation_queries: list[TrainingQuery],
    static_profile: Profile,
    direct: DirectProfileScorer,
    independent: IndependentGroupScorer,
    interaction: CausalInteractionPlanner,
    context_provider: _SyntheticPrefixContext,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = ("static", "direct", "independent", "interaction")
    hits: dict[str, list[int]] = {method: [] for method in methods}
    group_hits: dict[str, list[list[int]]] = {
        method: [[] for _ in range(interaction.num_groups)] for method in methods
    }
    selected_profiles: dict[str, list[Profile]] = {method: [] for method in methods}

    for query in validation_queries:
        target = query.target_profiles[0]
        direct_profile = direct.select(query.features)
        independent_profile = independent.select(query.features)
        interaction_result = interaction.plan(
            query.query_id,
            query.features,
            context_provider,
        )
        if not interaction_result.valid or interaction_result.profile is None:
            raise RuntimeError(
                f"interaction fixture invalid: {interaction_result.reason_code}"
            )
        predictions = {
            "static": static_profile,
            "direct": direct_profile,
            "independent": independent_profile,
            "interaction": interaction_result.profile,
        }
        row: dict[str, Any] = {
            "schema_version": _PREDICTION_SCHEMA_VERSION,
            "evidence_class": _EVIDENCE_CLASS,
            "claim_boundary": _CLAIM_BOUNDARY,
            "query_id": query.query_id,
            "target_profile_id": profile_id(target),
        }
        for method, selected in predictions.items():
            selected_profiles[method].append(selected)
            hit = int(selected in query.target_profiles)
            hits[method].append(hit)
            row[f"{method}_profile_id"] = profile_id(selected)
            row[f"{method}_hit"] = hit
            for group_index, (selected_bit, target_bit) in enumerate(
                zip(selected, target, strict=True)
            ):
                group_hits[method][group_index].append(
                    int(selected_bit == target_bit)
                )
        rows.append(row)

    method_metrics: dict[str, Any] = {}
    for method in methods:
        method_metrics[method] = {
            "feasible_profile_hit_rate": _mean(hits[method]),
            "mean_selected_bit_width": float(
                np.mean(
                    [bit for profile in selected_profiles[method] for bit in profile]
                )
            ),
            "per_group_bit_accuracy": [
                _mean(group_values) for group_values in group_hits[method]
            ],
        }

    direct_vs_static = paired_bootstrap_difference(
        hits["direct"],
        hits["static"],
        seed=20260804,
    )
    interaction_vs_independent = paired_bootstrap_difference(
        hits["interaction"],
        hits["independent"],
        seed=20260805,
    )
    comparisons = {
        "direct_minus_static": {
            "point_estimate": direct_vs_static.point_estimate,
            "lower_95": direct_vs_static.lower_95,
            "upper_95": direct_vs_static.upper_95,
            "replicates": direct_vs_static.replicates,
            "seed": direct_vs_static.seed,
        },
        "interaction_minus_independent": {
            "point_estimate": interaction_vs_independent.point_estimate,
            "lower_95": interaction_vs_independent.lower_95,
            "upper_95": interaction_vs_independent.upper_95,
            "replicates": interaction_vs_independent.replicates,
            "seed": interaction_vs_independent.seed,
        },
    }
    return rows, {"methods": method_metrics, "paired_comparisons": comparisons}


def _write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(*, output_dir: Path, seed: int) -> dict[str, Any]:
    generator = np.random.Generator(np.random.PCG64(seed))
    embedding_dim = 8
    raw_training = _generate_raw_queries(
        generator=generator,
        repeats=8,
        split="train",
        embedding_dim=embedding_dim,
    )
    raw_validation = _generate_raw_queries(
        generator=generator,
        repeats=4,
        split="validation",
        embedding_dim=embedding_dim,
    )
    normalizer = StructuralNormalizer.fit(
        [query.structural for query in raw_training]
    )
    training_queries = _normalize_queries(raw_training, normalizer)
    validation_queries = _normalize_queries(raw_validation, normalizer)
    profiles = enumerate_profiles(embedding_dim)
    context_provider = _SyntheticPrefixContext(embedding_dim)

    static_profile = select_pareto_static_profile(
        profiles,
        _analytical_costs(profiles),
    )
    direct = DirectProfileScorer.fit(training_queries, profiles)
    independent = IndependentGroupScorer.fit(training_queries, profiles)
    interaction = CausalInteractionPlanner.fit(
        training_queries,
        profiles,
        context_provider,
    )
    rows, evaluation = _evaluate(
        validation_queries=validation_queries,
        static_profile=static_profile,
        direct=direct,
        independent=independent,
        interaction=interaction,
        context_provider=context_provider,
    )

    metrics: dict[str, Any] = {
        "schema_version": "qbitplan.controller-prototype.v1",
        "evidence_class": _EVIDENCE_CLASS,
        "claim_boundary": _CLAIM_BOUNDARY,
        "dataset_seed": seed,
        "bootstrap_seeds": {
            "direct_minus_static": 20260804,
            "interaction_minus_independent": 20260805,
        },
        "training_queries": len(training_queries),
        "validation_queries": len(validation_queries),
        "executable_profiles": len(profiles),
        "num_groups": embedding_dim,
        "structural_dim": training_queries[0].features.structural_dim,
        "embedding_dim": embedding_dim,
        "static_profile_id": profile_id(static_profile),
        "training_mse": {
            "direct": direct.model.training_mse,
            "independent": independent.model.training_mse,
            "interaction_by_group": [
                model.training_mse for model in interaction.models
            ],
        },
        **evaluation,
    }
    _write_predictions(rows, output_dir / "predictions.csv")
    _write_json(metrics, output_dir / "metrics.json")
    _write_svg(metrics, output_dir / "controller-prototype.svg")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/controller-prototype"),
    )
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = run(output_dir=args.output_dir, seed=args.seed)
    summary = {
        method: round(
            values["feasible_profile_hit_rate"],
            6,
        )
        for method, values in metrics["methods"].items()
    }
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "evidence_class": _EVIDENCE_CLASS,
                "claim_boundary": "non-evidentiary",
                "hit_rates": summary,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

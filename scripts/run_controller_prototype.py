"""Run the deterministic, non-evidentiary QBitPlan controller prototype."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
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
_EVIDENTIARY_STATUS = "non-evidentiary"
_CLAIM_BOUNDARY = (
    "Deterministic synthetic engineering fixture only. This is not a Stage-1 "
    "signal, interaction, oracle, quality, cost, memory, latency, or serving "
    "result. Synthetic target-profile equality/membership is not Stage-1 "
    "feasible-set membership plus external task correctness."
)
_PREDICTION_SCHEMA_VERSION = "qbitplan.controller-prototype.prediction.v2"
_METRICS_SCHEMA_VERSION = "qbitplan.controller-prototype.v2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BootstrapSeedConfiguration:
    direct_minus_static: int = 20260804
    interaction_minus_independent: int = 20260805


@dataclass(frozen=True)
class SyntheticFixtureConfiguration:
    model_identifier: str = "qbitplan.synthetic-controller-model"
    model_revision: str = "target-profile-recurrence-v1"
    dataset_identifier: str = "qbitplan.synthetic-controller-dataset"
    dataset_revision: str = "sign-patterns-v1"
    num_groups: int = 8
    embedding_dim: int = 8
    train_repeats: int = 8
    validation_repeats: int = 4
    profile_bits: tuple[int, int] = (4, 8)
    target_definition: str = (
        "One constructed synthetic target profile per query; matching is exact "
        "profile equality/membership."
    )


@dataclass(frozen=True)
class ControllerPrototypeRunConfiguration:
    source_git_sha: str
    seed: int
    fixture: SyntheticFixtureConfiguration = field(
        default_factory=SyntheticFixtureConfiguration
    )
    bootstrap_seeds: BootstrapSeedConfiguration = field(
        default_factory=BootstrapSeedConfiguration
    )
    quantization_mode: str = "not-executed/synthetic-profile-labels-only"
    backend_mode: str = "numpy-ols/synthetic-fixture-only"
    query_manifest_filename: str = "query-manifest.json"
    raw_artifact_location: str = "."
    prediction_schema_version: str = _PREDICTION_SCHEMA_VERSION
    metrics_schema_version: str = _METRICS_SCHEMA_VERSION
    evidence_class: str = _EVIDENCE_CLASS
    evidentiary_status: str = _EVIDENTIARY_STATUS
    claim_boundary: str = _CLAIM_BOUNDARY


def _validate_source_git_sha(source_git_sha: str) -> str:
    if not _SHA256_PATTERN.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be a full 40-character lowercase git SHA")
    return source_git_sha


def _resolve_source_git_sha(source_git_sha: str | None) -> str:
    if source_git_sha is not None:
        return _validate_source_git_sha(source_git_sha)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "source git SHA is required and could not be resolved from the repository"
        ) from exc
    return _validate_source_git_sha(resolved)


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"raw artifact location is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"raw artifact directory is not empty; choose a unique run directory: {path}"
            )
    else:
        path.mkdir(parents=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_query_manifest(
    *,
    training_queries: list[Any],
    validation_queries: list[Any],
    config: ControllerPrototypeRunConfiguration,
    path: Path,
) -> str:
    payload = {
        "schema_version": "qbitplan.controller-prototype.query-manifest.v1",
        "evidence_class": config.evidence_class,
        "evidentiary_status": config.evidentiary_status,
        "model": {
            "identifier": config.fixture.model_identifier,
            "revision": config.fixture.model_revision,
        },
        "dataset": {
            "identifier": config.fixture.dataset_identifier,
            "revision": config.fixture.dataset_revision,
        },
        "train_query_ids": [query.query_id for query in training_queries],
        "validation_query_ids": [query.query_id for query in validation_queries],
    }
    _write_json(payload, path)
    return _sha256(path)


def _provenance(
    *,
    config: ControllerPrototypeRunConfiguration,
    query_manifest_sha256: str,
) -> dict[str, Any]:
    fixture = config.fixture
    return {
        "source_git_sha": config.source_git_sha,
        "configuration": asdict(config),
        "random_seeds": {
            "dataset": config.seed,
            "bootstrap": asdict(config.bootstrap_seeds),
        },
        "model": {
            "identifier": fixture.model_identifier,
            "revision": fixture.model_revision,
        },
        "dataset": {
            "identifier": fixture.dataset_identifier,
            "revision": fixture.dataset_revision,
        },
        "query_id_manifest": {
            "path": config.query_manifest_filename,
            "sha256": query_manifest_sha256,
        },
        "quantization_mode": config.quantization_mode,
        "backend_mode": config.backend_mode,
        "software": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "hardware": {
            "identity_status": "unavailable",
            "accelerator": "unavailable",
            "device_name": "unavailable",
            "device_uuid": "unavailable",
            "driver_version": "unavailable",
        },
        "raw_artifact_location": config.raw_artifact_location,
    }


def _evaluate(
    *,
    validation_queries: list[TrainingQuery],
    static_profile: Profile,
    direct: DirectProfileScorer,
    independent: IndependentGroupScorer,
    interaction: CausalInteractionPlanner,
    context_provider: _SyntheticPrefixContext,
    config: ControllerPrototypeRunConfiguration,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = ("static", "direct", "independent", "interaction")
    matches: dict[str, list[int]] = {method: [] for method in methods}
    group_matches: dict[str, list[list[int]]] = {
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
            "schema_version": config.prediction_schema_version,
            "evidence_class": config.evidence_class,
            "evidentiary_status": config.evidentiary_status,
            "claim_boundary": config.claim_boundary,
            "query_id": query.query_id,
            "target_profile_id": profile_id(target),
        }
        for method, selected in predictions.items():
            selected_profiles[method].append(selected)
            match = int(selected in query.target_profiles)
            matches[method].append(match)
            row[f"{method}_profile_id"] = profile_id(selected)
            row[f"{method}_target_profile_match"] = match
            for group_index, (selected_bit, target_bit) in enumerate(
                zip(selected, target, strict=True)
            ):
                group_matches[method][group_index].append(
                    int(selected_bit == target_bit)
                )
        rows.append(row)

    method_metrics: dict[str, Any] = {}
    for method in methods:
        method_metrics[method] = {
            "synthetic_target_profile_match_rate": _mean(matches[method]),
            "mean_selected_bit_width": float(
                np.mean(
                    [bit for profile in selected_profiles[method] for bit in profile]
                )
            ),
            "per_group_bit_accuracy": [
                _mean(group_values) for group_values in group_matches[method]
            ],
        }

    direct_vs_static = paired_bootstrap_difference(
        matches["direct"],
        matches["static"],
        seed=config.bootstrap_seeds.direct_minus_static,
    )
    interaction_vs_independent = paired_bootstrap_difference(
        matches["interaction"],
        matches["independent"],
        seed=config.bootstrap_seeds.interaction_minus_independent,
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    output_dir: Path,
    seed: int,
    source_git_sha: str | None = None,
) -> dict[str, Any]:
    config = ControllerPrototypeRunConfiguration(
        source_git_sha=_resolve_source_git_sha(source_git_sha),
        seed=seed,
    )
    _prepare_output_dir(output_dir)
    generator = np.random.Generator(np.random.PCG64(config.seed))
    fixture = config.fixture
    raw_training = _generate_raw_queries(
        generator=generator,
        repeats=fixture.train_repeats,
        split="train",
        embedding_dim=fixture.embedding_dim,
    )
    raw_validation = _generate_raw_queries(
        generator=generator,
        repeats=fixture.validation_repeats,
        split="validation",
        embedding_dim=fixture.embedding_dim,
    )
    normalizer = StructuralNormalizer.fit(
        [query.structural for query in raw_training]
    )
    training_queries = _normalize_queries(raw_training, normalizer)
    validation_queries = _normalize_queries(raw_validation, normalizer)
    profiles = enumerate_profiles(fixture.num_groups)
    context_provider = _SyntheticPrefixContext(fixture.embedding_dim)
    query_manifest_path = output_dir / config.query_manifest_filename
    query_manifest_sha256 = _write_query_manifest(
        training_queries=training_queries,
        validation_queries=validation_queries,
        config=config,
        path=query_manifest_path,
    )

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
        config=config,
    )

    metrics: dict[str, Any] = {
        "schema_version": config.metrics_schema_version,
        "evidence_class": config.evidence_class,
        "evidentiary_status": config.evidentiary_status,
        "claim_boundary": config.claim_boundary,
        "run_configuration": asdict(config),
        "random_seeds": {
            "dataset": config.seed,
            "bootstrap": asdict(config.bootstrap_seeds),
        },
        "training_queries": len(training_queries),
        "validation_queries": len(validation_queries),
        "executable_profiles": len(profiles),
        "num_groups": fixture.num_groups,
        "structural_dim": training_queries[0].features.structural_dim,
        "embedding_dim": fixture.embedding_dim,
        "static_profile_id": profile_id(static_profile),
        "training_mse": {
            "direct": direct.model.training_mse,
            "independent": independent.model.training_mse,
            "interaction_by_group": [
                model.training_mse for model in interaction.models
            ],
        },
        "provenance": _provenance(
            config=config,
            query_manifest_sha256=query_manifest_sha256,
        ),
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
    parser.add_argument(
        "--source-git-sha",
        type=str,
        default=None,
        help="full committed source SHA recorded in generated artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = run(
        output_dir=args.output_dir,
        seed=args.seed,
        source_git_sha=args.source_git_sha,
    )
    summary = {
        method: round(
            values["synthetic_target_profile_match_rate"],
            6,
        )
        for method, values in metrics["methods"].items()
    }
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "evidence_class": _EVIDENCE_CLASS,
                "evidentiary_status": _EVIDENTIARY_STATUS,
                "claim_boundary": _CLAIM_BOUNDARY,
                "synthetic_target_profile_match_rates": summary,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Functional-quality orchestration at the public execution adapter seam."""

from __future__ import annotations

import importlib.util
import math
import re
import subprocess
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypeGuard

from ..identity import sha256_canonical
from ..plan import MATH_SOURCE_REVISION, ExperimentPlan


class QualityExecutor(Protocol):
    """The existing profile executor plus the quality diagnostic operation."""

    evidence_class: str

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        """Run one complete profile/query execution."""

        ...

    def teacher_forced_diagnostic(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        reference_tokens: list[int],
    ) -> Mapping[str, Any]:
        """Compare this variant with the already-generated BF16 token sequence."""

        ...


@dataclass(frozen=True)
class QualityRun:
    """Externally visible quality records and summaries."""

    quality_records: tuple[dict[str, Any], ...]
    diagnostic_records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class FunctionalQualityRunner:
    """Compose profile execution with paired correctness and diagnostics.

    Model loading, quantization, generation, tokenization, and teacher-forced
    forward passes remain executor responsibilities. This class only joins
    their public observations with the declared external targets and produces
    auditable quality records.
    """

    def __init__(
        self,
        plan: ExperimentPlan | Mapping[str, Any],
        executor: QualityExecutor,
    ) -> None:
        raw_plan = plan.to_mapping() if isinstance(plan, ExperimentPlan) else dict(plan)
        self._reject_final_queries(raw_plan)
        self.plan = (
            plan
            if isinstance(plan, ExperimentPlan)
            else ExperimentPlan.from_mapping(raw_plan)
        )
        if self.plan.data["mode"] != "functional-quality":
            raise ValueError("FunctionalQualityRunner requires functional-quality mode")
        self.executor = executor
        if getattr(executor, "evidence_class", None) not in {
            "simulated",
            "directly measured",
        }:
            raise ValueError(
                "quality execution requires simulated or directly measured evidence"
            )

    def run(self) -> QualityRun:
        quality_records: list[dict[str, Any]] = []
        diagnostic_records: list[dict[str, Any]] = []
        reference_observations: dict[str, dict[str, Any]] = {}
        reference_quality: dict[str, dict[str, Any]] = {}
        reference_tokens: dict[str, list[int]] = {}
        observations_by_query: dict[str, dict[str, dict[str, Any]]] = {}
        profiles = self.plan.data.get("profiles")
        if not isinstance(profiles, list):
            raise TypeError("quality plans require a profile list")
        variant_ids = ["BF16", *[str(profile_id) for profile_id in profiles]]

        for query in self.plan.data["queries"]:
            for variant_id in variant_ids:
                observation = self._execute(query, variant_id)
                observations_by_query.setdefault(query["query_id"], {})[variant_id] = (
                    observation
                )
                if variant_id == "BF16":
                    reference_observations[query["query_id"]] = observation
                    tokens = observation.get("generated_tokens")
                    if observation.get("status") == "complete" and self._valid_tokens(
                        tokens
                    ):
                        reference_tokens[query["query_id"]] = list(tokens)
                quality_record = self._quality_record(
                    query,
                    variant_id,
                    observation,
                    reference_quality.get(query["query_id"]),
                )
                quality_records.append(quality_record)
                if variant_id == "BF16":
                    reference_quality[query["query_id"]] = quality_record

        for variant_id in variant_ids:
            for query in self.plan.data["queries"]:
                reference = reference_observations[query["query_id"]]
                diagnostic_records.append(
                    self._diagnostic_record(
                        query,
                        variant_id,
                        reference,
                        reference_tokens.get(query["query_id"]),
                        observations_by_query[query["query_id"]][variant_id],
                    )
                )

        return QualityRun(
            quality_records=tuple(quality_records),
            diagnostic_records=tuple(diagnostic_records),
            summary=self._summary(quality_records),
        )

    @staticmethod
    def parse_mmlu_answer(text: str) -> str | None:
        """Parse only the accepted single-option MMLU-Pro answer forms."""

        normalized = _normalize_text(text)
        match = re.fullmatch(
            r"(?:([A-J])(?:[.)])?|Answer: ([A-J])|Final answer: ([A-J]))",
            normalized,
        )
        if match is None:
            return None
        return next(group for group in match.groups() if group is not None)

    @staticmethod
    def _reject_final_queries(raw_plan: Mapping[str, Any]) -> None:
        manifest = raw_plan.get("source_manifest")
        queries = raw_plan.get("queries")
        if not isinstance(manifest, Mapping) or not isinstance(queries, list):
            return
        final_ids = manifest.get("final_source_ids", [])
        if not isinstance(final_ids, list):
            return
        final_set = set(final_ids)
        for query in queries:
            if (
                isinstance(query, Mapping)
                and query.get("source_id") in final_set
                and query.get("dataset") != "MMLU-Pro"
            ):
                raise ValueError(
                    "final-only query IDs are rejected from quality construction"
                )

    def _execute(self, query: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
        try:
            observation = self.executor.execute(query, variant_id)
        except (
            RuntimeError,
            ValueError,
            OSError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            MemoryError,
        ) as exc:
            return {
                "status": "invalid",
                "transform_status": "invalid"
                if variant_id != "BF16"
                else "not_applicable",
                "forward_status": "not_attempted",
                "reason_code": f"EXECUTOR_EXCEPTION:{type(exc).__name__}",
                "transform_reason_code": f"EXECUTOR_EXCEPTION:{type(exc).__name__}",
                "forward_reason_code": "EXECUTOR_EXCEPTION",
                "observed_group_prefix": [],
            }
        if not isinstance(observation, Mapping):
            raise TypeError("profile executor observations must be objects")
        normalized = dict(observation)
        status = normalized.get("status")
        if status not in {"complete", "invalid", "incomplete", "aborted"}:
            raise ValueError("profile executor returned an unsupported quality status")
        if (
            normalized.get("status") == "complete"
            and normalized.get("reason_code") is None
        ):
            normalized["reason_code"] = "completed"
        if (
            not isinstance(normalized.get("reason_code"), str)
            or not normalized["reason_code"]
        ):
            raise ValueError("profile executor observations require a reason code")
        return normalized

    def _quality_record(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        observation: Mapping[str, Any],
        reference: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = self._lineage(
            artifact_type="functional-quality-outcome",
            schema_version="qbitplan.stage1.functional-quality-outcome.v1",
            query=query,
            profile_id=variant_id,
        )
        record.update(
            {
                "status": observation["status"],
                "execution_status": observation["status"],
                "transform_status": observation.get("transform_status"),
                "forward_status": observation.get("forward_status"),
                "reason_code": observation["reason_code"],
                "transform_reason_code": observation.get("transform_reason_code"),
                "forward_reason_code": observation.get("forward_reason_code"),
                "evidence_class": self.executor.evidence_class,
            }
        )
        for field in ("output_hash", "output_text", "token_count"):
            if field in observation:
                record[field] = observation[field]
        if observation["status"] != "complete":
            return record
        if not isinstance(observation.get("output_text"), str):
            record.update(
                {
                    "status": "incomplete",
                    "execution_status": "incomplete",
                    "reason_code": "MISSING_GENERATED_TEXT",
                }
            )
            return record
        correct, parse_status = self._grade(query, observation["output_text"])
        record.update(
            {
                "correct": correct,
                "parse_status": parse_status,
                "parse_failure": parse_status != "parsed",
            }
        )
        if variant_id == "BF16":
            record["degradation"] = 0
        elif (
            reference is None
            or reference.get("status") != "complete"
            or "correct" not in reference
        ):
            record.update(
                {
                    "status": "incomplete",
                    "execution_status": "incomplete",
                    "reason_code": "REFERENCE_OUTCOME_UNAVAILABLE",
                }
            )
        else:
            record["degradation"] = int(bool(reference["correct"])) - int(correct)
        return record

    def _diagnostic_record(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        reference: Mapping[str, Any],
        tokens: list[int] | None,
        profile_observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = self._lineage(
            artifact_type="functional-quality-diagnostic",
            schema_version="qbitplan.stage1.functional-quality-diagnostic.v1",
            query=query,
            profile_id=variant_id,
        )
        record["evidence_class"] = self.executor.evidence_class
        if reference.get("status") != "complete":
            record.update(
                {
                    "status": "omitted/unavailable",
                    "reason_code": "REFERENCE_EXECUTION_INVALID",
                }
            )
            return record
        if tokens is None:
            record.update(
                {
                    "status": "omitted/unavailable",
                    "reason_code": "REFERENCE_TOKENS_UNAVAILABLE",
                }
            )
            return record
        record["reference_token_count"] = len(tokens)
        if variant_id != "BF16" and profile_observation.get("status") != "complete":
            record.update(
                {
                    "status": "omitted/unavailable",
                    "reason_code": "PROFILE_EXECUTION_INVALID",
                }
            )
            return record
        try:
            diagnostic = self.executor.teacher_forced_diagnostic(
                query, variant_id, tokens
            )
        except (
            RuntimeError,
            ValueError,
            OSError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            MemoryError,
        ) as exc:
            record.update(
                {
                    "status": "omitted/unavailable",
                    "reason_code": f"DIAGNOSTIC_EXCEPTION:{type(exc).__name__}",
                }
            )
            return record
        if (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("status") != "complete"
        ):
            record.update(
                {"status": "omitted/unavailable", "reason_code": "DIAGNOSTIC_FAILED"}
            )
            return record
        kl_mean = diagnostic.get("kl_mean")
        if (
            isinstance(kl_mean, bool)
            or not isinstance(kl_mean, (int, float))
            or not math.isfinite(kl_mean)
            or kl_mean < 0
        ):
            record.update({"status": "invalid", "reason_code": "DIAGNOSTIC_NON_FINITE"})
            return record
        record.update(
            {
                "status": "complete",
                "reason_code": diagnostic.get("reason_code", "completed"),
                "kl_mean": kl_mean,
                "reference_token_count": len(tokens),
                "diagnostic_evidence_class": diagnostic.get(
                    "evidence_class", self.executor.evidence_class
                ),
            }
        )
        return record

    def _grade(self, query: Mapping[str, Any], output_text: str) -> tuple[bool, str]:
        dataset = query.get("dataset")
        if dataset == "MATH":
            return self._grade_math(output_text, query["record"]["solution"])
        if dataset == "MMLU-Pro":
            answer = self.parse_mmlu_answer(output_text)
            target = self._mmlu_target(query["record"])
            if answer is None:
                return False, "parse_failed"
            return answer == target, "parsed"
        raise ValueError(f"unsupported quality dataset: {dataset!r}")

    @staticmethod
    def _grade_math(output_text: str, solution: str) -> tuple[bool, str]:
        last_boxed_only_string, is_equiv = _load_pinned_math_evaluator()
        output_box = last_boxed_only_string(_normalize_text(output_text))
        target_box = last_boxed_only_string(_normalize_text(solution))
        if output_box is None or target_box is None:
            return False, "parse_failed"
        output = _strip_source_box(output_box)
        target = _strip_source_box(target_box)
        if output is None or target is None:
            return False, "parse_failed"
        return bool(is_equiv(output, target)), "parsed"

    @staticmethod
    def _mmlu_target(record: Mapping[str, Any]) -> str:
        target = record.get("answer")
        if (
            isinstance(target, int)
            and not isinstance(target, bool)
            and 0 <= target <= 9
        ):
            return chr(ord("A") + target)
        if isinstance(target, str):
            normalized = _normalize_text(target)
            if re.fullmatch(r"[A-J]", normalized):
                return normalized
            if re.fullmatch(r"[0-9]", normalized):
                return chr(ord("A") + int(normalized))
        raise ValueError("MMLU-Pro record has no valid external answer target")

    def _summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[(record["phase"], record["dataset"], record["profile_id"])].append(
                record
            )
        summaries = []
        for (phase, dataset, profile_id), group in sorted(grouped.items()):
            scored = [
                record
                for record in group
                if "correct" in record and record["status"] == "complete"
            ]
            invalid_count = len(group) - len(scored)
            summary: dict[str, Any] = {
                "phase": phase,
                "dataset": dataset,
                "profile_id": profile_id,
                "record_count": len(group),
                "scored_record_count": len(scored),
                "invalid_or_incomplete_record_count": invalid_count,
                "accuracy": (
                    sum(bool(record["correct"]) for record in scored) / len(scored)
                )
                if scored
                else None,
                "degradation": (
                    sum(
                        record["degradation"]
                        for record in scored
                        if "degradation" in record
                    )
                    / len(scored)
                    if scored and all("degradation" in record for record in scored)
                    else None
                ),
            }
            summaries.append(summary)
        return {
            "artifact_type": "functional-quality-summary",
            "schema_version": "qbitplan.stage1.functional-quality-summary.v1",
            "evidence_class": self.executor.evidence_class,
            "source_manifest_id": self.plan.source_manifest_id,
            "source_artifact_id": self.plan.data["source_manifest"]["artifact_id"],
            "source_artifact_ids": self._source_artifact_ids(),
            "configuration_hash": sha256_canonical(
                {
                    "model": self.plan.data.get("model"),
                    "tokenizer": self.plan.data.get("tokenizer"),
                    "software": self.plan.data.get("software"),
                    "decoder": self.plan.data.get("decoder"),
                    "runtime": self.plan.data.get("runtime"),
                    "seed": self.plan.data.get("seed"),
                    "determinism": self.plan.data.get("determinism"),
                }
            ),
            "record_count": len(records),
            "groups": summaries,
        }

    def _source_artifact_ids(self) -> list[str]:
        artifact_ids = [self.plan.data["source_manifest"]["artifact_id"]]
        if self.plan.data["source_manifest"]["dataset"] == "MMLU-Pro":
            artifact_ids.append(self.plan.data["profile_inventory"]["artifact_id"])
        return artifact_ids

    def _lineage(
        self,
        *,
        artifact_type: str,
        schema_version: str,
        query: Mapping[str, Any],
        profile_id: str,
    ) -> dict[str, Any]:
        return {
            "artifact_type": artifact_type,
            "schema_version": schema_version,
            "source_manifest_id": self.plan.source_manifest_id,
            "source_artifact_id": self.plan.data["source_manifest"]["artifact_id"],
            "source_artifact_ids": self._source_artifact_ids(),
            "configuration_hash": sha256_canonical(
                {
                    "model": self.plan.data.get("model"),
                    "tokenizer": self.plan.data.get("tokenizer"),
                    "software": self.plan.data.get("software"),
                    "decoder": self.plan.data.get("decoder"),
                    "runtime": self.plan.data.get("runtime"),
                    "seed": self.plan.data.get("seed"),
                    "determinism": self.plan.data.get("determinism"),
                }
            ),
            "record_count": 1,
            "phase": query["phase"],
            "dataset": query["dataset"],
            "query_id": query["query_id"],
            "source_id": query["source_id"],
            "profile_id": profile_id,
        }

    @staticmethod
    def _valid_tokens(tokens: Any) -> TypeGuard[list[int]]:
        return isinstance(tokens, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in tokens
        )


def _normalize_text(value: str) -> str:
    return unicodedata.normalize(
        "NFKC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def _strip_source_box(value: str) -> str | None:
    for prefix in ("\\boxed{", "\\fbox{"):
        if value.startswith(prefix) and value.endswith("}"):
            return value[len(prefix) : -1]
    return None


@lru_cache(maxsize=1)
def _load_pinned_math_evaluator() -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[2] / "data" / "pinned" / "math"
    if not root.is_dir():
        raise RuntimeError("pinned MATH evaluator checkout is unavailable")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("pinned MATH evaluator revision cannot be verified") from exc
    if revision != MATH_SOURCE_REVISION:
        raise RuntimeError("pinned MATH evaluator revision does not match the plan")

    util_spec = importlib.util.spec_from_file_location(
        "qbitplan_pinned_math_util", root / "modeling" / "dataset" / "util.py"
    )
    equivalence_spec = importlib.util.spec_from_file_location(
        "qbitplan_pinned_math_equivalence", root / "modeling" / "math_equivalence.py"
    )
    if (
        util_spec is None
        or util_spec.loader is None
        or equivalence_spec is None
        or equivalence_spec.loader is None
    ):
        raise RuntimeError("pinned MATH evaluator modules cannot be loaded")
    util_module = importlib.util.module_from_spec(util_spec)
    equivalence_module = importlib.util.module_from_spec(equivalence_spec)
    util_spec.loader.exec_module(util_module)
    equivalence_spec.loader.exec_module(equivalence_module)
    return util_module.last_boxed_only_string, equivalence_module.is_equiv

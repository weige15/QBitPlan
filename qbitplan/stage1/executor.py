"""The real TorchAO-backed profile executor for issue #25 smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import os
import random
import subprocess
import sys
from typing import Any

from .contract import ExperimentPlan, ProfileExecutionResult


class RuntimeDependencyError(RuntimeError):
    """Raised when the declared executable profile cannot be initialized."""


def pin_gpu_uuid(expected_uuid: str) -> tuple[str, str, str]:
    """Pin this process to exactly the declared physical GPU UUID."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeDependencyError("nvidia-smi is required to validate the GPU UUID") from exc

    matches = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == expected_uuid:
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeDependencyError(f"declared GPU UUID is not uniquely available: {expected_uuid}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != expected_uuid:
        raise RuntimeDependencyError(
            "CUDA_VISIBLE_DEVICES conflicts with the plan's UUID-pinned GPU"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = expected_uuid
    return tuple(matches[0])  # type: ignore[return-value]


@dataclass
class _PreparedProfile:
    model: Any | None
    reason_code: str | None = None
    detail: str = ""


class TorchAOProfileExecutor:
    """Own profile transforms and complete ordered model execution.

    The class imports the accepted runtime stack lazily so contract tests can
    run without importing or selecting a test adapter in the scientific CLI.
    """

    evidence_class = "directly measured"

    def __init__(self, plan: ExperimentPlan) -> None:
        self.plan = plan
        pin_gpu_uuid(plan.data["hardware"]["gpu_uuid"])
        self._torch, self._tokenizer, self._auto_model = self._load_runtime()
        self._device = self._torch.device("cuda:0")
        self._configure_runtime()
        self._prepared: dict[str, _PreparedProfile] = {}

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        try:
            import accelerate
            import datasets
            import numpy as np
            import safetensors
            import torch
            import transformers
            import torchao
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeDependencyError(
                "the declared Python/Torch/TorchAO/Transformers stack is unavailable"
            ) from exc

        expected = self.plan.data["backend"]
        versions = {
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "torchao": torchao.__version__,
            "numpy": np.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
            "python": sys.version.split()[0],
        }
        for key, actual in versions.items():
            expected_key = {"pytorch": "pytorch", "transformers": "transformers", "torchao": "version"}.get(
                key, key
            )
            declared = expected.get(expected_key)
            if key == "torchao":
                declared = expected["version"]
            if declared != actual:
                raise RuntimeDependencyError(
                    f"{key} version mismatch: plan={declared!r}, runtime={actual!r}"
                )
        if not torch.cuda.is_available():
            raise RuntimeDependencyError("CUDA is unavailable for the real TorchAO executor")

        identity = pin_gpu_uuid(self.plan.data["hardware"]["gpu_uuid"])
        declared_hardware = self.plan.data["hardware"]
        if identity[1] != declared_hardware["gpu_name"] or identity[2] != declared_hardware["driver"]:
            raise RuntimeDependencyError("GPU name or driver does not match the plan")
        if torch.version.cuda != self.plan.data["backend"]["cuda"]:
            raise RuntimeDependencyError("CUDA runtime does not match the plan")
        model_config = self.plan.data["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            use_fast=True,
            trust_remote_code=False,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeDependencyError("the accepted fast tokenizer is unavailable")
        try:
            from huggingface_hub import hf_hub_download

            for filename, expected_hash in self.plan.data["tokenizer"]["file_hashes"].items():
                tokenizer_path = hf_hub_download(
                    repo_id=model_config["id"],
                    filename=filename,
                    revision=model_config["revision"],
                )
                with open(tokenizer_path, "rb") as handle:
                    actual_hash = hashlib.sha256(handle.read()).hexdigest()
                if actual_hash != expected_hash:
                    raise RuntimeDependencyError(f"tokenizer file hash mismatch: {filename}")
        except Exception as exc:
            raise RuntimeDependencyError("tokenizer file hashes could not be verified") from exc
        return torch, tokenizer, AutoModelForCausalLM

    def _configure_runtime(self) -> None:
        torch = self._torch
        runtime = self.plan.data["runtime"]
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is not None and workspace != runtime["cublas_workspace_config"]:
            raise RuntimeDependencyError("CUBLAS_WORKSPACE_CONFIG conflicts with the plan")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = runtime["cublas_workspace_config"]
        random.seed(self.plan.data["seeds"]["python"])
        torch.manual_seed(self.plan.data["seeds"]["torch_cpu"])
        import numpy as np

        np.random.seed(self.plan.data["seeds"]["numpy"])
        torch.cuda.manual_seed_all(self.plan.data["seeds"]["torch_cuda"])
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = runtime["tf32"]
        torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = runtime["deterministic_cudnn"]
        torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])

    def execute(self, profile_id: str, query: dict[str, str]) -> ProfileExecutionResult:
        prepared = self._prepare_profile(profile_id)
        boundaries = tuple(f"g{index}[{index * 4}:{(index + 1) * 4}]" for index in range(8))
        if prepared.model is None:
            return ProfileExecutionResult(
                profile_id=profile_id,
                query_id=query["query_id"],
                transform_status="not_applicable" if profile_id == "bf16" else "failed",
                forward_status="not_run",
                terminal_status="invalid",
                executable=False,
                reason_code=prepared.reason_code or "transform_failed",
                evidence_class=self.evidence_class,
                detail=prepared.detail,
                group_boundaries=boundaries,
                upstream_state="not_available",
            )
        return self._forward(profile_id, query, prepared.model, boundaries)

    def _prepare_profile(self, profile_id: str) -> _PreparedProfile:
        if profile_id in self._prepared:
            return self._prepared[profile_id]
        try:
            model = self._load_model()
            if profile_id != "bf16":
                self._apply_profile(model, profile_id)
            prepared = _PreparedProfile(model=model)
        except (RuntimeError, ValueError, OSError, AssertionError, AttributeError, TypeError, ImportError, KeyError, IndexError, MemoryError) as exc:
            prepared = _PreparedProfile(
                model=None,
                reason_code=self._transform_reason(exc),
                detail=f"{type(exc).__name__}: {exc}",
            )
        self._prepared[profile_id] = prepared
        return prepared

    def _load_model(self) -> Any:
        model_config = self.plan.data["model"]
        model = self._auto_model.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            torch_dtype=self._torch.bfloat16,
            trust_remote_code=False,
        )
        model = model.to(self._device)
        if getattr(model.config, "model_type", None) != "llama":
            raise RuntimeError("loaded model architecture is not Llama")
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != self.plan.data["model"]["layers"]:
            raise RuntimeError("loaded model layer count does not match the plan")
        first_parameter = next(model.parameters(), None)
        if first_parameter is None or first_parameter.dtype != self._torch.bfloat16:
            raise RuntimeError("loaded model parameters are not BF16")
        model.eval()
        return model

    def _apply_profile(self, model: Any, profile_id: str) -> None:
        bits = self._profile_bits(profile_id)
        named_modules = dict(model.named_modules())
        all_target_fqns = {
            fqn for group_index in range(8) for fqn in self._group_fqns(group_index)
        }
        missing = sorted(fqn for fqn in all_target_fqns if fqn not in named_modules)
        if missing:
            raise RuntimeError(f"unsupported_transform: missing target modules {missing[:3]}")
        before_weight_ids = {
            fqn: id(module.weight)
            for fqn, module in named_modules.items()
            if hasattr(module, "weight")
        }
        missing_weights = sorted(
            fqn for fqn in all_target_fqns if fqn not in before_weight_ids
        )
        if missing_weights:
            raise RuntimeError(
                f"unsupported_transform: target modules lack weights {missing_weights[:3]}"
            )
        for group_index, bit in enumerate(bits):
            fqns = self._group_fqns(group_index)
            missing = sorted(fqn for fqn in fqns if fqn not in named_modules)
            if missing:
                raise RuntimeError(f"unsupported_transform: missing target modules {missing[:3]}")
            target_fqns = set(fqns)
            if bit == 4:
                from torchao.quantization import int4_weight_only

                config = int4_weight_only(group_size=self.plan.data["groups"]["int4_group_size"])
            else:
                from torchao.quantization import int8_weight_only

                config = int8_weight_only()

            def filter_fn(module: Any, fqn: str, targets: set[str] = target_fqns) -> bool:
                return fqn in targets

            from torchao.quantization import quantize_

            quantize_(model, config, filter_fn=filter_fn, device=self._device)
        after_modules = dict(model.named_modules())
        untransformed = [
            fqn for fqn in all_target_fqns if id(after_modules[fqn].weight) == before_weight_ids[fqn]
        ]
        if untransformed:
            raise RuntimeError(f"unsupported_transform: target weights did not transform {untransformed[:3]}")
        changed_non_targets = [
            fqn for fqn, before_id in before_weight_ids.items()
            if fqn not in all_target_fqns and id(after_modules[fqn].weight) != before_id
        ]
        if changed_non_targets:
            raise RuntimeError(f"unsupported_transform: non-target weights changed {changed_non_targets[:3]}")

    def _profile_bits(self, profile_id: str) -> tuple[int, ...]:
        if profile_id not in self.plan.profile_ids or profile_id == "bf16":
            raise RuntimeError(f"unsupported_transform: unknown profile {profile_id}")
        if len(profile_id) != 8 or any(bit not in "01" for bit in profile_id):
            raise RuntimeError(f"unsupported_transform: invalid profile {profile_id}")
        return tuple(4 if bit == "0" else 8 for bit in profile_id)

    def _group_fqns(self, group_index: int) -> tuple[str, ...]:
        projections = self.plan.data["groups"]["target_projections"]
        names = []
        for layer in range(group_index * 4, group_index * 4 + 4):
            names.extend(
                [
                    f"model.layers.{layer}.self_attn.{projection}"
                    for projection in projections[:4]
                ]
            )
            names.extend(
                [
                    f"model.layers.{layer}.mlp.{projection}"
                    for projection in projections[4:]
                ]
            )
        return tuple(names)

    @contextmanager
    def _ordered_group_observer(self, model: Any) -> Any:
        observed_groups: list[int] = []
        named_modules = dict(model.named_modules())
        handles = []
        for group_index in range(self.plan.data["groups"]["count"]):
            first_group_fqn = self._group_fqns(group_index)[0]
            try:
                module = named_modules[first_group_fqn]
            except KeyError as exc:
                raise RuntimeError(
                    f"incomplete_group_order: missing boundary module {first_group_fqn}"
                ) from exc
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _inputs, group=group_index: observed_groups.append(group)
                )
            )
        try:
            yield observed_groups
        finally:
            for handle in handles:
                handle.remove()

    def _forward(
        self,
        profile_id: str,
        query: dict[str, str],
        model: Any,
        boundaries: tuple[str, ...],
    ) -> ProfileExecutionResult:
        try:
            tokenizer = self._tokenizer
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is None:
                raise RuntimeError("missing eos token ID")
            encoded = tokenizer(
                query["prompt"],
                return_tensors="pt",
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            max_positions = getattr(model.config, "max_position_embeddings", None)
            max_new_tokens = self.plan.data["decoder"]["max_new_tokens"]
            if max_positions is not None and input_ids.shape[-1] + max_new_tokens > max_positions:
                return self._failed_forward(
                    profile_id,
                    query["query_id"],
                    "context_overflow",
                    "prompt plus generation exceeds the model context window",
                    boundaries,
                )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            decoder = self.plan.data["decoder"]
            with self._ordered_group_observer(model) as observed_groups:
                self._torch.cuda.synchronize()
                with self._torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=decoder["max_new_tokens"],
                        num_beams=decoder["num_beams"],
                        num_return_sequences=decoder["num_return_sequences"],
                        do_sample=decoder["do_sample"],
                        early_stopping=decoder["early_stopping"],
                        length_penalty=decoder["length_penalty"],
                        repetition_penalty=decoder["repetition_penalty"],
                        no_repeat_ngram_size=decoder["no_repeat_ngram_size"],
                        num_beam_groups=decoder["num_beam_groups"],
                        diversity_penalty=decoder["diversity_penalty"],
                        pad_token_id=eos_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=self.plan.data["runtime"]["use_cache"],
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                self._torch.cuda.synchronize()
            if tuple(observed_groups[:8]) != tuple(range(8)):
                return self._failed_forward(
                    profile_id,
                    query["query_id"],
                    "incomplete_group_order",
                    "group execution did not observe the ordered eight-group path",
                    boundaries,
                )
            for score in getattr(generated, "scores", ()) or ():
                if not bool(self._torch.isfinite(score).all()):
                    return self._failed_forward(
                        profile_id,
                        query["query_id"],
                        "non_finite_output",
                        "generation score contained a non-finite value",
                        boundaries,
                    )
            sequences = generated.sequences.detach().cpu().numpy().tobytes()
            return ProfileExecutionResult(
                profile_id=profile_id,
                query_id=query["query_id"],
                transform_status="not_applicable" if profile_id == "bf16" else "complete",
                forward_status="complete",
                terminal_status="complete",
                executable=True,
                reason_code="completed",
                evidence_class=self.evidence_class,
                output_digest=hashlib.sha256(sequences).hexdigest(),
                group_boundaries=boundaries,
                upstream_state=f"ordered-groups:{','.join(str(group) for group in observed_groups[:8])}",
            )
        except (RuntimeError, ValueError, OSError, AssertionError, AttributeError, TypeError, ImportError, KeyError, IndexError, MemoryError) as exc:
            return self._failed_forward(
                profile_id,
                query["query_id"],
                self._forward_reason(exc),
                f"{type(exc).__name__}: {exc}",
                boundaries,
            )

    def _failed_forward(
        self,
        profile_id: str,
        query_id: str,
        reason_code: str,
        detail: str,
        boundaries: tuple[str, ...],
    ) -> ProfileExecutionResult:
        return ProfileExecutionResult(
            profile_id=profile_id,
            query_id=query_id,
            transform_status="not_applicable" if profile_id == "bf16" else "complete",
            forward_status="failed",
            terminal_status="invalid",
            executable=False,
            reason_code=reason_code,
            evidence_class=self.evidence_class,
            detail=detail,
            group_boundaries=boundaries,
            upstream_state="not_available",
        )

    @staticmethod
    def _forward_reason(exc: Exception) -> str:
        message = str(exc).lower()
        if "group_order" in message or "group execution" in message:
            return "incomplete_group_order"
        if "deterministic" in message:
            return "unsupported_determinism"
        if "overflow" in message:
            return "context_overflow"
        if "non-finite" in message or "nonfinite" in message:
            return "non_finite_output"
        if "out of memory" in message or "cuda out of memory" in message:
            return "oom"
        if isinstance(exc, MemoryError):
            return "oom"
        return "forward_failed"

    @staticmethod
    def _transform_reason(exc: Exception) -> str:
        if "out of memory" in str(exc).lower():
            return "oom"
        if isinstance(exc, MemoryError):
            return "oom"
        if "deterministic" in str(exc).lower():
            return "unsupported_determinism"
        if "unsupported_transform" in str(exc) or "unsupported" in str(exc).lower():
            return "unsupported_transform"
        return "transform_failed"

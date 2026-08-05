"""Real pinned-model TorchAO executor for smoke and functional-quality runs.

The optional ML libraries are imported only when this real adapter is used.
The adapter owns the target-module mapping, profile transformation, complete
forward, and ordered group-boundary observation.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..identity import canonical_json_bytes
from ..plan import MODEL_IDENTIFIER, MODEL_REVISION, ExperimentPlan


class _ExecutionFailure(RuntimeError):
    def __init__(self, reason_code: str, *, transform: bool = False) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.transform = transform


@dataclass(frozen=True)
class _FailureRecord:
    """Traceback-free cached failure for a profile already known to be invalid."""

    reason_code: str
    transform: bool

    @classmethod
    def from_exception(cls, exc: _ExecutionFailure) -> _FailureRecord:
        return cls(reason_code=exc.reason_code, transform=exc.transform)

    def to_exception(self) -> _ExecutionFailure:
        return _ExecutionFailure(self.reason_code, transform=self.transform)


class TorchAOProfileExecutor:
    """Execute every declared profile on the pinned RTX 3090.

    Quantized variants are prepared and validated on CPU before the completed
    representation is transferred to the selected GPU. This avoids holding the
    full BF16 CUDA allocation while TorchAO constructs replacement weights.
    """

    evidence_class = "directly measured"

    def __init__(self, plan: ExperimentPlan) -> None:
        self.plan = plan
        self._device_info: dict[str, Any] | None = None
        self._tokenizer: Any | None = None
        self._active_variant: str | None = None
        self._active_model: Any | None = None
        self._profile_failures: dict[str, _FailureRecord] = {}

    def hardware_identity(self) -> Mapping[str, Any]:
        return dict(self._resolve_device())

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        try:
            model = self._model_for_variant(variant_id)
            tokenizer = self._load_tokenizer()
            return self._run_forward(model, tokenizer, query["prompt"], variant_id)
        except _ExecutionFailure as exc:
            if exc.transform:
                return {
                    "status": "invalid",
                    "transform_status": "invalid",
                    "forward_status": "not_attempted",
                    "reason_code": exc.reason_code,
                    "transform_reason_code": exc.reason_code,
                    "forward_reason_code": "TRANSFORM_FAILED",
                    "observed_group_prefix": [],
                }
            return {
                "status": "invalid",
                "transform_status": "complete" if variant_id != "BF16" else "not_applicable",
                "forward_status": "invalid",
                "reason_code": exc.reason_code,
                "transform_reason_code": "REFERENCE_UNQUANTIZED" if variant_id == "BF16" else None,
                "forward_reason_code": exc.reason_code,
                "observed_group_prefix": [],
            }
        except (RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, MemoryError) as exc:
            return {
                "status": "invalid",
                "transform_status": "complete" if variant_id != "BF16" else "not_applicable",
                "forward_status": "invalid",
                "reason_code": f"FORWARD_EXCEPTION:{type(exc).__name__}",
                "transform_reason_code": "REFERENCE_UNQUANTIZED" if variant_id == "BF16" else None,
                "forward_reason_code": f"FORWARD_EXCEPTION:{type(exc).__name__}",
                "observed_group_prefix": [],
            }

    def _resolve_device(self) -> dict[str, Any]:
        if self._device_info is not None:
            return self._device_info
        gpu_uuid = self.plan.data["gpu_uuid"]
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _ExecutionFailure(f"GPU_QUERY_FAILED:{type(exc).__name__}") from exc
        rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
        selected: dict[str, Any] | None = None
        for row in rows:
            if len(row) != 4:
                continue
            index, name, uuid, driver = (item.strip() for item in row)
            if uuid == gpu_uuid:
                selected = {
                    "device_index": int(index),
                    "gpu_name": name,
                    "gpu_uuid": uuid,
                    "driver_version": driver,
                }
                break
        if selected is None:
            raise _ExecutionFailure("GPU_UUID_NOT_FOUND")
        if selected["gpu_name"] != "NVIDIA GeForce RTX 3090":
            raise _ExecutionFailure("GPU_MODEL_MISMATCH")
        try:
            import torch

            self._validate_software(torch, selected["driver_version"])
            if not torch.cuda.is_available():
                raise _ExecutionFailure("CUDA_UNAVAILABLE")
            if selected["device_index"] >= torch.cuda.device_count():
                raise _ExecutionFailure("GPU_DEVICE_INDEX_UNAVAILABLE")
            torch.cuda.set_device(selected["device_index"])
            self._apply_determinism(torch)
        except ImportError as exc:
            raise _ExecutionFailure("PYTORCH_UNAVAILABLE") from exc
        self._device_info = selected
        return selected

    def _validate_software(self, torch: Any, driver_version: str) -> None:
        try:
            import accelerate
            import datasets
            import numpy as np
            import safetensors
            import torchao
            import transformers
        except ImportError as exc:
            raise _ExecutionFailure("SOFTWARE_TUPLE_UNAVAILABLE") from exc
        actual = {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "torchao": torchao.__version__,
            "numpy": np.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
            "cuda": torch.version.cuda,
            "nvidia_driver": driver_version,
        }
        if actual != self.plan.data["software"]:
            raise _ExecutionFailure("SOFTWARE_TUPLE_MISMATCH")


    def _apply_determinism(self, torch: Any) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.plan.data["determinism"]["cublas_workspace_config"]
        try:
            import numpy as np

            np.random.seed(self.plan.data["seed"])
        except ImportError as exc:
            raise _ExecutionFailure("NUMPY_UNAVAILABLE") from exc
        random.seed(self.plan.data["seed"])
        torch.manual_seed(self.plan.data["seed"])
        torch.cuda.manual_seed_all(self.plan.data["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            raise _ExecutionFailure("UNSUPPORTED_DETERMINISTIC_OPERATION") from exc

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise _ExecutionFailure("TRANSFORMERS_UNAVAILABLE") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_IDENTIFIER,
            revision=MODEL_REVISION,
            use_fast=True,
            trust_remote_code=False,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise _ExecutionFailure("FAST_TOKENIZER_UNAVAILABLE")
        if tokenizer.eos_token_id is None:
            raise _ExecutionFailure("EOS_TOKEN_ID_UNAVAILABLE")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer
        return tokenizer

    def _model_for_variant(self, variant_id: str) -> Any:
        if variant_id not in self.plan.data["profiles"]:
            raise _ExecutionFailure("PROFILE_NOT_DECLARED", transform=True)
        cached_failure = self._profile_failures.get(variant_id)
        if cached_failure is not None:
            raise cached_failure.to_exception()
        if self._active_variant == variant_id and self._active_model is not None:
            return self._active_model
        self._release_model()
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            failure = _ExecutionFailure("MODEL_RUNTIME_UNAVAILABLE")
            self._profile_failures[variant_id] = _FailureRecord.from_exception(failure)
            raise failure from exc
        device_info = self._resolve_device()
        cpu_device = torch.device("cpu")
        cuda_device = torch.device("cuda", device_info["device_index"])
        model: Any | None = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_IDENTIFIER,
                revision=MODEL_REVISION,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
            )
            model.to(cpu_device)  # type: ignore[arg-type]
            model.eval()
            if variant_id != "BF16":
                self._transform_profile(model, variant_id, cpu_device)
            self._validate_group_structure(model)
            model.to(cuda_device)  # type: ignore[arg-type]
        except _ExecutionFailure as exc:
            self._profile_failures[variant_id] = _FailureRecord.from_exception(exc)
            model = None
            self._collect_released_memory()
            raise
        except RuntimeError as exc:
            reason = "OOM" if "out of memory" in str(exc).lower() else "TRANSFORM_OR_LOAD_RUNTIME_ERROR"
            failure = _ExecutionFailure(reason, transform=variant_id != "BF16")
            self._profile_failures[variant_id] = _FailureRecord.from_exception(failure)
            model = None
            self._collect_released_memory()
            raise failure from exc
        except OSError as exc:
            failure = _ExecutionFailure("MODEL_LOAD_FAILED", transform=variant_id != "BF16")
            self._profile_failures[variant_id] = _FailureRecord.from_exception(failure)
            model = None
            self._collect_released_memory()
            raise failure from exc
        if model is None:
            raise _ExecutionFailure("MODEL_LOAD_FAILED", transform=variant_id != "BF16")
        self._active_variant = variant_id
        self._active_model = model
        return model

    @staticmethod
    def _collect_released_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _release_model(self) -> None:
        active_model = self._active_model
        self._active_model = None
        self._active_variant = None
        if active_model is not None:
            del active_model
        self._collect_released_memory()

    @staticmethod
    def _group_targets(group_index: int) -> set[str]:
        suffixes = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        return {
            f"model.layers.{layer_index}.{suffix}"
            for layer_index in range(group_index * 4, (group_index + 1) * 4)
            for suffix in suffixes
        }

    def _transform_profile(self, model: Any, variant_id: str, device: Any) -> None:
        try:
            from torchao.quantization import (
                int4_weight_only,
                int8_weight_only,
                quantize_,
            )
        except ImportError as exc:
            raise _ExecutionFailure("TORCHAO_UNAVAILABLE", transform=True) from exc
        all_targets = {fqn for group_index in range(8) for fqn in self._group_targets(group_index)}
        named_modules = dict(model.named_modules())
        missing = sorted(all_targets - set(named_modules))
        if missing:
            raise _ExecutionFailure("TRANSFORM_TARGET_MAPPING_INCOMPLETE", transform=True)
        before_weight_ids: dict[str, int] = {}
        for fqn in all_targets:
            module = named_modules[fqn]
            if not hasattr(module, "weight"):
                raise _ExecutionFailure("TRANSFORM_TARGET_WEIGHT_MISSING", transform=True)
            before_weight_ids[fqn] = id(module.weight)
        for group_index, bit in enumerate(variant_id):
            targets = self._group_targets(group_index)
            seen: set[str] = set()

            def filter_fn(
                module: Any,
                fqn: str,
                targets: set[str] = targets,
                seen: set[str] = seen,
            ) -> bool:
                del module
                if fqn in targets:
                    seen.add(fqn)
                    return True
                return False

            config = int4_weight_only(group_size=128) if bit == "0" else int8_weight_only()
            try:
                quantize_(  # type: ignore[arg-type]
                    model,
                    config,
                    filter_fn=filter_fn,
                    device=device,
                    set_inductor_config=False,
                )
            except (RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, MemoryError) as exc:
                raise _ExecutionFailure(
                    f"TRANSFORM_FAILED_GROUP_{group_index}:{type(exc).__name__}", transform=True
                ) from exc
            if seen != targets:
                raise _ExecutionFailure(
                    f"TRANSFORM_TARGET_MAPPING_INCOMPLETE_GROUP_{group_index}", transform=True
                )

        after_modules = dict(model.named_modules())
        unchanged = [
            fqn for fqn, before_id in before_weight_ids.items()
            if id(after_modules[fqn].weight) == before_id
        ]
        if unchanged:
            raise _ExecutionFailure("TRANSFORM_REPRESENTATION_UNCHANGED", transform=True)
        non_target_before = {
            name: id(module.weight)
            for name, module in named_modules.items()
            if hasattr(module, "weight") and name not in all_targets
        }
        changed_non_targets = [
            fqn for fqn, before_id in non_target_before.items()
            if fqn in after_modules
            and hasattr(after_modules[fqn], "weight")
            and id(after_modules[fqn].weight) != before_id
        ]
        if changed_non_targets:
            raise _ExecutionFailure("TRANSFORM_NON_TARGET_CHANGED", transform=True)

    @staticmethod
    def _validate_group_structure(model: Any) -> None:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != 32:
            raise _ExecutionFailure("GROUP_STRUCTURE_UNSUPPORTED", transform=True)
        names = dict(model.named_modules())
        for group_index in range(8):
            targets = TorchAOProfileExecutor._group_targets(group_index)
            if not targets.issubset(names):
                raise _ExecutionFailure(
                    f"TRANSFORM_TARGET_MAPPING_INCOMPLETE_GROUP_{group_index}", transform=True
                )

    def _run_forward(self, model: Any, tokenizer: Any, prompt: str, variant_id: str) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise _ExecutionFailure("PYTORCH_UNAVAILABLE") from exc
        device_info = self._resolve_device()
        device = torch.device("cuda", device_info["device_index"])
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_length = int(encoded["input_ids"].shape[-1])
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions is not None and input_length + self.plan.data["decoder"]["max_new_tokens"] > max_positions:
            raise _ExecutionFailure("PROMPT_GENERATION_OVERFLOW")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prefix: list[dict[str, Any]] = []
        observed_groups: set[int] = set()
        handles = []
        layers = model.model.layers
        bits = "BF16" if variant_id == "BF16" else variant_id
        for group_index in range(8):
            layer = layers[(group_index + 1) * 4 - 1]

            def hook(module: Any, inputs: Any, output: Any, *, index: int = group_index) -> None:
                del module, inputs, output
                if index not in observed_groups:
                    observed_groups.add(index)
                    prefix.append({"group_index": index, "prefix_bits": bits[: index + 1]})

            handles.append(layer.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=True, return_dict=True)
                if not bool(torch.isfinite(outputs.logits).all().item()):
                    raise _ExecutionFailure("NON_FINITE_OUTPUT")
                generated = model.generate(
                    **encoded,
                    max_new_tokens=1024,
                    num_beams=4,
                    num_return_sequences=1,
                    do_sample=False,
                    early_stopping=True,
                    length_penalty=1.0,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    num_beam_groups=1,
                    diversity_penalty=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        except _ExecutionFailure:
            raise
        except RuntimeError as exc:
            reason = "OOM" if "out of memory" in str(exc).lower() else f"FORWARD_RUNTIME_ERROR:{type(exc).__name__}"
            raise _ExecutionFailure(reason) from exc
        finally:
            for handle in handles:
                handle.remove()
        if [entry["group_index"] for entry in prefix] != list(range(8)):
            raise _ExecutionFailure("INCOMPLETE_GROUP_ORDER")
        token_values = generated[0].detach().cpu().tolist()
        return {
            "status": "complete",
            "transform_status": "not_applicable" if variant_id == "BF16" else "complete",
            "forward_status": "complete",
            "reason_code": None,
            "transform_reason_code": "REFERENCE_UNQUANTIZED" if variant_id == "BF16" else None,
            "forward_reason_code": None,
            "observed_group_prefix": prefix,
            "token_count": len(token_values),
            "output_hash": hashlib.sha256(canonical_json_bytes(token_values)).hexdigest(),
        }

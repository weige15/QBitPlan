"""Real pinned-model TorchAO smoke executor.

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
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ..identity import canonical_json_bytes
from ..plan import (
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    TOKENIZER_FILE_HASHES,
    ExperimentPlan,
)


class _ExecutionFailure(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        *,
        transform: bool = False,
        phase: str | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.transform = transform
        self.phase = phase


class TorchAOProfileExecutor:
    """Execute the four issue-25 smoke variants on the pinned RTX 3090."""

    evidence_class = "directly measured"

    def __init__(self, plan: ExperimentPlan) -> None:
        self.plan = plan
        self._device_info: dict[str, Any] | None = None
        self._tokenizer: Any | None = None
        self._active_variant: str | None = None
        self._active_model: Any | None = None
        self._profile_failures: dict[str, tuple[str, bool, str | None]] = {}
        self._preparation_details: dict[str, dict[str, Any]] = {}

    def hardware_identity(self) -> Mapping[str, Any]:
        return dict(self._resolve_device())

    def prepare_variant(self, variant_id: str) -> Mapping[str, Any]:
        """Load and transform one profile without executing a query."""

        self._preparation_details.pop(variant_id, None)
        try:
            self._model_for_variant(variant_id)
            preparation: dict[str, Any] = {
                "status": "complete",
                "reason_code": "PREPARED",
                "transform_status": "not_applicable"
                if variant_id == "BF16"
                else "complete",
                "forward_status": "not_attempted",
            }
        except _ExecutionFailure as exc:
            phase = exc.phase or "setup"
            preparation = {
                "status": "invalid",
                "reason_code": exc.reason_code,
                "transform_status": (
                    "not_applicable"
                    if variant_id == "BF16"
                    else ("invalid" if exc.transform else "complete")
                ),
                "forward_status": "not_attempted",
            }
            preparation["failure_phase"] = phase
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
            preparation = {
                "status": "invalid",
                "reason_code": f"PREPARATION_EXCEPTION:{type(exc).__name__}",
                "transform_status": "invalid"
                if variant_id != "BF16"
                else "not_applicable",
                "forward_status": "not_attempted",
                "failure_phase": "setup",
            }
        preparation["preparation_phases"] = self._preparation_details.pop(
            variant_id, {}
        )
        return preparation

    def execute_prepared(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        *,
        trace: bool = False,
    ) -> Mapping[str, Any]:
        """Execute a query against the already prepared profile."""

        try:
            model = self._model_for_variant(variant_id)
            return self._run_forward(
                model,
                self._load_tokenizer(),
                query["prompt"],
                variant_id,
                trace=trace,
            )
        except _ExecutionFailure as exc:
            return self._failure_observation(
                variant_id, exc.reason_code, transform=exc.transform
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
            return self._failure_observation(
                variant_id, f"FORWARD_EXCEPTION:{type(exc).__name__}", transform=False
            )

    def release_variant(self, variant_id: str) -> None:
        """Release the active profile and its CUDA allocations."""

        if self._active_variant == variant_id:
            self._release_model()

    def execute(self, query: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
        return self.execute_prepared(query, variant_id)

    @staticmethod
    def _failure_observation(
        variant_id: str, reason_code: str, *, transform: bool
    ) -> dict[str, Any]:
        return {
            "status": "invalid",
            "transform_status": (
                "invalid"
                if transform
                else ("complete" if variant_id != "BF16" else "not_applicable")
            ),
            "forward_status": "not_attempted" if transform else "invalid",
            "reason_code": reason_code,
            "transform_reason_code": reason_code
            if transform
            else ("REFERENCE_UNQUANTIZED" if variant_id == "BF16" else None),
            "forward_reason_code": "TRANSFORM_FAILED" if transform else reason_code,
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
                physical_index = int(index)
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                logical_index: int | None
                if visible is None or not visible.strip():
                    logical_index = physical_index
                else:
                    visible_tokens = [
                        token.strip()
                        for token in visible.split(",")
                        if token.strip() and token.strip() != "-1"
                    ]
                    if gpu_uuid in visible_tokens:
                        logical_index = visible_tokens.index(gpu_uuid)
                    else:
                        logical_index = next(
                            (
                                logical
                                for logical, token in enumerate(visible_tokens)
                                if token.isdigit() and int(token) == physical_index
                            ),
                            None,
                        )
                    if logical_index is None:
                        raise _ExecutionFailure("GPU_UUID_NOT_VISIBLE")
                selected = {
                    "device_index": logical_index,
                    "physical_device_index": physical_index,
                    "gpu_name": name,
                    "gpu_uuid": uuid,
                    "driver_version": driver,
                    "cuda_visible_devices": visible or "unset",
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
            get_properties = getattr(torch.cuda, "get_device_properties", None)
            if callable(get_properties):
                properties = get_properties(selected["device_index"])
                actual_uuid = getattr(properties, "uuid", None)
                if actual_uuid is not None and str(actual_uuid) != gpu_uuid:
                    raise _ExecutionFailure("GPU_UUID_RUNTIME_MISMATCH")
            self._apply_determinism(torch)
        except ImportError as exc:
            raise _ExecutionFailure("PYTORCH_UNAVAILABLE") from exc
        self._device_info = selected
        return selected

    def _validate_software(self, torch: Any, driver_version: str) -> None:
        try:
            import accelerate  # type: ignore[import-untyped]
            import datasets  # type: ignore[import-untyped]
            import numpy as np
            import safetensors
            import torchao  # type: ignore[import-untyped]
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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.plan.data["determinism"][
            "cublas_workspace_config"
        ]
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
        self._verify_tokenizer_files()
        if not getattr(tokenizer, "is_fast", False):
            raise _ExecutionFailure("FAST_TOKENIZER_UNAVAILABLE")
        if tokenizer.eos_token_id is None:
            raise _ExecutionFailure("EOS_TOKEN_ID_UNAVAILABLE")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer
        return tokenizer

    def _verify_tokenizer_files(self) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise _ExecutionFailure("TOKENIZER_FILES_UNAVAILABLE") from exc
        try:
            snapshot_root = snapshot_download(
                MODEL_IDENTIFIER,
                revision=MODEL_REVISION,
                allow_patterns=list(TOKENIZER_FILE_HASHES),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _ExecutionFailure("TOKENIZER_FILES_UNAVAILABLE") from exc
        for filename, expected in TOKENIZER_FILE_HASHES.items():
            path = os.path.join(snapshot_root, filename)
            try:
                actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as exc:
                raise _ExecutionFailure("TOKENIZER_FILES_UNAVAILABLE") from exc
            if actual != expected:
                raise _ExecutionFailure("TOKENIZER_FILE_HASH_MISMATCH")

    def _model_for_variant(self, variant_id: str) -> Any:
        if variant_id != "BF16" and variant_id not in self.plan.data["profiles"]:
            raise _ExecutionFailure("PROFILE_NOT_DECLARED", transform=True)
        if variant_id in self._profile_failures:
            reason_code, transform, phase = self._profile_failures[variant_id]
            raise _ExecutionFailure(reason_code, transform=transform, phase=phase)
        if self._active_variant == variant_id and self._active_model is not None:
            return self._active_model
        self._release_model()
        details = self._preparation_details.setdefault(
            variant_id,
            {
                "placement": "cpu-before-device-transfer",
                "phase_status": "in_progress",
            },
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError:
            failure = _ExecutionFailure("MODEL_RUNTIME_UNAVAILABLE", phase="setup")
            self._profile_failures[variant_id] = (
                failure.reason_code,
                failure.transform,
                failure.phase,
            )
            raise failure from None
        device_info = self._resolve_device()
        device = torch.device("cuda", device_info["device_index"])
        details["device_index"] = device_info["device_index"]
        try:
            stage_started = time.perf_counter_ns()
            details["active_phase"] = "model_load"
            try:
                model = cast(
                    Any,
                    AutoModelForCausalLM.from_pretrained(
                        MODEL_IDENTIFIER,
                        revision=MODEL_REVISION,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=False,
                    ),
                )
                model.eval()
            finally:
                details["model_load_ns"] = time.perf_counter_ns() - stage_started

            if variant_id != "BF16":
                stage_started = time.perf_counter_ns()
                details["active_phase"] = "quantization"
                try:
                    self._transform_profile(model, variant_id)
                finally:
                    details["quantization_ns"] = time.perf_counter_ns() - stage_started

            stage_started = time.perf_counter_ns()
            details["active_phase"] = "group_validation"
            try:
                self._validate_group_structure(model)
            finally:
                details["group_validation_ns"] = time.perf_counter_ns() - stage_started

            stage_started = time.perf_counter_ns()
            details["active_phase"] = "device_transfer"
            try:
                model.to(device)
            finally:
                details["device_transfer_ns"] = time.perf_counter_ns() - stage_started

            if variant_id != "BF16":
                stage_started = time.perf_counter_ns()
                details["active_phase"] = "representation_validation"
                try:
                    self._validate_quantized_representation(model, variant_id, device)
                finally:
                    details["representation_validation_ns"] = (
                        time.perf_counter_ns() - stage_started
                    )
        except _ExecutionFailure as exc:
            if exc.phase is None:
                exc.phase = str(details.get("active_phase", "setup"))
            details["phase_status"] = "failed"
            details["failure_phase"] = exc.phase
            details.pop("active_phase", None)
            self._profile_failures[variant_id] = (
                exc.reason_code,
                exc.transform,
                exc.phase,
            )
            self._release_model()
            raise
        except RuntimeError as exc:
            phase = str(details.get("active_phase", "setup"))
            details["phase_status"] = "failed"
            details["failure_phase"] = phase
            details.pop("active_phase", None)
            reason = (
                "OOM"
                if "out of memory" in str(exc).lower()
                else "TRANSFORM_OR_LOAD_RUNTIME_ERROR"
            )
            failure = _ExecutionFailure(
                reason,
                transform=variant_id != "BF16",
                phase=phase,
            )
            self._profile_failures[variant_id] = (
                failure.reason_code,
                failure.transform,
                failure.phase,
            )
            self._release_model()
            raise failure from None
        except OSError:
            phase = str(details.get("active_phase", "model_load"))
            details["phase_status"] = "failed"
            details["failure_phase"] = phase
            details.pop("active_phase", None)
            failure = _ExecutionFailure(
                "MODEL_LOAD_FAILED",
                transform=variant_id != "BF16",
                phase=phase,
            )
            self._profile_failures[variant_id] = (
                failure.reason_code,
                failure.transform,
                failure.phase,
            )
            self._release_model()
            raise failure from None
        details["phase_status"] = "complete"
        details.pop("active_phase", None)
        self._active_variant = variant_id
        self._active_model = model
        return model

    @staticmethod
    def _validate_quantized_representation(
        model: Any, variant_id: str, device: Any
    ) -> None:
        try:
            from torchao.dtypes.affine_quantized_tensor import (  # type: ignore[import-untyped]
                AffineQuantizedTensor,
            )
        except ImportError as exc:
            raise _ExecutionFailure(
                "TORCHAO_REPRESENTATION_UNAVAILABLE", transform=True
            ) from exc
        modules = dict(model.named_modules())
        for group_index in range(8):
            for fqn in TorchAOProfileExecutor._group_targets(group_index):
                weight = getattr(modules[fqn], "weight", None)
                if not isinstance(weight, AffineQuantizedTensor):
                    raise _ExecutionFailure(
                        f"TRANSFORM_REPRESENTATION_NOT_TORCHAO:{variant_id}",
                        transform=True,
                    )
                if weight.device != device:
                    raise _ExecutionFailure(
                        "TRANSFORM_REPRESENTATION_DEVICE_MISMATCH", transform=True
                    )

    def _release_model(self) -> None:
        if self._active_model is not None:
            del self._active_model
            gc.collect()
        self._active_model = None
        self._active_variant = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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

    def _transform_profile(self, model: Any, variant_id: str) -> None:
        try:
            from torchao.quantization import (  # type: ignore[import-untyped]
                int4_weight_only,
                int8_weight_only,
                quantize_,
            )
        except ImportError as exc:
            raise _ExecutionFailure("TORCHAO_UNAVAILABLE", transform=True) from exc
        all_targets = {
            fqn for group_index in range(8) for fqn in self._group_targets(group_index)
        }
        named_modules = dict(model.named_modules())
        missing = sorted(all_targets - set(named_modules))
        if missing:
            raise _ExecutionFailure(
                "TRANSFORM_TARGET_MAPPING_INCOMPLETE", transform=True
            )
        before_weight_ids: dict[str, int] = {}
        for fqn in all_targets:
            module = named_modules[fqn]
            if not hasattr(module, "weight"):
                raise _ExecutionFailure(
                    "TRANSFORM_TARGET_WEIGHT_MISSING", transform=True
                )
            before_weight_ids[fqn] = id(module.weight)
        for group_index, bit in enumerate(variant_id):
            targets = self._group_targets(group_index)
            seen: set[str] = set()

            def filter_fn(
                module: Any,
                fqn: str,
                *,
                target_set: set[str] = targets,
                seen_set: set[str] = seen,
            ) -> bool:
                del module
                if fqn in target_set:
                    seen_set.add(fqn)
                    return True
                return False

            config = (
                int4_weight_only(group_size=128) if bit == "0" else int8_weight_only()
            )
            try:
                quantize_(model, config, filter_fn=filter_fn, set_inductor_config=False)
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
                raise _ExecutionFailure(
                    f"TRANSFORM_FAILED_GROUP_{group_index}:{type(exc).__name__}",
                    transform=True,
                ) from exc
            if seen != targets:
                raise _ExecutionFailure(
                    f"TRANSFORM_TARGET_MAPPING_INCOMPLETE_GROUP_{group_index}",
                    transform=True,
                )

        after_modules = dict(model.named_modules())
        unchanged = [
            fqn
            for fqn, before_id in before_weight_ids.items()
            if id(after_modules[fqn].weight) == before_id
        ]
        if unchanged:
            raise _ExecutionFailure(
                "TRANSFORM_REPRESENTATION_UNCHANGED", transform=True
            )
        non_target_before = {
            name: id(module.weight)
            for name, module in named_modules.items()
            if hasattr(module, "weight") and name not in all_targets
        }
        changed_non_targets = [
            fqn
            for fqn, before_id in non_target_before.items()
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
                    f"TRANSFORM_TARGET_MAPPING_INCOMPLETE_GROUP_{group_index}",
                    transform=True,
                )

    def _run_forward(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        variant_id: str,
        *,
        trace: bool = False,
    ) -> Mapping[str, Any]:
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
        if (
            max_positions is not None
            and input_length + self.plan.data["decoder"]["max_new_tokens"]
            > max_positions
        ):
            raise _ExecutionFailure("PROMPT_GENERATION_OVERFLOW")
        if trace:
            with torch.profiler.record_function("qbitplan.h2d"):
                encoded = {key: value.to(device) for key, value in encoded.items()}
        else:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        prefix: list[dict[str, Any]] = []
        observed_groups: set[int] = set()
        group_scopes: dict[int, Any] = {}
        handles = []
        layers = model.model.layers
        for group_index in range(8) if trace else ():
            layer = layers[(group_index + 1) * 4 - 1]

            bit = (
                "BF16"
                if variant_id == "BF16"
                else variant_id[group_index]
            )

            def pre_hook(
                module: Any,
                inputs: Any,
                *,
                index: int = group_index,
                profile_bit: str = bit,
            ) -> None:
                del module, inputs
                scope = torch.profiler.record_function(
                    f"qbitplan.group.{index}.{profile_bit}"
                )
                scope.__enter__()
                group_scopes[index] = scope

            def hook(
                module: Any, inputs: Any, output: Any, *, index: int = group_index
            ) -> None:
                del module, inputs, output
                scope = group_scopes.pop(index, None)
                if scope is not None:
                    scope.__exit__(None, None, None)
                if index not in observed_groups:
                    observed_groups.add(index)
                    prefix.append(
                        {
                            "group_index": index,
                            "prefix_bits": "BF16"
                            if variant_id == "BF16"
                            else variant_id[: index + 1],
                        }
                    )

            handles.append(layer.register_forward_pre_hook(pre_hook))
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
            reason = (
                "OOM"
                if "out of memory" in str(exc).lower()
                else f"FORWARD_RUNTIME_ERROR:{type(exc).__name__}"
            )
            raise _ExecutionFailure(reason) from exc
        finally:
            for scope in group_scopes.values():
                scope.__exit__(None, None, None)
            for handle in handles:
                handle.remove()
        if [entry["group_index"] for entry in prefix] != list(range(8)):
            if trace:
                raise _ExecutionFailure("INCOMPLETE_GROUP_ORDER")
            prefix = [
                {
                    "group_index": index,
                    "prefix_bits": "BF16"
                    if variant_id == "BF16"
                    else variant_id[: index + 1],
                }
                for index in range(8)
            ]
        token_values = generated[0].detach().cpu().tolist()
        completion_tokens = token_values[input_length:]
        result = {
            "status": "complete",
            "transform_status": "not_applicable"
            if variant_id == "BF16"
            else "complete",
            "forward_status": "complete",
            "reason_code": None,
            "transform_reason_code": "REFERENCE_UNQUANTIZED"
            if variant_id == "BF16"
            else None,
            "forward_reason_code": None,
            "observed_group_prefix": prefix,
            "token_count": len(completion_tokens),
            "generated_tokens": completion_tokens,
            "output_text": tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            ),
            "output_hash": hashlib.sha256(
                canonical_json_bytes(token_values)
            ).hexdigest(),
        }
        return result

    def teacher_forced_diagnostic(
        self,
        query: Mapping[str, Any],
        variant_id: str,
        reference_tokens: list[int],
    ) -> Mapping[str, Any]:
        """Compute full-vocabulary forward KL on the BF16 completion tokens."""

        try:
            import torch

            if not reference_tokens:
                raise _ExecutionFailure("DIAGNOSTIC_EMPTY_REFERENCE")
            tokenizer = self._load_tokenizer()
            device_info = self._resolve_device()
            device = torch.device("cuda", device_info["device_index"])
            encoded = tokenizer(
                query["prompt"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(device)
            continuation = torch.tensor(
                [reference_tokens], dtype=prompt_ids.dtype, device=device
            )
            teacher_input = torch.cat((prompt_ids, continuation[:, :-1]), dim=-1)

            def forward_logits(model: Any) -> Any:
                with torch.inference_mode():
                    result = model(
                        input_ids=teacher_input, use_cache=True, return_dict=True
                    )
                logits = result.logits.float()
                start = prompt_ids.shape[-1] - 1
                position_count = (
                    1 if query["dataset"] == "MMLU-Pro" else len(reference_tokens)
                )
                stop = start + position_count
                selected = logits[:, start:stop, :]
                if selected.shape[1] != position_count:
                    raise _ExecutionFailure("DIAGNOSTIC_LOGIT_LENGTH_MISMATCH")
                if not bool(torch.isfinite(selected).all().item()):
                    raise _ExecutionFailure("DIAGNOSTIC_NON_FINITE")
                return selected

            profile_model = self._model_for_variant(variant_id)
            max_positions = getattr(
                profile_model.config, "max_position_embeddings", None
            )
            if max_positions is not None and teacher_input.shape[-1] > max_positions:
                raise _ExecutionFailure("DIAGNOSTIC_CONTEXT_OVERFLOW")
            profile_logits = forward_logits(profile_model)
            if variant_id == "BF16":
                reference_logits = profile_logits
            else:
                reference_logits = forward_logits(self._model_for_variant("BF16"))
            reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
            profile_log_probs = torch.log_softmax(profile_logits, dim=-1)
            reference_probs = reference_log_probs.exp()
            kl = (
                (reference_probs * (reference_log_probs - profile_log_probs))
                .sum(dim=-1)
                .mean()
            )
            if not bool(torch.isfinite(kl).item()) or float(kl.item()) < 0:
                raise _ExecutionFailure("DIAGNOSTIC_NON_FINITE")
            return {
                "status": "complete",
                "reason_code": "completed",
                "kl_mean": float(kl.item()),
                "evidence_class": self.evidence_class,
            }
        except _ExecutionFailure as exc:
            return {"status": "invalid", "reason_code": exc.reason_code}
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
                "reason_code": f"DIAGNOSTIC_EXCEPTION:{type(exc).__name__}",
            }

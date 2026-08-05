from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qbitplan.plan import ExperimentPlan
from qbitplan.stage1.executor import TorchAOProfileExecutor, _ExecutionFailure, _FailureMetadata


class _FakeModel:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.model = SimpleNamespace(layers=[object() for _ in range(32)])
        self._target_modules = {
            fqn: SimpleNamespace(weight=object())
            for group_index in range(8)
            for fqn in TorchAOProfileExecutor._group_targets(group_index)
        }

    def named_modules(self):
        yield "", self
        yield from self._target_modules.items()

    def to(self, device: Any) -> _FakeModel:
        self.events.append(("to", device))
        return self

    def eval(self) -> _FakeModel:
        self.events.append(("eval",))
        return self


class _FakeAutoModel:
    events: list[Any]

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeModel:
        cls.events.append(("load", args, kwargs))
        return _FakeModel(cls.events)


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, events: list[Any]) -> None:
    torch = ModuleType("torch")
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.device = lambda kind, index=None: (kind, index)  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        empty_cache=lambda: events.append(("empty_cache",)),
    )

    transformers = ModuleType("transformers")
    _FakeAutoModel.events = events
    transformers.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]

    torchao_quantization = ModuleType("torchao.quantization")
    torchao_quantization.int4_weight_only = lambda group_size: ("int4", group_size)  # type: ignore[attr-defined]
    torchao_quantization.int8_weight_only = lambda: ("int8",)  # type: ignore[attr-defined]

    def quantize_(model: _FakeModel, config: Any, **kwargs: Any) -> None:
        events.append(("quantize", config, dict(kwargs)))
        filter_fn = kwargs["filter_fn"]
        for fqn, module in model.named_modules():
            if filter_fn(module, fqn):
                module.weight = object()

    torchao_quantization.quantize_ = quantize_  # type: ignore[attr-defined]
    torchao = ModuleType("torchao")
    torchao.quantization = torchao_quantization  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", torchao_quantization)


def _executor() -> TorchAOProfileExecutor:
    return TorchAOProfileExecutor(
        ExperimentPlan(
            {
                "profiles": ["BF16", "00000000"],
                "gpu_uuid": "test-gpu",
            }
        )
    )


def test_quantized_variant_is_prepared_on_cpu_before_cuda_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    _install_fake_runtime(monkeypatch, events)
    executor = _executor()

    monkeypatch.setattr(executor, "_resolve_device", lambda: {"device_index": 2})
    model = executor._model_for_variant("00000000")

    assert model is executor._active_model
    quantize_events = [event for event in events if event[0] == "quantize"]
    assert len(quantize_events) == 8
    assert all("device" not in event[2] for event in quantize_events)
    assert all(event[2]["set_inductor_config"] is False for event in quantize_events)
    cuda_transfer_index = events.index(("to", ("cuda", 2)))
    assert all(events.index(event) < cuda_transfer_index for event in quantize_events)
    assert events.index(("to", ("cpu", None))) < events.index(quantize_events[0])


def test_failed_profiles_cache_traceback_free_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    _install_fake_runtime(monkeypatch, events)
    executor = _executor()

    monkeypatch.setattr(executor, "_resolve_device", lambda: {"device_index": 0})

    transform_calls = 0

    def fail_transform(model: Any, variant_id: str) -> None:
        nonlocal transform_calls
        del model, variant_id
        transform_calls += 1
        raise _ExecutionFailure(
            "TRANSFORM_FAILED_GROUP_0:OutOfMemoryError",
            transform=True,
            failure_metadata=_FailureMetadata(
                failure_phase="cpu_transform",
                group_index=0,
                bit_width=4,
                failing_fqn="model.layers.0.self_attn.q_proj",
                exception_type="ValueError",
                exception_message="TensorCoreTiledAQTLayout is only available for cuda",
            ),
        )

    monkeypatch.setattr(executor, "_transform_profile", fail_transform)
    monkeypatch.setattr(executor, "_validate_group_structure", lambda model: None)

    first = executor.execute({"prompt": "unused"}, "00000000")
    cached = executor._profile_failures["00000000"]
    second = executor.execute({"prompt": "unused"}, "00000000")

    assert first["reason_code"] == "TRANSFORM_FAILED_GROUP_0:OutOfMemoryError"
    assert second["reason_code"] == first["reason_code"]
    assert first["transform_status"] == second["transform_status"] == "invalid"
    assert first["cuda_transfer_status"] == second["cuda_transfer_status"] == "not_attempted"
    assert first["forward_status"] == second["forward_status"] == "not_attempted"
    assert first["failure_metadata"] == {
        "failure_phase": "cpu_transform",
        "group_index": 0,
        "bit_width": 4,
        "failing_fqn": "model.layers.0.self_attn.q_proj",
        "exception_type": "ValueError",
        "exception_message": "TensorCoreTiledAQTLayout is only available for cuda",
    }
    assert transform_calls == 1
    assert not isinstance(cached, BaseException)
    assert cached.reason_code == first["reason_code"]
    assert cached.transform is True
    assert cached.failure_metadata == _FailureMetadata(
        failure_phase="cpu_transform",
        group_index=0,
        bit_width=4,
        failing_fqn="model.layers.0.self_attn.q_proj",
        exception_type="ValueError",
        exception_message="TensorCoreTiledAQTLayout is only available for cuda",
    )


def test_releasing_active_model_clears_identity_before_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    _install_fake_runtime(monkeypatch, events)
    executor = _executor()
    executor._active_variant = "00000000"
    executor._active_model = object()

    executor._release_model()

    assert executor._active_model is None
    assert executor._active_variant is None
    assert events[-1] == ("empty_cache",)

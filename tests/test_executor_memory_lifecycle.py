from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qbitplan.plan import ExperimentPlan
from qbitplan.stage1.executor import TorchAOProfileExecutor, _ExecutionFailure


class _FakeModel:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

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

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


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
    monkeypatch.setattr(
        executor,
        "_transform_profile",
        lambda model, variant_id, device: events.append(
            ("transform", variant_id, device)
        ),
    )
    monkeypatch.setattr(
        executor,
        "_validate_group_structure",
        lambda model: events.append(("validate",)),
    )

    model = executor._model_for_variant("00000000")

    assert model is executor._active_model
    lifecycle = [event for event in events if event[0] != "empty_cache"]
    assert lifecycle == [
        (
            "load",
            ("meta-llama/Llama-3.1-8B",),
            {
                "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "torch_dtype": sys.modules["torch"].bfloat16,
                "trust_remote_code": False,
            },
        ),
        ("to", ("cpu", None)),
        ("eval",),
        ("transform", "00000000", ("cpu", None)),
        ("validate",),
        ("to", ("cuda", 2)),
    ]


def test_failed_profiles_cache_traceback_free_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    _install_fake_runtime(monkeypatch, events)
    executor = _executor()

    monkeypatch.setattr(executor, "_resolve_device", lambda: {"device_index": 0})

    transform_calls = 0

    def fail_transform(model: Any, variant_id: str, device: Any) -> None:
        nonlocal transform_calls
        del model, variant_id, device
        transform_calls += 1
        raise _ExecutionFailure(
            "TRANSFORM_FAILED_GROUP_0:OutOfMemoryError",
            transform=True,
        )

    monkeypatch.setattr(executor, "_transform_profile", fail_transform)
    monkeypatch.setattr(executor, "_validate_group_structure", lambda model: None)

    first = executor.execute({"prompt": "unused"}, "00000000")
    cached = executor._profile_failures["00000000"]
    second = executor.execute({"prompt": "unused"}, "00000000")

    assert first["reason_code"] == "TRANSFORM_FAILED_GROUP_0:OutOfMemoryError"
    assert second["reason_code"] == first["reason_code"]
    assert first["transform_status"] == second["transform_status"] == "invalid"
    assert transform_calls == 1
    assert not isinstance(cached, BaseException)
    assert cached.reason_code == first["reason_code"]
    assert cached.transform is True


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

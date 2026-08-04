"""Accepted smoke-plan configuration materialization."""

from __future__ import annotations

from typing import Any

from .plan import (
    EXPECTED_DECODER,
    EXPECTED_DETERMINISM,
    EXPECTED_RUNTIME,
    EXPECTED_SOFTWARE,
    TOKENIZER_FILE_HASHES,
)


def accepted_smoke_configuration(
    *, artifact_root: str, attempt_id: str, gpu_uuid: str
) -> dict[str, Any]:
    """Return all non-source fields required by the issue-25 smoke plan."""

    return {
        "schema_version": "qbitplan.stage1.experiment-plan.v1",
        "artifact_root": artifact_root,
        "attempt_id": attempt_id,
        "mode": "smoke",
        "model": {
            "identifier": "meta-llama/Llama-3.1-8B",
            "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
            "architecture": "LlamaForCausalLM",
            "layers": 32,
            "dtype": "bfloat16",
        },
        "tokenizer": {
            "identifier": "meta-llama/Llama-3.1-8B",
            "file_hashes": dict(TOKENIZER_FILE_HASHES),
            "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
            "use_fast": True,
            "trust_remote_code": False,
        },
        "software": dict(EXPECTED_SOFTWARE),
        "decoder": dict(EXPECTED_DECODER),
        "runtime": dict(EXPECTED_RUNTIME),
        "seed": 20260807,
        "determinism": dict(EXPECTED_DETERMINISM),
        "gpu_uuid": gpu_uuid,
        "profiles": ["BF16", "00000000", "11111111", "01010101"],
    }

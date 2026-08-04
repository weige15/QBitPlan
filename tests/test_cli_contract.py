from __future__ import annotations

import pytest

from qbitplan.cli import main


def test_scientific_cli_has_no_fake_adapter_selector(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "stage1",
                "run",
                "--plan",
                "plan.json",
                "--mode",
                "smoke",
                "--gpu-uuid",
                "GPU-test",
                "--executor",
                "fake",
            ]
        )

    assert error.value.code == 2
    assert "unrecognized arguments: --executor fake" in capsys.readouterr().err

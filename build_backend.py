"""Dependency-free PEP 517/660 backend for the small stdlib package."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zipfile


NAME = "qbitplan"
VERSION = "0.1.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"


def _metadata() -> str:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Query-conditioned causal bit planning for budgeted LLM inference\n"
        "Requires-Python: >=3.12,<3.13\n"
    )


def _wheel_metadata() -> str:
    return "Wheel-Version: 1.0\nGenerator: qbitplan-build-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n"


def _entry_points() -> str:
    return "[console_scripts]\nqbitplan = qbitplan.cli:main\n"


def _dist_info(directory: Path) -> Path:
    target = directory / DIST_INFO
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_text(_metadata(), encoding="utf-8")
    (target / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8")
    (target / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")
    return target


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings: dict | None = None) -> str:
    del config_settings
    return _dist_info(Path(metadata_directory)).name


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: dict | None = None) -> str:
    del config_settings
    return _dist_info(Path(metadata_directory)).name


def get_requires_for_build_editable(config_settings: dict | None = None) -> list[str]:
    del config_settings
    return []


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    del config_settings
    return []


def _record_path(path: str, content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
    return f"{path},sha256={digest},{len(content)}\n"


def _build(wheel_directory: str, editable: bool) -> str:
    wheel_name = f"{NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    source_root = Path(__file__).resolve().parent
    files: dict[str, bytes] = {}
    if editable:
        files["qbitplan_editable.pth"] = (str(source_root) + "\n").encode("utf-8")
    else:
        for path in (source_root / "qbitplan").rglob("*.py"):
            files[path.relative_to(source_root).as_posix()] = path.read_bytes()
    files[f"{DIST_INFO}/METADATA"] = _metadata().encode("utf-8")
    files[f"{DIST_INFO}/WHEEL"] = _wheel_metadata().encode("utf-8")
    files[f"{DIST_INFO}/entry_points.txt"] = _entry_points().encode("utf-8")
    record_path = f"{DIST_INFO}/RECORD"
    record = "".join(_record_path(path, content) for path, content in files.items())
    record += f"{record_path},,\n"
    files[record_path] = record.encode("utf-8")
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return wheel_name


def build_editable(wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None) -> str:
    del config_settings, metadata_directory
    return _build(wheel_directory, editable=True)


def build_wheel(wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None) -> str:
    del config_settings, metadata_directory
    return _build(wheel_directory, editable=False)

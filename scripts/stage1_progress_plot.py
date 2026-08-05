"""Render the local-only Stage-1 progress figure from immutable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_METHODS = (
    ("static", "Static", "#A8B0BF"),
    ("direct", "Direct query-only", "#2F6BFF"),
    ("independent", "Independent", "#F2A93B"),
    ("interaction", "Interaction-aware", "#14A87B"),
)
_SMOKE_PROFILES = (
    ("BF16", "BF16", "#475569"),
    ("00000000", "all-4", "#2F6BFF"),
    ("01010101", "mixed", "#14A87B"),
    ("11111111", "all-8", "#F2A93B"),
)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = "#172033",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{value}</text>'
    )


def _bar_chart(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    values: list[tuple[str, str, float, str]],
    y_max: float,
    y_label: str,
    value_formatter,
) -> None:
    horizontal_axis = (
        f'<line x1="{x:.1f}" y1="{y + height:.1f}" x2="{x + width:.1f}" '
        f'y2="{y + height:.1f}" stroke="#98A2B3"/>'
    )
    vertical_axis = (
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" '
        f'y2="{y + height:.1f}" stroke="#98A2B3"/>'
    )
    parts.extend((horizontal_axis, vertical_axis, _text(x, y - 16, y_label, size=13, fill="#5B6475")))
    for tick in range(6):
        value = y_max * tick / 5
        tick_y = y + height - height * value / y_max
        parts.append(
            f'<line x1="{x:.1f}" y1="{tick_y:.1f}" x2="{x + width:.1f}" '
            f'y2="{tick_y:.1f}" stroke="#E7EBF1"/>'
        )
        parts.append(
            _text(
                x - 10,
                tick_y + 5,
                value_formatter(value),
                size=11,
                fill="#5B6475",
                anchor="end",
            )
        )
    bar_width = width / (len(values) * 1.35)
    gap = bar_width * 0.35
    start = x + (width - (len(values) * bar_width + (len(values) - 1) * gap)) / 2
    for index, (key, label, value, color) in enumerate(values):
        bar_x = start + index * (bar_width + gap)
        bar_height = height * value / y_max
        bar_y = y + height - bar_height
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="7" fill="{color}"/>'
        )
        parts.append(
            _text(
                bar_x + bar_width / 2,
                max(y + 16, bar_y - 9),
                value_formatter(value),
                size=12,
                weight=700,
                anchor="middle",
            )
        )
        parts.append(
            _text(
                bar_x + bar_width / 2,
                y + height + 27,
                label,
                size=11,
                anchor="middle",
            )
        )
        parts.append(_text(bar_x + bar_width / 2, y + height + 43, key, size=9, fill="#7A8495", anchor="middle"))


def render(controller_metrics: dict[str, Any], smoke_summary: dict[str, Any]) -> str:
    width = 1280
    height = 700
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F9FC"/>',
        '<rect x="22" y="22" width="1236" height="656" rx="20" fill="white" stroke="#DCE2EC"/>',
        _text(56, 68, "QBitPlan Stage-1 progress", size=28, weight=700),
        _text(56, 96, "Two evidence classes shown separately; this figure makes no combined quality-cost claim.", size=14, fill="#5B6475"),
        '<rect x="48" y="124" width="570" height="470" rx="14" fill="#F3F6FF" stroke="#C9D7FF"/>',
        '<rect x="646" y="124" width="586" height="470" rx="14" fill="#F2FBF7" stroke="#BFE8D7"/>',
        _text(76, 162, "Simulated controller match rates", size=19, weight=700),
        _text(76, 186, "Synthetic target-profile equality/membership only", size=12, fill="#5B6475"),
        _text(76, 205, "Evidence class: simulated | non-evidentiary", size=12, fill="#5B6475"),
        _text(674, 162, "Directly measured smoke resident bytes", size=19, weight=700),
        _text(674, 186, "Absolute NVML device-used bytes; baseline not subtracted", size=12, fill="#5B6475"),
        _text(674, 205, "Evidence class: directly measured | smoke only", size=12, fill="#5B6475"),
    ]

    method_values = [
        (
            key,
            label,
            100.0 * float(controller_metrics["methods"][key]["synthetic_target_profile_match_rate"]),
            color,
        )
        for key, label, color in _METHODS
    ]
    _bar_chart(
        parts,
        x=92,
        y=246,
        width=480,
        height=270,
        values=method_values,
        y_max=100.0,
        y_label="match rate (%)",
        value_formatter=lambda value: f"{value:.1f}%",
    )

    peaks = smoke_summary["nvml_profile_peaks"]
    memory_values = [
        (key, label, float(peaks[key]["peak_absolute_nvml_bytes"]) / 2**30, color)
        for key, label, color in _SMOKE_PROFILES
    ]
    _bar_chart(
        parts,
        x=696,
        y=246,
        width=490,
        height=270,
        values=memory_values,
        y_max=20.0,
        y_label="absolute resident accelerator bytes (GiB)",
        value_formatter=lambda value: f"{value:.0f}",
    )

    parts.extend(
        (
            _text(76, 558, "Not Stage-1 quality, external correctness, memory benefit, latency benefit, or serving evidence.", size=12, weight=600, fill="#7A3340"),
            _text(674, 558, "Host-to-device bytes, kernel latency, prefetch stalls, kernel switches, and end-to-end latency: unavailable in this smoke artifact.", size=12, weight=600, fill="#7A3340"),
            _text(56, 632, "Smoke absolute resident bytes are not plotted against average bit-width and are not used as a simulated cost proxy.", size=13, weight=700, fill="#172033"),
            _text(56, 654, "The panels must not be read as a single metric or as evidence that the controller changes hardware cost.", size=12, fill="#5B6475"),
            "</svg>",
        )
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-metrics", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller_metrics = json.loads(args.controller_metrics.read_text(encoding="utf-8"))
    smoke_summary = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(controller_metrics, smoke_summary), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

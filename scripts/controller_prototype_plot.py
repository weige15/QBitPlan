"""Dependency-light SVG report for the controller prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}">{text}</text>'
    )


def _write_svg(metrics: dict[str, Any], path: Path) -> None:
    width = 1180
    height = 620
    methods = ("static", "direct", "independent", "interaction")
    labels = {
        "static": "Static",
        "direct": "Direct query-only",
        "independent": "Independent",
        "interaction": "Interaction-aware",
    }
    colors = {
        "static": "#A8B0BF",
        "direct": "#2F6BFF",
        "independent": "#F2A93B",
        "interaction": "#14A87B",
    }
    method_metrics = metrics["methods"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F9FC"/>',
        '<rect x="24" y="24" width="1132" height="572" rx="18" fill="white" '
        'stroke="#DCE2EC"/>',
        _svg_text(58, 72, "QBitPlan controller prototype", size=26, weight=700),
        _svg_text(
            58,
            100,
            "Deterministic synthetic fixture — simulated, non-evidentiary",
            size=14,
            fill="#5B6475",
        ),
        _svg_text(80, 148, "Exact profile hit rate", size=18, weight=700),
        _svg_text(650, 148, "Per-group bit accuracy", size=18, weight=700),
    ]

    chart_x = 80
    chart_y = 180
    chart_w = 470
    chart_h = 310
    parts.append(
        f'<line x1="{chart_x}" y1="{chart_y + chart_h}" '
        f'x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" '
        'stroke="#98A2B3"/>'
    )
    parts.append(
        f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" '
        f'y2="{chart_y + chart_h}" stroke="#98A2B3"/>'
    )
    for tick in range(0, 101, 20):
        y = chart_y + chart_h - chart_h * tick / 100
        parts.append(
            f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" '
            f'y2="{y:.1f}" stroke="#E7EBF1"/>'
        )
        parts.append(_svg_text(chart_x - 12, y + 5, f"{tick}%", anchor="end", size=12))

    bar_width = 82
    gap = 30
    for index, method in enumerate(methods):
        value = float(method_metrics[method]["feasible_profile_hit_rate"])
        bar_height = chart_h * value
        x = chart_x + 25 + index * (bar_width + gap)
        y = chart_y + chart_h - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" '
            f'height="{bar_height:.1f}" rx="7" fill="{colors[method]}"/>'
        )
        parts.append(
            _svg_text(
                x + bar_width / 2,
                max(chart_y + 20, y - 10),
                f"{value * 100:.1f}%",
                anchor="middle",
                size=13,
                weight=700,
            )
        )
        parts.append(
            _svg_text(
                x + bar_width / 2,
                chart_y + chart_h + 25,
                labels[method],
                anchor="middle",
                size=11,
            )
        )

    line_x = 650
    line_y = 180
    line_w = 450
    line_h = 310
    parts.append(
        f'<line x1="{line_x}" y1="{line_y + line_h}" '
        f'x2="{line_x + line_w}" y2="{line_y + line_h}" '
        'stroke="#98A2B3"/>'
    )
    parts.append(
        f'<line x1="{line_x}" y1="{line_y}" x2="{line_x}" '
        f'y2="{line_y + line_h}" stroke="#98A2B3"/>'
    )
    for tick in range(0, 101, 20):
        y = line_y + line_h - line_h * tick / 100
        parts.append(
            f'<line x1="{line_x}" y1="{y:.1f}" x2="{line_x + line_w}" '
            f'y2="{y:.1f}" stroke="#E7EBF1"/>'
        )
        parts.append(_svg_text(line_x - 12, y + 5, f"{tick}%", anchor="end", size=12))
    for group_index in range(8):
        x = line_x + group_index * line_w / 7
        parts.append(
            f'<line x1="{x:.1f}" y1="{line_y + line_h}" x2="{x:.1f}" '
            f'y2="{line_y + line_h + 5}" stroke="#98A2B3"/>'
        )
        parts.append(
            _svg_text(x, line_y + line_h + 25, f"g{group_index}", anchor="middle", size=12)
        )

    for method in ("independent", "interaction"):
        values = method_metrics[method]["per_group_bit_accuracy"]
        points: list[str] = []
        for group_index, value in enumerate(values):
            x = line_x + group_index * line_w / 7
            y = line_y + line_h - line_h * float(value)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[method]}" stroke-width="3"/>'
        )
        for point in points:
            x_value, y_value = point.split(",")
            parts.append(
                f'<circle cx="{x_value}" cy="{y_value}" r="5" '
                f'fill="{colors[method]}" stroke="white" stroke-width="2"/>'
            )

    legend_y = 535
    for index, method in enumerate(("independent", "interaction")):
        x = 680 + index * 210
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 32}" y2="{legend_y}" '
            f'stroke="{colors[method]}" stroke-width="4"/>'
        )
        parts.append(_svg_text(x + 42, legend_y + 5, labels[method], size=13))

    parts.append(
        _svg_text(
            58,
            570,
            (
                "No LLM, GPU, quantized weight, task-correctness, memory, "
                "or latency result is represented."
            ),
            size=13,
            fill="#7A3340",
            weight=600,
        )
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")

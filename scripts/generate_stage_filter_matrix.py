#!/usr/bin/env python3
"""Render a slide-ready curriculum stage filter matrix as PNG and SVG."""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from staged_rl.config import build_default_stage_specs


PLOTS_DIR = ROOT / "results" / "plots"

STAGE_ORDER = [
    "stage1_easy_numeric",
    "stage2_float_numeric",
    "stage3_hard_numeric",
]

STAGE_COLORS = {
    "stage1_easy_numeric": "#4c78a8",
    "stage2_float_numeric": "#f58518",
    "stage3_hard_numeric": "#e45756",
}

STAGE_TITLES = {
    "stage1_easy_numeric": "S1\nEasy Numeric",
    "stage2_float_numeric": "S2\nMedium / Float",
    "stage3_hard_numeric": "S3\nHard Numeric",
}

STAGE_GOALS = {
    "stage1_easy_numeric": "Structure and easy numeric stabilization",
    "stage2_float_numeric": "Broader numeric coverage and moderate precision",
    "stage3_hard_numeric": "Harder multistep numeric reasoning",
}


@dataclass(frozen=True)
class TableSpec:
    col_widths: tuple[float, ...]
    header_height: float
    row_heights: tuple[float, ...]


def _lighten(color: str, amount: float = 0.86) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def _join_items(values: tuple[str, ...], *, wrap_width: int | None = None) -> str:
    cleaned = [value.replace("_", " ") for value in values if value]
    text = ", ".join(cleaned) if cleaned else "-"
    return textwrap.fill(text, width=wrap_width) if wrap_width else text


def _priority_text(stage_spec) -> str:
    parts = []
    if stage_spec.priority_context_families:
        parts.append("Contexts: " + ", ".join(stage_spec.priority_context_families))
    if stage_spec.priority_skills:
        parts.append("Skills: " + ", ".join(stage_spec.priority_skills))
    if stage_spec.priority_grades:
        parts.append("Grades: " + ", ".join(stage_spec.priority_grades))
    if stage_spec.hard_only:
        parts.append("Extra boost for high school / college")
    return textwrap.fill("; ".join(parts), width=32)


def _stage_matrix_rows() -> list[tuple[str, dict[str, str]]]:
    specs = build_default_stage_specs()
    rows: list[tuple[str, dict[str, str]]] = []
    rows.append(("Stage ID", {name: name for name in STAGE_ORDER}))
    rows.append(("Goal", {name: textwrap.fill(STAGE_GOALS[name], width=28) for name in STAGE_ORDER}))
    rows.append(("Question type", {name: _join_items(specs[name].filter_spec.question_types) for name in STAGE_ORDER}))
    rows.append(("Answer type", {name: _join_items(specs[name].filter_spec.answer_types) for name in STAGE_ORDER}))
    rows.append(("Language", {name: _join_items(specs[name].filter_spec.languages) for name in STAGE_ORDER}))
    rows.append(("Answer mode", {name: specs[name].answer_mode for name in STAGE_ORDER}))
    rows.append(
        (
            "Context families",
            {name: textwrap.fill(_join_items(specs[name].filter_spec.context_families), width=30) for name in STAGE_ORDER},
        )
    )
    rows.append(
        (
            "Skills (any)",
            {name: textwrap.fill(_join_items(specs[name].filter_spec.skills_any), width=30) for name in STAGE_ORDER},
        )
    )
    rows.append(("Hard only", {name: "True" if specs[name].hard_only else "False" for name in STAGE_ORDER}))
    rows.append(("Priority bias", {name: _priority_text(specs[name]) for name in STAGE_ORDER}))
    return rows


def _draw_cell(ax, x: float, y: float, w: float, h: float, *, facecolor: str, edgecolor: str, linewidth: float = 1.2):
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1,
        )
    )


def _draw_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float,
    weight: str = "normal",
    ha: str = "center",
    linespacing: float = 1.12,
):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1f1f1f",
        linespacing=linespacing,
        zorder=2,
    )


def _line_count(text: str) -> int:
    return max(1, len(str(text).splitlines()))


def render_stage_filter_matrix() -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _stage_matrix_rows()
    row_units = []
    for label, values in rows:
        max_lines = max([_line_count(label)] + [_line_count(values[stage_name]) for stage_name in STAGE_ORDER])
        base_units = 1.0
        extra_per_line = 0.62
        if label == "Priority bias":
            base_units = 1.35
            extra_per_line = 0.76
        elif label == "Skills (any)":
            base_units = 1.18
            extra_per_line = 0.70
        elif label == "Context families":
            base_units = 1.14
            extra_per_line = 0.68
        elif label == "Goal":
            base_units = 1.10
            extra_per_line = 0.64
        row_units.append(base_units + extra_per_line * (max_lines - 1))

    header_units = 1.95
    total_units = header_units + sum(row_units)
    spec = TableSpec(
        col_widths=(0.18, 0.273, 0.273, 0.274),
        header_height=header_units / total_units,
        row_heights=tuple(unit / total_units for unit in row_units),
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.955, "Curriculum Stage Filter Matrix", ha="left", va="top", fontsize=24, fontweight="bold")
    ax.text(
        0.05,
        0.905,
        "Each stage is a filtered view of the same MathVista split. Stages can overlap.",
        ha="left",
        va="top",
        fontsize=12.5,
        color="#4f4f4f",
    )

    table_left = 0.05
    table_bottom = 0.155
    table_width = 0.90
    table_height = 0.69

    x_positions = [table_left]
    for width in spec.col_widths:
        x_positions.append(x_positions[-1] + table_width * width)

    y_positions = [table_bottom + table_height]
    y_positions.append(y_positions[-1] - table_height * spec.header_height)
    for height in spec.row_heights:
        y_positions.append(y_positions[-1] - table_height * height)

    header_y_top = y_positions[0]
    header_y_bottom = y_positions[1]

    _draw_cell(ax, x_positions[0], header_y_bottom, x_positions[1] - x_positions[0], header_y_top - header_y_bottom, facecolor="#f3f4f6", edgecolor="#d0d4d9", linewidth=1.5)
    _draw_text(ax, (x_positions[0] + x_positions[1]) / 2, (header_y_top + header_y_bottom) / 2, "Filter", fontsize=13, weight="bold")

    for index, stage_name in enumerate(STAGE_ORDER, start=1):
        stage_color = STAGE_COLORS[stage_name]
        _draw_cell(
            ax,
            x_positions[index],
            header_y_bottom,
            x_positions[index + 1] - x_positions[index],
            header_y_top - header_y_bottom,
            facecolor=stage_color,
            edgecolor=stage_color,
            linewidth=1.5,
        )
        ax.text(
            (x_positions[index] + x_positions[index + 1]) / 2,
            (header_y_top + header_y_bottom) / 2 - 0.002,
            STAGE_TITLES[stage_name],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
            linespacing=1.12,
            zorder=2,
        )

    for row_index, (label, values) in enumerate(rows, start=1):
        row_y_top = y_positions[row_index]
        row_y_bottom = y_positions[row_index + 1]
        row_height = row_y_top - row_y_bottom
        label_fill = "#fafafa" if row_index % 2 else "#f5f6f8"
        _draw_cell(
            ax,
            x_positions[0],
            row_y_bottom,
            x_positions[1] - x_positions[0],
            row_height,
            facecolor=label_fill,
            edgecolor="#d8dce2",
        )
        _draw_text(
            ax,
            x_positions[0] + 0.012,
            (row_y_top + row_y_bottom) / 2,
            label,
            fontsize=11,
            weight="bold",
            ha="left",
        )

        for stage_col, stage_name in enumerate(STAGE_ORDER, start=1):
            facecolor = _lighten(STAGE_COLORS[stage_name], 0.82 if row_index % 2 else 0.88)
            _draw_cell(
                ax,
                x_positions[stage_col],
                row_y_bottom,
                x_positions[stage_col + 1] - x_positions[stage_col],
                row_height,
                facecolor=facecolor,
                edgecolor="#d8dce2",
            )
            text = values[stage_name]
            fontsize = 10.2
            if label == "Stage ID":
                fontsize = 8.8
            elif label == "Goal":
                fontsize = 9.4
            elif label in {"Context families", "Skills (any)", "Priority bias"}:
                fontsize = 9.6
            _draw_text(
                ax,
                (x_positions[stage_col] + x_positions[stage_col + 1]) / 2,
                (row_y_top + row_y_bottom) / 2,
                text,
                fontsize=fontsize,
                linespacing=1.10,
            )

    footer_y = 0.085
    ax.text(
        0.05,
        footer_y,
        "Stage 1 prioritizes easier integer numeric examples. Stage 2 broadens to medium / float-style numeric reasoning. Stage 3 targets harder geometry, algebra, and scientific numeric tasks.",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#444444",
    )
    ax.text(
        0.05,
        0.053,
        "Important: the stages are filtered views, not disjoint partitions. One example may match more than one stage.",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#444444",
        fontweight="bold",
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / "stage_filter_matrix.png"
    svg_path = PLOTS_DIR / "stage_filter_matrix.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    png_path, svg_path = render_stage_filter_matrix()
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a DeepSeek-style high-level training pipeline plot for the staged RL repo."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from staged_rl.config import build_default_run_config


PLOTS_DIR = ROOT / "results" / "plots"


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    w: float
    h: float

    def left(self, dy: float = 0.0) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2 + dy)

    def right(self, dy: float = 0.0) -> tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2 + dy)

    def top(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.x + self.w / 2 + dx, self.y + self.h)

    def bottom(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.x + self.w / 2 + dx, self.y)

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


def _lighten(color: str, amount: float = 0.16) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def add_round_box(
    ax,
    node: Node,
    text: str,
    *,
    fill: str,
    edge: str,
    fontsize: float = 12,
    weight: str = "normal",
    rounding: float = 1.2,
    linewidth: float = 1.7,
    zorder: int = 3,
):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=fill,
        zorder=zorder,
    )
    ax.add_patch(patch)
    text_obj = ax.text(
        node.x + node.w / 2,
        node.y + node.h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1f1f1f",
        zorder=zorder + 1,
    )
    fit_text_in_node(ax, text_obj, node, pad_x=0.55, pad_y=0.42, min_fontsize=7.0)
    return patch


def add_container(
    ax,
    node: Node,
    title: str,
    *,
    fill: str = "#fbfbfb",
    edge: str = "#8a8a8a",
    title_size: float = 15,
):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle="round,pad=0.04,rounding_size=1.6",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        node.x + 1.5,
        node.y + node.h - 1.5,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#222222",
        zorder=8,
        bbox={"facecolor": fill, "edgecolor": "none", "pad": 0.22},
    )
    return patch


def add_cylinder(
    ax,
    node: Node,
    text: str,
    *,
    fill: str,
    edge: str,
    fontsize: float = 12,
    weight: str = "normal",
    linewidth: float = 1.6,
):
    from matplotlib.patches import Ellipse, Rectangle

    ellipse_h = max(node.h * 0.22, 1.2)
    body_y = node.y + ellipse_h / 2
    body_h = max(node.h - ellipse_h, 0.5)
    body = Rectangle(
        (node.x, body_y),
        node.w,
        body_h,
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
    )
    top = Ellipse(
        (node.x + node.w / 2, node.y + node.h - ellipse_h / 2),
        node.w,
        ellipse_h,
        facecolor=_lighten(fill, 0.22),
        edgecolor=edge,
        linewidth=linewidth,
        zorder=3,
    )
    bottom = Ellipse(
        (node.x + node.w / 2, node.y + ellipse_h / 2),
        node.w,
        ellipse_h,
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(bottom)
    ax.add_patch(body)
    ax.add_patch(top)
    text_obj = ax.text(
        node.x + node.w / 2,
        node.y + node.h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1f1f1f",
        zorder=4,
    )
    fit_text_in_node(ax, text_obj, node, pad_x=0.7, pad_y=1.0, min_fontsize=7.0)
    return body


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str | None = None,
    color: str = "#6b6b6b",
    linestyle: str = "-",
    linewidth: float = 1.9,
    rad: float = 0.0,
    text_xy: tuple[float, float] | None = None,
    text_size: float = 10,
):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": linewidth,
            "color": color,
            "linestyle": linestyle,
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 4,
            "shrinkB": 4,
        },
        zorder=5,
    )
    if text:
        tx, ty = text_xy if text_xy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            tx,
            ty,
            text,
            ha="center",
            va="center",
            fontsize=text_size,
            color="#4f4f4f",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            zorder=6,
        )


def add_routed_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    text: str | None = None,
    color: str = "#6b6b6b",
    linestyle: str = "-",
    linewidth: float = 1.9,
    text_xy: tuple[float, float] | None = None,
    text_size: float = 10,
):
    if len(points) < 2:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=2.4,
    )
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={
            "arrowstyle": "->",
            "linewidth": linewidth,
            "color": color,
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 4,
        },
        zorder=5,
    )
    if text:
        tx, ty = text_xy if text_xy is not None else ((points[0][0] + points[-1][0]) / 2, (points[0][1] + points[-1][1]) / 2)
        ax.text(
            tx,
            ty,
            text,
            ha="center",
            va="center",
            fontsize=text_size,
            color="#4f4f4f",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            zorder=6,
        )


def fit_text_in_node(
    ax,
    text_obj,
    node: Node,
    *,
    pad_x: float,
    pad_y: float,
    min_fontsize: float = 7.0,
    step: float = 0.35,
):
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inner = Bbox.from_bounds(
        node.x + pad_x,
        node.y + pad_y,
        max(node.w - 2 * pad_x, 0.1),
        max(node.h - 2 * pad_y, 0.1),
    )
    inner_display = ax.transData.transform_bbox(inner)
    current_size = float(text_obj.get_fontsize())
    text_box = text_obj.get_window_extent(renderer=renderer)

    while (
        (text_box.width > inner_display.width or text_box.height > inner_display.height)
        and current_size > min_fontsize
    ):
        current_size = max(min_fontsize, current_size - step)
        text_obj.set_fontsize(current_size)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_box = text_obj.get_window_extent(renderer=renderer)


def save_plot_dual(fig, stem: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{stem}.svg", bbox_inches="tight")


def phase_card_text(name: str, stage_mix: str, focus: str) -> str:
    return f"{name}\n{stage_mix}\n{focus}"


def build_pipeline_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_config = build_default_run_config()
    phases = run_config.phases
    save_steps = run_config.trainer_defaults.save_steps
    base_model = run_config.model.base_model_name

    fig, ax = plt.subplots(figsize=(17.5, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    palette = {
        "model_fill": "#dbeaf7",
        "model_edge": "#6f89a8",
        "process_fill": "#fff2cc",
        "process_edge": "#b5963e",
        "data_fill": "#f3cfc9",
        "data_edge": "#8a6a65",
        "container_fill": "#fafafa",
        "container_edge": "#909090",
        "arrow": "#6d6d6d",
        "result_fill": "#d9e8f7",
        "result_edge": "#5f7f9e",
    }

    ax.text(
        1.0,
        61.8,
        "Qwen2.5-VL Staged RL pipeline",
        fontsize=34,
        ha="left",
        va="top",
        fontweight="normal",
        color="#111111",
    )
    ax.text(
        50.0,
        1.0,
        "repo-specific flow: phase data curation, metric-gated GRPO, checkpoint scoring, and alias-driven continuation",
        fontsize=11.5,
        ha="center",
        va="bottom",
        color="#4e4e4e",
    )

    train_split = Node(76.4, 53.6, 16.0, 8.0)
    phase_cfg = Node(18.8, 39.0, 14.8, 5.8)
    prompt_builder = Node(59.9, 54.3, 14.0, 6.2)
    base_node = Node(37.8, 49.4, 17.5, 4.6)
    loop_container = Node(35.3, 16.0, 42.4, 30.0)
    init_node = Node(38.0, 37.0, 16.8, 5.6)
    grpo_node = Node(58.0, 37.0, 17.8, 5.8)
    reward_node = Node(38.0, 27.2, 17.8, 8.0)
    eval_node = Node(58.0, 27.2, 17.8, 8.0)
    controller_node = Node(43.5, 19.0, 23.0, 5.8)
    eval_split = Node(79.0, 40.8, 16.0, 8.0)
    artifact_node = Node(79.0, 26.8, 16.0, 9.0)
    alias_node = Node(79.0, 15.2, 16.0, 8.5)
    curriculum_container = Node(4.0, 2.5, 73.0, 12.8)
    final_node = Node(79.7, 4.0, 16.0, 8.8)

    phase_a = Node(7.2, 5.0, 14.0, 6.8)
    phase_b = Node(24.8, 5.0, 14.0, 6.8)
    phase_c = Node(42.4, 5.0, 14.0, 6.8)
    phase_d = Node(60.0, 5.0, 14.0, 6.8)

    add_cylinder(
        ax,
        train_split,
        "MathVista\ntrain split\n(test or testmini)",
        fill=palette["data_fill"],
        edge=palette["data_edge"],
        fontsize=12.4,
        weight="bold",
    )
    add_round_box(
        ax,
        phase_cfg,
        "Phase config\nA / B / C / D",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=12.6,
        weight="bold",
    )
    add_round_box(
        ax,
        prompt_builder,
        "Stage / prompt builder\nfilters + priorities\n<REASONING>/<SOLUTION>",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=11.4,
        weight="bold",
    )
    add_round_box(
        ax,
        base_node,
        f"Base VLM\n{base_model}",
        fill=palette["model_fill"],
        edge=palette["model_edge"],
        fontsize=12.6,
        weight="bold",
        rounding=0.6,
    )
    add_container(ax, loop_container, "Per-Phase Metric-Gated GRPO Loop", fill=palette["container_fill"], edge=palette["container_edge"])
    add_round_box(
        ax,
        init_node,
        "Model prep / resume plan\nfresh LoRA, trainer resume,\nor adapter warm-start",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.7,
        weight="bold",
    )
    add_round_box(
        ax,
        grpo_node,
        "GRPO fine-tuning\nTRL GRPOTrainer\nUnsloth runtime",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=11.8,
        weight="bold",
    )
    add_round_box(
        ax,
        reward_node,
        "Rule-based reward stack\nformat | parseable | finished\ncorrectness | brevity | tolerance",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.8,
        weight="bold",
    )
    add_round_box(
        ax,
        eval_node,
        f"Checkpoint eval\nexact | tolerance | parseable\ntag compliance | truncation\nsave / eval every {save_steps} steps",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.7,
        weight="bold",
    )
    add_round_box(
        ax,
        controller_node,
        "Reward controller\nmetric-gated weight updates\nfor the next training segment",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.9,
        weight="bold",
    )
    add_cylinder(
        ax,
        eval_split,
        "MathVista\neval split\n(eval subsets)",
        fill=palette["data_fill"],
        edge=palette["data_edge"],
        fontsize=11.8,
        weight="bold",
    )
    add_cylinder(
        ax,
        artifact_node,
        "Checkpoint artifacts\nmetrics + reward weights\ncontroller state + samples",
        fill=palette["data_fill"],
        edge=palette["data_edge"],
        fontsize=10.8,
        weight="bold",
    )
    add_round_box(
        ax,
        alias_node,
        "Scores + aliases\nstructure | correctness | composite\nlatest | best_structure |\nbest_correctness | best_composite",
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.3,
        weight="bold",
    )

    add_container(ax, curriculum_container, "Phase-to-Phase Curriculum", fill=palette["container_fill"], edge=palette["container_edge"])
    add_round_box(
        ax,
        phase_a,
        phase_card_text("Phase A", "S1 only", "structure stabilization"),
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=11.0,
        weight="bold",
    )
    add_round_box(
        ax,
        phase_b,
        phase_card_text("Phase B", "S1 70% + S2 30%", "correctness strengthening"),
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.5,
        weight="bold",
    )
    add_round_box(
        ax,
        phase_c,
        phase_card_text("Phase C", "S2 60% + S3 40%", "tolerance + harder reasoning"),
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.5,
        weight="bold",
    )
    add_round_box(
        ax,
        phase_d,
        phase_card_text("Phase D", "S3 only", "hard-subset specialization"),
        fill=palette["process_fill"],
        edge=palette["process_edge"],
        fontsize=10.8,
        weight="bold",
    )
    add_round_box(
        ax,
        final_node,
        "Recommended final\n(report layer)\nLarge Phase C\nbest_composite",
        fill=palette["result_fill"],
        edge=palette["result_edge"],
        fontsize=11.4,
        weight="bold",
        rounding=0.7,
    )

    add_arrow(ax, train_split.left(), prompt_builder.right(), color=palette["arrow"])
    add_arrow(ax, phase_cfg.right(), init_node.left(0.8), color=palette["arrow"])
    add_arrow(ax, prompt_builder.bottom(), grpo_node.top(), color=palette["arrow"])
    add_arrow(ax, base_node.bottom(), init_node.top(), color=palette["arrow"])
    add_arrow(ax, init_node.right(), grpo_node.left(1.0), color=palette["arrow"])
    add_routed_arrow(
        ax,
        [
            reward_node.right(1.3),
            (56.7, reward_node.right(1.3)[1]),
            (56.7, grpo_node.left(-1.0)[1]),
            grpo_node.left(-1.0),
        ],
        color=palette["arrow"],
    )
    add_arrow(ax, grpo_node.bottom(), eval_node.top(), color=palette["arrow"])
    add_routed_arrow(
        ax,
        [
            eval_split.left(-0.4),
            (77.2, eval_split.left(-0.4)[1]),
            (77.2, eval_node.right(1.1)[1]),
            eval_node.right(1.1),
        ],
        color=palette["arrow"],
    )
    add_arrow(ax, eval_node.bottom(-11.9), controller_node.top(), color=palette["arrow"])
    add_arrow(ax, controller_node.top(-8.1), reward_node.bottom(), color=palette["arrow"])
    add_arrow(ax, eval_node.right(0.1), artifact_node.left(), color=palette["arrow"])
    add_arrow(ax, artifact_node.bottom(), alias_node.top(), color=palette["arrow"])
    add_routed_arrow(
        ax,
        [
            alias_node.left(0.6),
            (77.4, alias_node.left(0.6)[1]),
            (77.4, 46.9),
            (56.2, 46.9),
            (56.2, init_node.right(1.2)[1]),
            init_node.right(1.2),
        ],
        color=palette["arrow"],
        linestyle="--",
    )
    add_arrow(ax, loop_container.bottom(), curriculum_container.top(18.0), color=palette["arrow"], text="same loop runs for each phase", text_xy=(55.8, 17.0), text_size=9.5)

    add_arrow(ax, phase_a.right(), phase_b.left(), color=palette["arrow"], text="best_structure", text_xy=(23.2, 10.0), text_size=9.5)
    add_arrow(ax, phase_b.right(), phase_c.left(), color=palette["arrow"], text="best_composite", text_xy=(40.8, 10.0), text_size=9.5)
    add_arrow(ax, phase_c.right(), phase_d.left(), color=palette["arrow"], text="best_composite", text_xy=(58.4, 10.0), text_size=9.5)
    add_routed_arrow(
        ax,
        [
            phase_c.top(),
            (phase_c.top()[0], 16.2),
            (final_node.top()[0], 16.2),
            final_node.top(),
        ],
        color=palette["arrow"],
        text="recommended branch",
        text_xy=(69.8, 14.8),
        text_size=9.4,
    )
    add_arrow(ax, phase_d.right(1.3), final_node.top(-1.0), color=palette["arrow"], linestyle="--", rad=-0.15, text="matched, did not beat", text_xy=(78.0, 16.6), text_size=9.0)

    ax.text(7.8, 13.4, "start from base", fontsize=9.4, color="#565656")
    ax.text(81.0, 2.2, "phase C remained the practical winner in the exported artifacts", fontsize=9.4, color="#5a5a5a")

    save_plot_dual(fig, "staged_training_pipeline")


def main() -> None:
    build_pipeline_plot()


if __name__ == "__main__":
    main()

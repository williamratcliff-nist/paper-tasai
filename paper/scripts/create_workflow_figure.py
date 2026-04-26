#!/usr/bin/env python
"""Create workflow and TOC figures for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def create_workflow_figure():
    """Create clean workflow diagram with GK focus."""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    main_boxes = [
        (1.0, 4.8, "Crystal\nStructure\n(CIF)", "#b8d4e8", 1.4, 1.1),
        (3.0, 4.8, "GNN / GK\nHypothesis\nGenerator", "#fff2cc", 1.6, 1.1),
        (5.0, 4.8, "Candidate\nHamiltonians", "#d5e8d4", 1.4, 1.1),
        (7.0, 4.8, "TAS-AI\nMeasurement\nPlanning", "#f8cecc", 1.6, 1.1),
        (9.0, 4.8, "Bayesian\nDiscrimination", "#e1d5e7", 1.4, 1.1),
        (11.0, 4.8, "Validated\nHamiltonian", "#fff2cc", 1.4, 1.1),
    ]

    for x, y, text, color, w, h in main_boxes:
        bbox = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor="black",
            lw=2,
        )
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")

    arrow_style = dict(arrowstyle="->", color="black", lw=2, connectionstyle="arc3,rad=0")
    for i in range(len(main_boxes) - 1):
        x1 = main_boxes[i][0] + main_boxes[i][4] / 2
        x2 = main_boxes[i + 1][0] - main_boxes[i + 1][4] / 2
        ax.annotate("", xy=(x2, 4.8), xytext=(x1, 4.8), arrowprops=arrow_style)

    ax.annotate(
        "",
        xy=(3.0, 4.8 - 1.1 / 2),
        xytext=(11.0, 4.8 - 1.1 / 2),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#27ae60",
            lw=3,
            connectionstyle="arc3,rad=0.5",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax.text(
        7.0,
        2.6,
        "Feedback: Validated J values\nimprove predictions for new materials",
        ha="center",
        fontsize=9,
        color="#27ae60",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#27ae60", alpha=0.9),
    )

    detail_x, detail_y = 2.0, 1.3
    detail_box = FancyBboxPatch(
        (detail_x - 1.5, detail_y - 0.7),
        3.0,
        1.4,
        boxstyle="round,pad=0.03",
        facecolor="#fffde7",
        edgecolor="#f9a825",
        lw=1.5,
    )
    ax.add_patch(detail_box)
    ax.text(detail_x, detail_y + 0.35, "GNN / GK Output:", ha="center", fontsize=9, fontweight="bold")
    ax.text(detail_x, detail_y, "• Exchange pathways", ha="center", fontsize=8)
    ax.text(detail_x, detail_y - 0.25, "• AFM/FM predictions", ha="center", fontsize=8)
    ax.text(detail_x, detail_y - 0.5, "• Ranked |J| estimates", ha="center", fontsize=8)
    ax.annotate(
        "",
        xy=(detail_x + 0.5, detail_y + 0.7),
        xytext=(3.0, 4.8 - 1.1 / 2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"),
    )

    detail2_x, detail2_y = 10.0, 1.3
    detail2_box = FancyBboxPatch(
        (detail2_x - 1.5, detail2_y - 0.7),
        3.0,
        1.4,
        boxstyle="round,pad=0.03",
        facecolor="#e8f5e9",
        edgecolor="#4caf50",
        lw=1.5,
    )
    ax.add_patch(detail2_box)
    ax.text(detail2_x, detail2_y + 0.35, "TAS-AI Output:", ha="center", fontsize=9, fontweight="bold")
    ax.text(detail2_x, detail2_y, "• Best-fit Hamiltonian", ha="center", fontsize=8)
    ax.text(detail2_x, detail2_y - 0.25, "• Parameter uncertainties", ha="center", fontsize=8)
    ax.text(detail2_x, detail2_y - 0.5, "• Model selection (Bayes factor)", ha="center", fontsize=8)
    ax.annotate(
        "",
        xy=(detail2_x - 0.5, detail2_y + 0.7),
        xytext=(8.6, 4.8 - 1.1 / 2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"),
    )

    ax.text(
        6.0,
        5.7,
        "Closed-Loop Autonomous Spin Wave Characterization",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def create_swimlane_figure():
    """Generate improved workflow/example swimlane diagram."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    box_w, box_h = 2.0, 1.3
    gap, x_start = 0.5, 0.5
    y_top, y_bot = 6.0, 2.5

    c_blue = "#dae8fc"
    c_yellow = "#fff2cc"
    c_green = "#d5e8d4"
    c_red = "#f8cecc"
    c_purple = "#e1d5e7"
    c_db = "#f5f5f5"

    nodes = [
        ("Crystal\nStructure", c_blue, "Input", "Fe-O Square Lattice\n(CIF file)"),
        ("GNN / GK\nHypothesis", c_yellow, "Hypothesis Gen", "GK Rules:\n180° Fe-O-Fe $\\to$ Strong $J_1$\n117° Fe-Fe $\\to$ Weak $J_2$"),
        ("Candidate\nHamiltonians", c_green, "Model Pool", "$M_1$: NN ($J_1$)\n$M_3$: $J_1+J_2$\n$M_4$: Full ($J_1+J_2+D$)"),
        ("TAS-AI Planning\n(Hybrid Strategy)", c_red, "Adaptive Planner", "1. Symmetry Seeding (Prior)\n2. Diversity-JSD (Explore)\n3. Superset Refinement (Exploit)"),
        ("Bayesian\nDiscrimination", c_purple, "Inference", "Posterior Update:\nSuperset plan confirms $J_2$\nBayes Factor > 100"),
        ("Validated\nHamiltonian", c_blue, "Result", "Final Selection:\n$M_4$: >99% (Correct)\n$M_1$: <1% (Rejected)"),
    ]

    for i, (title, color, header, text) in enumerate(nodes):
        x_coord = x_start + i * (box_w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x_coord + 0.05, y_top - 0.05),
                box_w,
                box_h,
                boxstyle="round,pad=0.1",
                fc="#cccccc",
                ec="none",
                zorder=9,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x_coord, y_top),
                box_w,
                box_h,
                boxstyle="round,pad=0.1",
                ec="#333333",
                fc=color,
                lw=1.5,
                zorder=10,
            )
        )
        ax.text(
            x_coord + box_w / 2,
            y_top + box_h / 2,
            title,
            ha="center",
            va="center",
            fontweight="bold",
            color="#222222",
            zorder=11,
        )

        ax.add_patch(patches.Rectangle((x_coord, y_bot), box_w, box_h * 1.2, ec="#999999", fc="white", zorder=5))
        ax.add_patch(
            patches.Rectangle((x_coord, y_bot + box_h * 1.2 - 0.3), box_w, 0.3, fc=color, alpha=0.6, zorder=6)
        )
        ax.text(
            x_coord + box_w / 2,
            y_bot + box_h * 1.2 - 0.15,
            header,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
            zorder=7,
        )
        ax.text(
            x_coord + box_w / 2,
            y_bot + box_h * 0.5,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
            zorder=7,
        )

        ax.plot([x_coord + box_w / 2, x_coord + box_w / 2], [y_top, y_bot + box_h * 1.2], ":", color="#AAAAAA", lw=1.5, zorder=1)

        if i < len(nodes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x_coord + box_w, y_top + box_h / 2),
                    (x_coord + box_w + gap, y_top + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color="#333333",
                    zorder=10,
                )
            )

    db_x = x_start + 2.5 * (box_w + gap)
    db_y = 4.5
    ax.add_patch(patches.Rectangle((db_x - 0.75, db_y - 0.4), 1.5, 0.8, fc=c_db, ec="#999999", zorder=1))
    ax.add_patch(patches.Ellipse((db_x, db_y + 0.4), 1.5, 0.3, fc="white", ec="#999999", zorder=2))
    ax.add_patch(patches.Ellipse((db_x, db_y - 0.4), 1.5, 0.3, fc=c_db, ec="#999999", zorder=1))
    ax.text(db_x, db_y, "Training Data\n(Structure $\\to$ $J_{ij}$)", ha="center", va="center", fontweight="bold", fontsize=9, color="#555555", zorder=3)

    end_x = x_start + 5 * (box_w + gap) + box_w / 2
    ax.add_patch(
        FancyArrowPatch(
            (end_x, y_top + box_h),
            (db_x + 0.8, db_y),
            connectionstyle="arc3,rad=-0.2",
            arrowstyle="->",
            mutation_scale=15,
            color="#27ae60",
            lw=2,
            linestyle="--",
            zorder=5,
        )
    )

    hyp_x = x_start + 1 * (box_w + gap) + box_w / 2
    ax.add_patch(
        FancyArrowPatch(
            (db_x - 0.8, db_y),
            (hyp_x, y_top + box_h),
            connectionstyle="arc3,rad=-0.2",
            arrowstyle="->",
            mutation_scale=15,
            color="#27ae60",
            lw=2,
            linestyle="--",
            zorder=5,
        )
    )

    ax.text(
        db_x,
        5.5,
        "Closed-Loop Feedback: Refine Priors",
        ha="center",
        color="#27ae60",
        fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.8),
    )
    ax.text(0.2, y_top + box_h / 2, "(a) Conceptual\nWorkflow", fontsize=12, fontweight="bold", ha="center", va="center", rotation=90)
    ax.text(0.2, y_bot + box_h / 2, "(b) Example:\nFe-O Lattice", fontsize=12, fontweight="bold", ha="center", va="center", rotation=90)

    plt.tight_layout()
    return fig


def _draw_heatmap_panel(ax, x0, y0, w, h):
    xs = np.linspace(0.0, 1.0, 120)
    ys = np.linspace(0.0, 1.0, 120)
    xx, yy = np.meshgrid(xs, ys)
    ridge = np.exp(-((yy - (0.23 + 0.48 * xx + 0.08 * np.sin(5 * xx))) ** 2) / 0.01)
    branch = 0.6 * np.exp(-((yy - (0.68 - 0.32 * xx)) ** 2) / 0.015)
    img = ridge + branch
    ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), origin="lower", cmap="Blues", alpha=0.95, zorder=1)

    pts = np.array(
        [
            [0.10, 0.18], [0.18, 0.28], [0.28, 0.25], [0.36, 0.36], [0.42, 0.30],
            [0.52, 0.48], [0.62, 0.41], [0.72, 0.55], [0.79, 0.50], [0.87, 0.62],
        ]
    )
    pts[:, 0] = x0 + pts[:, 0] * w
    pts[:, 1] = y0 + pts[:, 1] * h
    ax.scatter(pts[:, 0], pts[:, 1], s=22, c="white", edgecolors="#0b5394", linewidths=1.0, zorder=3)
    ax.text(x0 + 0.08 * w, y0 + 0.86 * h, "Enhanced\nLog-GP", fontsize=11, fontweight="bold", color="#173f5f", va="top")


def _draw_inference_panel(ax, x0, y0, w, h):
    x = np.linspace(x0 + 0.08 * w, x0 + 0.92 * w, 200)
    t = (x - x.min()) / (x.max() - x.min())
    y_main = y0 + h * (0.22 + 0.46 * t)
    y_alt1 = y0 + h * (0.28 + 0.33 * t + 0.05 * np.sin(4 * np.pi * t))
    y_alt2 = y0 + h * (0.72 - 0.30 * t)

    ax.plot(x, y_alt1, color="#f6b26b", lw=2.0, alpha=0.9)
    ax.plot(x, y_alt2, color="#b4a7d6", lw=2.0, alpha=0.9)
    ax.plot(x, y_main, color="#cc0000", lw=3.2)

    bar_x = x0 + 0.73 * w
    bars = [0.18, 0.22, 0.60]
    colors = ["#f6b26b", "#b4a7d6", "#cc0000"]
    for i, (height, color) in enumerate(zip(bars, colors)):
        bx = bar_x + i * 0.07 * w
        ax.add_patch(patches.Rectangle((bx, y0 + 0.10 * h), 0.045 * w, height * h, fc=color, ec="none", alpha=0.95))
    ax.text(x0 + 0.08 * w, y0 + 0.86 * h, "Physics-aware\ninference", fontsize=11, fontweight="bold", color="#7a1f1f", va="top")


def _draw_audit_panel(ax, x0, y0, w, h):
    for cx, cy, r, alpha in [(0.32, 0.60, 0.18, 0.10), (0.55, 0.36, 0.22, 0.08), (0.72, 0.72, 0.16, 0.07)]:
        ax.add_patch(patches.Circle((x0 + cx * w, y0 + cy * h), r * min(w, h), fc="#d9ead3", ec="none", alpha=alpha))

    curve_x = np.linspace(x0 + 0.12 * w, x0 + 0.88 * w, 150)
    t = (curve_x - curve_x.min()) / (curve_x.max() - curve_x.min())
    curve_y = y0 + h * (0.25 + 0.22 * t + 0.06 * np.sin(2.3 * np.pi * t))
    ax.plot(curve_x, curve_y, color="#38761d", lw=2.4)

    probe_x = x0 + 0.58 * w
    probe_y = y0 + 0.48 * h
    ax.scatter([probe_x], [probe_y], s=70, c="#ffd966", edgecolors="#a61c00", linewidths=1.6, zorder=4)
    ax.annotate(
        "",
        xy=(probe_x, probe_y),
        xytext=(x0 + 0.28 * w, y0 + 0.78 * h),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#a61c00"),
    )

    shield = np.array(
        [
            [0.80, 0.76], [0.88, 0.76], [0.91, 0.67], [0.84, 0.56], [0.77, 0.67],
        ]
    )
    shield[:, 0] = x0 + shield[:, 0] * w
    shield[:, 1] = y0 + shield[:, 1] * h
    ax.add_patch(patches.Polygon(shield, closed=True, fc="#93c47d", ec="#274e13", lw=1.5))
    ax.text(x0 + 0.84 * w, y0 + 0.67 * h, "?", ha="center", va="center", fontsize=12, fontweight="bold", color="#274e13")
    ax.text(x0 + 0.08 * w, y0 + 0.86 * h, "Guarded\nfalsification", fontsize=11, fontweight="bold", color="#274e13", va="top")


def create_toc_figure():
    """Create a simple TOC graphic centered on the paper's three control stages."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    panels = [
        (0.35, 0.55, 2.75, 2.65, "#d9eaf7", "Discovery"),
        (3.40, 0.55, 2.75, 2.65, "#f9dfdc", "Inference"),
        (6.45, 0.55, 2.70, 2.65, "#deefd5", "Audit"),
    ]

    for x0, y0, w, h, color, title in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.18",
                fc=color,
                ec="#2f2f2f",
                lw=1.6,
            )
        )
        ax.text(x0 + 0.12, y0 + h + 0.20, title, fontsize=14, fontweight="bold", color="#1f1f1f")

    _draw_heatmap_panel(ax, 0.35, 0.55, 2.75, 2.65)
    _draw_inference_panel(ax, 3.40, 0.55, 2.75, 2.65)
    _draw_audit_panel(ax, 6.45, 0.55, 2.70, 2.65)

    for x1, x2 in [(3.12, 3.33), (6.17, 6.38)]:
        ax.add_patch(
            FancyArrowPatch((x1, 1.88), (x2, 1.88), arrowstyle="-|>", mutation_scale=16, lw=2.2, color="#333333")
        )

    ax.text(
        4.75,
        3.92,
        "Autonomous neutron spectroscopy with hybrid control",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        4.75,
        0.18,
        "Agnostic mapping localizes signal, physics planning resolves models, and a guarded audit layer breaks algorithmic myopia.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )
    plt.tight_layout(pad=0.2)
    return fig


def _save_figure(fig, output_base: Path):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    if output_base.suffix.lower() == ".pdf":
        fig.savefig(output_base, bbox_inches="tight", facecolor="white", edgecolor="none")


def main():
    parser = argparse.ArgumentParser(description="Create workflow and TOC figures.")
    parser.add_argument("--save-dir", default="figures", help="Directory for generated outputs.")
    parser.add_argument("--toc-only", action="store_true", help="Generate only the TOC figure.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (Path.cwd() / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.toc_only:
        fig = create_toc_figure()
        fig.savefig(save_dir / "figure_toc.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {save_dir / 'figure_toc.png'}")
        return

    fig1 = create_workflow_figure()
    fig1.savefig(save_dir / "closed_loop_workflow.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig1.savefig(save_dir / "closed_loop_workflow.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig1)
    print(f"Saved: {save_dir / 'closed_loop_workflow.png'}")

    fig2 = create_swimlane_figure()
    fig2.savefig(save_dir / "workflow_with_example.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig2)
    print(f"Saved: {save_dir / 'workflow_with_example.png'}")

    fig3 = create_toc_figure()
    fig3.savefig(save_dir / "figure_toc.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig3)
    print(f"Saved: {save_dir / 'figure_toc.png'}")


if __name__ == "__main__":
    main()

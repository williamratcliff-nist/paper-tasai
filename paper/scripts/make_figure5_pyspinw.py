#!/usr/bin/env python3
"""Generate PySpinW benchmark panels for the main text."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "figure5_pyspinw.png"
OUT_LEGACY = ROOT / "figure5_pyspinw.png"


def _load_npz(path: Path):
    data = np.load(path)
    return data["H"], data["E"], data["intensity"]


def _plot_intensity(ax, label, title, H, E, values):
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color="white")
    im = ax.pcolormesh(H, E, values.T, shading="auto", cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("[H H 0] (r.l.u.)")
    ax.set_ylabel("Energy (meV)")
    return im


def _plot_pyspinw_benchmark(ax, label):
    agnostic_path = ROOT / "data" / "final" / "benchmark_summary_fair_pyspinw_20260402c.json"
    tasai_path = ROOT / "data" / "final" / "benchmark_tasai_pyspinw_20260402d.json"
    with agnostic_path.open() as f:
        agnostic = json.load(f)
    with tasai_path.open() as f:
        tasai = json.load(f)

    scenarios = ["pyspinw_single", "pyspinw_gapped"]
    scenario_labels = ["Single", "Gapped"]
    methods = ["grid", "random", "log_gp", "tasai"]
    colors = {
        "grid": "#4c566a",
        "random": "#f6c343",
        "log_gp": "#2d7dd2",
        "tasai": "#d1495b",
    }

    merged = {
        "pyspinw_single": {
            "grid": agnostic["pyspinw_single"]["grid"],
            "random": agnostic["pyspinw_single"]["random"],
            "log_gp": agnostic["pyspinw_single"]["log_gp"],
            "tasai": tasai["pyspinw_single"]["tasai"],
        },
        "pyspinw_gapped": {
            "grid": agnostic["pyspinw_gapped"]["grid"],
            "random": agnostic["pyspinw_gapped"]["random"],
            "log_gp": agnostic["pyspinw_gapped"]["log_gp"],
            "tasai": tasai["pyspinw_gapped"]["tasai"],
        },
    }

    x = np.arange(len(scenarios))
    width = 0.18
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        xpos = x + (i - 1.5) * width
        means = [merged[s][method]["mean_final_error"] for s in scenarios]
        stds = [merged[s][method]["std_final_error"] for s in scenarios]
        ax.bar(
            xpos,
            means,
            width=width,
            color=colors[method],
            alpha=0.9,
            yerr=stds,
            capsize=4,
            label=method.replace("_", "-"),
        )
        for j, scenario in enumerate(scenarios):
            pts = np.array([run["final_error"] for run in merged[scenario][method]["runs"]], dtype=float)
            jitter = rng.uniform(-0.03, 0.03, size=len(pts))
            ax.scatter(
                np.full_like(pts, xpos[j]) + jitter,
                pts,
                s=16,
                color="white",
                edgecolors=colors[method],
                linewidths=0.8,
                zorder=3,
            )

    ax.set_title("PySpinW benchmark (final error at N=300)")
    ax.set_ylabel("Final weighted reconstruction error")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))

    H, E, values = _load_npz(ROOT / "data" / "pyspinw_cn_single_branch.npz")
    im1 = _plot_intensity(axes[0], "(a)", "PySpinW Single (CN)", H, E, values)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity (a.u.)")

    H, E, values = _load_npz(ROOT / "data" / "pyspinw_cn_gapped_branch.npz")
    im2 = _plot_intensity(axes[1], "(b)", "PySpinW Gapped (CN)", H, E, values)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Intensity (a.u.)")

    _plot_pyspinw_benchmark(axes[2], "(c)")

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_LEGACY, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

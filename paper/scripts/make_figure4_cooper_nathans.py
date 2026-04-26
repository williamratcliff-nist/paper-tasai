#!/usr/bin/env python3
"""Generate JCNS benchmark summary with Cooper-Nathans convolution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "legacy" / "benchmark_summary_thr035_cooper_nathans.json"
OUT = ROOT / "figure4_benchmark_cooper_nathans.png"

SCENARIOS = [
    "single_branch",
    "two_branches",
    "weak_signal",
    "sharp_feature",
    "gap_mode",
]

METHODS = ["grid", "random", "log_gp", "tasai", "sunny"]
METHOD_LABELS = {
    "grid": "Grid",
    "random": "Random",
    "log_gp": "Log-GP",
    "tasai": "TAS-AI",
    "sunny": "TAS-AI (Sunny)",
}
COLORS = {
    "grid": "#1f77b4",
    "random": "#ff7f0e",
    "log_gp": "#2ca02c",
    "tasai": "#d62728",
    "sunny": "#17becf",
}


def load_summary():
    with DATA.open() as f:
        return json.load(f)


def scenario_ticks():
    x = np.arange(len(SCENARIOS))
    labels = [s.replace("_", "\n") for s in SCENARIOS]
    return x, labels


def build_metric_arrays(summary):
    meas = np.full((len(METHODS), len(SCENARIOS)), np.nan)
    errs = np.full_like(meas, np.nan)
    for j, scenario in enumerate(SCENARIOS):
        for i, method in enumerate(METHODS):
            entry = summary.get(scenario, {}).get(method)
            if not entry:
                continue
            meas[i, j] = entry.get("mean_converge")
            errs[i, j] = entry.get("mean_final_error")
    return meas, errs


def plot_bars(ax, data, ylabel, title):
    x, labels = scenario_ticks()
    width = 0.15
    for idx, method in enumerate(METHODS):
        offset = (idx - (len(METHODS) - 1) / 2) * width
        ax.bar(
            x + offset,
            data[idx],
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_convergence(ax, summary):
    for method in METHODS:
        run = summary["single_branch"][method]["runs"][0]
        arr = np.asarray(run["errors"])
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Measurements")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Representative convergence (single_branch)")
    ax.set_ylim(bottom=0)


def plot_speedup(ax, meas):
    x, labels = scenario_ticks()
    baseline = meas[METHODS.index("grid")]
    for method in METHODS:
        if method == "grid":
            continue
        comp = meas[METHODS.index(method)]
        speed = baseline / comp
        ax.plot(
            x,
            speed,
            marker="o",
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speed-up vs grid")
    valid = np.nan_to_num(baseline / meas[1:], nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.nanmax(valid) if np.nanmax(valid) > 0 else 1.0
    ax.set_ylim(0, max(1.1, max_val * 1.1))
    ax.set_title("Relative efficiency (lower is faster)")


def add_panel_labels(axes):
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes.ravel(), labels):
        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )


def main():
    summary = load_summary()
    meas, errs = build_metric_arrays(summary)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plot_bars(
        axes[0, 0],
        meas,
        "Measurements to converge (<=200)",
        "JCNS benchmark convergence (Cooper-Nathans)",
    )
    plot_bars(
        axes[0, 1],
        errs,
        "Final reconstruction error",
        "Final error after 200 measurements",
    )
    plot_convergence(axes[1, 0], summary)
    plot_speedup(axes[1, 1], meas)
    add_panel_labels(axes)
    axes[0, 0].legend(ncol=2, fontsize=8, loc="upper right")
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("JCNS benchmark summary (Cooper-Nathans, threshold 0.35)")
    fig.text(
        0.5,
        0.01,
        "Cooper-Nathans convolution: 40' horizontal, 120' vertical, Ef=14.7 meV",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUT, dpi=300)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

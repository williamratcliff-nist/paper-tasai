#!/usr/bin/env python
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "legacy" / "benchmark_pyspinw_groundtruth.json"
OUT_PATH = ROOT / "figureS_pyspinw_alignment.png"


def main() -> None:
    with DATA_PATH.open() as f:
        data = json.load(f)

    scenarios = [k for k in data.keys() if k != "metadata"]
    methods = ["grid", "random", "log_gp", "tasai"]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.2), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        means = [data[scenario][m]["mean_converge"] for m in methods]
        x = list(range(len(methods)))
        ax.bar(x, means, color=["#808080", "#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylabel("Measurements to converge")
        ax.set_ylim(0, max(means) * 1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")

    fig.suptitle("PySpinW ground-truth benchmark (threshold 0.35, 200 max)")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

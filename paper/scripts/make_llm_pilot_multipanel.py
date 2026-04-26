#!/usr/bin/env python3
"""Assemble multi-panel figure for LLM pilot runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fig_dir = repo_root / "paper" / "figures" / "llm_progress"
    out_path = fig_dir / "llm_pilot_multipanel.png"

    panels = [
        ("(a) Grid + physics (no LLM)", fig_dir / "scenario1_analysis_coverage_posteriors.png"),
        ("(b) Grid + physics + symmetry", fig_dir / "scenario2_sym_preloggp_rerun4_analysis_coverage_posteriors.png"),
        ("(c) Grid + physics + LLM", fig_dir / "scenario3a_analysis_coverage_posteriors.png"),
        ("(d) Grid + physics + symmetry + LLM", fig_dir / "scenario3b_sym_mailbox_rerun1_analysis_coverage_posteriors.png"),
    ]

    for title, path in panels:
        if not path.exists():
            raise SystemExit(f"Missing panel image: {path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (title, path) in zip(axes, panels):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(out_path)


if __name__ == "__main__":
    main()

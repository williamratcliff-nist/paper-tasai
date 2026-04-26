#!/usr/bin/env python3
"""Build a Figure 11-style comparison for the archived bilayer audit ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulations.run_audit_ablation import (  # noqa: E402
    BilayerFMConfig,
    _bilayer_model_posteriors,
    _bilayer_models,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--none", required=True)
    p.add_argument("--hybrid", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_summary(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def _config_from_summary(summary: Dict) -> BilayerFMConfig:
    cfg = dict(summary["bilayer_fm_config"])
    return BilayerFMConfig(**cfg)


def _mode_style(mode: str) -> Tuple[str, str, float]:
    if mode == "bilayer_seed":
        return ("o", "#FFD166", 38.0)
    if mode == "bilayer_loggp":
        return ("s", "#2A9D8F", 40.0)
    if mode == "llm_audit":
        return ("D", "#E63946", 44.0)
    return ("^", "#457B9D", 40.0)


def _group_measurements(measurements: List[Dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for meas in measurements:
        grouped.setdefault(str(meas.get("mode", "unknown")), []).append((float(meas["h"]), float(meas["E"])))
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mode, points in grouped.items():
        arr = np.asarray(points, dtype=float)
        out[mode] = (arr[:, 0], arr[:, 1])
    return out


def _posterior_trace(summary: Dict, config: BilayerFMConfig) -> Dict[str, np.ndarray]:
    measurements = summary["measurements"]
    initial_n = int(summary.get("initial_measurement_count", 0))
    xs: List[int] = []
    post_a: List[float] = []
    post_b: List[float] = []
    for n in range(initial_n, len(measurements) + 1, 2):
        res = _bilayer_model_posteriors(measurements[:n], config)
        xs.append(n)
        post_a.append(float(res["M_A: Monolayer FM"]["posterior"]))
        post_b.append(float(res["M_B: Bilayer FM"]["posterior"]))
    if xs[-1] != len(measurements):
        res = _bilayer_model_posteriors(measurements, config)
        xs.append(len(measurements))
        post_a.append(float(res["M_A: Monolayer FM"]["posterior"]))
        post_b.append(float(res["M_B: Bilayer FM"]["posterior"]))
    return {
        "n": np.asarray(xs, dtype=int),
        "M_A": np.asarray(post_a, dtype=float),
        "M_B": np.asarray(post_b, dtype=float),
    }


def _draw_panel(ax_cov, ax_post, summary: Dict, title: str, config: BilayerFMConfig) -> None:
    _, bilayer = _bilayer_models(config)
    h_vals = np.linspace(config.h_min, config.h_max, 240)
    e_vals = np.linspace(config.e_min, config.e_max, 280)
    hh, ee = np.meshgrid(h_vals, e_vals, indexing="ij")
    inten = np.zeros_like(hh)
    for i in range(hh.shape[0]):
        for j in range(hh.shape[1]):
            inten[i, j] = bilayer.intensity_bilayer(float(hh[i, j]), float(hh[i, j]), config.L_fixed, float(ee[i, j]))

    ax_cov.pcolormesh(hh, ee, np.log10(inten + 1e-3), shading="auto", cmap="magma")
    ax_cov.plot(h_vals, [bilayer.omega_ac(h, h) for h in h_vals], "--", color="white", lw=1.7, alpha=0.9)
    ax_cov.plot(h_vals, [bilayer.omega_op(h, h) for h in h_vals], "--", color="#8ECAE6", lw=1.7, alpha=0.95)
    h_guide = 0.34
    e_guide = bilayer.omega_op(h_guide, h_guide)
    ax_cov.annotate(
        "Weak optic branch\n(guide to eye)",
        xy=(h_guide, e_guide),
        xytext=(0.06, min(config.e_max - 1.2, e_guide + 3.0)),
        textcoords="data",
        fontsize=7.5,
        color="#DFF6FF",
        arrowprops=dict(arrowstyle="->", color="#DFF6FF", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(0, 0, 0, 0.25), edgecolor="#DFF6FF", linewidth=0.5),
    )

    labels = {
        "bilayer_seed": "Seed",
        "bilayer_loggp": "Hybrid GP",
        "bilayer_physics": "Physics",
        "llm_audit": "LLM audit",
    }
    grouped = _group_measurements(summary["measurements"])
    for mode in ["bilayer_seed", "bilayer_loggp", "bilayer_physics", "llm_audit"]:
        if mode not in grouped:
            continue
        x, y = grouped[mode]
        marker, color, size = _mode_style(mode)
        ax_cov.scatter(x, y, s=size, c=color, marker=marker, edgecolor="black", linewidth=0.45, label=labels[mode], zorder=4)

    ax_cov.set_xlim(config.h_min, config.h_max)
    ax_cov.set_ylim(config.e_min, config.e_max)
    ax_cov.set_xlabel("H along [H,H,L]")
    ax_cov.set_ylabel("Energy (meV)")
    ax_cov.set_title(title)
    ax_cov.legend(loc="upper left", fontsize=7.4, frameon=True)

    trace = _posterior_trace(summary, config)
    ax_post.plot(trace["n"], trace["M_A"], color="#457B9D", lw=1.9, marker="o", ms=3, label="M_A monolayer")
    ax_post.plot(trace["n"], trace["M_B"], color="#E63946", lw=1.9, marker="o", ms=3, label="M_B bilayer")
    ax_post.axhline(0.99, color="0.4", lw=0.8, ls=":")
    ax_post.set_ylim(0.0, 1.02)
    ax_post.set_xlabel("Measurement #")
    ax_post.set_ylabel("Posterior")
    ax_post.grid(alpha=0.25, linewidth=0.6)
    ax_post.xaxis.set_major_locator(MaxNLocator(integer=True))

    tdc = summary.get("time_to_decisive_correct")
    dwell = summary.get("wrong_leader_dwell_time")
    fals = summary.get("falsification_probe_fraction")
    if tdc is None:
        note = f"no decisive recovery\ndwell={dwell:.0f}, falsify={fals:.2f}"
    else:
        note = f"decisive @ {int(tdc)}\ndwell={dwell:.0f}, falsify={fals:.2f}"
    ax_post.text(
        0.97,
        0.08,
        note,
        transform=ax_post.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, linewidth=0.5),
    )


def main() -> None:
    args = parse_args()
    summaries = [
        ("(a) No audit", _load_summary(args.none)),
        ("(b) Hybrid explorer + physics", _load_summary(args.hybrid)),
        ("(c) LLM audit", _load_summary(args.llm)),
    ]
    config = _config_from_summary(summaries[0][1])

    fig = plt.figure(figsize=(15.0, 7.6))
    outer = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.20, wspace=0.18)
    for col, (title, summary) in enumerate(summaries):
        ax_cov = fig.add_subplot(outer[0, col])
        ax_post = fig.add_subplot(outer[1, col])
        _draw_panel(ax_cov, ax_post, summary, title, config)

    fig.suptitle(
        "Bilayer FM audit ablation: archived one-seed comparison",
        fontsize=15,
        y=0.98,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

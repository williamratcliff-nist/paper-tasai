#!/usr/bin/env python3
"""Plot current closed-loop measurements over the full synthetic data map.

Usage:
  python paper/scripts/plot_llm_inloop_snapshot.py \
    --checkpoint /path/to/closed_loop_checkpoint.json \
    --out figures/llm_inloop_snapshot.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import sys
from matplotlib.patches import Rectangle

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulations.toy_closed_loop import (
    SquareLatticeDispersion,
    create_toy_structure,
    generate_hypotheses,
    discriminate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to closed_loop_checkpoint.json")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--log", default=None, help="Optional log file for posterior values")
    parser.add_argument("--results-csv", default=None, help="Optional CSV with model posteriors")
    parser.add_argument("--grid-h", type=int, default=220)
    parser.add_argument("--grid-e", type=int, default=220)
    parser.add_argument("--hmin", type=float, default=0.5)
    parser.add_argument("--hmax", type=float, default=1.7)
    parser.add_argument("--emin", type=float, default=0.5)
    parser.add_argument("--emax", type=float, default=30.0)
    parser.add_argument("--label-batches", action="store_true",
                        help="Label points with llm_batch_idx to show mode hopping")
    parser.add_argument("--batch-label-size", type=float, default=6.5,
                        help="Font size for batch labels")
    parser.add_argument("--true-j1", type=float, default=1.25)
    parser.add_argument("--true-j2", type=float, default=0.2)
    parser.add_argument("--true-d", type=float, default=0.02)
    parser.add_argument("--true-s", type=float, default=2.5)
    parser.add_argument("--outline-mode-regions", action="store_true",
                        help="Draw simple bounding boxes around mode-specific point clusters.")
    return parser.parse_args()


def _parse_posteriors_from_log(log_path: Path) -> dict:
    text = log_path.read_text()
    marker = "Model comparison (with fitted parameters):"
    idx = text.rfind(marker)
    if idx == -1:
        return {}
    chunk = text[idx:]
    post = {}
    current = None
    for line in chunk.splitlines():
        line = line.strip()
        if line.startswith("M") and ":" in line:
            current = line.split(":")[0]
        if "Posterior:" in line and current:
            try:
                val = float(line.split("Posterior:")[-1].strip().rstrip("%")) / 100.0
                post[current] = val
            except ValueError:
                continue
    return post


def _parse_true_params_from_log(log_path: Path) -> dict:
    text = log_path.read_text()
    marker = "M4: Full model"
    idx = text.find(marker)
    if idx == -1:
        return {}
    chunk = text[idx:].splitlines()
    params = {}
    for line in chunk[:40]:
        line = line.strip()
        if "J1 =" in line:
            params["J1"] = float(line.split("J1 =")[-1].split("±")[0].strip().split()[0])
        if "J2 =" in line:
            params["J2"] = float(line.split("J2 =")[-1].split("±")[0].strip().split()[0])
        if "D =" in line:
            params["D"] = float(line.split("D =")[-1].split("±")[0].strip().split()[0])
    return params


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.checkpoint).read_text())
    measurements = payload.get("measurements", [])

    # True model: prefer explicit args, else parse from log if provided.
    true_params = {}
    if args.log:
        true_params = _parse_true_params_from_log(Path(args.log))
    j1 = args.true_j1 if args.true_j1 is not None else true_params.get("J1", 1.25)
    j2 = args.true_j2 if args.true_j2 is not None else true_params.get("J2", 0.2)
    dval = args.true_d if args.true_d is not None else true_params.get("D", 0.02)
    true_model = SquareLatticeDispersion(J1=j1, J2=j2, D=dval, S=args.true_s)

    # Build intensity grid
    h = np.linspace(args.hmin, args.hmax, args.grid_h)
    e = np.linspace(args.emin, args.emax, args.grid_e)
    hh, ee = np.meshgrid(h, e, indexing="ij")
    intensity = np.zeros_like(hh)
    for i in range(hh.shape[0]):
        for j in range(hh.shape[1]):
            intensity[i, j] = true_model.intensity(hh[i, j], hh[i, j], ee[i, j])

    # Compute posteriors with fast fitting (or parse from log if provided)
    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    names = [c["name"] for c in candidates]
    labels = [f"M{i+1}" for i in range(len(names))]

    posteriors = None
    if args.log:
        log_post = _parse_posteriors_from_log(Path(args.log))
        if log_post:
            posteriors = [log_post.get(label, 0.0) for label in labels]

    if posteriors is None:
        if args.results_csv:
            import pandas as pd
            df = pd.read_csv(args.results_csv)
            post_map = {row["model"]: row["posterior"] for _, row in df.iterrows()}
            posteriors = [post_map.get(label, 0.0) for label in names]
        else:
            results = discriminate_models(measurements, candidates, use_bumps=False)
            posteriors = [results[n]["posterior"] for n in names]
    best_idx = int(np.argmax(posteriors))

    # Keep only points inside the plotted window to avoid off-axis markers/labels.
    in_window = (
        (np.array([m["h"] for m in measurements]) >= args.hmin)
        & (np.array([m["h"] for m in measurements]) <= args.hmax)
        & (np.array([m["E"] for m in measurements]) >= args.emin)
        & (np.array([m["E"] for m in measurements]) <= args.emax)
    )

    # Separate measurement types
    h_all = np.array([m["h"] for m in measurements])[in_window]
    e_all = np.array([m["E"] for m in measurements])[in_window]
    mode = np.array([m.get("mode", "") for m in measurements])[in_window]
    batch_idx = np.array(
        [m.get("llm_batch_idx", -1) if m.get("llm_batch_idx") is not None else -1 for m in measurements]
    )[in_window]
    llm_inject_mask = (
        np.array([bool(m.get("llm_inject_id")) for m in measurements])[in_window]
        | (mode == "loggp_inject")
        | (mode == "llm_points")
    )
    loggp_init_mask = (mode == "loggp_grid") & ~llm_inject_mask
    loggp_active_mask = (mode == "loggp_active") & ~llm_inject_mask
    physics_mask = (mode == "physics") & ~llm_inject_mask
    loggp_mask = np.array([bool(m.get("loggp_hint")) for m in measurements])[in_window] & ~llm_inject_mask
    symmetry_mask = np.array([bool(m.get("symmetry")) for m in measurements])[in_window] & ~llm_inject_mask

    h_llm = h_all[llm_inject_mask]
    e_llm = e_all[llm_inject_mask]
    h_loggp_init = h_all[loggp_init_mask]
    e_loggp_init = e_all[loggp_init_mask]
    h_loggp_active = h_all[loggp_active_mask]
    e_loggp_active = e_all[loggp_active_mask]
    h_loggp = h_all[loggp_mask & ~loggp_init_mask & ~loggp_active_mask & ~symmetry_mask]
    e_loggp = e_all[loggp_mask & ~loggp_init_mask & ~loggp_active_mask & ~symmetry_mask]
    h_sym = h_all[symmetry_mask]
    e_sym = e_all[symmetry_mask]
    h_tas = h_all[physics_mask]
    e_tas = e_all[physics_mask]

    # Backward-compatible fallback: older checkpoints may not include mode labels.
    if (
        len(h_all) > 0
        and np.sum(loggp_init_mask) == 0
        and np.sum(loggp_active_mask) == 0
        and np.sum(physics_mask) == 0
        and np.sum(llm_inject_mask) == 0
    ):
        h_tas = h_all
        e_tas = e_all

    # Plot
    fig = plt.figure(figsize=(10.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    data = np.log10(intensity + 1e-3)
    im = ax0.pcolormesh(hh, ee, data, shading="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label("log10(Intensity)")

    ax0.scatter(h_tas, e_tas, s=32, c="#00B3B3", edgecolor="black", linewidth=0.4, label="TAS-AI")
    if len(h_loggp_init) > 0:
        ax0.scatter(h_loggp_init, e_loggp_init, s=40, c="#FFD166", edgecolor="black",
                    linewidth=0.4, label="Log-GP init")
    if len(h_loggp_active) > 0:
        ax0.scatter(h_loggp_active, e_loggp_active, s=40, c="#118AB2", edgecolor="black",
                    linewidth=0.4, label="Log-GP active")
    if len(h_loggp) > 0:
        ax0.scatter(h_loggp, e_loggp, s=34, c="#5DBB63", edgecolor="black", linewidth=0.4, label="Log-GP")
    if len(h_sym) > 0:
        ax0.scatter(h_sym, e_sym, s=40, c="#D4AF37", edgecolor="black", linewidth=0.4, label="Symmetry seed")
    if len(h_llm) > 0:
        ax0.scatter(h_llm, e_llm, s=42, c="#EF476F", edgecolor="black", linewidth=0.5, label="LLM inject")

    if args.outline_mode_regions:
        region_specs = [
            ("Grid region", h_loggp_init, e_loggp_init, "#FFD166"),
            ("Active region", h_loggp_active, e_loggp_active, "#118AB2"),
            ("Physics region", h_tas, e_tas, "#00B3B3"),
        ]
        for _, hx, ex, color in region_specs:
            if len(hx) < 2 or len(ex) < 2:
                continue
            pad_h = max(0.03, 0.04 * (float(np.max(hx)) - float(np.min(hx)) + 1e-6))
            pad_e = max(0.25, 0.06 * (float(np.max(ex)) - float(np.min(ex)) + 1e-6))
            rect = Rectangle(
                (float(np.min(hx)) - pad_h, float(np.min(ex)) - pad_e),
                float(np.max(hx) - np.min(hx)) + 2 * pad_h,
                float(np.max(ex) - np.min(ex)) + 2 * pad_e,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                edgecolor=color,
                alpha=0.95,
                zorder=3,
            )
            ax0.add_patch(rect)

    if args.label_batches:
        for h, e, b, m in zip(h_all, e_all, batch_idx, mode):
            # Backward-compatible display: legacy checkpoints used batch=0 for both
            # pre-overseer grid and pre-overseer active points.
            b_disp = b
            if b == 0 and m == "loggp_grid":
                b_disp = -1
            if b_disp < 0:
                continue
            ax0.text(h + 0.01, e + 0.15, str(int(b_disp)),
                     fontsize=args.batch_label_size,
                     color="white",
                     ha="left",
                     va="bottom",
                     alpha=0.75,
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.3),
                     clip_on=True)

    ax0.set_xlim(args.hmin, args.hmax)
    ax0.set_ylim(args.emin, args.emax)
    ax0.set_xlabel("H (r.l.u.)")
    ax0.set_ylabel("E (meV)")
    ax0.set_title("Pilot overseer run")
    ax0.legend(loc="upper left", frameon=True)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(names))
    bars = ax1.bar(x, posteriors, color="#7AA6C2", edgecolor="black")
    bars[best_idx].set_color("#E27D60")
    ax1.set_ylim(0.0, max(0.6, max(posteriors) * 1.2))
    ax1.set_ylabel("Posterior")
    ax1.set_title(f"Favored: {labels[best_idx]}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, ha="center")

    for idx, val in enumerate(posteriors):
        ax1.text(x[idx], val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()

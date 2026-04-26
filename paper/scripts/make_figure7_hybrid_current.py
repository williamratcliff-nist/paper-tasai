#!/usr/bin/env python3
"""Build an updated Figure 7 from a checkpoint produced by hybrid runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulations.toy_closed_loop import (  # noqa: E402
    SquareLatticeDispersion,
    create_toy_structure,
    generate_hypotheses,
    discriminate_models,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hmin", type=float, default=0.5)
    p.add_argument("--hmax", type=float, default=1.7)
    p.add_argument("--emin", type=float, default=0.5)
    p.add_argument("--emax", type=float, default=30.0)
    p.add_argument("--grid-h", type=int, default=220)
    p.add_argument("--grid-e", type=int, default=220)
    p.add_argument("--true-j1", type=float, default=1.25)
    p.add_argument("--true-j2", type=float, default=0.2)
    p.add_argument("--true-d", type=float, default=0.02)
    p.add_argument("--true-s", type=float, default=2.5)
    p.add_argument("--posterior-step", type=int, default=5)
    p.add_argument("--compute-posteriors", action="store_true")
    p.add_argument("--fit-step", type=int, default=5,
                   help="Measurement stride for parameter-convergence fitting")
    return p.parse_args()


def _split_modes(measurements: List[Dict]) -> Dict[str, np.ndarray]:
    mode = np.array([m.get("mode", "") for m in measurements], dtype=object)
    mode_norm = np.array([
        "" if (v is None or str(v).strip() == "" or str(v).strip().lower() == "none") else str(v)
        for v in mode
    ], dtype=object)
    idx = np.arange(len(measurements))
    has_modes = np.any(mode_norm != "")
    if has_modes:
        return {
            "loggp_grid": mode_norm == "loggp_grid",
            "loggp_active": mode_norm == "loggp_active",
            "physics": mode_norm == "physics",
        }

    # Legacy checkpoints: infer phases from per-point flags, then by index as fallback.
    loggp_init = np.array([bool(m.get("loggp_init", False)) for m in measurements], dtype=bool)
    loggp_active = np.array([bool(m.get("loggp_active", False)) for m in measurements], dtype=bool)
    loggp_hint = np.array([bool(m.get("loggp_hint", False)) for m in measurements], dtype=bool)

    if np.any(loggp_init) or np.any(loggp_active) or np.any(loggp_hint):
        grid_mask = loggp_init.copy()
        active_mask = (~grid_mask) & (loggp_active | loggp_hint)
        physics_mask = ~(grid_mask | active_mask)
        return {
            "loggp_grid": grid_mask,
            "loggp_active": active_mask,
            "physics": physics_mask,
        }

    grid_n = min(31, len(measurements))
    active_n = min(30, max(0, len(measurements) - grid_n))
    return {
        "loggp_grid": idx < grid_n,
        "loggp_active": (idx >= grid_n) & (idx < grid_n + active_n),
        "physics": idx >= grid_n + active_n,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.checkpoint).read_text())
    measurements = payload.get("measurements", [])
    if not measurements:
        raise SystemExit("No measurements in checkpoint")

    model = SquareLatticeDispersion(
        J1=args.true_j1, J2=args.true_j2, D=args.true_d, S=args.true_s
    )

    h = np.linspace(args.hmin, args.hmax, args.grid_h)
    e = np.linspace(args.emin, args.emax, args.grid_e)
    hh, ee = np.meshgrid(h, e, indexing="ij")
    intensity = np.zeros_like(hh)
    for i in range(hh.shape[0]):
        for j in range(hh.shape[1]):
            intensity[i, j] = model.intensity(hh[i, j], hh[i, j], ee[i, j])

    h_all = np.array([m["h"] for m in measurements])
    e_all = np.array([m["E"] for m in measurements])
    in_window = (
        (h_all >= args.hmin) & (h_all <= args.hmax) &
        (e_all >= args.emin) & (e_all <= args.emax)
    )
    masks_all = _split_modes(measurements)
    idx_all = np.arange(len(measurements))

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)

    def draw_panel(ax, include_mask: np.ndarray, title: str, legend_loc: str = "upper left",
                   legend_bbox=None) -> None:
        im = ax.pcolormesh(hh, ee, np.log10(intensity + 1e-3), shading="auto", cmap="magma")
        n_grid = int(np.sum(include_mask & masks_all["loggp_grid"] & in_window))
        n_active = int(np.sum(include_mask & masks_all["loggp_active"] & in_window))
        n_phys = int(np.sum(include_mask & masks_all["physics"] & in_window))
        m = include_mask & masks_all["loggp_grid"] & in_window
        if np.any(m):
            ax.scatter(h_all[m], e_all[m], s=42, c="#FFD166", marker="o",
                       edgecolor="black", linewidth=0.55,
                       alpha=0.95, zorder=4, label="Log-GP grid")
        m = include_mask & masks_all["loggp_active"] & in_window
        if np.any(m):
            ax.scatter(h_all[m], e_all[m], s=50, c="#2A9D8F", marker="s",
                       edgecolor="black", linewidth=0.55,
                       alpha=0.95, zorder=5, label="Log-GP active")
        m = include_mask & masks_all["physics"] & in_window
        if np.any(m):
            ax.scatter(h_all[m], e_all[m], s=76, c="#FF006E", marker="P",
                       edgecolor="black", linewidth=0.70,
                       alpha=1.0, zorder=8, label="Physics")
        ax.set_xlim(args.hmin, args.hmax)
        ax.set_ylim(args.emin, args.emax)
        ax.set_xlabel("H (r.l.u.)")
        ax.set_ylabel("E (meV)")
        ax.set_title(f"{title} (grid={n_grid}, active={n_active}, physics={n_phys})")
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            fontsize=8,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
            title="Point types",
            title_fontsize=7.5,
        )
        return im

    mask_grid = masks_all["loggp_grid"]
    mask_active = masks_all["loggp_grid"] | masks_all["loggp_active"]
    mask_full = np.ones_like(idx_all, dtype=bool)
    grid_end = int(np.max(idx_all[masks_all["loggp_grid"]])) + 1 if np.any(masks_all["loggp_grid"]) else 0
    active_end = int(np.max(idx_all[masks_all["loggp_active"]])) + 1 if np.any(masks_all["loggp_active"]) else grid_end
    im = draw_panel(fig.add_subplot(gs[0, 0]), mask_grid, "(a) Phase 0: Log-GP grid")
    draw_panel(fig.add_subplot(gs[0, 1]), mask_active, "(b) Grid + Log-GP active")
    draw_panel(
        fig.add_subplot(gs[1, 0]),
        mask_full,
        "(c) Hybrid full path",
        legend_loc="upper right",
        legend_bbox=(0.985, 0.985),
    )

    ax = fig.add_subplot(gs[1, 1])
    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    steps = list(range(max(5, args.fit_step), len(measurements) + 1, args.fit_step))
    if steps[-1] != len(measurements):
        steps.append(len(measurements))
    j1_fit, j2_fit, d_fit = [], [], []
    post_traces = {
        "M1: NN Heisenberg": [],
        "M2: NN + anisotropy": [],
        "M3: J1-J2 no gap": [],
        "M4: Full model": [],
    }
    for k in steps:
        res = discriminate_models(measurements[:k], candidates, use_bumps=False)
        fit = res.get("M4: Full model", {})
        params = fit.get("params", {})
        j1 = float(params["J1"]) if "J1" in params else np.nan
        j2 = float(params["J2"]) if "J2" in params else np.nan
        d = float(params["D"]) if "D" in params else np.nan
        j1_fit.append(j1)
        j2_fit.append(j2)
        d_fit.append(d)
        for model_name in post_traces:
            post_traces[model_name].append(float(res.get(model_name, {}).get("posterior", 0.0)))

    colors = {
        "M1: NN Heisenberg": "#1F77B4",
        "M2: NN + anisotropy": "#FF7F0E",
        "M3: J1-J2 no gap": "#2CA02C",
        "M4: Full model": "#D62728",
    }
    markers = {
        "M1: NN Heisenberg": "o",
        "M2: NN + anisotropy": "s",
        "M3: J1-J2 no gap": "^",
        "M4: Full model": "D",
    }
    for model_name, vals in post_traces.items():
        ax.plot(
            steps, np.asarray(vals, dtype=float),
            lw=1.9,
            color=colors[model_name], markerfacecolor="white",
            markeredgecolor=colors[model_name], markeredgewidth=1.4,
            label=model_name.split(":")[0],
        )

    # Phase-coded background bands for posterior evolution.
    # Modes may interleave (overseer routing), so shade contiguous mode segments.
    n_meas = len(measurements)
    mode_by_idx = np.full(n_meas, "other", dtype=object)
    mode_by_idx[masks_all["loggp_grid"]] = "grid"
    mode_by_idx[masks_all["loggp_active"]] = "active"
    mode_by_idx[masks_all["physics"]] = "physics"
    mode_fill = {"grid": "#FFD166", "active": "#2A9D8F", "physics": "#FF006E"}
    mode_label = {
        "grid": "Log-GP grid",
        "active": "Log-GP active",
        "physics": "Physics",
    }
    i0 = 0
    while i0 < n_meas:
        m0 = mode_by_idx[i0]
        i1 = i0 + 1
        while i1 < n_meas and mode_by_idx[i1] == m0:
            i1 += 1
        if m0 in mode_fill:
            ax.axvspan(i0 + 1, i1, color=mode_fill[m0], alpha=0.10, zorder=0)
            xmid = 0.5 * ((i0 + 1) + i1)
            y_axes = 0.18 if xmid < 0.30 * n_meas else 0.97
            va = "bottom" if y_axes < 0.5 else "top"
            ax.text(
                xmid,
                y_axes,
                mode_label[m0],
                rotation=90,
                ha="center",
                va=va,
                transform=ax.get_xaxis_transform(),
                fontsize=7.6,
                fontweight="bold",
                color=mode_fill[m0],
                alpha=0.8,
                zorder=1,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.7),
            )
        i0 = i1

    ax.set_xlabel("Measurement #")
    ax.set_ylabel("Posterior")
    ax.set_title("(d) Model posterior evolution")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(1, n_meas)

    ax.legend(loc="upper left", fontsize=7.4, ncol=2, frameon=True)

    cax = fig.add_axes([0.92, 0.56, 0.015, 0.33])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log10(Intensity)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

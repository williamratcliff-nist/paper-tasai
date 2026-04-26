#!/usr/bin/env python3
"""Build Figure 11 with updated data, lattice context, and posterior evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    p.add_argument("--baseline", required=True, help="Checkpoint JSON for baseline run")
    p.add_argument("--mitigated", required=True, help="Checkpoint JSON for mitigated run")
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
    p.add_argument("--no-bumps", action="store_true",
                   help="Use fast (non-bumps) posterior evolution")
    p.add_argument("--legacy-grid-n", type=int, default=None,
                   help="Force grid length for legacy checkpoints without mode labels")
    p.add_argument("--legacy-active-n", type=int, default=None,
                   help="Force active length for legacy checkpoints without mode labels")
    return p.parse_args()


def _load_measurements(path: str) -> List[Dict]:
    payload = json.loads(Path(path).read_text())
    return payload.get("measurements", [])


def _mode_masks(measurements: List[Dict], legacy_grid_n: int | None = None,
                legacy_active_n: int | None = None) -> Dict[str, np.ndarray]:
    n = len(measurements)
    idx = np.arange(n)
    mode = np.array([m.get("mode", "") for m in measurements], dtype=object)
    mode = np.array([
        "" if (v is None or str(v).strip() == "" or str(v).strip().lower() == "none") else str(v)
        for v in mode
    ], dtype=object)
    if np.any(mode != ""):
        return {
            "grid": mode == "loggp_grid",
            "active": mode == "loggp_active",
            "physics": mode == "physics",
        }

    if legacy_grid_n is not None and legacy_active_n is not None:
        g = max(0, min(int(legacy_grid_n), n))
        a = max(0, min(int(legacy_active_n), n - g))
        return {
            "grid": idx < g,
            "active": (idx >= g) & (idx < g + a),
            "physics": idx >= g + a,
        }

    loggp_init = np.array([bool(m.get("loggp_init", False)) for m in measurements], dtype=bool)
    loggp_active = np.array([bool(m.get("loggp_active", False)) for m in measurements], dtype=bool)
    loggp_hint = np.array([bool(m.get("loggp_hint", False)) for m in measurements], dtype=bool)
    if np.any(loggp_init) or np.any(loggp_active) or np.any(loggp_hint):
        grid_mask = loggp_init.copy()
        active_mask = (~grid_mask) & (loggp_active | loggp_hint)
        physics_mask = ~(grid_mask | active_mask)
        return {"grid": grid_mask, "active": active_mask, "physics": physics_mask}

    grid_n = min(31, n)
    active_n = min(30, max(0, n - grid_n))
    return {
        "grid": idx < grid_n,
        "active": (idx >= grid_n) & (idx < grid_n + active_n),
        "physics": idx >= grid_n + active_n,
    }


def _draw_lattice_inset(ax):
    iax = inset_axes(ax, width="35%", height="35%", loc="upper right", borderpad=1.0)
    iax.set_xlim(-0.1, 1.1)
    iax.set_ylim(-0.1, 1.1)
    fe_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    o_pts = np.array([[0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]], dtype=float)
    iax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.25", lw=0.8)
    iax.scatter(fe_pts[:, 0], fe_pts[:, 1], s=45, c="#C1121F",
                edgecolors="black", linewidths=0.5, zorder=3)
    iax.scatter(o_pts[:, 0], o_pts[:, 1], s=25, c="#3A86FF", marker="s",
                edgecolors="black", linewidths=0.4, zorder=2)
    iax.text(0.50, -0.12, "Fe–O–Fe", ha="center", va="top", fontsize=6)
    iax.set_xticks([])
    iax.set_yticks([])
    iax.set_facecolor((1, 1, 1, 0.88))
    for spine in iax.spines.values():
        spine.set_linewidth(0.6)


def _posterior_trace(measurements: List[Dict], step: int, use_bumps: bool) -> Dict[str, np.ndarray]:
    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    names = [c["name"] for c in candidates]
    n = len(measurements)
    steps = list(range(max(5, step), n + 1, step))
    if not steps or steps[-1] != n:
        steps.append(n)
    post = {name: [] for name in names}
    for k in steps:
        res = discriminate_models(measurements[:k], candidates, use_bumps=use_bumps)
        for name in names:
            post[name].append(float(res[name]["posterior"]))
    return {
        "steps": np.asarray(steps, dtype=int),
        "names": np.asarray(names, dtype=object),
        "posteriors": {k: np.asarray(v, dtype=float) for k, v in post.items()},
    }


def _plot_column(fig, gs_col, measurements: List[Dict], title: str, model, hlim, elim,
                 posterior_step: int, use_bumps: bool,
                 legacy_grid_n: int | None, legacy_active_n: int | None):
    ax_cov = fig.add_subplot(gs_col[0, 0])
    ax_post = fig.add_subplot(gs_col[1, 0])

    h = np.linspace(hlim[0], hlim[1], 220)
    e = np.linspace(elim[0], elim[1], 220)
    hh, ee = np.meshgrid(h, e, indexing="ij")
    inten = np.zeros_like(hh)
    for i in range(hh.shape[0]):
        for j in range(hh.shape[1]):
            inten[i, j] = model.intensity(hh[i, j], hh[i, j], ee[i, j])
    ax_cov.pcolormesh(hh, ee, np.log10(inten + 1e-3), shading="auto", cmap="magma")

    masks_full = _mode_masks(measurements, legacy_grid_n=legacy_grid_n, legacy_active_n=legacy_active_n)

    h_all = np.array([m["h"] for m in measurements])
    e_all = np.array([m["E"] for m in measurements])
    inw = (
        (h_all >= hlim[0]) & (h_all <= hlim[1]) &
        (e_all >= elim[0]) & (e_all <= elim[1])
    )
    h_all = h_all[inw]
    e_all = e_all[inw]
    mwin = [m for i, m in enumerate(measurements) if inw[i]]
    masks = {k: v[inw] for k, v in masks_full.items()}

    if np.any(masks["grid"]):
        ax_cov.scatter(h_all[masks["grid"]], e_all[masks["grid"]], s=35, c="#FFD166",
                       edgecolor="black", linewidth=0.5, label="Log-GP grid", zorder=4)
    if np.any(masks["active"]):
        ax_cov.scatter(h_all[masks["active"]], e_all[masks["active"]], s=38, c="#2A9D8F", marker="s",
                       edgecolor="black", linewidth=0.5, label="Log-GP active", zorder=5)
    if np.any(masks["physics"]):
        ax_cov.scatter(h_all[masks["physics"]], e_all[masks["physics"]], s=44, c="#E76F51", marker="^",
                       edgecolor="black", linewidth=0.5, label="Physics", zorder=6)

    ax_cov.set_xlim(*hlim)
    ax_cov.set_ylim(*elim)
    ax_cov.set_xlabel("H (r.l.u.)")
    ax_cov.set_ylabel("E (meV)")
    ax_cov.set_title(title)
    ax_cov.legend(loc="upper left", fontsize=7.7, frameon=True)
    _draw_lattice_inset(ax_cov)

    tr = _posterior_trace(measurements, posterior_step, use_bumps)
    color_map = {
        "M1: NN Heisenberg": "#457B9D",
        "M2: NN + anisotropy": "#F4A261",
        "M3: J1-J2 no gap": "#6D597A",
        "M4: Full model": "#E63946",
    }
    for name in tr["names"]:
        y = tr["posteriors"][str(name)]
        ax_post.plot(tr["steps"], y, lw=1.8, marker="o", ms=3,
                     color=color_map.get(str(name), None), label=str(name).split(":")[0])
    ax_post.set_ylim(0.0, 1.02)
    ax_post.set_xlabel("Measurement #")
    ax_post.set_ylabel("Posterior")
    ax_post.grid(alpha=0.25, linewidth=0.6)
    ax_post.legend(loc="upper left", fontsize=7.3, ncol=2, frameon=True)


def main() -> None:
    args = parse_args()
    baseline = _load_measurements(args.baseline)
    mitigated = _load_measurements(args.mitigated)
    model = SquareLatticeDispersion(J1=args.true_j1, J2=args.true_j2, D=args.true_d, S=args.true_s)

    fig = plt.figure(figsize=(12.2, 8.0))
    outer = fig.add_gridspec(1, 2, wspace=0.22)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.20)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.20)

    _plot_column(
        fig, left, baseline, "(a) Baseline closed loop", model,
        (args.hmin, args.hmax), (args.emin, args.emax),
        posterior_step=args.posterior_step, use_bumps=not args.no_bumps,
        legacy_grid_n=args.legacy_grid_n, legacy_active_n=args.legacy_active_n,
    )
    _plot_column(
        fig, right, mitigated, "(b) Mitigated closed loop", model,
        (args.hmin, args.hmax), (args.emin, args.emax),
        posterior_step=args.posterior_step, use_bumps=not args.no_bumps,
        legacy_grid_n=args.legacy_grid_n, legacy_active_n=args.legacy_active_n,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

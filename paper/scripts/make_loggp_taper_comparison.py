#!/usr/bin/env python3
"""Regenerate SI Figure S1 from the current library-backed Log-GP phase."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulations.toy_closed_loop import (  # noqa: E402
    H_RANGE_MAX,
    H_RANGE_MIN,
    LOGGP_E_MAX,
    SquareLatticeDispersion,
    run_loggp_phase,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO_ROOT / "paper" / "figures" / "loggp_taper_comparison.png"))
    p.add_argument("--measurements", type=int, default=24)
    p.add_argument("--grid-h", type=int, default=24)
    p.add_argument("--grid-e", type=int, default=18)
    return p.parse_args()


def build_true_model() -> SquareLatticeDispersion:
    return SquareLatticeDispersion(J1=1.25, J2=0.2, D=0.02, S=2.5)


def intensity_map(model: SquareLatticeDispersion, h_vals: np.ndarray, e_vals: np.ndarray) -> np.ndarray:
    out = np.zeros((len(h_vals), len(e_vals)), dtype=float)
    for i, h in enumerate(h_vals):
        for j, e in enumerate(e_vals):
            out[i, j] = model.intensity(float(h), float(h), float(e))
    return out


def split_points(measurements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    init = np.array([(m["h"], m["E"]) for m in measurements if m.get("loggp_init")], dtype=float)
    active = np.array([(m["h"], m["E"]) for m in measurements if m.get("loggp_active")], dtype=float)
    return init, active


def main() -> None:
    args = parse_args()
    true_model = build_true_model()
    modes = [
        ("none", "No taper"),
        ("energy", "1D energy taper"),
        ("energy_h", "2D taper"),
    ]

    h_vals = np.linspace(H_RANGE_MIN, H_RANGE_MAX, 220)
    e_vals = np.linspace(0.0, LOGGP_E_MAX, 260)
    intensity = intensity_map(true_model, h_vals, e_vals)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (mode, title) in zip(axes, modes):
        measurements, _, first_active = run_loggp_phase(
            true_model=true_model,
            n_measurements=args.measurements,
            hmin=H_RANGE_MIN,
            hmax=H_RANGE_MAX,
            emin=0.0,
            emax=LOGGP_E_MAX,
            grid_h=args.grid_h,
            grid_e=args.grid_e,
            taper_mode=mode,
        )
        init_pts, active_pts = split_points(measurements)

        mesh = ax.pcolormesh(
            h_vals,
            e_vals,
            np.log10(intensity.T + 1e-3),
            shading="auto",
            cmap="magma",
        )
        if len(init_pts):
            ax.scatter(init_pts[:, 0], init_pts[:, 1], s=26, c="#FFD166", edgecolor="black", linewidth=0.35, label="Init")
        if len(active_pts):
            ax.scatter(active_pts[:, 0], active_pts[:, 1], s=34, c="#2A9D8F", edgecolor="black", linewidth=0.4, label="Active")
        ax.set_title(title)
        ax.set_xlabel("H along [H,H,0]")
        if first_active is not None:
            ax.text(
                0.97,
                0.04,
                f"first active #{first_active}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, linewidth=0.4),
            )
        ax.set_xlim(H_RANGE_MIN, H_RANGE_MAX)
        ax.set_ylim(0.0, LOGGP_E_MAX)
        ax.grid(alpha=0.12, linewidth=0.4)

    axes[0].set_ylabel("Energy (meV)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label(r"$\log_{10}(I + 10^{-3})$")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

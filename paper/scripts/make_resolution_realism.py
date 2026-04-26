#!/usr/bin/env python3
"""Generate a resolution-broadened realism panel for a JCNS scenario."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure4_realism.png"

# Ensure the bundled resolution dependency is on path.
DEPS_ROOT = ROOT.parent / "deps"
sys.path.insert(0, str(DEPS_ROOT))

try:
    from tasai.instrument import TASResolutionCalculator, create_default_tas_config
    HAS_RESOLUTION = True
except Exception:
    HAS_RESOLUTION = False


def intensity_single_branch(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    j = 5.0
    d = 0.1
    gamma = 0.5
    amp = 1.0
    background = 0.01
    omega = 2 * j * (1 - np.cos(2 * np.pi * h)) + d
    return amp * gamma / ((e - omega) ** 2 + gamma**2) / np.pi + background


def resolution_sigma(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    if not HAS_RESOLUTION:
        return np.full_like(e, 0.15)
    res_calc = TASResolutionCalculator(
        lattice_params=(4.0, 4.0, 10.0, 90, 90, 90),
        orient1=[1, 0, 0],
        orient2=[0, 1, 0],
        exp_config=create_default_tas_config(efixed=14.7, hcol=(40, 40, 40, 40)),
    )
    sigma = np.empty_like(e, dtype=float)
    for idx, (hh, ee) in enumerate(zip(h, e)):
        fwhm, _ = res_calc.get_resolution_fwhm(h=float(hh), k=0.0, l=0.0, E=float(ee))
        sigma[idx] = fwhm["E"] / 2.355
    return sigma


def convolve_energy(intensity: np.ndarray, energy: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    out = np.zeros_like(intensity)
    for i, (e0, sig) in enumerate(zip(energy, sigma)):
        if not np.isfinite(sig) or sig <= 0:
            out[i] = intensity[i]
            continue
        weights = np.exp(-0.5 * ((energy - e0) / sig) ** 2)
        weights /= weights.sum()
        out[i] = np.sum(intensity * weights)
    return out


def main() -> None:
    h_vals = np.linspace(0.02, 0.48, 120)
    e_vals = np.linspace(0.5, 20.0, 160)
    h_grid, e_grid = np.meshgrid(h_vals, e_vals)

    base = intensity_single_branch(h_grid, e_grid)

    conv = np.zeros_like(base)
    for idx, h in enumerate(h_vals):
        sigma = resolution_sigma(np.full_like(e_vals, h), e_vals)
        conv[:, idx] = convolve_energy(base[:, idx], e_vals, sigma)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    vmin, vmax = np.percentile(base, [1, 99])

    im0 = axes[0].imshow(
        base,
        origin="lower",
        aspect="auto",
        extent=(h_vals.min(), h_vals.max(), e_vals.min(), e_vals.max()),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[0].set_title("Single-branch (analytic)")
    axes[0].set_xlabel("H (r.l.u.)")
    axes[0].set_ylabel("Energy (meV)")

    im1 = axes[1].imshow(
        conv,
        origin="lower",
        aspect="auto",
        extent=(h_vals.min(), h_vals.max(), e_vals.min(), e_vals.max()),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[1].set_title("Cooper-Nathans broadened")
    axes[1].set_xlabel("H (r.l.u.)")

    fig.colorbar(im1, ax=axes, location="right", label="Intensity (a.u.)")
    fig.suptitle("Resolution realism (single_branch)", fontsize=11)
    fig.savefig(OUT, dpi=300)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Figure 4 scenario panels for the analytic JCNS benchmarks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tasai.examples import benchmark_jcns as jcns
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure4_scenarios.png"

SCENARIOS = [
    "single_branch",
    "two_branches",
    "weak_signal",
    "sharp_feature",
    "gap_mode",
]


def _make_resolution_calculator(backend: str = "numba"):
    try:
        from tasai.instrument import TASResolutionCalculator, create_default_tas_config
    except Exception as exc:  # pragma: no cover - runtime import check
        raise RuntimeError(f"Resolution calculator unavailable: {exc}") from exc
    return TASResolutionCalculator(
        lattice_params=(4.0, 4.0, 10.0, 90, 90, 90),
        orient1=[1, 0, 0],
        orient2=[0, 1, 0],
        exp_config=create_default_tas_config(
            efixed=14.7,
            hcol=(40, 40, 40, 40),
            vcol=(120, 120, 120, 120),
        ),
        backend=backend,
    )


def _plot_panel(ax, title, bounds, intensity, label, smoothing_sigma=None, n_h=160, n_e=200):
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')
    H = np.linspace(bounds[0, 0], bounds[0, 1], n_h)
    E = np.linspace(bounds[1, 0], bounds[1, 1], n_e)
    HH, EE = np.meshgrid(H, E, indexing='ij')
    try:
        values = intensity(HH, EE)
    except Exception:
        vectorized = np.vectorize(intensity)
        values = vectorized(HH, EE)
    if smoothing_sigma and gaussian_filter is not None:
        values = gaussian_filter(values, sigma=smoothing_sigma)
    im = ax.pcolormesh(H, E, values.T, shading='auto', cmap='inferno')
    ax.set_title(title)
    ax.set_xlabel('[H H 0] (r.l.u.)')
    ax.set_ylabel('Energy (meV)')
    return im


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.4), sharey=False)
    axes = np.atleast_2d(axes)
    panel_labels = [f"({chr(97 + i)})" for i in range(5)]

    # Analytic JCNS scenarios (panels a-e).
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for idx, name in enumerate(SCENARIOS):
        row, col = layout[idx]
        scenario = jcns.BENCHMARK_SCENARIOS[name]
        im = _plot_panel(
            axes[row, col],
            name.replace('_', ' ').title(),
            scenario['bounds'],
            scenario['function'],
            panel_labels[idx],
        )
        for label, func in scenario.get('dispersions', []):
            H = np.linspace(scenario['bounds'][0, 0], scenario['bounds'][0, 1], 160)
            axes[row, col].plot(H, func(H), lw=1.2, label=label)
        if scenario.get('dispersions'):
            axes[row, col].legend(fontsize=8, loc='upper right')

    cax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    fig.colorbar(im, cax=cax, label='Intensity (a.u.)')
    axes[1, 2].axis('off')
    plt.subplots_adjust(right=0.9, hspace=0.45, wspace=0.3)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

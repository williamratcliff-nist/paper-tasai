#!/usr/bin/env python3
"""Build a schematic of the archived ghost-optic benchmark used in Section 5.3.1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "ablation_runs" / "ghost_optic" / "none_seed000_summary.json"
OUT = ROOT / "figures" / "ghost_optic_schematic_20260415.png"


def lorentz(energy: np.ndarray, center: float, amplitude: float, gamma: float) -> np.ndarray:
    return amplitude * gamma / (((energy - center) ** 2) + gamma**2)


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    cfg = payload["ghost_optic_config"]

    energy = np.linspace(cfg["energy_min"], cfg["energy_max"], 1000, dtype=float)
    acoustic = lorentz(energy, cfg["acoustic_energy"], cfg["acoustic_amplitude"], cfg["gamma"]) + cfg["background"]
    acoustic_optic = acoustic + lorentz(
        energy,
        cfg["optic_energy"],
        cfg["acoustic_amplitude"] * cfg["optic_fraction"],
        cfg["gamma"],
    )

    seed_energies = np.array(cfg["init_energies"], dtype=float)
    seed_y = lorentz(seed_energies, cfg["acoustic_energy"], cfg["acoustic_amplitude"], cfg["gamma"]) + cfg["background"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    ax.plot(energy, acoustic, color="#254b6b", lw=2.6, label=r"$M_A$: acoustic only")
    ax.plot(energy, acoustic_optic, color="#bf5b04", lw=2.2, ls="--", label=r"$M_B$: acoustic + weak optic")
    ax.scatter(seed_energies, seed_y, color="#111111", s=28, zorder=4, label="common seed")
    ax.axvspan(cfg["optic_energy"] - 1.2, cfg["optic_energy"] + 1.2, color="#f3c98b", alpha=0.35, label="falsification region")
    ax.axvline(cfg["optic_energy"], color="#bf5b04", lw=1.2, alpha=0.7)

    ax.text(
        6.9,
        acoustic.max() * 0.22,
        "dominant acoustic branch",
        color="#254b6b",
        fontsize=10,
        ha="left",
        va="center",
    )
    ax.text(
        15.25,
        acoustic.max() * 0.22,
        "weak optic branch",
        color="#8f4303",
        fontsize=10,
        ha="left",
        va="center",
    )

    ax.set_xlim(cfg["energy_min"], cfg["energy_max"])
    ax.set_xlabel("Energy transfer E (meV)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.set_title("Ghost-Optic Benchmark")
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.47, 0.77),
        frameon=False,
        fontsize=9,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

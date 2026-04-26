#!/usr/bin/env python
"""
Closed-Loop Autonomous Magnetism Demo: Complete Toy System

This demonstrates the full closed-loop workflow on a simple 2D square lattice
antiferromagnet. The demo uses heuristic-based hypothesis generation (which
would be replaced by CHGNet/M3GNet in production).

The toy system:
- 2D square lattice with Fe ions
- True Hamiltonian: H = J₁ Σ S_i·S_j + J₂ Σ S_i·S_k + D Σ (S_i^z)²
- True parameters: J₁ = 5.0 meV, J₂ = 0.8 meV, D = 0.1 meV
- Dispersion: ω(q) = S√[(J₁(cos(πh)+cos(πk)) + J₂(cos(2πh)+cos(2πk)) + D)² - ...]

Workflow:
1. Define crystal structure
2. GNN (heuristic) proposes candidate Hamiltonians
3. MCTS plans measurement batch
4. Simulate measurements with true model (realistic TAS resolution)
5. Bayesian discrimination identifies correct model
6. Report validated parameters

TAS Configuration:
- Cooper-Nathans resolution calculation
- 40' horizontal collimations throughout
- Fixed Ef = 14.7 meV
- Includes kinematic accessibility checks

Usage:
    python toy_closed_loop.py

Output:
    - Console log of full workflow
    - figures/closed_loop_*.png visualization files
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import argparse
import sys
import json
import multiprocessing
import time
import subprocess
import shutil
import shlex
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Any, Dict, List, Tuple, Optional
import itertools
import logging

# Ensure the real rescalculator module is available before importing TAS helpers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
deps_dir = PROJECT_ROOT / 'deps'
rescalc_candidates = [deps_dir]
env_path = os.environ.get('RESCALCULATOR_PATH')
if env_path:
    rescalc_candidates.append(Path(env_path))
for candidate in rescalc_candidates:
    if candidate and candidate.exists():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

# Provide a lattice_calculator shim if the external dependency is missing.
try:
    import lattice_calculator  # noqa: F401
except ImportError:
    from tasai.instrument._lattice_compat import (
        Lattice, modvec, scalar, Orientation
    )

    import types

    lattice_module = types.ModuleType('lattice_calculator')
    lattice_module.Lattice = Lattice
    lattice_module.modvec = modvec
    lattice_module.scalar = scalar
    lattice_module.Orientation = Orientation
    sys.modules['lattice_calculator'] = lattice_module

BAND_EDGE_SCAN_RANGE = np.linspace(0.65, 0.85, 5)
H_RANGE_MIN = 0.0
H_RANGE_MAX = 2.0
LOGGP_E_MAX = 30.0
OMEGA_GUARD_MIN = 0.5
OMEGA_GUARD_MARGIN = 0.5
LOGGP_LEVEL_BACKGR_DIFFS_REL_MAX = 0.35
LOGGP_LEVEL_BACKGR_DIFFS_ABS_MIN = 0.4
LOGGP_LEVEL_BACKGR_DECILE_MAX = 5
LOGGP_THRESH_INTENS_FACT = 0.65
LOGGP_ELLIPSE_H_WIDTH = 0.08
LOGGP_ELLIPSE_E_SCALE = 3.0
LOGGP_MOVE_VH = 0.12
LOGGP_MOVE_VE = 2.0
LOGGP_MOVE_OVERHEAD = 3.0
LOGGP_INIT_GRID = 11
LOGGP_EDGE_TAPER = 0.10  # fraction of energy range to taper near boundaries
LOGGP_VAR_CLAMP = 2.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import TAS resolution calculator
try:
    from tasai.instrument import TASResolutionCalculator, create_default_tas_config
    HAS_RESOLUTION = True
except Exception as exc:  # pragma: no cover - logging fallback
    HAS_RESOLUTION = False
    logger.exception(
        "TAS resolution calculator not available, using simple Gaussian"
    )

BASE_HAS_RESOLUTION = HAS_RESOLUTION

try:
    from tasai.core.gaussian_process import LogGaussianProcess
except Exception as exc:  # pragma: no cover - fallback for partial envs
    LogGaussianProcess = None  # type: ignore[assignment]
    logger.warning("tasai.core.gaussian_process unavailable (%s); hybrid loop will fall back to local GP", exc)

# Create figures directory (output to ../figures for paper repo structure)
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

PROMPTS_DIR = Path(__file__).parent.parent / 'prompts'
LLM_TEMPLATE = PROMPTS_DIR / 'llm_inloop_template.txt'


def format_measurements_for_llm(measurements: List[Dict],
                                max_points: int = 20) -> str:
    """Create a compact, redacted measurement table for LLM prompts."""
    if not measurements:
        return "No measurements yet."

    recent = measurements[-max_points:]
    lines = ["idx,H,E,intensity,uncertainty,count_time"]
    offset = max(len(measurements) - len(recent), 0)
    for i, m in enumerate(recent, 1):
        idx = offset + i
        lines.append(
            f"{idx},{m['h']:.3f},{m['E']:.3f},{m['intensity']:.3f},"
            f"{m['uncertainty']:.3f},{m.get('count_time', 0.0):.1f}"
        )
    return "\n".join(lines)


def build_llm_prompt(measurements: List[Dict], max_points: int) -> str:
    """Render the in-loop LLM prompt with redacted measurements."""
    template = LLM_TEMPLATE.read_text()
    table = format_measurements_for_llm(measurements, max_points=max_points)
    return template.replace("{{MEASUREMENT_TABLE}}", table)


def _run_llm_command(command: str, prompt: str) -> str:
    """Run an LLM CLI command with prompt on stdin, returning stdout."""
    cmd = shlex.split(command)
    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout.decode("utf-8", errors="ignore")


def _extract_json_block(text: str) -> Optional[Dict]:
    """Extract the first JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_llm_suggestions(text: str) -> List[Dict]:
    """Parse LLM JSON suggestions into a list of point dicts."""
    payload = _extract_json_block(text)
    if not payload or "suggestions" not in payload:
        return []
    suggestions = []
    for item in payload["suggestions"]:
        try:
            h = float(item["h"])
            e = float(item["e"])
        except Exception:
            continue
        suggestions.append({
            "h": h,
            "k": 0.0,
            "E": e,
            "llm_hint": True,
            "reason": str(item.get("reason", ""))[:200]
        })
    return suggestions


class SimpleGaussianProcess:
    """Lightweight GP surrogate for agnostic exploration (log-intensity)."""

    def __init__(self, length_scale: float = 0.2, noise: float = 1.0):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []

    def add_observation(self, x: np.ndarray, y: float) -> None:
        self.X_train.append(x)
        self.y_train.append(y)

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        d = np.sum((x1 - x2) ** 2)
        return float(np.exp(-d / (2 * self.length_scale ** 2)))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X)) * 10.0

        X_train = np.array(self.X_train)
        y_train = np.array(self.y_train)
        n = len(X_train)

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(X_train[i], X_train[j])
        K += self.noise * np.eye(n)

        K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
        means = []
        variances = []
        for x in X:
            k_star = np.array([self._kernel(x, xi) for xi in X_train])
            mean = k_star @ K_inv @ y_train
            var = self._kernel(x, x) - k_star @ K_inv @ k_star
            means.append(mean)
            variances.append(max(var, 1e-8))
        return np.array(means), np.array(variances)


class LibraryBackedLogGPSurrogate:
    """
    Paper-loop adapter around the reusable library Log-GP.

    The closed-loop paper demo still applies its own acquisition safeguards
    (tau/gamma clamp, energy taper, consumed-area exclusion, motion penalty),
    but the predictive model itself is the shared TAS-AI Log-GP rather than
    the earlier paper-local surrogate.
    """

    def __init__(self,
                 hmin: float,
                 hmax: float,
                 emin: float,
                 emax: float,
                 background: float = 1.0):
        self.hmin = float(hmin)
        self.hmax = float(hmax)
        self.emin = float(emin)
        self.emax = float(emax)
        self.background = max(float(background), 1e-3)
        self._fallback = SimpleGaussianProcess(length_scale=0.25, noise=1.0)
        self._gp = None
        if LogGaussianProcess is not None:
            self._gp = LogGaussianProcess(
                length_scales=np.array([
                    max((self.hmax - self.hmin) / 5.0, 0.05),
                    max((self.emax - self.emin) / 5.0, 1.0),
                ]),
                background=self.background,
                noise_level=0.1,
                n_dims=2,
            )

    @property
    def using_library(self) -> bool:
        return self._gp is not None

    def refit(self,
              measurements: List[Dict],
              level_backgr: Optional[float],
              thresh_intens: Optional[float]) -> None:
        if self._gp is None:
            self._fallback = SimpleGaussianProcess(length_scale=0.25, noise=1.0)
            for meas in measurements:
                capped = min(
                    float(meas["intensity"]),
                    float(thresh_intens) if thresh_intens is not None else float(meas["intensity"]),
                )
                adjusted = max(
                    0.0,
                    capped - (float(level_backgr) if level_backgr is not None else 0.0),
                )
                self._fallback.add_observation(
                    np.array([float(meas["h"]), float(meas["E"]) / LOGGP_E_MAX]),
                    np.log1p(adjusted),
                )
            return

        self._gp = LogGaussianProcess(
            length_scales=np.array([
                max((self.hmax - self.hmin) / 5.0, 0.05),
                max((self.emax - self.emin) / 5.0, 1.0),
            ]),
            background=self.background,
            noise_level=0.1,
            n_dims=2,
        )
        for meas in measurements:
            capped = min(
                float(meas["intensity"]),
                float(thresh_intens) if thresh_intens is not None else float(meas["intensity"]),
            )
            adjusted = max(
                0.0,
                capped - (float(level_backgr) if level_backgr is not None else 0.0),
            )
            sigma = max(float(meas.get("uncertainty", 1.0)), 1e-4)
            self._gp.add_observation(
                np.array([float(meas["h"]), float(meas["E"])]),
                adjusted,
                sigma,
            )

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._gp is None:
            X_scaled = np.column_stack([X[:, 0], X[:, 1] / LOGGP_E_MAX])
            mu_z, var_z = self._fallback.predict(X_scaled)
            var_clamped = np.minimum(var_z, LOGGP_VAR_CLAMP)
            mean = np.expm1(mu_z + 0.5 * var_clamped)
            std = np.sqrt(np.maximum((np.exp(var_clamped) - 1.0) * np.exp(2.0 * mu_z + var_clamped), 1e-8))
            return mean, std
        return self._gp.predict_batch(np.asarray(X, dtype=float))


def _cosine_edge_weight(norm_vals: np.ndarray, taper: float) -> np.ndarray:
    weights = np.ones_like(norm_vals, dtype=float)
    if taper <= 0:
        return weights
    mask_low = norm_vals < taper
    if np.any(mask_low):
        weights[mask_low] = 0.5 * (1.0 - np.cos(np.pi * norm_vals[mask_low] / taper))
    mask_high = norm_vals > (1.0 - taper)
    if np.any(mask_high):
        weights[mask_high] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - norm_vals[mask_high]) / taper))
    return weights


def _loggp_window_weight(
    h_vals: np.ndarray,
    e_vals: np.ndarray,
    hmin: float,
    hmax: float,
    emin: float,
    emax: float,
    taper_mode: str,
) -> np.ndarray:
    taper = LOGGP_EDGE_TAPER
    if taper_mode == "none":
        return np.ones_like(e_vals, dtype=float)
    e_norm = (e_vals - emin) / max(emax - emin, 1e-9)
    w_e = _cosine_edge_weight(e_norm, taper)
    if taper_mode == "energy":
        return w_e
    if taper_mode == "energy_h":
        h_norm = (h_vals - hmin) / max(hmax - hmin, 1e-9)
        w_h = _cosine_edge_weight(h_norm, taper)
        return w_e * w_h
    raise ValueError(f"Unknown Log-GP taper mode: {taper_mode}")


def compute_heuris_experi_param(intens: np.ndarray,
                                level_backgr_diffs_rel_max: float,
                                level_backgr_diffs_abs_min: float,
                                level_backgr_decile_max: int,
                                thresh_intens_fact: float,
                                level_backgr: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
    """Compute background level and intensity threshold from decile statistics."""
    if len(intens) < 10:
        return level_backgr, None

    deciles = [-np.inf] + [np.percentile(intens, q=10 * n) for n in range(1, 10)] + [np.inf]
    buckets = [intens[np.logical_and(deciles[n] < intens, intens <= deciles[n + 1])]
               for n in range(10)]
    medians = np.array([np.median(bucket) if len(bucket) > 0 else deciles[n]
                        for n, bucket in enumerate(buckets)])

    diffs_abs = medians[1:] - medians[:-1]
    diffs_rel = diffs_abs / np.maximum(medians[:-1], 1e-9)

    if level_backgr is None:
        index_level_backgr = min(level_backgr_decile_max - 1,
                                 np.argmax(
                                     np.logical_or(
                                         np.logical_and(
                                             diffs_rel > level_backgr_diffs_rel_max,
                                             diffs_abs >= level_backgr_diffs_abs_min),
                                         np.arange(len(diffs_abs)) == len(diffs_abs) - 1)))
        level_backgr = medians[index_level_backgr]

    thresh_intens = level_backgr + thresh_intens_fact * (max(medians) - level_backgr)
    return level_backgr, thresh_intens


def run_loggp_phase(true_model: "SquareLatticeDispersion",
                    n_measurements: int,
                    hmin: float,
                    hmax: float,
                    emin: float,
                    emax: float,
                    grid_h: int = 24,
                    grid_e: int = 18,
                    seed_measurements: Optional[List[Dict]] = None,
                    taper_mode: str = "energy") -> Tuple[List[Dict], int, Optional[int]]:
    """Agnostic Log-GP exploration to locate signal before TAS-AI refinement."""
    gp = LibraryBackedLogGPSurrogate(hmin=hmin, hmax=hmax, emin=emin, emax=emax, background=1.0)
    if gp.using_library:
        logger.info("Log-GP backend: tasai.core.LogGaussianProcess")
    else:
        logger.info("Log-GP backend fallback: local simple GP surrogate")
    measurements: List[Dict] = []
    consumed: List[Tuple[float, float, float, float]] = []
    level_backgr = None
    thresh_intens = None

    h_grid = np.linspace(hmin, hmax, grid_h)
    e_grid = np.linspace(emin, emax, grid_e)
    candidates = np.array([(h, e) for h in h_grid for e in e_grid])
    # Pre-filter candidates by kinematic accessibility to avoid scoring invalid points.
    valid_mask = np.array([TAS.is_accessible(h, h, E) for h, E in candidates])
    candidates = candidates[valid_mask]
    if len(candidates) == 0:
        logger.warning("Log-GP candidate set empty after accessibility filtering.")
        return measurements, 0, None

    if seed_measurements:
        measurements = [dict(m) for m in seed_measurements]
        n_init = min(n_measurements, len(measurements))
        for meas in measurements[:n_init]:
            h = float(meas["h"])
            E = float(meas["E"])
            sigma_E = TAS.get_energy_resolution(h, h, E)
            dh = LOGGP_ELLIPSE_H_WIDTH
            dE = max(0.5, sigma_E * LOGGP_ELLIPSE_E_SCALE)
            consumed.append((h, E, dh, dE))
    else:
        init_rows = max(3, int(LOGGP_INIT_GRID) | 1)
        h_init = np.linspace(hmin, hmax, init_rows)
        e_init = np.linspace(emin, emax, init_rows)
        init_points: List[Tuple[float, float]] = []
        for j in range(init_rows):
            irange = range(init_rows) if j % 2 == 0 else range(init_rows - 1, -1, -1)
            for i in irange:
                if i % 2 == j % 2:
                    init_points.append((float(h_init[i]), float(e_init[j])))

        n_init = min(n_measurements, len(init_points))
        for i in range(n_init):
            h, E = init_points[i]
            if not TAS.is_accessible(h, h, E):
                continue
            batch = simulate_measurements([{'h': float(h), 'k': float(h), 'E': float(E)}], true_model)
            if batch:
                meas = batch[0]
                meas['loggp_hint'] = True
                meas['loggp_init'] = True
                meas['mode'] = 'loggp_grid'
                measurements.append(meas)
                sigma_E = TAS.get_energy_resolution(h, h, E)
                dh = LOGGP_ELLIPSE_H_WIDTH
                dE = max(0.5, sigma_E * LOGGP_ELLIPSE_E_SCALE)
                consumed.append((h, E, dh, dE))

    if measurements:
        if len(measurements) >= 10:
            intens = np.array([m['intensity'] for m in measurements])
            level_backgr, thresh_intens = compute_heuris_experi_param(
                intens,
                LOGGP_LEVEL_BACKGR_DIFFS_REL_MAX,
                LOGGP_LEVEL_BACKGR_DIFFS_ABS_MIN,
                LOGGP_LEVEL_BACKGR_DECILE_MAX,
                LOGGP_THRESH_INTENS_FACT,
                level_backgr
            )
            if level_backgr is not None and thresh_intens is not None:
                logger.info("Log-GP background level: %.3f, threshold: %.3f", level_backgr, thresh_intens)
        for meas in measurements:
            if level_backgr is None:
                level_backgr = 0.0
            if thresh_intens is None:
                thresh_intens = meas['intensity']
        gp.refit(measurements, level_backgr, thresh_intens)

    def _acq_score(h: float, E: float) -> float:
        x = np.array([[h, E]], dtype=float)
        _, std = gp.predict_batch(x)
        linear_var = np.minimum(np.square(std), LOGGP_VAR_CLAMP * LOGGP_VAR_CLAMP)
        # Soft boundary penalty (cosine taper in energy) to reduce edge locking.
        window = float(_loggp_window_weight(
            np.array([h], dtype=float),
            np.array([E], dtype=float),
            hmin, hmax, emin, emax,
            taper_mode,
        )[0])
        last = measurements[-1]
        dh = abs(h - last["h"])
        dE = abs(E - last["E"])
        move_time = max(dh / LOGGP_MOVE_VH, dE / LOGGP_MOVE_VE) + LOGGP_MOVE_OVERHEAD
        return float(linear_var[0] * window) / (1.0 + move_time)

    def _maximize_acq_with_bumps(seed: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if os.environ.get("LOGGP_DISABLE_BUMPS"):
            return None
        try:
            from bumps.fitproblem import FitProblem
            from bumps.fitters import fit
            from bumps.curve import Curve
        except Exception:
            return None

        def acq_proxy(idx, h, E):
            score = _acq_score(h, E)
            return np.array([-np.tanh(score)])

        curve = Curve(acq_proxy, np.array([0.0]), np.array([-1.0]), np.array([0.1]),
                      h=seed[0], E=seed[1], name="loggp_acq")
        curve.h.range(hmin, hmax)
        curve.E.range(emin, emax)
        problem = FitProblem(curve)
        try:
            result = fit(problem, method='dream', burn=50, steps=100)
        except Exception:
            return None
        return float(result.x[0]), float(result.x[1])

    first_active_index: Optional[int] = None
    active_attempts = 0
    active_added = 0
    skipped_consumed = 0
    skipped_inaccessible = 0
    for _ in range(n_measurements - n_init):
        active_attempts += 1
        if thresh_intens is None and len(measurements) >= 10:
            intens = np.array([m['intensity'] for m in measurements])
            level_backgr, thresh_intens = compute_heuris_experi_param(
                intens,
                LOGGP_LEVEL_BACKGR_DIFFS_REL_MAX,
                LOGGP_LEVEL_BACKGR_DIFFS_ABS_MIN,
                LOGGP_LEVEL_BACKGR_DECILE_MAX,
                LOGGP_THRESH_INTENS_FACT,
                level_backgr
            )
            if level_backgr is not None and thresh_intens is not None:
                logger.info("Log-GP background level: %.3f, threshold: %.3f", level_backgr, thresh_intens)

        if len(measurements) == 0:
            idx = np.random.randint(0, len(candidates))
        else:
            if consumed:
                ch = np.array([c[0] for c in consumed])
                ce = np.array([c[1] for c in consumed])
                cdh = np.array([c[2] for c in consumed])
                cdE = np.array([c[3] for c in consumed])
                dh_all = (candidates[:, 0][:, None] - ch[None, :]) / cdh[None, :]
                de_all = (candidates[:, 1][:, None] - ce[None, :]) / cdE[None, :]
                inside = np.any(dh_all ** 2 + de_all ** 2 <= 1.0, axis=1)
                avail_mask = ~inside
            else:
                avail_mask = np.ones(len(candidates), dtype=bool)

            if not np.any(avail_mask):
                logger.warning("Log-GP active selection: all candidates consumed; stopping.")
                break

            candidates_avail = candidates[avail_mask]
            X = np.column_stack([candidates_avail[:, 0], candidates_avail[:, 1]])
            _, std = gp.predict_batch(X)
            linear_var = np.minimum(np.square(std), LOGGP_VAR_CLAMP * LOGGP_VAR_CLAMP)
            # Soft boundary penalty (cosine taper) in E only.
            window = _loggp_window_weight(
                candidates_avail[:, 0],
                candidates_avail[:, 1],
                hmin, hmax, emin, emax,
                taper_mode,
            )
            last = measurements[-1]
            dh = np.abs(candidates_avail[:, 0] - last["h"])
            dE = np.abs(candidates_avail[:, 1] - last["E"])
            move_time = np.maximum(dh / LOGGP_MOVE_VH, dE / LOGGP_MOVE_VE) + LOGGP_MOVE_OVERHEAD
            # Use a short ramp so the first active picks are not dominated by
            # the terminal grid point, then restore full motion awareness.
            if active_added <= 0:
                move_scale = 0.0
            elif active_added == 1:
                move_scale = 0.33
            elif active_added == 2:
                move_scale = 0.66
            else:
                move_scale = 1.0
            score = (linear_var * window) / (1.0 + move_scale * move_time)
            seed_idx = int(np.argmax(score))
            seed = (float(candidates_avail[seed_idx][0]), float(candidates_avail[seed_idx][1]))
            bumps_pick = _maximize_acq_with_bumps(seed)
            if bumps_pick is None:
                h, E = seed
            else:
                h, E = bumps_pick
            # Snap to the nearest available candidate to avoid immediately re-hitting consumed points.
            idx_avail = int(np.argmin((candidates_avail[:, 0] - h) ** 2 + (candidates_avail[:, 1] - E) ** 2))
            h, E = candidates_avail[idx_avail]
            idx = int(np.where(avail_mask)[0][idx_avail])

        h, E = candidates[idx]
        if not TAS.is_accessible(h, h, E):
            idx = np.random.randint(0, len(candidates))
            h, E = candidates[idx]
            skipped_inaccessible += 1

        if consumed:
            dh = np.array([(h - c[0]) / c[2] for c in consumed])
            de = np.array([(E - c[1]) / c[3] for c in consumed])
            if np.any(dh ** 2 + de ** 2 <= 1.0):
                skipped_consumed += 1
                continue

        batch = simulate_measurements([{'h': float(h), 'k': float(h), 'E': float(E)}], true_model)
        if batch:
            meas = batch[0]
            meas['loggp_hint'] = True
            meas['loggp_active'] = True
            meas['mode'] = 'loggp_active'
            measurements.append(meas)
            active_added += 1
            if first_active_index is None:
                first_active_index = len(measurements)
            if level_backgr is None:
                level_backgr = 0.0
            if thresh_intens is None:
                thresh_intens = meas['intensity']
            gp.refit(measurements, level_backgr, thresh_intens)

            sigma_E = TAS.get_energy_resolution(h, h, E)
            dh = LOGGP_ELLIPSE_H_WIDTH
            dE = max(0.5, sigma_E * LOGGP_ELLIPSE_E_SCALE)
            consumed.append((h, E, dh, dE))

    if active_attempts:
        logger.info(
            "Log-GP active summary: attempts=%d added=%d skipped_consumed=%d skipped_inaccessible=%d",
            active_attempts, active_added, skipped_consumed, skipped_inaccessible,
        )
    if active_attempts and active_added == 0:
        logger.warning("Log-GP active added zero points; candidate set may be fully consumed.")

    return measurements, n_init, first_active_index


def log_llm_suggestions(output_dir: Path,
                        batch_idx: int,
                        suggestions: List[Dict]) -> None:
    if not suggestions:
        return
    lines = ["idx,h,e,reason"]
    for idx, item in enumerate(suggestions, 1):
        reason = str(item.get("reason", "")).replace("\n", " ").strip()
        lines.append(f"{idx},{item.get('h'):.3f},{item.get('E'):.3f},{reason}")
    path = output_dir / f"llm_justifications_batch{batch_idx:03d}.csv"
    path.write_text("\n".join(lines))


def consensus_suggestions(all_suggestions: Dict[str, List[Dict]],
                          max_points: int) -> List[Dict]:
    """Simple consensus: pick most frequent points after rounding."""
    bucket = {}
    for model, suggestions in all_suggestions.items():
        for s in suggestions:
            key = (round(s["h"], 3), round(s["E"], 3))
            bucket.setdefault(key, []).append(s)
    ranked = sorted(bucket.items(), key=lambda kv: len(kv[1]), reverse=True)
    chosen = []
    for (h, e), group in ranked[:max_points]:
        merged = group[0].copy()
        merged["h"] = h
        merged["E"] = e
        merged["llm_votes"] = len(group)
        chosen.append(merged)
    return chosen


def maybe_get_llm_points(measurements: List[Dict],
                         output_dir: Path,
                         max_points: int,
                         history: int,
                         decider: str,
                         rotate_state: Dict[str, Any],
                         dry_run: bool,
                         external: bool = False,
                         mailbox_url: Optional[str] = None,
                         mailbox_token: Optional[str] = None,
                         mailbox_run_id: Optional[str] = None,
                         wait_seconds: int = 0,
                         batch_idx: int = 0) -> List[Dict]:
    """Query LLMs for suggestions and return consensus points."""
    timing_rows: List[str] = ["mode,model,seconds,status"]
    prompt = build_llm_prompt(measurements, max_points=history)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prompt_path = output_dir / f"prompt_{timestamp}_batch{batch_idx:03d}.txt"
    prompt_path.write_text(prompt)
    timing_path = output_dir / f"llm_timing_batch{batch_idx:03d}.csv"

    model_cmds = {
        "claude": os.environ.get("LLM_CLAUDE_CMD", "claude"),
        "gemini": os.environ.get("LLM_GEMINI_CMD", "gemini"),
        "codex": os.environ.get("LLM_CODEX_CMD", "codex"),
    }
    available = {k: v for k, v in model_cmds.items() if shutil.which(shlex.split(v)[0])}

    if external and mailbox_url and mailbox_token:
        # Mailbox mode: push prompt, then wait for suggestions flag (blocking).
        import urllib.request

        def _http_get(url: str) -> Optional[Dict]:
            req = urllib.request.Request(url, headers={"X-LLM-Token": mailbox_token})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                return None

        def _http_post(url: str, payload: Dict) -> bool:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "X-LLM-Token": mailbox_token},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    _ = resp.read().decode("utf-8")
                    return True
            except Exception:
                return False

        batch_key = f"{mailbox_run_id}_{batch_idx:03d}" if mailbox_run_id else f"{batch_idx:03d}"
        status = _http_get(f"{mailbox_url}/status/{batch_key}")
        if not status or not status.get("prompt_ready"):
            prompt_payload = {
                "prompt": prompt,
                "checkpoint": {
                    "batch": f"{batch_idx:03d}",
                    "n_measurements": len(measurements),
                    "measurements": measurements[-max(history, 1):],
                },
                "meta": {
                    "batch": f"{batch_idx:03d}",
                    "timestamp": timestamp,
                },
            }
            ok = _http_post(f"{mailbox_url}/prompt/{batch_key}", prompt_payload)
            logger.info("LLM mailbox batch %d: prompt posted=%s", batch_idx, ok)
        # Blocking wait for suggestions
        import time as _time
        waited = 0
        while True:
            status = _http_get(f"{mailbox_url}/status/{batch_key}")
            if status and status.get("suggestions_ready"):
                payload = _http_get(f"{mailbox_url}/suggestions/{batch_key}")
                if payload:
                    suggestions = payload.get("suggestions", [])
                    parsed = parse_llm_suggestions(json.dumps({"suggestions": suggestions}))
                    log_llm_suggestions(output_dir, batch_idx, parsed)
                    logger.info("LLM mailbox batch %d: injected %d suggestions",
                                batch_idx, len(parsed))
                    timing_rows.append(f"mailbox,ready,{waited},ok")
                    timing_path.write_text("\n".join(timing_rows))
                    return parsed
            if wait_seconds > 0 and waited >= wait_seconds:
                logger.info("LLM mailbox batch %d: timed out after %ds", batch_idx, waited)
                timing_rows.append(f"mailbox,timeout,{waited},empty")
                timing_path.write_text("\n".join(timing_rows))
                return []
            _time.sleep(10)
            waited += 10

    if external:
        suggestion_path = output_dir / f"suggestions_batch{batch_idx:03d}.json"
        if wait_seconds > 0:
            import time as _time
            waited = 0
            while waited < wait_seconds and not suggestion_path.exists():
                _time.sleep(2)
                waited += 2
            timing_rows.append(f"external,wait,{waited},ok")
            logger.info("LLM external wait batch %d: waited %ds (exists=%s)",
                        batch_idx, waited, suggestion_path.exists())
        if suggestion_path.exists():
            payload = json.loads(suggestion_path.read_text())
            suggestions = payload.get("suggestions", [])
            parsed = parse_llm_suggestions(json.dumps({"suggestions": suggestions}))
            log_llm_suggestions(output_dir, batch_idx, parsed)
            logger.info("LLM external batch %d: injected %d suggestions",
                        batch_idx, len(parsed))
            timing_path.write_text("\n".join(timing_rows))
            return parsed
        logger.info("LLM external mode: no suggestions found for batch %d", batch_idx)
        timing_rows.append("external,missing,0,empty")
        timing_path.write_text("\n".join(timing_rows))
        return []

    if dry_run or not available:
        logger.info("LLM dry run or no available CLIs; prompt written only.")
        timing_rows.append("local,skip,0,dry_run")
        timing_path.write_text("\n".join(timing_rows))
        return []

    responses = {}
    suggestions = {}
    for name, cmd in available.items():
        try:
            start = time.time()
            out = _run_llm_command(cmd, prompt)
            elapsed = time.time() - start
            responses[name] = out
            suggestions[name] = parse_llm_suggestions(out)
            (output_dir / f"response_{name}_{timestamp}.txt").write_text(out)
            timing_rows.append(f"local,{name},{elapsed:.2f},ok")
        except Exception as exc:
            logger.warning("LLM %s failed: %s", name, exc)
            timing_rows.append(f"local,{name},0,fail")

    if not suggestions:
        timing_path.write_text("\n".join(timing_rows))
        return []

    if decider == "rotate":
        rotate_list = rotate_state.setdefault("order", list(available.keys()))
        rotate_idx = rotate_state.setdefault("idx", 0)
        decider_name = rotate_list[rotate_idx % len(rotate_list)]
        rotate_state["idx"] = rotate_idx + 1
    else:
        decider_name = decider

    if decider_name in available and len(suggestions) > 1:
        consensus_prompt = {
            "task": "Select up to N points that best distinguish gapped vs gapless behavior.",
            "max_points": max_points,
            "candidates": suggestions,
        }
        decider_text = json.dumps(consensus_prompt, indent=2)
        try:
            start = time.time()
            out = _run_llm_command(available[decider_name], decider_text)
            elapsed = time.time() - start
            responses[f"decider_{decider_name}"] = out
            (output_dir / f"decider_{decider_name}_{timestamp}.txt").write_text(out)
            final = parse_llm_suggestions(out)
            if final:
                log_llm_suggestions(output_dir, batch_idx, final)
                timing_rows.append(f"local,decider_{decider_name},{elapsed:.2f},ok")
                timing_path.write_text("\n".join(timing_rows))
                return final[:max_points]
        except Exception as exc:
            logger.warning("LLM decider %s failed: %s", decider_name, exc)
            timing_rows.append(f"local,decider_{decider_name},0,fail")

    final = consensus_suggestions(suggestions, max_points=max_points)
    log_llm_suggestions(output_dir, batch_idx, final)
    timing_path.write_text("\n".join(timing_rows))
    return final


def _checkpoint_payload(measurements: List[Dict],
                        measurement_plan: List[Dict],
                        planned_points: List[Dict],
                        llm_state: Dict[str, Any],
                        llm_batch_idx: int,
                        note: str) -> Dict[str, Any]:
    return {
        "note": note,
        "timestamp": datetime.utcnow().isoformat(),
        "measurements": measurements,
        "measurement_plan": measurement_plan,
        "planned_points": planned_points,
        "llm_state": llm_state,
        "llm_batch_idx": llm_batch_idx,
    }


def save_checkpoint(checkpoint_dir: Path,
                    measurements: List[Dict],
                    measurement_plan: List[Dict],
                    planned_points: List[Dict],
                    llm_state: Dict[str, Any],
                    llm_batch_idx: int,
                    note: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        measurements=measurements,
        measurement_plan=measurement_plan,
        planned_points=planned_points,
        llm_state=llm_state,
        llm_batch_idx=llm_batch_idx,
        note=note,
    )
    path = checkpoint_dir / "closed_loop_checkpoint.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_checkpoint(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "measurements": payload.get("measurements", []),
        "measurement_plan": payload.get("measurement_plan", []),
        "planned_points": payload.get("planned_points", []),
        "llm_state": payload.get("llm_state", {}),
        "llm_batch_idx": payload.get("llm_batch_idx", 0),
        "note": payload.get("note", ""),
    }


# =============================================================================
# TAS Instrument Configuration
# =============================================================================

# Physical constants for TAS kinematics
CONVERT2 = 2.072  # meV to k^2 conversion: E = CONVERT2 * k^2


class TASConfig:
    """
    Triple-axis spectrometer configuration with kinematic checks.

    Default: 40' collimations, fixed Ef=14.7 meV, Cooper-Nathans.
    """

    def __init__(self, lattice_a: float = 4.0, efixed: float = 14.7,
                 hcol: Tuple[float, ...] = (40, 40, 40, 40)):
        self.lattice_a = lattice_a  # Angstroms
        self.efixed = efixed  # meV
        self.hcol = hcol  # arc-minutes

        # Wave vector at fixed energy
        self.kf = np.sqrt(efixed / CONVERT2)

        # Resolution calculator (if available)
        self.res_calc = None
        if HAS_RESOLUTION:
            self.res_calc = TASResolutionCalculator(
                lattice_params=(lattice_a, lattice_a, 10.0, 90, 90, 90),
                orient1=[1, 0, 0],
                orient2=[0, 1, 0],
                exp_config=create_default_tas_config(
                    efixed=efixed,
                    hcol=hcol
                )
            )

    def h_to_Q(self, h: float, k: float = 0) -> float:
        """Convert (h, k, 0) to Q in inverse Angstroms."""
        return 2 * np.pi / self.lattice_a * np.sqrt(h**2 + k**2)

    def is_accessible(self, h: float, k: float, E: float) -> bool:
        """
        Check if (h, k, E) point is kinematically accessible.

        For fixed Ef mode:
        - Ei = Ef + E
        - Q must satisfy: |ki - kf| <= Q <= ki + kf

        Note: For this toy demonstration, we use relaxed constraints
        to show the full dispersion. Real TAS experiments would have
        stricter kinematic limits based on Ef and Q range.
        """
        # Relaxed constraints for toy demo
        # Real TAS would enforce: Q_min <= Q <= Q_max
        return 0 < E < 50  # Simple energy range

    def get_energy_resolution(self, h: float, k: float, E: float) -> float:
        """
        Get energy resolution (sigma, not FWHM) at (h, k, E).

        Uses Cooper-Nathans calculation for realistic Q-dependent resolution.
        With 80'-40'-40'-80' collimations and Ef=14.7 meV, typical FWHM ~0.5-1.0 meV.
        """
        if not self.is_accessible(h, k, E):
            return np.inf

        if self.res_calc is not None:
            try:
                fwhm, _ = self.res_calc.get_resolution_fwhm(h=h, k=k, l=0.0, E=E)
                if 0 < fwhm['E'] < 10:  # Valid result
                    return fwhm['E'] / 2.355  # Convert FWHM to sigma
            except (ValueError, RuntimeWarning):
                pass

        # Fallback: moderate TAS resolution
        return 0.30  # meV (FWHM ~0.70 meV)


def init_tas(enable_resolution: bool):
    """
    Initialize the global TAS configuration, optionally disabling
    the Cooper-Nathans resolution calculator for reproducibility.
    """
    global HAS_RESOLUTION, TAS
    HAS_RESOLUTION = enable_resolution and BASE_HAS_RESOLUTION
    TAS = TASConfig(lattice_a=4.0, efixed=14.7, hcol=(80, 40, 40, 80))


# Global TAS configuration
init_tas(BASE_HAS_RESOLUTION)


# =============================================================================
# Toy Physical System: 2D Square Lattice Antiferromagnet
# =============================================================================

class SquareLatticeDispersion:
    """
    Spin wave dispersion for 2D square-lattice AFM in LSWT.

    H = J₁ Σ_<ij> S_i·S_j + J₂ Σ_<<ik>> S_i·S_k + D Σ_i (S_i^z)²

    We plot in absolute Q along (H,H,0), but the analytic dispersion is
    defined in reduced wavevector q = Q - Q_AF, with AFM ordering
    Q_AF = (1.5, 1.5, 0) r.l.u. (equivalently (0.5, 0.5, 0) by reciprocity).

    For the AFM ground state, the dispersion is:
    ω(q) = 2S√[(J₁γ_1 + J₂γ_2 + D)² - (J₁γ_1)²]

    where γ_1 = (cos(π qh) + cos(π qk))/2, γ_2 = (cos(2π qh) + cos(2π qk))/2.
    """

    def __init__(self, J1: float = 5.0, J2: float = 0.0, D: float = 0.0,
                 S: float = 2.5, background: float = 0.5,
                 ordering_vector: Tuple[float, float] = (0.5, 0.5)):
        self.J1 = J1
        self.J2 = J2
        self.D = D
        self.S = S
        self.background = background
        self.ordering_vector = ordering_vector
        self.name = "SquareLattice"
        self.a = 4.0

    @staticmethod
    def _maybe_scalar(arr: np.ndarray):
        """Return a Python scalar for 0-d/size-1 arrays, else the array."""
        return arr.item() if np.ndim(arr) == 0 or arr.size == 1 else arr

    def _afm_kernel(self, h, k):
        """
        Vectorized Néel-phase LSWT kernel for the square-lattice J1-J2 AFM.

        Uses reduced momentum relative to the AF ordering vector. This is safe
        for the dispersion and for the one-magnon weight so long as the same
        reduced-q convention is used consistently in both places.
        """
        H, K = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(k, dtype=float))

        # This branch is only valid in the Néel phase.
        if self.J1 <= 0.0 or self.J2 >= 0.5 * self.J1:
            nan = np.full(H.shape, np.nan, dtype=float)
            return H, K, nan, nan, nan

        if self.S <= 0.5:
            d_eff = 0.0
        else:
            # Preserve the easy-axis convention H_aniso = -D sum_i (Sz_i)^2.
            d_eff = self.D * (1.0 - 1.0 / (2.0 * self.S))

        qh = H - self.ordering_vector[0]
        qk = K - self.ordering_vector[1]
        qx = 2.0 * np.pi * qh
        qy = 2.0 * np.pi * qk

        gamma1 = 0.5 * (np.cos(qx) + np.cos(qy))
        gamma2 = np.cos(qx) * np.cos(qy)

        A = (
            4.0 * self.S * self.J1
            - 4.0 * self.S * self.J2 * (1.0 - gamma2)
            + 2.0 * self.S * d_eff
        )
        B = 4.0 * self.S * self.J1 * gamma1

        arg = A * A - B * B
        tol = 1e-12 * np.maximum(1.0, np.maximum(np.abs(A * A), np.abs(B * B)))

        omega2 = np.full_like(arg, np.nan, dtype=float)
        valid = arg >= -tol
        omega2[valid] = np.maximum(arg[valid], 0.0)
        omega = np.sqrt(omega2)
        return H, K, A, B, omega

    def omega(self, h, k):
        """Spin wave energy at (h, k) in r.l.u. (input is absolute Q)."""
        _, _, _, _, omega = self._afm_kernel(h, k)
        return self._maybe_scalar(omega)

    def _form_factor_fe3(self, q_mag):
        """Fe3+ magnetic form factor (j0) using International Tables coefficients."""
        s = q_mag / (4 * np.pi)
        s2 = s ** 2
        A, a = 0.396, 13.244
        B, b = 0.629, 4.903
        C, c = -0.0314, 0.35
        D_const = 0.0044
        return A * np.exp(-a * s2) + B * np.exp(-b * s2) + C * np.exp(-c * s2) + D_const

    def intensity(self, h, k, E,
                  sigma_E: Optional[float] = None, I0: float = 100.0,
                  use_realistic_resolution: bool = True):
        """
        Predicted intensity at (h, k, E).

        Parameters
        ----------
        h, k : float
            Position in reciprocal lattice units
        E : float
            Energy transfer in meV
        sigma_E : float, optional
            Energy resolution sigma. If None and use_realistic_resolution=True,
            uses Cooper-Nathans calculation from TAS config.
        I0 : float
            Peak intensity scale
        use_realistic_resolution : bool
            If True, use TAS resolution calculator

        Returns
        -------
        float
            Intensity (Gaussian peak + background)
        """
        H, K, EE = np.broadcast_arrays(
            np.asarray(h, dtype=float),
            np.asarray(k, dtype=float),
            np.asarray(E, dtype=float),
        )

        out = np.full(EE.shape, float(self.background), dtype=float)
        _, _, A_q, B_q, omega = self._afm_kernel(H, K)

        if sigma_E is None:
            if use_realistic_resolution:
                sigma_E = np.vectorize(TAS.get_energy_resolution, otypes=[float])(H, K, EE)
            else:
                sigma_E = np.full(EE.shape, 0.5, dtype=float)
        else:
            sigma_E = np.broadcast_to(np.asarray(sigma_E, dtype=float), EE.shape).copy()

        accessible = np.vectorize(TAS.is_accessible, otypes=[bool])(H, K, EE)
        if np.any(accessible):
            # Reduced-q convention: one-magnon transverse weight is (A+B)/omega.
            with np.errstate(divide="ignore", invalid="ignore"):
                sw_weight = (A_q + B_q) / omega

            q_mag = (2.0 * np.pi / self.a) * np.sqrt(H * H + K * K)
            form_factor = self._form_factor_fe3(q_mag)

            with np.errstate(divide="ignore", invalid="ignore"):
                x = (EE - omega) / sigma_E
                profile = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma_E)

            bad = accessible & (
                ~np.isfinite(omega) |
                ~np.isfinite(sw_weight) |
                ~np.isfinite(profile) |
                (sigma_E <= 0.0)
            )
            out[bad] = np.nan

            good = accessible & ~bad
            out[good] += I0 * (form_factor[good] ** 2) * sw_weight[good] * profile[good]

        return self._maybe_scalar(out)


# =============================================================================
# Crystal Structure Definition
# =============================================================================

def create_toy_structure() -> Dict:
    """
    Create a toy 2D square lattice structure.
    
    This represents a simplified Fe-O perovskite layer.
    """
    a = 4.0  # Lattice parameter in Angstroms
    
    return {
        'name': 'ToySquareLattice',
        'lattice': np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, 10.0]  # Large c for 2D behavior
        ]),
        'species': ['Fe', 'O', 'O'],
        'coords': np.array([
            [0.0, 0.0, 0.0],   # Fe at origin
            [0.5, 0.0, 0.0],   # O bridging along a
            [0.0, 0.5, 0.0],   # O bridging along b
        ]),
        'description': '2D square lattice with Fe-O-Fe superexchange'
    }


# =============================================================================
# Hypothesis Generation (Heuristic - would use GNN in production)
# =============================================================================

def generate_hypotheses(structure: Dict) -> List[Dict]:
    """
    Generate candidate Hamiltonians from structure.
    
    In production, this would use CHGNet/M3GNet. Here we use
    physics-based heuristics that mimic what a GNN would learn.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Hypothesis Generation")
    logger.info("="*60)
    
    # Analyze structure
    lattice = structure['lattice']
    a = lattice[0, 0]
    
    # Fe-Fe distance along edges
    d_nn = a
    
    # Fe-Fe distance along diagonal
    d_nnn = a * np.sqrt(2)
    
    logger.info(f"Structure: {structure['name']}")
    logger.info(f"Lattice parameter: a = {a:.2f} Å")
    logger.info(f"NN Fe-Fe distance: {d_nn:.2f} Å")
    logger.info(f"NNN Fe-Fe distance: {d_nnn:.2f} Å")
    
    # Heuristic J estimates based on distance
    # These rules approximate what a GNN would learn from DFT data
    
    # J1: NN exchange via 180° Fe-O-Fe superexchange → AFM
    # Scaled model for this demo
    J1_est = 1.25  # meV (GK estimate - true value is 1.25)
    J1_err = 0.5  # Uncertainty from GK analysis

    # J2: NNN exchange, typically weaker, often FM for frustration relief
    # GK predicts weak NNN coupling - estimate within factor of 2
    J2_est = 0.2  # meV (GK estimate - true value is 0.2)
    J2_err = 0.2

    # D: Single-ion anisotropy, small for Fe³⁺
    # GK provides order-of-magnitude estimate
    D_est = 0.02  # meV (GK estimate - true value is 0.02)
    D_err = 0.02
    
    # Four candidate models matching paper Section 5.3
    candidates = [
        {
            'name': 'M1: NN Heisenberg',
            'description': 'J₁ only, J₂ = D = 0',
            'params': {'J1': J1_est, 'J2': 0.0, 'D': 0.0, 'background': 0.5},
            'uncertainties': {'J1': J1_err, 'J2': 0.0, 'D': 0.0, 'background': 0.0},
            'prior': 0.10
        },
        {
            'name': 'M2: NN + anisotropy',
            'description': 'J₁ + D, J₂ = 0',
            'params': {'J1': J1_est, 'J2': 0.0, 'D': D_est, 'background': 0.5},
            'uncertainties': {'J1': J1_err, 'J2': 0.0, 'D': D_err, 'background': 0.0},
            'prior': 0.10
        },
        {
            'name': 'M3: J1-J2 no gap',
            'description': 'J₁ + J₂, D = 0',
            'params': {'J1': J1_est, 'J2': J2_est, 'D': 0.0, 'background': 0.5},
            'uncertainties': {'J1': J1_err, 'J2': J2_err, 'D': 0.0, 'background': 0.0},
            'prior': 0.10
        },
        {
            'name': 'M4: Full model',
            'description': 'J₁ + J₂ + D (all terms)',
            'params': {'J1': J1_est, 'J2': J2_est, 'D': D_est, 'background': 0.5},
            'uncertainties': {'J1': J1_err, 'J2': J2_err, 'D': D_err, 'background': 0.0},
            'prior': 0.70
        }
    ]
    
    logger.info("\nProposed candidate Hamiltonians:")
    for cand in candidates:
        logger.info(f"\n  {cand['name']} (prior = {cand['prior']:.0%})")
        logger.info(f"    {cand['description']}")
        for p, v in cand['params'].items():
            if v != 0:
                err = cand['uncertainties'][p]
                logger.info(f"    {p} = {v:.2f} ± {err:.2f} meV")
    
    return candidates


# =============================================================================
# Measurement Planning (Simplified MCTS)
# =============================================================================

def create_zone_coverage_points(true_model: SquareLatticeDispersion) -> List[Dict]:
    """
    Seed measurements at the gap edge (Van Hove) and include a single
    boundary anchor so the planner always has decisive contrast data.
    """
    coverage: List[Dict] = []
    # Target magnetic zone center at H≈0.5 (AFM ordering vector)
    center_h = 0.5
    gap_positions = (center_h, center_h + 0.05, center_h + 0.10)
    gap_energies = (0.60, 0.70, 0.80, 0.90)
    for h in gap_positions:
        omega = true_model.omega(h, h)
        for E in gap_energies:
            if not TAS.is_accessible(h, h, E):
                continue
            sigma_E = TAS.get_energy_resolution(h, h, E)
            coverage.append({
                'h': h,
                'k': h,
                'E': E,
                'sigma_E': sigma_E,
                'omega_pred': omega,
                'coverage': True
            })
    boundary_h = min(1.0, H_RANGE_MAX)
    boundary_E = true_model.omega(boundary_h, boundary_h)
    if TAS.is_accessible(boundary_h, boundary_h, boundary_E):
        sigma_E = TAS.get_energy_resolution(boundary_h, boundary_h, boundary_E)
        coverage.append({
            'h': boundary_h,
            'k': boundary_h,
            'E': boundary_E,
            'sigma_E': sigma_E,
            'omega_pred': boundary_E,
            'coverage': True
        })
    return coverage


def create_human_gap_hint_points(candidates: List[Dict],
                                 hint_positions: Tuple[float, ...] = (H_RANGE_MIN, H_RANGE_MIN + 0.15, H_RANGE_MIN + 0.30)
                                 ) -> List[Dict]:
    """
    Mimic a human suggesting gap-sensitive measurements near the zone center.

    We use the most complex candidate (typically J1-J2-D) as the reference model.
    """
    if not candidates:
        return []

    # Use the richest candidate for the gap estimate (falls back to first)
    reference = candidates[-1]
    model = SquareLatticeDispersion(**reference['params'])

    hints: List[Dict] = []
    energy_grid = BAND_EDGE_SCAN_RANGE
    for h in hint_positions:
        omega = model.omega(h, h)
        for E in energy_grid:
            if not TAS.is_accessible(h, h, E):
                continue
            sigma_E = TAS.get_energy_resolution(h, h, E)
            hints.append({
                'h': h,
                'k': h,
                'E': E,
                'sigma_E': sigma_E,
                'omega_pred': omega,
                'human_hint': True
            })
    return hints


# Updated for diagonal scan (H, H, 0) with AFM ordering at Q_AF=(0.5,0.5)
# Gamma (silent): 0.0, 1.0, 2.0; M (bright): 0.5, 1.5
HIGH_SYMMETRY_POINTS = [
    ('M1', 0.5),     # Magnetic zone center (bright)
    ('Gamma', 1.0),  # Nuclear zone center (silent)
    ('M2', 1.5),     # Magnetic zone center (weaker form factor)
]


def create_symmetry_seed_points(candidates: List[Dict],
                                total_budget: int,
                                fraction: float = 0.1,
                                min_points: int = 0) -> List[Dict]:
    """
    Reserve a fraction of the measurement budget for Γ/X/M points.
    """
    if total_budget <= 0:
        return []

    n_seed = max(min_points, int(np.ceil(total_budget * fraction)))
    if n_seed <= 0:
        return []

    reference = candidates[-1] if candidates else None
    model = SquareLatticeDispersion(**reference['params']) if reference else None

    seeds: List[Dict] = []
    cycle_points = itertools.cycle(HIGH_SYMMETRY_POINTS)

    while len(seeds) < n_seed:
        label, h_val = next(cycle_points)
        omega = model.omega(h_val, h_val) if model else 5.0
        E = max(0.5, omega)
        if not TAS.is_accessible(h_val, h_val, E):
            continue
        sigma_E = TAS.get_energy_resolution(h_val, h_val, E)
        seeds.append({
            'h': h_val,
            'k': h_val,
            'E': E,
            'sigma_E': sigma_E,
            'omega_pred': omega,
            'symmetry': True,
            'label': label
        })

    return seeds


def intensity_to_probability(intensity: float,
                             background: float = 0.5,
                             eps: float = 1e-9) -> np.ndarray:
    """Convert an expected intensity to a 2-state probability vector."""
    signal = max(intensity, 0.0)
    denom = signal + max(background, eps) + eps
    p_signal = signal / denom
    return np.array([p_signal, 1.0 - p_signal])


def jsd_between_intensities(intensities: List[float]) -> float:
    """
    Compute mean JSD between all model intensity predictions at a point.
    """
    if len(intensities) < 2:
        return 0.0

    jsd_vals = []
    for i, j in itertools.combinations(range(len(intensities)), 2):
        p = intensity_to_probability(intensities[i])
        q = intensity_to_probability(intensities[j])
        m = 0.5 * (p + q)
        kl_p = np.sum(np.where(p > 0, p * np.log(p / m), 0.0))
        kl_q = np.sum(np.where(q > 0, q * np.log(q / m), 0.0))
        jsd_vals.append(0.5 * (kl_p + kl_q))
    return float(np.mean(jsd_vals))


def repulsion_factor(h: float, E: float,
                     history: List[Dict],
                     sigma: float = 0.05) -> float:
    """Penalize sampling near existing points to promote spatial diversity."""
    if not history:
        return 1.0
    factor = 1.0
    sigma_sq = sigma**2
    for pt in history:
        dh = h - pt['h']
        dE = (E - pt['E']) / 15.0  # scale energy to comparable units
        dist_sq = dh*dh + dE*dE
        penalty = np.exp(-dist_sq / (2 * sigma_sq))
        factor *= max(0.02, 1 - 0.97 * penalty)
    return factor


def estimate_fisher_information(model: Optional['SquareLatticeDispersion'],
                                h: float, k: float, E: float,
                                delta: float = 0.05,
                                mask_j1: bool = False,
                                normalize: bool = True,
                                return_base: bool = False) -> Any:
    """Crude Fisher information proxy with optional gradient masking."""
    if model is None:
        return 0.0

    params = ['J1', 'J2', 'D']
    base_intensity = model.intensity(h, k, E)
    info = 0.0

    for idx, param in enumerate(params):
        original = getattr(model, param)
        setattr(model, param, original + delta)
        plus = model.intensity(h, k, E)
        setattr(model, param, original - delta)
        minus = model.intensity(h, k, E)
        setattr(model, param, original)
        gradient = (plus - minus) / (2 * delta)
        if mask_j1 and param == 'J1':
            gradient = 0.0
        info += gradient**2

    if normalize:
        result = info / max(1.0, base_intensity)
    else:
        result = info

    if return_base:
        return result, base_intensity
    return result


def determine_planning_phase(meas_count: int,
                             posterior_hint: Optional[List[float]],
                             thresholds: Tuple[int, int],
                             posterior_threshold: float) -> int:
    """Return active phase (1, 2, or 3) based on measurement count/posterior."""
    phase1_limit, phase2_limit = thresholds
    if meas_count < phase1_limit:
        return 1
    if posterior_hint and max(posterior_hint) >= posterior_threshold:
        return 3
    if meas_count >= phase2_limit:
        return 3
    return 2


def plan_measurements(candidates: List[Dict],
                      n_points: int = 12,
                      force_zone_coverage: bool = False,
                      adaptive_coverage: bool = False,
                      existing_points: Optional[List[Dict]] = None,
                      measurement_history: Optional[List[Dict]] = None,
                      use_jsd: bool = False,
                      phase_thresholds: Tuple[int, int] = (5, 30),
                      posterior_hint: Optional[List[float]] = None,
                      precision_model: Optional['SquareLatticeDispersion'] = None,
                      enable_phases: bool = False,
                      posterior_phase3_threshold: float = 0.95,
                      force_phase: Optional[int] = None,
                      use_projected_fisher: bool = False,
                      dwell_multiplier: float = 1.0,
                      hmin: Optional[float] = None,
                      hmax: Optional[float] = None) -> List[Dict]:
    """
    Plan measurements using information-theoretic criteria.

    For this toy demo, we use a simplified strategy:
    - Sample along high-symmetry directions
    - For each Q point, sample energies near predicted dispersions
    - Include wider energy range to account for GK estimate uncertainty (~20%)
    - Focus on energies where models disagree
    - Only consider kinematically accessible (Q, E) points
    - Use recent measurement history to decide when to escalate to gap hunting

    Note: This assumes GK estimates are within ~20% of true values. For
    cases where GK estimates may be very wrong, the hybrid exploration
    approach (agnostic GP phase first) is more robust. See
    hybrid_exploration_demo.py for that workflow.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Measurement Planning")
    logger.info("="*60)

    # Log TAS configuration
    logger.info(f"\nTAS Configuration:")
    logger.info(f"  Fixed Ef: {TAS.efixed} meV")
    logger.info(f"  Collimations: {TAS.hcol} arcmin")
    logger.info(f"  kf = {TAS.kf:.3f} Å⁻¹")

    hmin = H_RANGE_MIN if hmin is None else hmin
    hmax = H_RANGE_MAX if hmax is None else hmax

    # Create models from candidates
    models = [
        SquareLatticeDispersion(**cand['params'])
        for cand in candidates
    ]

    existing_points = existing_points or []
    measurement_history = measurement_history or []
    gap_energy_threshold = 3.5
    super_dwell_time = 60.0 * max(2.0, dwell_multiplier)
    has_deep_gap_data = any(
        (classify_branch(meas['h']) == 'gap' and meas.get('E', 99.0) <= gap_energy_threshold and
         meas.get('count_time', 60.0) >= 0.9 * super_dwell_time)
        for meas in measurement_history
    )

    forced_phase3_logged = False

    def resolve_phase(meas_so_far: int) -> int:
        nonlocal forced_phase3_logged
        if force_phase is not None:
            return force_phase
        if not enable_phases:
            return 2
        phase_candidate = determine_planning_phase(
            meas_so_far, posterior_hint, phase_thresholds, posterior_phase3_threshold
        )
        if (phase_candidate == 2 and measurement_history and
                should_force_phase3_transition(measurement_history, posterior_hint)):
            if not forced_phase3_logged:
                logger.info("  [Planner] Early Phase-3 trigger: J1 settled, targeting gap.")
                forced_phase3_logged = True
            return 3
        return phase_candidate

    initial_phase = resolve_phase(len(existing_points))
    phase3_gap_mode = (enable_phases and use_projected_fisher and initial_phase == 3)
    need_super_dwell = (
        phase3_gap_mode and dwell_multiplier > 1.01 and not has_deep_gap_data
    )

    # Candidate measurement points along (h, h) direction
    if force_zone_coverage:
        # Include explicit zone-center sampling when coverage is requested
        h_values = np.concatenate([
            np.linspace(hmin, min(hmin + 0.2, hmax), 6, endpoint=True),
            np.linspace(min(hmin + 0.3, hmax), hmax, 20, endpoint=True)
        ])
    else:
        # Default: focus on zone boundary unless Phase 3 needs Γ coverage
        if phase3_gap_mode:
            logger.debug("Phase-3 planning (projected Fisher) using full H grid")
            h_values = np.linspace(hmin, hmax, 25)
        else:
            h_values = np.linspace(hmin, hmax, 25)

    # Track coverage density along H so we can adaptively encourage rarer regions
    coverage_bins = np.linspace(hmin, hmax, 8)
    coverage_counts = np.zeros(len(coverage_bins) + 1, dtype=float)
    if adaptive_coverage:
        for pt in existing_points:
            idx = np.digitize(pt['h'], coverage_bins)
            coverage_counts[idx] += 1

    def is_too_close(pt: Dict, others: List[Dict], h_thresh: float = 0.06,
                     e_thresh: float = 1.5) -> bool:
        for other in others:
            if abs(pt['h'] - other['h']) < h_thresh and abs(pt['E'] - other['E']) < e_thresh:
                return True
        return False

    # Score points by model disagreement, filtering for accessibility
    point_scores = []
    n_accessible = 0
    n_total = 0

    for h in h_values:
        # Get predicted dispersion energies from all models
        omega_predictions = [m.omega(h, h) for m in models]
        omega_min = min(omega_predictions)
        omega_max = max(omega_predictions)
        omega_avg = np.mean(omega_predictions)

        # Sample energies around the predicted dispersions
        # Include ±20% margin to account for GK estimate uncertainty
        # This ensures we measure near where signal is expected even if GK is off
        if omega_avg > 0:
            if phase3_gap_mode:
                E_samples = np.linspace(
                    max(0.35, omega_min - 2.0),
                    min(20.0, omega_max + 2.0),
                    8
                )
            else:
                margin = max(3.0, 0.2 * omega_avg)  # 20% or at least 3 meV
                E_samples = np.linspace(
                    max(0.5, omega_min - margin),
                    omega_max + margin,
                    10
                )
        else:
            # Sample energy range appropriate for this Q (dispersion is 0-44 meV)
            if phase3_gap_mode:
                E_samples = np.linspace(0.5, 20.0, 8)
            else:
                # Avoid elastic-line probing at high energy when omega≈0.
                E_samples = np.linspace(0.5, min(LOGGP_E_MAX, 5.0), 6)

        for E in E_samples:
            n_total += 1

            # Guard against omega≈0 selections at high energy (elastic-line territory).
            if omega_avg < OMEGA_GUARD_MIN and E > (OMEGA_GUARD_MIN + OMEGA_GUARD_MARGIN):
                continue

            # Check kinematic accessibility
            if not TAS.is_accessible(h, h, E):
                continue

            n_accessible += 1

            # Get predictions with realistic resolution
            sigma_E = TAS.get_energy_resolution(h, h, E)
            predictions = [m.intensity(h, h, E, sigma_E=sigma_E) for m in models]

            variance = np.var(predictions)
            jsd_value = jsd_between_intensities(predictions)
            fisher_score = 0.0
            fisher_proj = 0.0
            precision_intensity = 0.0
            if precision_model is not None:
                raw_proj = (enable_phases and force_phase == 3 and use_projected_fisher)
                fisher_score, base_intensity = estimate_fisher_information(
                    precision_model, h, h, E, mask_j1=False,
                    normalize=not raw_proj, return_base=True
                )
                fisher_proj, _ = estimate_fisher_information(
                    precision_model, h, h, E, mask_j1=True,
                    normalize=not raw_proj, return_base=True
                )
                precision_intensity = base_intensity

            # Bonus score for being near a dispersion (will have signal)
            dist_to_disp = min([abs(E - omega) for omega in omega_predictions])
            on_resonance_bonus = np.exp(-dist_to_disp**2 / (2 * 1.0**2))
            if enable_phases and force_phase == 3 and use_projected_fisher:
                on_resonance_bonus = 0.0

            zone_bonus = 1.0
            if force_zone_coverage:
                zone_bonus += 2.5 * np.exp(-((h - 0.1) / 0.08)**2)

            point_scores.append({
                'h': h, 'k': h, 'E': E,
                'variance': variance,
                'jsd': jsd_value,
                'fisher': fisher_score,
                'proj_fisher': fisher_proj,
                'on_res_bonus': (1 + on_resonance_bonus),
                'zone_bonus': zone_bonus,
                'sigma_E': sigma_E,
                'omega_pred': omega_avg,
                'precision_intensity': precision_intensity,
                'count_time': 60.0,
                'dwell': False
            })

    logger.info(f"\nKinematic accessibility: {n_accessible}/{n_total} points "
                f"({100*n_accessible/n_total:.0f}%)")

    initial_phase = resolve_phase(len(existing_points))
    phase3_gap_mode = (enable_phases and use_projected_fisher and initial_phase == 3)
    need_super_dwell = (
        phase3_gap_mode and dwell_multiplier > 1.01 and not has_deep_gap_data
    )
    if need_super_dwell:
        gap_candidates = []
        for pt in point_scores:
            if pt['E'] > gap_energy_threshold:
                continue
            if classify_branch(pt['h']) != 'gap':
                continue
            intensity = max(pt.get('precision_intensity', 0.0), 1e-3)
            gap_candidates.append((pt['proj_fisher'] / intensity, pt))
        if gap_candidates:
            gap_candidates.sort(key=lambda x: -x[0])
            _, best_gap = gap_candidates[0]
            best_gap = best_gap.copy()
            best_gap['dwell'] = True
            best_gap['count_time'] = best_gap.get('count_time', 60.0) * dwell_multiplier
            logger.info("  [Planner] Phase-3 super-dwell injection at h=%.3f, E=%.2f meV "
                        "for %.0fs", best_gap['h'], best_gap['E'], best_gap['count_time'])
            return [best_gap]
        else:
            logger.warning("  [Planner] Super-dwell requested but no qualifying gap candidates.")

    # Add exploratory measurements at key h values with wide energy range
    # This makes MCTS robust to GK estimate errors (similar to hybrid exploration)
    exploratory = []
    if not phase3_gap_mode:
        # Keep exploratory samples within the analysis energy range.
        e_high = min(50.0, LOGGP_E_MAX)
        e_low = 35.0 if LOGGP_E_MAX >= 35.0 else max(0.5, LOGGP_E_MAX - 5.0)
        for h in [0.75, 1.0, 1.25]:  # Key h values
            omega_avg = np.mean([m.omega(h, h) for m in models])
            if omega_avg < OMEGA_GUARD_MIN:
                continue
            for E in np.linspace(e_low, e_high, 6):  # Wide energy range to catch dispersion
                if TAS.is_accessible(h, h, E):
                    sigma_E = TAS.get_energy_resolution(h, h, E)
                    exploratory.append({
                        'h': h, 'k': h, 'E': E,
                        'score': 0,  # Exploratory - no model-based score
                        'sigma_E': sigma_E,
                        'omega_pred': 0,
                        'exploratory': True,
                        'count_time': 60.0,
                        'dwell': False
                    })

    # Take top model-disagreement points, ensuring diversity
    selected: List[Dict] = []
    available = point_scores.copy()

    # Cap exploratory points so they don't consume the entire batch.
    n_exploratory_slots = 0
    if exploratory:
        n_exploratory_slots = max(1, n_points // 4)
        if len(exploratory) > n_exploratory_slots:
            step = max(1, len(exploratory) // n_exploratory_slots)
            exploratory = exploratory[::step][:n_exploratory_slots]

    target_points = max(0, n_points - n_exploratory_slots)
    while available and len(selected) < target_points:
        meas_so_far = len(existing_points) + len(selected)
        phase = resolve_phase(meas_so_far)

        for pt in available:
            base_metric = pt['variance']
            if enable_phases:
                if phase == 3 and precision_model is not None:
                    if use_projected_fisher:
                        intensity = max(pt.get('precision_intensity', 0.0), 1e-3)
                        base_metric = max(pt['proj_fisher'] / intensity, 1e-12)
                    else:
                        base_metric = max(pt['fisher'], 1e-12)
                elif phase == 2:
                    base_metric = pt['jsd'] if use_jsd else pt['variance']
                else:
                    base_metric = pt['variance']
            else:
                base_metric = pt['jsd'] if use_jsd else pt['variance']

            base_metric *= pt['on_res_bonus'] * pt['zone_bonus']

            history_points = existing_points + selected
            repulse = repulsion_factor(pt['h'], pt['E'], history_points) if not (enable_phases and phase == 3) else 1.0
            weight = 1.0
            if adaptive_coverage:
                idx = np.digitize(pt['h'], coverage_bins)
                weight = 1.0 / (1.0 + coverage_counts[idx])

            pt['adaptive_score'] = base_metric * weight * repulse

        available.sort(key=lambda x: -x['adaptive_score'])
        point = available.pop(0)

        if not point.get('dwell') and (is_too_close(point, selected) or is_too_close(point, existing_points)):
            continue

        point.setdefault('count_time', 60.0)
        selected.append(point)
        if adaptive_coverage and not point.get('dwell'):
            idx = np.digitize(point['h'], coverage_bins)
            coverage_counts[idx] += 1

        # Prune nearby candidates to maintain diversity
        pruned = []
        for candidate in available:
            if is_too_close(candidate, [point]):
                continue
            pruned.append(candidate)
        available = pruned

        # Force a dwell measurement immediately after a gap shot when desired
        if (enable_phases and phase == 3 and use_projected_fisher and
                dwell_multiplier > 1.01 and len(selected) < target_points):
            dwell_copy = point.copy()
            dwell_copy['dwell'] = True
            dwell_copy['count_time'] = point.get('count_time', 60.0) * dwell_multiplier
            selected.append(dwell_copy)
            if len(selected) >= target_points:
                break

    # Add exploratory measurements
    for pt in exploratory:
        if not is_too_close(pt, selected):
            selected.append(pt)
            if len(selected) >= n_points:
                break

    logger.info(f"\nPlanned {len(selected)} measurements:")
    for i, pt in enumerate(selected, 1):
        dwell_tag = " [dwell]" if pt.get('dwell') else ""
        logger.info(f"  {i:2d}. (h,k) = ({pt['h']:.3f}, {pt['k']:.3f}), "
                   f"E = {pt['E']:.1f} meV (ω≈{pt['omega_pred']:.1f}), "
                   f"σ_E = {pt['sigma_E']:.3f} meV, t = {pt.get('count_time',60.0):.0f}s{dwell_tag}")

    return selected


# =============================================================================
# Branch-Aware Priors and Weights
# =============================================================================

BRANCH_PRIORS_DEFAULT = {
    'gap': 0.35,        # Zone center where gaps live
    'mid': 0.30,        # Intermediate h where curvature encodes J2
    'boundary': 0.35    # Zone boundary carrying J1 scale
}

BRANCH_RANGES = {
    'gap': (0.0, 0.18),
    'mid': (0.18, 0.45),
    'boundary': (0.45, 1.8)
}


def classify_branch(h: float) -> str:
    """Map an h-value to a qualitative branch label."""
    for name, (lo, hi) in BRANCH_RANGES.items():
        if lo <= h < hi:
            return name
    return 'boundary'


def count_branch_measurements(measurements: List[Dict], branch: str) -> int:
    """Return how many measurements fall into a given branch."""
    return sum(1 for meas in measurements if classify_branch(meas['h']) == branch)


def should_force_phase3_transition(measurements: List[Dict],
                                   posterior_hint: Optional[List[float]],
                                   min_total: int = 15,
                                   boundary_threshold: int = 8,
                                   gap_threshold: int = 3) -> bool:
    """
    Decide whether to bypass Phase 2 once the boundary physics is settled.

    Trigger when we have already collected plenty of boundary points but still
    lack decisive gap coverage and the posterior remains ambiguous.
    """
    if len(measurements) < min_total:
        return False
    if boundary_threshold > 0:
        boundary_hits = count_branch_measurements(measurements, 'boundary')
        if boundary_hits < boundary_threshold:
            return False
    if gap_threshold >= 0:
        gap_hits = count_branch_measurements(measurements, 'gap')
        if gap_hits >= gap_threshold:
            return False
    if posterior_hint and max(posterior_hint) >= 0.95:
        return False
    return True


def compute_branch_weights(measurements: List[Dict],
                           priors: Optional[Dict[str, float]] = None,
                           prior_strength: float = 0.75) -> Dict[str, float]:
    """
    Build branch weights that combine heuristic priors with observed coverage.

    Parameters
    ----------
    measurements : list of dict
        Current measurement set
    priors : dict
        Desired probability mass per branch (sums to 1)
    prior_strength : float
        How aggressively undersampled branches are boosted
    """
    priors = priors or BRANCH_PRIORS_DEFAULT
    counts = {branch: 0 for branch in BRANCH_RANGES}

    for meas in measurements:
        counts[classify_branch(meas['h'])] += 1

    weights: Dict[str, float] = {}
    for branch, prior in priors.items():
        weights[branch] = prior / (1 + counts[branch])**prior_strength

    # Normalize so the mean weight is 1
    mean_weight = np.mean(list(weights.values()))
    for branch in weights:
        weights[branch] /= mean_weight if mean_weight > 0 else 1.0

    return weights


def apply_gap_focus(branch_weights: Optional[Dict[str, float]],
                    enabled: bool) -> Optional[Dict[str, float]]:
    """Amplify gap-sensitive weights when requested."""
    if not enabled:
        return branch_weights
    if branch_weights is None:
        branch_weights = {branch: 1.0 for branch in BRANCH_RANGES}
    focus_map = {'gap': 1.0, 'mid': 0.05, 'boundary': 0.01}
    return {
        branch: branch_weights.get(branch, 1.0) * focus_map.get(branch, 1.0)
        for branch in BRANCH_RANGES
    }


def build_measurement_weight_array(measurements: List[Dict],
                                   branch_weights: Optional[Dict[str, float]]) -> np.ndarray:
    """Return per-measurement weights based on branch assignments."""
    if not measurements:
        return np.array([])

    arr = np.ones(len(measurements))
    if not branch_weights:
        return arr

    for idx, meas in enumerate(measurements):
        branch = classify_branch(meas['h'])
        arr[idx] = branch_weights.get(branch, 1.0)

    arr /= np.mean(arr)
    return arr


def compute_weighted_chi2(model: SquareLatticeDispersion,
                          measurements: List[Dict],
                          weight_arr: np.ndarray) -> float:
    """Compute χ² with optional per-measurement weights."""
    if weight_arr is None or len(weight_arr) == 0:
        weight_arr = np.ones(len(measurements))

    chi2 = 0.0
    for meas, weight in zip(measurements, weight_arr):
        pred = model.intensity(meas['h'], meas['k'], meas['E'])
        if not np.isfinite(pred):
            # Invalid model prediction should strongly disfavor the model,
            # but must remain finite so AIC/posterior normalization works.
            chi2 += float(weight) * 1e12
            continue
        residual = (meas['intensity'] - pred) / meas['uncertainty']
        if not np.isfinite(residual):
            chi2 += float(weight) * 1e12
            continue
        chi2 += weight * residual**2
    return chi2


# =============================================================================
# Measurement Simulation
# =============================================================================
# =============================================================================
# Measurement Simulation
# =============================================================================

def simulate_measurements(measurement_points: List[Dict], 
                         true_model: SquareLatticeDispersion,
                         count_time: float = 60.0) -> List[Dict]:
    """
    Simulate measurements with Poisson noise.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Measurements (Simulated)")
    logger.info("="*60)
    
    results = []
    
    for pt in measurement_points:
        # True intensity
        I_true = true_model.intensity(pt['h'], pt['k'], pt['E'])
        local_time = pt.get('count_time', count_time)
        
        # Poisson counts plus systematic noise floor
        counts = np.random.poisson(max(1, I_true * local_time))
        I_meas = counts / local_time

        sigma_poisson = np.sqrt(max(counts, 1)) / local_time
        sigma_syst = max(0.03 * I_meas, 0.02)
        final_sigma = np.sqrt(sigma_poisson**2 + sigma_syst**2)
        final_sigma = max(final_sigma, 1e-4)
        
        results.append({
            'h': pt['h'], 'k': pt['k'], 'E': pt['E'],
            'intensity': I_meas,
            'uncertainty': final_sigma,
            'counts': counts,
            'coverage': bool(pt.get('coverage')),
            'symmetry': bool(pt.get('symmetry')),
            'human_hint': bool(pt.get('human_hint')),
            'llm_hint': bool(pt.get('llm_hint')),
            'llm_reason': str(pt.get('reason', ''))[:200],
            'mode': pt.get('mode'),
            'llm_batch_idx': pt.get('llm_batch_idx'),
            'llm_provider': pt.get('llm_provider'),
            'llm_decision_reason': str(pt.get('llm_decision_reason', ''))[:200],
            'loggp_hint': bool(pt.get('loggp_hint')),
            'count_time': local_time,
            'dwell': bool(pt.get('dwell'))
        })
    
    logger.info(f"\nMeasured {len(results)} points:")
    for i, r in enumerate(results, 1):
        logger.info(f"  {i:2d}. E={r['E']:5.1f} meV: I = {r['intensity']:6.1f} ± {r['uncertainty']:.1f}")
    
    return results


# =============================================================================
# Bayesian Model Discrimination
# =============================================================================

def fit_model_parameters(model: SquareLatticeDispersion,
                         measurements: List[Dict],
                         free_params: List[str],
                         use_bumps: bool = True,
                         use_dream: bool = True,
                         weight_arr: Optional[np.ndarray] = None,
                         param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                         bumps_mp: Optional[int] = None,
                         bumps_pop: Optional[int] = None) -> Tuple[Dict, float, Optional[Dict]]:
    """
    Fit model parameters to measurements using bumps/DREAM or scipy.

    Parameters
    ----------
    model : SquareLatticeDispersion
        Model with initial parameter guesses
    measurements : List[Dict]
        Measured data points
    free_params : List[str]
        Which parameters to fit (e.g., ['J1'] or ['J1', 'J2', 'D'])
    use_bumps : bool
        If True, use bumps with DREAM for Bayesian inference

    Returns
    -------
    best_params : Dict
        Optimized parameter values
    min_chi2 : float
        Chi-squared at best fit
    uncertainties : Dict or None
        Parameter uncertainties (std) if using bumps, else None
    """
    if use_bumps and len(measurements) <= len(free_params):
        logger.warning(
            "Bumps requires more data than parameters (n=%d, p=%d); using LM only",
            len(measurements), len(free_params)
        )
        return _fit_with_bumps(model, measurements, free_params, weight_arr,
                               param_bounds=param_bounds, use_dream=False,
                               mp_procs=bumps_mp, pop=bumps_pop)

    if use_bumps:
        try:
            return _fit_with_bumps(model, measurements, free_params, weight_arr,
                                   param_bounds=param_bounds, use_dream=use_dream,
                                   mp_procs=bumps_mp, pop=bumps_pop)
        except Exception as e:
            logger.warning(f"Bumps fitting failed: {e}, falling back to scipy")

    return _fit_with_scipy(model, measurements, free_params, weight_arr,
                           param_bounds=param_bounds)


def _fit_with_scipy(model: SquareLatticeDispersion,
                    measurements: List[Dict],
                    free_params: List[str],
                    weight_arr: Optional[np.ndarray],
                    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Dict, float, None]:
    """Scipy L-BFGS-B fitting (point estimate only)."""
    from scipy.optimize import minimize

    param_bounds = param_bounds or {}
    x0 = []
    bounds = []
    for p in free_params:
        val = getattr(model, p)
        x0.append(val)
        if p in param_bounds:
            bounds.append(param_bounds[p])
        elif p in ['J1', 'J2']:
            bounds.append((0.1, 20.0))
        elif p == 'D':
            bounds.append((0.0, 2.0))
        elif p == 'background':
            bounds.append((0.0, 2.0))

    def objective(x):
        params = {p: x[i] for i, p in enumerate(free_params)}
        test_model = SquareLatticeDispersion(
            J1=params.get('J1', model.J1),
            J2=params.get('J2', model.J2),
            D=params.get('D', model.D),
            S=model.S,
            background=params.get('background', model.background)
        )
        chi2 = compute_weighted_chi2(test_model, measurements, weight_arr)
        return chi2

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    best_params = {p: result.x[i] for i, p in enumerate(free_params)}
    return best_params, result.fun, None


def _fit_with_bumps(model: SquareLatticeDispersion,
                    measurements: List[Dict],
                    free_params: List[str],
                    weight_arr: Optional[np.ndarray],
                    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                    use_dream: bool = True,
                    mp_procs: Optional[int] = None,
                    pop: Optional[int] = None) -> Tuple[Dict, float, Dict]:
    """
    Bumps/DREAM fitting with Bayesian uncertainties.

    Uses two-stage fitting:
    1. DREAM for global exploration
    2. Levenberg-Marquardt for local refinement
    """
    try:
        from bumps.fitproblem import FitProblem
        from bumps.fitters import fit
        from bumps.curve import Curve
    except Exception as exc:
        # Let fit_model_parameters catch this and fall back to scipy.
        raise RuntimeError(f"bumps unavailable: {exc}") from exc

    # Prepare data arrays
    h_arr = np.array([m['h'] for m in measurements])
    k_arr = np.array([m['k'] for m in measurements])
    E_arr = np.array([m['E'] for m in measurements])
    I_arr = np.array([m['intensity'] for m in measurements])
    dI_arr = np.array([m['uncertainty'] for m in measurements])
    if weight_arr is not None and len(weight_arr) == len(dI_arr):
        dI_arr = dI_arr / np.sqrt(weight_arr)

    param_bounds = param_bounds or {}

    def get_bounds(p: str, default: Tuple[float, float]) -> Tuple[float, float]:
        return param_bounds.get(p, default)

    S = model.S
    n_meas = len(measurements)

    # Build intensity model function dynamically based on free params
    # Bumps Curve expects: f(x, param1, param2, ...) -> y
    if free_params == ['J1']:
        fixed_J2, fixed_D = model.J2, model.D
        def intensity_func(idx, J1):
            test = SquareLatticeDispersion(J1=J1, J2=fixed_J2, D=fixed_D, S=S, background=model.background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)

    elif free_params == ['J1', 'D']:
        fixed_J2 = model.J2
        def intensity_func(idx, J1, D):
            test = SquareLatticeDispersion(J1=J1, J2=fixed_J2, D=D, S=S, background=model.background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, D=model.D, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('D', (0.0, 2.0))
        curve.D.range(lo, hi)

    elif free_params == ['J1', 'J2']:
        fixed_D = model.D
        def intensity_func(idx, J1, J2):
            test = SquareLatticeDispersion(J1=J1, J2=J2, D=fixed_D, S=S, background=model.background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, J2=model.J2, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('J2', (0.0, 5.0))
        curve.J2.range(lo, hi)

    elif free_params == ['J1', 'J2', 'D']:
        def intensity_func(idx, J1, J2, D):
            test = SquareLatticeDispersion(J1=J1, J2=J2, D=D, S=S, background=model.background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, J2=model.J2, D=model.D, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('J2', (0.0, 5.0))
        curve.J2.range(lo, hi)
        lo, hi = get_bounds('D', (0.0, 2.0))
        curve.D.range(lo, hi)

    elif free_params == ['J1', 'background']:
        fixed_J2, fixed_D = model.J2, model.D
        def intensity_func(idx, J1, background):
            test = SquareLatticeDispersion(J1=J1, J2=fixed_J2, D=fixed_D, S=S, background=background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, background=model.background, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('background', (0.0, 2.0))
        curve.background.range(lo, hi)

    elif free_params == ['J1', 'D', 'background']:
        fixed_J2 = model.J2
        def intensity_func(idx, J1, D, background):
            test = SquareLatticeDispersion(J1=J1, J2=fixed_J2, D=D, S=S, background=background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, D=model.D, background=model.background, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('D', (0.0, 2.0))
        curve.D.range(lo, hi)
        lo, hi = get_bounds('background', (0.0, 2.0))
        curve.background.range(lo, hi)

    elif free_params == ['J1', 'J2', 'background']:
        fixed_D = model.D
        def intensity_func(idx, J1, J2, background):
            test = SquareLatticeDispersion(J1=J1, J2=J2, D=fixed_D, S=S, background=background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, J2=model.J2, background=model.background, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('J2', (0.0, 5.0))
        curve.J2.range(lo, hi)
        lo, hi = get_bounds('background', (0.0, 2.0))
        curve.background.range(lo, hi)

    elif free_params == ['J1', 'J2', 'D', 'background']:
        def intensity_func(idx, J1, J2, D, background):
            test = SquareLatticeDispersion(J1=J1, J2=J2, D=D, S=S, background=background)
            return np.array([test.intensity(h_arr[i], k_arr[i], E_arr[i]) for i in range(n_meas)])
        curve = Curve(intensity_func, np.arange(n_meas), I_arr, dI_arr,
                      J1=model.J1, J2=model.J2, D=model.D, background=model.background, name='spinwave')
        lo, hi = get_bounds('J1', (0.1, 20.0))
        curve.J1.range(lo, hi)
        lo, hi = get_bounds('J2', (0.0, 5.0))
        curve.J2.range(lo, hi)
        lo, hi = get_bounds('D', (0.0, 2.0))
        curve.D.range(lo, hi)
        lo, hi = get_bounds('background', (0.0, 2.0))
        curve.background.range(lo, hi)
    else:
        raise ValueError(f"Unsupported free_params: {free_params}")

    model_name = getattr(model, 'name', '')
    if 'M4' in model_name:
        if hasattr(curve, 'J1'):
            curve.J1.value = 1.25
            curve.J1.range(0.8, 1.8)
        if hasattr(curve, 'J2'):
            curve.J2.value = 0.2
            curve.J2.range(0.05, 0.6)
        if hasattr(curve, 'D'):
            curve.D.value = 0.02
            curve.D.range(0.002, 0.08)

    problem = FitProblem(curve)

    result_dream = None
    mapper = None
    pool = None
    if mp_procs and mp_procs > 1:
        pool = multiprocessing.Pool(processes=mp_procs)
        mapper = pool.map
        # Avoid passing mapper twice through fit() + problem; use problem.mapper.
        problem.mapper = mapper
    if use_dream:
        # Stage 1: DREAM for global exploration
        try:
            result_dream = fit(problem, method='dream', burn=100, steps=200,
                               pop=pop or 20)
        except Exception as e:
            logger.warning(f"DREAM fitting failed: {e}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    # Stage 2: L-M refinement
    try:
        result_lm = fit(problem, method='lm')
        result_final = result_lm
    except Exception as e:
        if result_dream is None:
            logger.warning(f"L-M refinement failed: {e}, no DREAM result available")
            raise
        logger.warning(f"L-M refinement failed: {e}, using DREAM result")
        result_final = result_dream

    # Extract best-fit values and uncertainties
    best_params = {}
    uncertainties = {}

    labels = problem.labels()
    for j, label in enumerate(labels):
        for p in free_params:
            if p in label:
                best_params[p] = result_final.x[j]
                if hasattr(result_final, 'dx') and result_final.dx is not None:
                    uncertainties[p] = result_final.dx[j]
                else:
                    uncertainties[p] = 0.0

    # Get chi2 properly: result.fun is the total chi2
    try:
        chi2_total = result_final.fun
        chi2_reduced = chi2_total / problem.dof if problem.dof > 0 else chi2_total
    except:
        chi2_total = 0.0

    return best_params, chi2_total, uncertainties


def run_partial_mcmc(model: SquareLatticeDispersion,
                     free_params: List[str],
                     measurements: List[Dict],
                     weight_arr: np.ndarray,
                     prior_means: Dict[str, float],
                     prior_stds: Dict[str, float],
                     n_steps: int = 1200,
                     burn_in: int = 300) -> Dict[str, Any]:
    """
    Lightweight Metropolis sampler that perturbs the most uncertain parameters.
    """
    if not free_params:
        return {}

    current = np.array([getattr(model, p) for p in free_params], dtype=float)
    step_sizes = []
    for p in free_params:
        sigma = prior_stds.get(p, 1.0)
        if sigma <= 0:
            sigma = 1.0
        step_sizes.append(max(0.05, 0.3 * sigma))

    def log_prior(theta: np.ndarray) -> float:
        total = 0.0
        for val, p in zip(theta, free_params):
            sigma = prior_stds.get(p, 1.0)
            if sigma <= 0:
                sigma = 1.0
            mu = prior_means.get(p, getattr(model, p))
            total += -0.5 * ((val - mu) / sigma)**2 - np.log(np.sqrt(2 * np.pi) * sigma)
        return total

    def log_likelihood(theta: np.ndarray) -> float:
        trial = SquareLatticeDispersion(J1=model.J1, J2=model.J2, D=model.D, S=model.S)
        for val, p in zip(theta, free_params):
            setattr(trial, p, val)
        chi2 = compute_weighted_chi2(trial, measurements, weight_arr)
        return -0.5 * chi2

    def within_bounds(theta: np.ndarray) -> bool:
        for val, p in zip(theta, free_params):
            if p in ['J1', 'J2'] and not (0.05 <= val <= 20.0):
                return False
            if p == 'D' and not (0.0 <= val <= 2.5):
                return False
        return True

    log_post = log_likelihood(current) + log_prior(current)
    samples = []
    accepts = 0

    for step in range(n_steps):
        proposal = current + np.random.normal(scale=step_sizes)
        if not within_bounds(proposal):
            continue
        log_post_prop = log_likelihood(proposal) + log_prior(proposal)
        if np.log(np.random.rand()) < (log_post_prop - log_post):
            current = proposal
            log_post = log_post_prop
            accepts += 1
        if step >= burn_in:
            samples.append(current.copy())

    if not samples:
        return {}

    samples_arr = np.array(samples)
    stats = {
        'means': {p: samples_arr[:, i].mean() for i, p in enumerate(free_params)},
        'stds': {p: samples_arr[:, i].std(ddof=1) for i, p in enumerate(free_params)},
        'accept_rate': accepts / max(1, n_steps)
    }
    return stats


def discriminate_models(measurements: List[Dict],
                       candidates: List[Dict],
                       measurement_weights: Optional[np.ndarray] = None,
                       use_partial_mcmc: bool = False,
                       branch_weight_dict: Optional[Dict[str, float]] = None,
                       use_bumps: bool = True,
                       use_dream: bool = True,
                       fit_background: bool = True,
                       lock_gapless_background: bool = True,
                       clamp_measurements: bool = False,
                       bumps_mp: Optional[int] = None,
                       bumps_pop: Optional[int] = None) -> Dict:
    """
    Perform Bayesian model comparison with parameter fitting.

    Each model's free parameters are optimized before computing likelihoods.
    This ensures fair comparison - we compare the best version of each model.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Bayesian Model Discrimination")
    logger.info("="*60)

    results = {}
    if branch_weight_dict:
        logger.info("Using branch-aware weights: %s", branch_weight_dict)

    if measurement_weights is None or len(measurement_weights) != len(measurements):
        measurement_weights = np.ones(len(measurements))

    effective_n = measurement_weights.sum() if len(measurement_weights) else len(measurements)
    clamp_info = None
    if clamp_measurements and len(measurements) >= 10:
        intens = np.array([m['intensity'] for m in measurements])
        level_backgr, thresh_intens = compute_heuris_experi_param(
            intens,
            LOGGP_LEVEL_BACKGR_DIFFS_REL_MAX,
            LOGGP_LEVEL_BACKGR_DIFFS_ABS_MIN,
            LOGGP_LEVEL_BACKGR_DECILE_MAX,
            LOGGP_THRESH_INTENS_FACT,
            None
        )
        if level_backgr is not None and thresh_intens is not None:
            clamp_info = {'level_backgr': level_backgr, 'thresh_intens': thresh_intens}
            logger.info("Physics clamp (tau/gamma): background=%.3f, threshold=%.3f",
                        level_backgr, thresh_intens)
            clamped = []
            for m in measurements:
                m2 = dict(m)
                m2['intensity'] = min(m2['intensity'], thresh_intens)
                clamped.append(m2)
            measurements = clamped

    for cand in candidates:
        # Determine which parameters are free for this model
        free_params = []
        if cand['params']['J1'] != 0:
            free_params.append('J1')
        if cand['params']['J2'] != 0:
            free_params.append('J2')
        if cand['params']['D'] != 0:
            free_params.append('D')
        if fit_background:
            free_params.append('background')

        # Create initial model
        model = SquareLatticeDispersion(**cand['params'])
        model.name = cand['name']
        model.prior = cand['prior']

        if clamp_info and fit_background:
            model.background = max(0.0, clamp_info['level_backgr'])

        if model.name.startswith('M1') or model.name.startswith('M3'):
            eps = 1e-6
            # --- PHYSICS CONSTRAINT ---
            # Gapless NN-only models are prone to "cheating" by inflating a
            # floating background to mimic the band-edge peak. We lock that
            # nuisance parameter to a tiny positive floor so it cannot absorb
            # the gap intensity (the "Silent Data" mitigation described in the
            # manuscript) while still avoiding log(0) issues in the likelihood.
            model.background = max(eps, getattr(model, 'background', eps))
            if lock_gapless_background and 'background' in free_params:
                free_params = [p for p in free_params if p != 'background']

        mcmc_stats = None
        param_bounds = None
        if model.name.startswith('M4'):
            # Bound only by sign and broad scale (avoid excluding true parameters).
            j1_max = max(0.1, 5.0 * abs(getattr(model, 'J1', 1.0)))
            j2_max = max(0.1, 5.0 * abs(getattr(model, 'J2', 0.2)))
            d_max = 1.0
            param_bounds = {
                'J1': (0.0, j1_max),
                'J2': (0.0, j2_max),
                'D': (0.0, d_max)
            }
        if fit_background:
            bg_default = (0.0, 2.0)
            if clamp_info:
                level_backgr = clamp_info['level_backgr']
                bg_default = (max(0.0, level_backgr * 0.5), max(level_backgr * 3.0, 0.2))
            if param_bounds is None:
                param_bounds = {}
            param_bounds.setdefault('background', bg_default)

        # Fit the free parameters
        if free_params:
            best_params, chi2, uncertainties = fit_model_parameters(
                model, measurements, free_params, use_bumps=use_bumps,
                use_dream=use_dream,
                weight_arr=measurement_weights,
                param_bounds=param_bounds,
                bumps_mp=bumps_mp,
                bumps_pop=bumps_pop
            )
            # Update model with fitted parameters
            for p, v in best_params.items():
                setattr(model, p, v)
            logger.info(f"\n  {cand['name']}: fitted {free_params}")
            for p in free_params:
                unc = uncertainties.get(p, 0.0) if uncertainties else 0.0
                if unc > 0:
                    logger.info(f"    {p} = {getattr(model, p):.3f} ± {unc:.3f} meV")
                else:
                    logger.info(f"    {p} = {getattr(model, p):.3f} meV")
        else:
            chi2 = compute_weighted_chi2(model, measurements, measurement_weights)

        if use_partial_mcmc and free_params:
            mcmc_stats = run_partial_mcmc(
                model,
                free_params,
                measurements,
                measurement_weights,
                cand['params'],
                cand.get('uncertainties', {})
            )
            if mcmc_stats:
                for p, mean in mcmc_stats['means'].items():
                    setattr(model, p, mean)
                if uncertainties is None:
                    uncertainties = {}
                uncertainties.update(mcmc_stats['stds'])
                chi2 = compute_weighted_chi2(model, measurements, measurement_weights)
                logger.info("    (MCMC refinement) accept=%.0f%%",
                            100 * mcmc_stats['accept_rate'])

        # Compute log-likelihood at best-fit
        if not np.isfinite(chi2):
            chi2 = 1e12
        log_like = -0.5 * chi2

        # Compute predictions at best-fit parameters
        predictions = []
        for meas in measurements:
            pred = model.intensity(meas['h'], meas['k'], meas['E'])
            predictions.append(pred)

        # Information criterion (AIC) for model complexity
        n_params = len(free_params)
        n_data = effective_n if effective_n > 0 else len(measurements)
        aic = chi2 + 2.0 * n_params
        prior = max(float(model.prior), 1e-12)
        log_posterior = (-0.5 * aic) + np.log(prior)

        # Penalize fits that sit on parameter bounds (constraint-active).
        constraint_active = False
        if param_bounds:
            for p, (lo, hi) in param_bounds.items():
                if not hasattr(model, p):
                    continue
                val = getattr(model, p)
                span = max(hi - lo, 1e-9)
                if abs(val - lo) < 0.01 * span or abs(val - hi) < 0.01 * span:
                    constraint_active = True
                    break
        if constraint_active:
            log_posterior -= 2.0

        results[model.name] = {
            'log_likelihood': log_like,
            'log_prior': np.log(prior),
            'log_posterior': log_posterior,
            'predictions': predictions,
            'chi2': chi2 / max(n_data, 1),
            'n_params': n_params,
            'aic': aic,
            'fitted_model': model,
            'uncertainties': uncertainties,
            'mcmc_stats': mcmc_stats,
            'constraint_active': constraint_active
        }
    
    # Normalize posteriors using BIC-based approximation
    log_posts = np.array([res['log_posterior'] for res in results.values()], dtype=float)
    finite_mask = np.isfinite(log_posts)
    if not np.any(finite_mask):
        posteriors = np.full(len(results), 1.0 / max(len(results), 1), dtype=float)
    else:
        safe_logs = log_posts.copy()
        safe_logs[~finite_mask] = -1e12
        max_log = np.max(safe_logs)
        posteriors = np.exp(safe_logs - max_log)
        post_sum = posteriors.sum()
        if not np.isfinite(post_sum) or post_sum <= 0:
            posteriors = np.full(len(results), 1.0 / max(len(results), 1), dtype=float)
        else:
            posteriors /= post_sum

    for i, name in enumerate(results.keys()):
        results[name]['posterior'] = posteriors[i]

    # Report
    logger.info("\nModel comparison (with fitted parameters):")
    logger.info("-" * 50)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]['posterior'])

    for name, res in sorted_results:
        logger.info(f"\n  {name}:")
        logger.info(f"    Free params: {res['n_params']}")
        logger.info(f"    χ²/N:        {res['chi2']:6.2f}")
        logger.info(f"    AIC:         {res['aic']:6.1f}")
        logger.info(f"    Posterior:   {res['posterior']:6.1%}")

    # Bayes factor
    best_name = sorted_results[0][0]
    second_name = sorted_results[1][0]
    if results[second_name]['posterior'] > 0:
        bf = results[best_name]['posterior'] / results[second_name]['posterior']
        logger.info(f"\n  Bayes factor ({best_name.split(':')[0]} vs {second_name.split(':')[0]}): {bf:.1f}")
    else:
        bf = np.inf
        logger.info(f"\n  Bayes factor ({best_name.split(':')[0]} vs {second_name.split(':')[0]}): ∞")

    if bf > 100:
        logger.info("  → Decisive evidence")
    elif bf > 10:
        logger.info("  → Strong evidence")
    elif bf > 3:
        logger.info("  → Substantial evidence")
    else:
        logger.info("  → Inconclusive")

    return results


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(structure: Dict,
                         candidates: List[Dict],
                         measurements: List[Dict],
                         results: Dict,
                         true_model: SquareLatticeDispersion):
    """Create publication-quality figures."""
    
    logger.info("\n" + "="*60)
    logger.info("Creating Visualizations")
    logger.info("="*60)
    
    # Figure 1: Dispersion comparison
    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
    
    h_plot = np.linspace(0.5, 1.7, 100)
    
    # True dispersion
    omega_true = [true_model.omega(h, h) for h in h_plot]
    
    for ax, cand in zip(axes1, candidates):
        model = SquareLatticeDispersion(**cand['params'])
        omega = [model.omega(h, h) for h in h_plot]
        
        ax.plot(h_plot, omega_true, 'k-', lw=2, label='True')
        ax.plot(h_plot, omega, 'r--', lw=2, label='Candidate')
        
        # Measurement points
        for meas in measurements:
            ax.scatter(meas['h'], meas['E'], s=meas['intensity']*3 + 10, 
                      c='blue', alpha=0.6, edgecolors='black')
        
        ax.set_xlabel('(h, h, 0) [r.l.u.]')
        ax.set_ylabel('Energy [meV]')
        ax.set_title(f"{cand['name']}\nP = {results[cand['name']]['posterior']:.1%}")
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0.5, 1.7)
        ax.set_ylim(0, 50)
    
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / 'closed_loop_dispersion.png', dpi=150)
    plt.close(fig1)
    logger.info(f"  Saved: {FIGURES_DIR / 'closed_loop_dispersion.png'}")
    
    # Figure 2: Model probabilities
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    names = [c['name'].replace('Model-', 'M') for c in candidates]
    priors = [c['prior'] for c in candidates]
    posteriors = [results[c['name']]['posterior'] for c in candidates]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, priors, width, label='Prior', color='lightblue', edgecolor='black')
    bars2 = ax2.bar(x + width/2, posteriors, width, label='Posterior', color='steelblue', edgecolor='black')
    
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Add values on bars
    for bar, val in zip(bars2, posteriors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Bayesian Model Comparison')
    
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / 'closed_loop_probabilities.png', dpi=150)
    plt.close(fig2)
    logger.info(f"  Saved: {FIGURES_DIR / 'closed_loop_probabilities.png'}")
    
    # Figure 3: Closed-loop workflow diagram
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    
    # Boxes
    boxes = [
        (1, 4.5, 'Crystal\nStructure'),
        (3.5, 4.5, 'GNN\nHypothesis'),
        (6, 4.5, 'MCTS\nPlanning'),
        (8, 4.5, 'TAS-AI\nMeasure'),
        (6, 2, 'Bayesian\nDiscrimination'),
        (3.5, 2, 'Validated\nHamiltonian'),
        (1, 2, 'GNN\nTraining'),
    ]
    
    for x, y, text in boxes:
        rect = plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8, 
                             facecolor='lightsteelblue', edgecolor='black', lw=2)
        ax3.add_patch(rect)
        ax3.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        (1.6, 4.5, 0.8, 0),      # Structure → GNN
        (4.1, 4.5, 0.8, 0),      # GNN → MCTS
        (6.6, 4.5, 0.8, 0),      # MCTS → TAS-AI
        (8, 4.1, 0, -1.0),       # TAS-AI → down
        (7.4, 2, -0.8, 0),       # → Bayes
        (5.4, 2, -0.8, 0),       # Bayes → Validated
        (2.9, 2, -0.8, 0),       # Validated → GNN Training
        (1, 2.6, 0, 1.0),        # GNN Training → up (feedback)
    ]
    
    for x, y, dx, dy in arrows:
        ax3.arrow(x, y, dx*0.8, dy*0.8, head_width=0.15, head_length=0.1, 
                 fc='gray', ec='gray')
    
    # Feedback arc
    ax3.annotate('', xy=(1, 4.1), xytext=(1, 2.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.5, 3.3, 'Feedback', rotation=90, ha='center', va='center', 
            fontsize=9, color='green', fontweight='bold')
    
    ax3.set_title('Closed-Loop Autonomous Discovery Workflow', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / 'closed_loop_workflow.png', dpi=150)
    plt.close(fig3)
    logger.info(f"  Saved: {FIGURES_DIR / 'closed_loop_workflow.png'}")
    
    # Figure 4: Summary infographic
    fig4 = plt.figure(figsize=(12, 8))
    
    # Create grid
    gs = fig4.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Structure
    ax_a = fig4.add_subplot(gs[0, 0])
    ax_a.set_xlim(-0.5, 1.5)
    ax_a.set_ylim(-0.5, 1.5)
    
    # Draw unit cell
    for i in range(2):
        for j in range(2):
            ax_a.scatter(i, j, s=300, c='red', edgecolors='black', zorder=3)
            ax_a.text(i, j-0.15, 'Fe', ha='center', va='top', fontsize=8)
    
    # Oxygen
    ax_a.scatter(0.5, 0, s=150, c='blue', marker='s', zorder=2)
    ax_a.scatter(0, 0.5, s=150, c='blue', marker='s', zorder=2)
    ax_a.scatter(0.5, 1, s=150, c='blue', marker='s', zorder=2)
    ax_a.scatter(1, 0.5, s=150, c='blue', marker='s', zorder=2)
    
    ax_a.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=1)
    ax_a.set_aspect('equal')
    ax_a.set_title('Crystal structure overview', fontweight='bold')
    ax_a.axis('off')
    ax_a.text(0.5, -0.3, 'Fe-O-Fe superexchange\na = 4.0 Å', ha='center', fontsize=9)
    
    # Panel B: Candidates - show all 4 models
    ax_b = fig4.add_subplot(gs[0, 1])

    cand_text = """GK Candidate Hamiltonians:

M1: H = J₁ Σ Sᵢ·Sⱼ  (NN only)

M2: H = J₁ Σ + D Σ(Sᶻ)²  (NN+D)

M3: H = J₁ Σ + J₂ Σ  (J₁-J₂)

M4: H = J₁ + J₂ + D  (Full)"""

    ax_b.text(0.05, 0.95, cand_text, transform=ax_b.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             linespacing=1.4)
    ax_b.set_title('Goodenough-Kanamori hypotheses', fontweight='bold')
    ax_b.axis('off')
    
    # Panel C: Measurements
    ax_c = fig4.add_subplot(gs[0, 2])

    # Plot dispersion and measurements
    h_plot = np.linspace(0.5, 1.7, 100)
    omega_true = [true_model.omega(h, h) for h in h_plot]
    ax_c.plot(h_plot, omega_true, 'k-', lw=2, label='Dispersion')

    # Plot all measurements with style cues for coverage/hints
    style_map = {
        'coverage': dict(color='#f6c343', marker='s', label='Forced coverage seed'),
        'loggp_init': dict(color='#6c8ef5', marker='s', label='Log-GP init sweep'),
        'loggp_active': dict(color='#3b6fb6', marker='o', label='Log-GP active'),
        'human': dict(color='#2d7dd2', marker='^', label='Human gap hint'),
        'auto': dict(color='#d1495b', marker='o', label='Autonomous selection')
    }
    legend_used = set()

    for i, meas in enumerate(measurements, 1):
        if meas.get('loggp_init'):
            style_key = 'loggp_init'
        elif meas.get('loggp_active'):
            style_key = 'loggp_active'
        elif meas.get('human_hint'):
            style_key = 'human'
        elif meas.get('coverage'):
            style_key = 'coverage'
        else:
            style_key = 'auto'
        style = style_map[style_key]
        label = style['label'] if style_key not in legend_used else "_nolegend_"
        legend_used.add(style_key)
        ax_c.scatter(meas['h'], meas['E'], s=65, c=style['color'], alpha=0.85,
                     marker=style['marker'], edgecolors='black', linewidths=0.8,
                     zorder=5, label=label)
        ax_c.annotate(str(i), (meas['h'], meas['E']), fontsize=7,
                      xytext=(3, 3), textcoords='offset points', color='black')

    ax_c.set_xlabel('(h, h, 0) [r.l.u.]')
    ax_c.set_ylabel('E [meV]')
    ax_c.set_title(f'Measurement queue (n={len(measurements)})', fontweight='bold')
    ax_c.set_xlim(0.5, 1.7)
    ax_c.set_ylim(0, 50)
    ax_c.legend(loc='upper left', fontsize=8, frameon=False)
    
    # Panel D: Results
    ax_d = fig4.add_subplot(gs[1, :])
    
    # Bar chart
    names_short = ['M1: NN', 'M2: NN+D', 'M3: J1-J2', 'M4: Full']
    posteriors = [results[c['name']]['posterior'] for c in candidates]
    colors = ['#ff6b6b', '#ffa94d', '#4ecdc4', '#45b7d1']
    
    bars = ax_d.barh(names_short, posteriors, color=colors, edgecolor='black', height=0.5)
    
    # Highlight winner
    winner_idx = np.argmax(posteriors)
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)
    
    ax_d.set_xlabel('Posterior Probability')
    ax_d.set_xlim(0, 1)
    ax_d.set_title('Bayesian model selection result', fontweight='bold')
    
    # Add probability labels
    for bar, prob in zip(bars, posteriors):
        ax_d.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1%}', va='center', fontsize=11, fontweight='bold')
    
    # Winner annotation
    winner_name = candidates[winner_idx]['name']
    winner_params = candidates[winner_idx]['params']
    param_str = ', '.join([f'{k}={v:.1f}' for k, v in winner_params.items() if v != 0])
    
    ax_d.text(0.98, 0.02, f'Winner: {winner_name}\nParameters: {param_str} meV',
             transform=ax_d.transAxes, ha='right', va='bottom',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Closed-Loop Autonomous Magnetism: Toy System Demo',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / 'closed_loop_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    logger.info(f"  Saved: {FIGURES_DIR / 'closed_loop_summary.png'}")


def create_posterior_evolution_figure(candidates: List[Dict],
                                      measurements: List[Dict],
                                      true_model: SquareLatticeDispersion):
    """
    Create figure showing how posteriors evolve as measurements accumulate.

    Reorders measurements by information content (least informative first) to
    show gradual evolution from uniform priors to confident discrimination.
    """
    logger.info("  Creating posterior evolution figure...")

    from scipy.optimize import minimize

    # Create models
    models = []
    for cand in candidates:
        model = SquareLatticeDispersion(**cand['params'])
        model.name = cand['name']
        model.prior = cand['prior']
        models.append(model)

    # Use actual measurement order (MCTS-planned sequence)
    # This shows how posteriors evolve as the experiment progresses
    meas_sorted = measurements  # Keep original MCTS order

    # Define fitting function - uses same bumps fitting as main discrimination
    def fit_and_compute_posterior(meas_subset, models, candidates):
        """Fit each model and compute posteriors using bumps (same as main)."""
        results = {}

        for cand in candidates:
            # Determine free params
            free_params = []
            if cand['params']['J1'] != 0:
                free_params.append('J1')
            if cand['params']['J2'] != 0:
                free_params.append('J2')
            if cand['params']['D'] != 0:
                free_params.append('D')

            # Create model with initial guesses
            model = SquareLatticeDispersion(**cand['params'])
            param_bounds = None
            if cand['name'].startswith('M4'):
                j1_max = max(0.1, 5.0 * abs(cand['params'].get('J1', 1.0)))
                j2_max = max(0.1, 5.0 * abs(cand['params'].get('J2', 0.2)))
                d_max = 1.0
                param_bounds = {
                    'J1': (0.0, j1_max),
                    'J2': (0.0, j2_max),
                    'D': (0.0, d_max)
                }

            # Use same fitting as main discrimination
            if free_params:
                best_params, chi2, _ = fit_model_parameters(
                    model, meas_subset, free_params, use_bumps=True,
                    param_bounds=param_bounds
                )
            else:
                chi2 = sum(((m['intensity'] - model.intensity(m['h'], m['k'], m['E'])) / m['uncertainty'])**2
                          for m in meas_subset)

            n_params = len(free_params)
            n_data = len(meas_subset)
            log_like = -0.5 * chi2
            aic = chi2 + 2.0 * n_params
            log_post = (-0.5 * aic) + np.log(cand['prior'])

            results[cand['name']] = {'log_posterior': log_post, 'chi2': chi2 / n_data}

        # Normalize posteriors
        log_posts = [r['log_posterior'] for r in results.values()]
        max_log = max(log_posts)
        posteriors = np.exp(np.array(log_posts) - max_log)
        posteriors /= posteriors.sum()

        for i, name in enumerate(results.keys()):
            results[name]['posterior'] = posteriors[i]

        return results

    # Compute posteriors at finer granularity to show evolution
    n_points = [3, 5, 7, 10, 15]
    n_points = [n for n in n_points if n <= len(meas_sorted)]

    evolution_data = {}
    for n in n_points:
        meas_subset = meas_sorted[:n]
        results = fit_and_compute_posterior(meas_subset, models, candidates)
        evolution_data[n] = results

        # Log intermediate results
        m4_post = results['M4: Full model']['posterior']
        logger.info(f"    n={n}: M4 posterior = {m4_post:.1%}")

    # Create figure
    fig, axes = plt.subplots(1, len(n_points), figsize=(3.5*len(n_points), 4), sharey=True)
    if len(n_points) == 1:
        axes = [axes]

    colors = ['#ff6b6b', '#ffa94d', '#4ecdc4', '#45b7d1']
    names_short = ['M1', 'M2', 'M3', 'M4']
    cand_names = [c['name'] for c in candidates]

    for i, (n, ax) in enumerate(zip(n_points, axes)):
        posteriors = [evolution_data[n][name]['posterior'] for name in cand_names]

        bars = ax.barh(range(4), posteriors, color=colors, edgecolor='black', height=0.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels(names_short)

        # Highlight winner
        winner_idx = np.argmax(posteriors)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(2)

        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Posterior Probability')
        ax.set_title(f'n = {n}', fontweight='bold', fontsize=12)

        # Add probability labels
        for bar, prob in zip(bars, posteriors):
            if prob > 0.01:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.0%}', va='center', fontsize=10)

    axes[0].set_ylabel('Model')

    plt.suptitle('Posterior Evolution During MCTS-Planned Experiment',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'posterior_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {FIGURES_DIR / 'posterior_evolution.png'}")


def create_evolution_animation(candidates: List[Dict],
                               measurements: List[Dict],
                               true_model: SquareLatticeDispersion):
    """
    Create animated GIF showing posterior evolution as measurements arrive.
    """
    logger.info("  Creating evolution animation...")

    def fit_and_compute_posteriors(meas_subset: List[Dict]) -> np.ndarray:
        """Compute AIC-weighted posteriors using the same fitting logic as Figure 10."""
        results = {}
        for cand in candidates:
            free_params = []
            if cand['params']['J1'] != 0:
                free_params.append('J1')
            if cand['params']['J2'] != 0:
                free_params.append('J2')
            if cand['params']['D'] != 0:
                free_params.append('D')

            model = SquareLatticeDispersion(**cand['params'])
            param_bounds = None
            if cand['name'].startswith('M4'):
                j1_max = max(0.1, 5.0 * abs(cand['params'].get('J1', 1.0)))
                j2_max = max(0.1, 5.0 * abs(cand['params'].get('J2', 0.2)))
                d_max = 1.0
                param_bounds = {
                    'J1': (0.0, j1_max),
                    'J2': (0.0, j2_max),
                    'D': (0.0, d_max)
                }

            if free_params:
                _, chi2, _ = fit_model_parameters(
                    model, meas_subset, free_params, use_bumps=True,
                    param_bounds=param_bounds
                )
            else:
                chi2 = sum(((m['intensity'] - model.intensity(m['h'], m['k'], m['E'])) / m['uncertainty'])**2
                           for m in meas_subset)

            n_params = len(free_params)
            aic = chi2 + 2.0 * n_params
            log_post = (-0.5 * aic) + np.log(cand['prior'])
            results[cand['name']] = log_post

        log_posts = [results[name] for name in results.keys()]
        max_log = max(log_posts)
        posteriors = np.exp(np.array(log_posts) - max_log)
        posteriors /= posteriors.sum()
        return posteriors

    posteriors_history = []
    for n_meas in range(1, len(measurements) + 1):
        meas_subset = measurements[:n_meas]
        posteriors_history.append(fit_and_compute_posteriors(meas_subset))

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    h_plot = np.linspace(0.5, 1.7, 100)
    omega_true = [true_model.omega(h, h) for h in h_plot]

    colors = ['#ff6b6b', '#ffa94d', '#4ecdc4', '#45b7d1']
    names = ['M1: NN', 'M2: NN+D', 'M3: J1-J2', 'M4: Full']

    def init():
        ax1.clear()
        ax2.clear()
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()

        n = min(frame + 1, len(measurements))

        # Left panel: dispersion with measurements
        ax1.plot(h_plot, omega_true, 'k-', lw=2, label='True dispersion')
        for i, meas in enumerate(measurements[:n]):
            ax1.scatter(meas['h'], meas['E'], s=80, c='red', alpha=0.8,
                       edgecolors='black', zorder=5)
        ax1.set_xlabel('(h, h, 0) [r.l.u.]')
        ax1.set_ylabel('E [meV]')
        ax1.set_xlim(0.5, 1.7)
        ax1.set_ylim(0, 50)
        ax1.set_title(f'Measurements: {n}/{len(measurements)}')

        # Right panel: posterior bars
        if n > 0 and n <= len(posteriors_history):
            posteriors = posteriors_history[n-1]
            bars = ax2.barh(names, posteriors, color=colors, edgecolor='black')
            for bar, prob in zip(bars, posteriors):
                if prob > 0.01:
                    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{prob:.1%}', va='center', fontsize=10)
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel('Posterior Probability')
        ax2.set_title('Model Probabilities')

        plt.suptitle('Closed-Loop Model Discrimination', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(measurements) + 3,  # Extra frames at end
                        interval=600, blit=False)

    gif_path = FIGURES_DIR / 'closed_loop_evolution.gif'
    anim.save(gif_path, writer=PillowWriter(fps=2))
    plt.close(fig)
    logger.info(f"  Saved: {gif_path}")


# =============================================================================
# Main Demo
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop toy demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-zone-coverage", action="store_true",
                        help="Seed initial measurements near zone center and edge")
    parser.add_argument("--human-gap-hints", action="store_true",
                        help="Inject human-suggested measurements near the gap")
    parser.add_argument("--human-hint-after", type=int, default=-1,
                        help="Insert human hints after N planned points (-1 appends at end)")
    parser.add_argument("--adaptive-coverage", action="store_true",
                        help="Adaptively boost under-sampled h segments during planning")
    parser.add_argument("--branch-priors", action="store_true",
                        help="Weight likelihoods using per-branch priors")
    parser.add_argument("--gap-only-evidence", action="store_true",
                        help="Down-weight non-gap measurements when computing evidence")
    parser.add_argument("--partial-mcmc", action="store_true",
                        help="Run a lightweight MCMC refinement after fitting")
    parser.add_argument("--fit-background", action="store_true",
                        help="Fit background as a nuisance parameter during discrimination")
    parser.add_argument("--no-fit-background", dest="fit_background", action="store_false",
                        help="Disable background fitting during discrimination")
    parser.set_defaults(fit_background=True)
    parser.add_argument("--lock-gapless-background", action="store_true",
                        help="Lock background for gapless models (M1/M3) to avoid silent-data cheating")
    parser.add_argument("--unlock-gapless-background", dest="lock_gapless_background", action="store_false",
                        help="Allow background to float for the gapless NN-only model")
    parser.set_defaults(lock_gapless_background=True)
    parser.add_argument("--physics-use-loggp-clamp", action="store_true",
                        help="Apply Log-GP tau/gamma clamp when computing physics-model evidence")
    parser.add_argument("--no-physics-loggp-clamp", dest="physics_use_loggp_clamp", action="store_false",
                        help="Disable Log-GP tau/gamma clamp for physics-model evidence")
    parser.set_defaults(physics_use_loggp_clamp=True)
    parser.add_argument("--symmetry-seed", action="store_true",
                        help="Reserve 10%% of the plan for Γ/X/M measurements")
    parser.add_argument("--jsd-acquisition", action="store_true",
                        help="Use JSD-based disagreement scoring when planning")
    parser.add_argument("--phase-planner", action="store_true",
                        help="Enable three-phase planning (seed/JSD/Fisher)")
    parser.add_argument("--projected-fisher", action="store_true",
                        help="Use projected Fisher scores during phase-3 refinement")
    parser.add_argument("--phase2-limit", type=int, default=30,
                        help="Measurement count threshold for entering phase 3")
    parser.add_argument("--total-measurements", type=int, default=None,
                        help="Maximum number of measurements (default max(phase2+5, 25))")
    parser.add_argument("--phase3-dwell", action="store_true",
                        help="Allow stacking measurements in phase 3 to reduce noise")
    parser.add_argument("--phase3-dwell-multiplier", type=float, default=3.0,
                        help="Multiplier for count time when dwelling in phase 3")
    parser.add_argument("--phase3-threshold", type=float, default=0.5,
                        help="Posterior threshold to enter phase 3 (default 0.5)")
    parser.add_argument("--disable-bumps", action="store_true",
                        help="Skip bumps/DREAM fitting and fall back to deterministic L-BFGS")
    parser.add_argument("--no-animations", action="store_true",
                        help="Skip GIF creation to speed up headless runs")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all figure generation")
    parser.add_argument("--bumps-interval", type=int, default=5,
                        help="Run BUMPS every N measurements (default 5)")
    parser.add_argument("--bumps-mp", type=int, default=0,
                        help="Use multiprocessing for BUMPS/DREAM with N workers")
    parser.add_argument("--bumps-pop", type=int, default=0,
                        help="Population size for DREAM (0 = default)")
    parser.add_argument("--simple-resolution", action="store_true",
                        help="Use an empirical Gaussian energy width instead of Cooper-Nathans")
    parser.add_argument("--equal-priors", action="store_true",
                        help="Override candidate model priors to be uniform")
    parser.add_argument("--disable-symmetry-seed", action="store_true",
                        help="Skip mandatory symmetry seeding in phase 1")
    parser.add_argument("--symmetry-seed-count", type=int, default=0,
                        help="Override the number of symmetry seed points to inject (0 = default)")
    parser.add_argument("--hybrid-loggp", action="store_true",
                        help="Run agnostic Log-GP phase before TAS-AI planning")
    parser.add_argument("--loggp-measurements", type=int, default=10,
                        help="Number of Log-GP measurements in hybrid phase")
    parser.add_argument("--loggp-init-from", type=str, default=None,
                        help="Path to JSON with Log-GP init measurements to reuse")
    parser.add_argument("--loggp-pre-symmetry", action="store_true",
                        help="Inject symmetry points after Log-GP init, before Log-GP active picks")
    parser.add_argument("--loggp-grid-h", type=int, default=24,
                        help="Log-GP grid H resolution")
    parser.add_argument("--loggp-grid-e", type=int, default=18,
                        help="Log-GP grid E resolution")
    parser.add_argument("--loggp-taper-mode", choices=["none", "energy", "energy_h"],
                        default="energy",
                        help="Boundary taper used in the Log-GP active phase.")
    parser.add_argument("--demo-hmin", type=float, default=None,
                        help="Optional shared H minimum for paper-demo Log-GP and physics phases")
    parser.add_argument("--demo-hmax", type=float, default=None,
                        help="Optional shared H maximum for paper-demo Log-GP and physics phases")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a closed-loop checkpoint JSON file")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for closed-loop checkpoint output")
    parser.add_argument("--llm-in-loop", action="store_true",
                        help="Query LLMs for measurement suggestions during phase 3")
    parser.add_argument("--llm-max-points", type=int, default=3,
                        help="Number of LLM-suggested points to inject per batch")
    parser.add_argument("--llm-history", type=int, default=20,
                        help="Number of recent measurements to include in LLM prompt")
    parser.add_argument("--llm-cadence", type=int, default=1,
                        help="Query LLMs every N phase-3 batches")
    parser.add_argument("--llm-decider", type=str, default="rotate",
                        choices=["rotate", "claude", "gemini", "codex", "none"],
                        help="Choose which model decides consensus (rotate by default)")
    parser.add_argument("--llm-output-dir", type=str, default=str(PROMPTS_DIR / "llm_inloop_runs"),
                        help="Directory for LLM prompts and responses")
    parser.add_argument("--llm-dry-run", action="store_true",
                        help="Write prompts but do not call LLM CLIs")
    parser.add_argument("--llm-external", action="store_true",
                        help="Read LLM suggestions from JSON files instead of calling CLIs")
    parser.add_argument("--llm-mailbox-url", type=str, default=None,
                        help="Mailbox base URL (e.g., http://tripleaxis.org/tasai_mailbox)")
    parser.add_argument("--llm-mailbox-token", type=str, default=None,
                        help="Mailbox shared token (X-LLM-Token).")
    parser.add_argument("--llm-mailbox-run-id", type=str, default=None,
                        help="Optional run identifier prefix to avoid mailbox collisions.")
    parser.add_argument("--llm-one-shot", action="store_true",
                        help="Inject only the first LLM suggestion, save checkpoint, then exit.")
    parser.add_argument("--llm-wait-seconds", type=int, default=0,
                        help="Seconds to wait for external LLM suggestions per batch")
    parser.add_argument("--llm-phase", type=str, default="phase3",
                        choices=["phase0", "phase1", "phase2", "phase3", "all"],
                        help="Which phase(s) to inject LLM suggestions")
    return parser.parse_args()


def main(args=None):
    """Run the complete closed-loop demonstration."""

    if args is None:
        args = parse_args()

    if args.simple_resolution:
        logger.info("Using simplified Gaussian resolution model (Cooper-Nathans disabled).")
        init_tas(False)
    else:
        init_tas(BASE_HAS_RESOLUTION)

    use_bumps_default = not args.disable_bumps
    if not use_bumps_default:
        logger.info("Bumps fitting disabled; using deterministic L-BFGS estimates only.")
    bumps_interval = max(1, args.bumps_interval)

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     Closed-Loop Autonomous Magnetism: Toy System Demo             ║
║     With Realistic TAS Resolution (Cooper-Nathans)                ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Log TAS configuration
    logger.info("TAS Instrument Configuration:")
    logger.info(f"  Fixed Ef: {TAS.efixed} meV")
    logger.info(f"  Collimations: {TAS.hcol} arcmin (horizontal)")
    logger.info(f"  Method: Cooper-Nathans")
    if TAS.res_calc is not None:
        logger.info("  Resolution: Realistic (from rescalculator)")
    else:
        logger.info("  Resolution: Empirical approximation")

    # Define the TRUE model (what nature has - unknown to the algorithm)
    TRUE_J1 = 1.25
    TRUE_J2 = 0.2
    TRUE_D = 0.02
    true_model = SquareLatticeDispersion(J1=TRUE_J1, J2=TRUE_J2, D=TRUE_D, S=2.5)

    logger.info(f"\n[Ground truth - hidden from algorithm]")
    logger.info(f"  J1 = {TRUE_J1:.2f} meV, J2 = {TRUE_J2:.2f} meV, D = {TRUE_D:.2f} meV")
    
    # Step 1: Define structure
    structure = create_toy_structure()
    
    # Step 2: Generate candidate hypotheses (GNN or heuristic)
    candidates = generate_hypotheses(structure)
    if args.equal_priors:
        uniform = 1.0 / len(candidates)
        for cand in candidates:
            cand['prior'] = uniform
    
    measurements: List[Dict] = []
    measurement_plan: List[Dict] = []
    results = None
    superset_model = None
    checkpoint_dir = None
    demo_hmin = H_RANGE_MIN if args.demo_hmin is None else args.demo_hmin
    demo_hmax = H_RANGE_MAX if args.demo_hmax is None else args.demo_hmax

    if args.phase_planner:
        phase2_limit = max(args.phase2_limit, 5)
        phase_thresholds = (5, phase2_limit)
        phase3_threshold = args.phase3_threshold
        total_limit = args.total_measurements
        if total_limit is None:
            total_limit = max(phase2_limit + 5, 25)
        total_limit = max(total_limit, phase2_limit)

        planned_points: List[Dict] = []
        posterior_hint = [cand['prior'] for cand in candidates]
        best_model = None
        llm_state: Dict[str, Any] = {}
        llm_batch_idx = 0
        checkpoint_dir = Path(
            args.checkpoint_dir
            or args.llm_output_dir
            or str(FIGURES_DIR)
        )

        if args.resume_from:
            payload = load_checkpoint(Path(args.resume_from))
            measurements = payload["measurements"]
            measurement_plan = payload["measurement_plan"]
            planned_points = payload["planned_points"]
            llm_state = payload["llm_state"]
            llm_batch_idx = payload["llm_batch_idx"]
            logger.info("Resumed from checkpoint: %s", args.resume_from)

        def append_points(points: List[Dict]):
            measurement_plan.extend(points)
            planned_points.extend(points)

        def take_measurements(points: List[Dict]):
            append_points(points)
            new_data = simulate_measurements(points, true_model)
            measurements.extend(new_data)

        def run_discrimination():
            nonlocal results, posterior_hint, best_model, superset_model
            if not measurements:
                return
            use_bumps_now = use_bumps_default and (
                (len(measurements) % bumps_interval == 0) or (len(measurements) >= total_limit)
            )
            bumps_mp = args.bumps_mp if args.bumps_mp and args.bumps_mp > 0 else None
            bumps_pop = args.bumps_pop if args.bumps_pop and args.bumps_pop > 0 else None
            branch_weight_dict = compute_branch_weights(measurements) if args.branch_priors else None
            branch_weight_dict = apply_gap_focus(branch_weight_dict, args.gap_only_evidence)
            weight_arr = build_measurement_weight_array(measurements, branch_weight_dict)
            results = discriminate_models(
                measurements,
                candidates,
                measurement_weights=weight_arr,
                use_partial_mcmc=args.partial_mcmc,
                branch_weight_dict=branch_weight_dict,
                use_bumps=use_bumps_now,
                fit_background=args.fit_background,
                lock_gapless_background=args.lock_gapless_background,
                clamp_measurements=args.physics_use_loggp_clamp,
                bumps_mp=bumps_mp,
                bumps_pop=bumps_pop
            )
            posterior_hint = [results[c['name']]['posterior'] for c in candidates]
            best_name = max(results.items(), key=lambda x: x[1]['posterior'])[0]
            best_model = results[best_name]['fitted_model']
            plausible = [(name, res) for name, res in results.items() if res['posterior'] >= 0.05]
            if not plausible:
                plausible = list(results.items())
            superset_choice = max(plausible, key=lambda x: x[1]['n_params'])
            if args.projected_fisher:
                superset_cand = next(c for c in candidates if c['name'] == superset_choice[0])
                superset_model = SquareLatticeDispersion(**superset_cand['params'])
            else:
                fitted = superset_choice[1]['fitted_model']
                superset_model = SquareLatticeDispersion(J1=fitted.J1, J2=fitted.J2, D=fitted.D, S=fitted.S)

        def llm_enabled(phase: str) -> bool:
            return args.llm_in_loop and args.llm_decider != "none" and (args.llm_phase in (phase, "all"))

        def maybe_llm_phase0_injection(note: str) -> bool:
            """Run a phase0 LLM injection and optionally one-shot exit."""
            nonlocal llm_batch_idx
            if not llm_enabled("phase0"):
                return False
            llm_points = maybe_get_llm_points(
                measurements,
                output_dir=Path(args.llm_output_dir),
                max_points=max(args.llm_max_points, 1),
                history=max(args.llm_history, 1),
                decider=args.llm_decider,
                rotate_state=llm_state,
                dry_run=args.llm_dry_run,
                external=args.llm_external,
                mailbox_url=args.llm_mailbox_url,
                mailbox_token=args.llm_mailbox_token,
                mailbox_run_id=args.llm_mailbox_run_id,
                wait_seconds=args.llm_wait_seconds,
                batch_idx=llm_batch_idx
            )
            if llm_points:
                if args.llm_one_shot:
                    llm_points = llm_points[:1]
                logger.info("Injecting %d LLM-suggested points (phase0)", len(llm_points))
                take_measurements(llm_points)
                run_discrimination()
                save_checkpoint(
                    checkpoint_dir,
                    measurements,
                    measurement_plan,
                    planned_points,
                    llm_state,
                    llm_batch_idx,
                    note=note
                )
                llm_batch_idx += 1
                if args.llm_one_shot:
                    return True
            return False

        # Phase 1: mandatory symmetry seeds + optional coverage
        if args.hybrid_loggp and not measurements:
            logger.info("\nPhase 0: Log-GP agnostic exploration (%d measurements)", args.loggp_measurements)
            seed_measurements = None
            if args.loggp_init_from:
                payload = json.loads(Path(args.loggp_init_from).read_text())
                if isinstance(payload, dict) and "measurements" in payload:
                    seed_measurements = payload["measurements"]
                elif isinstance(payload, list):
                    seed_measurements = payload
                if seed_measurements:
                    measurements = [dict(m) for m in seed_measurements]
                    measurement_plan = measurements.copy()
                    planned_points = measurements.copy()

            if args.loggp_pre_symmetry and measurements:
                pre_sym = create_symmetry_seed_points(candidates, total_budget=5,
                                                      fraction=1.0, min_points=5)
                if pre_sym:
                    logger.info("\nPre-LogGP symmetry seeding: %d points", len(pre_sym))
                    take_measurements(pre_sym)
                    run_discrimination()
                    save_checkpoint(
                        checkpoint_dir,
                        measurements,
                        measurement_plan,
                        planned_points,
                        llm_state,
                        llm_batch_idx,
                        note="after_loggp_pre_symmetry"
                    )

            if measurements:
                seed_measurements = measurements

            loggp_meas, loggp_init_count, loggp_first_active = run_loggp_phase(
                true_model,
                n_measurements=max(1, args.loggp_measurements),
                hmin=demo_hmin,
                hmax=demo_hmax,
                emin=0.5,
                emax=LOGGP_E_MAX,
                grid_h=args.loggp_grid_h,
                grid_e=args.loggp_grid_e,
                seed_measurements=seed_measurements,
                taper_mode=args.loggp_taper_mode,
            )
            if seed_measurements:
                measurements = loggp_meas
                measurement_plan = loggp_meas.copy()
                planned_points = loggp_meas.copy()
            else:
                measurements.extend(loggp_meas)
                measurement_plan.extend(loggp_meas)
                planned_points.extend(loggp_meas)
            if loggp_init_count:
                save_checkpoint(
                    checkpoint_dir,
                    measurements[:loggp_init_count],
                    measurement_plan[:loggp_init_count],
                    planned_points[:loggp_init_count],
                    llm_state,
                    llm_batch_idx,
                    note="after_loggp_init"
                )
            if loggp_first_active is not None:
                save_checkpoint(
                    checkpoint_dir,
                    measurements[:loggp_first_active],
                    measurement_plan[:loggp_first_active],
                    planned_points[:loggp_first_active],
                    llm_state,
                    llm_batch_idx,
                    note="after_loggp_kernel"
                )
            run_discrimination()
            save_checkpoint(
                checkpoint_dir,
                measurements,
                measurement_plan,
                planned_points,
                llm_state,
                llm_batch_idx,
                note="after_loggp"
            )
            if maybe_llm_phase0_injection("after_llm_phase0"):
                return
        elif measurements and args.hybrid_loggp:
            # Resumed with existing measurements (e.g., log-GP init grid)
            if maybe_llm_phase0_injection("after_llm_phase0_resume"):
                return

        phase1_points: List[Dict] = []
        if not measurements and not args.disable_symmetry_seed:
            phase1_points = create_symmetry_seed_points(candidates, total_budget=5,
                                                        fraction=1.0, min_points=5)
            logger.info("\nPhase 1 seeding: %d high-symmetry measurements", len(phase1_points))

        if args.force_zone_coverage:
            coverage_points = create_zone_coverage_points(true_model)
            if coverage_points:
                logger.info("\nSeeding %d coverage measurements near zone center/edge", len(coverage_points))
                phase1_points.extend(coverage_points)

        if args.symmetry_seed:
            if args.symmetry_seed_count and args.symmetry_seed_count > 0:
                extra_sym = create_symmetry_seed_points(
                    candidates,
                    total_budget=args.symmetry_seed_count,
                    fraction=1.0,
                    min_points=args.symmetry_seed_count
                )
            else:
                extra_sym = create_symmetry_seed_points(candidates, total_budget=5)
            phase1_points.extend(extra_sym)

        if phase1_points:
            take_measurements(phase1_points)
            run_discrimination()
            save_checkpoint(
                checkpoint_dir,
                measurements,
                measurement_plan,
                planned_points,
                llm_state,
                llm_batch_idx,
                note="after_phase1"
            )
            if llm_enabled("phase1"):
                llm_points = maybe_get_llm_points(
                    measurements,
                    output_dir=Path(args.llm_output_dir),
                    max_points=max(args.llm_max_points, 1),
                    history=max(args.llm_history, 1),
                    decider=args.llm_decider,
                    rotate_state=llm_state,
                    dry_run=args.llm_dry_run,
                    external=args.llm_external,
                    mailbox_url=args.llm_mailbox_url,
                    mailbox_token=args.llm_mailbox_token,
                    mailbox_run_id=args.llm_mailbox_run_id,
                    wait_seconds=args.llm_wait_seconds,
                    batch_idx=llm_batch_idx
                )
                if llm_points:
                    logger.info("Injecting %d LLM-suggested points (phase1)", len(llm_points))
                    take_measurements(llm_points)
                    run_discrimination()
                    save_checkpoint(
                        checkpoint_dir,
                        measurements,
                        measurement_plan,
                        planned_points,
                        llm_state,
                        llm_batch_idx,
                        note="after_llm_phase1"
                    )
                llm_batch_idx += 1
        elif measurements:
            run_discrimination()

        # Phase 2: disagreement hunter
        batch_size = 5
        while len(measurements) < phase2_limit and len(measurements) < total_limit:
            needed = min(batch_size, phase2_limit - len(measurements))
            if needed <= 0:
                break
            new_points = plan_measurements(
                candidates,
                n_points=needed,
                force_zone_coverage=args.force_zone_coverage,
                adaptive_coverage=args.adaptive_coverage,
                existing_points=planned_points,
                measurement_history=measurements,
                use_jsd=True,
                phase_thresholds=phase_thresholds,
                posterior_hint=posterior_hint,
                precision_model=superset_model,
                enable_phases=True,
                posterior_phase3_threshold=phase3_threshold,
                force_phase=None,
                use_projected_fisher=args.projected_fisher,
                dwell_multiplier=(args.phase3_dwell_multiplier if args.phase3_dwell else 1.0),
                hmin=demo_hmin,
                hmax=demo_hmax
            )
            for pt in new_points:
                pt.setdefault('mode', 'physics')
            take_measurements(new_points)
            run_discrimination()
            save_checkpoint(
                checkpoint_dir,
                measurements,
                measurement_plan,
                planned_points,
                llm_state,
                llm_batch_idx,
                note="after_phase2_batch"
            )
            if llm_enabled("phase2"):
                if llm_batch_idx % max(args.llm_cadence, 1) == 0:
                    llm_points = maybe_get_llm_points(
                        measurements,
                        output_dir=Path(args.llm_output_dir),
                        max_points=max(args.llm_max_points, 1),
                        history=max(args.llm_history, 1),
                        decider=args.llm_decider,
                        rotate_state=llm_state,
                        dry_run=args.llm_dry_run,
                        external=args.llm_external,
                        mailbox_url=args.llm_mailbox_url,
                        mailbox_token=args.llm_mailbox_token,
                        mailbox_run_id=args.llm_mailbox_run_id,
                        wait_seconds=args.llm_wait_seconds,
                        batch_idx=llm_batch_idx
                    )
                    if llm_points:
                        logger.info("Injecting %d LLM-suggested points (phase2)", len(llm_points))
                        take_measurements(llm_points)
                        run_discrimination()
                        save_checkpoint(
                            checkpoint_dir,
                            measurements,
                            measurement_plan,
                            planned_points,
                            llm_state,
                            llm_batch_idx,
                            note="after_llm_phase2"
                        )
                llm_batch_idx += 1
            if results and max(posterior_hint) > 0.95:
                break

        # Phase 3: precision gatherer
        while len(measurements) < total_limit:
            needed = min(batch_size, total_limit - len(measurements))
            if needed <= 0:
                break
            precision_model = superset_model
            if precision_model is None:
                precision_model = SquareLatticeDispersion(**candidates[-1]['params'])

            new_points = plan_measurements(
                candidates,
                n_points=needed,
                force_zone_coverage=False,
                adaptive_coverage=args.adaptive_coverage,
                existing_points=planned_points,
                measurement_history=measurements,
                use_jsd=False,
                phase_thresholds=phase_thresholds,
                posterior_hint=posterior_hint,
                precision_model=precision_model,
                enable_phases=True,
                posterior_phase3_threshold=phase3_threshold,
                force_phase=3,
                use_projected_fisher=args.projected_fisher,
                dwell_multiplier=(args.phase3_dwell_multiplier if args.phase3_dwell else 1.0),
                hmin=demo_hmin,
                hmax=demo_hmax
            )
            for pt in new_points:
                pt.setdefault('mode', 'physics')
            take_measurements(new_points)
            run_discrimination()
            save_checkpoint(
                checkpoint_dir,
                measurements,
                measurement_plan,
                planned_points,
                llm_state,
                llm_batch_idx,
                note="after_phase3_batch"
            )

            if llm_enabled("phase3"):
                if llm_batch_idx % max(args.llm_cadence, 1) == 0:
                    llm_points = maybe_get_llm_points(
                        measurements,
                        output_dir=Path(args.llm_output_dir),
                        max_points=max(args.llm_max_points, 1),
                        history=max(args.llm_history, 1),
                        decider=args.llm_decider,
                        rotate_state=llm_state,
                        dry_run=args.llm_dry_run,
                        external=args.llm_external,
                        mailbox_url=args.llm_mailbox_url,
                        mailbox_token=args.llm_mailbox_token,
                        mailbox_run_id=args.llm_mailbox_run_id,
                        wait_seconds=args.llm_wait_seconds,
                        batch_idx=llm_batch_idx
                    )
                    if llm_points:
                        logger.info("Injecting %d LLM-suggested points", len(llm_points))
                        take_measurements(llm_points)
                        run_discrimination()
                        save_checkpoint(
                            checkpoint_dir,
                            measurements,
                            measurement_plan,
                            planned_points,
                            llm_state,
                            llm_batch_idx,
                            note="after_llm_injection"
                        )
                llm_batch_idx += 1

        if results is None:
            run_discrimination()

    else:
        base_points = 15

        coverage_points = []
        if args.force_zone_coverage:
            coverage_points = create_zone_coverage_points(true_model)
            if coverage_points:
                logger.info("\nSeeding %d coverage measurements near zone center/edge", len(coverage_points))

        symmetry_points: List[Dict] = []
        if args.symmetry_seed:
            symmetry_points = create_symmetry_seed_points(candidates, base_points)
            if symmetry_points:
                logger.info("\nReserving %d symmetry-point measurements (Γ/X/M)", len(symmetry_points))

        seeded_points = coverage_points + symmetry_points
        n_plan_points = max(0, base_points - len(seeded_points))

        measurement_plan = seeded_points + plan_measurements(
            candidates,
            n_points=n_plan_points,
            force_zone_coverage=args.force_zone_coverage,
            adaptive_coverage=args.adaptive_coverage,
            existing_points=seeded_points,
            measurement_history=measurements,
            use_jsd=args.jsd_acquisition,
            phase_thresholds=(5, 30),
            posterior_hint=None,
            precision_model=None,
            enable_phases=False
        )

        human_hint_points: List[Dict] = []
        if args.human_gap_hints:
            human_hint_points = create_human_gap_hint_points(candidates)
            if human_hint_points:
                insert_at = len(measurement_plan) if args.human_hint_after < 0 else max(
                    0, min(args.human_hint_after, len(measurement_plan))
                )
                logger.info(
                    "\nInjecting %d human-in-the-loop gap hints after measurement %d",
                    len(human_hint_points),
                    insert_at
                )
                measurement_plan[insert_at:insert_at] = human_hint_points

        measurements = simulate_measurements(measurement_plan, true_model)

        branch_weight_dict = compute_branch_weights(measurements) if args.branch_priors else None
        branch_weight_dict = apply_gap_focus(branch_weight_dict, args.gap_only_evidence)
        measurement_weight_arr = build_measurement_weight_array(measurements, branch_weight_dict)
        results = discriminate_models(
            measurements,
            candidates,
            measurement_weights=measurement_weight_arr,
            use_partial_mcmc=args.partial_mcmc,
            branch_weight_dict=branch_weight_dict,
            use_bumps=use_bumps_default,
            fit_background=args.fit_background,
            lock_gapless_background=args.lock_gapless_background,
            clamp_measurements=args.physics_use_loggp_clamp
        )
    
    # Step 6: Report winner
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    # Persist final state so post-run analysis includes TAS-AI batches.
    if checkpoint_dir is not None:
        save_checkpoint(
            checkpoint_dir,
            measurements,
            measurement_plan,
            planned_points,
            llm_state,
            llm_batch_idx,
            note="final"
        )
    
    winner = max(results.items(), key=lambda x: x[1]['posterior'])
    winner_cand = next(c for c in candidates if c['name'] == winner[0])
    
    logger.info(f"\nBest model: {winner[0]}")
    logger.info(f"Posterior probability: {winner[1]['posterior']:.1%}")
    logger.info(f"\nValidated parameters:")
    for p, v in winner_cand['params'].items():
        if v != 0:
            err = winner_cand['uncertainties'][p]
            logger.info(f"  {p} = {v:.2f} ± {err:.2f} meV")
    
    logger.info(f"\n[Comparison to ground truth]")
    logger.info(f"  True: J1={TRUE_J1:.2f}, J2={TRUE_J2:.2f}, D={TRUE_D:.2f}")
    logger.info(f"  Est:  J1={winner_cand['params']['J1']:.2f}, "
               f"J2={winner_cand['params']['J2']:.2f}, "
               f"D={winner_cand['params']['D']:.2f}")
    
    if args.no_plots:
        logger.info("\nSkipping figure generation (--no-plots).")
    else:
        # Create visualizations
        create_visualizations(structure, candidates, measurements, results, true_model)

        # Create posterior evolution figure (showing discrimination from 5→15 measurements)
        create_posterior_evolution_figure(candidates, measurements, true_model)

        # Create animation (optional)
        if not args.no_animations:
            create_evolution_animation(candidates, measurements, true_model)

    logger.info("\n" + "="*60)
    logger.info("Demo complete!")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info("="*60)
    
    return {
        'structure': structure,
        'candidates': candidates,
        'measurements': measurements,
        'results': results,
        'winner': winner[0]
    }


if __name__ == '__main__':
    main()

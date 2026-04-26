#!/usr/bin/env python3
"""Reviewer-facing sensitivity and aggregate analyses for items 2-5.

This script is intentionally paper-side: it consumes the current library and
paper code paths without changing the online controller implementations.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASAI_ROOT = ROOT.parent / "tasai"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASAI_ROOT))

from tasai.examples.example_model_discrimination import ModelDiscriminator  # type: ignore
from tasai.examples.example_parameter_determination import (  # type: ignore
    BenchmarkConfig,
    GridPolicy,
    LogGPPolicy,
    MeasurementPolicy,
    MotionAwareAcquisition,
    ParameterEstimator,
    RandomPolicy,
    run_policy,
)


OUTPUT = ROOT / "paper" / "data" / "reviewer_sensitivity_20260403.json"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def summarize_numeric(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"median": None, "iqr": None, "mean": None}
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "median": float(statistics.median(values)),
        "iqr": float(q3 - q1),
        "mean": float(statistics.mean(values)),
    }


def _finite_difference_hessian(func, x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Symmetric finite-difference Hessian for a scalar objective."""
    x = np.array(x, dtype=float)
    steps = np.array(steps, dtype=float)
    n = len(x)
    hessian = np.zeros((n, n), dtype=float)
    f0 = float(func(x))

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = steps[i]
        f_plus = float(func(x + ei))
        f_minus = float(func(x - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = steps[j]
            f_pp = float(func(x + ei + ej))
            f_pm = float(func(x + ei - ej))
            f_mp = float(func(x - ei + ej))
            f_mm = float(func(x - ei - ej))
            mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed

    return hessian


def _laplace_summary_from_estimator(estimator: ParameterEstimator) -> Dict:
    """Approximate marginal parameter uncertainties from the final chi^2 curvature."""
    if len(estimator.measurements) < 4:
        raise ValueError("Need at least four measurements for a curvature estimate")

    H = np.array([m.H for m in estimator.measurements], dtype=float)
    E = np.array([m.E for m in estimator.measurements], dtype=float)
    I = np.array([m.I for m in estimator.measurements], dtype=float)
    sigma = np.array([m.sigma for m in estimator.measurements], dtype=float)

    bounds = np.array(estimator.param_bounds, dtype=float)
    theta_hat = np.array([estimator.est_model.J1, estimator.est_model.J2, estimator.est_model.D], dtype=float)

    def chi2_for(params: np.ndarray) -> float:
        clipped = np.clip(np.array(params, dtype=float), bounds[:, 0], bounds[:, 1])
        model = estimator.est_model.__class__(J1=float(clipped[0]), J2=float(clipped[1]), D=float(clipped[2]))
        return float(model.chi_squared(H, np.zeros_like(H), E, I, sigma))

    # Stay comfortably inside the box constraints for symmetric differences.
    max_steps = 0.45 * np.minimum(theta_hat - bounds[:, 0], bounds[:, 1] - theta_hat)
    base_steps = np.maximum(0.01 * np.maximum(np.abs(theta_hat), 1.0), 1e-3)
    steps = np.minimum(base_steps, np.maximum(max_steps, 1e-4))
    hessian = _finite_difference_hessian(chi2_for, theta_hat, steps)

    try:
        cov = 2.0 * np.linalg.pinv(hessian, hermitian=True)
    except TypeError:
        cov = 2.0 * np.linalg.pinv(hessian)
    cov = 0.5 * (cov + cov.T)

    diag = np.clip(np.diag(cov), 0.0, None)
    stds = np.sqrt(diag)
    return {
        "theta_hat": {
            "J1": float(theta_hat[0]),
            "J2": float(theta_hat[1]),
            "D": float(theta_hat[2]),
        },
        "stds": {
            "J1": float(stds[0]),
            "J2": float(stds[1]),
            "D": float(stds[2]),
        },
        "hessian_condition_number": float(np.linalg.cond(hessian)),
    }


class EtaParameterEstimator(ParameterEstimator):
    """Parameter estimator with configurable motion exponent eta."""

    def __init__(self, config: BenchmarkConfig, eta: float):
        super().__init__(config=config)
        self.eta = float(eta)

    def suggest_tas_point(self) -> Tuple[float, float]:
        H_candidates = np.linspace(self.H_range[0], self.H_range[1], 25)
        candidates: List[Tuple[float, float]] = []
        infos: List[float] = []
        for H in H_candidates:
            E0 = self.est_model.dispersion(H, 0.0)
            for dE in (-0.8, -0.3, 0, 0.3, 0.8):
                E = E0 + dE
                if not (self.E_range[0] < E < self.E_range[1]):
                    continue
                info = self._info_gain_heuristic(H, E) * self._novelty(H, E)
                candidates.append((float(H), float(E)))
                infos.append(float(info))

        if not candidates:
            return 0.25, 15.0

        acquisition = MotionAwareAcquisition(
            motor_model=self.motor,
            eta=self.eta,
            count_time=self.count_time,
        )
        cand_array = np.array([[H, H, 0.0, E] for H, E in candidates], dtype=float)
        scores = acquisition.score_batch(cand_array, np.array(infos, dtype=float))
        return candidates[int(np.argmax(scores))]


class EtaTASPolicy(MeasurementPolicy):
    def __init__(self):
        super().__init__("tas_ai")

    def select(self, estimator: ParameterEstimator, iteration: int) -> Tuple[float, float]:
        return estimator.suggest_tas_point()


def run_tas_policy_with_eta(config: BenchmarkConfig,
                            threshold: float,
                            max_measurements: int,
                            seed: int,
                            eta: float) -> Dict:
    np.random.seed(seed)
    estimator = EtaParameterEstimator(config=config, eta=eta)
    estimator.add_initial_points()
    policy = EtaTASPolicy()
    policy.reset(estimator)

    rms_history = estimator.rms_history.copy()
    time_history = estimator.time_history.copy()
    measurement_count = len(estimator.measurements)
    converge_iter = None
    converge_time_s = None

    while measurement_count < max_measurements:
        H, E = policy.select(estimator, measurement_count)
        estimator.measure(H, E, source=policy.name)
        estimator.fit_model()
        rms_history.append(estimator.current_rms_error())
        time_history.append(float(estimator.total_time_s))
        measurement_count += 1
        if rms_history[-1] <= threshold and converge_iter is None:
            converge_iter = measurement_count
            converge_time_s = float(estimator.total_time_s)

    return {
        "policy": "tas_ai",
        "history": rms_history,
        "time_history_s": time_history,
        "converged_at": converge_iter,
        "converged_time_s": converge_time_s,
        "final_rms": rms_history[-1],
        "final_time_s": float(estimator.total_time_s),
    }


def refinement_analysis() -> Dict:
    config = BenchmarkConfig(
        true_J1=5.3,
        true_J2=0.6,
        true_D=0.28,
        prior_J1=4.2,
        prior_J2=0.35,
        prior_D=0.12,
        init_points=3,
        init_mode="prior_triplet",
        count_time=10.0,
        use_motion=True,
    )
    threshold = 0.05
    max_measurements = 18
    seeds = list(range(3))
    coverage_seeds = list(range(10))

    policies = {
        "grid": GridPolicy(),
        "random": RandomPolicy(),
    }

    aggregate: Dict[str, Dict] = {
        "config": asdict(config),
        "threshold": threshold,
        "max_measurements": max_measurements,
        "seed_summaries": {},
        "eta_sensitivity": {},
    }

    for policy_name, policy in policies.items():
        rows = []
        for seed in seeds:
            np.random.seed(seed)
            res = run_policy(policy, max_measurements=max_measurements, threshold=threshold, config=config)
            rows.append({
                "seed": seed,
                "converged_at": res["converged_at"],
                "converged_time_s": res["converged_time_s"],
                "final_rms": float(res["final_rms"]),
                "final_time_s": float(res["final_time_s"]),
            })
        aggregate["seed_summaries"][policy_name] = rows

    tas_rows = []
    for seed in seeds:
        tas_rows.append({
            "seed": seed,
            **{
                k: v for k, v in run_tas_policy_with_eta(
                    config=config,
                    threshold=threshold,
                    max_measurements=max_measurements,
                    seed=seed,
                    eta=0.7,
                ).items()
                if k in {"converged_at", "converged_time_s", "final_rms", "final_time_s"}
            },
        })
    aggregate["seed_summaries"]["tas_ai"] = tas_rows

    for eta in (0.5, 0.7, 0.9):
        rows = []
        for seed in seeds:
            res = run_tas_policy_with_eta(
                config=config,
                threshold=threshold,
                max_measurements=max_measurements,
                seed=seed,
                eta=eta,
            )
            rows.append({
                "seed": seed,
                "converged_at": res["converged_at"],
                "converged_time_s": res["converged_time_s"],
                "final_rms": float(res["final_rms"]),
                "final_time_s": float(res["final_time_s"]),
            })
        aggregate["eta_sensitivity"][str(eta)] = rows

    z90 = 1.6448536269514722
    coverage_rows = []
    for seed in coverage_seeds:
        np.random.seed(seed)
        estimator = EtaParameterEstimator(config=config, eta=0.7)
        estimator.add_initial_points()
        policy = EtaTASPolicy()
        policy.reset(estimator)
        measurement_count = len(estimator.measurements)

        while measurement_count < max_measurements:
            H, E = policy.select(estimator, measurement_count)
            estimator.measure(H, E, source=policy.name)
            estimator.fit_model()
            measurement_count += 1

        laplace = _laplace_summary_from_estimator(estimator)
        row = {
            "seed": seed,
            "mean": laplace["theta_hat"],
            "std": laplace["stds"],
            "condition_number": laplace["hessian_condition_number"],
            "contains_truth_90": {},
            "interval_90": {},
        }
        truths = {
            "J1": config.true_J1,
            "J2": config.true_J2,
            "D": config.true_D,
        }
        for name, truth in truths.items():
            mean = row["mean"][name]
            std = row["std"][name]
            lo = float(mean - z90 * std)
            hi = float(mean + z90 * std)
            row["interval_90"][name] = [lo, hi]
            row["contains_truth_90"][name] = bool(lo <= truth <= hi)
        coverage_rows.append(row)

    aggregate["laplace_coverage_90"] = {
        "n_seeds": len(coverage_rows),
        "nominal_level": 0.90,
        "z_value": z90,
        "seed_rows": coverage_rows,
        "summary": {},
    }

    for name in ("J1", "J2", "D"):
        hits = [1.0 if row["contains_truth_90"][name] else 0.0 for row in coverage_rows]
        widths = [row["interval_90"][name][1] - row["interval_90"][name][0] for row in coverage_rows]
        stds = [row["std"][name] for row in coverage_rows]
        aggregate["laplace_coverage_90"]["summary"][name] = {
            "empirical_coverage": float(sum(hits) / len(hits)),
            "hit_count": int(sum(hits)),
            "n_seeds": len(hits),
            "median_interval_width": float(statistics.median(widths)),
            "mean_std": float(statistics.mean(stds)),
        }

    aggregate["summary"] = {}
    for name, rows in aggregate["seed_summaries"].items():
        conv_counts = [r["converged_at"] for r in rows if r["converged_at"] is not None]
        conv_times = [r["converged_time_s"] for r in rows if r["converged_time_s"] is not None]
        final_rms = [r["final_rms"] for r in rows]
        final_time = [r["final_time_s"] for r in rows]
        aggregate["summary"][name] = {
            "success_rate": f"{len(conv_counts)}/{len(rows)}",
            "converged_at": summarize_numeric(conv_counts),
            "converged_time_s": summarize_numeric(conv_times),
            "final_rms": summarize_numeric(final_rms),
            "final_time_s": summarize_numeric(final_time),
        }

    return aggregate


def run_discrimination_seed(seed: int,
                            include_j2: bool = True,
                            n_iterations: int = 8,
                            points_per_iter: int = 1) -> Dict:
    np.random.seed(seed)
    disc = ModelDiscriminator(include_j2=include_j2)
    disc.add_initial_points(6)
    disc.fit_models()

    decisive_at = None
    decisive_ratio = None
    target_key = "J1J2" if include_j2 else "NN"

    def maybe_update_decisive(n_meas: int):
        nonlocal decisive_at, decisive_ratio
        w_target = disc.weights[target_key]
        w_other = 1.0 - w_target
        ratio = math.inf if w_other <= 0 else w_target / max(w_other, 1e-12)
        if decisive_at is None and ratio > 100.0:
            decisive_at = n_meas
            decisive_ratio = ratio

    maybe_update_decisive(len(disc.measurements))
    for _ in range(n_iterations):
        for _ in range(points_per_iter):
            H, E = disc.suggest_next_point()
            disc.measure(H, E)
        disc.fit_models()
        maybe_update_decisive(len(disc.measurements))

    return {
        "seed": seed,
        "measurements": len(disc.measurements),
        "final_weights": disc.weights.copy(),
        "decisive_at": decisive_at,
        "decisive_ratio": decisive_ratio,
    }


def _gaussian_logpdf(y: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-12)
    return -0.5 * math.log(2.0 * math.pi * sigma * sigma) - 0.5 * ((y - mu) / sigma) ** 2


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.sum(weights * values)
    return float(np.sum(weights * (values - mean) ** 2))


def discrimination_waic(seed: int = 0) -> Dict:
    np.random.seed(seed)
    disc = ModelDiscriminator(include_j2=True)
    disc.add_initial_points(6)
    for _ in range(2):
        H, E = disc.suggest_next_point()
        disc.measure(H, E)
    disc.fit_models()

    measurements = disc.measurements
    H_data = np.array([m.H for m in measurements], dtype=float)
    L_data = np.array([m.L for m in measurements], dtype=float)
    E_data = np.array([m.E for m in measurements], dtype=float)
    I_data = np.array([m.I for m in measurements], dtype=float)
    sigma_data = np.array([m.sigma for m in measurements], dtype=float)

    model_specs = {
        "NN": [(J1, D) for J1 in np.linspace(2.0, 10.0, 20)
               for D in np.linspace(0.01, 0.3, 10)],
        "J1J2": [(J1, J2, D) for J1 in np.linspace(2.0, 10.0, 15)
                 for J2 in np.linspace(-0.5, 2.0, 12)
                 for D in np.linspace(0.01, 0.3, 8)],
    }

    waic_rows = {}
    for model_name, grid in model_specs.items():
        pointwise = []
        log_posteriors = []
        for theta in grid:
            if model_name == "NN":
                model = disc.model_nn.__class__(J1=theta[0], D=theta[1])
            else:
                model = disc.model_j1j2.__class__(J1=theta[0], J2=theta[1], D=theta[2])
            loglikes = np.array([
                _gaussian_logpdf(I, model.intensity(H, 0, E), sig)
                for H, E, I, sig in zip(H_data, E_data, I_data, sigma_data)
            ], dtype=float)
            pointwise.append(loglikes)
            log_posteriors.append(float(np.sum(loglikes)))
        pointwise_arr = np.array(pointwise)
        log_posteriors = np.array(log_posteriors, dtype=float)
        max_log = np.max(log_posteriors)
        weights = np.exp(log_posteriors - max_log)
        weights /= np.sum(weights)

        lppd = 0.0
        p_waic = 0.0
        for j in range(pointwise_arr.shape[1]):
            ll_j = pointwise_arr[:, j]
            max_j = np.max(ll_j)
            lppd += max_j + math.log(np.sum(weights * np.exp(ll_j - max_j)))
            p_waic += _weighted_var(ll_j, weights)
        waic = -2.0 * (lppd - p_waic)

        if model_name == "NN":
            aic = -2.0 * disc.log_evidence["NN"]
        else:
            aic = -2.0 * disc.log_evidence["J1J2"]

        waic_rows[model_name] = {
            "waic": float(waic),
            "aic": float(aic),
            "posterior_weight_from_aic": float(disc.weights["NN" if model_name == "NN" else "J1J2"]),
        }

    waic_values = np.array([waic_rows["NN"]["waic"], waic_rows["J1J2"]["waic"]], dtype=float)
    max_like = np.max(-0.5 * waic_values)
    waic_weights = np.exp(-0.5 * waic_values - max_like)
    waic_weights /= np.sum(waic_weights)
    waic_rows["NN"]["posterior_weight_from_waic"] = float(waic_weights[0])
    waic_rows["J1J2"]["posterior_weight_from_waic"] = float(waic_weights[1])

    return {
        "seed": seed,
        "n_measurements": len(measurements),
        "models": waic_rows,
    }


def discrimination_sensitivity() -> Dict:
    from simulations.toy_closed_loop import (  # type: ignore
        create_toy_structure,
        discriminate_models,
        generate_hypotheses,
        load_checkpoint,
    )
    import logging

    logging.getLogger().setLevel(logging.ERROR)

    checkpoint = load_checkpoint(
        ROOT / "paper" / "data" / "fig7_library_loggp_window_20260401_checkpoint.json"
    )
    measurements = checkpoint["measurements"]
    seeds = list(range(3))
    candidates_default = generate_hypotheses(create_toy_structure())
    candidates_equal = generate_hypotheses(create_toy_structure())
    uniform = 1.0 / len(candidates_equal)
    for cand in candidates_equal:
        cand["prior"] = uniform

    variants = {
        "default": {
            "candidates": candidates_default,
            "fit_background": True,
            "lock_gapless_background": True,
        },
        "equal_priors": {
            "candidates": candidates_equal,
            "fit_background": True,
            "lock_gapless_background": True,
        },
        "unlock_gapless_background": {
            "candidates": candidates_default,
            "fit_background": True,
            "lock_gapless_background": False,
        },
    }

    variant_results: Dict[str, Dict] = {}
    sampled_prefixes = sorted(set([
        min(40, len(measurements)),
        min(60, len(measurements)),
        len(measurements),
    ]))

    for name, spec in variants.items():
        rows = []
        decisive_at = None
        for n in sampled_prefixes:
            results = discriminate_models(
                measurements[:n],
                spec["candidates"],
                use_partial_mcmc=False,
                use_bumps=False,
                use_dream=False,
                fit_background=spec["fit_background"],
                lock_gapless_background=spec["lock_gapless_background"],
            )
            ranked = sorted(results.items(), key=lambda item: item[1]["posterior"], reverse=True)
            top_name, top_res = ranked[0]
            second_name, second_res = ranked[1]
            ratio = float(top_res["posterior"] / max(second_res["posterior"], 1e-12))
            rows.append({
                "n_measurements": n,
                "leader": top_name,
                "leader_posterior": float(top_res["posterior"]),
                "second": second_name,
                "ratio_vs_second": ratio,
            })
            if decisive_at is None and top_name.startswith("M4") and ratio > 100.0:
                decisive_at = n
        variant_results[name] = {
            "decisive_at": decisive_at,
            "trajectory": rows,
        }

    seed_rows = [run_discrimination_seed(seed=s) for s in seeds]
    return {
        "seed_summary": seed_rows,
        "seed_summary_stats": {
            "success_rate": f"{sum(r['decisive_at'] is not None for r in seed_rows)}/{len(seed_rows)}",
            "decisive_at": summarize_numeric([r["decisive_at"] for r in seed_rows if r["decisive_at"] is not None]),
        },
        "closed_loop_variant_sensitivity": variant_results,
        "waic_check": discrimination_waic(seed=0),
    }


def implementation_details() -> Dict:
    return {
        "reconstruction_metric": {
            "formula": "sum(|I_pred - I_true| * I_true) / sum(I_true^2)",
            "source": "tasai/examples/benchmark_jcns.py::compute_reconstruction_error",
            "notes": "I_pred is reconstructed from raw measured intensities using inverse-distance interpolation over a 30x30 reference grid.",
        },
        "count_time": {
            "refinement_default_s": 10.0,
            "jointly_optimized_with_location": False,
            "source": "tasai/examples/example_time_aware_refinement.py + example_parameter_determination.py",
        },
        "mcts": {
            "core_defaults": {
                "n_simulations": 100,
                "exploration_constant": 1.41,
                "n_candidates": 20,
                "max_depth": "set to requested batch size",
                "rollout_depth": 3,
                "discount_factor": 0.95,
            },
            "source": "tasai/core/mcts.py",
        },
    }


def main() -> None:
    payload = {
        "refinement": refinement_analysis(),
        "discrimination": discrimination_sensitivity(),
        "implementation_details": implementation_details(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run closed-loop audit-layer ablations without using the live mailbox path."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from tasai.core.gaussian_process import AgnosticExplorer
except Exception:
    AgnosticExplorer = None  # type: ignore[assignment]

try:
    from toy_closed_loop import (
        BASE_HAS_RESOLUTION,
        H_RANGE_MIN,
        LOGGP_E_MAX,
        LOGGP_MOVE_OVERHEAD,
        LOGGP_MOVE_VE,
        LOGGP_MOVE_VH,
        SquareLatticeDispersion,
        create_toy_structure,
        discriminate_models,
        generate_hypotheses,
        init_tas,
        plan_measurements,
        run_loggp_phase,
        simulate_measurements,
        TAS,
        SimpleGaussianProcess,
    )
    _TOY_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    # The analytic ghost-optic and bilayer ablations can run without the full
    # toy TAS stack, which pulls in plotting dependencies on some remote hosts.
    BASE_HAS_RESOLUTION = False
    H_RANGE_MIN = 0.0
    LOGGP_E_MAX = 30.0
    LOGGP_MOVE_OVERHEAD = 0.0
    LOGGP_MOVE_VE = 1.0
    LOGGP_MOVE_VH = 1.0
    SquareLatticeDispersion = Any  # type: ignore[assignment]
    create_toy_structure = None  # type: ignore[assignment]
    discriminate_models = None  # type: ignore[assignment]
    generate_hypotheses = None  # type: ignore[assignment]
    init_tas = None  # type: ignore[assignment]
    plan_measurements = None  # type: ignore[assignment]
    run_loggp_phase = None  # type: ignore[assignment]
    simulate_measurements = None  # type: ignore[assignment]

    class _FallbackTAS:
        @staticmethod
        def is_accessible(h: float, k: float, e: float) -> bool:
            return math.isfinite(float(h)) and math.isfinite(float(k)) and math.isfinite(float(e))

    TAS = _FallbackTAS()  # type: ignore[assignment]

    class SimpleGaussianProcess:
        """Lightweight GP surrogate for analytic ablations."""

        def __init__(self, length_scale: float = 0.2, noise: float = 1.0):
            self.length_scale = float(length_scale)
            self.noise = float(noise)
            self.X_train: List[np.ndarray] = []
            self.y_train: List[float] = []

        def add_observation(self, x: np.ndarray, y: float) -> None:
            self.X_train.append(np.asarray(x, dtype=float))
            self.y_train.append(float(y))

        def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
            d = np.sum((x1 - x2) ** 2)
            return float(np.exp(-d / (2.0 * self.length_scale**2)))

        def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            X = np.asarray(X, dtype=float)
            if len(self.X_train) == 0:
                return np.zeros(len(X)), np.ones(len(X)) * 10.0

            X_train = np.array(self.X_train, dtype=float)
            y_train = np.array(self.y_train, dtype=float)
            n = len(X_train)
            K = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    K[i, j] = self._kernel(X_train[i], X_train[j])
            K += self.noise * np.eye(n)
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))

            means: List[float] = []
            variances: List[float] = []
            for x in X:
                k_star = np.array([self._kernel(x, xi) for xi in X_train], dtype=float)
                mean = float(k_star @ K_inv @ y_train)
                var = float(self._kernel(x, x) - k_star @ K_inv @ k_star)
                means.append(mean)
                variances.append(max(var, 1e-8))
            return np.array(means), np.array(variances)

    _TOY_IMPORT_ERROR = exc

logger = logging.getLogger("audit_ablation")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class _LibraryLogGPAdapter:
    """Small wrapper around the shared AgnosticExplorer for fixed candidate sets."""

    def __init__(self, bounds: np.ndarray, background: float = 0.01):
        if AgnosticExplorer is None:
            raise RuntimeError("tasai.core.AgnosticExplorer is unavailable")
        self.bounds = np.asarray(bounds, dtype=float)
        self.explorer = AgnosticExplorer(self.bounds, background=float(background), use_log_gp=True)

    def add_observation(self, x: Sequence[float], intensity: float, sigma: float) -> None:
        self.explorer.add_observation(np.asarray(x, dtype=float), float(intensity), float(sigma))

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if len(self.explorer.observations) >= 2:
            self.explorer.gp.fit()
        return self.explorer.gp.predict_batch(X)


def _estimate_measurement_sigma(mu: float, count_time: float = 60.0) -> float:
    mu = float(max(mu, 0.0))
    counts = max(1.0, mu * float(count_time))
    sigma_poisson = math.sqrt(counts) / float(count_time)
    sigma_syst = max(0.03 * mu, 0.02)
    sigma = float(math.sqrt(sigma_poisson**2 + sigma_syst**2))
    return max(sigma, 1e-4)


def _move_time(last_h: float, last_E: float, h: float, e: float) -> float:
    dh = abs(float(h) - float(last_h))
    d_e = abs(float(e) - float(last_E))
    return float(max(dh / LOGGP_MOVE_VH, d_e / LOGGP_MOVE_VE) + LOGGP_MOVE_OVERHEAD)


def _too_close(
    h: float,
    e: float,
    pts: Sequence[Dict[str, Any]],
    h_thresh: float = 0.06,
    e_thresh: float = 1.5,
) -> bool:
    for pt in pts:
        if abs(float(pt["h"]) - float(h)) < h_thresh and abs(float(pt["E"]) - float(e)) < e_thresh:
            return True
    return False


def _top_two_models(results: Dict[str, Any]) -> Tuple[str, str]:
    items = sorted(results.items(), key=lambda kv: -float(kv[1].get("posterior", 0.0)))
    if len(items) < 2:
        raise ValueError("Need at least two models")
    return items[0][0], items[1][0]


def _classify_probe_family(h: float, e: float) -> str:
    if h <= 0.65 and e <= 2.0:
        return "gap_check"
    if h <= 0.9:
        return "midzone_curvature"
    return "band_edge"


def _is_falsification_probe(h: float, e: float, z_score: float) -> bool:
    return bool(z_score >= 2.0 and (_classify_probe_family(h, e) in {"gap_check", "midzone_curvature"}))


def build_discrimination_menu(
    results: Dict[str, Any],
    measurements: List[Dict[str, Any]],
    h_bounds: Tuple[float, float],
    e_bounds: Tuple[float, float],
    menu_size: int = 12,
    max_candidates: int = 900,
    count_time: float = 60.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    best_name, second_name = _top_two_models(results)
    m_best: SquareLatticeDispersion = results[best_name]["fitted_model"]
    m_second: SquareLatticeDispersion = results[second_name]["fitted_model"]

    h_vals = np.linspace(h_bounds[0], h_bounds[1], 29)
    cand: List[Tuple[float, float]] = []
    for h in h_vals:
        o1 = float(m_best.omega(h, h))
        o2 = float(m_second.omega(h, h))
        centers = sorted({o1, o2})
        for center in centers:
            for delta in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
                e = center + delta
                if e_bounds[0] <= e <= e_bounds[1]:
                    cand.append((float(h), float(e)))
        for e in (0.6, 1.0, 1.5, 2.0):
            if e_bounds[0] <= e <= e_bounds[1]:
                cand.append((float(h), float(e)))

    cand = list({(round(h, 4), round(e, 4)) for h, e in cand})
    if len(cand) > max_candidates:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cand), size=max_candidates, replace=False)
        cand = [cand[int(i)] for i in idx]

    last_h = float(measurements[-1]["h"]) if measurements else float(h_bounds[0])
    last_e = float(measurements[-1]["E"]) if measurements else float(e_bounds[0])

    scored: List[Dict[str, Any]] = []
    for h, e in cand:
        if not TAS.is_accessible(h, h, e):
            continue
        if measurements and _too_close(h, e, measurements):
            continue
        sigma_e = TAS.get_energy_resolution(h, h, e)
        mu_best = float(m_best.intensity(h, h, e, sigma_E=sigma_e))
        mu_second = float(m_second.intensity(h, h, e, sigma_E=sigma_e))
        if max(mu_best, mu_second) < 1.0:
            continue
        s1 = _estimate_measurement_sigma(mu_best, count_time=count_time)
        s2 = _estimate_measurement_sigma(mu_second, count_time=count_time)
        denom = float(math.sqrt(s1**2 + s2**2))
        z_score = float(abs(mu_best - mu_second) / max(denom, 1e-6))
        move_t = _move_time(last_h, last_e, h, e)
        utility = float(z_score / (1.0 + move_t))
        probe_family = _classify_probe_family(h, e)
        scored.append(
            {
                "h": float(h),
                "k": float(h),
                "E": float(e),
                "score": utility,
                "z": z_score,
                "move_time": move_t,
                "mu_best": mu_best,
                "mu_second": mu_second,
                "sigma_E": float(sigma_e),
                "best_model": best_name,
                "second_model": second_name,
                "probe_family": probe_family,
                "is_falsification_probe": _is_falsification_probe(h, e, z_score),
                "expected_contrast": float(abs(mu_best - mu_second)),
            }
        )

    scored.sort(key=lambda row: (-float(row["score"]), -float(row["z"])))
    menu: List[Dict[str, Any]] = []
    for row in scored:
        if len(menu) >= menu_size:
            break
        if _too_close(row["h"], row["E"], menu):
            continue
        menu.append(row)

    menu_lookup: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(menu):
        pid = f"D{idx:02d}"
        row = dict(row)
        row["id"] = pid
        menu_lookup[pid] = row
        menu[idx] = row
    return menu, menu_lookup


def build_all_model_disagreement_menu(
    results: Dict[str, Any],
    measurements: Sequence[Dict[str, Any]],
    h_bounds: Tuple[float, float],
    e_bounds: Tuple[float, float],
    menu_size: int = 12,
    max_candidates: int = 250,
    count_time: float = 60.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ranked = _model_rank(results)
    if len(ranked) < 2:
        return [], {}

    fitted_models: Dict[str, SquareLatticeDispersion] = {}
    for name, _ in ranked:
        model = results.get(name, {}).get("fitted_model")
        if model is not None:
            fitted_models[name] = model
    if len(fitted_models) < 2:
        return [], {}

    leader_name = ranked[0][0]
    if leader_name not in fitted_models:
        return [], {}
    leader_model = fitted_models[leader_name]

    h_vals = np.linspace(h_bounds[0], h_bounds[1], 29)
    cand: List[Tuple[float, float]] = []
    for h in h_vals:
        centers = set()
        for model in fitted_models.values():
            centers.add(float(model.omega(h, h)))
        for center in sorted(centers):
            for delta in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
                e = center + delta
                if e_bounds[0] <= e <= e_bounds[1]:
                    cand.append((float(h), float(e)))
        for e in (0.6, 1.0, 1.5, 2.0):
            if e_bounds[0] <= e <= e_bounds[1]:
                cand.append((float(h), float(e)))

    cand = list({(round(h, 4), round(e, 4)) for h, e in cand})
    if len(cand) > max_candidates:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cand), size=max_candidates, replace=False)
        cand = [cand[int(i)] for i in idx]

    last_h = float(measurements[-1]["h"]) if measurements else float(h_bounds[0])
    last_e = float(measurements[-1]["E"]) if measurements else float(e_bounds[0])
    scored: List[Dict[str, Any]] = []
    for h, e in cand:
        if not TAS.is_accessible(h, h, e):
            continue
        if measurements and _too_close(h, e, measurements):
            continue
        sigma_e = TAS.get_energy_resolution(h, h, e)
        leader_mu = float(leader_model.intensity(h, h, e, sigma_E=sigma_e))
        if leader_mu < 1.0:
            continue
        best_other_name: Optional[str] = None
        best_other_mu = 0.0
        best_delta = -1.0
        for name, model in fitted_models.items():
            if name == leader_name:
                continue
            mu = float(model.intensity(h, h, e, sigma_E=sigma_e))
            delta = abs(leader_mu - mu)
            if delta > best_delta:
                best_delta = delta
                best_other_name = name
                best_other_mu = mu
        if best_other_name is None or max(leader_mu, best_other_mu) < 1.0:
            continue
        s1 = _estimate_measurement_sigma(leader_mu, count_time=count_time)
        s2 = _estimate_measurement_sigma(best_other_mu, count_time=count_time)
        denom = float(math.sqrt(s1**2 + s2**2))
        z_score = float(best_delta / max(denom, 1e-6))
        move_t = _move_time(last_h, last_e, h, e)
        utility = float(z_score / (1.0 + move_t))
        probe_family = _classify_probe_family(h, e)
        scored.append(
            {
                "h": float(h),
                "k": float(h),
                "E": float(e),
                "score": utility,
                "z": z_score,
                "move_time": move_t,
                "mu_best": leader_mu,
                "mu_second": best_other_mu,
                "sigma_E": float(sigma_e),
                "best_model": leader_name,
                "second_model": best_other_name,
                "probe_family": probe_family,
                "is_falsification_probe": _is_falsification_probe(h, e, z_score),
                "expected_contrast": float(best_delta),
            }
        )

    scored.sort(key=lambda row: (-float(row["score"]), -float(row["z"])))
    menu: List[Dict[str, Any]] = []
    for row in scored:
        if len(menu) >= menu_size:
            break
        if _too_close(row["h"], row["E"], menu):
            continue
        menu.append(row)

    menu_lookup: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(menu):
        pid = f"D{idx:02d}"
        row = dict(row)
        row["id"] = pid
        menu_lookup[pid] = row
        menu[idx] = row
    return menu, menu_lookup


def _model_rank(results: Dict[str, Any]) -> List[Tuple[str, float]]:
    return sorted(
        ((name, float(data.get("posterior", 0.0))) for name, data in results.items()),
        key=lambda item: -item[1],
    )


def sanitize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    finite_scores: List[Tuple[str, float]] = []
    for name, row in results.items():
        clean = dict(row)
        log_post = float(clean.get("log_posterior", float("-inf")))
        aic = float(clean.get("aic", float("inf")))
        chi2 = float(clean.get("chi2", float("inf")))
        if not math.isfinite(log_post):
            log_post = float("-inf")
        if not math.isfinite(aic):
            aic = float("inf")
        if not math.isfinite(chi2):
            chi2 = float("inf")
        clean["log_posterior"] = log_post
        clean["aic"] = aic
        clean["chi2"] = chi2
        clean["posterior"] = 0.0
        safe[name] = clean
        if math.isfinite(log_post):
            finite_scores.append((name, log_post))

    if finite_scores:
        max_log = max(score for _, score in finite_scores)
        weights = {name: math.exp(score - max_log) for name, score in finite_scores}
        total = sum(weights.values())
        if total > 0:
            for name, weight in weights.items():
                safe[name]["posterior"] = float(weight / total)
    else:
        n = max(len(safe), 1)
        for name in safe:
            safe[name]["posterior"] = 1.0 / n
    return safe


def _posterior_entropy(results: Dict[str, Any]) -> float:
    posts = [max(float(data.get("posterior", 0.0)), 1e-12) for data in results.values()]
    total = sum(posts)
    if total <= 0:
        return 0.0
    probs = [p / total for p in posts]
    return float(-sum(p * math.log(p) for p in probs))


def _gap_region_hit(meas: Dict[str, Any]) -> bool:
    return float(meas["h"]) <= 0.65 and float(meas["E"]) <= 2.0


def _coverage_ratio(measurements: Sequence[Dict[str, Any]]) -> float:
    if not measurements:
        return 0.0
    hits = sum(1 for meas in measurements if _gap_region_hit(meas))
    return float(hits / len(measurements))


def _wrong_leader_state(history: Sequence[Dict[str, Any]], true_model_name: str) -> Tuple[bool, int]:
    saw_wrong = any(row["leader"] != true_model_name for row in history)
    dwell = sum(int(row["batch_size"]) for row in history if row["leader"] != true_model_name)
    return saw_wrong, dwell


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _salvage_mailbox_decision(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    suggestions = response.get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        first = suggestions[0]
        if isinstance(first, dict):
            if "decision" in first and isinstance(first["decision"], dict):
                return first["decision"]
            return first

    decision = response.get("decision", response)
    if isinstance(decision, dict):
        return decision

    meta = response.get("meta", {})
    raw = meta.get("raw", {}) if isinstance(meta, dict) else {}
    for value in raw.values() if isinstance(raw, dict) else []:
        if not isinstance(value, str):
            continue
        parsed = _extract_json(value)
        if isinstance(parsed, dict):
            return parsed.get("decision", parsed) if isinstance(parsed, dict) else None
        ids = re.findall(r"\"(D\d+)\"", value)
        if ids or "reason" in value or "inject_ids" in value:
            reason_match = re.search(r'reason\s*:\s*"([^"]*)"', value)
            return {
                "inject_ids": ids,
                "reason": reason_match.group(1) if reason_match else value[:200],
            }

    if decision is None:
        return {"inject_ids": [], "reason": "empty_mailbox_decision"}
    return None


def _run_llm_command(command: str, prompt: str) -> Dict[str, Any]:
    proc = subprocess.run(
        shlex.split(command),
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    payload = _extract_json(proc.stdout.decode("utf-8", errors="ignore"))
    if not payload:
        raise RuntimeError("LLM command did not return JSON")
    if "decision" in payload and isinstance(payload["decision"], dict):
        return payload["decision"]
    return payload


def _http_get(url: str, token: str) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"X-LLM-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_post(url: str, token: str, payload: Dict[str, Any]) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-LLM-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read().decode("utf-8")
            return True
    except Exception:
        return False


def _mailbox_llm_decision(
    prompt: str,
    mailbox_url: str,
    mailbox_token: str,
    run_id: str,
    batch_index: int,
    batch_key: str,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    base = mailbox_url.rstrip("/")
    payload = {
        "prompt": prompt,
        "checkpoint": checkpoint,
        "meta": {
            "run_id": run_id,
            "batch": batch_index,
            "mailbox_key": batch_key,
        },
    }
    status = _http_get(f"{base}/status/{batch_key}", mailbox_token)
    if not status or not status.get("prompt_ready"):
        ok = _http_post(f"{base}/prompt/{batch_key}", mailbox_token, payload)
        if not ok:
            raise RuntimeError(f"failed to post prompt for {batch_key}")

    waited = 0
    while True:
        status = _http_get(f"{base}/status/{batch_key}", mailbox_token)
        if status and status.get("suggestions_ready"):
            response = _http_get(f"{base}/suggestions/{batch_key}", mailbox_token)
            if response:
                decision = _salvage_mailbox_decision(response)
                if isinstance(decision, dict):
                    return decision
                raise RuntimeError(f"mailbox suggestions for {batch_key} did not contain a decision object")
        time.sleep(10)
        waited += 10
        if waited % 60 == 0:
            logger.info("Waiting for mailbox LLM decision (%s): %ss", batch_key, waited)


def _parse_overseer_mode_payload(
    payload: Dict[str, Any],
    expected_batch_size: int,
    max_inject: int,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "selected_mode": "physics",
            "inject_ids": [],
            "reason": "invalid_payload",
        }
    mode = payload.get("selected_mode", payload.get("mode", "physics"))
    if mode == "llm_points":
        mode = "physics"
    if mode not in {"loggp_active", "physics"}:
        mode = "physics"
    inject_ids = payload.get("inject_ids", [])
    if not isinstance(inject_ids, list):
        inject_ids = []
    clean_ids: List[str] = []
    for pid in inject_ids:
        if isinstance(pid, str) and pid not in clean_ids:
            clean_ids.append(pid)
            if len(clean_ids) >= max_inject:
                break
    batch_n = payload.get("n_points", payload.get("batch_size", expected_batch_size))
    try:
        batch_n = int(batch_n)
    except Exception:
        batch_n = expected_batch_size
    if batch_n != expected_batch_size:
        batch_n = expected_batch_size
    return {
        "selected_mode": mode,
        "inject_ids": clean_ids,
        "reason": str(
            payload.get("decision_reason", payload.get("reason", ""))
        )[:240],
        "n_points": batch_n,
    }


def build_llm_policy_prompt(
    policy_state: Dict[str, Any],
    menu: Sequence[Dict[str, Any]],
    max_inject: int,
    batch_size: int,
) -> str:
    if str(policy_state.get("scenario", "")) == "multimodel-trap":
        ranked = policy_state.get("ranked_models", [])
        third_line = ""
        if len(ranked) >= 3:
            third_line = f"- Third model: {ranked[2][0]} ({float(ranked[2][1]):.3f})"
        lines = [
            "You are the strategic audit overseer for a controlled multi-model neutron benchmark.",
            "This is an intentional trap scenario.",
            "The leader and runner-up may agree on the bright branch, while a lower-ranked model may differ in a weak hidden pocket.",
            "Your job is to spend scarce audit probes on strategically decisive falsification measurements, not on local refinement of the bright branch.",
            "Rules:",
            "1. You may only choose IDs from the shared menu.",
            "2. You may inject at most the allowed number of probes.",
            "3. Prefer a lower-ranked-model falsifier when it can eliminate a plausible hidden alternative.",
            '4. Return strict JSON only: {"inject_ids": ["D00"], "reason": "brief reason"}.',
            f"Select at most {max_inject} inject_ids.",
            f"Current batch size: {batch_size}.",
            "Current status:",
            f"- Best model: {policy_state['leader']} ({policy_state['leader_post']:.3f})",
            f"- Runner-up: {policy_state['runner_up']} ({policy_state['runner_up_post']:.3f})",
            third_line,
            f"- Posterior entropy: {policy_state['entropy']:.3f}",
            f"- Hidden-pocket coverage fraction: {policy_state['gap_coverage']:.3f}",
            f"- Wrong-leader dwell so far: {policy_state['wrong_leader_dwell']}",
            "Task:",
            "Decide whether the best use of the audit budget is a weak-pocket falsifier against a lower-ranked model rather than another bright-branch refinement point.",
            "Menu:",
        ]
        for row in menu:
            lines.append(
                f"  {row['id']}: H={row['h']:.3f}, E={row['E']:.3f}, competitor={row.get('competitor','')}, "
                f"contrast={row['expected_contrast']:.3f}, family={row['probe_family']}, falsify={row['is_falsification_probe']}"
            )
        return "\n".join(line for line in lines if line != "")

    if str(policy_state.get("scenario", "")) == "bilayer-fm":
        lines = [
            "You are the strategic audit overseer for an autonomous triple-axis neutron spectrometer.",
            "Your role is to monitor the numerical planner and intervene if it over-refines the bright acoustic branch while neglecting measurements that could reveal a weak bilayer optic mode.",
            "Rules:",
            "1. You may not fit models, alter likelihoods, or bypass kinematic boundaries.",
            "2. Choose a routing mode from {loggp_active, physics}.",
            "3. You may optionally add a small number of shared-menu inject_ids to force falsification probes ahead of the selected mode.",
            "4. Use loggp_active when broader remapping or branch verification is needed; use physics when the planner should refine the current leading model.",
            "5. Prefer probes that test under-sampled branch structure or bilayer discrimination over probes that only sharpen the dominant branch.",
            '6. Return strict JSON only: {"decision_reason": "...", "selected_mode": "physics", "inject_ids": ["D00"]}.',
            f"Select at most {max_inject} inject_ids.",
            f"Current batch size: {batch_size}.",
            "Scientific context:",
            "- This is an [H,H,L_fixed] inelastic cut of a square-lattice ferromagnet.",
            "- The leading model is a monolayer ferromagnet with one acoustic branch.",
            "- The competing model is a weakly coupled bilayer ferromagnet with an additional optic branch whose weight is suppressed but nonzero at the chosen L.",
            "- The dominant acoustic signal can lock refinement-driven planning onto the wrong model if the optic branch is not tested directly.",
            "Current status:",
            f"- Best model: {policy_state['leader']} ({policy_state['leader_post']:.3f})",
            f"- Runner-up: {policy_state['runner_up']} ({policy_state['runner_up_post']:.3f})",
            f"- Posterior entropy: {policy_state['entropy']:.3f}",
            f"- Falsification-region coverage fraction: {policy_state['gap_coverage']:.3f}",
            f"- Wrong-leader dwell so far: {policy_state['wrong_leader_dwell']}",
            f"- Last mode: {policy_state.get('last_mode', 'physics')}",
            f"- Points since last Log-GP batch: {policy_state.get('since_loggp', 0)}",
            f"- Audit recommended: {bool(policy_state.get('audit_recommended', False))}",
            "Task:",
            "Decide whether the planner is sufficiently testing the weak bilayer branch or is trapped refining the acoustic branch.",
            "Menu:",
        ]
        for row in menu:
            lines.append(
                f"  {row['id']}: H={row['h']:.3f}, E={row['E']:.2f}, score={row['score']:.3f}, "
                f"z={row['z']:.2f}, family={row['probe_family']}, falsify={row['is_falsification_probe']}"
            )
        return "\n".join(lines)

    if str(policy_state.get("scenario", "")) == "ghost-optic":
        lines = [
            "You are the strategic audit overseer for an autonomous triple-axis neutron spectrometer.",
            "Your role is to monitor the numerical Bayesian planner ('physics' mode) and intervene if it suffers from algorithmic myopia, such as repeatedly sampling a dominant high-intensity feature to refine the current leading model while neglecting lower-intensity measurements that could falsify it.",
            "Rules:",
            "1. You may not fit models, alter numerical likelihoods, or bypass kinematic boundaries.",
            "2. Choose up to the allowed number of IDs from the shared menu only.",
            "3. Prefer probes that test under-sampled discriminative spectral structure over probes that only refine the already-sampled dominant feature.",
            '4. Return strict JSON only: {"inject_ids": ["D00"], "reason": "brief reason"}.',
            f"Select at most {max_inject} inject_ids.",
            f"Current batch size: {batch_size}.",
            "Scientific context:",
            "- This is a fixed-Q inelastic spectrum.",
            "- Both candidate models explain the dominant low-energy excitation already observed.",
            "- A competing model may also contain an additional weaker secondary excitation carrying only a small fraction of the dominant spectral weight.",
            "- Because the dominant feature is much brighter, a refinement-driven planner can become locked into repeatedly sampling that region while failing to test less-sampled energies that would be more decisive for model discrimination.",
            "Current status:",
            f"- Best model: {policy_state['leader']} ({policy_state['leader_post']:.3f})",
            f"- Runner-up: {policy_state['runner_up']} ({policy_state['runner_up_post']:.3f})",
            f"- Posterior entropy: {policy_state['entropy']:.3f}",
            f"- Falsification-region coverage fraction: {policy_state['gap_coverage']:.3f}",
            f"- Wrong-leader dwell so far: {policy_state['wrong_leader_dwell']}",
            "Task:",
            "Decide whether the planner is still productively refining a discriminative region or is instead trapped on a dominant feature while neglecting strategically important falsification measurements.",
            "Menu:",
        ]
        for row in menu:
            lines.append(
                f"  {row['id']}: E={row['E']:.2f}, score={row['score']:.3f}, "
                f"z={row['z']:.2f}, family={row['probe_family']}, falsify={row['is_falsification_probe']}"
            )
        return "\n".join(lines)

    lines = [
        "You are selecting audit probes for a neutron closed-loop ablation.",
        "Choose up to the allowed number of IDs from the menu.",
        "Return strict JSON: {inject_ids: [...], reason: \"...\"}.",
        f"Select at most {max_inject} inject_ids.",
        f"Current batch size: {batch_size}.",
        f"Best model: {policy_state['leader']} ({policy_state['leader_post']:.3f}).",
        f"Runner-up: {policy_state['runner_up']} ({policy_state['runner_up_post']:.3f}).",
        f"Posterior entropy: {policy_state['entropy']:.3f}.",
        f"Falsification-region coverage fraction: {policy_state['gap_coverage']:.3f}.",
        f"Wrong-leader dwell so far: {policy_state['wrong_leader_dwell']}.",
        f"Silent-data active: {policy_state['silent_data_active']}.",
        "Prefer true falsification probes when the leader may be wrong or under-tested.",
        "Menu:",
    ]
    for row in menu:
        lines.append(
            f"  {row['id']}: h={row['h']:.3f}, E={row['E']:.2f}, score={row['score']:.3f}, "
            f"z={row['z']:.2f}, family={row['probe_family']}, falsify={row['is_falsification_probe']}"
        )
    return "\n".join(lines)


@dataclass
class SilentDataConfig:
    enabled: bool = False
    until_measurement: int = 0
    attenuation: float = 0.25
    h_max: float = 0.65
    e_max: float = 2.0
    background_boost: float = 0.30


@dataclass
class HiddenGapInitConfig:
    enabled: bool = False
    coarse_stride: int = 1
    init_hmin: float = 0.75
    init_emin: float = 2.5
    grid_h: int = 9
    grid_e: int = 7


@dataclass
class GhostOpticConfig:
    energy_min: float = 0.0
    energy_max: float = 20.0
    gamma: float = 0.5
    background: float = 0.10
    acoustic_energy: float = 5.0
    optic_energy: float = 15.0
    acoustic_amplitude: float = 100.0
    optic_fraction: float = 0.05
    sigma_floor: float = 0.10
    sigma_scale: float = 0.03
    leader_prior: float = 0.95
    mixed_eta: float = 0.05
    refine_sigma_amp: float = 25.0
    refine_sigma_energy: float = 0.10
    init_energies: Tuple[float, ...] = (4.25, 4.75, 5.25, 5.75)
    candidate_points: int = 401
    optic_probe_min: float = 11.0


@dataclass
class BilayerFMConfig:
    S: float = 1.0
    J_par: float = 2.0
    J_perp: float = 0.45
    D: float = 0.10
    z_bi: float = 0.35
    z_perp: int = 1
    gamma: float = 0.60
    amp: float = 100.0
    background: float = 0.10
    L_fixed: float = 0.16
    h_min: float = 0.0
    h_max: float = 0.5
    e_min: float = 0.0
    e_max: float = 22.0
    candidate_h: int = 101
    candidate_e: int = 181
    sigma_floor: float = 0.10
    sigma_scale: float = 0.03
    leader_prior: float = 0.992
    refine_sigma_amp: float = 25.0
    refine_sigma_jpar: float = 0.20
    refine_sigma_d: float = 0.02
    init_h: Tuple[float, ...] = (0.24,)
    init_energy_offsets: Tuple[float, ...] = (-0.03, 0.0, 0.03)
    optic_probe_min: float = 6.0
    optic_region_tolerance: float = 0.35


@dataclass
class MultimodelTrapConfig:
    sigma: float = 1.5
    background: float = 0.3
    ridge_amp: float = 120.0
    pocket_amp: float = 9.0
    seed_coords: Tuple[Tuple[float, float], ...] = (
        (0.72, 0.58),
        (0.82, 0.64),
        (0.42, 0.42),
        (0.50, 0.46),
    )
    h_min: float = 0.0
    h_max: float = 1.0
    e_min: float = 0.13
    e_max: float = 1.0
    candidate_h: int = 61
    candidate_e: int = 61
    pocket_h: float = 0.30
    pocket_e: float = 0.75
    pocket_h_tol: float = 0.08
    pocket_e_tol: float = 0.08
    decisive_ratio: float = 20.0
    runner_ridge_offset: float = 0.001
    runner_ridge_amp_delta: float = 0.6
    min_falsification_contrast: float = 0.25


class SilentDataModel:
    def __init__(self, base: SquareLatticeDispersion, config: SilentDataConfig) -> None:
        self.base = base
        self.config = config
        self.measurement_index = 0

    def _silent_region(self, h: float, e: float) -> bool:
        return float(h) <= self.config.h_max and float(e) <= self.config.e_max

    def intensity(self, h: float, k: float, e: float) -> float:
        value = float(self.base.intensity(h, k, e))
        if (
            self.config.enabled
            and self.measurement_index < self.config.until_measurement
            and self._silent_region(h, e)
        ):
            return float(self.config.background_boost + self.config.attenuation * value)
        return value

    def advance(self, n: int) -> None:
        self.measurement_index += int(n)


def _ghost_lorentz(E: float, E0: float, A: float, gamma: float) -> float:
    return float(A * gamma / (((float(E) - float(E0)) ** 2) + gamma**2))


def _ghost_model_intensity(E: float, config: GhostOpticConfig, with_optic: bool) -> float:
    value = _ghost_lorentz(E, config.acoustic_energy, config.acoustic_amplitude, config.gamma) + config.background
    if with_optic:
        value += _ghost_lorentz(
            E,
            config.optic_energy,
            config.acoustic_amplitude * config.optic_fraction,
            config.gamma,
        )
    return float(value)


def _ghost_sigma2(E: float, config: GhostOpticConfig) -> float:
    ref = max(_ghost_model_intensity(E, config, with_optic=False), config.background)
    return float(max(config.sigma_scale * ref + config.sigma_floor, 1e-6))


def _ghost_refine_utility(E: float, config: GhostOpticConfig) -> float:
    dE = 1e-3
    dI_dA = _ghost_lorentz(E, config.acoustic_energy, 1.0, config.gamma)
    plus = _ghost_lorentz(E, config.acoustic_energy + dE, config.acoustic_amplitude, config.gamma)
    minus = _ghost_lorentz(E, config.acoustic_energy - dE, config.acoustic_amplitude, config.gamma)
    dI_dE = (plus - minus) / (2.0 * dE)
    fisher_like = (dI_dA**2) * (config.refine_sigma_amp**2) + (dI_dE**2) * (config.refine_sigma_energy**2)
    return float(0.5 * math.log1p(fisher_like / _ghost_sigma2(E, config)))


def _ghost_false_value(E: float, config: GhostOpticConfig) -> float:
    delta = _ghost_model_intensity(E, config, with_optic=True) - _ghost_model_intensity(E, config, with_optic=False)
    return float((delta**2) / (2.0 * _ghost_sigma2(E, config)))


def _ghost_energy_grid(config: GhostOpticConfig) -> np.ndarray:
    return np.linspace(config.energy_min, config.energy_max, max(21, int(config.candidate_points)), dtype=float)


def _ghost_refine_scores(config: GhostOpticConfig) -> np.ndarray:
    e_vals = _ghost_energy_grid(config)
    return np.array([_ghost_refine_utility(float(energy), config) for energy in e_vals], dtype=float)


def _ghost_exclusion_mask(
    e_vals: np.ndarray,
    measurements: Sequence[Dict[str, Any]],
    e_thresh: float,
) -> np.ndarray:
    mask = np.ones(len(e_vals), dtype=bool)
    if not measurements:
        return mask
    existing_e = np.array([float(meas["E"]) for meas in measurements], dtype=float)
    if existing_e.size == 0:
        return mask
    # Exclude any candidate within the requested energy threshold of an existing point.
    delta = np.abs(e_vals[:, None] - existing_e[None, :])
    mask &= ~np.any(delta < e_thresh, axis=1)
    return mask


def _ghost_select_top_energies(
    e_vals: np.ndarray,
    scores: np.ndarray,
    existing_measurements: Sequence[Dict[str, Any]],
    n_points: int,
    existing_thresh: float = 0.35,
    intra_batch_thresh: float = 0.5,
) -> List[float]:
    if n_points <= 0 or len(e_vals) == 0:
        return []
    mask = _ghost_exclusion_mask(e_vals, existing_measurements, existing_thresh)
    avail_idx = np.flatnonzero(mask)
    if avail_idx.size == 0:
        return []
    avail_scores = scores[avail_idx]
    order = avail_idx[np.argsort(-avail_scores, kind="stable")]
    selected: List[float] = []
    for idx in order:
        energy = float(e_vals[idx])
        if any(abs(energy - prev) < intra_batch_thresh for prev in selected):
            continue
        selected.append(energy)
        if len(selected) >= n_points:
            break
    return selected


def _ghost_optic_region_hit(meas: Dict[str, Any], config: GhostOpticConfig) -> bool:
    return float(meas["E"]) >= config.optic_probe_min


def _ghost_coverage_ratio(measurements: Sequence[Dict[str, Any]], config: GhostOpticConfig) -> float:
    if not measurements:
        return 0.0
    hits = sum(1 for meas in measurements if _ghost_optic_region_hit(meas, config))
    return float(hits / len(measurements))


def _ghost_model_posteriors(
    measurements: Sequence[Dict[str, Any]],
    config: GhostOpticConfig,
) -> Dict[str, Dict[str, float]]:
    names = (
        "M_A: Acoustic-only",
        "M_B: Acoustic+optic",
    )
    prior_a = min(max(float(config.leader_prior), 1e-6), 1.0 - 1e-6)
    prior_logs = {
        names[0]: math.log(prior_a),
        names[1]: math.log(1.0 - prior_a),
    }
    loglikes = {name: prior_logs[name] for name in names}
    for meas in measurements:
        energy = float(meas["E"])
        obs = float(meas["intensity"])
        sigma2 = float(meas.get("variance", _ghost_sigma2(energy, config)))
        for name, with_optic in ((names[0], False), (names[1], True)):
            mu = _ghost_model_intensity(energy, config, with_optic=with_optic)
            loglikes[name] += -0.5 * ((obs - mu) ** 2) / sigma2 - 0.5 * math.log(2.0 * math.pi * sigma2)

    max_log = max(loglikes.values())
    weights = {name: math.exp(val - max_log) for name, val in loglikes.items()}
    total = sum(weights.values())
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        out[name] = {
            "posterior": float(weights[name] / max(total, 1e-12)),
            "log_posterior": float(loglikes[name]),
            "aic": float("nan"),
            "chi2": float("nan"),
        }
    return out


def _ghost_discrimination_menu(
    measurements: Sequence[Dict[str, Any]],
    config: GhostOpticConfig,
    menu_size: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    e_vals = np.linspace(config.energy_min, config.energy_max, max(21, int(config.candidate_points)))
    scored: List[Dict[str, Any]] = []
    existing = [{"h": 1.0, "E": float(meas["E"])} for meas in measurements]
    for energy in e_vals:
        if _too_close(1.0, float(energy), existing, h_thresh=0.2, e_thresh=0.35):
            continue
        false_val = _ghost_false_value(float(energy), config)
        delta = abs(
            _ghost_model_intensity(float(energy), config, with_optic=True)
            - _ghost_model_intensity(float(energy), config, with_optic=False)
        )
        sigma = math.sqrt(_ghost_sigma2(float(energy), config))
        scored.append(
            {
                "h": 1.0,
                "k": 1.0,
                "E": float(energy),
                "score": false_val,
                "z": float(delta / max(sigma, 1e-6)),
                "move_time": 0.0,
                "mu_best": _ghost_model_intensity(float(energy), config, with_optic=False),
                "mu_second": _ghost_model_intensity(float(energy), config, with_optic=True),
                "sigma_E": 0.0,
                "best_model": "M_A: Acoustic-only",
                "second_model": "M_B: Acoustic+optic",
                "probe_family": "ghost_optic",
                "is_falsification_probe": bool(float(energy) >= config.optic_probe_min and false_val > 0.05),
                "expected_contrast": delta,
            }
        )
    scored.sort(key=lambda row: (-float(row["score"]), -float(row["z"])))
    menu: List[Dict[str, Any]] = []
    for row in scored:
        if len(menu) >= menu_size:
            break
        if _too_close(float(row["h"]), float(row["E"]), menu, h_thresh=0.2, e_thresh=0.5):
            continue
        menu.append(dict(row))
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(menu):
        pid = f"D{idx:02d}"
        row["id"] = pid
        lookup[pid] = row
    return menu, lookup


def _ghost_plan_refinement_points(
    measurements: Sequence[Dict[str, Any]],
    config: GhostOpticConfig,
    n_points: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    e_vals = _ghost_energy_grid(config)
    scores = _ghost_refine_scores(config)
    for energy in _ghost_select_top_energies(e_vals, scores, measurements, n_points):
        out.append(
            {
                "h": 1.0,
                "k": 1.0,
                "E": energy,
                "mode": "ghost_refine",
                "audit_policy": "",
                "audit_injected": False,
                "is_falsification_probe": False,
                "audit_probe_family": "ghost_refine",
                "expected_contrast": 0.0,
                "count_time": 60.0,
            }
        )
    return out


def _ghost_build_loggp(measurements: Sequence[Dict[str, Any]], config: GhostOpticConfig) -> Any:
    if AgnosticExplorer is None:
        gp = SimpleGaussianProcess(length_scale=0.12, noise=0.25)
        for meas in measurements:
            energy = float(meas["E"])
            intensity = float(meas["intensity"])
            gp.add_observation(np.array([energy / 20.0]), np.log1p(max(intensity, 0.0)))
        return gp

    gp = _LibraryLogGPAdapter(
        bounds=np.array([[float(config.energy_min), float(config.energy_max)]], dtype=float),
        background=float(config.background),
    )
    for meas in measurements:
        energy = float(meas["E"])
        intensity = max(float(meas["intensity"]), 0.0)
        sigma = float(meas.get("uncertainty", math.sqrt(float(meas.get("variance", 0.04)))))
        gp.add_observation([energy], intensity, sigma)
    return gp


def _ghost_plan_loggp_points(
    measurements: Sequence[Dict[str, Any]],
    config: GhostOpticConfig,
    n_points: int,
) -> List[Dict[str, Any]]:
    gp = _ghost_build_loggp(measurements, config)
    e_vals = _ghost_energy_grid(config)
    if AgnosticExplorer is None:
        X = (e_vals / 20.0).reshape(-1, 1)
        _, score_vals = gp.predict(X)
    else:
        X = e_vals.reshape(-1, 1)
        _, std = gp.predict_batch(X)
        score_vals = std**2
    out: List[Dict[str, Any]] = []
    for energy in _ghost_select_top_energies(
        e_vals,
        np.asarray(score_vals, dtype=float),
        measurements,
        n_points,
    ):
        out.append(
            {
                "h": 1.0,
                "k": 1.0,
                "E": energy,
                "mode": "ghost_loggp",
                "audit_policy": "loggp",
                "audit_injected": False,
                "is_falsification_probe": bool(energy >= config.optic_probe_min),
                "audit_probe_family": "ghost_loggp",
                "expected_contrast": float(score_vals[np.argmin(np.abs(e_vals - energy))]),
                "count_time": 60.0,
            }
        )
    return out


def _ghost_simulate_measurements(
    points: Sequence[Dict[str, Any]],
    config: GhostOpticConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pt in points:
        energy = float(pt["E"])
        mu_true = _ghost_model_intensity(energy, config, with_optic=True)
        sigma2 = _ghost_sigma2(energy, config)
        sigma = math.sqrt(sigma2)
        obs = float(rng.normal(mu_true, sigma))
        out.append(
            {
                "h": 1.0,
                "k": 1.0,
                "E": energy,
                "intensity": obs,
                "uncertainty": sigma,
                "variance": sigma2,
                "counts": float("nan"),
                "coverage": False,
                "symmetry": False,
                "human_hint": False,
                "llm_hint": bool(pt.get("llm_hint")),
                "llm_reason": str(pt.get("llm_reason", ""))[:200],
                "mode": pt.get("mode", "ghost_refine"),
                "llm_batch_idx": pt.get("llm_batch_idx"),
                "llm_provider": pt.get("llm_provider"),
                "llm_decision_reason": str(pt.get("llm_decision_reason", ""))[:200],
                "loggp_hint": False,
                "count_time": float(pt.get("count_time", 60.0)),
                "dwell": False,
                "audit_policy": pt.get("audit_policy"),
                "audit_injected": bool(pt.get("audit_injected")),
                "audit_probe_family": pt.get("audit_probe_family"),
                "is_falsification_probe": bool(pt.get("is_falsification_probe")),
                "expected_contrast": float(pt.get("expected_contrast", 0.0)),
                "silent_data_active": False,
            }
        )
    return out


class SquareFMBilayerAnalytic:
    def __init__(
        self,
        S: float,
        J_par: float,
        J_perp: float,
        D: float,
        z_bi: float,
        z_perp: int,
        gamma: float,
        amp: float,
        background: float,
    ) -> None:
        self.S = float(S)
        self.J_par = float(J_par)
        self.J_perp = float(J_perp)
        self.D = float(D)
        self.z_bi = float(z_bi)
        self.z_perp = int(z_perp)
        self.gamma = float(gamma)
        self.amp = float(amp)
        self.background = float(background)

    def anisotropy_gap(self) -> float:
        return float(self.D * (2.0 * self.S - 1.0))

    def omega_mono(self, H: float, K: float) -> float:
        return float(
            2.0 * self.J_par * self.S * (2.0 - math.cos(2.0 * math.pi * H) - math.cos(2.0 * math.pi * K))
            + self.anisotropy_gap()
        )

    def optic_shift(self) -> float:
        return float(2.0 * self.z_perp * self.S * self.J_perp)

    def omega_ac(self, H: float, K: float) -> float:
        return self.omega_mono(H, K)

    def omega_op(self, H: float, K: float) -> float:
        return float(self.omega_mono(H, K) + self.optic_shift())

    def bilayer_weights(self, L: float) -> Tuple[float, float]:
        phi = math.pi * self.z_bi * float(L)
        return float(math.cos(phi) ** 2), float(math.sin(phi) ** 2)

    def lorentzian(self, E: float, E0: float) -> float:
        return float(self.gamma / (((float(E) - float(E0)) ** 2) + self.gamma**2))

    def intensity_bilayer(self, H: float, K: float, L: float, E: float) -> float:
        w_ac, w_op = self.bilayer_weights(L)
        return float(
            self.background
            + self.amp * (
                w_ac * self.lorentzian(E, self.omega_ac(H, K))
                + w_op * self.lorentzian(E, self.omega_op(H, K))
            )
        )

    def intensity_monolayer(self, H: float, K: float, L: float, E: float) -> float:
        w_ac, _ = self.bilayer_weights(L)
        return float(self.background + self.amp * w_ac * self.lorentzian(E, self.omega_mono(H, K)))


def _bilayer_models(config: BilayerFMConfig) -> Tuple[SquareFMBilayerAnalytic, SquareFMBilayerAnalytic]:
    mono = SquareFMBilayerAnalytic(
        S=config.S,
        J_par=config.J_par,
        J_perp=0.0,
        D=config.D,
        z_bi=config.z_bi,
        z_perp=config.z_perp,
        gamma=config.gamma,
        amp=config.amp,
        background=config.background,
    )
    bi = SquareFMBilayerAnalytic(
        S=config.S,
        J_par=config.J_par,
        J_perp=config.J_perp,
        D=config.D,
        z_bi=config.z_bi,
        z_perp=config.z_perp,
        gamma=config.gamma,
        amp=config.amp,
        background=config.background,
    )
    return mono, bi


def _bilayer_sigma2(H: float, E: float, config: BilayerFMConfig) -> float:
    _, bi = _bilayer_models(config)
    ref = max(bi.intensity_bilayer(H, H, config.L_fixed, E), config.background)
    return float(max(config.sigma_scale * ref + config.sigma_floor, 1e-6))


def _bilayer_refine_utility(H: float, E: float, config: BilayerFMConfig) -> float:
    mono, _ = _bilayer_models(config)
    dA = mono.lorentzian(E, mono.omega_mono(H, H))
    dJ = 1e-3
    plus_j = SquareFMBilayerAnalytic(config.S, config.J_par + dJ, 0.0, config.D, config.z_bi, config.z_perp, config.gamma, config.amp, config.background)
    minus_j = SquareFMBilayerAnalytic(config.S, config.J_par - dJ, 0.0, config.D, config.z_bi, config.z_perp, config.gamma, config.amp, config.background)
    dI_dJ = (plus_j.intensity_monolayer(H, H, config.L_fixed, E) - minus_j.intensity_monolayer(H, H, config.L_fixed, E)) / (2.0 * dJ)
    plus_d = SquareFMBilayerAnalytic(config.S, config.J_par, 0.0, config.D + dJ, config.z_bi, config.z_perp, config.gamma, config.amp, config.background)
    minus_d = SquareFMBilayerAnalytic(config.S, config.J_par, 0.0, config.D - dJ, config.z_bi, config.z_perp, config.gamma, config.amp, config.background)
    dI_dD = (plus_d.intensity_monolayer(H, H, config.L_fixed, E) - minus_d.intensity_monolayer(H, H, config.L_fixed, E)) / (2.0 * dJ)
    fisher_like = (
        (dA**2) * (config.refine_sigma_amp**2)
        + (dI_dJ**2) * (config.refine_sigma_jpar**2)
        + (dI_dD**2) * (config.refine_sigma_d**2)
    )
    return float(0.5 * math.log1p(fisher_like / _bilayer_sigma2(H, E, config)))


def _bilayer_false_value(H: float, E: float, config: BilayerFMConfig) -> float:
    mono, bi = _bilayer_models(config)
    delta = bi.intensity_bilayer(H, H, config.L_fixed, E) - mono.intensity_monolayer(H, H, config.L_fixed, E)
    return float((delta**2) / (2.0 * _bilayer_sigma2(H, E, config)))


def _bilayer_optic_region_hit(meas: Dict[str, Any], config: BilayerFMConfig) -> bool:
    _, bi = _bilayer_models(config)
    h = float(meas["h"])
    energy = float(meas["E"])
    optic_center = float(bi.omega_op(h, h))
    return bool(
        energy >= config.optic_probe_min
        and abs(energy - optic_center) <= float(config.optic_region_tolerance)
    )


def _bilayer_coverage_ratio(measurements: Sequence[Dict[str, Any]], config: BilayerFMConfig) -> float:
    if not measurements:
        return 0.0
    hits = sum(1 for meas in measurements if _bilayer_optic_region_hit(meas, config))
    return float(hits / len(measurements))


def _bilayer_model_posteriors(
    measurements: Sequence[Dict[str, Any]],
    config: BilayerFMConfig,
) -> Dict[str, Dict[str, float]]:
    names = ("M_A: Monolayer FM", "M_B: Bilayer FM")
    mono, bi = _bilayer_models(config)
    prior_a = min(max(float(config.leader_prior), 1e-6), 1.0 - 1e-6)
    loglikes = {
        names[0]: math.log(prior_a),
        names[1]: math.log(1.0 - prior_a),
    }
    for meas in measurements:
        h = float(meas["h"])
        e = float(meas["E"])
        obs = float(meas["intensity"])
        sigma2 = float(meas.get("variance", _bilayer_sigma2(h, e, config)))
        preds = {
            names[0]: mono.intensity_monolayer(h, h, config.L_fixed, e),
            names[1]: bi.intensity_bilayer(h, h, config.L_fixed, e),
        }
        for name in names:
            mu = preds[name]
            loglikes[name] += -0.5 * ((obs - mu) ** 2) / sigma2 - 0.5 * math.log(2.0 * math.pi * sigma2)
    max_log = max(loglikes.values())
    weights = {name: math.exp(val - max_log) for name, val in loglikes.items()}
    total = sum(weights.values())
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        out[name] = {
            "posterior": float(weights[name] / max(total, 1e-12)),
            "log_posterior": float(loglikes[name]),
            "aic": float("nan"),
            "chi2": float("nan"),
        }
    return out


def _bilayer_candidate_grid(config: BilayerFMConfig) -> List[Tuple[float, float]]:
    hs = np.linspace(config.h_min, config.h_max, max(11, int(config.candidate_h)))
    es = np.linspace(config.e_min, config.e_max, max(21, int(config.candidate_e)))
    return [(float(h), float(e)) for h in hs for e in es]


def _bilayer_discrimination_menu(
    measurements: Sequence[Dict[str, Any]],
    config: BilayerFMConfig,
    menu_size: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    mono, bi = _bilayer_models(config)
    scored: List[Dict[str, Any]] = []
    existing = [{"h": float(meas["h"]), "E": float(meas["E"])} for meas in measurements]
    for h, energy in _bilayer_candidate_grid(config):
        if not TAS.is_accessible(h, h, energy):
            continue
        if _too_close(h, energy, existing, h_thresh=0.03, e_thresh=0.4):
            continue
        mu_a = mono.intensity_monolayer(h, h, config.L_fixed, energy)
        mu_b = bi.intensity_bilayer(h, h, config.L_fixed, energy)
        sigma = math.sqrt(_bilayer_sigma2(h, energy, config))
        delta = abs(mu_b - mu_a)
        false_val = (delta**2) / (2.0 * max(sigma**2, 1e-8))
        scored.append(
            {
                "h": h,
                "k": h,
                "E": energy,
                "score": float(false_val),
                "z": float(delta / max(sigma, 1e-6)),
                "move_time": 0.0,
                "mu_best": mu_a,
                "mu_second": mu_b,
                "sigma_E": 0.0,
                "best_model": "M_A: Monolayer FM",
                "second_model": "M_B: Bilayer FM",
                "probe_family": "bilayer_optic",
                "is_falsification_probe": bool(_bilayer_optic_region_hit({"h": h, "E": energy}, config) and false_val > 0.05),
                "expected_contrast": float(delta),
            }
        )
    scored.sort(key=lambda row: (-float(row["score"]), -float(row["z"])))
    menu: List[Dict[str, Any]] = []
    for row in scored:
        if len(menu) >= menu_size:
            break
        if _too_close(float(row["h"]), float(row["E"]), menu, h_thresh=0.03, e_thresh=0.5):
            continue
        menu.append(dict(row))
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(menu):
        pid = f"D{idx:02d}"
        row["id"] = pid
        lookup[pid] = row
    return menu, lookup


def _bilayer_plan_refinement_points(
    measurements: Sequence[Dict[str, Any]],
    config: BilayerFMConfig,
    n_points: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        (
            (_bilayer_refine_utility(h, energy, config), h, energy)
            for h, energy in _bilayer_candidate_grid(config)
            if TAS.is_accessible(h, h, energy)
        ),
        reverse=True,
    )
    existing = [{"h": float(meas["h"]), "E": float(meas["E"])} for meas in measurements]
    out: List[Dict[str, Any]] = []
    for _, h, energy in ranked:
        if len(out) >= n_points:
            break
        if _too_close(h, energy, existing, h_thresh=0.03, e_thresh=0.4):
            continue
        if _too_close(h, energy, out, h_thresh=0.03, e_thresh=0.5):
            continue
        out.append(
            {
                "h": h,
                "k": h,
                "E": energy,
                "mode": "bilayer_physics",
                "audit_policy": "",
                "audit_injected": False,
                "is_falsification_probe": False,
                "audit_probe_family": "bilayer_refine",
                "expected_contrast": 0.0,
                "count_time": 60.0,
            }
        )
    return out


def _bilayer_plan_hybrid_points(
    measurements: Sequence[Dict[str, Any]],
    config: BilayerFMConfig,
    n_points: int,
) -> List[Dict[str, Any]]:
    if AgnosticExplorer is None:
        gp = SimpleGaussianProcess(length_scale=0.12, noise=0.25)
        for meas in measurements:
            if "intensity" not in meas:
                continue
            gp.add_observation(
                np.array([float(meas["h"]), float(meas["E"]) / max(config.e_max, 1e-6)]),
                np.log1p(max(float(meas["intensity"]), 0.0)),
            )
    else:
        gp = _LibraryLogGPAdapter(
            bounds=np.array(
                [[float(config.h_min), float(config.h_max)], [float(config.e_min), float(config.e_max)]],
                dtype=float,
            ),
            background=float(config.background),
        )
        for meas in measurements:
            if "intensity" not in meas:
                continue
            sigma = float(meas.get("uncertainty", math.sqrt(float(meas.get("variance", 0.04)))))
            gp.add_observation([float(meas["h"]), float(meas["E"])], max(float(meas["intensity"]), 0.0), sigma)
    candidates = [(h, e) for h, e in _bilayer_candidate_grid(config) if TAS.is_accessible(h, h, e)]
    existing = [{"h": float(meas["h"]), "E": float(meas["E"])} for meas in measurements]
    if AgnosticExplorer is None:
        X = np.array([[h, e / max(config.e_max, 1e-6)] for h, e in candidates], dtype=float)
        mu, var = gp.predict(X)
        var_clamped = np.minimum(var, 10.0)
        linear_var = (np.exp(var_clamped) - 1.0) * np.exp(2.0 * mu + var_clamped)
    else:
        X = np.array([[h, e] for h, e in candidates], dtype=float)
        mean, std = gp.predict_batch(X)
        linear_var = std**2
    e_vals = np.array([e for _, e in candidates], dtype=float)
    if config.e_max > config.e_min:
        e_norm = (e_vals - float(config.e_min)) / (float(config.e_max) - float(config.e_min))
    else:
        e_norm = np.zeros_like(e_vals)
    taper = 0.10
    w_e = np.ones_like(e_vals)
    mask_low = e_norm < taper
    if np.any(mask_low):
        w_e[mask_low] = 0.5 * (1.0 - np.cos(np.pi * e_norm[mask_low] / taper))
    mask_high = e_norm > (1.0 - taper)
    if np.any(mask_high):
        w_e[mask_high] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - e_norm[mask_high]) / taper))
    if measurements:
        last = measurements[-1]
        dh = np.array([abs(h - float(last["h"])) for h, _ in candidates], dtype=float)
        dE = np.array([abs(e - float(last["E"])) for _, e in candidates], dtype=float)
        move_time = np.maximum(dh / max(LOGGP_MOVE_VH, 1e-6), dE / max(LOGGP_MOVE_VE, 1e-6)) + LOGGP_MOVE_OVERHEAD
    else:
        move_time = np.zeros(len(candidates), dtype=float)
    score = (linear_var * w_e) / (1.0 + move_time)
    ranked = sorted(((float(score[idx]), candidates[idx][0], candidates[idx][1]) for idx in range(len(candidates))), reverse=True)
    out: List[Dict[str, Any]] = []
    for score, h, energy in ranked:
        if len(out) >= n_points:
            break
        if _too_close(h, energy, existing, h_thresh=0.03, e_thresh=0.4):
            continue
        if _too_close(h, energy, out, h_thresh=0.03, e_thresh=0.5):
            continue
        out.append(
            {
                "h": h,
                "k": h,
                "E": energy,
                "mode": "bilayer_loggp",
                "audit_policy": "hybrid",
                "audit_injected": False,
                "is_falsification_probe": bool(_bilayer_optic_region_hit({"h": h, "E": energy}, config)),
                "audit_probe_family": "bilayer_loggp",
                "expected_contrast": score,
                "count_time": 60.0,
            }
        )
    return out


def _bilayer_simulate_measurements(
    points: Sequence[Dict[str, Any]],
    config: BilayerFMConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    _, bi = _bilayer_models(config)
    out: List[Dict[str, Any]] = []
    for pt in points:
        h = float(pt["h"])
        e = float(pt["E"])
        mu_true = bi.intensity_bilayer(h, h, config.L_fixed, e)
        sigma2 = _bilayer_sigma2(h, e, config)
        sigma = math.sqrt(sigma2)
        obs = float(rng.normal(mu_true, sigma))
        out.append(
            {
                "h": h,
                "k": h,
                "E": e,
                "intensity": obs,
                "uncertainty": sigma,
                "variance": sigma2,
                "counts": float("nan"),
                "coverage": False,
                "symmetry": False,
                "human_hint": False,
                "llm_hint": bool(pt.get("llm_hint")),
                "llm_reason": str(pt.get("llm_reason", ""))[:200],
                "mode": pt.get("mode", "bilayer_physics"),
                "llm_batch_idx": pt.get("llm_batch_idx"),
                "llm_provider": pt.get("llm_provider"),
                "llm_decision_reason": str(pt.get("llm_decision_reason", ""))[:200],
                "loggp_hint": bool(pt.get("mode", "").endswith("loggp")),
                "count_time": float(pt.get("count_time", 60.0)),
                "dwell": False,
                "audit_policy": pt.get("audit_policy"),
                "audit_injected": bool(pt.get("audit_injected")),
                "audit_probe_family": pt.get("audit_probe_family"),
                "is_falsification_probe": bool(pt.get("is_falsification_probe")),
                "expected_contrast": float(pt.get("expected_contrast", 0.0)),
                "silent_data_active": False,
            }
        )
    return out


def _trap_gaussian2d(h: float, e: float, h0: float, e0: float, sh: float, se: float, amp: float) -> float:
    return float(amp * math.exp(-0.5 * (((float(h) - h0) / sh) ** 2 + ((float(e) - e0) / se) ** 2)))


def _trap_ridge(h: float, e: float, *, offset: float = 0.0, amp: float = 120.0) -> float:
    center = 0.18 + 0.55 * float(h) + offset
    return float(amp * math.exp(-0.5 * ((float(e) - center) / 0.05) ** 2))


def _trap_models(config: MultimodelTrapConfig) -> Dict[str, Any]:
    def m4(h: float, e: float) -> float:
        return float(
            config.background
            + _trap_ridge(h, e, offset=0.0, amp=config.ridge_amp)
            + _trap_gaussian2d(h, e, config.pocket_h, config.pocket_e, 0.06, 0.04, config.pocket_amp)
        )

    def m2(h: float, e: float) -> float:
        return float(
            config.background
            + _trap_ridge(h, e, offset=config.runner_ridge_offset, amp=config.ridge_amp - config.runner_ridge_amp_delta)
            + _trap_gaussian2d(h, e, config.pocket_h, config.pocket_e, 0.06, 0.04, config.pocket_amp)
        )

    def m3(h: float, e: float) -> float:
        return float(config.background + _trap_ridge(h, e, offset=0.0005, amp=config.ridge_amp - 0.1))

    return {
        "M4: Ridge+pocket": m4,
        "M2: Shifted ridge+pocket": m2,
        "M3: Ridge only": m3,
    }


def _trap_is_pocket_probe(h: float, e: float, config: MultimodelTrapConfig) -> bool:
    return bool(
        abs(float(h) - config.pocket_h) <= config.pocket_h_tol
        and abs(float(e) - config.pocket_e) <= config.pocket_e_tol
    )


def _trap_coverage_ratio(measurements: Sequence[Dict[str, Any]], config: MultimodelTrapConfig) -> float:
    if not measurements:
        return 0.0
    hits = sum(1 for meas in measurements if _trap_is_pocket_probe(float(meas["h"]), float(meas["E"]), config))
    return float(hits / max(1, len(measurements)))


def _trap_model_posteriors(
    measurements: Sequence[Dict[str, Any]],
    config: MultimodelTrapConfig,
) -> Dict[str, Dict[str, Any]]:
    models = _trap_models(config)
    priors = {
        "M4: Ridge+pocket": 0.45,
        "M2: Shifted ridge+pocket": 0.40,
        "M3: Ridge only": 0.15,
    }
    logw = {name: math.log(priors[name]) for name in models}
    var = float(config.sigma**2)
    for meas in measurements:
        if "intensity" not in meas:
            continue
        h = float(meas["h"])
        e = float(meas["E"])
        obs = float(meas["intensity"])
        for name, fn in models.items():
            mu = float(fn(h, e))
            logw[name] += -0.5 * ((obs - mu) ** 2) / var
    m = max(logw.values())
    w = {name: math.exp(val - m) for name, val in logw.items()}
    z = sum(w.values())
    return {name: {"posterior": float(w[name] / z), "fitted_model": fn} for name, fn in models.items()}


def _trap_candidate_menu(
    measurements: Sequence[Dict[str, Any]],
    config: MultimodelTrapConfig,
    top_two_only: bool,
    menu_size: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    results = _trap_model_posteriors(measurements, config)
    ranked = _model_rank(results)
    leader = ranked[0][0]
    competitors = [ranked[1][0]] if top_two_only else [name for name, _ in ranked[1:]]
    leader_fn = results[leader]["fitted_model"]
    rows: List[Dict[str, Any]] = []
    counter = 0
    for h in np.linspace(config.h_min, config.h_max, config.candidate_h):
        for e in np.linspace(config.e_min, config.e_max, config.candidate_e):
            leader_mu = float(leader_fn(float(h), float(e)))
            if leader_mu < 0.5:
                continue
            best_name: Optional[str] = None
            best_delta = -1.0
            best_mu = 0.0
            for name in competitors:
                mu = float(results[name]["fitted_model"](float(h), float(e)))
                delta = abs(leader_mu - mu)
                if delta > best_delta:
                    best_name = str(name)
                    best_delta = float(delta)
                    best_mu = float(mu)
            if best_name is None:
                continue
            rows.append(
                {
                    "id": f"D{counter:02d}",
                    "h": float(h),
                    "k": float(h),
                    "E": float(e),
                    "score": float(best_delta),
                    "z": float(best_delta / max(config.sigma, 1e-6)),
                    "expected_contrast": float(best_delta),
                    "probe_family": "hidden_pocket" if _trap_is_pocket_probe(float(h), float(e), config) else "bright_branch",
                    "is_falsification_probe": bool(_trap_is_pocket_probe(float(h), float(e), config)),
                    "effective_falsification_probe": bool(
                        _trap_is_pocket_probe(float(h), float(e), config) and best_delta >= config.min_falsification_contrast
                    ),
                    "competitor": best_name,
                    "mu_best": float(leader_mu),
                    "mu_second": float(best_mu),
                }
            )
            counter += 1
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            not bool(row["effective_falsification_probe"]),
            -float(row["expected_contrast"]),
            -float(row["z"]),
            -float(row["score"]),
        ),
    )[:menu_size]
    return ranked_rows, {str(row["id"]): row for row in ranked_rows}


def _trap_plan_refinement_points(
    measurements: Sequence[Dict[str, Any]],
    config: MultimodelTrapConfig,
    n_points: int,
) -> List[Dict[str, Any]]:
    if n_points <= 0:
        return []
    results = _trap_model_posteriors(measurements, config)
    leader = _model_rank(results)[0][0]
    leader_fn = results[leader]["fitted_model"]
    rows: List[Tuple[float, float, float]] = []
    for h in np.linspace(config.h_min, config.h_max, config.candidate_h):
        center = 0.18 + 0.55 * float(h)
        e = min(max(config.e_min, center), config.e_max)
        if _too_close(float(h), float(e), measurements, h_thresh=0.04, e_thresh=0.04):
            continue
        rows.append((float(leader_fn(float(h), float(e))), float(h), float(e)))
    rows.sort(key=lambda item: -item[0])
    out: List[Dict[str, Any]] = []
    for _, h, e in rows[:n_points]:
        out.append(
            {
                "h": float(h),
                "k": float(h),
                "E": float(e),
                "mode": "trap_refine",
                "audit_policy": "",
                "audit_injected": False,
                "audit_probe_family": "bright_branch",
                "is_falsification_probe": False,
                "expected_contrast": 0.0,
                "count_time": 60.0,
            }
        )
    return out


def _trap_simulate_measurements(
    points: Sequence[Dict[str, Any]],
    config: MultimodelTrapConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    truth_fn = _trap_models(config)["M4: Ridge+pocket"]
    out: List[Dict[str, Any]] = []
    for pt in points:
        h = float(pt["h"])
        e = float(pt["E"])
        obs = float(rng.normal(float(truth_fn(h, e)), config.sigma))
        out.append(
            {
                "h": h,
                "k": float(pt.get("k", h)),
                "E": e,
                "intensity": obs,
                "uncertainty": float(config.sigma),
                "variance": float(config.sigma**2),
                "counts": float("nan"),
                "coverage": False,
                "symmetry": False,
                "human_hint": False,
                "llm_hint": bool(pt.get("llm_hint")),
                "llm_reason": str(pt.get("llm_reason", ""))[:200],
                "mode": pt.get("mode", "trap_refine"),
                "llm_batch_idx": pt.get("llm_batch_idx"),
                "llm_provider": pt.get("llm_provider"),
                "llm_decision_reason": str(pt.get("llm_decision_reason", ""))[:200],
                "loggp_hint": False,
                "count_time": float(pt.get("count_time", 60.0)),
                "dwell": False,
                "audit_policy": pt.get("audit_policy"),
                "audit_injected": bool(pt.get("audit_injected")),
                "audit_probe_family": pt.get("audit_probe_family"),
                "is_falsification_probe": bool(pt.get("is_falsification_probe")),
                "expected_contrast": float(pt.get("expected_contrast", 0.0)),
                "silent_data_active": False,
            }
        )
    return out


def build_hidden_gap_seed_points(
    true_model: SquareLatticeDispersion,
    total_points: int,
    hmax: float,
    config: HiddenGapInitConfig,
) -> List[Dict[str, Any]]:
    target_points = max(1, int(math.ceil(total_points / max(1, config.coarse_stride))))
    init_hmin = min(max(config.init_hmin, H_RANGE_MIN), hmax)
    init_emin = min(max(config.init_emin, 0.5), LOGGP_E_MAX)
    stressed: List[Dict[str, Any]] = []
    h_vals = np.linspace(init_hmin, hmax, max(3, int(config.grid_h)))
    e_vals = np.linspace(init_emin, LOGGP_E_MAX, max(3, int(config.grid_e)))
    seed_grid: List[Tuple[float, float]] = []
    for j, e in enumerate(e_vals):
        h_iter = h_vals if j % 2 == 0 else h_vals[::-1]
        for h in h_iter:
            seed_grid.append((float(h), float(e)))

    kept = 0
    for h, e in seed_grid:
        if kept >= target_points:
            break
        if not TAS.is_accessible(h, h, e):
            continue
        row = {
            "h": h,
            "k": h,
            "E": e,
            "mode": "fixed_sparse_seed",
            "audit_policy": "",
            "audit_injected": False,
            "is_falsification_probe": False,
            "audit_probe_family": _classify_probe_family(h, e),
            "expected_contrast": 0.0,
            "count_time": 60.0,
            "hidden_gap_init_stress": True,
        }
        stressed.append(row)
        kept += 1
    return stressed


def simulate_measurements_with_silent_data(
    measurement_points: List[Dict[str, Any]],
    true_model: SquareLatticeDispersion,
    silent_data: SilentDataModel,
    count_time: float = 60.0,
) -> List[Dict[str, Any]]:
    patched = []
    for pt in measurement_points:
        pt_local = dict(pt)
        pt_local.setdefault("count_time", count_time)
        patched.append(pt_local)

    results = []
    for pt in patched:
        true_intensity = silent_data.intensity(pt["h"], pt["k"], pt["E"])
        local_time = pt.get("count_time", count_time)
        counts = np.random.poisson(max(1, true_intensity * local_time))
        i_meas = counts / local_time
        sigma_poisson = math.sqrt(max(counts, 1)) / local_time
        sigma_syst = max(0.03 * i_meas, 0.02)
        final_sigma = max(math.sqrt(sigma_poisson**2 + sigma_syst**2), 1e-4)
        row = {
            "h": pt["h"],
            "k": pt["k"],
            "E": pt["E"],
            "intensity": i_meas,
            "uncertainty": final_sigma,
            "counts": counts,
            "coverage": bool(pt.get("coverage")),
            "symmetry": bool(pt.get("symmetry")),
            "human_hint": bool(pt.get("human_hint")),
            "llm_hint": bool(pt.get("llm_hint")),
            "llm_reason": str(pt.get("reason", ""))[:200],
            "mode": pt.get("mode"),
            "llm_batch_idx": pt.get("llm_batch_idx"),
            "llm_provider": pt.get("llm_provider"),
            "llm_decision_reason": str(pt.get("llm_decision_reason", ""))[:200],
            "loggp_hint": bool(pt.get("loggp_hint")),
            "count_time": local_time,
            "dwell": bool(pt.get("dwell")),
            "audit_policy": pt.get("audit_policy"),
            "audit_injected": bool(pt.get("audit_injected")),
            "audit_probe_family": pt.get("audit_probe_family"),
            "is_falsification_probe": bool(pt.get("is_falsification_probe")),
            "expected_contrast": float(pt.get("expected_contrast", 0.0)),
            "silent_data_active": bool(
                silent_data.config.enabled
                and silent_data.measurement_index < silent_data.config.until_measurement
                and silent_data._silent_region(pt["h"], pt["E"])
            ),
        }
        results.append(row)
        silent_data.advance(1)
    return results


def choose_audit_points(
    policy: str,
    state: Dict[str, Any],
    menu: Sequence[Dict[str, Any]],
    menu_lookup: Dict[str, Dict[str, Any]],
    max_inject: int,
    rng: np.random.Generator,
    llm_command: Optional[str] = None,
    mailbox_url: Optional[str] = None,
    mailbox_token: Optional[str] = None,
    mailbox_run_id: Optional[str] = None,
    mailbox_batch_index: Optional[int] = None,
    mailbox_batch_key: Optional[str] = None,
    mailbox_checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    if max_inject <= 0 or not menu or policy == "none":
        return [], ""

    selected_ids: List[str] = []
    reason = ""

    if policy == "rule":
        need_gap_probe = state["entropy"] >= state["entropy_trigger"] and state["gap_coverage"] < state["gap_coverage_trigger"]
        if need_gap_probe:
            for row in menu:
                if row["probe_family"] == "gap_check" and row["is_falsification_probe"]:
                    selected_ids.append(row["id"])
                    break
        if state["wrong_leader_streak"] >= state["wrong_leader_trigger"] or state["margin"] < state["margin_trigger"]:
            for row in menu:
                if row["id"] in selected_ids:
                    continue
                if row["is_falsification_probe"]:
                    selected_ids.append(row["id"])
                    if len(selected_ids) >= max_inject:
                        break
        if not selected_ids and state["audit_recommended"]:
            selected_ids.append(menu[0]["id"])
        reason = "rule_audit"
    elif policy in {"max_disagreement", "max_disagreement_all"}:
        ranked = sorted(
            menu,
            key=lambda row: (
                not bool(row["is_falsification_probe"]),
                -float(row["expected_contrast"]),
                -float(row["z"]),
                -float(row["score"]),
            ),
        )
        for row in ranked:
            if row["id"] in selected_ids:
                continue
            if len(selected_ids) >= max_inject:
                break
            selected_ids.append(row["id"])
        reason = f"{policy}_audit"
    elif policy == "random":
        top_k = min(len(menu), max(state["random_topk"], max_inject))
        choices = list(rng.choice(top_k, size=min(max_inject, top_k), replace=False))
        selected_ids = [menu[int(idx)]["id"] for idx in choices]
        reason = "random_audit"
    elif policy == "llm":
        prompt = build_llm_policy_prompt(state, menu, max_inject=max_inject, batch_size=state["batch_size"])
        if mailbox_url and mailbox_token and mailbox_batch_key:
            payload = _mailbox_llm_decision(
                prompt=prompt,
                mailbox_url=mailbox_url,
                mailbox_token=mailbox_token,
                run_id=str(mailbox_run_id or ""),
                batch_index=int(mailbox_batch_index or 0),
                batch_key=mailbox_batch_key,
                checkpoint=mailbox_checkpoint or {},
            )
        elif llm_command:
            payload = _run_llm_command(llm_command, prompt)
        else:
            raise RuntimeError("LLM audit policy requires either --llm-command or mailbox settings")
        inject_ids = payload.get("inject_ids", [])
        if isinstance(inject_ids, list):
            for pid in inject_ids:
                if isinstance(pid, str) and pid in menu_lookup and pid not in selected_ids:
                    selected_ids.append(pid)
                    if len(selected_ids) >= max_inject:
                        break
        reason = str(payload.get("reason", "llm_audit"))[:200]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    points: List[Dict[str, Any]] = []
    for pid in selected_ids[:max_inject]:
        row = menu_lookup[pid]
        point = {
            "h": float(row["h"]),
            "k": float(row["k"]),
            "E": float(row["E"]),
            "audit_policy": policy,
            "audit_injected": True,
            "audit_probe_family": row["probe_family"],
            "is_falsification_probe": bool(row["is_falsification_probe"]),
            "expected_contrast": float(row["expected_contrast"]),
            "llm_hint": policy == "llm",
            "llm_reason": reason if policy == "llm" else "",
            "mode": f"{policy}_audit",
            "count_time": 60.0,
        }
        points.append(point)
    return points, reason


def choose_bilayer_overseer_action(
    policy: str,
    state: Dict[str, Any],
    menu: Sequence[Dict[str, Any]],
    menu_lookup: Dict[str, Dict[str, Any]],
    batch_size: int,
    max_inject: int,
    rng: np.random.Generator,
    llm_command: Optional[str] = None,
    mailbox_url: Optional[str] = None,
    mailbox_token: Optional[str] = None,
    mailbox_run_id: Optional[str] = None,
    mailbox_batch_index: Optional[int] = None,
    mailbox_batch_key: Optional[str] = None,
    mailbox_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    forced_mode: Optional[str] = None
    if int(state.get("mode_run_len", 0)) < int(state.get("min_run_points", 0)):
        forced_mode = str(state.get("last_mode", "physics"))
    elif float(state.get("leader_post", 0.0)) < float(state.get("phase3_threshold", 0.99)) and int(
        state.get("since_loggp", 0)
    ) >= int(state.get("loggp_forced_interval", 6)):
        forced_mode = "loggp_active"
    elif float(state.get("leader_post", 0.0)) >= 0.9 and int(state.get("since_loggp", 0)) >= int(
        state.get("loggp_verify_interval", 6)
    ):
        forced_mode = "loggp_active"

    if policy in {"hybrid", "max_disagreement"}:
        if forced_mode:
            return {"selected_mode": forced_mode, "inject_ids": [], "reason": f"{policy}_guardrail"}
        inject_ids: List[str] = []
        if policy == "max_disagreement":
            ranked = sorted(
                menu,
                key=lambda row: (
                    not bool(row["is_falsification_probe"]),
                    -float(row["expected_contrast"]),
                    -float(row["z"]),
                    -float(row["score"]),
                ),
            )
            for row in ranked:
                if len(inject_ids) >= max_inject:
                    break
                inject_ids.append(str(row["id"]))
        if (
            float(state.get("entropy", 0.0)) >= float(state.get("entropy_trigger", 0.0))
            or float(state.get("gap_coverage", 1.0)) < float(state.get("gap_coverage_trigger", 0.0))
            or float(state.get("margin", 1.0)) < float(state.get("margin_trigger", 0.0))
        ):
            return {
                "selected_mode": "loggp_active",
                "inject_ids": inject_ids if policy == "max_disagreement" else [],
                "reason": "hybrid_explore" if policy == "hybrid" else "max_disagreement_explore",
            }
        return {
            "selected_mode": "physics",
            "inject_ids": inject_ids if policy == "max_disagreement" else [],
            "reason": "hybrid_refine" if policy == "hybrid" else "max_disagreement_refine",
        }

    if policy != "llm":
        raise ValueError(f"Unknown bilayer overseer policy: {policy}")

    if forced_mode:
        return {"selected_mode": forced_mode, "inject_ids": [], "reason": "guardrail"}

    prompt = build_llm_policy_prompt(state, menu, max_inject=max_inject, batch_size=batch_size)
    if mailbox_url and mailbox_token and mailbox_batch_key:
        payload = _mailbox_llm_decision(
            prompt=prompt,
            mailbox_url=mailbox_url,
            mailbox_token=mailbox_token,
            run_id=str(mailbox_run_id or ""),
            batch_index=int(mailbox_batch_index or 0),
            batch_key=mailbox_batch_key,
            checkpoint=mailbox_checkpoint or {},
        )
    elif llm_command:
        payload = _run_llm_command(llm_command, prompt)
    else:
        raise RuntimeError("LLM bilayer policy requires either --llm-command or mailbox settings")

    action = _parse_overseer_mode_payload(payload, expected_batch_size=batch_size, max_inject=max_inject)
    inject_ids: List[str] = []
    for pid in action["inject_ids"]:
        if pid in menu_lookup and pid not in inject_ids:
            inject_ids.append(pid)
    action["inject_ids"] = inject_ids
    return action


def _decisive_true(results: Dict[str, Any], true_model_name: str, decisive_ratio: float) -> bool:
    ranked = _model_rank(results)
    if not ranked or ranked[0][0] != true_model_name:
        return False
    if len(ranked) == 1:
        return True
    runner = max(ranked[1][1], 1e-12)
    return bool(ranked[0][1] / runner >= decisive_ratio)


def _batch_summary(results: Dict[str, Any], true_model_name: str, batch_size: int) -> Dict[str, Any]:
    ranked = _model_rank(results)
    leader, leader_post = ranked[0]
    runner_up, runner_up_post = ranked[1]
    return {
        "leader": leader,
        "leader_post": leader_post,
        "runner_up": runner_up,
        "runner_up_post": runner_up_post,
        "margin": leader_post - runner_up_post,
        "entropy": _posterior_entropy(results),
        "leader_is_true": leader == true_model_name,
        "batch_size": batch_size,
    }


def _campaign_run_id(mailbox_run_id: str, policy: str, seed: int) -> str:
    return f"{mailbox_run_id}_{policy}_seed{seed:03d}"


def run_single_policy(
    policy: str,
    seed: int,
    output_dir: Path,
    total_measurements: int,
    batch_size: int,
    loggp_grid_points: int,
    loggp_active_points: int,
    phase3_threshold: float,
    audit_points: int,
    llm_command: Optional[str],
    silent_data_config: SilentDataConfig,
    hidden_gap_init_config: HiddenGapInitConfig,
    loggp_init_from: Optional[str],
    use_bumps: bool,
    mailbox_url: Optional[str],
    mailbox_token: Optional[str],
    mailbox_run_id: Optional[str],
    demo_hmin: Optional[float],
    demo_hmax: Optional[float],
) -> Dict[str, Any]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    init_tas(BASE_HAS_RESOLUTION)
    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    true_model_name = candidates[-1]["name"]
    true_model = SquareLatticeDispersion(**candidates[-1]["params"])
    silent_model = SilentDataModel(true_model, silent_data_config)
    hmin = H_RANGE_MIN if demo_hmin is None else float(demo_hmin)
    hmax = 1.7 if demo_hmax is None else float(demo_hmax)

    measurements: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    llm_campaign_run_id = (
        _campaign_run_id(mailbox_run_id, policy, seed)
        if mailbox_run_id and policy == "llm"
        else None
    )

    if hidden_gap_init_config.enabled:
        seed_points = build_hidden_gap_seed_points(
            true_model=true_model,
            total_points=loggp_grid_points + loggp_active_points,
            hmax=hmax,
            config=hidden_gap_init_config,
        )
        for row in seed_points:
            row["audit_policy"] = policy
    elif loggp_init_from:
        payload = json.loads(Path(loggp_init_from).read_text())
        init_measurements = payload.get("measurements", payload)
        seed_points = [
            {
                "h": float(meas["h"]),
                "k": float(meas.get("k", meas["h"])),
                "E": float(meas["E"]),
                "mode": meas.get("mode", "loggp_active"),
                "audit_policy": policy,
                "audit_injected": False,
                "is_falsification_probe": False,
                "audit_probe_family": _classify_probe_family(float(meas["h"]), float(meas["E"])),
                "expected_contrast": 0.0,
                "count_time": float(meas.get("count_time", 60.0)),
            }
            for meas in init_measurements[: min(len(init_measurements), loggp_grid_points + loggp_active_points)]
        ]
    else:
        loggp_meas, _, _ = run_loggp_phase(
            true_model,
            n_measurements=loggp_grid_points + loggp_active_points,
            hmin=hmin,
            hmax=hmax,
            emin=0.5,
            emax=LOGGP_E_MAX,
        )
        seed_points = [
            {
                "h": float(meas["h"]),
                "k": float(meas.get("k", meas["h"])),
                "E": float(meas["E"]),
                "mode": meas.get("mode", "loggp_active"),
                "audit_policy": policy,
                "audit_injected": False,
                "is_falsification_probe": False,
                "audit_probe_family": _classify_probe_family(float(meas["h"]), float(meas["E"])),
                "expected_contrast": 0.0,
            }
            for meas in loggp_meas[: min(len(loggp_meas), loggp_grid_points + loggp_active_points)]
        ]
    seeded = simulate_measurements_with_silent_data(seed_points, true_model=true_model, silent_data=silent_model)
    measurements.extend(seeded)

    wrong_leader_streak = 0
    decisive_idx: Optional[int] = None
    recovered = False
    saw_wrong_leader = False

    while len(measurements) < total_measurements:
        remaining = total_measurements - len(measurements)
        current_batch_size = min(batch_size, remaining)
        results = sanitize_results(discriminate_models(measurements, candidates, use_bumps=use_bumps))
        summary = _batch_summary(results, true_model_name=true_model_name, batch_size=current_batch_size)
        if summary["leader"] != true_model_name:
            wrong_leader_streak += 1
            saw_wrong_leader = True
        else:
            if wrong_leader_streak > 0:
                recovered = True
            wrong_leader_streak = 0

        if policy == "max_disagreement_all":
            menu, menu_lookup = build_all_model_disagreement_menu(
                results,
                measurements,
                h_bounds=(max(0.5, hmin), hmax),
                e_bounds=(0.5, LOGGP_E_MAX),
                menu_size=12,
            )
        else:
            menu, menu_lookup = build_discrimination_menu(
                results,
                measurements,
                h_bounds=(max(0.5, hmin), hmax),
                e_bounds=(0.5, LOGGP_E_MAX),
                menu_size=12,
            )
        gap_coverage = _coverage_ratio(measurements)
        state = {
            **summary,
            "gap_coverage": gap_coverage,
            "entropy_trigger": 0.35,
            "gap_coverage_trigger": 0.16,
            "margin_trigger": 0.20,
            "wrong_leader_trigger": 1,
            "wrong_leader_streak": wrong_leader_streak,
            "random_topk": 6,
            "audit_recommended": bool(summary["entropy"] >= 0.35 or gap_coverage < 0.16),
            "silent_data_active": bool(silent_data_config.enabled and len(measurements) < silent_data_config.until_measurement),
            "wrong_leader_dwell": sum(
                int(item["batch_size"]) for item in batch_history if item["leader"] != true_model_name
            ),
            "batch_size": current_batch_size,
        }
        inject_pts, inject_reason = choose_audit_points(
            policy=policy,
            state=state,
            menu=menu,
            menu_lookup=menu_lookup,
            max_inject=min(audit_points, current_batch_size),
            rng=rng,
            llm_command=llm_command,
            mailbox_url=mailbox_url,
            mailbox_token=mailbox_token,
            mailbox_run_id=llm_campaign_run_id,
            mailbox_batch_index=len(batch_history),
            mailbox_batch_key=(
                f"{llm_campaign_run_id}_{len(batch_history):03d}"
                if llm_campaign_run_id
                else None
            ),
            mailbox_checkpoint={
                "policy": policy,
                "seed": seed,
                "batch_index": len(batch_history),
                "n_measurements": len(measurements),
                "recent_measurements": measurements[-20:],
                "menu": menu,
                "state": state,
                "campaign_run_id": llm_campaign_run_id,
            },
        )
        inject_pts = [
            pt for pt in inject_pts if not _too_close(float(pt["h"]), float(pt["E"]), measurements)
        ][:current_batch_size]

        base_needed = max(current_batch_size - len(inject_pts), 0)
        physics_batch: List[Dict[str, Any]] = []
        if base_needed > 0:
            ranked = _model_rank(results)
            best_model = results[ranked[0][0]]["fitted_model"]
            best_candidate = {
                "name": ranked[0][0],
                "params": {
                    "J1": float(getattr(best_model, "J1", 0.0)),
                    "J2": float(getattr(best_model, "J2", 0.0)),
                    "D": float(getattr(best_model, "D", 0.0)),
                    "background": float(getattr(best_model, "background", 0.0)),
                },
            }
            planned = plan_measurements(
                [best_candidate],
                existing_points=measurements + inject_pts,
                measurement_history=measurements,
                n_points=base_needed,
                hmin=hmin,
                hmax=hmax,
            )
            for pt in planned[:base_needed]:
                row = dict(pt)
                row["mode"] = "physics"
                row["audit_policy"] = policy
                row["audit_injected"] = False
                row["is_falsification_probe"] = False
                row["audit_probe_family"] = _classify_probe_family(float(row["h"]), float(row["E"]))
                row["expected_contrast"] = 0.0
                physics_batch.append(row)

        batch_points = inject_pts + physics_batch
        new_meas = simulate_measurements_with_silent_data(
            batch_points,
            true_model=true_model,
            silent_data=silent_model,
        )
        measurements.extend(new_meas)

        results_after = sanitize_results(discriminate_models(measurements, candidates, use_bumps=use_bumps))
        batch_record = {
            **_batch_summary(results_after, true_model_name=true_model_name, batch_size=len(new_meas)),
            "n_measurements": len(measurements),
            "audit_injections": len(inject_pts),
            "audit_reason": inject_reason,
            "falsification_probes": sum(1 for meas in new_meas if meas.get("is_falsification_probe")),
            "gap_coverage": _coverage_ratio(measurements),
            "policy": policy,
        }
        batch_history.append(batch_record)

        if decisive_idx is None and _decisive_true(results_after, true_model_name, decisive_ratio=100.0):
            decisive_idx = len(measurements)

    final_results = sanitize_results(discriminate_models(measurements, candidates, use_bumps=use_bumps))
    saw_wrong, wrong_dwell = _wrong_leader_state(batch_history, true_model_name=true_model_name)

    summary = {
        "policy": policy,
        "seed": seed,
        "true_model": true_model_name,
        "total_measurements": len(measurements),
        "time_to_decisive_correct": decisive_idx,
        "recovered_after_wrong_leader": bool(saw_wrong and recovered),
        "wrong_leader_dwell_time": wrong_dwell,
        "falsification_probe_batches": sum(1 for row in batch_history if row["falsification_probes"] > 0),
        "falsification_probe_fraction": (
            float(sum(1 for row in batch_history if row["falsification_probes"] > 0) / len(batch_history))
            if batch_history
            else 0.0
        ),
        "falsification_probe_count": sum(1 for meas in measurements if meas.get("is_falsification_probe")),
        "silent_data_success": bool(_decisive_true(final_results, true_model_name, decisive_ratio=100.0)),
        "final_posteriors": {name: float(row["posterior"]) for name, row in final_results.items()},
        "batch_history": batch_history,
        "measurements": measurements,
        "silent_data_config": {
            "enabled": silent_data_config.enabled,
            "until_measurement": silent_data_config.until_measurement,
            "attenuation": silent_data_config.attenuation,
            "background_boost": silent_data_config.background_boost,
        },
        "hidden_gap_init_config": {
            "enabled": hidden_gap_init_config.enabled,
            "coarse_stride": hidden_gap_init_config.coarse_stride,
            "init_hmin": hidden_gap_init_config.init_hmin,
            "init_emin": hidden_gap_init_config.init_emin,
            "grid_h": hidden_gap_init_config.grid_h,
            "grid_e": hidden_gap_init_config.grid_e,
        },
        "initialization_source": (
            "hidden_gap_sparse_seed"
            if hidden_gap_init_config.enabled
            else ("checkpoint" if loggp_init_from else "fresh_loggp")
        ),
        "initial_measurement_count": len(seeded),
        "llm_campaign_run_id": llm_campaign_run_id,
    }

    run_dir = output_dir / f"{policy}_seed{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if llm_campaign_run_id and mailbox_url and mailbox_token:
        campaign_manifest = {
            "run_id": llm_campaign_run_id,
            "mailbox_url": mailbox_url,
            "token_env": "LLM_MAILBOX_TOKEN",
            "mode": "overseer",
            "policy": policy,
            "seed": seed,
            "prepared_root": "run_logs/llm_prepared",
            "usage_log_dir": "run_logs/llm_usage_manager",
        }
        (run_dir / "llm_campaign_manifest.json").write_text(json.dumps(campaign_manifest, indent=2))
    return summary


def run_single_policy_ghost_optic(
    policy: str,
    seed: int,
    output_dir: Path,
    total_measurements: int,
    batch_size: int,
    audit_points: int,
    llm_command: Optional[str],
    mailbox_url: Optional[str],
    mailbox_token: Optional[str],
    mailbox_run_id: Optional[str],
    ghost_config: GhostOpticConfig,
) -> Dict[str, Any]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    true_model_name = "M_B: Acoustic+optic"
    measurements: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    llm_campaign_run_id = (
        _campaign_run_id(mailbox_run_id, policy, seed)
        if mailbox_run_id and policy == "llm"
        else None
    )

    seed_points = [
        {
            "h": 1.0,
            "k": 1.0,
            "E": float(energy),
            "mode": "ghost_seed",
            "audit_policy": policy,
            "audit_injected": False,
            "is_falsification_probe": False,
            "audit_probe_family": "ghost_seed",
            "expected_contrast": 0.0,
            "count_time": 60.0,
        }
        for energy in ghost_config.init_energies
    ]
    measurements.extend(_ghost_simulate_measurements(seed_points, config=ghost_config, rng=rng))

    wrong_leader_streak = 0
    decisive_idx: Optional[int] = None
    recovered = False
    saw_wrong_leader = False

    while len(measurements) < total_measurements:
        remaining = total_measurements - len(measurements)
        current_batch_size = min(batch_size, remaining)
        results = _ghost_model_posteriors(measurements, ghost_config)
        summary = _batch_summary(results, true_model_name=true_model_name, batch_size=current_batch_size)
        if summary["leader"] != true_model_name:
            wrong_leader_streak += 1
            saw_wrong_leader = True
        else:
            if wrong_leader_streak > 0:
                recovered = True
            wrong_leader_streak = 0

        menu, menu_lookup = _ghost_discrimination_menu(measurements, ghost_config, menu_size=12)
        falsify_coverage = _ghost_coverage_ratio(measurements, ghost_config)
        state = {
            **summary,
            "scenario": "ghost-optic",
            "gap_coverage": falsify_coverage,
            "entropy_trigger": 0.20,
            "gap_coverage_trigger": 0.10,
            "margin_trigger": 0.35,
            "wrong_leader_trigger": 1,
            "wrong_leader_streak": wrong_leader_streak,
            "random_topk": 6,
            "audit_recommended": bool(summary["entropy"] >= 0.20 or falsify_coverage < 0.10),
            "silent_data_active": False,
            "wrong_leader_dwell": sum(
                int(item["batch_size"]) for item in batch_history if item["leader"] != true_model_name
            ),
            "batch_size": current_batch_size,
        }
        if policy == "loggp":
            inject_pts = []
            inject_reason = "ghost_loggp_explore"
            batch_points = _ghost_plan_loggp_points(measurements, ghost_config, current_batch_size)
        else:
            inject_pts, inject_reason = choose_audit_points(
                policy=policy,
                state=state,
                menu=menu,
                menu_lookup=menu_lookup,
                max_inject=min(audit_points, current_batch_size),
                rng=rng,
                llm_command=llm_command,
                mailbox_url=mailbox_url,
                mailbox_token=mailbox_token,
                mailbox_run_id=llm_campaign_run_id,
                mailbox_batch_index=len(batch_history),
                mailbox_batch_key=(
                    f"{llm_campaign_run_id}_{len(batch_history):03d}"
                    if llm_campaign_run_id
                    else None
                ),
                mailbox_checkpoint={
                    "scenario": "ghost-optic",
                    "policy": policy,
                    "seed": seed,
                    "batch_index": len(batch_history),
                    "n_measurements": len(measurements),
                    "recent_measurements": measurements[-20:],
                    "menu": menu,
                    "state": state,
                    "campaign_run_id": llm_campaign_run_id,
                },
            )
            inject_pts = [
                pt
                for pt in inject_pts
                if not _too_close(float(pt["h"]), float(pt["E"]), measurements, h_thresh=0.2, e_thresh=0.35)
            ][:current_batch_size]

            base_needed = max(current_batch_size - len(inject_pts), 0)
            physics_batch = _ghost_plan_refinement_points(measurements + inject_pts, ghost_config, base_needed)
            for row in physics_batch:
                row["audit_policy"] = policy
            batch_points = inject_pts + physics_batch
        if not batch_points:
            logger.info(
                "Ghost scenario exhausted candidate set for policy=%s seed=%s at n=%d",
                policy,
                seed,
                len(measurements),
            )
            break
        new_meas = _ghost_simulate_measurements(batch_points, config=ghost_config, rng=rng)
        if not new_meas:
            logger.info(
                "Ghost scenario produced no new measurements for policy=%s seed=%s at n=%d",
                policy,
                seed,
                len(measurements),
            )
            break
        measurements.extend(new_meas)

        results_after = _ghost_model_posteriors(measurements, ghost_config)
        batch_record = {
            **_batch_summary(results_after, true_model_name=true_model_name, batch_size=len(new_meas)),
            "n_measurements": len(measurements),
            "audit_injections": len(inject_pts),
            "audit_reason": inject_reason,
            "falsification_probes": sum(1 for meas in new_meas if meas.get("is_falsification_probe")),
            "gap_coverage": _ghost_coverage_ratio(measurements, ghost_config),
            "policy": policy,
        }
        batch_history.append(batch_record)

        if decisive_idx is None and _decisive_true(results_after, true_model_name, decisive_ratio=100.0):
            decisive_idx = len(measurements)

    final_results = _ghost_model_posteriors(measurements, ghost_config)
    saw_wrong, wrong_dwell = _wrong_leader_state(batch_history, true_model_name=true_model_name)
    summary = {
        "policy": policy,
        "seed": seed,
        "true_model": true_model_name,
        "scenario": "ghost-optic",
        "total_measurements": len(measurements),
        "target_measurements": int(total_measurements),
        "time_to_decisive_correct": decisive_idx,
        "recovered_after_wrong_leader": bool(saw_wrong and recovered),
        "wrong_leader_dwell_time": wrong_dwell,
        "falsification_probe_batches": sum(1 for row in batch_history if row["falsification_probes"] > 0),
        "falsification_probe_fraction": (
            float(sum(1 for row in batch_history if row["falsification_probes"] > 0) / len(batch_history))
            if batch_history
            else 0.0
        ),
        "falsification_probe_count": sum(1 for meas in measurements if meas.get("is_falsification_probe")),
        "silent_data_success": bool(_decisive_true(final_results, true_model_name, decisive_ratio=100.0)),
        "final_posteriors": {name: float(row["posterior"]) for name, row in final_results.items()},
        "batch_history": batch_history,
        "measurements": measurements,
        "ghost_optic_config": {
            "energy_min": ghost_config.energy_min,
            "energy_max": ghost_config.energy_max,
            "gamma": ghost_config.gamma,
            "background": ghost_config.background,
            "acoustic_energy": ghost_config.acoustic_energy,
            "optic_energy": ghost_config.optic_energy,
            "acoustic_amplitude": ghost_config.acoustic_amplitude,
            "optic_fraction": ghost_config.optic_fraction,
            "leader_prior": ghost_config.leader_prior,
            "mixed_eta": ghost_config.mixed_eta,
            "init_energies": list(ghost_config.init_energies),
        },
        "initialization_source": "ghost_fixed_seed",
        "initial_measurement_count": len(seed_points),
        "llm_campaign_run_id": llm_campaign_run_id,
    }

    run_dir = output_dir / f"{policy}_seed{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if llm_campaign_run_id and mailbox_url and mailbox_token:
        campaign_manifest = {
            "run_id": llm_campaign_run_id,
            "mailbox_url": mailbox_url,
            "token_env": "LLM_MAILBOX_TOKEN",
            "mode": "overseer",
            "policy": policy,
            "seed": seed,
            "prepared_root": "run_logs/llm_prepared",
            "usage_log_dir": "run_logs/llm_usage_manager",
            "scenario": "ghost-optic",
        }
        (run_dir / "llm_campaign_manifest.json").write_text(json.dumps(campaign_manifest, indent=2))
    return summary


def run_single_policy_multimodel_trap(
    policy: str,
    seed: int,
    output_dir: Path,
    total_measurements: int,
    batch_size: int,
    audit_points: int,
    llm_command: Optional[str],
    mailbox_url: Optional[str],
    mailbox_token: Optional[str],
    mailbox_run_id: Optional[str],
    trap_config: MultimodelTrapConfig,
) -> Dict[str, Any]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    true_model_name = "M4: Ridge+pocket"
    measurements: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    llm_campaign_run_id = (
        _campaign_run_id(mailbox_run_id, policy, seed)
        if mailbox_run_id and policy == "llm"
        else None
    )

    seed_points = [
        {
            "h": float(h),
            "k": float(h),
            "E": float(e),
            "mode": "trap_seed",
            "audit_policy": policy,
            "audit_injected": False,
            "is_falsification_probe": False,
            "audit_probe_family": "trap_seed",
            "expected_contrast": 0.0,
            "count_time": 60.0,
        }
        for h, e in trap_config.seed_coords
    ]
    measurements.extend(_trap_simulate_measurements(seed_points, trap_config, rng))

    wrong_leader_streak = 0
    decisive_idx: Optional[int] = None
    recovered = False

    while len(measurements) < total_measurements:
        remaining = total_measurements - len(measurements)
        current_batch_size = min(batch_size, remaining)
        results = _trap_model_posteriors(measurements, trap_config)
        summary = _batch_summary(results, true_model_name=true_model_name, batch_size=current_batch_size)
        if summary["leader"] != true_model_name:
            wrong_leader_streak += 1
        else:
            if wrong_leader_streak > 0:
                recovered = True
            wrong_leader_streak = 0

        if policy in {"max_disagreement_all", "llm"}:
            menu, menu_lookup = _trap_candidate_menu(measurements, trap_config, top_two_only=False, menu_size=12)
        else:
            menu, menu_lookup = _trap_candidate_menu(measurements, trap_config, top_two_only=True, menu_size=12)
        coverage = _trap_coverage_ratio(measurements, trap_config)
        state = {
            **summary,
            "scenario": "multimodel-trap",
            "gap_coverage": coverage,
            "entropy_trigger": 0.30,
            "gap_coverage_trigger": 0.08,
            "margin_trigger": 0.30,
            "wrong_leader_trigger": 1,
            "wrong_leader_streak": wrong_leader_streak,
            "random_topk": 6,
            "audit_recommended": bool(summary["entropy"] >= 0.30 or coverage < 0.08),
            "silent_data_active": False,
            "wrong_leader_dwell": sum(int(item["batch_size"]) for item in batch_history if item["leader"] != true_model_name),
            "batch_size": current_batch_size,
            "ranked_models": _model_rank(results),
        }
        inject_pts, inject_reason = choose_audit_points(
            policy=policy,
            state=state,
            menu=menu,
            menu_lookup=menu_lookup,
            max_inject=min(audit_points, current_batch_size),
            rng=rng,
            llm_command=llm_command,
            mailbox_url=mailbox_url,
            mailbox_token=mailbox_token,
            mailbox_run_id=llm_campaign_run_id,
            mailbox_batch_index=len(batch_history),
            mailbox_batch_key=(f"{llm_campaign_run_id}_{len(batch_history):03d}" if llm_campaign_run_id else None),
            mailbox_checkpoint={
                "scenario": "multimodel-trap",
                "policy": policy,
                "seed": seed,
                "batch_index": len(batch_history),
                "n_measurements": len(measurements),
                "recent_measurements": measurements[-20:],
                "menu": menu,
                "state": state,
                "campaign_run_id": llm_campaign_run_id,
            },
        )
        inject_pts = [
            pt for pt in inject_pts
            if not _too_close(float(pt["h"]), float(pt["E"]), measurements, h_thresh=0.04, e_thresh=0.04)
        ][:current_batch_size]
        base_needed = max(current_batch_size - len(inject_pts), 0)
        physics_batch = _trap_plan_refinement_points(measurements + inject_pts, trap_config, base_needed)
        for row in physics_batch:
            row["audit_policy"] = policy
        batch_points = inject_pts + physics_batch
        if not batch_points:
            break

        new_meas = _trap_simulate_measurements(batch_points, trap_config, rng)
        measurements.extend(new_meas)
        results_after = _trap_model_posteriors(measurements, trap_config)
        batch_record = {
            **_batch_summary(results_after, true_model_name=true_model_name, batch_size=len(new_meas)),
            "n_measurements": len(measurements),
            "audit_injections": len(inject_pts),
            "audit_reason": inject_reason,
            "falsification_probes": sum(1 for meas in new_meas if meas.get("is_falsification_probe")),
            "gap_coverage": _trap_coverage_ratio(measurements, trap_config),
            "policy": policy,
        }
        batch_history.append(batch_record)

        if decisive_idx is None and _decisive_true(results_after, true_model_name, decisive_ratio=trap_config.decisive_ratio):
            decisive_idx = len(measurements)

    final_results = _trap_model_posteriors(measurements, trap_config)
    saw_wrong, wrong_dwell = _wrong_leader_state(batch_history, true_model_name=true_model_name)
    summary = {
        "policy": policy,
        "seed": seed,
        "true_model": true_model_name,
        "scenario": "multimodel-trap",
        "total_measurements": len(measurements),
        "target_measurements": int(total_measurements),
        "time_to_decisive_correct": decisive_idx,
        "recovered_after_wrong_leader": bool(saw_wrong and recovered),
        "wrong_leader_dwell_time": wrong_dwell,
        "falsification_probe_batches": sum(1 for row in batch_history if row["falsification_probes"] > 0),
        "falsification_probe_fraction": (
            float(sum(1 for row in batch_history if row["falsification_probes"] > 0) / len(batch_history))
            if batch_history
            else 0.0
        ),
        "falsification_probe_count": sum(1 for meas in measurements if meas.get("is_falsification_probe")),
        "silent_data_success": bool(_decisive_true(final_results, true_model_name, decisive_ratio=trap_config.decisive_ratio)),
        "final_posteriors": {name: float(row["posterior"]) for name, row in final_results.items()},
        "batch_history": batch_history,
        "measurements": measurements,
        "trap_config": {
            "sigma": trap_config.sigma,
            "background": trap_config.background,
            "ridge_amp": trap_config.ridge_amp,
            "pocket_amp": trap_config.pocket_amp,
            "seed_coords": [[float(h), float(e)] for h, e in trap_config.seed_coords],
            "decisive_ratio": trap_config.decisive_ratio,
        },
        "initialization_source": "trap_fixed_seed",
        "initial_measurement_count": len(seed_points),
        "llm_campaign_run_id": llm_campaign_run_id,
    }

    run_dir = output_dir / f"{policy}_seed{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if llm_campaign_run_id and mailbox_url and mailbox_token:
        campaign_manifest = {
            "run_id": llm_campaign_run_id,
            "mailbox_url": mailbox_url,
            "token_env": "LLM_MAILBOX_TOKEN",
            "mode": "overseer",
            "policy": policy,
            "seed": seed,
            "prepared_root": "run_logs/llm_prepared",
            "usage_log_dir": "run_logs/llm_usage_manager",
            "scenario": "multimodel-trap",
        }
        (run_dir / "llm_campaign_manifest.json").write_text(json.dumps(campaign_manifest, indent=2))
    return summary


def run_single_policy_bilayer_fm(
    policy: str,
    seed: int,
    output_dir: Path,
    total_measurements: int,
    batch_size: int,
    audit_points: int,
    llm_command: Optional[str],
    mailbox_url: Optional[str],
    mailbox_token: Optional[str],
    mailbox_run_id: Optional[str],
    bilayer_config: BilayerFMConfig,
) -> Dict[str, Any]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    true_model_name = "M_B: Bilayer FM"
    measurements: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    llm_campaign_run_id = (
        _campaign_run_id(mailbox_run_id, policy, seed)
        if mailbox_run_id and policy == "llm"
        else None
    )

    mono, _ = _bilayer_models(bilayer_config)
    seed_points: List[Dict[str, Any]] = []
    for h in bilayer_config.init_h:
        e0 = mono.omega_mono(h, h)
        for delta in bilayer_config.init_energy_offsets:
            energy = min(max(bilayer_config.e_min, e0 + float(delta)), bilayer_config.e_max)
            seed_points.append(
                {
                    "h": float(h),
                    "k": float(h),
                    "E": float(energy),
                    "mode": "bilayer_seed",
                    "audit_policy": policy,
                    "audit_injected": False,
                    "is_falsification_probe": False,
                    "audit_probe_family": "bilayer_seed",
                    "expected_contrast": 0.0,
                    "count_time": 60.0,
                }
            )
    measurements.extend(_bilayer_simulate_measurements(seed_points, bilayer_config, rng))

    wrong_leader_streak = 0
    decisive_idx: Optional[int] = None
    recovered = False
    saw_wrong_leader = False
    controller_state = {
        "last_mode": "physics",
        "mode_run_len": batch_size,
        "since_loggp": 0,
    }

    while len(measurements) < total_measurements:
        remaining = total_measurements - len(measurements)
        current_batch_size = min(batch_size, remaining)
        results = _bilayer_model_posteriors(measurements, bilayer_config)
        summary = _batch_summary(results, true_model_name=true_model_name, batch_size=current_batch_size)
        if summary["leader"] != true_model_name:
            wrong_leader_streak += 1
            saw_wrong_leader = True
        else:
            if wrong_leader_streak > 0:
                recovered = True
            wrong_leader_streak = 0

        menu, menu_lookup = _bilayer_discrimination_menu(measurements, bilayer_config, menu_size=12)
        coverage = _bilayer_coverage_ratio(measurements, bilayer_config)
        state = {
            **summary,
            "scenario": "bilayer-fm",
            "true_model_name": true_model_name,
            "gap_coverage": coverage,
            "entropy_trigger": 0.20,
            "gap_coverage_trigger": 0.10,
            "margin_trigger": 0.35,
            "wrong_leader_trigger": 1,
            "wrong_leader_streak": wrong_leader_streak,
            "random_topk": 6,
            "audit_recommended": bool(summary["entropy"] >= 0.20 or coverage < 0.10),
            "silent_data_active": False,
            "wrong_leader_dwell": sum(int(item["batch_size"]) for item in batch_history if item["leader"] != true_model_name),
            "batch_size": current_batch_size,
            "last_mode": controller_state["last_mode"],
            "mode_run_len": controller_state["mode_run_len"],
            "since_loggp": controller_state["since_loggp"],
            "min_run_points": 2,
            "phase3_threshold": 0.95,
            "loggp_forced_interval": 6,
            "loggp_verify_interval": 6,
        }

        if policy in {"hybrid", "llm"}:
            action = choose_bilayer_overseer_action(
                policy=policy,
                state=state,
                menu=menu,
                menu_lookup=menu_lookup,
                batch_size=current_batch_size,
                max_inject=min(audit_points, current_batch_size, 2),
                rng=rng,
                llm_command=llm_command,
                mailbox_url=mailbox_url,
                mailbox_token=mailbox_token,
                mailbox_run_id=llm_campaign_run_id,
                mailbox_batch_index=len(batch_history),
                mailbox_batch_key=(f"{llm_campaign_run_id}_{len(batch_history):03d}" if llm_campaign_run_id else None),
                mailbox_checkpoint={
                    "scenario": "bilayer-fm",
                    "policy": policy,
                    "seed": seed,
                    "batch_index": len(batch_history),
                    "n_measurements": len(measurements),
                    "recent_measurements": measurements[-20:],
                    "menu": menu,
                    "state": state,
                    "campaign_run_id": llm_campaign_run_id,
                },
            )
            mode = str(action["selected_mode"])
            inject_reason = str(action.get("reason", ""))
            inject_pts: List[Dict[str, Any]] = []
            for pid in action.get("inject_ids", []):
                row = menu_lookup.get(pid)
                if not row:
                    continue
                if _too_close(float(row["h"]), float(row["E"]), measurements, h_thresh=0.03, e_thresh=0.4):
                    continue
                inject_pts.append(
                    {
                        "h": float(row["h"]),
                        "k": float(row["k"]),
                        "E": float(row["E"]),
                        "audit_policy": policy,
                        "audit_injected": True,
                        "audit_probe_family": row["probe_family"],
                        "is_falsification_probe": bool(row["is_falsification_probe"]),
                        "expected_contrast": float(row["expected_contrast"]),
                        "llm_hint": policy == "llm",
                        "llm_reason": inject_reason if policy == "llm" else "",
                        "mode": "llm_audit" if policy == "llm" else "hybrid_audit",
                        "count_time": 60.0,
                    }
                )
            inject_pts = inject_pts[:current_batch_size]
            base_needed = max(current_batch_size - len(inject_pts), 0)
            if mode == "loggp_active":
                base_batch = _bilayer_plan_hybrid_points(measurements + inject_pts, bilayer_config, base_needed)
            else:
                base_batch = _bilayer_plan_refinement_points(measurements + inject_pts, bilayer_config, base_needed)
                for row in base_batch:
                    row["audit_policy"] = policy
            batch_points = inject_pts + base_batch
            controller_state["since_loggp"] = 0 if mode == "loggp_active" else controller_state["since_loggp"] + len(batch_points)
            controller_state["mode_run_len"] = (
                controller_state["mode_run_len"] + len(batch_points)
                if controller_state["last_mode"] == mode
                else len(batch_points)
            )
            controller_state["last_mode"] = mode
        else:
            inject_pts, inject_reason = choose_audit_points(
                policy=policy,
                state=state,
                menu=menu,
                menu_lookup=menu_lookup,
                max_inject=min(audit_points, current_batch_size),
                rng=rng,
                llm_command=llm_command,
                mailbox_url=mailbox_url,
                mailbox_token=mailbox_token,
                mailbox_run_id=llm_campaign_run_id,
                mailbox_batch_index=len(batch_history),
                mailbox_batch_key=(f"{llm_campaign_run_id}_{len(batch_history):03d}" if llm_campaign_run_id else None),
                mailbox_checkpoint={
                    "scenario": "bilayer-fm",
                    "policy": policy,
                    "seed": seed,
                    "batch_index": len(batch_history),
                    "n_measurements": len(measurements),
                    "recent_measurements": measurements[-20:],
                    "menu": menu,
                    "state": state,
                    "campaign_run_id": llm_campaign_run_id,
                },
            )
            inject_pts = [
                pt
                for pt in inject_pts
                if not _too_close(float(pt["h"]), float(pt["E"]), measurements, h_thresh=0.03, e_thresh=0.4)
            ][:current_batch_size]
            base_needed = max(current_batch_size - len(inject_pts), 0)
            physics_batch = _bilayer_plan_refinement_points(measurements + inject_pts, bilayer_config, base_needed)
            for row in physics_batch:
                row["audit_policy"] = policy
            batch_points = inject_pts + physics_batch
            controller_state["since_loggp"] += len(batch_points)
            controller_state["mode_run_len"] = (
                controller_state["mode_run_len"] + len(batch_points)
                if controller_state["last_mode"] == "physics"
                else len(batch_points)
            )
            controller_state["last_mode"] = "physics"

        new_meas = _bilayer_simulate_measurements(batch_points, bilayer_config, rng)
        measurements.extend(new_meas)

        results_after = _bilayer_model_posteriors(measurements, bilayer_config)
        batch_record = {
            **_batch_summary(results_after, true_model_name=true_model_name, batch_size=len(new_meas)),
            "n_measurements": len(measurements),
            "audit_injections": sum(1 for pt in batch_points if pt.get("audit_injected")),
            "audit_reason": inject_reason,
            "falsification_probes": sum(1 for meas in new_meas if meas.get("is_falsification_probe")),
            "optic_region_hits": sum(1 for meas in new_meas if _bilayer_optic_region_hit(meas, bilayer_config)),
            "gap_coverage": _bilayer_coverage_ratio(measurements, bilayer_config),
            "policy": policy,
            "selected_mode": controller_state["last_mode"],
        }
        batch_history.append(batch_record)

        if decisive_idx is None and _decisive_true(results_after, true_model_name, decisive_ratio=100.0):
            decisive_idx = len(measurements)

    final_results = _bilayer_model_posteriors(measurements, bilayer_config)
    saw_wrong, wrong_dwell = _wrong_leader_state(batch_history, true_model_name=true_model_name)
    summary = {
        "policy": policy,
        "seed": seed,
        "true_model": true_model_name,
        "scenario": "bilayer-fm",
        "total_measurements": len(measurements),
        "time_to_decisive_correct": decisive_idx,
        "recovered_after_wrong_leader": bool(saw_wrong and recovered),
        "wrong_leader_dwell_time": wrong_dwell,
        "falsification_probe_batches": sum(1 for row in batch_history if row["falsification_probes"] > 0),
        "falsification_probe_fraction": (
            float(sum(1 for row in batch_history if row["falsification_probes"] > 0) / len(batch_history))
            if batch_history
            else 0.0
        ),
        "falsification_probe_count": sum(1 for meas in measurements if meas.get("is_falsification_probe")),
        "optic_region_hit_batches": sum(1 for row in batch_history if row["optic_region_hits"] > 0),
        "optic_region_hit_fraction": (
            float(sum(1 for row in batch_history if row["optic_region_hits"] > 0) / len(batch_history))
            if batch_history
            else 0.0
        ),
        "optic_region_hit_count": sum(1 for meas in measurements if _bilayer_optic_region_hit(meas, bilayer_config)),
        "silent_data_success": bool(_decisive_true(final_results, true_model_name, decisive_ratio=100.0)),
        "final_posteriors": {name: float(row["posterior"]) for name, row in final_results.items()},
        "batch_history": batch_history,
        "measurements": measurements,
        "bilayer_fm_config": {
            "S": bilayer_config.S,
            "J_par": bilayer_config.J_par,
            "J_perp": bilayer_config.J_perp,
            "D": bilayer_config.D,
            "z_bi": bilayer_config.z_bi,
            "z_perp": bilayer_config.z_perp,
            "gamma": bilayer_config.gamma,
            "amp": bilayer_config.amp,
            "background": bilayer_config.background,
            "L_fixed": bilayer_config.L_fixed,
        },
        "initialization_source": "bilayer_fixed_seed",
        "initial_measurement_count": len(seed_points),
        "llm_campaign_run_id": llm_campaign_run_id,
    }

    run_dir = output_dir / f"{policy}_seed{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if llm_campaign_run_id and mailbox_url and mailbox_token:
        campaign_manifest = {
            "run_id": llm_campaign_run_id,
            "mailbox_url": mailbox_url,
            "token_env": "LLM_MAILBOX_TOKEN",
            "mode": "overseer",
            "policy": policy,
            "seed": seed,
            "prepared_root": "run_logs/llm_prepared",
            "usage_log_dir": "run_logs/llm_usage_manager",
            "scenario": "bilayer-fm",
        }
        (run_dir / "llm_campaign_manifest.json").write_text(json.dumps(campaign_manifest, indent=2))
    return summary


def summarize_runs(run_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in run_summaries:
        grouped.setdefault(str(row["policy"]), []).append(row)

    out: Dict[str, Any] = {}
    for policy, rows in grouped.items():
        decisive_vals = [row["time_to_decisive_correct"] for row in rows if row["time_to_decisive_correct"] is not None]
        out[policy] = {
            "n_runs": len(rows),
            "time_to_decisive_correct_mean": float(np.mean(decisive_vals)) if decisive_vals else None,
            "time_to_decisive_correct_std": float(np.std(decisive_vals)) if decisive_vals else None,
            "recover_after_wrong_leader_rate": float(np.mean([bool(r["recovered_after_wrong_leader"]) for r in rows])),
            "wrong_leader_dwell_time_mean": float(np.mean([float(r["wrong_leader_dwell_time"]) for r in rows])),
            "falsification_probe_fraction_mean": float(np.mean([float(r["falsification_probe_fraction"]) for r in rows])),
            "optic_region_hit_fraction_mean": float(np.mean([float(r.get("optic_region_hit_fraction", 0.0)) for r in rows])),
            "silent_data_success_rate": float(np.mean([bool(r["silent_data_success"]) for r in rows])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--scenario",
        choices=["spinwave", "ghost-optic", "bilayer-fm", "multimodel-trap"],
        default="spinwave",
        help="Select the benchmark family to run.",
    )
    parser.add_argument("--policies", nargs="+", default=["none", "rule", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--total-measurements", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--loggp-grid-points", type=int, default=31)
    parser.add_argument("--loggp-active-points", type=int, default=15)
    parser.add_argument("--phase3-threshold", type=float, default=0.95)
    parser.add_argument("--audit-points", type=int, default=2)
    parser.add_argument("--loggp-init-from", type=str, default="paper/data/closed_loop_checkpoint_afm90_grid7_init.json")
    parser.add_argument("--use-bumps", action="store_true", default=True)
    parser.add_argument("--no-use-bumps", dest="use_bumps", action="store_false")
    parser.add_argument("--llm-command", type=str, default=None)
    parser.add_argument("--llm-mailbox-url", type=str, default=None)
    parser.add_argument("--llm-mailbox-token", type=str, default=None)
    parser.add_argument("--llm-mailbox-run-id", type=str, default=None)
    parser.add_argument("--demo-hmin", type=float, default=None)
    parser.add_argument("--demo-hmax", type=float, default=None)
    parser.add_argument("--silent-data", action="store_true", default=False)
    parser.add_argument("--silent-data-until", type=int, default=45)
    parser.add_argument("--silent-data-attenuation", type=float, default=0.25)
    parser.add_argument("--silent-data-background-boost", type=float, default=0.30)
    parser.add_argument(
        "--stress-regime",
        choices=["none", "hidden-gap-init"],
        default="none",
        help="Optional physically motivated stress mode for ablation runs.",
    )
    parser.add_argument(
        "--hidden-gap-coarse-stride",
        type=int,
        default=2,
        help="Reduce the number of initialization measurements by this factor under hidden-gap-init stress.",
    )
    parser.add_argument(
        "--hidden-gap-init-hmin",
        type=float,
        default=0.75,
        help="Start hidden-gap-init Log-GP above this h to avoid direct low-q gap coverage.",
    )
    parser.add_argument(
        "--hidden-gap-init-emin",
        type=float,
        default=2.5,
        help="Start hidden-gap-init Log-GP above this energy to avoid direct low-energy gap coverage.",
    )
    parser.add_argument(
        "--hidden-gap-grid-h",
        type=int,
        default=9,
        help="Log-GP h-grid size for hidden-gap-init stress.",
    )
    parser.add_argument(
        "--hidden-gap-grid-e",
        type=int,
        default=7,
        help="Log-GP energy-grid size for hidden-gap-init stress.",
    )
    parser.add_argument("--ghost-optic-energy-min", type=float, default=0.0)
    parser.add_argument("--ghost-optic-energy-max", type=float, default=20.0)
    parser.add_argument("--ghost-optic-gamma", type=float, default=0.5)
    parser.add_argument("--ghost-optic-background", type=float, default=0.10)
    parser.add_argument("--ghost-optic-acoustic-energy", type=float, default=5.0)
    parser.add_argument("--ghost-optic-optic-energy", type=float, default=15.0)
    parser.add_argument("--ghost-optic-acoustic-amplitude", type=float, default=100.0)
    parser.add_argument("--ghost-optic-optic-fraction", type=float, default=0.05)
    parser.add_argument("--ghost-optic-leader-prior", type=float, default=0.95)
    parser.add_argument("--ghost-optic-mixed-eta", type=float, default=0.05)
    parser.add_argument("--ghost-optic-sigma-floor", type=float, default=0.10)
    parser.add_argument("--ghost-optic-sigma-scale", type=float, default=0.03)
    parser.add_argument("--ghost-optic-candidate-points", type=int, default=401)
    parser.add_argument("--ghost-optic-optic-probe-min", type=float, default=11.0)
    parser.add_argument(
        "--ghost-optic-init-energies",
        nargs="+",
        type=float,
        default=[4.25, 4.75, 5.25, 5.75],
    )
    parser.add_argument("--bilayer-s", type=float, default=1.0)
    parser.add_argument("--bilayer-j-par", type=float, default=2.0)
    parser.add_argument("--bilayer-j-perp", type=float, default=0.45)
    parser.add_argument("--bilayer-d", type=float, default=0.10)
    parser.add_argument("--bilayer-z-bi", type=float, default=0.35)
    parser.add_argument("--bilayer-z-perp", type=int, default=1)
    parser.add_argument("--bilayer-gamma", type=float, default=0.60)
    parser.add_argument("--bilayer-amp", type=float, default=100.0)
    parser.add_argument("--bilayer-background", type=float, default=0.10)
    parser.add_argument("--bilayer-l-fixed", type=float, default=0.16)
    parser.add_argument("--bilayer-h-min", type=float, default=0.0)
    parser.add_argument("--bilayer-h-max", type=float, default=0.5)
    parser.add_argument("--bilayer-e-min", type=float, default=0.0)
    parser.add_argument("--bilayer-e-max", type=float, default=22.0)
    parser.add_argument("--bilayer-candidate-h", type=int, default=101)
    parser.add_argument("--bilayer-candidate-e", type=int, default=181)
    parser.add_argument("--bilayer-leader-prior", type=float, default=0.992)
    parser.add_argument("--bilayer-sigma-floor", type=float, default=0.10)
    parser.add_argument("--bilayer-sigma-scale", type=float, default=0.03)
    parser.add_argument("--bilayer-optic-probe-min", type=float, default=6.0)
    parser.add_argument("--bilayer-optic-region-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--bilayer-init-h",
        nargs="+",
        type=float,
        default=[0.24],
    )
    parser.add_argument(
        "--bilayer-init-energy-offsets",
        nargs="+",
        type=float,
        default=[-0.03, 0.0, 0.03],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    silent_cfg = SilentDataConfig(
        enabled=bool(args.silent_data),
        until_measurement=int(args.silent_data_until),
        attenuation=float(args.silent_data_attenuation),
        background_boost=float(args.silent_data_background_boost),
    )
    hidden_gap_cfg = HiddenGapInitConfig(
        enabled=(args.stress_regime == "hidden-gap-init"),
        coarse_stride=max(1, int(args.hidden_gap_coarse_stride)),
        init_hmin=float(args.hidden_gap_init_hmin),
        init_emin=float(args.hidden_gap_init_emin),
        grid_h=max(3, int(args.hidden_gap_grid_h)),
        grid_e=max(3, int(args.hidden_gap_grid_e)),
    )
    ghost_cfg = GhostOpticConfig(
        energy_min=float(args.ghost_optic_energy_min),
        energy_max=float(args.ghost_optic_energy_max),
        gamma=float(args.ghost_optic_gamma),
        background=float(args.ghost_optic_background),
        acoustic_energy=float(args.ghost_optic_acoustic_energy),
        optic_energy=float(args.ghost_optic_optic_energy),
        acoustic_amplitude=float(args.ghost_optic_acoustic_amplitude),
        optic_fraction=float(args.ghost_optic_optic_fraction),
        sigma_floor=float(args.ghost_optic_sigma_floor),
        sigma_scale=float(args.ghost_optic_sigma_scale),
        leader_prior=float(args.ghost_optic_leader_prior),
        mixed_eta=float(args.ghost_optic_mixed_eta),
        candidate_points=max(51, int(args.ghost_optic_candidate_points)),
        optic_probe_min=float(args.ghost_optic_optic_probe_min),
        init_energies=tuple(float(v) for v in args.ghost_optic_init_energies),
    )
    bilayer_cfg = BilayerFMConfig(
        S=float(args.bilayer_s),
        J_par=float(args.bilayer_j_par),
        J_perp=float(args.bilayer_j_perp),
        D=float(args.bilayer_d),
        z_bi=float(args.bilayer_z_bi),
        z_perp=int(args.bilayer_z_perp),
        gamma=float(args.bilayer_gamma),
        amp=float(args.bilayer_amp),
        background=float(args.bilayer_background),
        L_fixed=float(args.bilayer_l_fixed),
        h_min=float(args.bilayer_h_min),
        h_max=float(args.bilayer_h_max),
        e_min=float(args.bilayer_e_min),
        e_max=float(args.bilayer_e_max),
        candidate_h=max(11, int(args.bilayer_candidate_h)),
        candidate_e=max(21, int(args.bilayer_candidate_e)),
        sigma_floor=float(args.bilayer_sigma_floor),
        sigma_scale=float(args.bilayer_sigma_scale),
        leader_prior=float(args.bilayer_leader_prior),
        optic_probe_min=float(args.bilayer_optic_probe_min),
        optic_region_tolerance=float(args.bilayer_optic_region_tolerance),
        init_h=tuple(float(v) for v in args.bilayer_init_h),
        init_energy_offsets=tuple(float(v) for v in args.bilayer_init_energy_offsets),
    )
    trap_cfg = MultimodelTrapConfig()

    run_summaries: List[Dict[str, Any]] = []
    for policy in args.policies:
        for seed in args.seeds:
            logger.info("Running policy=%s seed=%s", policy, seed)
            if args.scenario == "ghost-optic":
                summary = run_single_policy_ghost_optic(
                    policy=policy,
                    seed=seed,
                    output_dir=output_dir,
                    total_measurements=args.total_measurements,
                    batch_size=args.batch_size,
                    audit_points=args.audit_points,
                    llm_command=args.llm_command,
                    mailbox_url=args.llm_mailbox_url,
                    mailbox_token=args.llm_mailbox_token,
                    mailbox_run_id=args.llm_mailbox_run_id,
                    ghost_config=ghost_cfg,
                )
            elif args.scenario == "bilayer-fm":
                summary = run_single_policy_bilayer_fm(
                    policy=policy,
                    seed=seed,
                    output_dir=output_dir,
                    total_measurements=args.total_measurements,
                    batch_size=args.batch_size,
                    audit_points=args.audit_points,
                    llm_command=args.llm_command,
                    mailbox_url=args.llm_mailbox_url,
                    mailbox_token=args.llm_mailbox_token,
                    mailbox_run_id=args.llm_mailbox_run_id,
                    bilayer_config=bilayer_cfg,
                )
            elif args.scenario == "multimodel-trap":
                summary = run_single_policy_multimodel_trap(
                    policy=policy,
                    seed=seed,
                    output_dir=output_dir,
                    total_measurements=args.total_measurements,
                    batch_size=args.batch_size,
                    audit_points=args.audit_points,
                    llm_command=args.llm_command,
                    mailbox_url=args.llm_mailbox_url,
                    mailbox_token=args.llm_mailbox_token,
                    mailbox_run_id=args.llm_mailbox_run_id,
                    trap_config=trap_cfg,
                )
            else:
                if _TOY_IMPORT_ERROR is not None:
                    raise RuntimeError(
                        "spinwave scenario requires toy_closed_loop dependencies"
                    ) from _TOY_IMPORT_ERROR
                summary = run_single_policy(
                    policy=policy,
                    seed=seed,
                    output_dir=output_dir,
                    total_measurements=args.total_measurements,
                    batch_size=args.batch_size,
                    loggp_grid_points=args.loggp_grid_points,
                    loggp_active_points=args.loggp_active_points,
                    phase3_threshold=args.phase3_threshold,
                    audit_points=args.audit_points,
                    llm_command=args.llm_command,
                    silent_data_config=silent_cfg,
                    hidden_gap_init_config=hidden_gap_cfg,
                    loggp_init_from=args.loggp_init_from,
                    use_bumps=args.use_bumps,
                    mailbox_url=args.llm_mailbox_url,
                    mailbox_token=args.llm_mailbox_token,
                    mailbox_run_id=args.llm_mailbox_run_id,
                    demo_hmin=args.demo_hmin,
                    demo_hmax=args.demo_hmax,
                )
            run_summaries.append(summary)

    aggregate = summarize_runs(run_summaries)
    (output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_dir / "all_runs.json").write_text(json.dumps(run_summaries, indent=2))
    logger.info("Wrote aggregate summary to %s", output_dir / "aggregate_summary.json")


if __name__ == "__main__":
    main()

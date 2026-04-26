#!/usr/bin/env python3
"""Closed-loop toy demo with LLM overseer (mode + points)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from toy_closed_loop import (
    BASE_HAS_RESOLUTION,
    H_RANGE_MIN,
    H_RANGE_MAX,
    LOGGP_E_MAX,
    LOGGP_MOVE_VH,
    LOGGP_MOVE_VE,
    LOGGP_MOVE_OVERHEAD,
    SquareLatticeDispersion,
    create_toy_structure,
    generate_hypotheses,
    run_loggp_phase,
    plan_measurements,
    simulate_measurements,
    discriminate_models,
    save_checkpoint,
    load_checkpoint,
    init_tas,
    TAS,
)

logger = logging.getLogger("tasai_overseer")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _http_get(url: str, token: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    req = urllib.request.Request(url, headers={"X-LLM-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_post(url: str, token: str, payload: Dict[str, Any]) -> bool:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-LLM-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            _ = resp.read().decode("utf-8")
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Discrimination menu + audit guardrail
# ---------------------------------------------------------------------------

def _estimate_measurement_sigma(mu: float, count_time: float = 60.0) -> float:
    mu = float(max(mu, 0.0))
    counts = max(1.0, mu * float(count_time))
    sigma_poisson = np.sqrt(counts) / float(count_time)
    sigma_syst = max(0.03 * mu, 0.02)
    sigma = float(np.sqrt(sigma_poisson**2 + sigma_syst**2))
    return max(sigma, 1e-4)


def _move_time(last_h: float, last_E: float, h: float, E: float) -> float:
    dh = abs(float(h) - float(last_h))
    dE = abs(float(E) - float(last_E))
    return float(max(dh / LOGGP_MOVE_VH, dE / LOGGP_MOVE_VE) + LOGGP_MOVE_OVERHEAD)


def _too_close(h: float, E: float, pts: List[Dict[str, Any]],
               h_thresh: float = 0.06, e_thresh: float = 1.5) -> bool:
    for p in pts:
        if abs(float(p["h"]) - float(h)) < h_thresh and abs(float(p["E"]) - float(E)) < e_thresh:
            return True
    return False


def _top_two_models(results: Dict[str, Any]) -> Tuple[str, str]:
    items = sorted(results.items(), key=lambda kv: -float(kv[1].get("posterior", 0.0)))
    if len(items) < 2:
        raise ValueError("Need at least two models for discrimination menu")
    return items[0][0], items[1][0]


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
        for c in centers:
            for d in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
                E = c + d
                if e_bounds[0] <= E <= e_bounds[1]:
                    cand.append((float(h), float(E)))
        for E in (0.6, 1.0, 1.5, 2.0):
            if e_bounds[0] <= E <= e_bounds[1]:
                cand.append((float(h), float(E)))

    cand = list({(round(h, 4), round(E, 4)) for h, E in cand})
    if len(cand) > max_candidates:
        rng = np.random.default_rng(0)
        cand = list(rng.choice(np.array(cand, dtype=float), size=max_candidates, replace=False))
        cand = [(float(h), float(E)) for h, E in cand]

    last_h = float(measurements[-1]["h"]) if measurements else float(h_bounds[0])
    last_E = float(measurements[-1]["E"]) if measurements else float(e_bounds[0])

    scored: List[Dict[str, Any]] = []
    for h, E in cand:
        if not TAS.is_accessible(h, h, E):
            continue
        if measurements and _too_close(h, E, measurements):
            continue
        sigma_E = TAS.get_energy_resolution(h, h, E)
        mu1 = float(m_best.intensity(h, h, E, sigma_E=sigma_E))
        mu2 = float(m_second.intensity(h, h, E, sigma_E=sigma_E))
        if max(mu1, mu2) < 1.0:
            continue
        s1 = _estimate_measurement_sigma(mu1, count_time=count_time)
        s2 = _estimate_measurement_sigma(mu2, count_time=count_time)
        denom = float(np.sqrt(s1**2 + s2**2))
        z = float(abs(mu1 - mu2) / max(denom, 1e-6))
        t_move = _move_time(last_h, last_E, h, E)
        utility = float(z / (1.0 + t_move))
        scored.append(
            {
                "h": float(h),
                "k": float(h),
                "E": float(E),
                "score": utility,
                "z": z,
                "move_time": t_move,
                "mu_best": mu1,
                "mu_second": mu2,
                "sigma_E": float(sigma_E),
                "best_model": best_name,
                "second_model": second_name,
            }
        )

    scored.sort(key=lambda d: -float(d["score"]))
    menu: List[Dict[str, Any]] = []
    for row in scored:
        if len(menu) >= menu_size:
            break
        if _too_close(row["h"], row["E"], menu):
            continue
        menu.append(row)

    menu_lookup: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(menu):
        pid = f"D{i:02d}"
        row = dict(row)
        row["id"] = pid
        menu_lookup[pid] = row
        menu[i] = row

    return menu, menu_lookup


def audit_needed(
    best_post: float,
    menu: List[Dict[str, Any]],
    measurements: List[Dict[str, Any]],
    conf_threshold: float = 0.90,
    topk: int = 6,
    h_thresh: float = 0.06,
    e_thresh: float = 1.5,
    min_measurements: int = 20,
) -> bool:
    if len(measurements) < min_measurements:
        return False
    if float(best_post) < conf_threshold:
        return False
    if not menu:
        return False
    top = menu[: max(1, int(topk))]
    for m in measurements:
        for pt in top:
            if abs(float(m["h"]) - float(pt["h"])) < h_thresh and abs(float(m["E"]) - float(pt["E"])) < e_thresh:
                return False
    return True


def _build_overseer_prompt(state: Dict[str, Any]) -> str:
    lines = [
        "You are an overseer for an autonomous neutron TAS experiment.",
        "Do not assume the answer; use only the data and general AFM intuition.",
        "Choose a base MODE and INTENT. Do NOT provide raw coordinates.",
        "Return STRICT JSON with fields: {mode, intent, n_points, inject_ids(optional), reason}.",
        "mode in {loggp_active, physics}.",
        "intent in {map, discriminate, refine}.",
        f"n_points must equal {state['batch_size']}.",
        f"inject_ids may include at most {state['max_inject']} IDs from the DISCRIMINATION MENU.",
        "If inject_ids are provided, they are executed FIRST; remaining points are filled by the selected mode.",
        f"Valid H range: {state['h_bounds'][0]} to {state['h_bounds'][1]}.",
        f"Valid E range: {state['e_bounds'][0]} to {state['e_bounds'][1]}.",
        f"Budget remaining: {state['budget_left']} of {state['budget_total']}",
        f"Last mode: {state['last_mode']}",
        f"Last reason: {state['last_reason']}",
        f"Posterior: {state['posterior']}",
        f"Best model: {state['best_model']} ({state['best_post']:.3f})",
        f"Points since last Log-GP batch: {state['since_loggp']}",
        f"Audit recommended: {bool(state.get('audit_recommended'))} (suggested: {state.get('audit_suggested','')})",
        "Guidance: use loggp_active to map when posterior is diffuse; use physics to refine once a model is favored; use discriminate intent when competing models are close.",
        "Guardrails (min-run + forced Log-GP interval) may override your choice.",
        "DISCRIMINATION MENU (pick inject_ids from here if needed):",
    ]
    discrim_menu = state.get("discrim_menu", [])
    if discrim_menu:
        for row in discrim_menu:
            lines.append(
                f"  {row['id']}: h={row['h']:.3f}, E={row['E']:.2f}, "
                f"score={row['score']:.3f} (z={row['z']:.2f}, move={row['move_time']:.1f}s)"
            )
    else:
        lines.append("  (no menu available yet)")
    if state.get("violation"):
        lines.append(f"Violation: {state['violation']}")
    if state.get("recent"):
        lines.append("Recent measurements:")
        for row in state["recent"]:
            lines.append(f"  {row}")
    return "\n".join(lines)


def _parse_decision(payload: Dict[str, Any], expected_n_points: int, max_inject: int) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    if "decision" in payload:
        payload = payload.get("decision")
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    n_points = payload.get("n_points")
    inject_ids = payload.get("inject_ids", [])
    reason = payload.get("reason", "")
    if mode == "llm_points":
        mode = "physics"
    if n_points != expected_n_points:
        return None
    if mode not in {"loggp_active", "physics"}:
        return None
    if not isinstance(inject_ids, list) or not all(isinstance(x, str) for x in inject_ids):
        inject_ids = []
    inject_ids = inject_ids[:max_inject]
    violation = ""
    if payload.get("points") or payload.get("inject_points"):
        violation = "raw coordinates are not allowed; use inject_ids"
    return {
        "mode": mode,
        "intent": payload.get("intent", ""),
        "n_points": n_points,
        "inject_ids": inject_ids,
        "reason": reason,
        "violation": violation,
    }


def _mailbox_overseer_decision(state: Dict[str, Any], mailbox_url: str, token: str, batch_key: str) -> Optional[Dict[str, Any]]:
    base_url = mailbox_url.rstrip("/")
    if base_url.endswith("/tasai_mailbox_overseer"):
        base = base_url
    else:
        base = base_url + "/tasai_mailbox_overseer"
    prompt = _build_overseer_prompt(state)
    payload = {
        "prompt": prompt,
        "checkpoint": {
            "batch": state["batch"],
            "n_measurements": state["n_measurements"],
            "measurements": state.get("recent", []),
        },
        "meta": {
            "batch": state["batch"],
            "timestamp": state["timestamp"],
        },
    }
    status = _http_get(f"{base}/status/{batch_key}", token)
    if not status or not status.get("prompt_ready"):
        _http_post(f"{base}/prompt/{batch_key}", token, payload)
    waited = 0
    while True:
        status = _http_get(f"{base}/status/{batch_key}", token)
        if status and status.get("suggestions_ready"):
            response = _http_get(f"{base}/suggestions/{batch_key}", token)
            if response:
                decision = _parse_decision(response, state["batch_size"], state["max_inject"])
                meta = response.get("meta", {}) if isinstance(response, dict) else {}
                return {"decision": decision, "meta": meta}
        time.sleep(10)
        waited += 10
        if waited % 60 == 0:
            logger.info("Waiting for LLM overseer decision (%s): %ss", batch_key, waited)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--phase-planner", action="store_true")
    parser.add_argument("--hybrid-loggp", action="store_true")
    parser.add_argument("--loggp-init-from", type=str, default=None)
    parser.add_argument("--total-measurements", type=int, default=90)
    parser.add_argument("--phase2-limit", type=int, default=30)
    parser.add_argument("--phase3-threshold", type=float, default=0.95)
    parser.add_argument("--projected-fisher", action="store_true", default=True,
                        help="Enable phase-3 projected-Fisher gap hunting (default on)")
    parser.add_argument("--bumps-interval", type=int, default=10)
    parser.add_argument("--bumps-mp", type=int, default=0)
    parser.add_argument("--bumps-pop", type=int, default=0)
    parser.add_argument("--loggp-grid-points", type=int, default=31)
    parser.add_argument("--loggp-active-points", type=int, default=15)
    parser.add_argument("--demo-hmin", type=float, default=None)
    parser.add_argument("--demo-hmax", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--llm-mailbox-url", type=str, required=True)
    parser.add_argument("--llm-mailbox-token", type=str, required=True)
    parser.add_argument("--llm-mailbox-run-id", type=str, default=None)
    parser.add_argument("--min-run-points", type=int, default=5)
    parser.add_argument("--loggp-forced-interval", type=int, default=10)
    parser.add_argument("--loggp-verify-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--checkpoint-snapshots", action="store_true", default=True,
                        help="Write per-batch checkpoint snapshots for posterior tracing")
    parser.add_argument("--no-checkpoint-snapshots", dest="checkpoint_snapshots", action="store_false",
                        help="Disable per-batch checkpoint snapshots")
    parser.add_argument("--audit-enabled", action="store_true", default=True)
    parser.add_argument("--audit-confidence", type=float, default=0.9)
    parser.add_argument("--audit-topk", type=int, default=10)
    parser.add_argument("--audit-window", type=int, default=20)
    parser.add_argument("--audit-points", type=int, default=2)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("batch-size must be > 0 for overseer")

    init_tas(BASE_HAS_RESOLUTION)
    np.random.seed(args.seed)

    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    true_model = SquareLatticeDispersion(**candidates[-1]["params"]) if candidates else SquareLatticeDispersion(J1=1.25, J2=0.2, D=0.02)

    checkpoint_dir = Path(args.checkpoint_dir)
    demo_hmin = H_RANGE_MIN if args.demo_hmin is None else args.demo_hmin
    demo_hmax = H_RANGE_MAX if args.demo_hmax is None else args.demo_hmax
    measurements: List[Dict[str, Any]] = []
    measurement_plan: List[Dict[str, Any]] = []
    planned_points: List[Dict[str, Any]] = []
    llm_state: Dict[str, Any] = {}
    llm_batch_idx = 0

    if args.resume_from:
        payload = load_checkpoint(Path(args.resume_from))
        measurements = payload["measurements"]
        measurement_plan = payload["measurement_plan"]
        planned_points = payload["planned_points"]
        llm_state = payload.get("llm_state", {}) or {}
        llm_batch_idx = int(payload.get("llm_batch_idx", 0) or 0)
        logger.info("Resumed overseer from checkpoint: %s", args.resume_from)
    elif args.hybrid_loggp:
        seed_measurements: List[Dict[str, Any]] = []
        if args.loggp_init_from:
            payload = json.loads(Path(args.loggp_init_from).read_text())
            seed_measurements = payload.get("measurements", [])
        else:
            grid_meas, _, _ = run_loggp_phase(
                true_model,
                n_measurements=args.loggp_grid_points,
                hmin=demo_hmin,
                hmax=demo_hmax,
                emin=0.5,
                emax=LOGGP_E_MAX,
            )
            for m in grid_meas:
                m["mode"] = "loggp_grid"
                # Use -1 for pre-overseer init grid; overseer-controlled batches start at 1.
                m["llm_batch_idx"] = -1
                m["llm_provider"] = "init"
                m["llm_decision_reason"] = "loggp grid"
            seed_measurements = grid_meas
            save_checkpoint(
                checkpoint_dir,
                seed_measurements,
                measurement_plan,
                planned_points,
                llm_state={},
                llm_batch_idx=-1,
                note="after_loggp_grid",
            )

        total_loggp = len(seed_measurements) + args.loggp_active_points
        loggp_meas, _, _ = run_loggp_phase(
            true_model,
            n_measurements=total_loggp,
            hmin=demo_hmin,
            hmax=demo_hmax,
            emin=0.5,
            emax=LOGGP_E_MAX,
            seed_measurements=seed_measurements,
        )
        active_added = max(0, len(loggp_meas) - len(seed_measurements))
        if active_added == 0 and args.loggp_active_points > 0:
            logger.warning("Log-GP active produced zero points; will proceed but LLM may not be prompted.")
        for idx, m in enumerate(loggp_meas):
            if idx < len(seed_measurements):
                m.setdefault("mode", "loggp_grid")
                m.setdefault("llm_batch_idx", -1)
                m.setdefault("llm_provider", "init")
                m.setdefault("llm_decision_reason", "loggp grid")
            else:
                m["mode"] = "loggp_active"
                m["llm_batch_idx"] = 0
                m["llm_provider"] = "init"
                m["llm_decision_reason"] = "loggp active"
        measurements = loggp_meas
        save_checkpoint(
            checkpoint_dir,
            measurements,
            measurement_plan,
            planned_points,
            llm_state={},
            llm_batch_idx=0,
            note="after_loggp_active",
        )
        measurement_plan = loggp_meas.copy()
        planned_points = loggp_meas.copy()
    else:
        raise SystemExit("Overseer requires either --resume-from or --hybrid-loggp and --loggp-init-from")

    if not llm_state:
        llm_state = {
            "last_mode": "loggp_active",
            "last_reason": "initial loggp phase",
            "mode_run_len": args.min_run_points,
            "since_loggp": 0,
            "violation": "",
            "last_decider": "",
            "last_overseer_meta": {},
        }

    def run_discrimination() -> Tuple[Dict[str, Any], List[float], str, float]:
        use_bumps_now = (len(measurements) % max(1, args.bumps_interval) == 0)
        bumps_mp = args.bumps_mp if args.bumps_mp > 0 else None
        bumps_pop = args.bumps_pop if args.bumps_pop > 0 else None
        results = discriminate_models(
            measurements,
            candidates,
            use_bumps=use_bumps_now,
            bumps_mp=bumps_mp,
            bumps_pop=bumps_pop,
        )
        post = [results[c["name"]]["posterior"] for c in candidates]
        best_idx = int(np.argmax(post))
        return results, post, candidates[best_idx]["name"], post[best_idx]

    while len(measurements) < args.total_measurements:
        results, posteriors, best_model, best_post = run_discrimination()
        remaining = args.total_measurements - len(measurements)
        batch_cap = min(args.batch_size, remaining)
        max_inject = min(args.audit_points, batch_cap, 2)
        discrim_menu, menu_lookup = build_discrimination_menu(
            results,
            measurements,
            h_bounds=(0.5, 1.7),
            e_bounds=(0.5, LOGGP_E_MAX),
            menu_size=max(12, args.audit_topk),
        )
        audit_recommended = False
        audit_ids: List[str] = []
        if args.audit_enabled:
            audit_recommended = audit_needed(
                best_post,
                discrim_menu,
                measurements,
                conf_threshold=args.audit_confidence,
                topk=args.audit_topk,
                min_measurements=max(20, args.loggp_grid_points),
            )
            if audit_recommended:
                audit_ids = [row["id"] for row in discrim_menu[: max(1, max_inject)]]

        # Decide mode
        forced_mode = None
        if llm_state["mode_run_len"] < args.min_run_points:
            forced_mode = llm_state["last_mode"]
        elif best_post < args.phase3_threshold and llm_state["since_loggp"] >= args.loggp_forced_interval:
            forced_mode = "loggp_active"
        elif best_post >= 0.9 and llm_state["since_loggp"] >= args.loggp_verify_interval:
            forced_mode = "loggp_active"

        decision = None
        reason = ""
        provider = ""
        overseer_meta: Dict[str, Any] = {}
        if forced_mode:
            decision = {"mode": forced_mode, "n_points": batch_cap, "points": [], "reason": "guardrail"}
            provider = "guardrail"
        else:
            batch_key = f"{args.llm_mailbox_run_id}_{llm_batch_idx:03d}" if args.llm_mailbox_run_id else f"{llm_batch_idx:03d}"
            state = {
                "batch": f"{llm_batch_idx:03d}",
                "timestamp": datetime.utcnow().isoformat(),
                "n_measurements": len(measurements),
                "budget_total": args.total_measurements,
                "budget_left": args.total_measurements - len(measurements),
                "last_mode": llm_state["last_mode"],
                "last_reason": llm_state["last_reason"],
                "posterior": {c["name"]: p for c, p in zip(candidates, posteriors)},
                "best_model": best_model,
                "best_post": best_post,
                "since_loggp": llm_state["since_loggp"],
                "h_bounds": (0.5, 1.7),
                "e_bounds": (0.5, LOGGP_E_MAX),
                "batch_size": args.batch_size,
                "recent": [[m["h"], m["E"], m["intensity"], m.get("mode", "")] for m in measurements[-20:]],
                "violation": llm_state.get("violation", ""),
                "discrim_menu": discrim_menu,
                "audit_recommended": audit_recommended,
                "audit_suggested": ",".join(audit_ids),
                "max_inject": max_inject,
            }
            response = _mailbox_overseer_decision(state, args.llm_mailbox_url, args.llm_mailbox_token, batch_key)
            if response:
                decision = response.get("decision")
                overseer_meta = response.get("meta", {})
                provider = overseer_meta.get("decider", "")
            llm_batch_idx += 1

        if decision is None:
            decision = {"mode": "loggp_active", "n_points": batch_cap, "points": [], "reason": "fallback"}
            provider = "fallback"

        mode = decision["mode"]
        intent = decision.get("intent", "")
        reason = decision.get("reason", "")
        inject_ids: List[str] = decision.get("inject_ids", [])
        if decision.get("violation"):
            llm_state["violation"] = decision.get("violation", "")
            inject_ids = []
            mode = "loggp_active"
            provider = "guardrail"
            reason = "invalid llm payload"
        else:
            llm_state["violation"] = ""

        def _propagate_meta(batch_points: List[Dict[str, Any]], new_measurements: List[Dict[str, Any]]) -> None:
            if not batch_points:
                return
            keys = (
                "mode",
                "llm_batch_idx",
                "llm_provider",
                "llm_decision_reason",
                "llm_hint",
                "llm_inject_id",
                "llm_inject_score",
                "llm_inject_z",
                "llm_intent",
                "human_hint",
                "symmetry",
                "coverage",
                "loggp_hint",
                "dwell",
                "count_time",
            )
            for m, pt in zip(new_measurements, batch_points):
                for key in keys:
                    if key in pt:
                        m[key] = pt[key]
                if "llm_decision_reason" not in m and pt.get("reason"):
                    m["llm_decision_reason"] = str(pt.get("reason"))[:200]

        # Allow inject_ids for loggp_active (cap to 2) and execute first.

        if mode == "loggp_active":
            llm_state["since_loggp"] = 0
            inject_pts: List[Dict[str, Any]] = []
            if inject_ids:
                for pid in inject_ids[:max_inject]:
                    row = menu_lookup.get(pid)
                    if not row:
                        continue
                    if _too_close(row["h"], row["E"], measurements):
                        continue
                    inject_pts.append(
                        {
                            "h": float(row["h"]),
                            "k": float(row["h"]),
                            "E": float(row["E"]),
                            "llm_hint": True,
                            "llm_inject_id": pid,
                            "llm_inject_score": float(row["score"]),
                            "llm_inject_z": float(row["z"]),
                            "mode": "loggp_inject",
                            "llm_batch_idx": llm_batch_idx,
                            "llm_provider": provider,
                            "llm_decision_reason": reason,
                            "llm_intent": intent,
                        }
                    )
            if inject_pts:
                new_meas = simulate_measurements(inject_pts, true_model)
                _propagate_meta(inject_pts, new_meas)
                measurements.extend(new_meas)
                measurement_plan.extend(inject_pts)
                planned_points.extend(inject_pts)

            remaining = max(0, batch_cap - len(inject_pts))
            if remaining > 0:
                loggp_meas, _, _ = run_loggp_phase(
                    true_model,
                    n_measurements=len(measurements) + remaining,
                    hmin=demo_hmin,
                    hmax=demo_hmax,
                    emin=0.5,
                    emax=LOGGP_E_MAX,
                    seed_measurements=measurements,
                )
                new_meas = loggp_meas[len(measurements):]
                for m in new_meas:
                    m["mode"] = "loggp_active"
                    m["llm_batch_idx"] = llm_batch_idx
                    m["llm_provider"] = provider
                    m["llm_decision_reason"] = reason
                    m["llm_intent"] = intent
                measurements = loggp_meas
                measurement_plan = loggp_meas.copy()
                planned_points = loggp_meas.copy()
        elif mode == "physics":
            llm_state["since_loggp"] += batch_cap
            best_model_obj = next((c for c in candidates if c["name"] == best_model), None)
            precision_model = None
            if best_model_obj is not None:
                precision_model = SquareLatticeDispersion(**best_model_obj["params"])
            inject_pts: List[Dict[str, Any]] = []
            if inject_ids:
                for pid in inject_ids[:max_inject]:
                    row = menu_lookup.get(pid)
                    if not row:
                        continue
                    if _too_close(row["h"], row["E"], measurements):
                        continue
                    inject_pts.append(
                        {
                            "h": float(row["h"]),
                            "k": float(row["h"]),
                            "E": float(row["E"]),
                            "llm_hint": True,
                            "llm_inject_id": pid,
                            "llm_inject_score": float(row["score"]),
                            "llm_inject_z": float(row["z"]),
                        }
                    )
            if audit_recommended and not inject_pts and audit_ids:
                for pid in audit_ids[:max_inject]:
                    row = menu_lookup.get(pid)
                    if not row:
                        continue
                    if _too_close(row["h"], row["E"], measurements):
                        continue
                    inject_pts.append(
                        {
                            "h": float(row["h"]),
                            "k": float(row["h"]),
                            "E": float(row["E"]),
                            "llm_hint": True,
                            "llm_inject_id": pid,
                            "llm_inject_score": float(row["score"]),
                            "llm_inject_z": float(row["z"]),
                            "llm_decision_reason": "audit_inject",
                        }
                    )
            inject_pts = inject_pts[:batch_cap]
            remaining = max(0, batch_cap - len(inject_pts))
            batch = plan_measurements(
                candidates,
                n_points=remaining,
                existing_points=measurements + inject_pts,
                measurement_history=measurements,
                enable_phases=True,
                phase_thresholds=(5, args.phase2_limit),
                posterior_hint=posteriors,
                posterior_phase3_threshold=args.phase3_threshold,
                precision_model=precision_model,
                use_projected_fisher=args.projected_fisher,
                hmin=demo_hmin,
                hmax=demo_hmax,
            )
            batch = inject_pts + batch
            if len(batch) > batch_cap:
                batch = batch[:batch_cap]
            for pt in batch:
                pt["mode"] = "physics"
                pt["llm_batch_idx"] = llm_batch_idx
                pt["llm_provider"] = provider
                pt["llm_decision_reason"] = reason
                pt["llm_intent"] = intent
            new_meas = simulate_measurements(batch, true_model)
            _propagate_meta(batch, new_meas)
            measurements.extend(new_meas)
            measurement_plan.extend(batch)
            planned_points.extend(batch)
        else:
            llm_state["since_loggp"] += batch_cap

        if mode == llm_state["last_mode"]:
            llm_state["mode_run_len"] += args.batch_size
        else:
            llm_state["mode_run_len"] = args.batch_size
        llm_state["last_mode"] = mode
        llm_state["last_reason"] = reason
        llm_state["last_decider"] = provider
        llm_state["last_overseer_meta"] = overseer_meta

        save_checkpoint(
            checkpoint_dir,
            measurements,
            measurement_plan,
            planned_points,
            llm_state,
            llm_batch_idx,
            note=f"after_{mode}_{len(measurements)}"
        )
        if args.checkpoint_snapshots:
            snapshot = checkpoint_dir / f"checkpoint_batch_{llm_batch_idx:03d}.json"
            try:
                snapshot.write_text(
                    json.dumps(
                        {
                            "note": f"after_{mode}_{len(measurements)}",
                            "measurements": measurements,
                            "measurement_plan": measurement_plan,
                            "planned_points": planned_points,
                            "llm_state": llm_state,
                            "llm_batch_idx": llm_batch_idx,
                        },
                        indent=2,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to write checkpoint snapshot: %s", exc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a compact final summary from a closed-loop checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulations.toy_closed_loop import create_toy_structure, generate_hypotheses, discriminate_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--use-bumps", action="store_true")
    return p.parse_args()


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _mode_counts(measurements: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter((m.get("mode") or "unknown") for m in measurements))


def _llm_batch_counts(measurements: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for m in measurements:
        idx = m.get("llm_batch_idx")
        key = "init" if idx is None else str(idx)
        counts[key] += 1
    return dict(counts)


def _model_rows(results: Dict[str, Dict[str, Any]], n_measurements: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, row in results.items():
        fitted_model = row.get("fitted_model")
        params = {
            "J1": getattr(fitted_model, "J1", None),
            "J2": getattr(fitted_model, "J2", None),
            "D": getattr(fitted_model, "D", None),
            "background": getattr(fitted_model, "background", None),
        }
        errs = row.get("uncertainties", {}) or {}
        chi2 = _as_float(row.get("chi2"))
        rows.append(
            {
                "model": name,
                "posterior": _as_float(row.get("posterior")),
                "aic": _as_float(row.get("aic")),
                "chi2": (chi2 / float(n_measurements)) if chi2 is not None and n_measurements > 0 else None,
                "J1": _as_float(params.get("J1")),
                "J2": _as_float(params.get("J2")),
                "D": _as_float(params.get("D")),
                "background": _as_float(params.get("background")),
                "J1_err": _as_float(errs.get("J1")),
                "J2_err": _as_float(errs.get("J2")),
                "D_err": _as_float(errs.get("D")),
                "background_err": _as_float(errs.get("background")),
            }
        )
    rows.sort(key=lambda r: (-(r["posterior"] or 0.0), r["model"]))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "posterior",
        "aic",
        "chi2",
        "J1",
        "J2",
        "D",
        "background",
        "J1_err",
        "J2_err",
        "D_err",
        "background_err",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    payload = json.loads(checkpoint.read_text())
    measurements = payload.get("measurements", [])
    if not measurements:
        raise SystemExit("checkpoint contains no measurements")

    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    results = discriminate_models(measurements, candidates, use_bumps=args.use_bumps)
    rows = _model_rows(results, len(measurements))
    best = max(rows, key=lambda r: (r["posterior"] or 0.0, -(r["aic"] or 1e99)))

    summary = {
        "checkpoint": str(checkpoint),
        "note": payload.get("note"),
        "n_measurements": len(measurements),
        "mode_counts": _mode_counts(measurements),
        "llm_batch_counts": _llm_batch_counts(measurements),
        "best_model": best["model"],
        "best_model_posterior": best["posterior"],
        "model_table": rows,
        "use_bumps": bool(args.use_bumps),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    _write_csv(Path(args.out_csv), rows)
    print(json.dumps({"ok": True, "best_model": best["model"], "n_measurements": len(measurements)}))


if __name__ == "__main__":
    main()

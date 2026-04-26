#!/usr/bin/env python3
"""Compute posterior evolution vs measurement count from a checkpoint.

Usage:
  python paper/scripts/compute_posterior_trace.py \
    --checkpoint /path/to/closed_loop_checkpoint.json \
    --out-csv /path/to/posterior_trace.csv \
    --out-plot /path/to/posterior_trace.png \
    --step 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulations.toy_closed_loop import create_toy_structure, generate_hypotheses, discriminate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-plot", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--n-values", type=str, default="",
                        help="Comma-separated measurement counts to evaluate (overrides --step).")
    parser.add_argument("--use-bumps", action="store_true")
    parser.add_argument("--bumps-mp", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=0,
                        help="Write intermediate CSV/plot every N steps (0=only at end).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.checkpoint).read_text())
    measurements = payload.get("measurements", [])
    if not measurements:
        raise SystemExit("No measurements found in checkpoint")

    structure = create_toy_structure()
    candidates = generate_hypotheses(structure)
    names = [c["name"] for c in candidates]

    rows = []
    n_total = len(measurements)
    if args.n_values:
        raw = [s.strip() for s in args.n_values.split(",") if s.strip()]
        steps = []
        seen = set()
        for token in raw:
            try:
                n = int(token)
            except ValueError as exc:
                raise SystemExit(f"Invalid --n-values entry: {token}") from exc
            if n < 1:
                continue
            if n > n_total:
                continue
            if n in seen:
                continue
            steps.append(n)
            seen.add(n)
        if not steps:
            raise SystemExit("No valid --n-values entries within measurement count.")
    else:
        steps = list(range(args.step, n_total + 1, args.step))
        if steps[-1] != n_total:
            steps.append(n_total)

    for n in steps:
        subset = measurements[:n]
        results = discriminate_models(
            subset,
            candidates,
            use_bumps=args.use_bumps,
            bumps_mp=args.bumps_mp,
            bumps_pop=args.bumps_mp,
        )
        post = {name: results[name]["posterior"] for name in names}
        best = max(post.items(), key=lambda x: x[1])[0]
        for name in names:
            rows.append({
                "n": n,
                "model": name,
                "posterior": post[name],
                "best_model": best,
            })

        if args.flush_every and (n % args.flush_every == 0 or n == steps[-1]):
            _write_outputs(rows, names, args.out_csv, args.out_plot)

    _write_outputs(rows, names, args.out_csv, args.out_plot)


def _write_outputs(rows, names, out_csv, out_plot):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "n,model,posterior,best_model\n"
    with out_csv.open("w") as f:
        f.write(header)
        for r in rows:
            f.write(f"{r['n']},{r['model']},{r['posterior']},{r['best_model']}\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        xs = [r["n"] for r in rows if r["model"] == name]
        ys = [r["posterior"] for r in rows if r["model"] == name]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=name)
    ax.set_xlabel("Measurements")
    ax.set_ylabel("Posterior")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Posterior evolution")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200)


if __name__ == "__main__":
    main()

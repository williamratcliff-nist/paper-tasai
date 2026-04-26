# Data Layout

This directory contains curated archived data supporting the paper.

## Contents

- [final/](final/)
  - benchmark summaries, pilot summaries, and table-supporting artifacts used by the main paper
- [ablation_runs/](ablation_runs/)
  - Section 5 audit-ablation outputs, including:
    - the original one-seed ghost-optic archive
    - the original cleaned one-seed bilayer archive
    - the multi-model trap archive
    - the five-seed ghost-optic rerun
    - the five-seed bilayer rerun
- top-level JSON files in this directory
  - Laplace-coverage and reviewer-sensitivity artifacts referenced in the SI

## Notes

- The five-seed ablation reruns are archived under:
  - [ablation_runs/ghost_optic_5seed_20260415c](ablation_runs/ghost_optic_5seed_20260415c)
  - [ablation_runs/bilayer_fm_5seed_20260415c](ablation_runs/bilayer_fm_5seed_20260415c)
- Each of those directories contains:
  - `aggregate_summary.json`
  - `all_runs.json`
  - per-seed `summary.json` files
- Historical machine-specific run roots in archived JSON files have been
  replaced with placeholders where needed for portability.
- Operational mailbox logs and local watcher state are intentionally excluded from this reviewer repo.

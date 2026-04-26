# Reproducibility

This paper-facing bundle is designed to let a reviewer inspect:

- the manuscript source
- the exact figures referenced by the current manuscript snapshot
- the archived JSON/CSV artifacts behind the quantitative claims
- the paper-facing scripts used to generate figures and summaries

## Canonical Entry Points

- Main manuscript:
  - [paper/digital_discovery_paper.md](paper/digital_discovery_paper.md)
- Supplementary information:
  - [paper/TAS-AI_Digital_Discovery_SI.md](paper/TAS-AI_Digital_Discovery_SI.md)
- Figure assets:
  - [paper/figures/](paper/figures/)
- Paper-facing scripts:
  - [paper/scripts/](paper/scripts/)
- Archived data:
  - [paper/data/](paper/data/)

## Build the Manuscript

The manuscript subtree is self-contained. From the repo root:

```bash
cd paper
bash scripts/build_manuscript.sh
```

This uses:

- [paper/Dockerfile.pandoc](paper/Dockerfile.pandoc)
- [paper/scripts/build_manuscript.sh](paper/scripts/build_manuscript.sh)

## Main Archived Data

The main paper tables and figure summaries are backed by:

- [paper/data/final/benchmark_summary_fair_analytic_20260218.json](paper/data/final/benchmark_summary_fair_analytic_20260218.json)
- [paper/data/final/benchmark_summary_fair_pyspinw_20260402c.json](paper/data/final/benchmark_summary_fair_pyspinw_20260402c.json)
- [paper/data/final/benchmark_tasai_analytic_sunny_20260318.json](paper/data/final/benchmark_tasai_analytic_sunny_20260318.json)
- [paper/data/final/benchmark_tasai_pyspinw_20260402d.json](paper/data/final/benchmark_tasai_pyspinw_20260402d.json)
- [paper/data/final/time_aware_refinement_full_20260327.json](paper/data/final/time_aware_refinement_full_20260327.json)
- [paper/data/final/time_aware_search_results_20260327.json](paper/data/final/time_aware_search_results_20260327.json)
- [paper/data/final/overseer_loggpfix_20260327_final_summary.json](paper/data/final/overseer_loggpfix_20260327_final_summary.json)

## Section 5 Ablation Archives

The Section 5 discussion is backed by:

- original one-seed ghost-optic archive:
  - [paper/data/ablation_runs/ghost_optic](paper/data/ablation_runs/ghost_optic)
- original one-seed cleaned bilayer archive:
  - [paper/data/ablation_runs/bilayer_fm_cleaned](paper/data/ablation_runs/bilayer_fm_cleaned)
- multi-model trap archive:
  - [paper/data/ablation_runs/multimodel_trap](paper/data/ablation_runs/multimodel_trap)
- five-seed ghost-optic rerun:
  - [paper/data/ablation_runs/ghost_optic_5seed_20260415c](paper/data/ablation_runs/ghost_optic_5seed_20260415c)
- five-seed bilayer rerun:
  - [paper/data/ablation_runs/bilayer_fm_5seed_20260415c](paper/data/ablation_runs/bilayer_fm_5seed_20260415c)

## Coverage and Sensitivity Checks

The SI robustness/calibration additions are backed by:

- [paper/data/laplace_coverage_refinement_20260415.json](paper/data/laplace_coverage_refinement_20260415.json)
- [paper/data/reviewer_sensitivity_20260403.json](paper/data/reviewer_sensitivity_20260403.json)

## Closed-Loop Drivers and Physics Backends

The manuscript-specific closed-loop drivers live under this repo at
[paper/scripts/](paper/scripts/):

- `toy_closed_loop.py` — end-to-end closed-loop pilot used for Figure 9
  (fixed 13+15+N handoff) and Figure 10 (90-measurement overseer run,
  four-model candidate set with the `[0.10, 0.10, 0.10, 0.70]` prior).
- `run_audit_ablation.py` — harness for the §5.3 ablations (ghost-optic,
  bilayer ferromagnet, multi-model trap).
- `toy_closed_loop_llm_overseer.py` — overseer-mode wrapper.
- `exchange_path_analysis.py` — Goodenough–Kanamori exchange-path
  enumeration used for Figure 13.
- `create_workflow_figure.py` — Figure 1 workflow diagram.

The analytic spin-wave physics used by these drivers is upstreamed into
the library so the closed-loop pilots can be rerun without copying
physics code out of this repo:

- `tasai.physics.SquareLatticeAFM` — Néel-phase J₁-J₂-D AFM on the
  square lattice (§3.6 / Fig 10).
- `tasai.physics.SquareFMBilayer` — square-lattice bilayer ferromagnet
  with acoustic + optic branches and L-dependent weights (§5.3.2).

## Citation Audit

A read-only citation checker that diffs `paper/references.bib` against
Crossref (author lists, year, DOIs) and scans the two markdown sources
for unresolved or unused `[@key]` citations is available at
[paper/scripts/check_citations.py](paper/scripts/check_citations.py).
It uses only the Python standard library and returns a non-zero exit
code when warnings are present.

## Relationship to the Code Repo

This repo does not duplicate the full TAS-AI library source. The public code repo is:

- [williamratcliff-nist/tasai](https://github.com/williamratcliff-nist/tasai)

The public paper-bundle mirror is:

- [williamratcliff-nist/paper-tasai](https://github.com/williamratcliff-nist/paper-tasai)

The manuscript text points to those two public repositories directly:

- `williamratcliff-nist/tasai` for the reusable library code
- `williamratcliff-nist/paper-tasai` for manuscript sources, paper-facing scripts, archived data, and provenance artifacts

# Reviewer Guide

This repository is meant to be a clean paper-facing bundle.

If you only want the shortest path through the evidence:

1. Read the current manuscript:
   - [paper/TAS-AI_Digital_Discovery.docx](paper/TAS-AI_Digital_Discovery.docx)
   - [paper/TAS-AI_Digital_Discovery_SI.docx](paper/TAS-AI_Digital_Discovery_SI.docx)
2. Check the figure assets used by the manuscript:
   - [paper/figures/](paper/figures/)
3. Check the archived data behind the claims:
   - [paper/data/final/](paper/data/final/)
   - [paper/data/ablation_runs/](paper/data/ablation_runs/)
4. If needed, inspect the paper-facing scripts:
   - [paper/scripts/](paper/scripts/)
5. If needed, inspect the markdown sources of record:
   - [paper/digital_discovery_paper.md](paper/digital_discovery_paper.md)
   - [paper/TAS-AI_Digital_Discovery_SI.md](paper/TAS-AI_Digital_Discovery_SI.md)

## What Is In Scope Here

This repo includes:

- manuscript markdown
- rendered manuscript outputs
- exact figure files referenced by the manuscript
- archived benchmark/pilot/ablation JSON and CSV files
- paper-facing scripts used to generate figures and summaries

This repo intentionally excludes:

- operational mailbox logs
- prompt experiments and local watcher state
- slides and cover letters
- old manuscript versions
- the full TAS-AI library source tree

## Key Provenance Pointers

### Main paper benchmark and pilot artifacts

- [paper/data/final/benchmark_summary_fair_analytic_20260218.json](paper/data/final/benchmark_summary_fair_analytic_20260218.json)
- [paper/data/final/benchmark_summary_fair_pyspinw_20260402c.json](paper/data/final/benchmark_summary_fair_pyspinw_20260402c.json)
- [paper/data/final/benchmark_tasai_analytic_sunny_20260318.json](paper/data/final/benchmark_tasai_analytic_sunny_20260318.json)
- [paper/data/final/benchmark_tasai_pyspinw_20260402d.json](paper/data/final/benchmark_tasai_pyspinw_20260402d.json)
- [paper/data/final/time_aware_refinement_full_20260327.json](paper/data/final/time_aware_refinement_full_20260327.json)
- [paper/data/final/time_aware_search_results_20260327.json](paper/data/final/time_aware_search_results_20260327.json)
- [paper/data/final/overseer_loggpfix_20260327_final_summary.json](paper/data/final/overseer_loggpfix_20260327_final_summary.json)

### Section 5 ablations

- one-seed ghost-optic:
  - [paper/data/ablation_runs/ghost_optic](paper/data/ablation_runs/ghost_optic)
- one-seed cleaned bilayer:
  - [paper/data/ablation_runs/bilayer_fm_cleaned](paper/data/ablation_runs/bilayer_fm_cleaned)
- multi-model trap:
  - [paper/data/ablation_runs/multimodel_trap](paper/data/ablation_runs/multimodel_trap)
- five-seed ghost-optic rerun:
  - [paper/data/ablation_runs/ghost_optic_5seed_20260415c](paper/data/ablation_runs/ghost_optic_5seed_20260415c)
- five-seed bilayer rerun:
  - [paper/data/ablation_runs/bilayer_fm_5seed_20260415c](paper/data/ablation_runs/bilayer_fm_5seed_20260415c)

### SI robustness additions

- [paper/data/laplace_coverage_refinement_20260415.json](paper/data/laplace_coverage_refinement_20260415.json)
- [paper/data/reviewer_sensitivity_20260403.json](paper/data/reviewer_sensitivity_20260403.json)

### Closed-loop drivers used to generate Figures 9, 10, and Section 5 tables

- [paper/scripts/toy_closed_loop.py](paper/scripts/toy_closed_loop.py) — Figures 9 and 10 (four-model closed-loop pilot with the `[0.10, 0.10, 0.10, 0.70]` chemically-informed prior).
- [paper/scripts/run_audit_ablation.py](paper/scripts/run_audit_ablation.py) — Section 5 ghost-optic, bilayer, and multi-model trap ablations.
- [paper/scripts/toy_closed_loop_llm_overseer.py](paper/scripts/toy_closed_loop_llm_overseer.py) — overseer wrapper for the LLM-audited pilot.
- [paper/scripts/exchange_path_analysis.py](paper/scripts/exchange_path_analysis.py) — Figure 13 exchange-path enumeration.
- [paper/scripts/create_workflow_figure.py](paper/scripts/create_workflow_figure.py) — Figure 1 workflow diagram.

The analytic spin-wave physics that these drivers call is upstreamed into the library as `tasai.physics.SquareLatticeAFM` (Néel-phase J₁-J₂-D AFM, §3.6 / Fig 10) and `tasai.physics.SquareFMBilayer` (square-lattice bilayer ferromagnet with optic branch, §5.3.2). Both modules are covered by unit tests at `tasai/tests/test_paper_backends.py` in the code repo.

### Citation audit tool

- [paper/scripts/check_citations.py](paper/scripts/check_citations.py) — stdlib-only read-only checker that (1) diffs each `references.bib` entry against Crossref (title, year, DOI, position-by-position author list) and (2) scans the two manuscript markdown files for unresolved or unused `[@key]` citations. Returns exit code 1 on warnings so it can be used as a pre-submit gate.

## Code Repository

The underlying reusable library code lives in:

- [usnistgov/tasai](https://github.com/usnistgov/tasai)

The public paper bundle is:

- [usnistgov/paper-tasai](https://github.com/usnistgov/paper-tasai)

# paper-tasai

Reviewer-facing companion repository for the TAS-AI paper.

Unless otherwise noted for third-party materials, the contents of this
repository are released under the NIST data/work statement in
[LICENSE](LICENSE). In the United States, works of NIST employees are not
subject to copyright protection under 17 U.S.C. 105.

This repo is intentionally narrower than the full working paper repository. It contains:

- the current manuscript and supplementary-information sources
- rendered manuscript outputs, with `docx` as the preferred reviewer-facing format
- the exact figure assets referenced by the current manuscript snapshot
- the paper-facing scripts used to generate figures and summaries
- curated archived data supporting the quantitative claims in the paper

It does not include local operational mailbox logs, prompt experiments, old manuscript versions, slides, or the full TAS-AI library source tree.

## Repository Layout

- [paper/](paper/)
  - canonical manuscript source and outputs
  - [paper/figures/](paper/figures/)
    - figures referenced by the current manuscript and SI
  - [paper/scripts/](paper/scripts/)
    - paper-facing figure/table generation scripts
  - [paper/data/](paper/data/)
    - curated archived benchmark, pilot, ablation, and sensitivity artifacts
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
  - current reproducibility guide copied from the working paper repo
- [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)
  - reviewer-facing provenance guide copied from the working paper repo

## Current Manuscript Snapshot

For reading, start with:

- [paper/TAS-AI_Digital_Discovery.docx](paper/TAS-AI_Digital_Discovery.docx)
- [paper/TAS-AI_Digital_Discovery_SI.docx](paper/TAS-AI_Digital_Discovery_SI.docx)

Source-of-record files in this repo are:

- [paper/digital_discovery_paper.md](paper/digital_discovery_paper.md)
- [paper/TAS-AI_Digital_Discovery_SI.md](paper/TAS-AI_Digital_Discovery_SI.md)

These correspond to the current v5 manuscript snapshot from the working paper repo, but are renamed here to canonical names so reviewers do not have to navigate version clutter.

## Data Scope

The [paper/data/](paper/data/) tree includes:

- `final/`
  - benchmark and pilot summaries used for the main paper tables and figures
- `ablation_runs/`
  - archived Section 5 ablation outputs, including the five-seed ghost-optic and bilayer reruns
- top-level coverage and sensitivity JSON files
  - Laplace-coverage and reviewer-sensitivity artifacts used in the SI

See [paper/data/README.md](paper/data/README.md) for more detail.

## Figure Generation

The [paper/scripts/](paper/scripts/) directory contains the paper-facing scripts used to generate the current figures and supporting summaries. This includes the scripts for:

- the Figure 4 benchmark panels
- the Figure 1 workflow diagram
- the closed-loop pilot and overseer drivers
- the Section 5 audit-ablation harness
- the exchange-path analysis figure
- the PySpinW benchmark figure
- the Figure 7 hybrid-routing figure
- the ghost-optic schematic
- the bilayer ablation figure
- the pilot LLM multipanel figure
- the taper comparison and sensitivity summaries

## Manuscript Build

The manuscript can be rebuilt with Pandoc using:

- [paper/Dockerfile.pandoc](paper/Dockerfile.pandoc)
- [paper/scripts/build_manuscript.sh](paper/scripts/build_manuscript.sh)

## Relationship to the Code Repo

This repo is paper-facing. The reusable TAS-AI library code lives separately in:

- [williamratcliff-nist/tasai](https://github.com/williamratcliff-nist/tasai)

Reviewers interested in the underlying library implementation should consult that repo, while reviewers focused on manuscript provenance should be able to work entirely within this one.

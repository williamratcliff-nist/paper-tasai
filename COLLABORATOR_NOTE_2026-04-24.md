# Collaborator Note (2026-04-24)

This note records which comments from the latest read-through were taken into
the manuscript, which were intentionally not taken, and why.

## Changes made in the paper

- **AIC vs BIC wording.**
  The discussion in §3.3 now explains the real concern more precisely: the
  issue is not that the accumulated `\chi^2` term stops mattering, but that in
  a sequential controller the relative BIC penalty drifts with `n` after each
  small batch, even when the new points mostly reinforce already identified
  structure.

- **Earlier model-library clarity.**
  The paper now defines the candidate Hamiltonian family earlier and states
  explicitly in the Discussion that the workflow assumes a candidate-model
  library even when the spectral support is initially unknown.

- **Figure 6 counting clarification.**
  The caption now states that the x-axis is cumulative measurements, including
  the seeded initialization points, so the “fewer than 10 measurements” claim is
  not misread as “fewer than 10 post-seed iterations.”

- **Figure 7 caption clarification.**
  The caption now states that the colors in panel (a) encode traversal order,
  and frames the figure explicitly as a scheduling diagnostic rather than a
  second adaptive-discovery benchmark.

- **LLM control-surface clarification.**
  Section 5.1 now states more explicitly that the LLM does not emit arbitrary
  continuous instrument coordinates. It can only choose between bounded routing
  options and nominate a small number of tactical audit probes from a candidate
  menu that has already been constructed and kinematically vetted by the
  numerical planner.

- **Prompt transparency in the SI.**
  Supplementary Note S5 now includes a representative schematic prompt packet:
  recent measured points, batch/measurement counts, allowed mode choices, a
  one-line automatically generated ambiguity description, a bounded audit menu,
  and the strict JSON response contract.

## Deliberate non-changes

- **Figure 10b was kept as-is.**
  The panel is being used as a compact final-state posterior summary for the
  pilot run, not as a posterior-evolution trace. The manuscript text now leans
  on that interpretation rather than trying to make Figure 10 do the same job as
  Figure 9d.

- **No “count while moving” future-work claim.**
  We did not add this. For the point-defined TAS measurements in this paper, the
  combination of position fidelity requirements and typical count-time balance
  makes it a poor fit for the present study.

- **No “refine while moving” claim.**
  We did not add this either. In the regimes studied here, numerical planning is
  already cheap relative to counting, so it is not the dominant wall-clock
  bottleneck.

## Current framing to keep in mind

- The paper is strongest when it argues for the **task decomposition**:
  discovery, discrimination, and refinement are different control problems.
- The Section 5 result should stay framed as a result about the
  **falsification channel** first, with LLM-specific advantages limited to
  interface flexibility and broader ambiguity handling.
- The paper now states explicitly that the present workflow assumes a
  **candidate-model library** for in-loop discrimination. In the current
  framing, that library can be seeded by crystal-chemical heuristics such as
  Goodenough–Kanamori reasoning, while more autonomous hypothesis generation
  remains future work rather than a demonstrated capability.

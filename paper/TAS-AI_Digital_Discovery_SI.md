---
title: "Supplementary Information for: Accelerating Quantum Materials Characterization: Hybrid Active Learning for Autonomous Spin Wave Spectroscopy"
---

## Supplementary Note S1. Enhanced Log-GP: the 1D taper and linear-intensity variance weighting

The underlying Log-GP reconstruction idea is due to Teixeira Parente *et al.* and the JCNS neutron active-learning work cited in the main text; it is not introduced in this manuscript. What is specific to the present implementation is the set of safeguards that stabilize the Log-GP policy for our TAS benchmark domain.

In the agnostic Log-GP phase, uncertainty sampling in **log-intensity space** can over-prioritize the boundaries of the search domain. GP predictive variance is naturally largest at the edges of a bounded box, and log-variance treats dim background regions as comparably valuable to bright signal regions. The resulting acquisition can become **edge-locked**, repeatedly sampling high-energy or high-$|H|$ boundary points that do not intersect real signal support.

We mitigate this failure mode with two complementary changes.

1. **Linear-intensity variance weighting.** Rather than ranking candidates
   by raw log-space variance alone, the acquisition is weighted by a
   linear-space variance proxy so that dim background regions are not
   treated as equally valuable as bright signal regions. When the
   surrogate exposes log-space mean and variance $(\mu, \sigma^2)$
   directly, this can be written as

$$
\mathrm{Var}(I) = \left(e^{\sigma^2}-1\right)e^{2\mu+\sigma^2},
$$

where $\mu$ and $\sigma^2$ are the GP posterior mean and variance in log-intensity space and $\mathrm{Var}(I)$ is the corresponding approximate variance in linear intensity units. In the current codebase, the live path may instead use the backend's directly exposed posterior standard deviation in linear space; the common design principle is that acquisition is ranked in linear-intensity variance units rather than raw log-space variance.

2. **A 1D cosine taper in energy.** We apply a soft window in $E$ that
   smoothly downweights the outer 10% of the energy domain while leaving the
   interior nearly unchanged.

A stronger 2D taper in both $E$ and $H$ further suppresses edge selection, but in this model it can over-penalize low-$|H|$ regions where the dispersion is strongest. We therefore retain the energy-only taper in the reported benchmarks.

![Effect of linear-intensity variance weighting and boundary tapering on Log-GP active selection. Left: no taper (edge locked). Middle: 1D energy taper used in this work. Right: stronger 2D taper, which was tested but suppresses low-$|H|$ ridge coverage in this model.](figures/loggp_taper_comparison.png){#fig:loggp-taper width=90%}

## Supplementary Note S2. Benchmark runtime accounting

Detailed benchmark provenance has been moved out of the SI and into the reproducibility/referee material:

- `REPRODUCIBILITY.md`
- `REVIEWER_GUIDE.md`
- `paper/data/README.md`
- `paper/data/table2_provenance.md`

The controller itself operates at sub-second algorithmic latency, but benchmark sweeps incur additional end-to-end digital-twin overhead. Table S1 reports the canonical planner-side quantity preserved in the benchmark JSON files: `mean_time_per_suggestion`. This is the reproducible per-suggestion compute cost of the benchmark harness itself. We do not report a separate elapsed-per-run column because those wall-clock values depend on execution environment and are not preserved consistently in the final benchmark artifacts.

*Table S1.* Mean planner-side compute time per simulated suggestion from the canonical benchmark JSON artifacts. Internal archive keys such as `faira_*` and `fairpf_*` are omitted here because they are implementation-facing provenance labels rather than scientific benchmark categories.

| **Benchmark family** | **Method**       | **Mean time per suggestion (ms)** | **Std. dev. (ms)** |
| -------------------- | ---------------- | --------------------------------: | -----------------: |
| Analytic             | Grid             |                           0.00028 |            0.00001 |
| Analytic             | Random           |                           0.00959 |            0.00034 |
| Analytic             | Enhanced Log-GP  |                             847.8 |               58.4 |
| Analytic             | TAS-AI (physics) |                              15.98 |               0.23 |
| PySpinW+CN           | Grid             |                           0.00081 |            0.00002 |
| PySpinW+CN           | Random           |                           0.01658 |            0.00032 |
| PySpinW+CN           | Enhanced Log-GP  |                             965.9 |               56.6 |
| PySpinW+CN           | TAS-AI (physics) |                              16.03 |               0.23 |

These values come directly from the archived benchmark outputs and are therefore reproducible from the current repository state. They should be interpreted as planner-side digital-twin cost, not beamtime or wall-clock queueing cost.

### S2.1 Blind-reconstruction metric and threshold

The blind benchmark figures and Table 1 use a common reconstruction metric defined on a fixed reference grid in $(H,E)$ space:

$$
\varepsilon_{\mathrm{recon}}
=
\frac{\sum_i \left|I_i^{\mathrm{pred}}-I_i^{\mathrm{true}}\right|\,I_i^{\mathrm{true}}}
{\sum_i \left(I_i^{\mathrm{true}}\right)^2}.
$$

Here $I_i^{\mathrm{true}}$ is the ground-truth intensity at reference-grid point $i$, and $I_i^{\mathrm{pred}}$ is the reconstructed intensity inferred from the method's raw measurements. In the current benchmark implementation, $I_i^{\mathrm{pred}}$ is obtained from inverse-distance interpolation over the raw observed intensities rather than from each method's internal surrogate state. This makes the score an acquisition-quality metric rather than a benchmark of surrogate-specific reconstruction machinery. The canonical implementation is `compute_reconstruction_error()` in `tasai/examples/benchmark_jcns.py`.

For the analytic benchmark families, a run is counted as successful when $\varepsilon_{\mathrm{recon}} \le 0.20$ within the fixed budget. For the corrected PySpinW+Cooper-Nathans rows, no method reaches that threshold within $N=300$, so the informative quantity is the final error at budget rather than the median measurements-to-threshold.

### S2.2 Fixed count time and MCTS settings

In the current time-aware refinement study, count time is **not** optimized jointly with location. The refinement benchmark uses a fixed dwell time of 10 s per measurement and optimizes only the location-dependent information-rate objective. This is why the wall-clock gains in Figure 5 come from route choice and information density rather than from adaptive dwell-time allocation.

When the optional MCTS batch planner is used, the current core defaults are:

- `n_simulations = 100`
- `exploration_constant = 1.41`
- `n_candidates = 20`
- `rollout_depth = 3`
- `discount_factor = 0.95`
- `max_depth =` requested batch size

These values are taken directly from the active implementation in `tasai/core/mcts.py`. They were chosen as practical short-horizon settings for motion-aware batched planning rather than as a separately optimized benchmark target.

## Supplementary Note S3. Additional remarks on stopping, local approximation, and escalation

Inside the physics-informed loop, TAS-AI uses fast multi-start local fits and a Laplace/Levenberg–Marquardt covariance approximation because the planner must evaluate many candidates in real time. These approximations are appropriate when the posterior is locally unimodal and the candidate model family is already close to the truth, but they can under-represent uncertainty in strongly multimodal settings.

To guard against that failure mode, the code uses simple escalation triggers. In the controlled single-branch tests, reduced $\chi^2$ values that remain high (for example, above approximately 5) or parameter estimates that repeatedly hit bounds trigger a heavier posterior stage. In current practice, that heavier stage is reserved for batch boundaries or offline validation rather than every in-loop update.

These escalation rules are part of the reason the manuscript distinguishes three operating regimes: agnostic discovery, physics-informed inference, and strategic audit. Each regime uses a different approximation budget and a different notion of what constitutes "useful" information.

### S3.1 Small-seed robustness checks for refinement and discrimination

To provide a first robustness check beyond the representative runs shown in the main text, we ran a small three-seed sensitivity sweep for the controlled time-aware refinement benchmark of Figure 5 and for the simple NN-vs-$J_1$-$J_2$ discrimination benchmark of Figure 6. These results are archived in `paper/data/reviewer_sensitivity_20260403.json`.

For the Figure 5 refinement setup, the ranking across three seeds is:

- `tas_ai` succeeds in 3/3 runs, with median convergence at 11 measurements and median convergence time 225 s.
- `random` also succeeds in 3/3 runs, with the same median measurement count but median convergence time 413 s.
- `grid` does not reach the threshold in any of the 3 runs within the same budget.

The multi-seed result therefore supports a **wall-clock** advantage for TAS-AI rather than a measurement-count advantage: the motion-aware planner is consistently faster in elapsed time even when the number of measurements needed is similar to the best random runs.

For the simple NN-vs-$J_1$-$J_2$ discrimination benchmark, the result is stable across the tested seeds: all 3/3 runs reach decisive evidence after the initial six-point seed set. This is consistent with the main-text claim that once the model family is appropriate and the discriminating region is already sampled, the in-loop model-selection signal is very strong.

### S3.2 Sensitivity to the exploration exponent

The motion-aware refinement policy uses an empirical exploration exponent $\eta$ (denoted $\gamma$ in the main text) in the information-rate score. A small three-seed sweep over $\eta \in \{0.5, 0.7, 0.9\}$ gives:

| **$\eta$** | **Success** | **Median measurements to threshold** | **Median convergence time (s)** | **Mean final RMS** |
| ---------: | ----------: | -----------------------------------: | ------------------------------: | -----------------: |
| 0.5        | 3/3         | 11                                   | 225                             | 0.0169             |
| 0.7        | 3/3         | 11                                   | 225                             | 0.0104             |
| 0.9        | 3/3         | 11                                   | 220                             | 0.0066             |

Within this limited sweep, the refinement result is not brittle across the tested range. The higher value $\eta=0.9$ is slightly better on final RMS and slightly faster in median elapsed time, while the current default $\eta=0.7$ remains safely inside the stable regime rather than at a knife-edge optimum.

### S3.3 AIC versus WAIC in the simple discrimination test

The main manuscript uses AIC-derived weights as a pragmatic real-time model-selection proxy. As a focused check on that choice, we computed an offline grid-based WAIC comparison for the simple NN-vs-$J_1$-$J_2$ discrimination setup after 8 measurements. In the tested seed, AIC and WAIC agree completely on the model ranking: both assign effectively unit weight to the correct $J_1$-$J_2$ model and negligible weight to the NN-only alternative.

This does **not** prove that AIC and WAIC are interchangeable in all TAS-AI regimes. It does show that, in the simple controlled discrimination setting corresponding to Figure 6, the manuscript's AIC-based conclusion is not being driven by a disagreement with this standard offline predictive criterion.

### S3.4 Empirical coverage of the Laplace credible intervals

To check whether the fast Laplace/Levenberg--Marquardt uncertainty estimates are actually calibrated in the controlled Figure 5 refinement setting, we ran a 10-seed coverage calculation on the TAS-AI policy. For each seed we computed the final local covariance from the numerical Hessian of the $\chi^2$ objective at the converged parameter estimate, formed nominal 90% marginal intervals for $(J_1,J_2,D)$, and counted the fraction of seeds in which those intervals contained the true parameter values. The seed-level results are archived in `paper/data/laplace_coverage_refinement_20260415.json`.

| **Parameter** | **Nominal coverage** | **Empirical coverage** | **Hit count** | **Median 90% interval width (meV)** |
| ------------: | -------------------: | ---------------------: | ------------: | ----------------------------------: |
| $J_1$         | 0.90                 | 0.30                   | 3/10          | 0.0171                              |
| $J_2$         | 0.90                 | 0.70                   | 7/10          | 0.0277                              |
| $D$           | 0.90                 | 0.10                   | 1/10          | 0.0258                              |

This is clear **under-coverage**, especially for $J_1$ and $D$. With 10 seeds, the coverage estimates themselves carry substantial sampling uncertainty (binomial standard error $\approx 0.14$ at coverage 0.30), but the qualitative conclusion of substantial under-coverage is unambiguous. Two effects contribute. First, the real-time estimator is built around deterministic local optimization rather than full posterior sampling, so curvature around the best fit does not capture global posterior mass. Second, some seeds approach parameter bounds or numerically stiff directions, causing the finite-curvature estimate to become overconfident or even collapse to near-zero marginal variance.

The practical implication is that the manuscript's uncertainty bars should be interpreted as **fast local error surrogates** rather than as fully validated Bayesian credible intervals. This does not undermine the main control argument of the paper, which depends primarily on ranking, discrimination speed, and routing behavior, but it does set a clear limit on how strongly one should interpret the nominal Laplace intervals until heavier posterior calibration is added.

## Supplementary Note S4. Mathematical origin of posterior lock-in and the ghost-optic audit ablation

This note formalizes the posterior lock-in mechanism identified in §3.4 of the main text and provides the detailed setup for the ghost-optic ablation benchmark reported in §5.3.1.

### S4.1 The lock-in mechanism

Consider two candidate spectral models:

- $M_A$: acoustic-only spectrum with one dominant bright branch.
- $M_B$: acoustic+optic spectrum, where the additional branch carries only a small fraction of the dominant spectral weight.

When the initial seed measurements are placed only around the bright acoustic feature, the wrong one-branch leader ($M_A$) already has high posterior weight before any explicit falsification probe is taken. The one-shot falsification value at energy $E$ is

$$
G_{\mathrm{false}}(E)=\frac{\left[I_B(E)-I_A(E)\right]^2}{2\sigma^2(E)}.
$$

Here $I_A(E)$ and $I_B(E)$ are the predicted intensities of the two competing models at energy $E$, and $\sigma(E)$ is the corresponding measurement uncertainty. This quantity peaks at the weak optic branch, whereas the local refinement utility of the current one-branch leader peaks on the steep flanks of the already observed acoustic branch. When the posterior already heavily favors the wrong leader, the refinement term is systematically over-weighted relative to the falsification term, so the planner is biased toward more acoustic-branch refinement even though a strongly discriminative optic probe remains kinematically accessible.

In other words, the Laplace approximation of parameter information concentrates utility on the bright feature precisely because that is where gradients $\partial S/\partial\theta$ are largest, while the cross-model intensity difference $I_B(E)-I_A(E)$ that would drive falsification is largest on the weak branch where the refinement gradient is small. This asymmetry is the mathematical origin of the silent-data posterior lock-in discussed in the main text.

### S4.2 Ghost-optic ablation details

The ghost-optic benchmark is a fixed-$Q$ two-Lorentzian toy spectrum over $E\in[0,20]$ with additive background 0.1. The acoustic-only comparator contains a dominant peak at $E=5$ with amplitude 100 and linewidth $\gamma=0.5$, while the truth adds a weak optic peak at $E=15$ carrying 5% of the acoustic amplitude with the same linewidth. The common seed consists of four acoustic-centered measurements at $E=\{4.25,4.75,5.25,5.75\}$, intentionally leaving the optic region unprobed at initialization.

From this common seed, four one-seed policies are compared:

- `None`: fixed seed followed by pure refinement of the current leader.
- `Log-GP`: fixed seed followed by a 1D GP variance explorer — the bare Log-GP variant described in §3.1 of the main text.
- `Max-disagreement`: a deterministic top-two falsification rule using the same bounded candidate menu as the LLM audit path.
- `LLM`: fixed seed followed by the same refinement loop plus the constrained LLM audit layer, using the same shared candidate menu and strict JSON contract as the main overseer.

The results are reported in Table 4 of the main text. All four policies eventually recover, but the timescale separation is large: None stays on the bright branch through a long wrong-leader episode; Log-GP reduces that dwell by allocating more falsification-oriented batches; and both Max-disagreement and LLM eliminate wrong-leader dwell by explicitly targeting the falsification region. A five-seed rerun of this comparison is reported in Supplementary Note S5.2 and preserves the same qualitative pattern.

These runs should be interpreted narrowly. They do not replace the full TAS-AI spin-wave benchmarks in the main text. Their purpose is to demonstrate, in a controlled setting, that a posterior-dominated refinement policy can accumulate substantial wrong-leader dwell on the bright branch, that exploration alone can recover, and that a falsification-oriented audit layer can recover much faster when the missing feature is strategically under-sampled.

## Supplementary Note S5. Bilayer ferromagnet audit ablation with shared action space

To move beyond the minimal ghost-optic benchmark, we implemented a simple analytic square-lattice bilayer ferromagnet backend in which the acoustic branch remains bright while a weak $L$-suppressed optic branch provides the falsifying signal. In the cleaned version reported in Table 5 and Figure 12 of the main text, the single-branch comparator is matched to the same $L$-dependent acoustic weight as the bilayer truth, so the models differ only through the presence or absence of the optic branch. The optic-region metric is tightened accordingly: `optic_region_hit_fraction` counts batches containing at least one measurement within a narrow tolerance of the optic branch.

Four controllers are compared: a refinement-only baseline (None), two deterministic non-LLM rules (Hybrid, Max-disagreement), and the constrained LLM overseer (LLM). All four operate over the same action space: a switch between bare Log-GP remapping (`loggp_active`) and `physics` refinement (see §3.1 of the main text for the distinction between bare and enhanced Log-GP). They differ only in how that mode decision is made and whether audit-probe injections are allowed. This is a stricter comparison than a setup in which the LLM is treated as a pure point selector, because every controller shares the same control interface and the same candidate menu; it therefore isolates whether the falsification gain comes from the LLM specifically or from the falsification principle itself, which the deterministic rules probe from different angles.

For reproducibility, the current overseer path uses the local mailbox watcher in `scripts/llm_danse2_watcher.py`. The watcher draws from three local CLI-backed providers: Claude Code (default model: Opus 4.5), Gemini CLI (default model: Gemini 3), and Codex CLI pinned to `gpt-5.2-codex`; in overseer mode the decider rotates across providers by batch unless explicitly pinned. The manuscript runs use provider CLI defaults without separate temperature sweeps or sampling overrides; reproducibility is enforced through the bounded prompt contract, strict JSON parsing, the fixed shared action menu, and guardrail fallbacks when malformed output is returned.

The "natural-language description" given to the overseer is generated automatically by the prompt builder rather than typed by a human during the run. In the main closed-loop pilot this prompt is assembled from current loop state — the posterior ranking, recent measurement history, time since the last Log-GP batch, an audit recommendation flag, and the bounded discrimination menu. In the bilayer ablation, the local prompt builder adds a scripted semantic hint about the operative ambiguity but does not expose hidden coordinates, the true model identity, or any action outside the shared menu.

A representative prompt packet is intentionally compact. In schematic form, it contains: `(i)` a short tabular history of recent measured points and intensities, `(ii)` the current batch and measurement counts, `(iii)` the allowed routing choices (`loggp_active` or `physics`), `(iv)` a one-line ambiguity description generated from loop state (for example, whether the remaining uncertainty is gap-vs-no-gap or whether a weak optic branch may be missing), `(v)` a small bounded menu of candidate audit probes already vetted by the numerical planner, and `(vi)` an instruction to return strict JSON without adding coordinates or actions outside the menu. The exact wording varies by benchmark, but the interface contract is fixed.

### S5.1 Deterministic hybrid-router specification

For the reported one-seed comparison, the deterministic Hybrid router uses the same menu of allowed actions and has no access to the true model. Its decision logic is:

1. **Minimum run length.** The current mode is held for at least two measurements before any switch is considered.
2. **Forced periodic exploration.** A `loggp_active` batch is forced whenever six measurements have elapsed since the previous Log-GP batch.
3. **Ambiguity triggers.** Outside the forced-exploration condition, the router selects `loggp_active` whenever any of the following hold: posterior entropy exceeds 0.20, falsification-region coverage remains below 0.10, or the posterior margin (difference between the top two model weights) falls below 0.35.
4. **Default.** If none of the above triggers fire, the router selects `physics` refinement.

These thresholds were set before examining the LLM comparison and were not tuned to favor or disadvantage any policy.

### S5.2 Five-seed robustness check for the Section 5 ablations

To test whether the one-seed ablation pattern was robust or merely anecdotal, we reran the ghost-optic and bilayer benchmarks over five seeds per policy. The five-seed medians naturally differ from the one-seed values in Tables 4 and 5 of the main text because those tables report a single representative run rather than aggregate statistics. Tables S5 and S6 summarize the resulting time-to-decisive and wrong-leader-dwell statistics as medians with interquartile ranges. For policies that do not reach decisive correct selection in every seed, we report the median and IQR over the successful runs and list the success rate explicitly.

*Table S5.* Five-seed ghost-optic audit ablation. Time to decisive is reported as median (IQR) over successful runs; success reports the number of successful seeds out of five.

| **Policy** | **Time to decisive, median (IQR)** | **Wrong-leader dwell, median (IQR)** | **Success** |
| ---------- | ---------------------------------: | -----------------------------------: | ----------: |
| None       | 30 (30--30)                        | 25 (21--25)                          | 2/5         |
| Log-GP     | 29 (24--30)                        | 15 (15--20)                          | 5/5         |
| Max-disagreement | 9 (9--9)                    | 0 (0--0)                             | 5/5         |
| LLM        | 9 (9--9)                           | 0 (0--0)                             | 5/5         |

*Table S6.* Five-seed bilayer ferromagnet audit ablation with the shared action space.

| **Policy** | **Time to decisive, median (IQR)** | **Wrong-leader dwell, median (IQR)** | **Success** |
| ---------- | ---------------------------------: | -----------------------------------: | ----------: |
| None       | 23 (18--23)                        | 5 (0--5)                             | 5/5         |
| Hybrid     | 8 (8--13)                          | 0 (0--0)                             | 5/5         |
| Max-disagreement | 8 (8--8)                    | 0 (0--0)                             | 5/5         |
| LLM        | 8 (8--8)                           | 0 (0--0)                             | 5/5         |

The five-seed rerun sharpens the conclusion suggested by the one-seed tables. In the ghost-optic benchmark, both Max-disagreement and LLM eliminate wrong-leader dwell and reach decisive correct selection at 9 measurements in every seed, while bare Log-GP remains much slower and the refinement-only baseline succeeds in only two of five seeds. In the bilayer benchmark, LLM and Max-disagreement match each other exactly across all five seeds, and the deterministic Hybrid router remains strong but slightly less consistent (one seed requires 23 measurements, giving an IQR upper bound of 13).

The LLM performs well and robustly in both analytic ablations, but the precise conclusion is that, for these two-model benchmarks, the bounded deterministic top-two falsification rule is already sufficient to capture the gain. This strengthens the narrower interpretation of the main text: the active ingredient is the falsification channel itself, while any stronger claim of an LLM-specific advantage requires broader multi-model or structurally harder benchmarks such as the trap in §S5.3. The LLM's generality across problem descriptions — handling the ghost, bilayer, and multi-model trap benchmarks through the same interface without per-problem engineering — remains its primary architectural advantage even when the two-model benchmarks show no performance gap.

### S5.3 Controlled multi-model trap for top-two versus broader falsification

The bilayer and ghost benchmarks show that a falsification-oriented audit channel can reduce wrong-leader dwell, but they leave open a sharper question: is a deterministic top-two disagreement heuristic already sufficient, making the LLM implementation unnecessary? In the easier analytic cases, that top-two ansatz is in fact quite strong. We therefore constructed a **targeted stress test** whose purpose is not to represent the average operating regime, but to isolate a specific failure mode of local top-two falsification.

The trap is a narrow-window synthetic three-model benchmark with:

- true model $M_4$: bright ridge plus a weak hidden pocket,
- runner-up model $M_2$: nearly the same bright ridge and the same pocket, and
- lower-ranked model $M_3$: nearly the same ridge but **no** pocket.

The fixed seed state is chosen so that the initial posterior ranking is $M_4 > M_2 > M_3$. Under this ranking, a top-two disagreement policy naturally prefers additional **bright-branch** refinement because $M_4$ and $M_2$ differ only weakly there, while the decisive falsifier against the lower-ranked $M_3$ sits in the weak hidden pocket. This setup is fair in the same sense as the ghost benchmark: all policies start from the same measurements, use the same bounded candidate menu, obey the same kinematic/selection rules, and differ only in how they rank the allowed audit actions.

The policies compared are:

- None: no explicit audit injection;
- Max-disagreement: deterministic top-two falsification, using only the current leader and runner-up;
- Max-disagreement-all: deterministic broader falsification, scoring the leader against all currently fitted competitors; and
- LLM: the constrained LLM audit layer, given the same bounded menu as Max-disagreement-all.

For the reported one-seed stress test, the outcome is:

| **Policy** | **Final $P(M_4)$ at $N=8$** | **Pocket probe used?** |
| ---------- | --------------------------: | ---------------------: |
| None                   | 0.663 | no  |
| Max-disagreement       | 0.668 | no  |
| Max-disagreement-all   | 0.843 | yes |
| LLM                    | 0.843 | yes |

Three observations follow. First, the None baseline and the top-two Max-disagreement rule are nearly indistinguishable: without a pocket probe, neither can suppress the pocket-free $M_3$ competitor, so the final posterior on $M_4$ remains under 0.67. Second, the broader Max-disagreement-all rule targets the hidden pocket and strongly suppresses $M_3$, raising $P(M_4)$ to 0.84. Third, the constrained LLM audit makes the same hidden-pocket choice from the same bounded menu and reaches the same posterior.

The lesson is narrow but useful. The top-two falsification rule is a serious baseline and should not be dismissed; in easier cases it works well. But it is **not universally sufficient** in multi-model settings, because the decisive falsifier may separate the current leader from a lower-ranked model rather than from the current runner-up. In such cases, both a broader deterministic falsification rule and the constrained LLM audit make the same strategically correct choice under identical guardrails. We therefore interpret this stress test as support for the **falsification-channel idea** rather than as proof that the LLM is uniquely superior to all deterministic alternatives. The cleaner conclusion is that local top-two disagreement is a strong ansatz but not a complete one, and a broader strategic audit layer is motivated precisely when the posterior trap involves more than two live hypotheses.

### S5.4 Prior and background sensitivity in the closed-loop discrimination stack

We ran a coarse sensitivity check on the pilot closed-loop discrimination stack (Figure 10 of the main text) at sampled checkpoints of 40, 60, and 90 measurements. Three variants were compared: the default chemically motivated prior weights, equal model priors, and the default priors with the gapless-background lock disabled.

At all three sampled checkpoints, the default setting ranks the full model $M_4$ first, but only moderately over $M_2$ (posterior ratio about 2.58 rather than decisive support). With **equal priors**, the ranking flips and $M_2$ becomes the leader over $M_4$ at all three checkpoints (ratio about 2.72). Disabling the gapless-background lock, by contrast, does not change the ranking at these checkpoints relative to the default setting.

The practical implication is that the prior weights matter substantially in this integrated closed-loop regime, whereas the background-freezing switch is not what controls the final ranking in this coarse sensitivity pass. This does not invalidate the rationale for freezing nuisance backgrounds; it indicates that the posterior evolution in Figure 10 should not be interpreted as strongly prior-insensitive.

# Repository Walkthrough

This document is for a reader who has already read the TAS-AI paper and now wants to understand how the code is organized, how the library is meant to be used, how the paper bundle relates to it, and where the important tradeoffs and pitfalls are.

It covers two codebases that are intentionally used together for this reproducibility bundle:

- the paper bundle repo: the manuscript, figure-generation scripts, and paper-specific closed-loop demos
- the public `tasai` code repo: the reusable library installed into the paper environment

The quickest companion references are:

- `README.md`
- `REPRODUCIBILITY.md`
- `REVIEWER_GUIDE.md`
- `MODULE_MAP.md`

This guide goes beyond those files and follows the actual code structure.

## 1. What lives where

The first thing to understand is that this repository is not a normal Python package repo. It is a paper/reproducibility repo that depends on an external `tasai` install from the public code repository.

At a high level:

- `paper/` contains manuscript sources, figure scripts, built artifacts, and provenance data.
- `simulations/` contains the paper-facing Python scripts that generate key figures and closed-loop demonstrations.
- `prompts/` and `scripts/` contain the LLM prompt templates and mailbox/watcher tooling used for the guarded audit-layer experiments.
- the reusable `tasai` library lives in a separate repository and is installed into the environment from the pinned paper tag.
- `deps/rescalculator/` contains the resolution backend used by the library and by the paper demos.

One current-provenance rule from the updated reproducibility docs is worth keeping in mind from the start:

- benchmark claims now treat `paper/data/final/` as canonical
- Section 5.3 audit-layer claims treat `paper/data/ablation_runs/` as canonical
- operational mailbox residue such as `run_logs/` is not canonical manuscript provenance

This split is deliberate:

- the reusable library lives in the separate public code repo `usnistgov/tasai`
- the paper repo owns the figure scripts, manuscript-specific orchestration, checkpoint conventions, and plot generation

![Repository architecture overview](paper/figures/repo_walkthrough_architecture.png)

That separation is good for reproducibility, but it means you must constantly ask:

- "Am I reading generic library code?"
- "Or am I reading paper-specific orchestration?"

If you do not keep that distinction in mind, the repo feels more inconsistent than it really is.

## 2. The core architectural idea

Conceptually, TAS-AI is built around a simple loop:

1. represent a physical model with tunable parameters
2. represent an instrument that can measure points in `(h, k, l, E)` space
3. infer current uncertainty from the accumulated data
4. score candidate measurements by expected value
5. execute or simulate the next measurement
6. repeat

In the reusable library, those responsibilities are split across modules:

- `tasai.physics.*`: forward models
- `tasai.instrument.*`: geometry, simulator, resolution, motion, remote proxy
- `tasai.inference.*`: posterior fitting and sampling
- `tasai.core.*`: acquisition, forecasting, GP exploration, MCTS
- `tasai.examples.*`: end-to-end entry points that show the intended usage

In the paper bundle, the most important script, `simulations/toy_closed_loop.py`, does not simply call those abstractions directly. Instead, it re-implements a substantial amount of orchestration in one file so that the paper logic, plotting, checkpointing, and policy variants remain self-contained and stable across manuscript revisions.

That is one of the most important architectural decisions in the repo:

- the library is modular
- the paper demo is intentionally monolithic

This is not an accident. It trades elegance for reproducibility and figure control.

![Closed-loop workflow](paper/figures/closed_loop_workflow.png)

## 3. How to read the code without getting lost

There are two good reading orders.

### 3.1 If you want to understand the reusable library first

Read in this order:

1. `README.md` in `usnistgov/tasai`
2. `tasai/examples/example_parameter_determination.py`
3. `tasai/examples/example_model_discrimination.py`
4. `tasai/examples/example_with_motor_motion.py`
5. `tasai/physics/base.py`
6. `tasai/instrument/base.py`
7. `tasai/core/acquisition.py`
8. `tasai/core/forecast.py`
9. `tasai/inference/mcmc.py`
10. `tasai/instrument/simulator.py`
11. `tasai/instrument/motors.py`
12. `tasai/instrument/resolution.py`
13. `tasai/physics/spinwave.py`

### 3.2 If you want to understand the paper implementation first

Read in this order:

1. `REPRODUCIBILITY.md`
2. `MODULE_MAP.md`
3. `simulations/toy_closed_loop.py`
4. `simulations/toy_closed_loop_llm_overseer.py`
5. `simulations/create_closed_loop_animation_from_log.py`
6. `simulations/run_audit_ablation.py`
7. `paper/scripts/plot_llm_inloop_snapshot.py`
8. `paper/scripts/make_figure4_scenarios.py`
9. `simulations/exchange_path_analysis.py`
10. `simulations/hybrid_exploration_demo.py`

The first path teaches the abstractions. The second path teaches the actual paper story.

## 4. The reusable `tasai` library

The library snapshot lives in the separate public code repository. The important thing to know is that its public architecture is cleaner than the paper bundle's architecture.

### 4.1 `tasai.physics`: what the model predicts

Start with `tasai/physics/base.py`.

This file defines the two central abstractions:

- `Parameter`: a named model parameter with bounds, prior type, and metadata
- `PhysicsModel`: an abstract base class that every inferable model is supposed to satisfy

`PhysicsModel` is the conceptual heart of the library. It standardizes:

- parameter storage
- prior evaluation
- parameter vector conversion for samplers
- `compute_intensity(...)`
- `log_likelihood(...)`
- `log_posterior(...)`

This is the right abstraction boundary. It means the rest of the stack can ask:

- "If I pick this measurement point, what intensity should I expect?"
- "How likely is the current data under this parameter setting?"

without needing to know whether the underlying model is a spin-wave Hamiltonian, an order-parameter curve, or something else.

That said, most of the end-to-end examples do not fully lean on the abstract interface. Several examples work with concrete model classes directly because they are pedagogical and optimized for clarity over perfect abstraction hygiene.

### 4.2 `tasai.instrument`: how the measurement is represented

Read `tasai/instrument/base.py`.

This file defines:

- `MeasurementPoint`: a configuration in `(h, k, l, E)` space, plus count time and optional angles
- `MeasurementResult`: the observed intensity and uncertainty
- `InstrumentInterface`: the abstract contract for a real instrument, simulator, or proxy
- `TASGeometry`: reciprocal-space to angle conversion and accessibility logic

This module is where the code stops being abstract machine learning and starts being instrument software.

Architecturally, this is important because it separates:

- scientific intent: "measure `(h, k, l, E)`"
- mechanical execution: "what motor angles produce that point?"

That separation is the reason the same planning code can work against:

- a simulator
- a remote proxy to a real beamline
- a potential replay backend

### 4.3 `tasai.inference`: how parameter uncertainty is estimated

Read `tasai/inference/mcmc.py`.

The design here is pragmatic rather than doctrinaire:

- use BUMPS/DREAM if available
- fall back to `emcee`
- fall back again to a simple Metropolis method

That fallback chain tells you a lot about the repo's priorities:

- it wants to preserve the "NIST neutron software" lineage via BUMPS
- it also wants examples to stay runnable in imperfect environments

`MCMCRunner` expects a `PhysicsModel` and data arrays, and produces posterior samples. Those samples are what the acquisition and forecasting layers operate on.

The important architectural point is that the library tries to keep:

- model definition
- inference engine
- acquisition policy

loosely coupled. In other words, acquisition operates on posterior samples, not on one specific optimizer's internal state.

### 4.4 `tasai.core`: what to measure next

This is the decision-making layer.

#### `acquisition.py`

This file defines the scoring functions.

The central abstraction is `AcquisitionFunction`, with concrete implementations like:

- `HHAcquisition`
- `UncertaintyAcquisition`
- `ANDiEAcquisition`
- `CompositeAcquisition`

The conceptual rule is:

- candidates go in
- posterior samples go in
- scored candidates come out

`HHAcquisition` is especially important because it encodes the information-rate idea:

- not just "how informative is this point?"
- but "how informative is this point per unit time?"

That denominator is where movement time and count time enter the science logic.

This is one of the strongest architectural ideas in the codebase: instrument cost is not bolted on later, it is part of the utility function.

#### `forecast.py`

This file implements the "forecast multiple future points from one posterior" idea.

Instead of:

1. measure one point
2. re-run full MCMC
3. choose the next point

the forecaster approximates posterior updates with importance reweighting between steps. That is a major latency optimization and directly reflects the practical reality of autonomous instruments: you often want the next few moves queued while the instrument is still in motion.

The architectural tradeoff is clear:

- exact posterior refresh after every step would be cleaner
- but forecasting is much more practical for online control

#### `gaussian_process.py`

This file contains the agnostic exploration side:

- `LogGaussianProcess`
- `AgnosticExplorer`
- `HybridExplorer`

This is the part of the library that speaks to the paper's "agnostic discovery first, physics-informed refinement later" framing.

The key design choice here is log-space GP modeling for non-negative intensities. That is not a cosmetic detail. It is a direct answer to a known weakness of naive GP modeling of scattering intensity.

This module is therefore not just an implementation choice; it is part of the scientific argument of the paper.

#### `mcts.py`

This file provides MCTS-based batch and trajectory planning.

It is the most algorithmically ambitious planner in the library, but it is not the first place you should start reading unless you already understand:

- the candidate representation
- the acquisition score
- the move-time model

Read it after the simpler examples. Otherwise it is easy to mistake it for the whole library instead of one planning option.

### 4.5 `tasai.instrument` support modules

#### `simulator.py`

`TASSimulator` is the main development-time execution backend.

It combines:

- geometry/accessibility checks
- optional resolution convolution
- Poisson counting noise
- simulated motion time
- measurement history

This is the bridge from "a scored candidate" to "a noisy synthetic observation".

#### `motors.py`

This file models motion cost explicitly. It includes:

- low-level motor configs
- a `TASMotorSystem`
- a `SimplifiedMotorModel`
- `MotionAwareAcquisition`

The code here encodes a strong architectural stance: motion is a first-class planning variable, not postprocessing.

That matters because many autonomous-science demos optimize only information, while real instruments pay a heavy price for movement.

#### `resolution.py`

This file wraps `rescalculator` and exposes:

- default TAS configs
- Cooper-Nathans resolution matrices
- energy/Q FWHM estimates
- convolution helpers

This is one of the most important realism modules, and also one of the most fragile parts of the stack because it depends on external scientific software and compatibility shims.

### 4.6 `tasai.physics.spinwave` and `tasai.sunny`

You need to read these carefully because the naming can mislead.

`tasai/physics/spinwave.py` is the general spin-wave backend wrapper. It supports backend discovery and model construction, but in this snapshot:

- `pyspinw` is the real implemented backend path
- the Sunny path in `SpinWaveModel` is still a placeholder and raises `NotImplementedError`

Separately, `tasai/sunny/__init__.py` contains the lightweight analytic `SquareLatticeFM` model used throughout the examples.

This distinction matters:

- "Sunny" in the repo does not always mean "full Julia backend is wired up everywhere"
- sometimes it means "analytic model inspired by or standing in for the Sunny workflow"

This is a real pitfall for readers who assume every backend name is equally production-ready.

### 4.7 `tasai.examples`: the intended user-facing entry points

The examples are the best practical introduction because they show how the authors expect the code to be used.

The three most important examples are:

- `example_parameter_determination.py`
- `example_model_discrimination.py`
- `example_with_motor_motion.py`

These examples are valuable because they are smaller and cleaner than the paper demos.

They also reveal an important design pattern in the codebase:

- the examples are often self-contained mini applications
- they use library classes selectively rather than building one giant framework runner

That makes them easy to read and easy to adapt, but it also means the "canonical pipeline" is distributed across examples rather than encoded in one master API.

## 5. Code walkthrough of the main library examples

### 5.1 `example_time_aware_refinement.py`

This is now the most direct code path to the paper's current Figure 5.

It wraps the shared refinement machinery from `example_parameter_determination.py`
in a concrete known-family, motion-aware benchmark that scores methods by
elapsed experiment time rather than only by final RMS at a fixed count budget.

What it teaches:

- how the paper's current Figure 5 scenario is instantiated
- how motion cost enters the refinement objective
- how TAS-AI can be evaluated on time-to-threshold instead of static end-state error
- how the benchmark is collapsed into a concrete manuscript-facing script with fixed configuration

What it does not teach:

- the deeper estimator internals, which still live in `example_parameter_determination.py`
- the generic blind-discovery benchmark harness

The important architectural point is that this file is intentionally thin. It
exists as a manuscript-facing concrete benchmark, while the reusable fitter and
policy machinery remain in `example_parameter_determination.py`.

### 5.2 `example_parameter_determination.py`

This remains the best lower-level walkthrough for understanding the underlying
physics-informed refinement scaffolding.

It defines:

- a local `Measurement` dataclass
- a `ParameterEstimator`
- several measurement policies
- a runner that compares those policies

What it teaches:

- how a simple model object (`SquareLatticeFM`) is used for synthetic data generation and fitting
- how policy objects can propose points
- how convergence metrics are recorded and plotted

What it does not teach:

- the generic `InstrumentInterface`
- the generic `AcquisitionFunction` stack

So treat it as an approachable conceptual walkthrough, not as the deepest architecture demo.

It should also be read as a controlled refinement demo. The task assumes the model family is already fixed, and several of the comparison policies use branch-following heuristics when placing candidate measurements. That means the example is intentionally set up to study parameter contraction once signal structure is already known, rather than to test neutral end-to-end autonomy.

### 5.3 `example_model_discrimination.py`

This is the clearest expression of the paper's "measure where models disagree" logic.

The main class is `ModelDiscriminator`. Its workflow is:

1. create a true model
2. maintain competing candidate models
3. accumulate measurements
4. fit each candidate
5. update approximate evidences
6. pick the next point by maximizing disagreement

This script is valuable because it makes the model-discrimination objective explicit. It is much easier to reason about than the larger paper script.

#### What the file is trying to teach

This example is not trying to be a full general-purpose Bayesian model comparison framework. It is trying to teach one idea very clearly:

- autonomous planning can be driven by model disagreement rather than by coverage alone

That is why the example is intentionally narrow:

- only two candidate families
- only one key scientific question: does `J2` matter?
- only one reciprocal-space cut

That narrowness is a feature. It strips away enough complexity that you can see the central logic directly.

#### The main objects and their roles

The file defines a small local `Measurement` dataclass and one main controller class, `ModelDiscriminator`.

Inside `ModelDiscriminator.__init__`, the code sets up:

- the true model, either `SquareLatticeFM` or `NNOnlyModel`
- the competing candidate models
- equal initial model weights
- parameter dictionaries that hold the current best-fit values
- a measurement list
- a few small bookkeeping structures for the history of weights and fit quality

This is a good example of the style used across many TAS-AI examples:

- keep all run state in one small object
- make the state transitions easy to follow

#### How measurements are generated

`measure(H, E)` calls the true model's `simulate_measurement(...)` method with fixed counting assumptions:

- `count_time = 30.0`
- `count_rate = 20.0`

This is intentionally simple. The example is not trying to benchmark counting-time optimization. It is trying to isolate the model-discrimination problem.

That is an important point for readers: if you are looking for realistic instrument-time accounting, this is not the example for that. This file is about evidence accumulation, not wall-clock optimization.

#### How fitting is done

`fit_models()` is the most important method in the file.

It:

- converts the measurement list into arrays
- performs a grid search for the NN model over `(J1, D)`
- performs a larger grid search for the J1-J2 model over `(J1, J2, D)`
- stores the best-fit parameters for each model
- computes an approximate log-evidence using an AIC-style penalty
- updates the two model weights by a softmax over those scores

This is a very deliberate simplification. Instead of full posterior inference for each model, the example uses:

- discrete parameter sweeps
- a simple complexity penalty
- normalized evidence-like weights

That makes the example much easier to read and much faster to run, while still teaching the core discrimination logic.

It also exposes a key architectural theme in this codebase:

- exactness is often sacrificed in examples so that the autonomy logic stays visible

#### How the next point is chosen

`suggest_next_point()` is the real conceptual center of the example.

It loops over candidate `H` values and then:

- computes the NN and J1-J2 dispersions
- identifies energies where the two models differ
- evaluates the predicted intensity under both models
- scores the point by how different the predicted responses are
- discounts points that are too close to existing measurements via a novelty term

That scoring rule is a compact implementation of the "measure where the models disagree most, but do not waste time re-measuring the same region" principle.

This is the point in the example where the paper's intuition becomes executable code.

The fact that the quarter-zone region is naturally emphasized is not hard-coded magic. It comes from the structure of the two competing dispersions and from the scoring logic.

#### The experiment loop

`run(...)` is straightforward but worth reading closely. It:

- seeds the experiment with a small set of strategic initial points
- fits the models once
- repeatedly:
  - selects a few more discriminating points
  - measures them
  - refits both models
  - reports the current evidence and leader
- stops early if one model crosses a decisive-confidence threshold

This is a nice miniature of the broader TAS-AI pattern:

- initialize
- update candidate beliefs
- choose measurements that maximize decision value
- stop when the model ranking becomes decisive

#### The plotting section is diagnostic, not decorative

`plot_results(...)` is useful because each panel corresponds to a different mental check:

- panel (a): do the fitted dispersions line up with the data?
- panel (b): did the model weights actually separate over time?
- panel (c): did reduced chi-squared behave sensibly?
- panel (d): where in reciprocal space did the discrimination power come from?

This is a good habit throughout the repo: the best plots are usually not presentation graphics first, but debugging and interpretation tools.

#### What this example does not cover

It is important not to over-generalize from this file. It does not demonstrate:

- generic `InstrumentInterface` usage
- motion-aware planning
- forecasting from posterior samples
- full MCMC-based model comparison
- a larger candidate library

So the right way to use this example is:

- learn the cleanest form of model-disagreement planning here
- then move to the paper scripts or richer library modules for realism and scale

Architecturally, this file shows a recurring codebase pattern:

- approximate Bayes/evidence methods are often used for speed and interpretability
- exactness is traded away in favor of a runnable autonomous loop

### 5.3 `example_with_motor_motion.py`

This is the best short demonstration of why motion matters.

It builds:

- a candidate list
- an information proxy
- different path-ordering strategies
- a motion-aware TAS strategy

If you want to explain TAS-AI to someone in one sentence, this example is close to the essence:

"It is not enough to find informative points; you must find informative points in an order that is cheap to execute."

#### What the file is trying to teach

This example is intentionally narrow. It is not trying to solve a full adaptive experiment. It is trying to isolate one practical lesson:

- once the candidate measurements are known, ordering them intelligently can save a large amount of experiment time

That makes it a scheduling benchmark more than a discovery algorithm.

More precisely, it should be read as a controlled fixed-candidate scheduling study, not as a fair benchmark of end-to-end adaptive autonomy.

This is why the example starts from a fixed candidate set rather than discovering candidates online.

#### The main data model

The local `Candidate` dataclass is tiny:

- `H`
- `E`
- `label`

That is enough because the point of the example is not complicated metadata. The example only needs a fixed list of physically meaningful measurements along a dispersion.

#### Building the candidate list

`create_candidates(model, n_points=30)` is deliberately simple:

- sample `H` uniformly
- evaluate the model dispersion at each `H`
- create one measurement candidate on the branch

This means the comparison is about ordering, not about whether the candidate generator itself is good.

That design matters because it keeps the comparison fair:

- every strategy sees the same candidate set
- only the ordering policy changes

But it does **not** make the example a fair comparison of full planners, because the candidate set itself has already been prefiltered onto the model dispersion.

#### The information proxy

`info_gain(...)` is intentionally lightweight. It is not a full Bayesian information-gain calculation. Instead, it builds a heuristic score from:

- distance from the current dispersion
- a small `J2`-sensitivity modulation

This gives the TAS-aware scheduler a notion of "scientific value" without dragging in a full inference loop.

That is another recurring pattern in these examples:

- use the lightest scoring surrogate that still exposes the intended tradeoff

#### The hidden star of the example: `simulate_order(...)`

This function is where the example becomes concrete.

For a proposed order of candidates, it:

- asks the motor model for move time to each point
- advances the motor model state with `move_to(...)`
- records move time, count time, and cumulative wall clock

This is the crucial point:

- the policies are not compared in abstract score space
- they are compared in elapsed experiment time

That is exactly the kind of realism that makes TAS-AI different from purely information-centric active learning demos.

#### The three strategies

The file defines three orderings:

- `random_strategy(...)`
- `nearest_strategy(...)`
- `tas_strategy(...)`

Each teaches something different.

`random_strategy(...)` is the baseline of indifference. It says:

- ignore both information structure and motion cost

`nearest_strategy(...)` is a common intuitive heuristic. It says:

- just minimize travel distance locally

This is important because it tests a plausible human rule of thumb, not only a straw man.

`tas_strategy(...)` is the real TAS-AI idea in miniature. It:

- computes heuristic information scores for the remaining candidates
- forms a candidate array in `(H, H, 0, E)` space
- uses `MotionAwareAcquisition.score_batch(...)`
- picks the best candidate by information rate
- updates the motor state and repeats

This is the cleanest short demonstration in the repo of the "science per time" objective.

#### Why separate motor instances are used

`build_orders(...)` creates a fresh `SimplifiedMotorModel()` for each strategy.

That is a small implementation detail, but it matters. Move time depends on the current position, so each strategy must start from the same initial state. Otherwise the comparison would be contaminated by previous strategy runs.

This is a good example of careful experimental coding practice in the repo.

#### The plotting section

`make_figure(...)` produces four panels:

- TAS-AI measurement sequence in `(H, E)`
- total move time by strategy
- stacked science versus motion time
- cumulative experiment duration

This is a compact but very effective design because it answers four different questions:

- what path did TAS-AI choose?
- did it actually reduce motor time?
- did that save total time, not just move time?
- how does the wall-clock advantage accumulate over the run?

#### What this example really proves

This file is not proving that TAS-AI always discovers better points. It is proving something more operational:

- for a fixed scientifically plausible set of measurements, motion-aware sequencing can materially reduce total experiment duration

That is a narrower claim, but it is also a much cleaner one.

#### What this example does not cover

It does not cover:

- online candidate generation
- posterior updating
- model comparison
- noisy decision-making under uncertainty
- the paper's hybrid agnostic-to-informed handoff

So this example should be read as the motion/scheduling slice of the architecture, not the whole loop.

## 6. The paper repo is an application layer, not just documentation

The top-level `simulations/` directory is not a thin wrapper around the library. It contains bespoke logic for the paper's figures and narratives.

The most important file by far is `simulations/toy_closed_loop.py`.

### 6.1 Why `toy_closed_loop.py` is so large

This file is large because it is doing all of the following in one place:

- bootstrapping local import paths
- shimming external dependencies
- defining a toy physical model
- defining a simple agnostic GP phase
- planning measurements
- simulating measurements
- fitting model parameters
- computing approximate posteriors
- checkpointing
- generating publication figures
- generating animations
- formatting LLM prompts
- parsing LLM suggestions

This is not "library code with bad factoring". It is better understood as a paper application that needs to keep many experimental branches stable and reproducible.

### 6.2 The bootstrapping section matters

The very top of `toy_closed_loop.py` adds `deps/` and `tasai/` to `sys.path` and installs a `lattice_calculator` shim if needed.

That section is easy to skip, but it explains a major architectural decision:

- the paper bundle is designed to run from this repo root without requiring a separately installed `tasai`

That improves archival reproducibility, but it also means the script has environment-management logic embedded inside it.

### 6.3 The toy closed-loop stack inside the paper script

The most important internal components are:

- `LibraryBackedLogGPSurrogate`: the paper-loop adapter around the reusable Log-GP
- `SimpleGaussianProcess`: retained as a lightweight fallback helper
- `run_loggp_phase(...)`: agnostic discovery / mapping
- `SquareLatticeDispersion`: the toy forward model used throughout the closed-loop demo
- `generate_hypotheses(...)`: candidate-model construction
- `plan_measurements(...)`: physics-aware point selection with phase logic
- `simulate_measurements(...)`: noisy synthetic data generation
- `fit_model_parameters(...)`: fitting via BUMPS/DREAM or scipy fallback
- `discriminate_models(...)`: model ranking and posterior approximation
- `save_checkpoint(...)` / `load_checkpoint(...)`: experiment state persistence
- visualization functions near the bottom of the file

This file is effectively the executable scientific argument of the paper.

### 6.4 A concrete walkthrough of `toy_closed_loop.py`

The best way to understand this file is to follow `main(...)` from the bottom upward.

At a high level, `main(...)` does this:

1. parse flags and initialize the TAS realism layer with `init_tas(...)`
2. instantiate the hidden ground-truth model with `SquareLatticeDispersion`
3. build the toy crystal structure with `create_toy_structure()`
4. generate candidate Hamiltonians with `generate_hypotheses(...)`
5. choose a control regime:
   - pure phase-planner loop
   - hybrid Log-GP handoff
   - optional LLM injection / mailbox path
6. repeatedly:
   - pick points
   - simulate them
   - fit candidates
   - update posteriors
   - checkpoint and plot

The important thing is that the file is not organized as one neat controller class. Instead, `main(...)` owns local helper closures like:

- `append_points(...)`
- `take_measurements(...)`
- `run_discrimination(...)`
- `maybe_llm_phase0_injection(...)`

That style looks unorthodox if you are expecting a library-style orchestration layer, but it works well for a paper script because the flow is explicit and nearby.

### 6.5 The early building blocks in `toy_closed_loop.py`

The top third of the file defines small, composable tools that later parts depend on.

#### `LibraryBackedLogGPSurrogate`

This is the current bridge between the paper script and the reusable library GP stack. The closed-loop demo still owns the acquisition policy details that are specific to the manuscript:

- the soft energy taper
- the empirical tau/gamma intensity clamp
- consumed-area exclusion
- motion-aware scoring during the agnostic phase

But the predictive model underneath those safeguards is now the shared `tasai.core.LogGaussianProcess`, not a paper-local GP implementation.

#### `SimpleGaussianProcess`

This class still exists in the file, but it is now a fallback helper rather than the main closed-loop agnostic model. It remains useful in degraded environments and for a few audit-side utilities that want a tiny dependency footprint.

#### `run_loggp_phase(...)`

This function is the paper's agnostic front end. Its job is:

- create an accessible `(h, E)` candidate grid
- initialize with a serpentine coarse scan or seed checkpoint
- estimate a background level and intensity threshold from empirical deciles
- fit the shared library Log-GP in log-intensity space
- score unexplored points by uncertainty and move cost
- avoid repeatedly sampling nearly identical points via consumed ellipses

This function is one of the clearest code realizations of the paper's "agnostic discovery first" claim. It is also where a lot of the JCNS-inspired heuristics live:

- soft energy taper
- background-floor estimation
- consumed-area exclusion

For the updated manuscript/docs, this combined policy is what we call **enhanced Log-GP** (§3.1 of the paper):

- the shared `tasai.core.LogGaussianProcess` backend (log-intensity space, following the Teixeira-Parente / JCNS idea),
- plus the paper-specific coarse-grid init, 1D cosine energy taper, consumed-area exclusion via resolution-sized ellipses, and linear-intensity variance weighting.

That distinction matters because other parts of the paper repo — especially the ghost and bilayer audit ablations in `simulations/run_audit_ablation.py` — use only the simpler library-backed `AgnosticExplorer(use_log_gp=True)` remapping path (what the paper calls **bare Log-GP**) rather than the full enhanced policy. SI Note S1 describes the variance-weighting implementation in detail: the design principle is that acquisition is ranked in linear-intensity variance units, either by the backend's directly exposed linear-space standard deviation or, when log-space μ and σ² are available, by the log-normal conversion `Var(I) = (e^{σ²} − 1) · e^{2μ+σ²}`.

#### `maybe_get_llm_points(...)`

This function is the bridge from the numerical loop to the advisory LLM layer. It:

- renders the prompt from recent measurements
- optionally dispatches through a local CLI or an external mailbox path
- parses JSON suggestions
- applies a consensus/decider step if multiple models are used

This function matters because it shows that the LLM is not coupled into the inference engine itself. It only proposes candidate measurements.

#### `save_checkpoint(...)` / `load_checkpoint(...)`

These functions are more important than they look. The paper workflows rely heavily on checkpoint compatibility so that:

- long runs can resume
- figure scripts can reconstruct posterior evolution
- overseer batches can be audited after the fact

In practice, the checkpoint schema is part of the paper interface.

#### `TASConfig` and `init_tas(...)`

These define the paper-local TAS realism object. `init_tas(...)` decides whether to use:

- full Cooper-Nathans resolution via `tasai.instrument.TASResolutionCalculator`
- or a simpler empirical fallback

This function is the choke point where the paper script connects to the installed tasai instrument layer.

#### `SquareLatticeDispersion`

This is the paper's local forward model for the closed-loop demonstration. It is not the generic library `PhysicsModel` interface, and that is an important design choice. The paper keeps a bespoke, explicit toy model here because:

- it needs figure-specific behavior
- it needs easy direct control of intensity and dispersion formulas
- it needs predictable fitting behavior across multiple study modes

### 6.6 Candidate generation and planning in detail

#### `generate_hypotheses(...)`

This function creates the candidate model library for the closed-loop run. In the paper, the candidates are heuristic stand-ins for what a future GNN- or chemistry-driven hypothesis generator would provide.

That means the code is doing two jobs at once:

- constructing concrete model dictionaries the planner can fit
- illustrating the paper's scientific story about structure-to-Hamiltonian proposal

This is why the repo also includes `simulations/exchange_path_analysis.py`, `tasai/extensions/goodenough_kanamori.py`, and `tasai/extensions/gnn_hypothesis.py`: they are conceptual neighbors, even though they are not called directly by this script. The reusable exchange-path engine now lives in `tasai`, while the paper script remains the plotting/demo front-end.

#### `plan_measurements(...)`

This is the single most important planning function in the paper repo.

Its responsibilities include:

- deciding which planning phase is active
- generating accessible `(h, E)` candidates
- evaluating model disagreement
- evaluating projected Fisher information
- injecting forced zone coverage or gap-focused measurements
- adjusting count times for super-dwell gap checks
- keeping spatial/energy diversity in the chosen batch

Read this function slowly. It contains the paper's real policy decisions, including several ideas that are easy to miss when reading only the manuscript:

- planning is phase-aware, not stationary
- the algorithm explicitly handles motion cost in the agnostic phase
- the algorithm has special-case behavior for gap hunting in late phases
- some policies are triggered by evidence state, not just measurement count

This function is also the clearest example of the paper script encoding a scientific workflow directly rather than calling a generic library planner.

### 6.7 Measurement, fitting, and discrimination in detail

#### `simulate_measurements(...)`

This is a paper-local simulation layer. It turns a planned point into:

- counts
- measured intensity
- Poisson plus systematic uncertainty
- metadata flags such as `coverage`, `human_hint`, `llm_hint`, `mode`, and `dwell`

That metadata is important because later figure-generation and audit code depends on it. A "measurement" in this script is not just intensity data; it is also provenance about why that point was taken.

#### `fit_model_parameters(...)`

This is the fitting dispatcher. It decides whether to use:

- `BUMPS`/`DREAM`
- scipy `L-BFGS-B`
- or a reduced fallback path

Its helper `_fit_with_bumps(...)` is especially revealing because it builds a different fitting surface depending on which parameters are free for the current candidate model. This is one of the places where the script's candidate-library logic and its fitting logic are tightly coupled.

That coupling is not ideal as a generic library pattern, but it is effective for a paper demo where the set of candidate families is known in advance.

#### `run_partial_mcmc(...)`

This is a lightweight local uncertainty-refinement layer. It is not trying to be the central inference engine. It exists to cheaply stabilize or probe uncertainty around fitted models when full heavyweight sampling is unnecessary or unavailable.

#### `discriminate_models(...)`

This is the paper script's main decision-updating function. It:

- computes weights over the existing measurements
- optionally branch-reweights the data
- fits each hypothesis with the appropriate free parameters
- computes approximate evidence-like scores
- converts those scores into posterior weights

This is the step that converts "the algorithm has some data" into "the algorithm now believes model X is leading with probability Y".

If you want to understand why later phases trigger, this is the state update to study.

### 6.8 The control flow inside `main(...)`

The lower part of the file, starting around `main(...)`, is where all of the preceding pieces are assembled into an actual experiment loop.

The most important state variables maintained there are:

- `measurements`
- `measurement_plan`
- `planned_points`
- `results`
- `posterior_hint`
- `best_model`
- `superset_model`
- `llm_state`
- `llm_batch_idx`

The main loop has several notable branches:

- resume from checkpoint
- hybrid Log-GP initialization
- optional phase-0 LLM injection
- repeated discrimination and replanning
- optional checkpoint snapshots after batches

This section is worth reading alongside the CLI argument parser because many manuscript modes are controlled by flags rather than by separate scripts.

The takeaway is that `toy_closed_loop.py` is not just "a simulation." It is the command-line application that encodes the paper's autonomous experiment variants.

### 6.9 `plan_measurements(...)` is where the paper's policy logic lives

This function is the best place to understand the paper's decision policy.

It combines:

- phase logic
- zone-coverage logic
- branch/gap emphasis
- model disagreement
- projected Fisher information
- accessibility checks from the TAS model
- dwell-time injection for gap-hunting

This is not a generic planner API. It is the paper's curated experimental policy, expressed in code.

That is an important architectural choice:

- the library exposes generic planners and abstractions
- the paper script encodes a particular scientific workflow directly

### 6.10 Inference in the paper script is tiered

`fit_model_parameters(...)` and its helpers show another pragmatic design choice.

The script supports several levels of inference:

- scipy optimization for a point estimate
- BUMPS/DREAM when available
- a lightweight partial MCMC for extra uncertainty estimation

This is a good example of the repo's general philosophy:

- use better scientific tooling when the environment supports it
- do not make the entire pipeline fail if optional inference tooling is missing

That is practical, but it creates a reproducibility caveat:

- two environments can both "run"
- yet not be using the same inference backend

If you are checking scientific claims, always verify which backend path actually executed.

### 6.11 Model comparison in the paper script

`discriminate_models(...)` is the central loop for the closed-loop demos.

At a high level it:

1. chooses which parameters are free for each hypothesis
2. fits each candidate model to the accumulated data
3. computes chi-squared / approximate information criteria
4. converts those into posterior-like weights

This is not meant to be a universal Bayesian model-comparison framework. It is a fast, paper-focused discrimination engine designed to support repeated in-loop evaluation.

## 7. The LLM/audit layer

The LLM-related code is not spread evenly through the repo. The main places are:

- `simulations/toy_closed_loop.py`
- `simulations/toy_closed_loop_llm_overseer.py`
- `simulations/run_audit_ablation.py`
- `scripts/llm_danse2_watcher.py`
- `scripts/llm_audit_mailbox_runner.py`
- `prompts/*`

The updated reproducibility docs make an important distinction here:

- the older redacted prompt files in `prompts/` belong to qualitative model-agnostic checks
- the current manuscript Section 5.3.2 bilayer audit discussion uses the inline prompt builder in `simulations/run_audit_ablation.py`

So if you are trying to understand the current paper claims, do not assume that `prompts/llm_prompt_redacted.txt` is the full or only prompt story.

### 7.1 `toy_closed_loop_llm_overseer.py`

This file should be read as an extension of `toy_closed_loop.py`, not as a separate framework.

It imports many functions from `toy_closed_loop.py` and adds:

- mailbox communication
- discrimination-menu construction
- an overseer prompt contract
- guardrails for injected audit points
- mode selection between `loggp_active` and `physics`

Architecturally, this is significant:

- the LLM is not replacing the numerical planner
- it is wrapped as an advisory/router layer above it

That separation reflects a cautious design philosophy and lines up with the paper's framing of the LLM results as pilot/guarded evidence rather than a replacement for symbolic methods.

### 7.2 A concrete walkthrough of the LLM infrastructure

There are four distinct layers in the LLM path:

1. prompt construction inside `toy_closed_loop.py`
2. overseer state construction and validation inside `toy_closed_loop_llm_overseer.py`
3. current audit-ablation prompt construction inside `simulations/run_audit_ablation.py`
4. watcher / local model execution inside `scripts/llm_danse2_watcher.py`
5. mailbox transport and operational utilities in `scripts/llm_mailbox_client.py` and `scripts/llm_audit_mailbox_runner.py`

If you do not separate those layers, the LLM infrastructure feels much more mysterious than it is.

#### Layer 1: prompt construction in `toy_closed_loop.py`

The base script contains:

- `format_measurements_for_llm(...)`
- `build_llm_prompt(...)`
- `_run_llm_command(...)`
- `_extract_json_block(...)`
- `parse_llm_suggestions(...)`
- `maybe_get_llm_points(...)`

This is the local "direct suggestion" path. It turns recent measurements into a compact tabular prompt, runs one or more LLM CLIs, and expects a small JSON object back.

This path is important because it keeps the symbolic control loop and the LLM contract very narrow:

- inputs are recent measurement summaries
- outputs are candidate points with short reasons

The LLM does not see hidden posterior weights unless a more explicit overseer path passes them.

#### Layer 2: the overseer in `toy_closed_loop_llm_overseer.py`

The overseer file adds structure and guardrails on top of the direct suggestion path.

Its key functions are:

- `_http_get(...)` / `_http_post(...)`: mailbox transport
- `build_discrimination_menu(...)`: precompute a safe menu of candidate falsification probes
- `audit_needed(...)`: decide when a strategic audit is warranted
- `_build_overseer_prompt(...)`: render the overseer state into a constrained prompt
- `_parse_decision(...)`: validate that the LLM output follows the JSON contract
- `_mailbox_overseer_decision(...)`: request a decision from the mailbox service

The big architectural decision here is that the LLM is not free to propose arbitrary raw coordinates. Instead, it operates within a constrained protocol:

- choose a mode
- choose an intent
- optionally choose IDs from a precomputed discrimination menu

That is a very important safety and interpretability choice. The numerical code still owns:

- accessibility
- menu construction
- execution ordering
- final batch composition

The LLM is acting more like a strategy selector than a direct controller.

#### Layer 3: `simulations/run_audit_ablation.py`

This file matters more after the recent manuscript updates because it now owns the current bilayer audit-ablation prompt regime used in Section 5.3.2.

Architecturally, this is different from the older redacted prompt path:

- it is not trying to be purely model-agnostic
- it is a semantically guided strategic-audit setup
- it still constrains the overseer to the shared discrimination menu rather than exposing hidden coordinates or the true model identity

That distinction is important when reading the current paper claims. The code is deliberately separating:

- older prompt-comparison artifacts in `prompts/`
- current archived ablation summaries in `paper/data/ablation_runs/`

If you are auditing the manuscript, read this file together with the archived one-seed summary JSONs rather than treating operational mailbox logs as the source of record.

#### Layer 4: `scripts/llm_danse2_watcher.py`

This script is the operational watcher that polls the mailbox and runs local models.

The main pieces are:

- `http_get(...)` / `http_post(...)`
- `run_llm_single(...)`
- `run_llms(...)`
- `run_llm_overseer(...)`
- `consensus_suggestions(...)`
- `main()`

Its role is:

1. poll the mailbox for a prompt
2. execute local `claude`, `gemini`, and/or `codex`
3. parse JSON output
4. build a consensus or decider result
5. post the response back

This file is not "model reasoning logic." It is process orchestration and glue. That distinction matters. If you want to change how the LLM is instructed, read the prompt-building code. If you want to change how CLIs are invoked or merged, read the watcher.

#### Layer 5: mailbox utilities

The operational support scripts are:

- `scripts/llm_mailbox_client.py`
- `scripts/llm_audit_mailbox_runner.py`

These are useful for:

- manually inspecting mailbox state
- posting prompts
- running controlled audit ablations without the live watcher

This code is infrastructural rather than algorithmic. It exists so the audit experiments can be run and debugged in a distributed environment.

### 7.3 The end-to-end LLM call flow

The cleanest mental model is this:

1. `toy_closed_loop.py` or `toy_closed_loop_llm_overseer.py` constructs experiment state
2. that state is rendered into a compact prompt
3. for current audit ablations, `run_audit_ablation.py` may build a semantically guided prompt around the shared action space
4. the prompt is either sent directly to a local CLI or posted to a mailbox
5. `llm_danse2_watcher.py` polls the mailbox and runs one or more local LLMs
6. the JSON response comes back
7. the numerical layer validates it
8. only then are suggested points or injected menu IDs turned into actual measurements

The important consequence is that the LLM layer is not trusted with direct actuation. It must pass through:

- structured parsing
- menu restrictions
- accessibility checks
- batch-filling logic

That is why the paper can honestly describe the LLM path as a guarded audit layer rather than an unconstrained autonomous controller.

### 7.4 Watcher scripts

The watcher scripts are process glue:

- poll a mailbox
- run one or more local LLM CLIs
- parse or merge JSON suggestions
- post results back

These are not polished library abstractions. They are operational tooling for the experimental workflow.

That is a good thing to understand early, because otherwise it is easy to over-read them as part of the core architecture.

## 8. Other paper scripts

### 8.1 `paper/scripts/make_figure4_scenarios.py`

This is the cleanest example of paper code importing the library cleanly.

It imports `tasai.examples.benchmark_jcns` and uses the benchmark scenario definitions directly. This is a nice pattern because it reuses scientific definitions without copying them into the paper layer.

### 8.2 `simulations/exchange_path_analysis.py`

This script is now best understood as a paper-facing wrapper around logic that also lives in the `tasai` library. The reusable orbital-aware exchange-path engine is in `tasai/extensions/goodenough_kanamori.py`; this script keeps the paper-specific visualization, ranking display, and figure layout.

The current version is more than a bond-angle cartoon. It now:

- enumerates exchange paths with periodic minimum-image geometry
- attaches simple orbital-filling states (`eg` / `t2g`) to the magnetic ions
- uses a lookup-table implementation of the orbital-dependent GK sign rules
- exposes the resulting channel labels directly in the ranked-path output and figure

It matters scientifically because it shows how the candidate-model library might be seeded, but it is not central to the library's core runtime architecture. The architecture point is that the chemistry/structure analysis now belongs in the library, while the manuscript figure generation stays here.

### 8.3 `simulations/hybrid_exploration_demo.py`

This script is also mostly self-contained. It is better thought of as a pedagogical figure generator than as a core reusable engine.

It now uses a library-backed Log-GP surrogate for the agnostic phase, but it is still deliberately structured as a didactic workflow example rather than a benchmark harness. The seed set is a small prior-based scaffold, not a blind random start and not an oracle truth path. That choice makes the hybrid handoff legible and stable in a single figure without turning the script into a fair comparative benchmark.

Its role is explanatory:

- show agnostic Log-GP exploration under a plausible prior
- show transition to informed refinement
- show how a weak prior plus agnostic mapping can bootstrap a usable initial model
- produce the visual argument for the hybrid workflow

The right way to read this script is:

- as a structured demonstration of hybrid control logic
- not as evidence that this exact seed/acquisition recipe is benchmark-optimal

### 8.4 `paper/scripts/plot_llm_inloop_snapshot.py`

This script matters because the updated reproducibility guide now treats it as the current Figure 10 renderer. It takes:

- a closed-loop checkpoint
- a model-table CSV
- plotting bounds and labeling options

and writes the compact final overseer closed-loop snapshot used in the manuscript.

This is a subtle but important provenance change. Earlier walkthroughs might have led you to expect the paper's late closed-loop figures to come mainly from the older side-by-side comparison scripts. The current manuscript Figure 10 instead routes through the refreshed library-backed overseer checkpoint `paper/data/overseer_loggpfix_20260327_checkpoint.json` and this plotting script.

## 9. Architectural decisions that shape the whole repo

### 9.1 Separate the library from the paper application

This is the biggest structural decision and it is the right one.

Why it helps:

- the paper bundle can be frozen
- figure scripts can be reproducible
- the library can still be thought of as a reusable package

What it costs:

- duplicated concepts
- some drift between "generic" and "paper" implementations
- more than one place to look for planning logic

### 9.2 Keep core interfaces abstract, but examples concrete

The library defines good abstractions:

- `PhysicsModel`
- `MeasurementPoint`
- `InstrumentInterface`
- `AcquisitionFunction`

But the examples often instantiate concrete workflows directly rather than building a giant orchestration framework on top.

This is a pragmatic design:

- examples stay readable
- extension by copy/adaptation is easy
- the code is approachable for scientists

But it also means:

- some concepts are repeated
- the "one true runner" does not exist

### 9.3 Treat realism as optional but important

Resolution, motion, and external spin-wave backends are optional in many paths.

This is sensible because scientific Python environments are messy. The code often prefers:

- degraded but runnable behavior

over

- strict failure

That is good for usability, but it is one of the biggest sources of hidden behavior differences.

### 9.4 Use multiple fidelity levels of inference

Across the repo you will see:

- grid search
- local optimization
- BUMPS/DREAM
- simple Metropolis
- posterior approximations via AIC/BIC-like surrogates

This is not inconsistency so much as a hierarchy of cost/accuracy tradeoffs. The code often chooses the lightest tool that preserves the behavior needed for the given demo or figure.

## 10. How to use the library in practice

If you want to use the reusable `tasai` code for a new autonomous workflow, do not start from `toy_closed_loop.py`.

Start from the library examples and build outward.

Recommended path:

1. pick the closest example:
   - parameter refinement: `example_parameter_determination.py`
   - model discrimination: `example_model_discrimination.py`
   - motion-aware ordering: `example_with_motor_motion.py`
2. identify which forward model you need
3. decide whether you are using:
   - a simulator
   - a proxy to a real instrument
4. decide whether you need:
   - full MCMC
   - a fast point-estimate plus evidence proxy
5. only then add forecasting, GP exploration, or MCTS

In other words:

- do not begin with the most sophisticated planner
- begin with the simplest end-to-end example that matches your scientific goal

## 11. Common pitfalls

These are the main things that will confuse a new reader or user.

### 11.1 The paper repo and the library are not the same thing

The paper repo depends on an installed tagged copy of the library, but the paper's flagship script is not just a thin client of that library.

### 11.2 Optional dependencies change behavior

Examples:

- `rescalculator` present vs absent
- `bumps` present vs absent
- `pyspinw` present vs absent
- sklearn GP present vs absent

The code usually falls back gracefully, which is convenient, but you need to know which path actually ran.

### 11.3 "Sunny" means two different things in practice

There is a lightweight analytic Sunny-inspired model used widely in examples, and there is also a general backend story around Sunny/Julia. They are related, but not identical in implementation maturity in this snapshot.

### 11.4 The dashboard is not the central orchestration layer

`tasai/dashboard/app.py` is useful, but it is not the main engine used by the paper closed-loop workflow. It is closer to a UI/demo surface than the canonical scientific driver.

### 11.5 Tests are focused, not comprehensive

The test coverage in this snapshot is not a full end-to-end reproducibility test suite. The most substantial tests are around spin-wave behavior and backend comparisons.

That means:

- passing tests is useful
- but it does not validate the entire paper workflow

### 11.6 The paper script is intentionally monolithic

A reader coming from a software-engineering mindset may want to refactor `toy_closed_loop.py` immediately. Resist that urge until you understand which pieces are there for scientific comparability, checkpoint compatibility, or figure generation.

## 12. Where the scientific story shows up directly in code

The paper argues for:

- agnostic discovery first
- physics-informed refinement second
- motion-aware planning
- guarded audit intervention for failure modes

You can map those claims directly to code:

- agnostic discovery: `tasai/core/gaussian_process.py`, `run_loggp_phase(...)`
- physics-informed refinement: `PhysicsModel`, fitting code, discrimination loops
- motion-aware planning: `instrument/motors.py`, `example_with_motor_motion.py`, acquisition denominators
- guarded audit layer: `toy_closed_loop_llm_overseer.py`, watcher scripts, prompts

That mapping is useful because it lets you separate:

- code that supports the central claims

from

- code that is mainly convenience, plotting, or infrastructure

## 13. What to read if you want specific answers

If your question is "How are candidate measurements scored?"

- read `tasai/core/acquisition.py`
- then `simulations/toy_closed_loop.py` around `plan_measurements(...)`

If your question is "How are posteriors updated?"

- read `tasai/inference/mcmc.py`
- then `tasai/core/forecast.py`
- then the fitting/discrimination helpers in `simulations/toy_closed_loop.py`

If your question is "Where does instrument realism enter?"

- read `tasai/instrument/base.py`
- `tasai/instrument/simulator.py`
- `tasai/instrument/motors.py`
- `tasai/instrument/resolution.py`

If your question is "Where does the paper's closed-loop policy actually live?"

- read `simulations/toy_closed_loop.py`
- then `simulations/toy_closed_loop_llm_overseer.py`

## 14. Final orientation

The cleanest way to understand this codebase is to hold two pictures in your head at once.

Picture one:

- `tasai` is a modular autonomous-science library with clear abstractions for models, instruments, inference, and acquisition.

Picture two:

- `tasai_paper_clean` is a reproducibility application that packages one pinned library snapshot together with manuscript-oriented orchestration and figure scripts.

Once you keep those pictures separate, most of the repo starts to make sense:

- read examples to understand the library
- read `toy_closed_loop.py` to understand the paper's actual closed-loop narrative
- read `REPRODUCIBILITY.md` to understand provenance
- read `MODULE_MAP.md` to keep the repo boundaries straight

If you want to build on the library, start from the examples.

If you want to audit the paper's claims, start from the paper scripts and then trace downward into the installed library only when a script crosses that boundary.

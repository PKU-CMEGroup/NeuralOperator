# Mechanistic Diagnostic Plan

Date: 2026-07-04
Status: Historical CPG/PCNO diagnostic plan

This document preserves the original reproduction-diagnostic logic. For current
Idea 2.1 method design, experiment ordering, and stop rules, follow
`RESEARCH_DIRECTION_DECISION.md` and `HANDOFF.md`. Unfinished B9/B10 work still
blocks claims about the CPG/PCNO scale-propagation mechanism and any recurrent
control motivated by that mechanism; it does not block the separately scoped
conservative-coordinate diagnostic or reference-flux export/closure audit.

## Problem

The current CPGNet reproduction learns accurate teacher-forced one-step updates on the supersonic bump dataset, but its autoregressive rollout is far worse than the paper's Table 2 values. We need to diagnose what the model learns, where rollout fails, and which failure modes are likely shared by PCNO/MPCNO/FNO-style neural operators.

## Current Evidence

Batch-size-2 CPGNet results from `artifacts/time_dependent_no/official_bs2_20260702_160451_results/`:

| Metric | Paper Table 2 | One-step rollout | Two-stage rollout | One-step teacher-forced | Two-stage teacher-forced |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rho` | 0.015 | 0.200242 | 0.158637 | 0.007548 | 0.007423 |
| `v1` | 0.007 | 0.094521 | 0.067247 | 0.004169 | 0.003972 |
| `v2` | 0.009 | 0.083955 | 0.086872 | 0.003083 | 0.003070 |
| `pres` | 0.017 | 0.384017 | 0.293593 | 0.014596 | 0.014151 |

Interpretation to test, not assume: the model has learned the local one-step map on the ground-truth state manifold, but rollout leaves that manifold and amplifies small errors. Two-stage training improves some rollout variables but does not close the stability gap.

## Current Mechanistic Readout

The completed diagnostics sharpen the initial interpretation. CPGNet does learn accurate teacher-forced one-step primitive updates, and the reproduction failure is therefore not explained by simple one-step underfitting. However, the strongest evidence does not support a simple global out-of-distribution drift or small random-noise amplification story either:

- Teacher-forced error remains close to the paper scale, while autoregressive rollout error is roughly one to two orders of magnitude larger.
- Sampled train-range drift is weak; final predictions almost never leave sampled train min/max ranges.
- Small perturbation probes do not reproduce the full autoregressive gap.
- Shock-local pressure/rho errors dominate smooth-region errors, but local best-shift alignment explains only a small fraction of the shock-front error.

The working hypothesis is now: CPGNet fails mainly by losing shock-local phase, shape, amplitude, or multi-scale structure under autoregressive rollout. Two-stage training partially reduces the defect in some variables and trajectories, but does not robustly stabilize shock geometry.

## Literature-Informed Diagnostic Principles

Recent failure-mode papers on neural PDE emulators suggest several rules for interpreting this result:

1. **One-step accuracy is not rollout stability.** APEBench and Recurrent Neural Operators both emphasize the mismatch between teacher-forced training and autoregressive inference. Diagnostics must always report teacher-forced and free-rollout curves separately.
2. **Temporal metrics should be curves or valid-time metrics, not only final averages.** APEBench uses rollout error curves and geometric aggregation; PDE-Refiner uses high-correlation time. CPGNet/PCNO reports should include per-time RMSE, relative L2, valid prediction time, and correlation-time style metrics.
3. **Low-amplitude or high-frequency residuals can control long rollout behavior.** PDE-Refiner and the spectral FNO analysis both show that MSE-trained models can prioritize dominant energy components while neglecting non-dominant components that later couple into visible dynamics. For Euler shocks, this motivates scale-local residual diagnostics near discontinuities.
4. **Unrolling is a training intervention with tradeoffs.** APEBench and RNO show that recurrent/unrolled training can improve temporal generalization, but it may sacrifice short-term accuracy and can interact strongly with architecture. Two-stage CPGNet should therefore be judged by variable-wise and trajectory-wise stability, not by a single aggregate.
5. **Dynamics difficulty and information propagation matter.** APEBench's CFL/difficulty framing suggests measuring how far physical structures move per step relative to graph spacing and model propagation radius. For CPGNet this means comparing shock-front displacement to median edge length and message-passing depth; for PCNO it means auditing spectral/point-cloud resolution and represented modes.
6. **Metrics can hide the failure mechanism.** Smooth-region RMSE, boundary-clamped errors, or aggregate rollout averages can make a model look more stable than it is. Shock-region splits, front overlays, and scale/spectral residuals should be treated as primary diagnostics for this dataset.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
| --- | --- | --- | --- |
| C1: The CPGNet failure is primarily autoregressive rollout degradation, not one-step underfitting. | Directs future work toward stabilization, multistep objectives, DA, or state correction instead of simply increasing capacity. | Teacher-forced error stays near paper scale while rollout error is much larger; state-drift and perturbation probes rule out overly simple explanations. | B1, B2, B3 |
| C2: The hard parts of long-time Euler rollout are localized in shocks, contact/wake regions, multi-scale residuals, and invariant drift rather than uniform smooth-field error. | Identifies what any neural operator must handle, including PCNO. | Error, shock-front position, shock centroid, thickness/strength, conservation drift, valid prediction time, and scale/spectral residuals are decomposed by time, region, and trajectory. | B2, B4, B5, B9 |
| C3: Structure preservation helps some failure modes but does not guarantee stable rollout in the released CPGNet setup. | Gives a fair account of CPGNet pros/cons before comparing with PCNO. | CPGNet has better positivity / conservation / shock behavior than raw state predictors or PCNO in at least some metrics, but still shows compounding, phase, or geometry-sensitive errors. | B4, B6, B7 |
| C4: The dominant defect may be an architecture/training mismatch with shock-scale information propagation. | Connects CPGNet/PCNO diagnostics to broader neural-operator failure literature. | Effective-CFL/receptive-field probes and scale-local residual spectra explain which structures move beyond the stable learned update. | B9, B10, B11 |

Anti-claim to rule out: "The reproduction failed only because of a simple implementation bug or because the model cannot learn one-step dynamics." The teacher-forced results already argue against this, but the diagnostics below should make the mechanism explicit.

## Experiment Blocks

### B1: Metric and Artifact Audit

- Claim tested: C1.
- Purpose: Establish a model-agnostic diagnostic artifact format before adding more runs.
- Data/task: Existing CPGNet one-step and two-stage rollout HDF5 outputs plus test HDF5 metadata.
- Systems: CPGNet one-step bs2, CPGNet two-stage bs2, optionally prior two-stage bs1.
- Metrics: Paper-compatible rollout RMSE, teacher-forced one-step RMSE, per-time RMSE, valid prediction time, finite/positivity/boundary checks.
- Setup: Pull a selected subset of rollout HDF5 files first, then all 20 trajectories only if storage and transfer are acceptable.
- Success criterion: A reproducible local script can regenerate the current summary tables and produce a per-trajectory CSV/JSON.
- Failure interpretation: If summaries cannot be reproduced from HDF5 arrays, fix the analysis pipeline before deeper conclusions.
- Output target: Diagnostic table 1 and a model-agnostic `analysis_summary.json`.
- Priority: MUST-RUN.

### B2: Error Growth and State-Distribution Drift

- Claim tested: C1.
- Purpose: Separate "bad one-step predictor" from "good one-step predictor used off manifold."
- Data/task: Test trajectories, full 80-step rollout.
- Systems: One-step checkpoint, two-stage checkpoint.
- Metrics: Teacher-forced RMSE by time, autoregressive RMSE by time, ratio of rollout error to teacher-forced error, first-failure time at thresholds, per-variable state mean/std/range drift, Mahalanobis or z-score distance to train-state statistics.
- Setup: Compute train-state statistics from a sampled subset of train HDF5 frames; evaluate rollout states at every step.
- Success criterion: Rollout error begins low and grows as predicted states leave the training distribution or enter high-gradient regions.
- Failure interpretation: If rollout states remain in distribution but error grows, prioritize local stability/Jacobian and shock phase diagnostics.
- Output target: Error-growth figure and state-drift figure.
- Priority: MUST-RUN.

### B3: Perturbation Amplification Probe

- Claim tested: C1.
- Purpose: Quantify local stability of the learned update around true states and rolled-out states.
- Data/task: Selected representative test trajectories, frames early/mid/late.
- Systems: One-step checkpoint, two-stage checkpoint.
- Metrics: Amplification factor after 1, 5, 10, 20 steps; sensitivity by variable; recovery vs monotone growth; valid prediction time under controlled perturbations.
- Setup: Start from ground-truth frame `t`, add small normal-node perturbations with scales matching observed one-step error and training noise, clamp boundaries as in rollout, and roll forward.
- Success criterion: Two-stage checkpoint should reduce amplification if scheduled training is doing what it is supposed to do. Strong amplification would explain why accurate teacher-forced prediction still fails.
- Failure interpretation: If amplification is mild, focus on shock phase, geometry normalization, or metric/data mismatch.
- Output target: Perturbation amplification curves.
- Priority: MUST-RUN.

### B4: Region, Shock Position, and Shock-Shape Diagnostics

- Claim tested: C2.
- Purpose: Identify whether failure is dominated by shock position / phase error, shock smearing, smooth flow, wake/contact regions, or boundary-adjacent cells.
- Data/task: Test rollouts with graph edges and positions.
- Systems: One-step and two-stage CPGNet first; PCNO later using the same output interface.
- Metrics: Near-shock vs smooth relative L2, pressure-gradient shock proxy, shock-front mask IoU/F1, nearest-front or Chamfer distance in physical coordinates, shock centroid distance, local best-shift error reduction, shock thickness ratio, shock strength ratio, region-local RMSE, per-node error heatmaps.
- Setup: Use target pressure gradient on the graph as the primary shock mask; evaluate multiple quantiles such as 0.85, 0.90, 0.95.
- Success criterion: A dominant spatial failure mode is named, for example shock displacement / phase lag, shock smearing, shock-strength damping, or smooth-region drift.
- Failure interpretation: If errors are spatially diffuse, prioritize global stability and invariant drift.
- Output target: Shock diagnostics table and overlay figures.
- Priority: MUST-RUN.

#### Shock Position Subtests

These are the specific checks for the suspected main failure mode:

1. Build target and predicted shock-front masks from graph pressure-gradient scores at several quantiles.
2. Measure front overlap with IoU/F1 and measure geometric displacement with nearest-neighbor or Chamfer distance between predicted and target front nodes.
3. Track shock centroid and front-distance curves over rollout time to detect phase lag or early/late shock motion.
4. Run a local alignment control: allow a small spatial shift or nearest-front remap before computing pressure error. If error drops sharply after alignment, the dominant defect is shock position rather than amplitude.
5. Split error into target-front, predicted-front-only, target-front-only, and smooth regions. This separates displaced shocks from diffuse background error.

### B5: Physics and Structure Diagnostics

- Claim tested: C2 and C3.
- Purpose: Measure what structure preservation actually preserves in this released setup.
- Data/task: Predicted and target primitive rollouts.
- Systems: One-step CPGNet, two-stage CPGNet, later PCNO/direct baselines.
- Metrics: Equal-node and approximate-weight mass/momentum/energy drift, positivity counts, pressure/density minima, total variation proxy, boundary leakage under clamped rollout, optional entropy proxy.
- Setup: Start with equal-node weights and clearly label them. Recover approximate geometric weights from mesh files or graph geometry as a second pass.
- Success criterion: We can state which invariants CPGNet protects in practice and which still drift over autoregressive rollout.
- Failure interpretation: If equal-node conservation suggests drift but approximate weights disagree, geometry weighting becomes a required audit before making conservation claims.
- Output target: Physics diagnostic table.
- Priority: MUST-RUN.

### B6: Structure-Preservation Controls

- Claim tested: C3.
- Purpose: Attribute CPGNet behavior to its finite-volume/flux structure rather than capacity or training budget.
- Data/task: Same bump train/test split.
- Systems: CPGNet two-stage, raw direct node-state predictor at similar message-passing capacity, CPGNet with post-hoc clipping/projection control if cheap, PCNO when available.
- Metrics: Same as B1-B5, plus runtime and memory.
- Setup: Do not launch until B1-B5 identify the dominant defect. Favor one strong direct-prediction control over many weak ablations.
- Success criterion: Structure preservation improves at least one meaningful physical diagnostic or stability metric, while its failure mode is named.
- Failure interpretation: If structure gives no diagnostic advantage, future method development should not assume flux-form structure is enough.
- Output target: Main comparison or appendix ablation.
- Priority: NICE-TO-HAVE until diagnostics justify training cost.

### B7: Visualization and Qualitative Readout

- Claim tested: C1-C3.
- Purpose: Build intuition for what is learned and what fails.
- Data/task: Representative trajectories selected by best/median/worst rollout RMSE and by high shock error.
- Systems: One-step CPGNet, two-stage CPGNet, PCNO later.
- Views: Pressure and density rollout animations, target/prediction/error triptych, predicted-vs-target shock-front overlay, front-displacement heatmap, time slider, and per-variable valid-time markers.
- Setup: Reuse `scripts/time_dependent_no/visualize_official_cpg_rollout.py` as the CPGNet reference; require outputs to remain under ignored artifacts.
- Success criterion: Visualizations clearly show whether failure is phase shift, smearing, amplitude damping, spurious oscillation, or global drift.
- Failure interpretation: If visuals do not clarify the failure, add region masks and trajectory selection rather than more animations.
- Output target: Internal animation gallery and selected paper figures.
- Priority: MUST-RUN for CPGNet subset.

### B8: PCNO Replay Protocol

- Claim tested: C2 and C3 across model families.
- Purpose: Ensure CPGNet and PCNO diagnostics are comparable.
- Data/task: Same train/test split and rollout horizon.
- Systems: PCNO checkpoint(s) once training completes.
- Metrics: Exactly B1-B5 plus PCNO-specific adapter/preprocessing diagnostics.
- Setup: Convert PCNO outputs into the same HDF5 or NumPy artifact contract: `predicteds`, `targets`, `node_type`, `edges`, `pos`, optional metadata.
- Success criterion: PCNO can be compared using the same tables and figures without changing metric definitions.
- Failure interpretation: If adapter artifacts differ, fix the output contract before interpreting PCNO physics.
- Output target: Side-by-side CPGNet/PCNO diagnostic report.
- Priority: MUST-RUN after PCNO training completes.

### B9: Scale and Spectral Residual Diagnostics

- Claim tested: C2 and C4.
- Purpose: Test the PDE-Refiner / spectral-FNO hypothesis that neglected low-amplitude or high-frequency residuals drive long rollout failure.
- Data/task: Existing CPGNet rollout arrays first; PCNO artifacts later.
- Systems: CPGNet one-step and two-stage; PCNO once available.
- Metrics: Residual energy by graph-Laplacian mode or edge-length scale, residual edge-jump spectra, target-energy-normalized residual spectra, and the same quantities split by target-front, predicted-front-only, target-front-only, and smooth regions.
- Setup: Prefer graph-native spectra or edge-length/variation bins on the unstructured mesh. Use interpolation-to-grid FFT only after a preprocessing-error sanity check.
- Success criterion: Identify whether rollout failure concentrates in non-dominant scales, shock-local high-gradient residuals, or low modes contaminated by earlier shock errors.
- Failure interpretation: If scale/spectral residuals are flat or uninformative, prioritize dynamics-difficulty and recurrent-training controls.
- Output target: Scale-residual table and per-time spectral/scale-error curves.
- Priority: MUST-RUN before designing PCNO method changes.

### B10: Dynamics Difficulty and Effective Receptive-Field Audit

- Claim tested: C4.
- Purpose: Connect observed failure to whether the learned update can propagate information far enough per time step.
- Data/task: Target trajectories with positions, edges, and shock-front masks.
- Systems: CPGNet first; PCNO with analogous point-cloud/spectral resolution audit.
- Metrics: Shock-front displacement per step in physical units, displacement in median-edge-length units, graph-hop distance covered by message passing, boundary-to-front distance, and per-trajectory correlation with rollout/shock error.
- Setup: Estimate target-front motion using nearest-front matching or centroid/front-distance curves. Compare with CPGNet's message-passing depth and graph edge-length distribution.
- Success criterion: State whether hard trajectories correspond to front motion or feature propagation that stresses the model's effective receptive field.
- Failure interpretation: If failure is not correlated with motion/difficulty, return to scale-local residual and invariant diagnostics.
- Output target: Effective-CFL/receptive-field audit table.
- Priority: MUST-RUN for interpreting both CPGNet and PCNO.

### B11: Recurrent/Unrolled Control Experiments

- Claim tested: C1, C3, and C4.
- Purpose: Test whether aligning training with inference improves CPGNet/PCNO rollout stability and what it sacrifices.
- Data/task: Supersonic bump train/test split; start with a small fine-tune or cheap control, not a broad sweep.
- Systems: CPGNet two-stage or closest available checkpoint, then PCNO if its one-step/rollout split reproduces the same failure.
- Metrics: Teacher-forced one-step RMSE, rollout RMSE, valid prediction time, correlation time, shock-front metrics, scale-residual metrics, and runtime/memory.
- Setup: Try a narrow set of unroll lengths such as 2, 5, and 10 steps. Report whether short-term accuracy is traded for long-term stability, as APEBench and RNO suggest.
- Success criterion: Determine whether recurrent exposure actually stabilizes shock geometry, rather than merely smoothing or damping the solution.
- Failure interpretation: If unrolling reduces aggregate error but worsens shock structure or specific variables, do not treat it as sufficient.
- Output target: Small recurrent-control report.
- Priority: NICE until B9/B10 clarify which mechanism to target.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| M0 | Reproduce current summaries from raw arrays | B1 on selected CPGNet trajectories | Local summaries match existing JSON within tolerance | CPU, small transfer | Missing rollout arrays locally |
| M1 | Diagnose rollout compounding | B2 and B3 on selected trajectories | Name whether failure is off-manifold drift or local amplification | 1-3 GPU-hours for probe rollouts | Perturbation scale chosen poorly |
| M2 | Localize failure spatially | B4 and B7 on best/median/worst trajectories | Identify shock-position, shock-smearing, smooth, boundary, or global dominant mode | CPU/GPU eval only | Shock proxy too coarse |
| M3 | Quantify physical structure | B5 with equal-node weights, then approximate weights | State what is actually preserved | CPU, possible mesh parsing | Mesh weights unavailable or ambiguous |
| M4 | Decide training controls | B6 only if M1-M3 require attribution | Launch no more than one direct-predictor control first | 1-2 full training runs | Expensive without clear value |
| M5 | Replay on PCNO | B8 after PCNO training | Same diagnostic report can be generated | Depends on PCNO outputs | Output contract mismatch |
| M6 | Explain shock-scale mechanism | B9 and B10 on CPGNet, then PCNO | Name whether scale residuals or effective propagation explain hard trajectories | CPU plus possible eigensolver cost | Unstructured spectral analysis can be misleading |
| M7 | Test stabilization controls | B11 only after M6 | Determine whether recurrent/unrolled training fixes the named mechanism | GPU fine-tune cost | Better aggregate RMSE may hide worse shock structure |

## Completed First Actions

1. Downloaded a selected CPGNet diagnostic subset and full-rollout summaries.
2. Wrote a model-agnostic rollout diagnostic script that consumes prediction/target arrays and emits B1-B5 scalar summaries plus per-time CSVs.
3. Ran teacher-forced per-time, state-drift, perturbation amplification, shock diagnostics, equal-node physics checks, and selected shock overlays.

## Next Diagnostic Actions

1. Add correlation-time and geometric rollout aggregation to the model-agnostic diagnostic report.
2. Implement graph-native scale/spectral residual diagnostics, starting with edge-jump/edge-length bins before graph-Laplacian eigenmodes.
3. Audit effective shock-front displacement per step against median edge length and CPGNet message-passing depth.
4. Prepare PCNO rollout export in the same `predicteds`, `targets`, `pos`, `edges`, `node_type` contract.
5. Only after B9/B10, decide whether to run a small recurrent/unrolled fine-tune control.

## What We Expect To Learn

- Learned well: teacher-forced one-step primitive updates, especially under ground-truth state inputs and clamped boundary values.
- Possibly learned poorly: autoregressive stability, shock phase/strength, invariant preservation under accumulated error, and robustness to predicted-state distribution shift.
- Hard for neural operators: discontinuities, phase-sensitive shocks, nonuniform geometry, long-horizon error compounding, physical constraints that are not directly optimized, and mismatch between teacher-forced training and free rollout.
- Structure preservation pros: physical inductive bias, interpretable flux path, possible gains in positivity/conservation/shock robustness, and better sample efficiency.
- Structure preservation cons: not automatically stable, can encode wrong geometric or time-step assumptions, may hide boundary failure through clamping, can be expensive, and may amplify biased learned interface states into coherent rollout drift.

## Stop Criteria

- Do not add or retrain a method until B9/B10 clarify whether scale residuals or effective propagation are the mechanism to target.
- Do not make a conservation claim until weight choice is explicit.
- Do not compare PCNO against CPGNet unless both use the same rollout artifact contract and diagnostic definitions.
- Do not interpret Table 2 reproduction failure as model underfitting unless teacher-forced and perturbation diagnostics contradict the current evidence.
- Do not use interpolation-to-grid FFT as primary evidence on the unstructured mesh until interpolation error is shown to be negligible.
- Do not treat two-stage or recurrent training as successful unless it improves shock-local and per-variable stability, not only aggregate RMSE.

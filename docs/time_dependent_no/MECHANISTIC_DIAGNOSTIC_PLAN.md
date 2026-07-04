# Mechanistic Diagnostic Plan

Date: 2026-07-04

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

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
| --- | --- | --- | --- |
| C1: The CPGNet failure is primarily autoregressive stability / state-distribution drift, not one-step underfitting. | Directs future work toward stabilization, multistep objectives, DA, or state correction instead of simply increasing capacity. | Teacher-forced error stays near paper scale while rollout error grows quickly; perturbation sweeps show high amplification; rollout states drift outside training-state statistics before final errors become large. | B1, B2, B3 |
| C2: The hard parts of long-time Euler rollout are localized in shocks, contact/wake regions, and invariant drift rather than uniform smooth-field error. | Identifies what any neural operator must handle, including PCNO. | Error, shock-front position, shock centroid, thickness/strength, conservation drift, and valid prediction time are decomposed by time, region, and trajectory. | B2, B4, B5 |
| C3: Structure preservation helps some failure modes but does not guarantee stable rollout in the released CPGNet setup. | Gives a fair account of CPGNet pros/cons before comparing with PCNO. | CPGNet has better positivity / conservation / shock behavior than raw state predictors or PCNO in at least some metrics, but still shows compounding, phase, or geometry-sensitive errors. | B4, B6, B7 |

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

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| M0 | Reproduce current summaries from raw arrays | B1 on selected CPGNet trajectories | Local summaries match existing JSON within tolerance | CPU, small transfer | Missing rollout arrays locally |
| M1 | Diagnose rollout compounding | B2 and B3 on selected trajectories | Name whether failure is off-manifold drift or local amplification | 1-3 GPU-hours for probe rollouts | Perturbation scale chosen poorly |
| M2 | Localize failure spatially | B4 and B7 on best/median/worst trajectories | Identify shock-position, shock-smearing, smooth, boundary, or global dominant mode | CPU/GPU eval only | Shock proxy too coarse |
| M3 | Quantify physical structure | B5 with equal-node weights, then approximate weights | State what is actually preserved | CPU, possible mesh parsing | Mesh weights unavailable or ambiguous |
| M4 | Decide training controls | B6 only if M1-M3 require attribution | Launch no more than one direct-predictor control first | 1-2 full training runs | Expensive without clear value |
| M5 | Replay on PCNO | B8 after PCNO training | Same diagnostic report can be generated | Depends on PCNO outputs | Output contract mismatch |

## First Three Actions

1. Download or generate a selected CPGNet diagnostic subset: best, median, worst, and trajectory 17 if it remains visually useful.
2. Write a model-agnostic rollout diagnostic script that consumes prediction/target arrays and emits B1-B5 scalar summaries plus per-time CSVs.
3. Run perturbation amplification on one-step and two-stage CPGNet checkpoints for 3 representative trajectories before launching any new training.

## What We Expect To Learn

- Learned well: teacher-forced one-step primitive updates, especially under ground-truth state inputs and clamped boundary values.
- Possibly learned poorly: autoregressive stability, shock phase/strength, invariant preservation under accumulated error, and robustness to predicted-state distribution shift.
- Hard for neural operators: discontinuities, phase-sensitive shocks, nonuniform geometry, long-horizon error compounding, physical constraints that are not directly optimized, and mismatch between teacher-forced training and free rollout.
- Structure preservation pros: physical inductive bias, interpretable flux path, possible gains in positivity/conservation/shock robustness, and better sample efficiency.
- Structure preservation cons: not automatically stable, can encode wrong geometric or time-step assumptions, may hide boundary failure through clamping, can be expensive, and may amplify biased learned interface states into coherent rollout drift.

## Stop Criteria

- Do not add a new method until B1-B5 identify a dominant CPGNet defect.
- Do not make a conservation claim until weight choice is explicit.
- Do not compare PCNO against CPGNet unless both use the same rollout artifact contract and diagnostic definitions.
- Do not interpret Table 2 reproduction failure as model underfitting unless teacher-forced and perturbation diagnostics contradict the current evidence.

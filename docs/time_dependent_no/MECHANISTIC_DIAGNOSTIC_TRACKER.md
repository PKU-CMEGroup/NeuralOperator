# Mechanistic Diagnostic Tracker

Date: 2026-07-04

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D001 | M0 | Recompute current summaries from rollout arrays | CPGNet one-step bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL under `artifacts/time_dependent_no/cpg_mechanistic_diagnostic_20260704_full/`; reproduced user-provided AR summary (`rho=0.200242`, `pres=0.384017`). |
| D002 | M0 | Recompute current summaries from rollout arrays | CPGNet two-stage bs2 | bump test full | paper RMSE, per-time RMSE, VPT | MUST | DONE | Full 20 trajectories analyzed on AutoDL; reproduced user-provided AR summary (`rho=0.158637`, `pres=0.293593`). |
| D003 | M1 | Teacher-forced vs autoregressive error by time | CPGNet one-step and two-stage | bump test full | TF RMSE, AR RMSE, AR/TF ratio | MUST | DONE | Per-time teacher-forced evaluator run on AutoDL. TF remains small over time; AR/TF ratios are ~22-26x for one-step and ~16-28x for two-stage depending on variable. |
| D004 | M1 | State-distribution drift | CPGNet one-step and two-stage | bump test/train stats | z-score drift, range violations | MUST | DONE | Sampled train stats from 30 train trajectories at stride 10. Final predictions almost never leave sampled train min/max ranges except tiny pressure fractions; gross range-OOD drift is not the main explanation. |
| D005 | M1 | Perturbation amplification | CPGNet one-step and two-stage | selected bump test | amplification factor, excess target RMSE | MUST | DONE | Sampled AutoDL probe over 5 trajectories, 4 start frames, two perturbation scales, and 5 channel modes. One-step is mostly locally damped; two-stage is more velocity-sensitive, but induced excess target RMSE is tiny versus full AR errors. |
| D006 | M2 | Shock position and region decomposition | CPGNet one-step and two-stage | bump test full | shock-front IoU/F1, Chamfer/front distance, centroid, near-shock/smooth error, thickness, strength | MUST | DONE | Full q=0.85/0.90/0.95 shock diagnostics run on AutoDL. Shock-local error dominates smooth error, but best-shift alignment explains only a small fraction, so failure is shock-local shape/amplitude/stability plus phase, not pure displacement. |
| D007 | M2 | Rollout animations and overlays | CPGNet one-step and two-stage | selected bump test | qualitative phase/smear/drift labels | MUST | DONE | Generated 32 selected-subset pressure/shock overlay PNGs plus JSON/Markdown gallery under `artifacts/time_dependent_no/cpg_shock_overlay_gallery_20260704/`. |
| D008 | M3 | Equal-node physics diagnostics | CPGNet one-step and two-stage | bump test full | conservation drift, positivity, TV proxy | MUST | DONE | Equal-node normal-node total mismatch, positivity, and clamped-boundary leakage computed for full split. No positivity failures; boundary error is exactly zero under clamped rollout. |
| D009 | M3 | Approximate-weight physics diagnostics | CPGNet one-step and two-stage | bump test | weighted conservation drift | NICE | BLOCKED | Requires mesh/cell weights or a validated geometric approximation; current diagnostics intentionally report equal-node totals only. |
| D010 | M4 | Direct state-predictor control | Direct GNN or simplest available comparable model | bump train/test | B1-B5 metrics | NICE | BLOCKED | Wait for dominant defect |
| D011 | M5 | PCNO diagnostic replay | PCNO completed checkpoint via CPG-HDF5 adapter | selected bump test | AR/TF RMSE, shock masks, positivity, boundary leakage, animations | MUST | DONE | D019 completed for trajectories 0, 6, 11, 13, 17. This is a branch-adapter replay, not the exact original PCNO preprocessing contract because AutoDL lacks `pcno_Euler_forward_data.npz` and raw `points/elems/features.npy`. |
| D012 | M6 | Correlation-time and geometric rollout aggregation | CPGNet one-step and two-stage | bump test full | high-correlation time, geometric relative error aggregation | MUST | TODO | Add APEBench/PDE-Refiner-style temporal metrics to existing raw-array diagnostic reports. |
| D013 | M6 | Scale/spectral residual diagnostics | CPGNet first, then PCNO | bump test full/subset | edge-jump residual spectra, edge-length bins, graph-mode residuals, region-split spectra | MUST | TODO | Start with graph-native bins; do not use interpolation-to-grid FFT as primary evidence until interpolation error is audited. |
| D014 | M6 | Effective-CFL / receptive-field audit | CPGNet first, then PCNO | bump test full/subset | shock-front motion per step, median-edge-length units, message-passing/hop coverage, correlation with hard trajectories | MUST | TODO | Tests whether hard trajectories exceed stable information-propagation capacity. |
| D015 | M7 | Recurrent/unrolled stabilization control | CPGNet or PCNO after D013/D014 | bump train/test | TF error, AR error, VPT, shock metrics, scale residuals | NICE | BLOCKED | Do not launch until scale/resolution and effective-CFL diagnostics identify which mechanism to target. |
| D016 | M6 | Interface-state latent instrumentation | CPGNet one-step and two-stage bs2 | selected bump test frames | `reconstruct_prims`, one-sided trace metrics, LLF central/dissipation split, induced FV update, wave-type strata | MUST | DONE | Added branch-local NumPy helpers, synthetic tests, and `scripts/time_dependent_no/dump_cpg_interface_latents.py`; script imports the author checkout at runtime and does not vendor author code. |
| D017 | M6 | Interface-state latent selected-frame run | CPGNet one-step and two-stage bs2 | trajectories 0, 6, 11, 13, 17; frames 0, 20, 40, 58, 78 | admissibility, trace-likeness, flux/update match, dissipation localization, speed projection, sampled edge table | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`; latents are admissible but not physical one-sided traces, induced update matches model delta but not true update exactly, and dissipation is only weakly shock-localized. |
| D018 | M6 | Interface-latent mechanism probe | CPGNet one-step and two-stage bs2 | selected trajectories/frames; teacher-forced and autoregressive state sources | physical projection sensitivity, constrained inverse flux fit, TF-vs-AR latent drift | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`; physical projections do not preserve learned flux/update, constrained physical inverse fits remain poor, and AR mode greatly increases learned-update error against the target next state. |
| D019 | M5 | PCNO CPG-HDF5 rollout replay | PCNO Euler checkpoint through current branch adapter | trajectories 0, 6, 11, 13, 17 | AR/TF RMSE ratio, shock IoU/F1, positivity, velocity blow-up, GIF/overlay gallery | MUST | DONE | AutoDL selected replay completed under `artifacts/time_dependent_no/pcno_cpg_rollout_20260705_selected_v2/`; teacher-forced pressure RMSE is 0.143, AR pressure RMSE is 2.377, AR/TF pressure ratio is 16.6x overall and 22.3x final-step, with v1/v2 overflow-scale finite blow-up. |

## 2026-07-04 Diagnostic Update

Artifacts created under ignored `artifacts/time_dependent_no/`:

| Artifact | Purpose |
| --- | --- |
| `cpg_mechanistic_subset_20260704/` | Local compact subset pulled from AutoDL: trajectories 0, 6, 11, 13, 17 for one-step and two-stage, with `predicteds`, `targets`, `pos`, `edges`, and `node_type`. |
| `cpg_mechanistic_diagnostic_20260704_subset/` | First selected-subset diagnostic and key findings. |
| `cpg_mechanistic_diagnostic_20260704_full/` | Full 20-trajectory AR rollout diagnostics for one-step and two-stage CPGNet. |
| `cpg_teacher_forced_per_time_20260704/` | Per-time teacher-forced RMSE for both bs2 checkpoints. |
| `cpg_state_drift_20260704_full/` | Sampled train-stat range/z-score drift diagnostic for full rollouts. |
| `cpg_perturbation_amplification_20260704/` | Controlled perturbation amplification probe for one-step and two-stage CPGNet. |
| `cpg_shock_overlay_gallery_20260704/` | Selected-subset pressure/shock overlay gallery for qualitative phase/shape readout. |
| `cpg_cloud_diagnostics_20260704_summary.md` | Compact interpretation of the full cloud diagnostics. |


## 2026-07-04 Literature-Informed Update

The diagnostic plan now incorporates lessons from four neural-operator failure-mode papers: spectral FNO analysis, APEBench, PDE-Refiner, and Recurrent Neural Operators. The main additions are:

- Treat one-step accuracy and autoregressive stability as separate quantities in every report.
- Add correlation-time and geometric rollout aggregation so long-horizon behavior is visible even when final-step averages are noisy.
- Add graph-native scale/spectral residual diagnostics to test whether low-amplitude or high-frequency shock-local residuals drive rollout failure.
- Add an effective-CFL/receptive-field audit to compare shock-front motion against graph spacing and model propagation depth.
- Defer recurrent/unrolled control experiments until the scale/resolution and propagation diagnostics identify a concrete target.

Current blockers:

- Approximate/mesh-weighted conservation remains blocked until geometric weights are validated.
- Exact-contract PCNO replay remains pending until the original preprocessed PCNO mesh-measure artifacts are available; the CPG-HDF5 adapter replay is complete as D019.
- Scale/spectral residual diagnostics are pending graph-native implementation and should precede method changes.
- Recurrent/unrolled stabilization controls remain blocked until D013/D014 identify the mechanism to target.

## 2026-07-05 Interface-Latent Run Result

D017 completed on AutoDL for the selected trajectories/frames. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `per_wave_type_summary.csv`, `sampled_edges.csv`, and `aggregate_analysis.md`.

Main evidence: decoded interface states remain density/pressure-admissible but are not physical one-sided traces; pressure boundedness between adjacent node states is near zero and owner-closer fractions are only about 0.52. The LLF/FV path is internally consistent with the model delta, but its true-update error remains nontrivial, especially around shocks. Dissipation is enriched near shocks only weakly: top-decile dissipative edges are about 17% target shock-front edges versus an 8.6% base shock-edge fraction. Frame-motion/wave-speed diagnostics are noisy and do not support a single too-fast or too-slow explanation.

## 2026-07-05 Interface-Latent Priority Update

Idea 2.2 is now the next diagnostic priority before Idea 2.1 target-ladder training. The new script should be run first on the selected hard/representative trajectories and frames for both bs2 checkpoints. Its immediate purpose is to decide whether learned `reconstruct_prims` behaves like physical one-sided traces, space-time averaged predictors, flux coordinates, hidden dissipation, or a wrong wave-speed / shock-shape mechanism.

Local implementation status:

- Reusable NumPy metric logic is under `utility/time_dependent_no/cpg_interface_latents.py`.
- Author-checkpoint execution and compact artifact writing are under `scripts/time_dependent_no/dump_cpg_interface_latents.py`.
- Synthetic tests are under `tests/time_dependent_no/test_cpg_interface_latents.py`.
- Full D017 evidence has now been generated under `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`.

## 2026-07-05 Interface Mechanism Probe Result

D018 completed on AutoDL for the same selected trajectories/frames as D017, now in both teacher-forced and autoregressive state-source modes. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `projection_summary.csv`, `inverse_fit_summary.csv`, `report.md`, and `aggregate_analysis.md`.

Main evidence: replacing learned interface states with physical candidates or componentwise-bounded projections changes the induced model update substantially. The best physical projection (`expanded_clip`) still has update-vs-model energy relative L2 about 1.7-1.9 and learned-flux relative L2 about 7.8-7.9. Constrained inverse fits inside strict local owner-neighbor boxes almost never match learned fluxes within 10%; one-jump-expanded boxes improve the median residual but still leave relative L2 medians around 1.4-3.9 and high p90 residuals. AR-mode inputs increase learned true-update energy relative L2 from about 0.075-0.080 teacher-forced to about 0.79-0.99 on frames >0, while the internal learned update still matches the model delta at roughly 1e-6.

Interpretation: `reconstruct_prims` is not just a strangely scaled physical trace. It behaves as a functional nonphysical flux-control coordinate; projection to plausible physical states destroys the model update, and the learned flux is generally not close to a local physical LLF flux manifold. This strengthens the case for Idea 2.1 target-ladder training with explicit physical/interface/flux targets rather than supervising PCNO-FV from raw CPGNet latents.

## 2026-07-05 PCNO CPG-HDF5 Rollout Result

D019 completed on AutoDL for selected trajectories 0, 6, 11, 13, and 17 using `artifacts/time_dependent_no/pcno_ckpt/PCNO_forward_euler_exp_model.pth`. The checkpoint was loaded with the original Euler-positive-output PCNO wrapper, while graph features came from the current branch's CPG-HDF5 adapter: `x = [x, y, node_rho, rho, v1, v2, pres]`, equal normalized auxiliary node weights, least-squares edge-gradient weights, official boundary clamping, and `node_rho_policy='ones'`.

Artifacts are under ignored `artifacts/time_dependent_no/`: `pcno_cpg_rollout_20260705_selected_v2/`, `pcno_cpg_teacher_forced_20260705_selected_v2/`, `pcno_cpg_mechanistic_diagnostic_20260705_selected_v2/`, `pcno_cpg_teacher_forced_diagnostic_20260705_selected_v2/`, `pcno_cpg_animation_gallery_20260705_selected_v2/`, `pcno_cpg_shock_overlay_gallery_20260705_selected_v2/`, and `pcno_cpg_rollout_analysis_20260705_selected_v2.md`.

Main evidence: teacher-forced PCNO through this adapter is stable but not CPGNet-level (`rho=0.0542`, `v1=0.0947`, `v2=0.0297`, `pres=0.1434` aggregate RMSE). Autoregressive PCNO is unstable (`rho=1.592`, `pres=2.377`, and overflow-scale finite velocity errors), with pressure AR/TF ratios about 16.6x overall and 22.3x at the final step. Shock masks remain good in teacher-forced mode (mean q0.90 IoU about 0.79) but collapse in AR mode (mean q0.90 IoU about 0.081 and final-step mean about 0.020). Density and pressure underflow to exact zeros at many normal-node frames in AR mode, while clamped boundary leakage remains zero by construction.

Important caveat: AutoDL does not currently contain the original PCNO `pcno_Euler_forward_data.npz` or raw `points.npy` / `elems.npy` / `features.npy` preprocessing tree. Therefore D019 is a valid evaluation of the current branch's PCNO-from-CPG-HDF5 conversion path, but it is not yet a definitive statement about the collaborator checkpoint under the exact data contract used during training.

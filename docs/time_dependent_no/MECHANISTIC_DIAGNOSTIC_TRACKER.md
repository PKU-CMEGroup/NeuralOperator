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
| D011 | M5 | PCNO diagnostic replay | PCNO completed checkpoint via corrected preprocessing contract | selected bump test | AR RMSE, shock masks, positivity, boundary leakage, animations | MUST | DONE | D019 completed for trajectories 0, 6, 11, 13, 17 using the corrected HDF5-to-npy-to-reconstructed-npz path. The retired raw-HDF5 graph adapter should not be used for model conclusions. |
| D012 | M6 | Correlation-time and geometric rollout aggregation | CPGNet one-step and two-stage | bump test full | high-correlation time, geometric relative error aggregation | MUST | TODO | Add APEBench/PDE-Refiner-style temporal metrics to existing raw-array diagnostic reports. |
| D013 | M6 | Scale/spectral residual diagnostics | CPGNet first, then PCNO | bump test full/subset | edge-jump residual spectra, edge-length bins, graph-mode residuals, region-split spectra | MUST | TODO | Start with graph-native bins; do not use interpolation-to-grid FFT as primary evidence until interpolation error is audited. |
| D014 | M6 | Effective-CFL / receptive-field audit | CPGNet first, then PCNO | bump test full/subset | shock-front motion per step, median-edge-length units, message-passing/hop coverage, correlation with hard trajectories | MUST | TODO | Tests whether hard trajectories exceed stable information-propagation capacity. |
| D015 | M7 | Recurrent/unrolled stabilization control | CPGNet or PCNO after D013/D014 | bump train/test | TF error, AR error, VPT, shock metrics, scale residuals | NICE | BLOCKED | Do not launch until scale/resolution and effective-CFL diagnostics identify which mechanism to target. |
| D016 | M6 | Interface-state latent instrumentation | CPGNet one-step and two-stage bs2 | selected bump test frames | `reconstruct_prims`, one-sided trace metrics, LLF central/dissipation split, induced FV update, wave-type strata | MUST | DONE | Historical probe completed. The one-off implementation was retired from the active tree after the result was recorded; recover from git history only if exact reproduction is needed. |
| D017 | M6 | Interface-state latent selected-frame run | CPGNet one-step and two-stage bs2 | trajectories 0, 6, 11, 13, 17; frames 0, 20, 40, 58, 78 | admissibility, trace-likeness, flux/update match, dissipation localization, speed projection, sampled edge table | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`; latents are admissible but not physical one-sided traces, induced update matches model delta but not true update exactly, and dissipation is only weakly shock-localized. |
| D018 | M6 | Interface-latent mechanism probe | CPGNet one-step and two-stage bs2 | selected trajectories/frames; teacher-forced and autoregressive state sources | physical projection sensitivity, constrained inverse flux fit, TF-vs-AR latent drift | MUST | DONE | AutoDL run completed under `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`; physical projections do not preserve learned flux/update, constrained physical inverse fits remain poor, and AR mode greatly increases learned-update error against the target next state. |
| D019 | M5 | PCNO corrected preprocessed rollout replay | PCNO Euler checkpoint through collaborator-compatible preprocessing | trajectories 0, 6, 11, 13, 17 | AR RMSE, positivity, velocity blow-up, GIF gallery | MUST | DONE | AutoDL selected replay completed under ignored corrected PCNO rollout artifacts. Visual readout: PCNO initially tracks shock position better than CPGNet, but Fourier-style ripples grow and can trigger long-rollout crash; pressure mean RMSE across selected trajectories is about 2.13 and velocity errors can overflow. |

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
- Corrected PCNO replay is complete through the collaborator-compatible preprocessing path. Future PCNO conclusions should use that path, not the retired raw-HDF5 adapter.
- Scale/spectral residual diagnostics are pending graph-native implementation and should precede method changes.
- Recurrent/unrolled stabilization controls remain blocked until D013/D014 identify the mechanism to target.

## 2026-07-05 Interface-Latent Run Result

D017 completed on AutoDL for the selected trajectories/frames. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_latent_diagnostic_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `per_wave_type_summary.csv`, `sampled_edges.csv`, and `aggregate_analysis.md`.

Main evidence: decoded interface states remain density/pressure-admissible but are not physical one-sided traces; pressure boundedness between adjacent node states is near zero and owner-closer fractions are only about 0.52. The LLF/FV path is internally consistent with the model delta, but its true-update error remains nontrivial, especially around shocks. Dissipation is enriched near shocks only weakly: top-decile dissipative edges are about 17% target shock-front edges versus an 8.6% base shock-edge fraction. Frame-motion/wave-speed diagnostics are noisy and do not support a single too-fast or too-slow explanation.

## 2026-07-05 Interface-Latent Priority Update

Idea 2.2 is now the next diagnostic priority before Idea 2.1 target-ladder training. The new script should be run first on the selected hard/representative trajectories and frames for both bs2 checkpoints. Its immediate purpose is to decide whether learned `reconstruct_prims` behaves like physical one-sided traces, space-time averaged predictors, flux coordinates, hidden dissipation, or a wrong wave-speed / shock-shape mechanism.

Local implementation status:

- The exact one-off implementation was retired from the active tree during cleanup after the D017/D018 results were recorded.
- Full D017 evidence has been generated under ignored interface-latent diagnostic artifacts.

## 2026-07-05 Interface Mechanism Probe Result

D018 completed on AutoDL for the same selected trajectories/frames as D017, now in both teacher-forced and autoregressive state-source modes. Compact outputs are under ignored `artifacts/time_dependent_no/cpg_interface_mechanism_probe_20260705_full/`, including `summary.json`, `per_frame_summary.csv`, `projection_summary.csv`, `inverse_fit_summary.csv`, `report.md`, and `aggregate_analysis.md`.

Main evidence: replacing learned interface states with physical candidates or componentwise-bounded projections changes the induced model update substantially. The best physical projection (`expanded_clip`) still has update-vs-model energy relative L2 about 1.7-1.9 and learned-flux relative L2 about 7.8-7.9. Constrained inverse fits inside strict local owner-neighbor boxes almost never match learned fluxes within 10%; one-jump-expanded boxes improve the median residual but still leave relative L2 medians around 1.4-3.9 and high p90 residuals. AR-mode inputs increase learned true-update energy relative L2 from about 0.075-0.080 teacher-forced to about 0.79-0.99 on frames >0, while the internal learned update still matches the model delta at roughly 1e-6.

Interpretation: `reconstruct_prims` is not just a strangely scaled physical trace. It behaves as a functional nonphysical flux-control coordinate; projection to plausible physical states destroys the model update, and the learned flux is generally not close to a local physical LLF flux manifold. This strengthens the case for Idea 2.1 target-ladder training with explicit physical/interface/flux targets rather than supervising PCNO-FV from raw CPGNet latents.

## 2026-07-05 Corrected PCNO Preprocessed Rollout Result

After collaborator feedback, D019 uses the corrected PCNO replay path: HDF5-to-npy conversion and reconstruction before loading the Euler PCNO `.npz` contract. The active script is `scripts/time_dependent_no/rollout_pcno_preprocessed.py`; the raw-HDF5 graph adapter was retired from the active tree.

Corrected selected-rollout artifacts are under ignored `artifacts/time_dependent_no/`, including `pcno_corrected_rollout_20260705_selected/` and `pcno_corrected_animation_gallery_20260706_selected/`.

Main evidence: PCNO initially places the shock front more accurately than CPGNet, but nonphysical ripple artifacts grow over time and can trigger rollout crash. In the selected corrected replay, pressure mean RMSE is about 2.13 and final pressure RMSE about 3.48; velocity errors can reach overflow scale. The next useful diagnostics are direct ripple-energy and shock-front-position metrics, not time-lag curves.
## 2026-07-06 Active-Tree Cleanup

The tracked `time_dependent_no` surface was reduced to reusable utilities, active rollout/visualization diagnostics, corrected PCNO replay, and compact documentation. One-off inspection, smoke, state-drift, perturbation, time-alignment, raw-HDF5 PCNO adapter, and CPGNet interface-latent probe implementations were removed from the active tree after their conclusions were recorded here. Use git history for exact reproduction; do not treat retired paths as current entry points.
## 2026-07-09 Idea 2.1 Priority Update

The active weekly objective is now Idea 2.1: solver-facing target diagnostics. Use 1D Euler as the fast pilot to compare target parameterizations before transferring only the useful stabilized variants to CPGNet-style and 2D bump runs.

The current target ladder should remain compact: choose one nonzero training-noise level from the active follow-up batch, then run the FNO stride-4 selector over `limited_residual`, `limited_flux`, and `positive_limited_interface` with noise `0` and the selected nonzero noise. Seed-confirm only the best one or two variants.

This target-ladder work also supports later Idea 2.2. By making models predict residuals, fluxes, or interface states through explicit adapters, the learned quantities become inspectable as physical traces, flux corrections, dissipation controls, or nonphysical update coordinates.

## 2026-07-09 1D Euler Noise Follow-up Result

The `euler1d_noise_followups_v2` batch completed on AutoDL. Compact local analysis was written under ignored `artifacts/time_dependent_no/euler1d_noise_followups_v2_analysis_20260709/`.

Main evidence: for `FNO + limited_residual + stride 4`, three-seed final rollout L2 was `0.0861 +/- 0.0329` with no noise, `0.0754 +/- 0.0173` with noise `0.003`, and `0.0755 +/- 0.0055` with noise `0.02`. Noise `0.003` gave the best mean rollout/conservation without pressure-floor hugging. Noise `0.02` gave much better shock MAE (`0.0058 +/- 0.0015`) but had worse one-step error and minimum pressure at the floor.

Limiter control: raw conservative residual did not crash in the small stride-4 control and beat limited residual at noise `0` for one seed, but at noise `0.02` limited residual had better final rollout and shock metrics. Interpret this as evidence that the current limiter may be conservative/diffusive, not that admissibility control is unnecessary.

Stride-2 check: zero-noise stride 2 had nonpositive raw-pressure counts and final L2 `0.1318`; noise `0.02` improved final L2 to `0.0962`, conservation to `0.0286`, and shock MAE to `0.0061`. Smaller stride does not automatically stabilize rollout because it increases the number of learned operator applications.

Decision: use noise `0.003` as the default nonzero setting for the first stabilized target selector. Keep noise `0.02` as a shock-stability stress setting, especially when a target family is shock-unstable after the `0`/`0.003` selector.

## 2026-07-09 1D Euler Stabilized Target Selector Launch

Implemented the shared `ConservativeUpdateLimiter`, `limited_flux`, and `positive_limited_interface` target adapters. `limited_flux` and `positive_limited_interface` use samplewise limiting so a single limiter coefficient scales the whole finite-volume update for each trajectory sample, preserving interior face-pair conservation accounting. Existing `limited_residual` keeps cellwise limiting for backward compatibility with the prior residual experiments.

Local verification before launch: `uvx ruff check` passed for the touched Python files, `tests/time_dependent_no` passed, and tiny FNO CPU smoke runs completed for both new targets. Remote CPU smoke runs on the real 1D Euler dataset also completed for `limited_flux` and `positive_limited_interface`.

The compact AutoDL selector `euler1d_target_selector_v1` was launched on 2026-07-09 at 23:08 CST. Matrix: FNO, stride 4, 40 epochs, 384 train cases, 64 test cases, targets `limited_residual`, `limited_flux`, and `positive_limited_interface`, with training noise `0` and `0.003`. The first run (`limited_residual`, noise `0`) reached epoch 2/40 with finite training and test metrics. Output directories use the relative pattern `artifacts/time_dependent_no/target_selector_v1_fno_*`; the latest log path is recorded in `artifacts/time_dependent_no/euler1d_target_selector_v1_latest_log.txt`.
## 2026-07-10 1D Euler Target Selector V1 Result

The compact FNO stride-4 stabilized-target selector completed all six runs. Ignored local analysis artifacts are under `artifacts/time_dependent_no/target_selector_v1_analysis_20260710/`, with downloaded lightweight metrics under `artifacts/time_dependent_no/target_selector_v1_metrics_20260710/`.

Main evidence: one-step relative L2 was similar across targets (`0.0059` to `0.0081`), but rollout separated sharply. `limited_residual` with noise `0.003` had the best final rollout L2 (`0.0933`), shock MAE (`0.0175`), and conservation error (`0.0457`), improving over zero-noise `limited_residual` (`0.1231`, `0.0305`, `0.0663`). However, it touched the pressure floor in a few rollout states, so it remains a cautionary baseline rather than a clean structure-preserving solution.

Current flux/interface target forms failed the selector. `limited_flux` final rollout L2 was `0.6138` at noise `0` and `0.4182` at noise `0.003`; `positive_limited_interface` was `1.3574` at noise `0` and `0.3594` at noise `0.003`. Both families had large shock MAE, pressure-floor hugging, nonpositive raw pressure after conservative decoding, and high limiter activation during rollout, despite stable one-step metrics.

Interpretation: conservation-form decoding alone is not enough. The current absolute flux and absolute interface-state parameterizations are structurally conservative but dynamically wrong under autoregressive distribution shift. The limiter prevents immediate crash but becomes an emergency clamp. Next target work should prioritize physical base flux plus bounded correction, direct macro-step face-flux supervision from the generator, and bounded interface corrections around local/Riemann base states.
## 2026-07-10 Physical Flux-Correction Target Launch

Implemented `physical_flux_correction` for the 1D Euler target ladder: the network predicts a bounded correction around the current-state Rusanov face flux, then the finite-volume update and shared samplewise conservative admissibility limiter decode the next state. The target exposes correction-scale diagnostics: mean/max absolute correction over bound and saturation fraction.

Local verification: focused solver-target tests passed, `tests/time_dependent_no` passed, script help/smoke checks passed, and ruff passed with the established E402 ignore for script entry points.

Remote smoke on the real 1D dataset passed. The zero-correction/base-Rusanov audit at stride 4 was poor (`one_step_l2` about `0.198`, final rollout L2 about `0.818`, shock MAE about `0.410`, and rollout limiter activation about `0.999`), so the learned correction must do real macro-step work rather than lightly polishing a good classical step.

The first full scale-1 launch was interrupted after early live diagnostics showed test relative L2 near `0.10` and correction saturation around `0.18-0.20` with no teacher-forced limiter activation. Decision: run a short bound-scale probe over correction scales `1`, `2`, and `4` before spending the full training budget. Active ignored outputs use the relative pattern `artifacts/time_dependent_no/physical_flux_scale_probe_v1_*`, with log `artifacts/time_dependent_no/logs/euler1d_physical_flux_scale_probe_v1.log`.
## 2026-07-10 Physical Flux-Correction Scale Probe Result

The short scale probe over correction scales `1`, `2`, and `4` completed on AutoDL. Ignored local analysis artifacts are under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_analysis_20260710/`, with lightweight downloaded metrics under `artifacts/time_dependent_no/physical_flux_scale_probe_v1_metrics_20260710/`.

Main evidence: all three physical-flux-correction scale settings were rejected. Scale `1` had one-step L2 `0.0991` and final rollout L2 `1.1100`; scale `2` had one-step L2 `0.0850` and final rollout L2 `0.8259`; scale `4` had one-step L2 `0.0805` and final rollout L2 `0.8568`. All reached pressure floor, had nonpositive raw pressure counts, and had rollout limiter activation about `0.95-0.96` with minimum theta `0`.

Interpretation: increasing the correction bound improves one-step fit but does not stabilize rollout. Scale `1` is correction-bound-limited, while scale `4` largely removes correction saturation but activates the teacher-forced limiter heavily. The base audit showed the core issue: one explicit Rusanov flux over the stride-4 macro step is a poor large-timestep anchor, so the network must cancel and replace the base flux rather than learn a small physical correction.

Decision: do not run the full noise `0`/`0.003` selector for this exact target family. Next flux-target work should prioritize direct macro-step time-integrated face-flux supervision or a stable/data-derived macro flux base.

## 2026-07-13 Corrected Solver-Level CPGNet M0

The earlier `CPGNetEuler1DHead` rows are now deprecated: that generic directed target head did not execute the paper's solver-level interface-state/FV recurrence. The corrected M0 is `CPGNetEuler1D` with the exclusive `cpg_interface` target. It uses physical ghost nodes, directed geometry-only edge encoding, 12 unshared message-passing layers, target-node interface reconstruction, positive density/pressure interface decoders, one shared oriented Rusanov flux per face, and an exact finite-volume update. Left inflow is anchored to the case state; the right-wall exterior interface is the reflected owner-side prediction. Exact 1D geometry replaces the release model's learned positive geometry factor.

No post-update cell-state floor or admissibility limiter is used in the corrected recurrence. Invalid raw density/pressure terminates and is counted by rollout diagnostics. Training now supports the paper-compatible two-stage schedule: one-step standardized next-state loss followed by three-step fully differentiable autoregressive fine-tuning with reduced learning rate and additive Gaussian primitive-input noise. The existing FNO path retains its admissibility-preserving log-normal density/pressure noise. Checkpoint selection prioritizes completed admissible validation rollouts by final error; if every candidate fails the horizon, one-step validation loss selects the best-fit failure for diagnosis rather than silently freezing on epoch 1.

Local verification: all `tests/time_dependent_no` tests passed (`55 passed`), and a small synthetic CPU gate with the additive CPG noise path completed without nonpositive raw states. With a 16-hidden-channel, 2-layer smoke model trained for five one-step epochs plus one three-step autoregressive epoch, one-step relative L2 was `0.00177` and four-step final-rollout L2 was `0.00696`; final conservation error was `0.00129`. These numbers validate the implementation path only and are not benchmark evidence.

Next decision gate: run the corrected full-width M0 on AutoDL before proposing architectural improvements. Require competitive one-step fit first, then judge raw recurrence by rollout completion, positivity failures, conservation, boundary leakage, and shock metrics. Remote launch is pending a credential-safe authenticated SSH session; no dataset, checkpoint, or private machine path is tracked.

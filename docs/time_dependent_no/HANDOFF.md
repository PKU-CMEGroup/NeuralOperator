# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

This week's active objective is Idea 2.1: the solver-facing representation
diagnostic. Use 1D Euler to isolate four design axes: state coordinates,
predicted quantity, supervision graph, and enforcement mechanism. The completed
primitive-loss target ladder is evidence and baseline history; it is not a
license to bundle all four axes into one new method. Medium-horizon open-loop
forecasting remains the gate before data assimilation.

## Current Diagnostic State

CPGNet did not reproduce the paper's Supersonic Bump Table 2 autoregressive rollout errors. Completed diagnostics show teacher-forced one-step errors near the paper scale, while autoregressive errors are roughly one to two orders of magnitude larger. The most defensible explanation is shock-local phase/shape/amplitude or multi-scale structure degradation under autoregressive rollout, not simple one-step underfitting, gross train-range drift, or small random-noise amplification.

The CPGNet interface-latent investigation concluded that `reconstruct_prims` behaves like a nonphysical flux-control coordinate rather than a physical one-sided trace. That code was retired from the active tree after the result was recorded; recover it from git history only if the exact probe must be rerun.

For PCNO, use `scripts/time_dependent_no/rollout_pcno_preprocessed.py`, not the retired raw-HDF5 graph adapter. The corrected PCNO rollouts show good early shock placement but visible Fourier-style rippling, followed by instability/crash in longer autoregressive rollout.

The active 1D Euler target-ladder work is the bridge from these diagnostics to method design. The current evidence says raw conservation structure is not enough: flux and interface objectives need admissibility, rollout, shock, conservation, and noise diagnostics before they can support a structure-preserving claim. The target ladder also makes learned quantities more interpretable for later Idea 2.2 investigation.

The two Section 1.2 rows formerly labeled CPGNet were produced by a generic
directed target head with `limited_residual`; they did not execute the paper's
interface reconstruction, Rusanov flux, and finite-volume update. They are now
deprecated. The corrected M0 path is `CPGNetEuler1D + cpg_interface`: positive
directed interface states, one shared face flux, exact 1D geometry, physical
ghost boundaries, no post-update limiter, and raw failure at the first
inadmissible recurrent state. The local CPU gate passes; AutoDL validation is
pending.

The accepted direction is now recorded in
`docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md`. The current 1D and CPG
paths use primitive inputs and primitive next-state loss even when their decoder
updates conservative variables. They therefore do not answer whether a model
should operate in conservative-variable coordinates. The next causal diagnostic
must keep the conservative-residual output fixed while separating input and
loss coordinates. Direct integrated-flux and dense-supervision work follows
only after that coordinate comparison and reference-flux closure check.

## Active Code Surface

Reusable utilities are under `utility/time_dependent_no/`. The active 1D Euler target-ladder utilities are:

- `utility/time_dependent_no/fv.py`
- `utility/time_dependent_no/euler1d.py`
- `utility/time_dependent_no/euler1d_data.py`
- `utility/time_dependent_no/euler1d_models.py`
- `utility/time_dependent_no/euler1d_targets.py`

Active 1D Euler scripts are:

- `scripts/time_dependent_no/euler1d_weno_hllc_rk3_dataset.py`
- `scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py`
- `scripts/time_dependent_no/train_euler1d_target_ladder.py`
- `scripts/time_dependent_no/analyze_euler1d_target_ladder.py`
- `scripts/time_dependent_no/generate_euler1d_rollout_animations.py`

Active 2D diagnostic scripts are:

- `scripts/time_dependent_no/run_euler_fixture_diagnostics.py`
- `scripts/time_dependent_no/diagnose_cpg_rollout_mechanisms.py`
- `scripts/time_dependent_no/visualize_official_cpg_rollout.py`
- `scripts/time_dependent_no/visualize_cpg_shock_overlays.py`
- `scripts/time_dependent_no/rollout_pcno_preprocessed.py`

Generated reports, HDF5 rollout arrays, GIFs, checkpoints, and large logs remain ignored under `artifacts/time_dependent_no/`.

## Key References

- `docs/time_dependent_no/README.md`
- `docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md`
- `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_PLAN.md`
- `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`
- `docs/time_dependent_no/CPGGNSPDES_REFERENCE_AUDIT.md`

## Next Steps

1. Finish the corrected solver-level CPGNet sanity and 15+5 baseline as a
   reference validation. Require competitive one-step fit, report raw
   admissibility, and do not interpret it as target-superiority evidence.
2. Keep `limited_residual + noise 0.003` only as the current fallback baseline.
   Do not seed-repeat or transfer the rejected absolute flux/interface targets.
3. Implement the four-way coordinate diagnostic with one matched backbone and
   conservative-residual output: primitive/conservative input crossed with
   primitive/conservative loss. Add no dense losses, noise changes, limiter
   changes, or architecture changes in this comparison.
4. Export exact macro-step time-integrated face flux from the 1D generator and
   verify state-transition and boundary-flux closure before training a flux
   model.
5. Compare residual, state-loss-only flux, direct-flux, and joint flux/state
   supervision with a shared parameter-matched backbone.
6. Add dense supervision only to the best identified primary path, one family
   at a time: conservative state, primitive state, shock/front labels, then
   short unrolled loss. Derive redundant states through the solver decoder.
7. Evaluate finalists with teacher-forced one-step, direct-horizon, and at least
   50 raw autoregressive calls over a documented physical horizon, followed by
   timestep/resolution transfer and a coarse-CFD comparison.
8. Trigger local invariant-domain correction, facewise routing, front tracking,
   or INR/adaptive-time work only when a named failure mode justifies it.

## Agent Execution Rules

- Read `RESEARCH_DIRECTION_DECISION.md` before changing the trainer, target
  adapters, generator contract, or experiment matrix.
- Change one causal axis per primary comparison. A multi-component engineering
  prototype is not a substitute for the coordinate and supervision diagnostics.
- Record input/recurrent coordinates, predicted quantity, loss coordinates,
  normalization, timestep, effective CFL, geometry scaling, and all hidden
  floors or clamps in every run summary.
- Preserve one primary inference path. Do not add independent state and flux
  heads that can disagree; compute derived states through the conservative
  update.
- New auxiliary labels require a documented source, identifiability argument,
  inference role, normalization, and ablation. Do not launch a broad dense-loss
  sweep by default.

## 1D Euler Experiment Todo

This section preserves experiment history. Unchecked historical items are not
automatically approved work; the ordered `Next Steps` and accepted research
direction above take precedence.

Completed follow-up batch on AutoDL:

- [x] Analyze `euler1d_noise_followups_v2`. Use only `v2` output directories; the earlier non-v2 launch was stopped because it did not pass the intended `--seed` argument.
- [x] Check seed stability for `FNO + limited_residual + stride 4` at noise `0`, `0.003`, and `0.02` over seeds `20260707`, `20260708`, and `20260709`.
- [x] Test whether noise can replace the limiter: compare `FNO + residual + stride 4` at noise `0` and `0.02` against `limited_residual`.
- [x] Test whether denoising helps shorter-step accumulation: compare `FNO + limited_residual + stride 2` at noise `0`, `0.003`, and `0.02`.

Result summary: over three stride-4 seeds, noise `0.003` and `0.02` both improved final rollout L2 by about 12% relative to zero noise. Noise `0.003` is the safer default for the next selector because it gives the best mean rollout/conservation without pressure-floor hugging; noise `0.02` is a useful shock-stability stress setting because it sharply improves shock MAE but worsens one-step error and sits at the pressure floor. Stride 2 also needed noise: zero-noise stride 2 had raw-pressure violations, while noise `0.02` greatly improved final, conservation, and shock metrics.

Ignored compact analysis artifacts are under `artifacts/time_dependent_no/euler1d_noise_followups_v2_analysis_20260709/`.

Target-objective extensions to implement after the active batch:

- [x] Select one noise level from the active follow-up before expanding the target matrix: use `0.003` as the default nonzero-noise selector value, and keep `0.02` only as a shock-stability stress setting.
- [x] Factor a shared conservative admissibility limiter for decoded updates so residual, flux, and interface adapters can all keep updated density and pressure admissible. Flux/interface targets use samplewise scaling to preserve finite-volume face-pair accounting.
- [x] Add `limited_flux`: predict face fluxes, apply the finite-volume update, then limit the induced conservative update to keep updated density and pressure admissible.
- [x] Add `positive_limited_interface`: decode positive interface density/pressure, compute Rusanov fluxes, apply the finite-volume update, then limit the updated cell averages.
- [x] Keep training noise objective-independent: perturb only the current primitive input during training, keep the clean next state as the supervised target, and evaluate clean rollouts.
- [x] Log limiter diagnostics for all limited objectives: activation fraction, mean/min `theta`, pressure-floor cases, and whether the limiter preserved the intended conservation accounting.
- [x] Run the first extended target selector on `FNO` at stride 4 with only noise `0` and `0.003`: `limited_residual`, `limited_flux`, and `positive_limited_interface`. Result: only `limited_residual` remained viable; current absolute flux/interface targets failed under rollout.
- [ ] Seed-confirm only the best one or two stabilized target variants; do not seed-repeat losing variants.
- [ ] Run only the corrected `cpg_interface` CPGNet baseline; do not spend GPU on the deprecated CPG residual target head.
- [ ] Generate animations for the best and worst noise/limiter cases, especially trajectories where pressure reaches the limiter floor.

Flux-indeterminacy and Hodge-style follow-ups:

- [ ] Prefer direct numerical face-flux supervision if the 1D Euler generator can export the solver's face fluxes; do not treat flux reconstructed only from state differences as a unique physical target.
- [x] Add a physical-flux-plus-correction objective before more exotic gauges: `F_pred = F_Rusanov(current) + delta_F`, with regularization/diagnostics on the learned correction rather than on the whole physical flux.
- [ ] For 2D only, test curl or loop-circulation regularization on the learned flux correction after mesh face orientations, areas, and cell volumes are validated.
- [ ] Treat scalar-potential flux prediction as an ablation rather than the first mainline method, because a gradient-only gauge may over-constrain shocks and multidimensional flow.
- [ ] Do not combine Hodge regularization with the broad noise/positivity target ladder until the stabilized FNO selector identifies at least one viable target family.


Physical flux-correction status:

- [x] Implemented `physical_flux_correction` as current-state Rusanov base flux plus bounded learned correction, decoded by the FV update and shared samplewise limiter.
- [x] Added live correction diagnostics: correction/bound mean, correction/bound max, and saturation fraction.
- [x] Ran zero-correction base audit on the real 1D dataset; stride-4 base Rusanov is a poor macro-step anchor and heavily activates the rollout limiter.
- [x] Finished the short correction-scale probe over scales `1`, `2`, and `4`; all variants were rejected, so do not run the full noise selector for this exact macro-step Rusanov-base target.
Engineering controls to consider after the selector:

- [x] Replace the mislabeled CPG residual head with the solver-level paper adaptation: geometry-only edge encoder, 12 directed flow layers, positive interface decoding, shared Rusanov flux, exact 1D geometry, physical ghosts, and 15+5 training.
- [ ] Evaluate that baseline without a cell-state limiter or recurrence floor; stop and record the first raw nonpositive state.
- [ ] For FNO, keep fixed-step operators without `dt` parameterization and keep identity/zero-update initialization for residual-like heads; treat spectral smoothing or high-frequency penalties as later ablations, not part of the first target selector.

## Do Not Do

- Do not touch collaborator-owned checkpoint or preprocessing artifacts unless explicitly asked.
- Do not use the retired raw-HDF5 PCNO adapter path for model conclusions.
- Do not launch broad new training sweeps before ripple and shock-position diagnostics identify the mechanism to target.
- Do not combine coordinate, target, dense-supervision, constraint, and
  architecture changes in one primary experiment.
- Do not impose blanket componentwise TVD or global spectral/variation damping
  on Euler without a smooth/shock-region contract.
- Do not make mesh-weighted conservation claims until cell/mesh weights are validated.
- Do not treat boundary leakage as evidence of model quality under clamped official rollout.
- Do not commit private paths, credentials, raw data, checkpoints, extracted rollout arrays, heavy figures, or local machine paths.

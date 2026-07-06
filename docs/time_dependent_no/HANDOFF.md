# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

Diagnose and compare time-dependent neural-operator rollout failure modes on the supersonic bump benchmark. The current completed baseline is CPGNet run `bs2_20260702_160451`; the next comparable model family is PCNO once rollout artifacts are available.

## Current Diagnostic State

CPGNet did not reproduce the paper's Supersonic Bump Table 2 autoregressive rollout errors. The completed cloud diagnostics show teacher-forced one-step errors near the paper scale, but autoregressive errors roughly 10x-23x above the paper targets depending on variable and training stage.

The working mechanism is not simple one-step underfitting, gross train-range drift, or small random-noise amplification. The strongest evidence points to shock-local phase/shape/amplitude or multi-scale structure degradation under autoregressive rollout. Two-stage training improves some variables and trajectories, but does not robustly stabilize shock geometry.

Reusable diagnostics are implemented under `utility/time_dependent_no/`, entry points under `scripts/time_dependent_no/`, and tests under `tests/time_dependent_no/`. Generated reports and galleries remain under ignored `artifacts/time_dependent_no/`.

## Key Local References

- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_PLAN.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md`
- `artifacts/time_dependent_no/cpg_cloud_diagnostics_20260704_summary.md`
- `artifacts/time_dependent_no/cpg_mechanistic_diagnostic_20260704_full/`
- `artifacts/time_dependent_no/cpg_teacher_forced_per_time_20260704/`
- `artifacts/time_dependent_no/cpg_state_drift_20260704_full/`
- `artifacts/time_dependent_no/cpg_perturbation_amplification_20260704/`
- `artifacts/time_dependent_no/cpg_shock_overlay_gallery_20260704/`

## Literature-Informed Next Steps

1. Add correlation-time and geometric rollout aggregation to the model-agnostic diagnostic report.
2. Implement graph-native scale/spectral residual diagnostics. Start with edge-jump and edge-length bins before graph-Laplacian eigenmodes; do not use interpolation-to-grid FFT as primary evidence until interpolation error is audited.
3. Audit effective shock-front motion per step in median-edge-length units against CPGNet message-passing depth and PCNO resolution/mode choices.
4. Prepare PCNO outputs in the same diagnostic artifact contract: `predicteds`, `targets`, `pos`, `edges`, `node_type`.
5. Only after the scale/resolution and effective-CFL diagnostics, decide whether to run a small recurrent/unrolled fine-tune control.

## Do Not Do Yet

- Do not launch broad new training sweeps before D013/D014 identify the mechanism to target.
- Do not compare PCNO against CPGNet unless both use the same rollout artifact contract and node masks.
- Do not make mesh-weighted conservation claims until cell/mesh weights are validated.
- Do not treat boundary leakage as evidence of model quality under clamped official rollout.
- Do not commit private paths, credentials, raw data, checkpoints, extracted rollout arrays, heavy figures, or local machine paths.

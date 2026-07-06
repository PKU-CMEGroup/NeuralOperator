# Handoff

Use this page when transferring work between human collaborators or AI agents.

## Current Objective

Diagnose and compare time-dependent neural-operator rollout failure modes on the supersonic bump benchmark. The active comparison is now CPGNet versus the corrected-preprocessing PCNO rollout path.

## Current Diagnostic State

CPGNet did not reproduce the paper's Supersonic Bump Table 2 autoregressive rollout errors. Completed diagnostics show teacher-forced one-step errors near the paper scale, while autoregressive errors are roughly one to two orders of magnitude larger. The most defensible explanation is shock-local phase/shape/amplitude or multi-scale structure degradation under autoregressive rollout, not simple one-step underfitting, gross train-range drift, or small random-noise amplification.

The CPGNet interface-latent investigation concluded that `reconstruct_prims` behaves like a nonphysical flux-control coordinate rather than a physical one-sided trace. That code was retired from the active tree after the result was recorded; recover it from git history only if the exact probe must be rerun.

For PCNO, use `scripts/time_dependent_no/rollout_pcno_preprocessed.py`, not the retired raw-HDF5 graph adapter. The corrected PCNO rollouts show good early shock placement but visible Fourier-style rippling, followed by instability/crash in longer autoregressive rollout.

## Active Code Surface

Reusable utilities are under `utility/time_dependent_no/`. Active scripts are:

- `scripts/time_dependent_no/run_euler_fixture_diagnostics.py`
- `scripts/time_dependent_no/diagnose_cpg_rollout_mechanisms.py`
- `scripts/time_dependent_no/visualize_official_cpg_rollout.py`
- `scripts/time_dependent_no/visualize_cpg_shock_overlays.py`
- `scripts/time_dependent_no/rollout_pcno_preprocessed.py`

Generated reports, HDF5 rollout arrays, GIFs, checkpoints, and large logs remain ignored under `artifacts/time_dependent_no/`.

## Key References

- `docs/time_dependent_no/README.md`
- `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md`
- `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_PLAN.md`
- `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`
- `docs/time_dependent_no/CPGGNSPDES_REFERENCE_AUDIT.md`

## Next Steps

1. Implement direct ripple metrics for PCNO: smooth-region graph high-pass energy, local overshoot/undershoot, total-variation inflation, and pre-crash positivity/finite-value curves.
2. Implement direct shock-position metrics for CPGNet and PCNO: pressure-gradient front masks, Chamfer/F1 under tolerance, signed front displacement, and front overlays over time.
3. Compare CPGNet and PCNO using the same rollout artifact contract and node masks.
4. Use those diagnostics to decide whether the next algorithmic change should be local-kernel PCNO, shock-aware/unrolled training, or a finite-volume/flux-target ladder.

## Do Not Do

- Do not touch collaborator-owned checkpoint or preprocessing artifacts unless explicitly asked.
- Do not use the retired raw-HDF5 PCNO adapter path for model conclusions.
- Do not launch broad new training sweeps before ripple and shock-position diagnostics identify the mechanism to target.
- Do not make mesh-weighted conservation claims until cell/mesh weights are validated.
- Do not treat boundary leakage as evidence of model quality under clamped official rollout.
- Do not commit private paths, credentials, raw data, checkpoints, extracted rollout arrays, heavy figures, or local machine paths.

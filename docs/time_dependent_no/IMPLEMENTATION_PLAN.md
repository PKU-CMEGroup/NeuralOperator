# Implementation Plan: Support The Selected Predictive Experiment

Updated: 2026-09-22

[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) owns scientific selection;
[PROJECT_PLAN.md](PROJECT_PLAN.md) owns completion. The owner now authorizes
coding, focused tests and the fixed-data corrective comparison on the selected
workstation configured in ignored `LOCAL_CONTEXT.md`. This supersedes the earlier
documentation-only pause. Exact previous implementation
recipes are preserved in the
[September 16 snapshot](archive/pre_predictive_reset_20260916a/IMPLEMENTATION_PLAN.md).

## Existing Reuse Boundaries

The [README](README.md) owns maintained-code navigation. Reuse only the pieces
needed for the selected experiment, preserving their scientific semantics.

| Existing component | Useful capability | Boundary |
| --- | --- | --- |
| Periodic PCNO adapter | Complete residual transition, shared canonical output restriction, raw/restricted attribution. | Restriction is already part of Clean; it is not an empirical-manifold projection. |
| Kolmogorov reference stepper | Explicit canonical/projected state maps, restart and refinement diagnostics. | Match the actual physical/state contract; sampled qualification is not a uniform solver guarantee. |
| Clean / coverage fits and rollouts | Training-only scale, matched teacher metrics, recurrence, physical diagnostics and censoring. | Preserve role mappings, checkpoint selection and frozen identities. |
| Common-response and paired-solver assays | Shared inputs, forcing/response decomposition, trusted response and numerical sensitivity. | Finite secants do not identify manifold coordinates or certify an operator norm. |
| Paired bank and fitter | Frozen signed Gaussian targets and matched continuation/recovery/dynamics preparation. | Prepared code is not a scientific selection or launch instruction. |
| Historical supporting pipelines | ODE, NACA and Bump evidence or presentation support. | Reuse only for an explicit question; no broad rerun or relocation. |

Some entry scripts serve as libraries, and paired fitting checks both dependency
hashes and function-origin paths. Preserve that closure. Existing tests and
historical pass counts describe their recorded versions, not fresh validation
of a future changed implementation.

## Current Bounded Implementation

The September 22 owner-selected design uses one new entry point,
`fit_kolmogorov_response_preserving.py`, and a narrow recipe-loading branch in
the existing recovery evaluator. The fitter reuses fixed-data loading, the
paired bank, samplers and PCNO; it trains from bank DYN with the matched clean
objective, optionally adding the difference-of-responses penalty in the
experiment plan. Run it as a Python module with training packet, bank, frozen
teacher fit/query, arm, phase and output arguments. No reusable core API or new
model is introduced. Its synthetic test verifies both gradient paths, the
clean-only control, frozen teacher identity, matched tapes and checkpoint loading.
The current 46-test dependency check passes locally. The handoff/tracker own
the isolated resource/fit status; the first worker stops after nonrecurrent
assays, before either candidate's autonomous outcomes. No new forecast platform
or automatic protected-population reveal is part of this implementation.

The clean-continuation/online-recovery fitter and separate assay/rollout evaluator
are complete. Preserve their historical recipes and attempt identities.
The evaluation-only `replay` and `reached_response` phases in
evaluate_kolmogorov_recovery.py are implemented and completed in
`kf_reached_response_20260917a`. They reuse the loading, prediction, complete-map
restriction and pair-metric functions; historical numerical functions are
unchanged. Dense early states, native donors, shared amplitudes and compact
signed response vectors are retained. All 36 paths reproduce saved anchors
bitwise; 33 focused evaluator/fitter CPU tests passed. The dated tracker owns
the scientific interpretation and verification record. Subsequent code changes
need their own appropriate checks.

The PDE/Galerkin amplitude envelope and sampled reference calibration are
complete. They establish late exclusion, not early manifold coordinates or a
uniform time-integrator guarantee. No additional tangent calculation is needed
before the selected target and gradient contrasts. Do not implement a global
manifold estimator, tangent/normal classifier or separate forecast platform.

Implementation status and remaining constraints follow the staged comparisons.

The K=4 pair is implemented in `fit_kolmogorov_unroll.py`; its CLI has only
resource/fit phases and detached/full arms. The existing evaluator accepts
the exact new checkpoint recipe while preserving recovery-recipe validation.
Forty focused tests cover analytic temporal gradients, matching forward losses
and sequence samples, immutable inputs, failure receipts and both checkpoint
contracts. The isolated attempt is complete; saved fits, assays and rollouts
are verified and incorporated into Section 4. The handoff and dated ledger
own the result. No changes to PCNO, the recovery fitter or solver semantics were needed.

- Paired REC/DYN: `generate_kolmogorov_target_bank.py` creates the separately
  bound onset training and unused-probe banks; `fit_kolmogorov_targets.py`
  reuses the tested clean/two-branch update and parent replay. The evaluator's
  `target_assay` scores both targets and trusted-response defect without
  composing a trajectory. Forty focused CPU tests pass. The isolated
  `kf_target_pair_20260918a` comparison is complete, including assays and the
  subsequently evaluated fixed rollouts. Saved-array reductions verify the
  target distinction, numerical labels and response/bias tradeoff. Its frozen
  source owns this attempt. The old sparse-bank fitter and its
  identity checks remain unchanged; no historical packet is relabeled.
- Short unrolling: share the forward recurrence, sequence sampler and averaged
  loss between detached and full-gradient variants. Use a synthetic gradient
  check that detects missing/extra temporal paths; forward equivalence alone
  does not verify training semantics.
- Refinement/generation/response priors: first write down the selected source
  objective, conditioning, full deployment recurrence and cost. Reuse compatible
  PCNO components and existing Refiner semantics; add only the code needed for
  the selected adaptation. Do not build a general method registry in advance.

Tests should protect consequential semantics: checkpoint replay, identical probe
inputs/amplitudes, exact clean versus actual-input targets, independent RNG,
gradient paths, role separation, the forcing/odd/even identity and failure
accounting. A stochastic comparison needs shared draws and variation accounting.
Use synthetic CPU fixtures, then one bounded real-data/resource check before
dataset-scale execution. Existing test pass counts are historical, not fresh
validation of changed code.

No new maintained entry script was needed for the completed diagnostic. A later
script must have a selected scientific purpose and invocation; attempt-local
packing/launch/retrieval helpers stay ignored. No generic experiment framework,
code migration or external-review cycle is part of this work.

## Minimal Retained Evidence

Use existing packet conventions where their semantics match. A selected
attempt needs only enough retained material to reconstruct its scientific claim:

- source, data/split, checkpoint, normalizer and trusted-map identities;
- declared probe inputs, directions, times, amplitudes and information cost;
- fitted predictive quantities, validity checks and uncertainty assumptions;
- numerical forecasts, decision rule, selected action and abstentions frozen
  before corresponding outcomes;
- confirmation outputs, failures/censoring, clean/phase/physical checks and cost.

Preserve historical packets exactly; a revised recipe has a new identity.
Truth and solver diagnostics remain offline unless a different deployment
interface is explicitly selected. Protected-role access stays separately
controlled. Do not create a new report format merely to duplicate the tracker.

## Focused Verification

Choose tests for the selected mathematical operation and its consequential
failure modes. Depending on the change, these may include affine forcing/
response identities, ordered nonnormal propagation, signed finite-amplitude
effects, omitted-space feedback, correction-induced clean bias, phase harm,
and explicit abstention outside the supported range.
Use synthetic CPU fixtures before a bounded real-data check.

Retain applicable existing coverage for input/target pairing, normalization,
precision, source binding, state validity, replay and honest failure accounting.
Run the narrowest relevant tests; broaden only for a concrete dependency risk.
Record newly executed checks separately from inherited test evidence.
No exhaustive audit or external-review cycle is imposed by this plan.

## Resources And Ownership

Keep reusable branch code under utility/time_dependent_no/, entry points under
scripts/time_dependent_no/, and focused tests under tests/time_dependent_no/.
Measure the authorized pilot's runtime, memory, storage and model/JVP call cost
on the owner-selected workstation; use those measurements for later estimates.
Preserve other agents' resources and local recovery copies. Infrastructure
extraction, cleanup, migration, purchase, commit and publication are separate
actions, not prerequisites silently added to the scientific task.

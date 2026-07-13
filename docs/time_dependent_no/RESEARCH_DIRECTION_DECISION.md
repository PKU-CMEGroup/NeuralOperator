# Research Direction Decision: Medium-Horizon Forecasting Before Data Assimilation

Date: 2026-07-12
Status: Accepted working direction

## Decision

The immediate objective is to build a credible neural-operator-based algorithm
for medium-horizon, open-loop prediction of time-dependent PDEs before adding
data assimilation.

Data assimilation remains central to the long-term program, particularly for
chaotic systems. It will not be used to compensate for forecast defects already
visible on nonchaotic or weakly chaotic benchmarks.

Idea 2.1 is the first method-design stage of this program. It asks which
solver-facing quantity a neural operator should learn so that a global model and
a structured numerical update jointly produce an accurate, stable, and
interpretable forecast.

## Position In The Grand Plan

The long-term framework is a learned numerical forecast-analysis system:

1. A neural operator predicts a solver-facing quantity such as a macro residual,
   time-integrated numerical flux, interface predictor, closure, or dissipation.
2. A numerical scaffold converts it into a forecast state while enforcing
   selected structure.
3. Reliability diagnostics expose nonphysical behavior, forecast error, and
   distribution shift.
4. Data assimilation conditions later forecasts on observations once the
   forecast model itself is credible.
5. The resulting framework supports many-query inverse problems, control, and
   design optimization.

The forecast layer has the factorization

z_n = T_theta(U_n, parameters, geometry, dt),

U_(n+1) = Phi_dt(U_n, z_n).

T_theta may use global neural-operator context while Phi_dt is a local or
structured numerical decoder. A global implicit numerical scheme means that
z_n represents the effect of eliminated fine-scale stages or substeps. It does
not make an unconstrained latent a physical interface state.

## Why Forecasting Comes First

- The current 1D Euler and supersonic-bump failures are not established
  consequences of chaos. They expose one-step/rollout mismatch, shock-local
  degradation, target non-identifiability, and time-discretization mismatch.
- Assimilation could make a poor forecast appear usable by repeatedly resetting
  it, obscuring what the model learned and why it failed.
- A trustworthy assimilation study needs a fixed forecast model whose open-loop
  error growth, physical defects, and predictability horizon are already known.
- Medium-horizon rollout is long enough to expose accumulated numerical error
  without demanding pointwise prediction beyond a chaotic predictability limit.

Open-loop rollout remains a diagnostic after assimilation is introduced. Later
chaotic-system evaluation will distinguish short-range forecast skill, filtered
state accuracy, and long-time statistical fidelity.

## What Current Work Has Established

- Teacher-forced one-step accuracy is not a reliable measure of autoregressive
  stability.
- Current CPGNet reproduction errors are dominated by shock-local phase, shape,
  amplitude, and stability defects rather than simple one-step underfitting.
- The released CPGNet interface latent behaves as a nonphysical flux-control
  coordinate, not a verified one-sided physical trace.
- Residual, flux, and interface parameterizations with similar one-step errors
  can have sharply different rollout behavior.
- Input noise can regularize residual rollout, but may act through denoising or
  damping rather than improved wave physics.
- The legacy toy CPG-style rows are diagnostic failures, not CPGNet evidence.
  Their interface rollouts survive mainly because a whole-sample limiter
  suppresses most proposed updates.
- Conservation-form decoding, positive interface variables, or a limiter alone
  do not guarantee an accurate or stable learned scheme.

These results motivate method design but do not show that directly supervised
flux or interface targets outperform residual prediction.

## Central Scientific Hypothesis

The learning target is a first-class algorithmic design variable. Target choice
acts like coordinate selection or preconditioning in operator space: it changes
identifiability, optimization conditioning, inductive bias, and error
amplification through the numerical decoder.

> Solver-facing learning improves medium-horizon prediction when the target is
> identifiable, aligned with the macro time integrator, representable by the
> decoder, and coupled to appropriate local admissibility control.

This conditional hypothesis is more defensible than assuming that flux or
interface prediction is intrinsically superior.

## Next Experimental Path

1. Export the 1D RK3 solver's actual time-integrated numerical face flux over
   every saved interval.
2. Verify that it reconstructs each conservative-state transition, including
   boundary accounting, to numerical precision.
3. With a shared, parameter-matched backbone, compare direct conservative
   residual supervision, state-loss-only flux, direct integrated-flux
   supervision, and joint flux/state supervision.
4. Test interface-state representability before training. Retain it only if a
   precisely defined target can reproduce the reference flux under meaningful
   constraints.
5. Add noise, positivity, and multistep training only after identifying a viable
   target.
6. Replace whole-sample emergency limiting with a local conservative
   admissibility mechanism before making structure-preservation claims.
7. Stress finalists across timestep, effective CFL, resolution, parameter
   shift, and fixed-split model seeds.
8. Transfer only the winner and matched controls to the official 2D CPG
   benchmark.

The CPG benchmark is a secondary geometry and shock-transfer test. Its current
HDF5 contract lacks validated control-volume geometry and reference face fluxes,
so it cannot be the primary causal target diagnostic without additional
solver-side data.

## Medium-Horizon Promotion Gate

Data-assimilation experiments begin only after a forecast method demonstrates:

- completed rollouts for nearly all held-out trajectories at a fixed horizon;
- rollout-curve and final-error gains over a matched residual baseline;
- consistent gains across fixed data splits and multiple model seeds;
- no nonphysical raw states hidden only by primitive floors;
- infrequent, nondegenerate limiter intervention;
- boundary-aware conservation accounting;
- accurate shock position, strength, thickness, and contact behavior;
- robustness to timestep and resolution changes; and
- successful transfer to a credible 2D setting.

Thresholds should follow reference-solver scales and baseline variability. A
method that remains finite only because its limiter nearly freezes the update
does not pass.

## Later Data-Assimilation Stage

After this gate, keep the forecast model fixed and first use a standard EnKF,
LETKF, or variational method. This isolates the learned forecast target from the
assimilation algorithm.

Initial questions include the useful observation interval, required sensor and
ensemble counts, analysis-increment magnitude, uncertainty calibration, and
which physical scales or unstable directions remain poorly modeled or observed.

Learned covariance, localization, analysis operators, adaptive observation, and
solver fallback belong after this controlled forecast-plus-classical-DA stage.

## Paper-Level Interpretation

The likely first paper is:

> Neural operators as learned macro-integrators: identifiable solver-facing
> targets for medium-horizon hyperbolic PDE prediction.

It should establish a causal target comparison, a credible learned numerical
scheme, medium-horizon 1D and 2D evidence, and a mechanistic account of how
target error and decoder conditioning affect rollout stability.

A subsequent paper can place the resulting forecast model in a closed-loop
data-assimilation system for genuinely chaotic PDEs.

## Work To Stop Or Defer

- Do not use the legacy toy CPG-style model as research evidence.
- Do not broadly sweep absolute flux or interface latents supervised only
  through next-state loss.
- Do not continue the instantaneous stride-4 Rusanov correction.
- Do not treat heavy whole-sample limiting as structure-preserving success.
- Do not scale architectures broadly before fixing target identifiability and
  the time-integration contract.
- Do not introduce observations into current 1D or bump experiments to hide
  forecast defects.
- Defer learned data assimilation, uncertainty-driven fallback, and inverse
  design integration until the medium-horizon forecast gate is passed.

## Related Records

- README.md
- HANDOFF.md
- MECHANISTIC_DIAGNOSTIC_TRACKER.md
- MECHANISTIC_DIAGNOSTIC_PLAN.md
- CPGGNSPDES_REFERENCE_AUDIT.md
- CPG_EULER_DATASET_CONTRACT.md

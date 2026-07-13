# Research Direction Decision: Medium-Horizon Forecasting Before Data Assimilation

Date: 2026-07-12
Updated: 2026-07-13
Status: Accepted working direction

## Decision

The immediate objective is to build a credible neural-operator-based algorithm
for medium-horizon, open-loop prediction of time-dependent PDEs before adding
data assimilation.

Data assimilation remains central to the long-term program, particularly for
chaotic systems. It will not be used to compensate for forecast defects already
visible on nonchaotic or weakly chaotic benchmarks.

Idea 2.1 is the first method-design stage of this program. It now treats the
full solver-facing representation as the design object: state coordinates,
predicted quantity, supervision graph, and enforcement mechanism. These axes
must be tested causally rather than bundled into one larger model.

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

## State-Coordinate Decision

The current 1D and reference CPG paths do not constitute a fully conservative-
coordinate experiment. They consume primitive flow variables, use conservative
updates for residual/flux/interface decoders, recur through primitive states,
and optimize standardized next-primitive-state loss. Current evidence therefore
does not answer whether operating in conservative-variable coordinates improves
optimization or rollout.

For the next coordinate diagnostic, keep the predicted quantity fixed as the
conservative increment and separate input coordinates from loss coordinates:

1. primitive input with primitive loss: current residual control;
2. conservative input with primitive loss: input-coordinate effect;
3. primitive input with conservative loss: loss-coordinate effect;
4. conservative input with conservative loss: fully conservative form; and
5. conservative plus primitive features with joint loss only after the causal
   four-way comparison.

New conservative-form variants should keep the recurrent state in conservative
variables and derive primitive features without hidden floors. Use fixed
physical nondimensional scales as the main normalization and retain empirical
standardization only as a documented control. A pure conservative formulation
is a hypothesis, not a foregone conclusion: conservative variables align with
weak-form updates, while primitive or characteristic features can make wave
speeds, pressure, and admissibility easier to represent.

Every run must record `input_coordinates`, `recurrent_coordinates`,
`predicted_quantity`, `loss_coordinates`, target normalization, timestep,
geometry scaling, and whether primitive conversion used a floor. Do not label a
conservative decoder trained through primitive loss as a conservative-space
model.

## Structured Dense Supervision

Dense supervision is approved as a staged representation-learning strategy,
not as a kitchen-sink loss. Preserve one primary solver path and derive
algebraically related outputs through known operators. For a flux model, the
network predicts one shared oriented face flux, while conservative increment,
next conservative state, and next primitive state are computed from that flux;
independent state heads must not bypass the conservative decoder at inference.

Classify each proposed auxiliary target before implementation:

- information-adding labels include exact time-integrated face flux, uniquely
  defined stage traces, shock/contact labels, front motion, wave speed,
  entropy production, admissibility margin, and solver error/fallback labels;
- redundant but potentially conditioning-improving losses include next
  conservative state, next primitive state, conservative increment, and flux
  divergence when they are linked by the deterministic decoder; and
- unsafe labels include underidentified macro interface states, arbitrary flux
  gauges, or solver-specific internal decisions that do not serve the intended
  inference algorithm.

The first dense objective after reference-flux validation should add losses in
this order: direct integrated flux, conservative next state, primitive next
state, shock/front auxiliary supervision, then short unrolled supervision.
Semigroup, cross-resolution, entropy, and error-estimator losses are later
additions. Add one loss family at a time and report target scales, gradient
norms, and gradient conflicts. Auxiliary accuracy alone does not establish a
better representation; require data-efficiency, transfer, or rollout gains.

## Constraint Hierarchy

Enforce inexpensive algebraic and convex properties; regularize conditional or
representation-dependent properties:

- enforce interior conservation with one shared face flux, validated geometry,
  and exact owner/neighbor signs;
- enforce known boundary conditions in the solver, while reporting free versus
  clamped boundary behavior explicitly;
- use local conservative invariant-domain or antidiffusive-flux limiting for
  density and internal-energy admissibility instead of whole-sample scaling;
- prefer entropy-stable flux parameterizations or bounded entropy corrections
  when they can be made compatible with the actual time integrator; and
- use soft losses for shock geometry, short rollout, semigroup composition,
  cross-resolution commutation, and calibrated failure prediction.

Do not impose blanket componentwise TVD on multidimensional Euler. TVD is an
appropriate hard diagnostic for scalar 1D problems and a possible constraint
on characteristic or face-correction variables. For Euler systems, prioritize
conservation, invariant domains, entropy admissibility, and local oscillation
control. Global spectral or total-variation penalties can erase physical shocks
and must not be introduced without smooth/shock-region separation.

## Capability And Escalation Gate

The project must test whether a strong neural operator can handle genuinely
hard time-dependent PDE rollout before accumulating method complexity. A
50-step result must state the physical horizon, characteristic travel
distance, timestep, effective CFL, and number of autoregressive model calls.
For each serious candidate, measure:

1. teacher-forced one-step accuracy;
2. direct prediction to selected future horizons as a representation oracle;
3. at least 50 raw autoregressive calls where the data contract supports them;
4. timestep, resolution, and parameter transfer; and
5. performance against a coarse classical solver at matched runtime or error.

The 1D ladder must include shocks, contacts, rarefactions, reflections, and wave
interactions. The CPG bump remains an irregular-geometry attached-shock transfer
test, but it is not sufficient evidence for dynamic moving shocks. A credible
story ultimately needs a dynamic 2D shock problem in addition to the bump.

Escalate engineering only in response to a named failure:

- use a stable coarse/subcycled solver plus learned time-integrated conservative
  correction when raw proposals are nonphysical or worse than coarse CFD;
- use a physics-routed facewise expert or dissipation model when error is
  localized to shocks/contacts, with one gate shared by both adjacent cells;
- use an explicit front/level-set plus one-sided smooth fields when shock phase
  and smearing remain dominant after local-global controls; and
- use INR space-time representations for smooth pieces, front geometry, or
  adaptive querying, not as an assumed cure for a raw discontinuous field.

Adaptive time stepping should be driven by a measurable defect such as one
full step versus two half steps, semigroup inconsistency, entropy violation, or
admissibility risk. Front tracking must use Rankine-Hugoniot-compatible motion
and conservative remapping; generic per-cell MoE and visually sharp but
nonconservative front prediction do not qualify.

## Next Experimental Path

1. Finish the corrected solver-level CPGNet baseline as a reference validation,
   not as evidence for solver-facing target superiority.
2. Run the four-way conservative-coordinate diagnostic with the output fixed as
   a conservative residual and no new dense losses or architectural changes.
3. Export the 1D RK3 solver's actual time-integrated numerical face flux over
   every saved interval and verify conservative transition and boundary closure
   to numerical precision.
4. With a shared, parameter-matched backbone, compare conservative residual,
   state-loss-only flux, direct integrated-flux, and joint flux/state
   supervision.
5. Add shock/front auxiliary supervision to the best identified primary target;
   do not add independent bypass heads.
6. Evaluate every finalist under the one-step/direct-horizon/50-step capability
   protocol before adding noise, unrolling, or safety mechanisms.
7. Add short unrolled training, semigroup consistency, cross-resolution
   consistency, and calibrated failure prediction one family at a time.
8. Replace whole-sample emergency limiting with a local conservative
   invariant-domain mechanism before making structure-preservation claims.
9. Trigger facewise routing, solver defect correction, or front tracking only
   when the capability diagnostics identify the corresponding failure mode.
10. Transfer the winner and matched controls to CPG and then to a dynamic 2D
    shock benchmark.

The CPG benchmark is a secondary geometry and shock-transfer test. Its current
HDF5 contract lacks validated control-volume geometry and reference face fluxes,
so it cannot be the primary causal target diagnostic without additional
solver-side data.

## Medium-Horizon Promotion Gate

Data-assimilation experiments begin only after a forecast method demonstrates:

- a documented one-step/direct-horizon/autoregressive capability decomposition;
- at least 50 raw autoregressive calls over a meaningful physical horizon where
  the dataset supports that protocol;
- completed rollouts for nearly all held-out trajectories at a fixed horizon;
- rollout-curve and final-error gains over a matched residual baseline;
- consistent gains across fixed data splits and multiple model seeds;
- no nonphysical raw states hidden only by primitive floors;
- infrequent, nondegenerate limiter intervention;
- boundary-aware conservation accounting;
- accurate shock position, strength, thickness, and contact behavior;
- robustness to timestep and resolution changes; and
- successful transfer to both irregular geometry and a credible dynamic 2D
  shock setting.

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

> Neural operators as learned conservative macro-solvers: coordinates,
> solver-facing supervision, and stability for shock-dominated PDEs.

It should establish causal comparisons rather than claim that every component
is necessary. The paper may center on the strongest supported axis among state
coordinates, identifiable target supervision, or constrained decoding, while
the other axes remain matched controls. It still needs a credible learned
numerical scheme, medium-horizon 1D and 2D evidence, and a mechanistic account
of how representation and decoder errors amplify under rollout.

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
- Do not combine coordinate changes, dense supervision, constraint layers,
  MoE routing, and front tracking in the first experiment.
- Do not train independent state and flux heads that can disagree at inference;
  derive redundant solver quantities through the primary update path.
- Do not impose global componentwise TVD, total-variation, or spectral damping
  on Euler without separating smooth and discontinuous regions.
- Do not describe a grid residual as resolution invariant without a verified
  cross-resolution operator contract.
- Do not implement generic per-cell MoE or raw-field INR as a shock solution
  before simpler conservative local-global controls fail.
- Do not start a full shock-fitting platform before phase/smearing diagnostics
  establish that moving-front representation is the dominant bottleneck.
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

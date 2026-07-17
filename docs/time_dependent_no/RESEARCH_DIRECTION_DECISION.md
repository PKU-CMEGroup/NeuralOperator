# Research Direction Decision: Medium-Horizon Forecasting Before Data Assimilation

Date: 2026-07-12
Updated: 2026-07-16
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
- On the fixed 1D Euler stride-4 dataset, corrected CPGNet becomes a competitive
  raw macro-step solver when message depth is raised from 12 to 28: completion
  rises from 34/64 to 64/64 and the effective-CFL/error correlation disappears.
  Parameter-matched wide-shallow and narrow-deep controls support receptive
  field rather than parameter count as the primary mechanism. Its remaining
  shock tail is a persistent stationary ghost front, magnified but not created
  by argmax front detection.
- Exact accepted-substep ADER face impulses close the saved state transition and
  can supervise a conservative flux-form FNO without an implementation-level
  label mismatch.
- In 1D, removing the one constant-flux null mode makes the remaining face field
  information-equivalent to the state increment. Its MSE changes the residual
  norm toward an inverse-divergence, low-frequency weighting; it is not an
  independent physical target after projection.
- The first midscale solver-flux screen confirms that distinction: the
  gauge-canonical joint row reduces divergence-active face error but worsens
  held-out state error and fails all paired rollouts earlier than state-only.
  Better native-target fit is therefore not sufficient evidence for a better
  learned solver.
- Short differentiable recurrence is a useful optimization intervention for the
  same conservative flux-form FNO. At 64/16/16 scale it improves one-step fit,
  raises raw survival by `1.464x`, and yields the first 2/16 completed rollouts.
  A matched smooth admissibility barrier neither improves survival nor reduces
  the 14 remaining nonpositive terminations, so the supported mechanism is
  recurrent-distribution training rather than that barrier.
- At the full 384/64/64 scale, the same recurrent recipe reaches one-step
  relative L2 `0.001305`, mean survival `0.97734`, and 57/64 completed 20-call
  raw rollouts at mean initial effective CFL `3.84`. It passes the scale and
  20-call promotion gates but misses the strict 90% completion gate by one
  trajectory. Seven nonpositive proposals remain, so this is not evidence that
  positivity constraints are unnecessary in general.
- The frozen flux checkpoint does not pass the actual 50-call capability gate.
  Only 1/64 cases completes 50 calls, mean survival is `0.566`, and no case
  reaches 100 calls; all terminate on raw nonpositive proposals by call 51.
- Direct next conservative state and conservative residual are label-valid and
  information-equivalent, but not optimization-equivalent. Direct state misses
  the tiny-fit threshold on both declared seeds, while residual passes. At
  64/16/16, recurrent residual reaches one-step relative L2 `0.002649`, 16/16
  completion, and zero nonpositive failures, versus `0.003438`, 2/16, and 14
  for the matched flux head. This supports output centering and the identity
  bypass as an important target-parameterization effect.
- At full 384/64/64 scale, recurrent residual reaches one-step relative L2
  `0.001123` and completes 64/64, 62/64, and 61/64 raw rollouts at 20, 50, and
  100 calls. Mean 100-call survival is `0.97781`; the three failures are
  interior pressure collapses at calls 33, 37, and 91, with no density or
  nonfinite termination. Residual is lower-error than the matched flux head on
  63/64 cases at their longer common endpoint. This establishes a strong
  fixed-setting baseline, but not a seed-, stride-, resolution-, or
  architecture-independent target ordering.
- The matched three-seed residual/projection screen resolves the first seed
  gate. Plain residual completes 182/192 pooled 100-call rollouts; projected
  residual completes 187/192 and closes exactly to a learned low-dimensional
  boundary budget. That stability gain does not become a robust accuracy gain:
  the projected three-seed mean state-error ratio is `1.016`, and among 181
  common completers its state L2 is 9.8% worse, conserved-total error 5.9%
  better, and shock MAE 4.8% worse. Only stability noninferiority passes the
  preregistered family gates. Projection is therefore a structural ablation,
  not the replacement accuracy baseline.
- A matched 64/16/16 generated-burn-in pilot on projected residual gives a
  strong but nonuniform distribution-exposure signal. With identical one-step
  histories and near-matched recurrent update counts, eight detached burn-in
  calls before four-step BPTT reduce 50- and 100-call state L2 by 23% and 36%.
  It wins 15/16 and 14/16 paired common-endpoint comparisons, respectively.
  The primary gate still fails because 20-call shock MAE rises 38% and 100-call
  survival drops from `0.9650` to `0.9394`: one clean failure is rescued, one
  new call-58 failure appears, and a shared failure moves three calls earlier.
  This supports later-state exposure as an error-drift intervention, not as a
  complete stability mechanism or a reason to promote projected residual.
- The same generated-exposure gate on plain residual gives a stronger
  medium-horizon signal. It lowers final state L2 by 4.6% and 32% at 20 and 50
  calls; at 100 calls it lowers common-endpoint state error by 51.4%, wins
  14/16 cases, and raises completion from 15/16 to 16/16. Conserved-total error
  also improves at all horizons. The preregistered gate remains formally
  failed because the legacy pressure-argmax shock MAE rises 45.6% at 20 calls.
  Focused replay separates two front-strength rank changes from one genuine
  spurious/displaced-front regression. Generated burn-in therefore remains a
  promising recurrent-distribution intervention, but a teacher-offset control
  is required to distinguish off-manifold exposure from later-time sampling.
- The matched teacher-offset control resolves that ambiguity on the fixed
  64/16/16 split and seed. Later-time sampling alone is 10.9% worse than clean
  at 20 calls and tied at 50/100 calls. Generated exposure beats teacher by
  13.9%, 33.9%, and 50.8% in common-endpoint state error at 20/50/100 calls,
  wins 15/16 H100 cases, and completes 16/16 versus 14/16. The bootstrap
  intervals exclude parity at all three horizons. Thus model-generated
  off-manifold exposure contributes beyond time offset in this setting.
  Method promotion remains blocked: generated versus clean top-two
  front-position error is 17.8% worse at H20 and 10.2% worse at H50, and the
  original-trajectory label is not the reference PDE advance from a perturbed
  generated state.
- The restriction-consistent resolution gate separates shared representation
  from grid-dependent numerical targets. One equal-presentation 64/24/4 FNO
  stays within `1.296x` same-grid-oracle one-step error and `1.418x` H20/H50/
  H100 state error on exact 128/256/512-cell restrictions, never loses
  completion, and improves frozen off-grid one-step error by 68.1%/78.3% at
  nx128/nx512. The same weights do not reproduce independently evolved native
  coarse maps: native nx128/nx256 one-step error is 5.27/3.65 times the
  restriction-consistent value. D029 was therefore substantially a target-
  consistency problem, not an FFT-capacity result. The shared model still loses
  three pressure-limited cases by H100, so neither native-solver invariance nor
  positivity-free medium-horizon reliability is established.

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

## Optimization Lens

There is no architecture-independent ordering of easy and hard targets. For a
fixed model and optimizer, target difficulty depends on the target's alignment
with the model parameterization, the loss metric, the data distribution, and
the supervision path. Representability does not imply that gradient descent
will find the represented operator on the available budget.

For a flux head linearized as `F_theta = F_0 + J delta_theta` and a conservative
decoder `U_next = U - A F_theta`, the local Gauss-Newton matrices are

`H_flux = J^T J`,

`H_state = J^T A^T A J`.

State-through-divergence supervision is blind to `ker(A)` and weights visible
flux modes by the singular spectrum of `A`. Direct flux supervision changes
both identifiability and conditioning. This is a local optimization statement,
not a proof that full flux loss gives a better solver: an exactly
divergence-free error is harmless under the same fixed decoder, and direct
loss may waste capacity on a solver-specific gauge.

The primary conditioning diagnostic should distinguish

1. raw decoded-state loss, `||A e||^2`;
2. whitened divergence-active or projected loss,
   `||(A A^T)^(dagger/2) A e||^2 = ||P_range(A^T) e||^2`; and
3. full direct-flux loss, `||e||^2`.

Raw versus divergence-active loss tests conditioning. Divergence-active versus
full flux tests whether selecting the reference solver's nullspace
representative adds useful information. These controls are analysis tools; do
not add all three to the first production matrix if the simpler comparison is
already decisive.

Every apparent failure must be classified before changing the method:

- failure to overfit a tiny supervised subset indicates an implementation,
  effective-capacity, scaling, or optimization problem;
- cold-start failure followed by a lower final-target training-loss floor after
  a horizon curriculum supports continuation-sensitive optimization only when
  the final-stage and total-exposure cold controls below are satisfied;
- low training error with poor held-out error indicates generalization, data
  coverage, or closure failure rather than optimizer failure;
- different reference labels for nearly identical model inputs indicate a
  non-Markovian or underresolved closure, for which deterministic MSE learns a
  conditional mean; and
- good teacher-forced and held-out one-step error with bad free rollout
  indicates recurrent distribution shift or unstable error propagation.

Record observations first and mechanisms second. Kernel alignment, spectral
bias, gradient conflict, shock rarity, and denoising are hypotheses until an
intervention changes the predicted optimization behavior.

## Macro-Flux Target Contract

Use `cumulative conservative face impulse` for the solver-facing quantity

`I_f[n:n+K] = sum_(k in accepted substeps) dt_k Fbar_(f,k)`,

where `Fbar_(f,k)` is the solver's owner-oriented normal numerical flux density
averaged with the actual quadrature for accepted substep `k`. For an RK solver
this average is the weighted stage flux; for ADER it is the returned
space-time-predictor quadrature flux. Rejected retry attempts contribute zero.
The final accepted retry or first-order fallback contributes exactly once.

Use `time-averaged face flux` only for `I_f / Delta T_sample`, where
`Delta T_sample = sum_k dt_k` is the actual saved-interval duration. Avoid the
unqualified term `mean field`, which can be confused with an ensemble or
statistical mean.

The current 1D artifact stores conservative flux density times time, oriented
owner-to-neighbor on interior faces and outward on boundaries; the discrete
decoder owns the face signs and `1 / dx` factor. For higher dimensions, record
whether face measure is stored in `I` or in `D_h`, and never apply it twice.

Before training, require serialized closure to numerical precision:

`U_(n+K) - U_n = -D_h I[n:n+K]`,

where `D_h` includes the documented face orientation, geometry, mass, and
boundary convention. Do not separately add boundary terms when boundary-face
impulses are already included in `I`. The current 1D gate is maximum float64
pre-serialization closure below `1e-12`, followed by float32 decoded closure at
`rtol=3e-5`, `atol=3e-6`; any changed dtype or scale requires a documented
replacement tolerance before data generation.

Temporal averaging can suppress fast temporal oscillations, but it does not
guarantee an easier operator from the initial state. Larger horizons expand the
domain of dependence, increase sensitivity to unresolved state, and introduce
kinks when shocks cross faces or solver branches change. Horizon difficulty is
therefore an empirical axis, not an assumed benefit of averaging.

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

Apply the same failure taxonomy to the coordinate matrix. Select a provisional
coordinate contract from matched train/validation and rollout evidence, but do
not treat a coordinate as intrinsically unsuitable when it failed label,
tiny-set-fit, or optimization checks.

## Target-Family Empirical Program

Empirical target exploration remains central. Theory supplies prior
predictions, controls, and falsifiers; it does not justify removing a target
family before a valid test. Positive and negative results are both evidence,
but only within the tested coordinates, decoder, stride, data regime,
normalization, and optimization budget.

Ask two questions for every family:

1. Can the final model optimize the target? Report tiny-set fit, training-loss
   floor, convergence time, seed variance, full-batch versus minibatch
   behavior, and per-target gradient norms or conflicts.
2. Does a well-fitted target induce a better solver? Report held-out
   supervised-objective error, decoded-state error, direct-horizon error, raw rollout,
   shock, conservation, positivity, boundary, limiter, and transfer
   diagnostics. For directly supervised rows, also report native-target error.
   For the state-loss-only flux row, decoded-state loss is its native supervised
   metric; report divergence-active impulse error separately and label full
   reference-impulse error as gauge-dependent.

The core matched screen is

1. direct next conserved state;
2. conservative residual or increment;
3. a flux head trained only through decoded-state loss;
4. directly supervised cumulative conservative face impulse; and
5. joint cumulative-impulse and decoded-state supervision.

Rows 1 and 2 isolate output centering and the identity bypass for two
information-equivalent state targets. Rows 3 and 4 isolate the supervision
graph while keeping the flux-form architecture fixed. Use a common decoded
state metric so native target scaling is not mistaken for target superiority.
The joint row is a prespecified supervision control, not the later
dense-supervision stage.

Conditional second-stage families are

- uniquely exported solver-stage interface states, or states selected by an
  explicit inference-relevant rule, decoded by one fixed Riemann solver;
  endpoint closure alone does not identify an interface-state label;
- a cumulative coarse-solver or other inference-computable base impulse plus a
  normalized learned correction, but only after auditing base error and
  correlation; the base may use the current coarse state, parameters, and
  geometry, but not future reference states or solver stages, while
  training-set statistics may be used only for normalization;
- characteristic wave or fluctuation targets when the basis and degeneracies
  are controlled; and
- divergence-active or gauge-aware flux targets when the full reference gauge
  is shown to be unnecessary or harmful.

Pre-register tendencies and their falsifiers rather than retrospective
mechanistic stories:

- residual prediction may optimize faster than direct next-state prediction
  because it centers a near-identity map; loss whitening or an explicit skip
  that removes the gap would attribute the effect to parameterization;
- direct impulse labels should improve common decoded-state fit,
  divergence-active impulse error, or data efficiency over the state-loss-only
  flux head if missing local supervision is important; full impulse error for
  the state-loss-only row is gauge-dependent, and no matched gain weakens the
  local-supervision mechanism;
- a fixed-Riemann interface target should help only when its label-selection
  rule is unique and inference-relevant, and its proposed states remain
  credible without constant emergency limiting;
- base-plus-correction should help only when the cumulative base is correlated
  with the reference macro impulse; otherwise the network must cancel the base
  before learning the target; and
- characteristic or dense targets may amplify rare wave and shock signals but
  can lose through basis degeneracy or conflicting gradients.

Surprises are expected. Update the hypothesis after the controlled result; do
not retrofit the mechanism to make the run agree with the prior.

The rejected absolute flux, absolute interface, and instantaneous stride-4
Rusanov-correction rows remain negative controls for those exact formulations.
They do not reject identifiable cumulative-impulse supervision, a
uniquely selected fixed-Riemann interface target, or a credible macro-flux
base.
Do not rerun the rejected formulations unchanged.

Run two distinct comparison regimes:

- In the `strict matched` regime, freeze coordinates, data, backbone capacity,
  seeds, compute, clean-input contract, and decoded-state metric. Declare the
  deterministic decoder required by each target family. Within the
  state-loss-only, direct, and joint flux supervision block, also freeze the
  exact decoder and output parameterization. Use this regime for causal
  statements about target and supervision choice.
- In the `best engineered` regime, give surviving families equal tuning budget
  for target-specific normalization, baseline correction, horizon curriculum,
  and optimizer settings. Use this regime to select the practical algorithm,
  not to claim intrinsic target superiority.

Use successive halving. Before a batch starts, its run manifest must declare
dtype-aware closure tolerances, normalized supervised-objective and decoded-state
tiny-set pass thresholds, the maximum diagnostic interventions, optimizer
updates and data exposure per row, and the seed policy. Do not choose pass
thresholds after seeing the results. First require applicable label closure and
tiny-set fit; then run matched train/validation curves; then direct-horizon and
raw rollout tests. An apparent failure receives one predeclared repeat
initialization before rejection. Only finalists receive full seed confirmation,
timestep/resolution transfer, and 2D evaluation. A one-step winner is not
promoted when raw rollout or structural diagnostics fail.

Send any label-valid family, with closure also validated where its decoder
contract requires it, that fits at short stride but fails cold at a larger
stride into the continuation diagnostic, regardless of its large-stride
ranking. The progressive arm uses sequential fixed-stride checkpoints over
`Delta t -> 2 Delta t -> 4 Delta t -> ...`; it is not mixed-stride training or
a newly timestep-conditioned model. Here `Delta t` is one saved reference
interval, not an internal accepted CFL substep. Transfer model weights only,
and reset the optimizer and scheduler at each stride; carrying optimizer state is a labeled
secondary ablation. At the final stride, hold architecture, data cases,
final-horizon labels, normalization, loss weights, optimizer/scheduler contract,
and final-stage sample presentations identical to a cold final-target arm.
Define sample presentations as batch size times optimizer updates, including
repeated examples, rather than epochs or nominal trajectory cases. Add a
total-exposure-matched cold arm whose final-target sample presentations equal
all continuation stages, plus composition of a small-step model trained on
`Delta t` labels. A lower final-target training-loss floor than both cold
controls supports a continuation-sensitive optimization claim; validation or
rollout gains without that fit gain indicate curriculum-induced generalization
or stability instead. None of these outcomes shows that the large-step operator
is intrinsically easier or physically better.

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

Direct and joint impulse supervision belong to the primary target screen, not
to this dense stage. After selecting a primary target and validating its
reference label, add only auxiliary losses not already present in that
objective, in this order: decoded conservative state, decoded primitive state,
uniquely defined stage or partial-impulse traces, shock/front supervision, then
short unrolled supervision.
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

1. Freeze h128/mp28 as the strong corrected CPGNet reference. The matched
   controls support a locality mechanism; do not turn this into a broad graph
   architecture sweep or target-superiority claim.
2. Freeze conservative input, conservative loss, fixed physical scaling, and
   conservative recurrence as the provisional coordinate contract within D021.
   The coordinate matrix is complete; do not reopen it during the target screen.
3. Preserve the validated cumulative-impulse dataset and the completed
   tiny/repeat/optimization/gauge controls. Stop the exact gauge-canonical joint
   face-value MSE row at midscale because it fails the decoded-state and rollout
   gates despite fitting its native target.
4. The named stability intervention is complete. Unroll-only passes the
   predeclared midscale gate; the tested training-only barrier fails attribution.
   Keep inference limiter-free and the conservative flux decoder fixed.
5. Freeze the full-scale flux-form row as a strong 20-call result and a failed
   50-call baseline. The 1/64 completion at call 50 rules out a medium-horizon
   stability claim for this checkpoint.
6. Stop uncentered direct next-state prediction in the strict matched regime
   after its two-seed tiny-fit failure. Residual is the surviving centered
   parameterization; adding an identity skip to direct state would make the two
   parameterizations algebraically equivalent.
7. Freeze the completed three-seed residual family as the strong fixed-setting
   accuracy baseline. Preserve pooled completion and matched common-case
   readouts; do not collapse failed prefixes into a single accuracy claim.
8. Freeze the completed projected-residual family as a structural ablation. Its
   exact learned-budget closure and pooled stability gain are useful, but its
   state, robust-conservation, and shock gates fail; do not promote it as the
   more accurate target or return to full face-value MSE.
9. The teacher-offset 8+4 control is complete. It rules out later physical-time
   sampling as the source of the generated-burn gain on this split and seed.
   D023 is complete on all 16 test cases, starts 0:10:80, and prefix depths
   0/2/4/8. At depth eight, generated/teacher error ratios are `0.992` to the
   original next state and `1.017` to the same-state solver continuation, with
   correction alignment `0.074`. It therefore passes neither the PDE-map nor
   strong trajectory-correction threshold. Treat generated burn-in as a useful
   closed-loop intervention with unresolved modified-dynamics mechanism, not a
   demonstrated PDE-consistent operator improvement. Do not full-scale while
   H20/H50 top-two front position remains worse than clean.
10. D013 is complete for 1D. Teacher-forced flux and residual fits have similar
    high-band content, but recurrent flux failure has 10.4 times the matched
    residual high-band error and 154 times its second-difference error.
    Classify this as recurrent spectral growth, not one-step spectral
    underfitting. A 2D extension must use audited graph-native scale bins.
11. D024 rejects blind conservative Laplacian viscosity on the frozen flux
    model. No tested coefficient improves H50 completion; small coefficients
    worsen state and Nyquist-tail error, and larger coefficients shorten
    survival. Do not repeat broad current-state smoothing sweeps.
12. D025 is complete and fails its successive-halving gate. The faithful global
    face-grid FNO uses relative positive traces, a shared Rusanov or central
    flux, physical boundaries, exact FV update, and raw recurrence. At 64/16/16
    scale, frame-zero weighting lowers selected test one-step relative L2 to
    `0.00791` and initial-call conservative error by 73%, but all 16 H50
    rollouts still fail within four calls. This rules out shock rarity as a
    sufficient explanation. Do not promote this parameterization to full scale
    or four-step training.
13. The D025 training-only pressure/internal-energy control is also complete.
    Barrier weight `0.1` eliminates the measured violation inside two-step
    training windows but leaves test survival below `0.05` with 0/16 complete.
    This does not reject risk-sensitive constraint losses in general, but it
    rejects another mean-barrier sweep for this row. Keep inference limiter-free
    and do not add blanket TV.
14. D026 closes the direct boundary-exchange auxiliary at the matched gate.
    Weights `0.01` and `0.1` improve the intended boundary and conserved-total
    metrics, but both worsen one-step, H20 state, and shock errors while leaving
    completion unchanged at 16/16. This is a structural Pareto ablation, not a
    promoted method. Do not search more coefficients or run full seed repeats.
15. D027 establishes a useful cold stride-2 residual operator at effective CFL
    maxima up to about 13.3. It improves common-case H20/H50/H100 state error by
    25.0%, 36.2%, and 48.6% over stride-1 composition and completes 16/16 at
    H100. Its direct frame-2 error is 1.216 times composition, so classify it as
    large-step capacity with an initial-jump defect, not unconditional timestep
    robustness.
16. D028 weight-only continuation repairs frame 2 and lowers final-target
    training floors by more than 40%, but it is not a uniformly better solver:
    H50 state and shock errors are 1.095 and 1.565 times cold. Stop before the
    conditional total-exposure control and make no continuation-sensitive
    optimization claim. Preserve cold stride 2 as the large-step accuracy
    baseline and warm stride 2 as a fit/stability tradeoff ablation.
17. D029 completes the frozen 128/256/512-cell transfer gate. Exact physical
    case identity and nx256 metric replay pass, but neither checkpoint matches
    the native off-grid one-step map: error rises by 5.5--8.3 times, fine-grid
    shock geometry regresses, and cold stride 2 loses one near-pressure-margin
    case on both new meshes. Do not call the current residual FNO mesh invariant.
    The larger-step advantage is more robust: cold stride 2 still beats
    stride-1 composition at H20/H50/H100 on both off-grid meshes with equal
    same-grid completion.
18. D030 resolves the immediate representation question. Exact restriction and
    update-label commutation pass, and the unchanged 24-mode shared FNO passes
    the equal-presentation same-grid-oracle and frozen-baseline usefulness
    gates. Do not add explicit cell width, timestep, or bandwidth to explain
    D029. The row fails native coarse-solver equivalence and still has three
    consistent pressure terminations by H100; it remains a fixed-step operator
    for one restriction-consistent flow map, not one timestep-conditioned or
    arbitrary-discretization model.
19. Isolate post-fit tail stability on the strong residual baseline. Compare an
    equal-update, equal-generated-exposure continuation control with one
    training-only tail-risk pressure/internal-energy penalty, keep raw inference,
    and predeclare accuracy, shock, conservation, and completion noninferiority.
    Do not bundle TV, viscosity, projection, or an interface decoder into the
    first row.
20. Give survivors an equal-budget best-engineered comparison, then add uniquely
    defined stage/partial-impulse and shock/front auxiliaries one family at a
    time. Derive redundant states through the primary decoder.
21. Evaluate every finalist under the one-step/direct-horizon/100-call raw
    capability protocol, timestep/resolution transfer, and a coarse-CFD
    comparison before transfer to CPG geometry and a dynamic 2D shock benchmark.

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

The target program has two legitimate outcomes. A positive algorithmic result
requires a strict matched gain, a competitive best-engineered method, and
seed-stable structural and rollout improvements. A negative result is useful
only after applicable label closure, target-fit checks, continuation checks
where short-stride fit and large-stride cold failure make them relevant, and a
clear failure classification across more than one credible architecture or PDE
setting. A failed toy head or one optimizer setting cannot reject a target
family.

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

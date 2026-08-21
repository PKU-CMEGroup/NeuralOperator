# W26-L2-P0/P1 Shock Representation And Differential-Path Preregistration

Updated: 2026-08-11

Status: A1 synthetic implementation and CPU-contract validation complete. The
P0 full-PCNO fit `W26-L2-P0-FULL-S1701` completed and its exact artifact bundle
was retrieved. It passed all 24 fixed-grid training discontinuities; the exact
held guards passed 9/12 step and 9/12 pulse cases, with systematic failures at
phase `0.875`. On 2026-08-11 the owner authorized the frozen checkpoint-bearing
P1 branch/pathway analysis and use of either available cloud resource. P1 uses
the lower-cost personal GPU first and performs no optimization. The working
labels are `W26-L2-P0` and `W26-L2-P1`; no D-series identity has been allocated.
P1 subsequently completed with a hash-verified 18-file result bundle. The owner
then authorized the mentor-prioritized P2 matched gradient-ablation matrix and
granted use of both available cloud resources. P2 is restricted to the exact
nine synthetic trainings bound below. No sealed population or non-synthetic
dataset access is authorized here.

Implementation base: `4d90436ffd4d9fd27797a8551b232196253ea623` on
`time-dependent-no`.

## Terminology And Claim Boundary

In this research line, **Gibbs is an analogy only**. Registered schemas,
metrics, decisions, and claims use `shock-local oscillation`, `ringing`,
`representation`, `phase`, `grid`, and `differential pathway`.

A sharp front, a shock-local bump, high physical-wavenumber energy, or a single
alternating profile is not classified as classical Gibbs. The optional direct
truncated-spectral comparator is an explanatory retrieval control, not the
definition of a full-PCNO failure. There is no `classical_gibbs=true/false`
field in this experiment.

The two A1 deliverables establish only that:

1. exact analytic cell-average cases, nested-grid contracts, physical metrics,
   and differential-path taps are implemented correctly; and
2. a no-write toy PCNO replay closes on CPU without parameter mutation.

They provide no trained capacity, phase generalization, resolution transfer,
checkpoint, rollout, conservation, or PDE result.

## Nonduplication And Current Mechanistic Boundary

- D052 closed its exact frozen branch-gain attenuation matrix; no arm passed.
- D070C supports full differential and nonlinear fixed-hop composite pathways,
  but not one additive gradient-only cause.
- D073-A already passed its same-hidden layer-3 physical-radius question. It is
  not repeated. This line adds analytic fronts, per-grid distributions,
  pre-Softsign/saturation taps, and a decoded ablation response.
- D080--D081 tested decoded-proposal graph dissipation and its early/late/dose
  schedules. No W26-L2 filter is applied there or given a renamed schedule.
- D073-B remains the recurrent A2/A0/A1 comparison. If authorized later, it
  must share W26-L5's transfer-native comparator, cases, timing, source
  contract, and no-more-than-5% per-case control limits.

Current core execution order is frozen as:

```text
raw = least_squares_gradient(hidden)
aggregated = self_plus_neighbor_average(raw, iterations=2)
pre_softsign = gw1 * aggregated
post_softsign = Softsign(pre_softsign)
differential = bias_free_gw2(post_softsign)
branch_sum = spectral + pointwise + differential
nonfinal hidden = hidden + GELU(branch_sum)
final hidden = branch_sum
decoded increment = fc2(GELU(fc1(final hidden)))
```

More exactly, for target cell `i` with face-neighbor coordinate-difference
matrix `DeltaX_i` (one row per neighbor), geometry construction stores column
`j` of `pinv(DeltaX_i, rcond=1e-12)` on directed edge `(i,j)`. Runtime scatter
then evaluates

```text
g_i = sum_(j in N(i)) w_ij * (hidden_j - hidden_i).
```

One graph-average application is
`A(y)_i=(y_i+sum_(j in N(i)) y_j)/(1+|N(i)|)`; the maintained differential
path uses the sequential result `A(A(g))`. Thus the self term, uniform incoming
neighbor weights, two applications, and boundary degree are all part of the
registered operator rather than an informal convolution description.

`gw1` is one learned scalar initialized to `0.01`. The maintained
`inv_L_scale_hyper` constructor argument is not applied in the differential
path. Therefore a fixed learned scale directly receives the grid-dependent raw
gradient.

## P0 Analytic Target Contract

### Domain, grids, and geometry

- Physical domain: `[0,1] x [0,1/2]`.
- Registered nested grids: `32x16`, `64x32`, and `128x64` square cells.
- Native/train grid: `64x32`, with `h0=1/64`.
- Cell centers, physical volumes, open four-face adjacency, and least-squares
  weights are regenerated independently on each grid.
- Gradient pseudoinverses use the maintained dynamic-FV contract with
  `rcond=1e-12` and must reproduce coordinate gradients to `1e-10` or better.
- PCNO quadrature weights are physical cell volumes divided by domain volume;
  they sum to one. Integral metrics use physical volumes, not normalized PCNO
  weights.

No boundary enforcement or Euler wrapper is used. This is a scalar core-PCNO
representation test, not an Euler rollout.

### Exact cell averages

For a cell `I=[a,b]`, the rectangle average is

```text
avg(1_[left,right); I)
  = max(0, min(b,right)-max(a,left)) / (b-a).
```

The four families are:

1. Moving left step, with `u_s(x)=1_{x<s}` and fixed displacement
   `delta=1/8`:

   ```text
   current = avg(u_s)
   next = avg(u_(s+delta))
   increment = next - current = avg(1_[s,s+delta)).
   ```

2. Translated rectangular pulse. Here `s` is the **left edge**, pulse width is
   `W=1/4`, and `delta=1/8`:

   ```text
   current = avg(1_[s,s+W))
   next = avg(1_[s+delta,s+delta+W))
   increment = next - current.
   ```

   Its signed increment integral is exactly zero.

3. Mollified-front control
   `0.5*(1-tanh((x-s)/(1/32)))`, integrated with a stable log-cosh primitive.
   This is explicitly a steep mollified-front control, not the clean smooth
   case.

4. Genuine smooth control `sin(2*pi*4*(x-s))`, with analytic cell integrals,
   unit amplitude, and zero background.

The fixed `delta=1/8` is an integer number of cells on all three grids. Any
positive result is therefore bounded to phase-conditioned discrete translation;
it is not arbitrary-displacement equivariance.

### Two phase populations

Anchors are `a in {1/4,3/8,1/2}`.

Fixed-grid phase cases use the spacing of the evaluated grid:

```text
s = a + alpha * h_grid
train alpha = {0, 1/4, 1/2, 3/4}
held alpha  = {1/8, 3/8, 5/8, 7/8}.
```

These diagnose grid-local phase generalization and are never used as
common-source coarse/fine pairs.

Resolution-transfer cases use one physical position defined from native `h0`:

```text
s = a + alpha * h0
```

and regenerate exact cell averages at that same `s` on every grid. Fine exact
averages must conservatively restrict to coarse exact averages before any model
commutator is interpreted.

Structured arrays use shape `[ny,nx]` and flatten with `x` fastest, matching the
maintained restriction and resolution-pathway utilities. The A1 tests compare
the new nested restriction directly with `restrict_uniform_cell_averages` so a
self-consistent but permuted layout cannot pass silently.

### Frozen scalar PCNO fitting contract

The P0 fitting contract is:

- Core `PCNO`, not `PCNOEuler2DResidual`.
- Input features: raw physical `x`, raw physical `y`, uniform quadrature
  density `1/|Omega|=2`, and current scalar cell average.
- Scalar increment output with identity target normalization.
- Physical Fourier periods `(1,1/2)`, `k_max=8`, one measure.
- Four PCNO blocks with width 128, decoder width 128, GELU, and current branch
  order.
- No boundary side path or mask.

Capacity arms use one shared initialization and functional branch masks:

```text
S, P, D, S+P, S+D, P+D, and S+P+D (full).
```

These unequal-parameter arms are descriptive capacity comparisons. They are
not substitutes for the later parameter-matched no-gradient causal study.

The primary future optimizer is full-batch AdamW with `lr=1e-3`,
`betas=(0.9,0.999)`, `eps=1e-8`, zero weight decay, 20,000 updates, and cosine
decay to `1e-5`; seeds are `1701`, `1702`, and `1703`. Selection uses the final
registered update, not a held-resolution result. If every full-model seed
fails fixed-grid fit, the only preregistered optimization rescue is a
full-batch strong-Wolfe L-BFGS continuation from each AdamW endpoint, learning
rate 1, history 20, and at most 2,000 iterations. The present authorization
covers only seed `1701` under the AdamW protocol; it does not authorize the
L-BFGS rescue.

### First fitting decision and visualization contract

The first fitting run is full PCNO only. A standalone no-gradient companion is
not added here: it would combine fewer active parameters with removal of the
differential pathway and would not answer the registered causal question. The
no-gradient question remains P2 and requires the full three-arm,
parameter-matched, three-seed design after P0/P1 analysis and a separate costed
A3 approval.

Working run label: `W26-L2-P0-FULL-S1701`. It trains on the 48 native-grid cases
formed by four families, three anchors, and four registered train phases, and
evaluates the final update on the corresponding 48 held-phase cases. The
physical-volume objective is the equally weighted mean over cases of squared
relative increment L2. Held phases never select a checkpoint; the final
registered AdamW update is evaluated.

The run must retain parseable history, final case metrics, predictions, a final
checkpoint, source/configuration/population digests, and an output manifest.
Two reproducible visualization packages are required in both vector PDF and
300-DPI PNG:

1. train/held mean and worst-case relative-increment-L2 curves versus optimizer
   update, with the fixed-grid training threshold shown; and
2. common-scale target/predicted next-state profiles, transverse prediction
   range, and increment-error profiles for representative train/held step and
   pulse cases, with fixed physical front bands marked.

Figures supplement the case metrics. They never determine checkpoint selection
or turn high-frequency energy alone into a ripple claim.

## P0 Metrics

All errors use the model-predicted increment against the exact increment. Front
geometry, extrema, and TV use the reconstructed next state
`current + predicted_increment` against the exact next-state cell averages.

Required case-level metrics are:

- volume-weighted relative increment L2;
- jump-normalized excess above the exact next-state maximum and below its
  minimum;
- outside the union of fixed `1/32` physical front bands, full two-dimensional
  positive and negative error mass per `x`-connected smooth component, with
  no transverse averaging before the oscillatory mass
  `2*min(M_positive,M_negative)` and signed bias reported separately;
- front midpoint, 10--90 thickness, plateau strength, number of midpoint
  crossings, and ambiguity, always compared with the same measurement of the
  exact cell-average reference;
- jump-normalized physical-volume RMS outside the front bands;
- signed `TV(predicted_next)-TV(exact_next)`, positive TV excess, and TV deficit;
- separate increment-integral and next-state-integral errors using physical
  cell volumes, plus pulse signed-integral closure for pulse cases (null for
  other families); and
- physical-wavenumber error spectrum on modes common to all compared grids.

High-frequency energy is descriptive. It is never equated with ripple when a
front-strength loss or TV deficit indicates blur.

A missing 10%, 50%, or 90% crossing is serialized as a null measurement with
`front_gate_valid=false`; it is an automatic structure-gate failure, never NaN
and never a silently selected zero error. Multiple crossings are likewise
flagged invalid even though the nearest crossing is retained descriptively.

### Fixed-grid existence gate

One full-PCNO seed establishes fixed-grid fitting capacity only if every
registered discontinuous case satisfies:

- relative increment L2 `<=1e-3`;
- normalized overshoot and undershoot `<=0.01`;
- outside-band oscillatory mass `<=1e-3`;
- midpoint error `<=h0/4`;
- strength error `<=0.02`;
- 10--90 thickness excess `<=h0/2`;
- smooth-region error `<=1e-3`;
- positive TV excess and TV deficit each `<=0.02`; and
- increment and next-state integral errors each `<=0.005`.

Failure of one optimizer/seed is `capacity not demonstrated under this
protocol`, not mathematical incapacity. A bounded full-configuration capacity
failure requires all three seeds, both preregistered optimization protocols,
finite gradients/budget completion, and a nodal-lookup plumbing control that
fits the exact discrete target.

### Phase and resolution gates

For each discontinuous family and held grid:

- at least 10/12 held phases meet every absolute structure guard;
- median relative increment L2 is at most
  `max(1.25*native-held-phase error, 5e-3)`;
- extrema `<=0.02`, midpoint error `<=h/2`, strength error `<=0.05`, thickness
  excess `<=h`, and both integral errors `<=0.01`; and
- conservative coarse/fine prediction commutator `<=0.05` of target-increment
  RMS after subtracting the measured analytic restriction floor.

Training-phase success with held-phase failure on one grid is a phase failure.
Phase-controlled native success with systematic fixed-physical nested-grid
failure is grid inconsistency. Neither is called strict Gibbs.

## P1 Differential-Path Contract

### Analytic and frozen stages

For each block and grid, retain:

```text
g  = raw least-squares gradient
a0 = native repeated two-hop self-plus-neighbor average of g
a1 = graph-ball average at the grid-local two-hop radius
a2 = graph-ball average at fixed physical radius R=1/32
z  = gw1 * selected aggregate
q  = Softsign(z)
d  = gw2(q)
```

At native `64x32`, the local and fixed radii are identical and their operator
result must be reused rather than recomputed as an independent arm.

The bounded A1 implementation separates two levels deliberately. On exact
scalar synthetic fields it implements `g`, `a0`, `a1`, `a2`, and the fixed
mollified diagnostic on every grid. Its random-toy PCNO replay implements the
native learned taps `g`, `a0`, `z`, `q`, and `d`, plus the decoded ablation
response. Learned-hidden `a1`/`a2` alternatives and adjacent-grid hidden-stage
commutators remain part of the registered P1 design but require a later
checkpoint-bearing authorization; A1 produces no result for them.

Off-path diagnostics are componentwise `(hx*gx, hy*gy)` and a fixed-physical
mollified derivative. They diagnose scaling; they do not alter the model.

For raw and native fixed-hop gradients, expected distributional behavior is:

- peak magnitude proportional to `1/h`;
- physical support proportional to `h`;
- support measured in cells approximately stable; and
- signed normal integral divided by jump strength approximately stable.

That expected peak growth is not an inconsistency. Fixed-radius `a2`
intentionally has fixed physical support and therefore a different peak limit.

### Learned-stage instrumentation

For `g`, `a`, `z`, `q`, and `d`, record maximum, q99, volume-weighted
quantiles, L1 strength, signed front-normal integral, minimum physical band
containing 50%/90% of magnitude, support in physical units and cells, and
conservative adjacent-grid commutators.

Analytic per-front signed strength uses the nearest-front Voronoi partition in
physical `x`, so the full discrete distributional mass is retained even when a
coarse off-face stencil extends beyond the separate `1/32` error band.

Softsign saturation is defined exactly as:

```text
abs(Softsign(z)) >= 0.9, equivalently abs(z) >= 9.
```

Report global and front-band volume-channel fractions, feature-energy fractions,
and volume-channel-weighted quantiles of the derivative `(1+abs(z))^-2`. The A1
trace accepts an explicit physical front-band mask; it never infers a front from
the prediction.

The downstream quantity is named
`decoded_differential_ablation_response`:

```text
native_full_output
  - output_after_zeroing_only_this_layer_differential_branch
    and recomputing all downstream layers.
```

It is a finite nonlinear counterfactual, not an additive contribution. The
native zero/no-intervention output is the already computed native output; it is
not generated by a duplicate replay and relabeled as zero.

### First-inconsistent-stage rule

A stage is implicated only when its upstream stage satisfies the analytic or
common-grid contract, while the candidate stage changes by at least 20% on
both adjacent grid pairs in at least 10/12 held phases and the change survives
to the decoded ablation response.

- Raw signed strength/support failure: least-squares/discretization.
- Raw passes, native aggregate fails, and fixed support repairs: aggregation
  physical-support inconsistency.
- Nondimensional aggregate is stable but `gw1*a` crosses operating ranges:
  learned-scale mismatch.
- A new saturation-fraction change of at least 0.20 accompanied by a
  post-Softsign commutator: Softsign saturation.
- Pointwise `gw2` can amplify/reweight an existing spatial mismatch; it is not
  labeled the creator of a new spatial commutator.
- Stable branch output followed by a decoded response change of at least 20%:
  activation/decoder interaction.

## Zero-Inclusive Frozen Filter Interface

The A1 implementation validates only the mathematical frozen filter and bypass
invariants. It does not apply a filter to a checkpoint.

At the one layer selected by P1, future arms are:

| Arm | One registered intervention |
| --- | --- |
| F0 | True bypass returning the already computed native tensor/output. |
| R | Replace native two-hop aggregation by fixed-radius graph-ball support. |
| S-grad | Replace native aggregation by a smooth physical-wavenumber filter of raw gradient. |
| H-grad | Same replacement with a hard cutoff; negative control only. |
| S-pre | Filter the summed branch preactivation once before block activation/residual addition. |

No arms are stacked with each other, the native graph average, or D080's
decoded-proposal correction.

The frozen spectral diagnostic uses an orthonormal DCT-II on structured cell
centers, corresponding to an even/cosine nonperiodic boundary extension, and
multiplies coefficient energy by physical cell volume for the discrete physical
Parseval measure. Cross-grid summaries explicitly truncate the rectangular mode
axes to the coarsest compared `(nx,ny)` before scoring. Physical frequency is
measured in cycles per unit length. DC gain is exactly one. The smooth transfer
is:

```text
sigma(q) = 1                                      for q <= 8
           0.5*(1+cos(pi*(q-8)/8))                for 8 < q < 16
           0                                      for q >= 16.
```

The hard negative control is `1_{q<=8}`. F0 bypasses the transform bitwise.
Only physical modes common to the compared grids enter a cross-grid score.

A frozen arm is eligible for later consideration only if it reduces the
diagnosed defect by at least 20% in at least 10/12 synthetic phase cases and,
after a separate A2 authorization, at least 5/6 named open cases, while every
extremum/front/thickness/strength/smooth/TV/integral control is at most 1.05 of
zero and no absolute gate fails. A frozen pass is not recurrent promotion.

## Conditional P2--P4 Matrices

### P2 matched gradient training

The owner has selected P2 as the first training matrix after the current P0 fit
and its final analysis. P2 may not start until the P0 metric report and
visualizations, P1 instrumentation, and the frozen branch-factorial report below
have been reviewed. Their outcomes route interpretation and lock the primary
defect metric before training; they do not silently substitute another training
matrix. An exact costed A3 launch remains separately authorized.

Before retraining, use the finalized full-PCNO checkpoint for two complementary
frozen analyses of the spectral (`S`), pointwise-linear (`P`), and differential
(`D`) branches:

1. branch-local same-hidden replay at every block, so each removed branch sees
   the native hidden input; and
2. the complete end-to-end `2^3` mask cube
   `(000, 100, 010, 001, 110, 101, 011, 111)` for one-step prediction and
   with bit order `(S,P,D)` and the selected masks applied at every block.

The pre-training P0 report uses this cube only on the registered one-step
train/held-phase and nested-grid populations. After P2 training, repeat it on
each arm's one-step evaluation and registered recurrent rollout. No synthetic
recurrent result is invented merely to unblock P2.

Report factorial main effects and interactions rather than assigning the full
change to one branch. Branch summation, activation, residual updates, decoding,
and recurrent state changes make the effects non-additive. Same-hidden replay is
an acute local response; end-to-end masking is acute network dependence; neither
is evidence that an independently trained restricted architecture will adapt in
the same way.

The training matrix has exactly three arms:

1. current full PCNO;
2. functional no-gradient PCNO, with every differential branch gain fixed to
   zero from initialization and its differential parameters excluded from the
   optimizer; and
3. parameter-matched local replacement
   `V2(Softsign(alpha*A_h^2*V1(x)))`, with two bias-free `C->C` pointwise maps
   and one scalar, exactly `2*C^2+1` active parameters per replaced block.

Use three paired seeds, identical common tensors, data order, presentations,
optimizer steps, selection, recurrence, and boundary policy. A causal win
requires at least 2/3 paired-seed wins, at least 5% median target improvement,
and every case-level control at most 1.05.

The primary defect metric is locked after P0/P1 review but before any P2
optimization: outside-band oscillatory mass if a registered ripple defect is
present; otherwise the first failed phase/front/grid metric. Required secondary
channels are normalized over/undershoot, front position/strength/thickness,
smooth-region error, TV excess, pulse integral, physical spectrum, state error,
admissibility, and boundary leakage. Reducing only high-frequency energy, or
blurring/weakening the front, is not a ripple improvement.

Required visualizations use identical physical windows, front bands, and color
limits across arms: target/prediction/error maps, front-normal cuts, physical
spectra, paired-seed metric plots, and a frozen `2^3` factorial-effect plot.
Training spectral-only, pointwise-only, gradient-only, or all seven nonzero
branch subsets is not part of P2; the frozen cube supplies that attribution
without multiplying the training matrix.

Interpret the matched results as follows:

- no-gradient improves ripple and preserves every control, while the matched
  replacement does not: the trained differential pathway is specifically
  implicated under this contract;
- no-gradient and matched replacement both improve: removed differential
  behavior or active local capacity/support is implicated, not gradient
  semantics uniquely;
- no-gradient worsens front/state metrics and the replacement recovers them:
  the differential branch carries useful capacity;
- frozen deletion helps but trained no-gradient does not: the defect is an acute
  checkpoint reliance that the retrained network fails to replace; and
- one-step masks are clean but recurrent masks regenerate the defect: call this
  recurrent regeneration only after the registered temporal onset and control
  gates pass.

A Sobolev/gradient-supervision study is not bundled into this ablation. It gains
causal headroom only if the differential branch carries useful front/state
capacity but the full model retains a measured defect that plain L2
underweights. Any later cross-grid gradient loss must register its physical
scaling or fixed-physical mollification; an unscaled raw shock-gradient L2 term
would inherit the expected `O(h^-1)` peak and is not resolution neutral.

This matrix is three arms by three paired seeds, hence nine production trainings.
A reduced-budget smoke may test plumbing only and cannot select an arm or support
a scientific claim. Filtering, loss/exposure, boundary, and REALM training remain
behind P2.

### P4 loss

Launch only if normalized L2 passes while a target front band occupying at most
15% of physical volume contains at least 50% of squared error and a registered
front metric fails.

- L0: physical-volume normalized L2.
- L1: L0 plus `0.1` times normalized target-derived fixed-front-band error.
- L2: L0 plus `0.1*TV(error)/(TV(target)+eps)`.

No prediction-derived front selector or raw-TV-of-prediction loss is used.

### P4 exposure

- E0: clean teacher forcing.
- E-IID: zero-mean IID perturbation, projected to zero componentwise volume
  integral and RMS matched to measured incoming error.
- E-structured: conservative exact subcell phase shift or equal-volume neighbor
  transfer along a current-state front normal, with matched RMS.
- E-rollout: detached two-step endogenous rollout with matched transition
  presentations and optimization budget.

Structured exposure requires measured support/covariance agreement. Endogenous
exposure requires at least 50% of the targeted defect to be propagated/input
error. Existing D053 evidence does not meet that trigger for smooth high-pass
ripple, which was predominantly fresh. Earlier D015/D022 exposure contracts
must be audited for nonduplication before any A3 request.

## Global Variant Cap

At most two method variants can become eligible for later training:

1. fixed-physical differential support if P1 and frozen R gates pass; and
2. exactly one defect-routed alternative: a smooth intermediate filter,
   front-aware loss, structured corruption, or short endogenous exposure.

There is no combined variant without a new preregistration. Hard cutoff and IID
noise remain negative controls. Full/no-gradient/parameter-matched arms are
causal controls rather than deployment variants.

## A1 Files, Invocation, And Checks

The bounded A1 surface consists of exactly:

1. this preregistration;
2. `utility/time_dependent_no/pcno_shock_representation.py`;
3. `scripts/time_dependent_no/analyze_pcno_shock_representation.py`; and
4. `tests/time_dependent_no/test_pcno_shock_representation.py`.

The utility has two maintained callers: the script and focused tests. No
`pcno/`, Euler wrapper, D070/D073 utility, active tracker, weekly plan, README,
or package initializer is changed.

The A1 CLI is deliberately no-write and refuses to run without `--dry-run`:

```powershell
python scripts/time_dependent_no/analyze_pcno_shock_representation.py --dry-run
```

It prints schema
`pcno_shock_representation_w26_l2_p0_p1_a1_dry_run_v1`, marks
`scientific_interpretation_allowed=false`, allocates no D-series ID, performs no
optimization, and writes no artifact.

Required CPU checks are:

- exact overlap/log-cosh/sine averages, residual identity, and known integrals;
- exact common-physical restriction closure on both adjacent grid pairs;
- explicit separation of grid-local phase and common-source cases;
- FV coordinate-gradient recovery and raw `1/h` scaling with stable `h*peak`,
  support in cells, signed strength, and negligible transverse gradient;
- local/fixed graph-ball identity on the native grid;
- bitwise zero-filter bypass, DC preservation, and smooth/hard distinction;
- exact-prediction metric zeros and distinct crafted bias, alternating ripple,
  overshoot, phase shift, and blur responses;
- exact raw/aggregation/pre-Softsign/post-Softsign/`gw2` algebra;
- decoded differential-ablation equality with the maintained branch-disable
  replay; and
- model-state immutability and JSON-finite no-write CLI output.

The A1 cost class is low CPU. Production fitting, named open-checkpoint screens,
and all training matrices stop at the M2 human review gate.

## Authorized P0 Implementation Surface And Checks

The owner subsequently authorized the first full-PCNO P0 fit and its
visualizations. The additional implementation surface is exactly:

1. `scripts/time_dependent_no/fit_pcno_shock_representation.py`;
2. `scripts/time_dependent_no/visualize_pcno_shock_overfit.py`; and
3. `tests/time_dependent_no/test_fit_pcno_shock_representation.py`.

The fit script is the maintained invocation path; the separate visualizer is a
second real caller of the retained run schema and can regenerate figures from
the parseable artifacts without retraining. No core `pcno/` file, package
initializer, Euler wrapper, active tracker, weekly plan, D070/D073 utility, or
unrelated experiment file is changed for P0.

Before a production launch, the CPU checks must confirm the exact production
population and model contract, nonzero target norms, exact-prediction gates,
finite full-model gradients, resumable atomic artifacts, final-only selection,
and parseable PDF/PNG visualization outputs. A local smoke run is explicitly
`science_result=false`. The production cost class is one moderate personal-GPU
fit; AutoDL is not selected or authorized by this registration.

## Authorized P1 Checkpoint Surface And Checks

The 2026-08-11 authorization adds exactly:

1. this status and P1 contract update;
2. `scripts/time_dependent_no/analyze_pcno_shock_pathways.py`; and
3. `tests/time_dependent_no/test_pcno_shock_pathways.py`.

P1 reuses the maintained P0 builder, structure metrics, exact branch replay,
and differential-stage trace. It changes no `pcno/` source, reusable utility,
Euler wrapper, dataset loader, active tracker, or unrelated experiment. Its
production input is only the completed P0 checkpoint and its hash-verified
synthetic artifact bundle.

The isolated execution snapshot also retains the unmodified
`pcno_euler2d.py` and `cpg_mesh_contract.py` transitive imports required by the
maintained branch tracer. They are hashed dependencies, not P1 edit surfaces.

The frozen populations are exact:

- all 96 native-grid train/held cases for the global `2^3` mask cube and all
  12 layer-local deletions;
- all 48 common-physical held cases on each registered nested grid for the
  global cube and prediction commutators; and
- all 24 held discontinuities plus the anchor-1 held smooth controls on each
  grid for learned differential-stage taps and adjacent-grid commutators.

A separate scalar current-field control uses both common-physical positions
and grid-local held phases on every grid. The latter is the contract for
expected `1/h` peak scaling; common-physical cases diagnose phase/resolution
behavior. Neither a changing raw peak nor a large hidden-tensor commutator alone
allocates a mechanism label.

The maintained invocation path is:

```powershell
python scripts/time_dependent_no/analyze_pcno_shock_pathways.py `
  --p0-run-dir artifacts/time_dependent_no/w26_l2_p0_full_s1701 `
  --output-dir artifacts/time_dependent_no/w26_l2_p1_frozen_s1701 `
  --device cuda --batch-size 8
```

Required CPU smoke checks cover exact mask-bit semantics, common-physical
restriction closure, mask-`111` checkpoint closure, all local deletion paths,
differential replay closure, model-state immutability, JSON-finite artifacts,
and six parseable PDF/PNG figures. Production additionally refuses CPU, source
drift, an incomplete/non-scientific P0 input, hash mismatch, population drift,
or a nonempty output directory. P1 is a low-to-moderate frozen GPU analysis; it
does not authorize optimization or turn an acute deletion into a trained
architecture result.

Compatibility amendment, frozen before any P1 metric artifact was written: the
first GPU invocation stopped after the global-mask forward passes because a
batch-size-dependent CUDA replay differed from the stored P0 array by
`6.58e-5`. A second diagnostic stop isolated a `1.14e-5` difference between
separate same-batch native and no-op CUDA scatter executions. Therefore mask
`111` follows the already registered true-bypass rule and reuses the native
direct output exactly. An independently recomputed no-op replay is retained
only as a reported numerical floor with maximum-absolute gate `1e-4`; current
direct batch versus stored P0 also uses `1e-4`, and differential manual replay
uses `1e-4`. No branch metric, pathway summary, or visualization from either
stopped invocation was available or used to choose these tolerances.

## P0/P1 Result And Claim Gate

The completed P0/P1 evidence is bounded as follows:

- full PCNO passed all 24 native training discontinuities;
- exact held guards passed 9/12 step and 9/12 pulse cases, with all six failures
  at phase `0.875`;
- at that phase, mean relative increment L2 was `0.01353`, step oscillatory mass
  was `0.001341`, pulse overshoot was `0.02532`, and mean positive TV excess was
  `0.08965`, while mean maximum front-position error was `1.51e-4`, front-
  strength error `3.77e-4`, thickness excess `0.00236`, and increment-integral
  error `2.92e-4`;
- common-physical off-grid guards passed 0/12 step and 0/12 pulse cases on both
  `32x16` and `128x64`; full-mask discontinuous mean L2 was `0.1171` coarse,
  `0.00444` native, and `0.1432` fine;
- deleting spectral, pointwise, or differential branches globally increased
  the phase-`0.875` L2 from `0.01353` to `0.8085`, `0.4425`, and `0.3493`,
  respectively, with train/smooth controls worsening by hundreds to thousands
  of times; all 12 layer-local deletions also worsened the primary and controls;
  and
- grid-local scalar current gradients obeyed exact `1/h` peak scaling, stable
  `h*peak`, L1 strength, support in cells, and signed strength through the
  native two-hop average. Learned layer-0 peak ratios remained approximately
  two through raw gradient, aggregation, and fixed `gw1`; Softsign compressed
  those ratios, while strict `abs(z)>=9` saturation remained zero except for a
  `0.00188` front-band volume-channel fraction at layer 3 on the fine grid.

Therefore this supports fixed-grid full-PCNO capacity plus a native subcell-
phase failure and severe finite-grid transfer failure. It does not support a
classical Gibbs claim, mathematical capacity failure, a unique spectral or
gradient ripple cause, recurrent regeneration, or a general operator-
convergence claim. The first architectural concern is the absence of
grid-aware nondimensionalization before a nonlinear Softsign operating range;
the current frozen evidence does not causally separate fixed learned scaling
from downstream nonlinear adaptation. The internal result-to-claim verdict is
`partial`; external Codex review is pending because unpublished results were not
sent outside the repository authorization boundary.

## Authorized P2 Matched Training Surface

P2 adds exactly:

1. `scripts/time_dependent_no/train_pcno_gradient_ablation.py`; and
2. `tests/time_dependent_no/test_train_pcno_gradient_ablation.py`.

The three arms are unchanged from the earlier registration:

1. maintained full PCNO;
2. functional no-gradient PCNO, where initialized differential parameters are
   retained frozen, excluded from the optimizer, and return exact zero; and
3. `V2(Softsign(alpha*A_h^2*V1(x)))`, with two bias-free `C->C` maps and one
   scalar, exactly `2*C^2+1` active parameters per block.

Seeds are exactly `1701`, `1702`, and `1703`. Each run uses the same 48-case
full-batch order, 20,000 updates, optimizer, cosine schedule, final-update
selection, native evaluation, and common-physical `32x16`/`128x64` held
evaluation. Shared non-gradient initialization is hashed per paired seed.

The primary is locked before optimization as mean native phase-`0.875`
discontinuous relative increment L2. Mandatory secondaries remain step
oscillatory mass, pulse over/undershoot, TV excess/deficit, front position,
strength and thickness, smooth-region error, both integrals, physical spectrum,
and both off-grid directions. A win requires at least 2/3 paired wins, at least
5% median improvement, and every case-level control at most 1.05. Lower
high-frequency fraction caused by blur or low-frequency underfit is not a win.

All nine runs are authorized on the two available GPU resources. The measured
P0 personal-GPU runtime gives an upper estimate near 13 GPU-hours if every arm
ran at that speed; splitting the queue reduces wall time, while actual AutoDL
speed is recorded rather than assumed. P2 performs no recurrent or sealed
evaluation. Post-training recurrent attribution remains coordinated with
D073-B and W26-L5.

## Decision Tree After A1

```text
CPU contracts fail
  -> fix the harness only; no scientific interpretation.

CPU contracts pass
  -> M2 human review.
     |
     +-- approve P0 fitting?
     |    +-- fixed grid fails only by seed/schedule -> optimization sensitivity
     |    +-- fixed grid passes, held phase fails -> phase failure
     |    +-- phase passes, fixed-physical grids fail -> grid inconsistency
     |    +-- all pass -> no registered static representation failure
     |
     +-- complete P1 and frozen S/P/D attribution
          +-- first inconsistent stage satisfies P1 rule -> route one mechanism
          +-- no stage satisfies rule -> retain bounded no-inconsistency result
          +-- in either case, prepare the selected P2 costed A3 request
```

Recurrent regeneration is a later result only when a one-step-clean prediction
develops new outside-band oscillatory mass under recurrence and the D053-style
fresh/propagated decomposition identifies its source. Static high-frequency
content alone cannot establish it.

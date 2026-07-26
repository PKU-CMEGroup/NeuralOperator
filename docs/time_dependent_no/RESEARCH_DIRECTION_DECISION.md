# Research Direction Decision: Frozen Four-Line Closeout

Updated: 2026-07-26
Status: Accepted normative source

## Authority And Change Protocol

This file is the sole normative source for scientific scope, claim language,
sealed populations, and experiment authorization on branch `time-dependent-no`.
[`HANDOFF.md`](HANDOFF.md) is a derived operational snapshot,
[`MECHANISTIC_DIAGNOSTIC_TRACKER.md`](MECHANISTIC_DIAGNOSTIC_TRACKER.md) is the
evidence ledger, and [`README.md`](README.md) is non-authoritative onboarding and
the sole active-code inventory. If they disagree, this file governs.

A result, artifact, TODO, available GPU, or open evidence gap never authorizes a
run. Changing the claim register or reopening research requires an explicit
human direction decision. Before such a change, bind the proposal to tracker run
IDs and immutable artifacts, separate observation from mechanism and claim,
predeclare the population, intervention, metric field, aggregation, threshold,
direction, seed scope, and cost, and preserve every existing non-claim. Update
derived documents only after this decision is changed.

Private paths, hosts, credentials, and machine-specific dataset locations remain
in ignored local context and are not part of this record.

## Current Verdict And Causal Thesis

The completed 2026-07-23 result-to-claim gate is `partial`, with high
confidence. The intended method claim - that this campaign identifies and
validates a generally improved, shock-stable geometry-aware neural-operator
solver - is not supported. The bounded scientific claim is supported: a serious
residual PCNO is useful on one validated dynamic finite-volume family, and
matched interventions separate several distinct failure mechanisms within that
frozen scope.

The report must use one causal thesis. On this shock-bearing finite-volume
family, the residual PCNO has a useful global map, but its error is not governed
by one removable branch. Propagated shock-supported state error, freshly
generated smooth high-pass error, geometry-conditioned transfer,
discrete-decoder conditioning, and unstable front identity are distinct coupled
obstacles. Do not reduce this to generic instability, optimizer failure, or
Gibbs ringing.

## Final Four-Line Status

| Line | Status | Frozen conclusion | Non-claim |
| --- | --- | --- | --- |
| 1: large-step flow maps | Closed | Under the frozen 1D contracts, the useful stride depends on horizon and metric: a harder one-call map can win after fewer recurrent compositions. | No universal optimal stride, learned CFL limit, timestep-conditioned solver, ripple cure, or native-grid transfer follows. |
| 2: CPGNet validity and mechanism | Closed | Corrected 1D controls support message reach rather than width alone; interface coordinates act as functional controls rather than verified physical traces. Legal-boundary training helps without closing the oracle-boundary gap. | No paper-faithful reproduction, implicit scheme, physical-interface, conservation, or PCNO-transfer claim follows. |
| 3: geometry-aware 2D rollout | Closed without method promotion | D044 is a useful one-seed residual-PCNO baseline on the frozen Mach-1.1 shock-vortex family. D060 lowers recurrent state error, but the targeted ripple/front conjunction and every registered stabilization route fail. | No generally improved or ripple-stable solver, facewise-conservative learned method, seed-robust result, strength-OOD result, or test result follows. |
| 4: latent forecasting and assimilation | Stopped before forecast training | Smooth-decoder and fixed-Haar capacity tests isolate a representation-capacity limitation; discontinuous regularity helps but still misses the reconstruction/front hierarchy. | No latent transition, autonomous recurrence, geometry transfer, neural-operator, or data-assimilation claim was tested. |

## Unified Claim Register

The RC-01--RC-15 entries below are the complete allowed claim set. Evidence
populations are part of each claim; they cannot be silently pooled or widened.

### RC-01. Larger learned macro steps can beat repeated smaller steps

- **Verdict:** `yes, bounded`.
- **Evidence and population:** D031-D036 and D038-D039 form the Line-1 closeout.
  Their primary population is 64 frozen 1D test cases through H96; D034 uses a
  fixed 16-case timing subset. Across three stride-8 seeds, all 192 H96 rollouts
  complete, and the seed-level mean fixed-scale conservative relative-L2 range
  is `0.01767--0.01821`. D033 same-state decomposition and D035/D038 frozen
  evolution show worse truth-state one-call error but eventual on-policy gains.
  D060 adds 24 position-OOD validation cases through physical H60 and the
  six-case D013 subset, with one 2D seed and sealed strength OOD.
- **Allowed wording:** The operating point trades one-call approximation
  difficulty against recurrence count. Lower endpoint state error can coexist
  with a rougher early defect and worse front or high-pass metrics.
- **Missing evidence and non-claim:** The winner varies with horizon and metric.
  No universal best stride, learned CFL limit, timestep transfer, ripple cure,
  or mesh invariance follows. Independent 2D seeds, strength OOD, other
  geometries, and longer uncensored horizons remain missing. Do not pool the
  full-split D031 regime with the earlier midscale D027 regime.

### RC-02. CPGNet stability depends materially on message reach

- **Verdict:** `yes, bounded`.
- **Evidence and population:** On the corrected 1D 384/64/64 split, h128/mp12
  gives one-step relative L2 `0.0334` with 34/64 raw completions, while mp28 gives
  `0.00614` with 64/64; the mp12/h193 and mp28/h85 parameter controls support hop
  coverage rather than width alone. Physical projection does not preserve the
  learned interface update. On the same 20 release-bundle 2D trajectories, the
  arithmetic mean of per-trajectory normal-node primitive RMSE
  `[rho,v1,v2,pres]` is
  `[0.177032,0.073653,0.082204,0.319612]` with released next-reference boundary
  injection, `[0.563023,0.263983,0.242344,1.137027]` for the frozen checkpoint
  under causal nodal boundaries, and `[0.368161,0.157947,0.133379,0.753202]`
  after one legal-boundary training seed.
- **Allowed wording:** Hop coverage is the supported 1D mechanism, not width
  alone. The interface coordinates are functional, nonphysical control
  coordinates; causal boundary training helps without closing the
  legal-versus-oracle gap.
- **Missing evidence and non-claim:** Paper dataset/checkpoint/evaluator identity,
  exact DG boundary replay, validation-based checkpoint selection, seed
  robustness, and grouped geometry holdout remain absent. This is not a
  paper-faithful mechanism, implicit scheme, physical trace, conservation, or
  architecture-transfer result.

### RC-03. D044 is a useful fixed-family 2D macro-map baseline

- **Verdict:** `yes, bounded`.
- **Evidence and population:** One seed completes 24/24 raw H60 position-OOD
  validation rollouts at mean physical-volume state error `0.00834190` and beats
  persistence plus both train-manifold controls on every paired case.
- **Allowed wording:** A conservative-coordinate residual PCNO gives a strong
  raw-rollout baseline on the frozen Mach-1.1 shock-vortex family.
- **Missing evidence and non-claim:** Strength OOD and test are sealed; seed
  uncertainty and other geometry or dataset families are missing. D044 bundles
  coordinates, residual prediction, primitive training noise, and rollout-based
  selection. Its recurrence is not conservative by construction.

### RC-04. D060 improves state error but is not a promoted solver

- **Verdict:** `partial for state; no for the joint claim`.
- **Evidence and population:** On the same 24 position-OOD validation cases,
  D060 completes 24/24 and lowers mean H60 state error to `0.00729369`, or
  `0.87434x` D044, winning every paired case. Direct frame 2 is `1.03246x` two
  composed D044 calls. The conjunctive mechanism fails: six-case high-pass RMS
  is `0.00548836` versus `0.00535199` (`1.02548x`), all-24 endpoint high-pass
  energy is worse in every case, and front-centroid distance is `0.0260282`
  versus `0.0181368` (`1.43511x`). The centroid ratio splits to `2.74982x` for
  `y00` and `0.74853x` for `y08`.
- **Allowed wording:** Halving recurrent calls improves accumulated state L2
  while leaving the fresh high-pass source and some front errors unresolved.
- **Missing evidence and non-claim:** There is one seed, no strength OOD, no
  extra stride, and a failed physical conjunction. Timing remains descriptive.
  Do not call D060 uniformly better, ripple-stable, or promoted.

### RC-05. Serious-PCNO ripple formation is composite

- **Verdict:** Within their respective frozen contracts, the pure
  Fourier/Gibbs, pure pointwise, pure differential, boundary-only, and
  strong-cancellation explanations are `no`.
- **Evidence and population:** D041 evaluates all 20 bump holdouts for boundary
  and failure routing; D042 uses five call-1 bump holdouts, and D043 uses two
  bump trajectories at call 10. D044/D013 and D052-D053 use six position-OOD
  validation trajectories from the separate dynamic-FV family over early,
  middle, and late calls. On the dynamic checkpoint the Fourier Gram matrix is
  well-conditioned and the spectral response is the smoothest branch; D043
  finds no strong paired cancellation on the bump checkpoint, and no repeated
  branch attenuation passes a complete D052 gate.
- **Allowed wording:** Within each frozen evidence regime, the spectral branch
  is globally coupled and comparatively smooth, and no isolated branch explains
  the failure. The mechanism is composite; the bump and dynamic populations
  are not pooled.
- **Missing evidence and non-claim:** Exact layerwise causation and any residual
  aliasing contribution remain unresolved. The dynamic causal result has one
  checkpoint and one model seed on one family; adaptive local bases, different
  global maps, and cross-family transfer remain untested.

### RC-06. Long-horizon state error and smooth ripple have distinct channels

- **Verdict:** `yes, bounded`.
- **Evidence and population:** D053 is a zero-training exact decomposition on six
  D052 validation trajectories at calls 1--60. At calls 30/60, median full-field
  propagated shares are `0.88995/0.89043`, while smooth-high-pass propagated
  shares are `0.15619/0.30733`; the additive identities close to numerical
  precision.
- **Allowed wording:** On this frozen cohort, shock-supported state error is
  carried mainly through recurrence, while smooth-region high-pass error is
  predominantly regenerated by the one-call map.
- **Missing evidence and non-claim:** There is one checkpoint, one shock-bearing
  family, and no matched smooth control. This is not a universal
  neural-operator law.

### RC-07. Native face-target fit is insufficient for a viable conservative update

- **Verdict:** `no for the exact tested objective`.
- **Evidence and population:** D048 trains on four immutable pairs for 3,200
  updates. Fifteen epochs pass every native face-space gate at roughly 10%
  face-field error, but 0/4 decoded states are admissible and relative
  divergence amplification is `1001--1581x`. D049 independently verifies the
  frequency dependence of the discrete decoder.
- **Allowed wording:** Target-space error must be checked after the physical
  discrete decoder. The tested canonical shared-face value objective is
  stopped.
- **Missing evidence and non-claim:** No serious divergence-conditioned,
  projected, or full reference-impulse model was run. This result does not
  reject every flux representation.

### RC-08. A sparse legal locator does not imply a safe correction

- **Verdict:** `yes` for the locator; `no` for the exact correction.
- **Evidence and population:** D055 and D056 are zero-training tests on six
  validation trajectories at calls 1/10/30/60, with strength OOD sealed. D055
  captures median `91.224%/80.212%` of fresh smooth-high-pass energy at calls
  30/60 using `17.304%/16.866%` interior support and passes every gate. D056
  preserves balance to `3.61508e-17`, but state-error reduction is only
  `7.185%/7.010%`, high-pass reduction is `23.763%/-3.592%`, and the
  joint-nonworse count falls from six cases at call 30 to two at call 60.
- **Allowed wording:** The proposal is a credible legal troubled-region locator,
  but the bounded balanced post-hoc correction lacks safe late-horizon
  headroom.
- **Missing evidence and non-claim:** This rejects the registered support-limited
  detail head, not every conservative local correction or an end-to-end global
  map change.

### RC-09. One static joint-objective direction is not geometry-robust

- **Verdict:** `no safe late-OOD continuation for the exact directions`.
- **Evidence and population:** D057 starts from four training pairs; D058 uses
  all 84 training trajectories and evaluates six position-OOD validation cases
  at calls 30/60. Both take zero optimizer steps. D058 closes its Frank-Wolfe
  gap to `5.55e-17`; its eight training-task cosines span
  `0.563997--0.824737`, and joint validation transfer is 6/6 at call 30 but
  3/6 at call 60. At call 60, clean-state cosines are negative for every `y00`
  case and positive for every `y08` case.
- **Allowed wording:** The objectives are jointly descendable on training data,
  but the frozen full-model direction is geometry-conditioned and unsafe as a
  late-OOD continuation.
- **Missing evidence and non-claim:** Finite-step optimization, state-dependent
  weighting, and other task parameterizations remain untested. This is not a
  generic optimizer-failure claim.

### RC-10. Scalar multirate state blending does not combine the parent gains

- **Verdict:** `no for the exact registered blend`.
- **Evidence and population:** D061 combines frozen D044 and D060 trajectories
  for six D013 cases at matched frames 2/10/30/60, producing 24 zero-training
  rows. At H60, the truth-informed scalar oracle reduces state and high-pass
  errors by only `8.14%/10.10%` versus required `10%/20%`; it is jointly
  nonworse in 0/6 and passes anti-smearing in 0/6 cases.
- **Allowed wording:** Complementary parent metrics are not composable by scalar
  state averaging. Target-free disagreement localization remains descriptive.
- **Missing evidence and non-claim:** No shared-backbone row is authorized. The
  failure does not reject every learned, state-dependent multirate method.

### RC-11. The independent-row front chart is not a stable representation

- **Verdict:** `no for the exact chart`.
- **Evidence and population:** D062 is a target-informed zero-training oracle on
  12 rows from six validation trajectories at H30/H60. Its conservative remap
  preserves row/component totals to `2.69e-15`. At H60, front-curve MAE improves
  by `62.82%`, while median state error worsens by `41.88%`, high-pass RMS
  worsens by `828.08%`, and joint nonworse is 0/6. Displacement reaches 10.54
  cells and adjacent-row jumps reach 10.51 cells.
- **Allowed wording:** Comparable pressure-jump branches exchange argmax
  identity; improving the scalar front coordinate can shear the 2D field despite
  exact conservation and admissibility.
- **Missing evidence and non-claim:** This rejects the exact independent-row
  argmax chart. A connected multi-front atlas, level set, or topology-aware
  chart is untested and would require a new zero-training identity and closure
  oracle.

### RC-12. The tested latent family is decoder-capacity limited

- **Verdict:** `yes, bounded`; the route is stopped before transition training.
- **Evidence and population:** L4A-002 is an 800-update matched autoencoder
  smoke, L4A-003 fits each held-out code directly, and L4A-004 is a zero-training
  fixed-Haar capacity test. The common population is 10 position-OOD validation
  states from two trajectories; strength OOD and test remain sealed. L4A-004
  changes mean L2 from `0.005430` to `0.004461`, strength ratio from `0.539` to
  `0.830`, thickness from `2.672x` to `1.554x`, and IoU from `0.390` to `0.500`.
  Only 1/10 states reaches the `0.0021` L2 gate; mean strength remains 17.0%
  weak and thickness 55.4% broad, and it is worse than L4A-003 in L2 on 10/10.
- **Allowed wording:** Discontinuous decoder regularity materially affects front
  fidelity at fixed state size, but the tested fixed chart is insufficient.
- **Missing evidence and non-claim:** No transition, recurrence, geometry
  transfer, or assimilation was tested. Any new chart must pass reconstruction
  and closure before forecast training.

### RC-13. Physical conservation claims are data-contract conditional

- **Verdict:** `yes` for benchmark diagnostics; `no` for a learned conservative
  solver.
- **Evidence and population:** D037 audits the Mach-1.1 family's 250x100 finite
  volumes, oriented faces, normals, boundary accounting, and cumulative
  accepted-substep impulses. D044 predicts state residuals, D060 changes the
  temporal stride, and D062 only post-processes frozen states.
- **Allowed wording:** Physical totals and boundary exchange are valid outcome
  diagnostics on this audited benchmark; D062 preserves its post-hoc row totals
  exactly.
- **Missing evidence and non-claim:** D044/D060 recurrence and D062 are not
  predicted-flux conservative neural solvers. Equal-node bump sums and
  unvalidated bump quadrature do not support physical conservation claims.
  The boundary-matched 2D SharpClaw reference is state-only; D037's
  cycle-impulse field is discretization-specific.

### RC-14. A general shock-stable geometry-aware neural operator is not established

- **Verdict:** `no`.
- **Evidence and population:** All serious 2D method evidence uses one seed and
  one dynamic family; every registered stabilization misses a predeclared joint
  gate. Strength OOD and test remain sealed.
- **Allowed wording:** No general solver or state-of-the-art claim is allowed.
- **Missing evidence and non-claim:** Multi-seed confirmation, new geometries,
  test OOD, and a promoted method are absent.

### RC-15. The completed campaign supports a bounded mechanistic contribution

- **Verdict:** `partial, high confidence`.
- **Evidence and population:** The validated benchmark and baseline, exact
  source decomposition, discrete-decoder audit, systematic geometry strata,
  and failed capacity oracles form one coherent evidence-graded account.
- **Allowed wording:** The contribution maps what works, what fails, and why
  several plausible shock remedies fail necessary conditions within the frozen
  scope.
- **Missing evidence and non-claim:** A positive replacement architecture is
  absent, and discontinuity specificity is not isolated by a matched smooth
  control.

## Limitations And Sealed Evidence

These limitations remain visible and do not become implicit follow-up
permissions:

1. D044 bundles coordinates, residual prediction, primitive training noise, and
   rollout-based selection; no matched 2D training ablation isolates them.
2. D044 and D060 are single-seed serious runs. Casewise repetition is not seed
   robustness.
3. Selected epochs 44 and 34 and their loss curves are provenance, not a
   controlled sample-efficiency comparison.
4. Branch traces and proposal scores are not learned hidden-state evidence for
   stable front topology, future failure, or correction sign.
5. Every current 2D cohort is shock-bearing; no matched smooth problem isolates
   discontinuity specificity.
6. Strength-OOD and test splits remain sealed and cannot strengthen a claim
   after method selection failed.
7. CPG release identity and exact DG boundary replay remain unresolved; the
   supersonic-bump HDF5 does not provide the validated finite-volume geometry
   needed for physical conservation.
8. Validation-only oracles do not license test, strength-OOD, or promoted-method
   claims.
9. Evidence populations with different splits, scales, or controls cannot be
   pooled without a new decision. In particular, D031 and the earlier D027
   regime remain distinct.
10. Reference trajectories and boundary impulses are solver- and
    discretization-specific. The boundary-matched 2D SharpClaw reference records
    states only and exports no face impulses; D037's cycle-impulse field is
    discretization-specific.

## Authorized Report-Only Queue

The only active queue uses frozen artifacts:

- plot D044 and D060 state, high-pass, front, and geometry-stratum curves;
- plot D053 propagated-versus-fresh shares beside the D052 branch
  falsification;
- place D048/D049 decoder amplification beside D062 front-identity failure;
- tabulate objective, presentations, selected epoch, intervention, population,
  evidence grade, and non-claim for every result; and
- retain failed implementation attempts as provenance, not scientific trials.

These actions execute no checkpoint, open no sealed split, change no threshold,
and consume no GPU time.

## Reopening Gate

No learned or oracle experiment is authorized. The next human decision is
exactly one of:

1. close and report the campaign under the bounded claim register; or
2. separately preregister a zero-training capacity oracle for a representation
   with connected front identity and transverse regularity.

A reopening proposal must define the exact current inputs, topology and identity
rule, conservative remap or closure rule, frozen validation population with no
test access, artifact-bound thresholds, and total cost. It must pass
reconstruction and closure before any encoder, transition, recurrent, or filter
training. Any other direction requires an explicit change to this decision.
Available compute does not relax the gate.

## Mechanism And Method Discipline

Before changing a model, distinguish at least six causal classes:

1. supervised fit or centering;
2. objective and discrete-decoder conditioning;
3. recurrent objective mismatch;
4. representation-manifold failure;
5. geometry-conditioned generalization; and
6. implementation or metric binding.

Every proposal must predeclare its mechanism variable, expected diagnostic
curve, smallest matched control, systematic strata, exact threshold field,
aggregation, population and direction, strongest alternative explanation, seed
scope, and full cost. Put optimized loss beside raw physical rollout, front,
admissibility, and budget metrics. A named gate, attention map, latent, branch,
or physical coordinate needs a readout or intervention showing that the model
uses it.

Apply the following interpretation rules:

- target-space fit is insufficient until evaluated after the physical decoder;
- fewer calls or lower endpoint state error does not imply lower roughness;
- conservation is necessary in some formulations but does not identify a
  correct front representation;
- a coordinate needs an identity and closure contract, not only a lower loss;
- a necessary-condition failure rejects the exact registered branch, not every
  method in its broad family; and
- reported lower high-pass energy is not improvement when it is caused by shock
  blurring or lost wave strength.

Keep one primary solver path and derive redundant outputs where possible. Add
only one dense-supervision family at a time. Prefer hard constraints through
shared face quantities, validated geometry, and explicit boundary accounting;
do not enforce them with whole-sample mean corrections, blanket total-variation
penalties, or independent state and flux heads that can disagree.

## Macro-Flux Contract

The solver-facing quantity is the `cumulative conservative face impulse`

`I_f[n:n+K] = sum_(k in accepted substeps) dt_k Fbar_(f,k)`,

where `Fbar_(f,k)` is the owner-oriented normal numerical flux density averaged
with the solver's actual quadrature for accepted substep `k`. For RK methods,
use the weighted stage flux; for ADER, use the returned space-time-predictor
quadrature flux. A rejected retry contributes zero, and the final accepted
retry or first-order fallback contributes exactly once.

Use `time-averaged face flux` only for `I_f / Delta T_sample`, where
`Delta T_sample = sum_k dt_k` is the actual saved-interval duration. Do not call
this a `mean field`.

The validated 1D artifact stores conservative flux density times time, oriented
owner-to-neighbor on interior faces and outward on boundaries. The discrete
decoder owns face signs and the `1 / dx` factor. In higher dimensions, state
whether face measure is included in `I` or in `D_h`; never apply it twice.

Before training, require serialized closure

`U_(n+K) - U_n = -D_h I[n:n+K]`,

where `D_h` includes the declared orientation, geometry, mass, and boundary
convention. Do not add separate boundary terms when boundary-face impulses are
already in `I`. The frozen 1D gate is maximum float64 pre-serialization closure
below `1e-12`, followed by float32 decoded closure at `rtol=3e-5` and
`atol=3e-6`. A dtype or scale change requires a replacement tolerance declared
before generation.

Temporal averaging can suppress fast temporal oscillations but does not make a
longer-horizon operator intrinsically easier. Larger horizons expand the domain
of dependence, increase unresolved-state sensitivity, and introduce kinks when
shocks cross faces or solver branches change. Treat horizon difficulty as an
empirical axis.

## State-Coordinate Contract

The completed 1D coordinate matrix held the conservative increment fixed and
separated input from loss coordinates:

1. primitive input with primitive loss;
2. conservative input with primitive loss;
3. primitive input with conservative loss;
4. conservative input with conservative loss; and
5. conservative plus primitive features with joint loss only after the matched
   four-way comparison.

The provisional 1D contract is conservative input, conservative loss,
conservative recurrence, and fixed physical scaling. It is benchmark-supported,
not a universal claim that primitive or characteristic features are inferior.
When this contract is used, keep the recurrent state conservative and derive
primitive features without hidden floors. Use fixed physical nondimensional
scales as the main normalization; empirical standardization is only a declared
control.

Every run record must bind `input_coordinates`, `recurrent_coordinates`,
`predicted_quantity`, `loss_coordinates`, target normalization, timestep,
geometry scaling, and whether primitive conversion used a floor. Do not call a
conservative decoder trained through primitive loss a conservative-space model.
D044 bundles several design choices, so its 2D result does not separately
validate this coordinate choice.

## Shared Evaluation And Artifact Contract

Every line must:

- treat one-step fit as an entry gate, never a rollout proxy;
- evaluate raw recurrence without clipping, primitive floors, future-reference
  boundaries, a nearly frozen limiter, or undeclared decode-reencode;
- report admissible completion before oscillation, shock position, strength,
  thickness, smooth-region error, conservation and boundary leakage;
- distinguish mixed-prefix, completed-case, and common-endpoint statistics;
- use grouped geometry or parameter OOD splits where possible;
- select checkpoints by a declared rollout rule; and
- report physical horizon, learned calls, characteristic travel, effective CFL,
  batch-1 latency, and amortized throughput together.

Retain

`C_eff = max_x(|u| + c) Delta t / h`

as a descriptive case property, not a universal neural stability condition. On
uniform 1D grids, a local architecture may compare physical reach
`r_phys(Delta t)` with architecture propagation radius `r_arch` through

`chi = r_phys(Delta t) / r_arch`.

For arbitrary hyperbolic inputs, `chi > 1` exposes a worst-case causal
obstruction. Do not reduce an irregular 2D graph to one scalar; report
node-, direction-, state-, and time-dependent predecessor coverage in a common
physical metric. A domain-global operator remains global at every timestep.
There is no universal learned CFL claim.

For 2D artifacts, preserve `predicteds`, `targets`, `pos`, `edges`, and
`node_type` when available, plus current or initial state, physical time and
`Delta t`, trajectory/sample mapping, parameters or Mach number, boundary mode,
valid length and failure cause, coordinate convention, checkpoint and
configuration digests, intervention flags, and the exact diagnostic weights.
Physical conservation additionally requires a validated mesh-to-graph map,
volumes, face measures, normals, oriented face connectivity, and boundary
accounting. A reference-flux claim additionally requires cumulative reference
impulses.

Latent artifacts must also bind state size or token contract, hierarchical
reconstruction, closure defect, decoder conditioning, and whether recurrence
includes decode-reencode. Preserve metric definitions, aggregation level,
population and split IDs, seed, normalization, selected checkpoint, source and
artifact digests, complete cost, and any post-hoc oracle access.

## Forecast-Before-Assimilation Boundary

Data assimilation cannot be used to hide an open-loop defect. A latent forecast
must first pass all of these jointly on an authorized population:

- hierarchical reconstruction with admissibility, front position, strength,
  thickness, overshoot, contact behavior, smooth-region error, and budgets;
- approximate Markov closure, including future ambiguity and at most one
  preregistered bounded-history control when ambiguity is material;
- decoder conditioning, including physical perturbation amplification,
  front-position sensitivity, and resolution scaling;
- raw autonomous recurrence without hidden correction or truth reset;
- direct-versus-composed timestep behavior beside truth error and semigroup
  defect;
- transfer across declared front location, strength, interaction, resolution,
  and geometry strata; and
- matched utility against the frozen physical-space baseline, including every
  encode/decode, completion, accuracy, physical diagnostic, state size, latency,
  and throughput.

Lower L2 obtained through broader or weaker fronts fails. Stable latent values
without a physically valid decoded rollout fail. No current candidate passes
these requirements, so latent transition and assimilation remain unauthorized.

A later assimilation-readiness decision must also verify dynamically or
observation-relevant perturbations, for example

`J_D J_G J_E delta U approximately equals J_(Phi_dt) delta U`,

or a declared ensemble analogue, and must report discarded directions, decoded
covariance and growth fidelity, latent observability, analysis-like off-manifold
states, innovation behavior, reparameterization sensitivity, and a localization
contract for spatial latents.

Only after an explicit later authorization may a frozen accepted forecast use
`y = H(D(a)) + eta` with a standard EnKF or LETKF. The required analysis
comparison holds observation operator, ensemble budget, initialization, and
cadence fixed across physical and latent coordinates. A separate end-to-end
comparison may change forecast systems but must be labeled accordingly. Both
must include credible numerical and persistence controls, observation-blackout
continuation, spread-error and innovation calibration, shock uncertainty,
admissibility, conservation, ripple, and online cost. Open-loop evidence remains
separate.

## Retained Records

- [`README.md`](README.md): onboarding and exact active-code inventory.
- [`HANDOFF.md`](HANDOFF.md): derived operational snapshot.
- [`MECHANISTIC_DIAGNOSTIC_TRACKER.md`](MECHANISTIC_DIAGNOSTIC_TRACKER.md):
  immutable run ledger and detailed contracts.
- [`SECTION_1_2_CORRECTED_BASELINES.md`](SECTION_1_2_CORRECTED_BASELINES.md):
  corrected 1D labels, final results, and frozen-run analyzer contract.
- [`CPG_EULER_DATASET_CONTRACT.md`](CPG_EULER_DATASET_CONTRACT.md): audited CPG
  dataset and evaluation contract.
- [`BUMP_300_DATASET_AUDIT.md`](BUMP_300_DATASET_AUDIT.md): supersonic-bump
  data limitations.
- [`CPGGNSPDES_REFERENCE_AUDIT.md`](CPGGNSPDES_REFERENCE_AUDIT.md): reference
  implementation audit.

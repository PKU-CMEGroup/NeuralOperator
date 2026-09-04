# B5 Supersonic-Bump Solver-Free Transfer Preregistration

Date: 2026-09-05

Experiment identity: `B5_BUMP_SOLVER_FREE_TRANSFER_20260905A`

Status: owner-selected secondary-PDE transfer study. Stage 0A/B local contract
and synthetic implementation work are authorized. Stage 0C and Stage 1 are a
frozen design envelope, not an executable training contract: they require an
identity/algorithm amendment before launch. A dataset-scale or remote run also
requires a named compute-resource decision and a launch receipt. The historical
20-trajectory test population remains sealed.

## Question And Role

This study asks one bounded question:

> Do framework-guided corrective mechanisms help a fixed PCNO when the clean
> solution family is geometry-dependent and advection-dominated, so the
> particular fixed-mesh global path projector used for NACA0012 does not
> transfer directly?

NACA0012 and the bump have deliberately different roles. NACA demonstrates a
special case in which a nearly linear low-rank trajectory makes a fixed PCA
projection unusually effective. B5 tests whether useful corrective information
extends beyond that special geometry. It is not a second primary case, a
prospective test of C2, or a search for a universally best intervention.

Advection of sharp structures can have low physical or nonlinear intrinsic
dimension while having poor linear reducibility in fixed Eulerian coordinates.
The study therefore distinguishes intrinsic structure from approximation by a
single linear subspace.

## Claim Boundary

B5 may support a secondary cross-problem statement about intervention value,
problem-dependent correction geometry, rollout no-harm, and transported-
structure preservation. It cannot establish:

- a literal or smooth data manifold;
- a universal failure of PCA or a universal advantage of any learned method;
- trusted dynamics, normal response, or solver-relative defect at displaced
  graph states;
- physical conservation from the reconstructed graph weights;
- prospective C2 generalization; or
- historical-test performance.

The bump graph bundle has 300 geometry-dependent meshes with different node
counts. Raw global PCA and kNN in a shared state vector are therefore not
defined without a new remapping representation. Per-trajectory temporal POD is
an offline diagnostic only. A basis fitted to the evaluated trajectory is not
a deployable corrector.

## Frozen Population And Representation

- Dataset identity: the audited 300-trajectory supersonic-bump training bundle.
- Split: the immutable D094 field-blind split, with 256 training trajectories
  and 44 open-development trajectories.
- Test: the separate historical 20-trajectory population remains unopened.
- State: first-order conservative variables
  `[rho, rho_v1, rho_v2, energy]` on each native graph.
- Normalization: fit on all 256 training trajectories using the maintained
  proxy-mass-weighted PCNO normalization at stride one.
- Physical caveat: reconstructed node weights are diagnostic proxies, not
  audited finite-volume measures.

No development state is used to fit a normalizer, POD basis, recovery scale,
or corrector. Development trajectories are used only after the corresponding
stage is frozen.

## Stage 0A: Identity And Leakage Audit

Before any state analysis or training, verify:

1. the exact split-manifest file and canonical payload hashes;
2. the prepared-shard manifest hash and all referenced train/development keys;
3. the exact clean-checkpoint, normalizer, model, boundary, and archived-source
   identities used for the phenotype replay;
4. a fresh B5 source/evaluator manifest rather than reuse of a D094 identity;
5. absence of historical-test access; and
6. refusal on any missing or mismatched source, checkpoint, split, or packet.

Any failure stops scientific execution. A historical packet may remain parent
evidence without being declared compatible with the mutable checkout.

Stage 0B requires only items 1--2 and the fresh B5 analysis-source manifest.
Stage 0C additionally requires items 3--4 and is blocked until an artifact
receipt records the exact checkpoint, normalizer, archived source snapshot,
boundary contract, and the exact selection/outside-selection key identities.

## Stage 0B: Train-Only Linear-Reducibility Audit

Use the frozen `n=32` nested training subset. Analyze every trajectory on its
own native mesh under two declared metrics:

1. uniform-node Euclidean distance in normalized PCNO state coordinates; and
2. the existing reconstructed proxy-mass metric as a robustness view.

Let `z_t=(u_t-mu)/s` use the four global state means and scales fitted on all
256 training trajectories. The POD population is frames `0--78`, exactly the
states used as inputs to the 79 clean one-step training pairs; frame 79 is a
target, not a supervised map input. For each trajectory and each metric,
normalize finite nonnegative node weights with positive total to sum to one
and give the four state components equal weight. The oracle mean is the
temporal mean of frames `0--78`; the prefix mean is the temporal mean of frames
`0--39`. POD eigenvalues are the descending eigenvalues
of the corresponding centered time--time Gram matrix. Negative roundoff is
clipped to zero, and numerical rank uses tolerance
`max(lambda_1 * 1e-12, eps * trace(G))`. Explained-variance rank is the smallest
available rank whose cumulative eigenvalue fraction reaches the threshold.
Participation rank is `(sum lambda)^2/sum lambda^2`; entropy rank is
`exp(-sum p log p)` over positive normalized eigenvalues.

For prefix reconstruction, project normalized frames `40--78` onto the prefix
basis using the same weighted inner product and add back the prefix mean. A
requested rank above numerical rank is reported with its smaller effective
rank. The primary relative reconstruction error is the weighted Frobenius norm
of reconstruction error divided by the weighted Frobenius norm of the future
states centered at the prefix mean. Physical diagnostics decode reconstructed
normalized conservative states with the fixed train normalizer and use the
maintained pressure-gradient shock, positivity, and boundary-leakage routines.
Smooth/high-gradient errors use the target pressure-gradient top-decile mask
and normalized conservative coordinates. All formulas and parameters are
written into the result packet.

For every trajectory compute:

- an oracle centered POD spectrum over the 79 model-input frames `0--78`;
- a prefix POD fitted only on frames `0--39` and evaluated on frames `40--78`;
- ranks reaching `90%`, `99%`, and `99.9%` explained variance;
- participation and entropy effective ranks;
- rank `2`, `4`, `7`, `16`, and `32` future reconstruction errors; and
- density/pressure admissibility, pressure-shock centroid, strength, thickness,
  boundary error, and smooth/high-gradient error proxies after reconstruction.

The rank-7 comparison is fixed because seven area-weighted components capture
`99.9%` of the NACA training variance; it is not selected from bump outcomes.
The intended linear-reducibility contrast passes only if at least 24 of the 32
bump trajectories require more than seven oracle uniform-metric components for
`99.9%` variance. Otherwise B5 is reframed as a no-harm control or stopped; the
rank is not increased post hoc to rescue the contrast.

High explained variance does not override a transported-structure failure.
Rank-truncated states that improve an aggregate distance while materially
moving, weakening, thickening, or invalidating the shock are recorded as a
linear-projection failure mode, not a successful correction.

## Stage 0C: Clean Phenotype Replay

Reproduce the retained `n=256`, seed `20260718`, step-`16384` clean PCNO using
its exact archived source and checkpoint. Keep the original 16 selection and
28 outside-selection open-development roles distinct. A separate current-
evaluator derivative, if produced, receives a fresh source identity.

The phenotype gate requires all 28 outside-selection trajectories to complete
79 calls, mean clean H1 error at most `0.02`, mean `H79/H1` at least `4`, and
either mean H79 error at least `0.08` or a registered admissibility failure in
at least `20%` of trajectories. If the gate fails, B5 remains a no-harm control
and no intervention training follows.

## Stage 1 Design Envelope: One-Seed Solver-Free Pilot

This section fixes the scientific comparison and forbids opportunistic arm
changes, but it is not yet an executable training contract. Before Stage 1,
an amendment must freeze the architecture and feature contract, deployed
boundary map, loss/node population, optimizer and complete learning-rate
schedule, presentation stream, EMA and exposure schedule, all keyed RNG rules,
evaluation formulas and thresholds, and checkpoint-selection rule. The
amendment must also state whether each reported transition deploys online or
EMA weights. No Stage 1 training may start without it.

Stage 1 uses seed `20260718`, the same 256 training trajectories, 16 selection
trajectories, 28 outside-selection trajectories, fixed 79-call horizon, and
`16384` optimizer updates for each end-to-end learned transition. No arm is a
fine-tuning continuation of the retained clean endpoint.

The intended first pilot contains exactly four deployed systems:

1. `CLEAN`: a fresh fixed-PCNO control;
2. `IID_RECOVERY`: a 0.5 clean / 0.5 perturbed objective whose admissible
   primitive Gaussian scale is matched, using train-only data, to the clean
   model's one-prefix normalized-RMS error; the target is the stored clean
   successor;
3. `CURRICULUM_EMA_PREFIX_K13`: detached EMA-generated prefixes with requested
   depth in `{1,2,3}`, a frozen warmup/exposure ramp, and the stored clean
   future as target; and
4. `PREFIX_ERROR_CORRECTOR_K13`: freeze the exact causally closed `CLEAN` map
   `F`, train a separate PCNO residual correction
   `C(x)=x+delta_theta(x)` with zero-initialized `delta_theta` on actual
   one-to-three-step `F` proposals paired with the corresponding stored clean
   state, mix in identity pairs `C(u)=u`, and deploy `C o F` with a second
   boundary closure and corrected-state feedback.

The first three models start from identical overlapping parameter bytes within
the seed. The operational corrector adds a second model call and receives
separate parameter, training-compute, and inference-cost accounting.

A variable-mesh `PCNO_PDEREFINER_K3_VPRED` port is a predeclared conditional
extension, not part of Stage 1. It may be activated only before confirmation
outcomes if Stage 1 passes and a source-faithful cross-problem learned-sampler
comparison remains necessary. It must retain its distinct four-call
conditional-residual-sampling semantics and must not be mislabeled as
`C o F`.

## Signed Predictions

1. `IID_RECOVERY` lowers small-scale ripple/roughness and modestly improves
   late error, but can broaden or weaken transported shocks.
2. `CURRICULUM_EMA_PREFIX_K13` lowers error on model-prefix inputs and improves
   late rollout more than IID recovery, with possible clean-H1 cost.
3. `PREFIX_ERROR_CORRECTOR_K13` gives the best H79 and rollout-AUC among the
   solver-free pilot arms because its training inputs match the coherent error
   directions generated by the frozen base map; its identity term limits clean-
   path damage.
4. If the explicit corrector loses to curriculum exposure, the proposed
   error-geometry advantage is falsified rather than explained away.
5. Any aggregate-L2 improvement accompanied by material shock/front or
   admissibility regression is smoothing, not successful correction.

## Metrics And Promotion Rule

Report paired per-trajectory H1/H20/H40/H60/H79 error and rollout AUC, clean
one-step no-harm, case wins, pressure-shock location/strength/thickness,
smooth-region high-pass error, boundary leakage, density/pressure/internal-
energy admissibility, correction dose, identity error, calls, parameters,
training time, and inference time. Proxy totals, if retained, remain explicitly
nonphysical.

The numeric thresholds below are the intended decision rule. The Stage 1
amendment must bind exact AUC, relative-improvement, case-win, admissibility,
and material shock-regression formulas before they are executable. Promote to
seeds `20260812` and `20260813` only if at least one intervention:

- improves both paired median H79 and rollout AUC by at least `10%`;
- wins on at least `60%` of the fixed 28 outside-selection trajectories;
- worsens clean H1 by at most `5%`; and
- causes no material registered shock/front or admissibility regression.

Otherwise close B5 as a preregistered null/no-harm boundary result. Do not add
methods, alter the noise rule, raise the POD rank, change the population, or
open the historical test to rescue the result.

## Paper Budget

B5 receives at most one reducibility panel and one compact transfer table in
the main paper. Detailed spectra, per-case results, and implementation contracts
belong in the appendix or repository documentation. NACA remains the primary
PDE case study.

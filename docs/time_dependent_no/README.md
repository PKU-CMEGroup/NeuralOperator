# Time-Dependent Neural Operators

This branch contains the summer 2026 time-dependent neural-operator work inside
the PKU-CME `NeuralOperator` codebase. The 1D solver-facing representation
program is now a mature evidence base rather than the active center of the
project. Work is organized into four coordinated lines: a bounded 1D
large-step flow-map closeout; a validity and mechanism audit of 1D/2D CPGNet;
a flagship geometry-aware 2D rollout program; and a cross-cutting line that
tests discontinuity-aware latent forecasting first and latent data assimilation
only after the raw forecast gates pass. The accepted hierarchy, shared
terminology, gates, and stop rules are in
`RESEARCH_DIRECTION_DECISION.md`.

Read-only Phase-0 kickoff prompts live in `docs/time_dependent_no/prompts/`:

- `LINE_1_LARGE_STEP_FLOW_MAP_PROMPT.md`
- `LINE_2_CPGNET_MECHANISM_PROMPT.md`
- `LINE_3_2D_ROLLOUT_PROMPT.md`
- `LINE_4_LATENT_FORECAST_ASSIMILATION_PROMPT.md`

Each prompt requests a bounded proposal before any code change or experiment.

The tracked code is intentionally lean. Historical one-off probes and scaffold scripts were removed from the active tree after their conclusions were recorded in `MECHANISTIC_DIAGNOSTIC_TRACKER.md`; recover them from git history if an old result must be reproduced exactly. Commit `729091b` is the exact pre-cleanup provenance checkpoint.

## Active Code

Reusable branch utilities live in `utility/time_dependent_no/`:

- `fv.py`: PDE-agnostic finite-volume geometry, gather/scatter, and conservative update helpers.
- `euler1d.py`: 1D Euler primitive/conservative conversion, fluxes, geometry, and batch helpers.
- `euler1d_data.py`: collaborator-compatible 1D Euler dataset loading and batching.
- `euler1d_models.py`: FNO target heads and the corrected solver-level 1D CPGNet adaptation.
- `euler1d_targets.py`: state/residual, flux, and interface target adapters.
- `euler2d.py`: CPG HDF5 schema inspection, primitive/conservative conversion, node-type helpers, and graph-frame materialization.
- `euler2d_synthetic.py`: deterministic CPG-style synthetic fixture for CPU tests.
- `euler2d_metrics.py`: rollout, positivity, conservation, shock-proxy, boundary, and compact-summary diagnostics.
- `euler2d_fixture.py`: end-to-end no-model fixture diagnostics.
- `cpg_release.py`: pinned-release provenance, rollout-metric, boundary-distance, and result-integrity helpers.
- `cpg_mesh_contract.py`: released graph-to-raw-mesh identity and legal nodal boundary metadata.
- `cpg_reach.py`: audited dependency-radius and characteristic-envelope helpers.
- `pcno_euler2d.py`: sharded full-resolution trajectory access, fixed conservative normalization, and the residual PCNO wrapper.
- `pcno_fv_geometry.py`: validated structured finite-volume geometry and oriented face incidence for shock--vortex PCNO shards.
- `pcno_ripple_diagnostics.py`: graph-spectral, Fourier-basis, branch-response, and raw-admissibility diagnostics for the residual PCNO.
- `shock_vortex_fv.py`: canonical shock--vortex WENO5-HLLC-SSPRK3 reference evolution, structured physical geometry, conservative restriction, and accepted-substep face impulses.
- `shock_vortex_coarse_cfd.py`: exact nested cell-average remapping and synchronized state-only coarse-CFD rollout with native boundary accounting.
- `shock_vortex_metrics.py`: shared D044/D051 physical-total, shock, vortex, front, and smooth-region metrics.
- `shock_vortex_family.py`: frozen dynamic-benchmark family contract, manifest validation, and trajectory loading.
- `fv_impulse_diagnostics.py`: oriented face-to-cell decoder gains, graph-frequency divergence modes, weighted divergence-active/cycle/boundary decomposition, algebraic boundary allocation, and canonical minimum-norm reconstruction for validated finite-volume impulse fields.
- `errors.py`: small NumPy error helpers used by diagnostics.

Active command-line entry points live in `scripts/time_dependent_no/`:

```bash
python scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py --help
python scripts/time_dependent_no/train_euler1d_target_ladder.py --help
python scripts/time_dependent_no/analyze_euler1d_target_ladder.py --help
python scripts/time_dependent_no/evaluate_euler1d_flow_map_frontier.py --help
python scripts/time_dependent_no/benchmark_euler1d_flow_map_runtime.py --help
python scripts/time_dependent_no/generate_euler1d_flow_map_ood.py --help
python scripts/time_dependent_no/visualize_euler1d_flow_map_frontier.py --help
python scripts/time_dependent_no/audit_cpg_release_provenance.py --help
python scripts/time_dependent_no/audit_cpg_mesh_contract.py --help
python scripts/time_dependent_no/evaluate_cpg_release.py --help
python scripts/time_dependent_no/diagnose_cpg_characteristic_reach.py --help
python scripts/time_dependent_no/train_cpg_legal_boundary.py --help
python scripts/time_dependent_no/prepare_pcno_euler2d_shards.py --help
python scripts/time_dependent_no/train_pcno_euler2d_residual.py --help
python scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py --help
python scripts/time_dependent_no/diagnose_pcno_euler2d_ripples.py --help
python scripts/time_dependent_no/generate_euler2d_shock_vortex_reference.py --help
python scripts/time_dependent_no/generate_euler2d_shock_vortex_pyro_reference.py --help
python scripts/time_dependent_no/generate_euler2d_shock_vortex_sharpclaw_reference.py --help
python scripts/time_dependent_no/audit_euler2d_shock_vortex_convergence.py --help
python scripts/time_dependent_no/build_euler2d_shock_vortex_family.py --help
python scripts/time_dependent_no/generate_euler2d_shock_vortex_family_case.py --help
python scripts/time_dependent_no/prepare_pcno_shock_vortex_shards.py --help
python scripts/time_dependent_no/evaluate_pcno_shock_vortex_baseline.py --help
python scripts/time_dependent_no/benchmark_pcno_shock_vortex_coarse_cfd.py --help
python scripts/time_dependent_no/visualize_official_cpg_rollout.py --help
python scripts/time_dependent_no/rollout_pcno_preprocessed.py --help
```

The CPG audit, evaluation, legal-training, and reach entry points preserve the
frozen Line-2 closeout contract. They are reproducibility surfaces, not an
active authorization for another CPG training or mechanism sweep.

`rollout_pcno_preprocessed.py` is the corrected PCNO evaluation path. It assumes the collaborator-compatible preprocessing contract: HDF5 trajectories are converted to per-trajectory arrays, reconstructed into the PCNO Euler `.npz` format, and then rolled out. The retired raw-HDF5 PCNO adapter path should not be used for checkpoint evaluation.

Tests live in `tests/time_dependent_no/`.

Current execution follows a mechanism-first rule across all four lines. Before
any learned row, classify the failure as fit/centering, objective or decoder
conditioning, recurrence, representation, geometry generalization, or an
implementation artifact; predeclare the expected plot, matched control, case
strata, promotion/kill decision, seed scope, and cost. Bind every threshold to
an exact stored field and aggregation. Lines 1 and 2 are closed, D060 closes
Line 3's temporal-step test without promotion, and Line 4 remains stopped
before serious representation training or data assimilation.

## Current Dataset State

The copied dataset folder label was `forward_300`, but extracted files identify the supersonic bump dataset:

- 300 train trajectories and 20 test trajectories;
- 80 HDF5 time steps per trajectory;
- roughly 19k to 23k nodes per trajectory;
- expected CPG keys are present;
- extracted test cases contain `Bump.jl`, `Bump.msh`, `Bump.inp`, `Mach.txt`, `params.txt`, and VTU snapshots.

See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md` and `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md` for stable schema facts. Exact AutoDL paths remain in ignored local context.

## Current Diagnostic Readout

The available local-bundle CPGNet run learns accurate teacher-forced one-step
updates but fails in autoregressive rollout, mainly around shock-local phase,
shape, amplitude, and stability rather than simple one-step underfitting. It is
not yet an apples-to-apples paper reproduction. The active PCNO baseline is now
the selected clean 19,155,720-parameter conservative-residual model, not the
historical positive-primitive checkpoint. On all 20 held-out bump trajectories,
the frozen checkpoint has mean one-step proxy relative L2 `0.006663`, completes
20/20 H20 rollouts at mean error `0.022257`, and completes 17/20 raw H79
rollouts with mean survival `0.905696`. The reconstructed weights used by this
evaluator remain diagnostic proxies, not physical control-volume measures.

Line 2 now has a frozen legal-boundary closeout on the same 20 release-bundle
test trajectories and archived closeout evaluator. Mean-per-trajectory
normal-node RMSE
for `[rho, v1, v2, pres]` is `[0.177032, 0.073653, 0.082204, 0.319612]` with
released next-reference boundary injection, `[0.563023, 0.263983, 0.242344,
1.137027]` when the public checkpoint is evaluated with causal nodal
boundaries, and `[0.368161, 0.157947, 0.133379, 0.753202]` after one
legal-boundary training run. Retraining improves all four variables on 19/20
trajectories and the q0.90 shock-centroid and Chamfer metrics on 20/20; every
rollout remains finite and positive on normal nodes. The oracle row is still
better by factors of roughly `1.6--2.4`, and the trained run's worst
post-policy boundary error is larger than the frozen sensitivity row.

This is release-bundle evidence, not a paper-table reproduction or a general
learned-solver result. Paper dataset/checkpoint and compared-baseline evaluator
identity remain unresolved; the legal policy is causal nodal boundary handling,
not exact DG replay; and the trained result is one seed, final-epoch selected,
without a validation split or geometry/parameter holdout. Later source,
mesh-stencil, recurrent-outflow, and termination-accounting checks are
prospective hardening and were not used to regenerate the reported arrays.
Line 2 is therefore closed as a bounded validity result. Do not use the released oracle-assisted
row as a fair autonomous baseline, and do not launch another CPG training or
decoder sweep from this evidence.

Frozen D013 plus the full held-out D041 evaluation classify the dominant
observed failure as shock/front-seeded recurrent amplification, not a pure
teacher-forced spectral-fit failure, pure Gibbs mechanism, or pointwise-only
defect. Smooth-region graph high-pass error grows rapidly after the first
shock-local error. A frozen `0.75` pointwise-tail gain worsens one-step error by
`3.74x`, lowers H79 completion from 17/20 to 13/20, and worsens paired H79 state
error by `7.21x`; the earlier paired-response selector was sensitivity evidence,
not causal attribution. D043's official-checkpoint paired error-response audit
subsequently falsifies strong measured branch cancellation: all 16
layer/case/proxy fractions are at most `0.00430` and their pooled median is
`-0.0196`. The pointwise response remains roughest in all four layers, but the
full learned branch composition must be retained; do not infer a safe branch
gain from that sensitivity. D042 also rejects one fixed 289-column
geometry-only compact span: it halves smooth-region high-pass projection error
but barely changes residual RMSE and fails every decoded anti-smearing/
admissibility row. This does not reject adaptive or learned local bases.

The full Line-2 nodal boundary policy improves completion to 19/20 but fails
state and anti-smearing gates. Fixing only exact freestream inflow leaves H20
metrics effectively unchanged, extends both inflow-triggered failures by 16
calls, and passes the tighter baseline-contract gate, but it remains a
checkpoint-mismatch sensitivity until training and evaluation use the same
policy.

Do not reopen CPGNet architecture work or launch a PCNO gain, smoothing, noise,
or exposure sweep. Because the bump artifact lacks validated physical volumes
and oriented faces, it cannot authorize a conservative learned correction. The
canonical Mach-1.1 shock--isentropic-vortex benchmark is now closed by
`shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720c`: its v3
audit reports `benchmark_contract_closed`, all checks true, and final-state,
time-mean-state, and centerline-density successive ratios
`0.484166/0.481540/0.397123`. A pinned Clawpack 5.9.0 build
`py311h3d4ca6a_1` SharpClaw run passes the frozen state envelope at final time
(`0.0315534 < 0.0466208`) and in time mean (`0.0156538 < 0.0233192`), with zero
final shock-position difference and `0.00215749` vortex-core relative
difference.

The same audit closes the direct-reference-impulse contract only for the
primary WENO5-HLLC-SSPRK3 solver's full common-250x100-mesh restricted face
vector: full, divergence-active, cycle, and boundary error ratios are
`0.463428/0.463362/0.463482/0.464003`. This does not validate native fine-grid
face vectors. SharpClaw is a state-only independent comparison and exports no
face impulses; the full cycle field remains discretization-specific. The
failed v2 audit is preserved as
superseded fail-closed provenance, and the earlier Pyro mismatch remains a
historical low-order comparison rather than the promotion decision.

That authorization has now been consumed by one frozen full-resolution
19,155,720-parameter conservative-variable residual PCNO. On all 24 validation
cases it completes raw H60 admissibly, with physical-volume state error
`0.00028009` at call 1 and `0.00834190` at H60, and wins every paired H60 case
against persistence, nearest-training-trajectory, and train-only parameter/time
interpolation. It emits no face exchange, however, and its H60 physical-total
mismatch relative to reference boundary exchange is `0.0513185`; this is a
strong state baseline, not a conservative or flux model.

Dynamic D013 classifies the ripple mechanism as `unresolved`. The Fourier Gram
contract is essentially orthogonal and the spectral response is smoother than
the local responses, while no single local branch wins both response-gain and
roughness criteria. Later teacher-forced calls develop almost the same smooth
high-band energy as rollout calls, so a Fourier-looking ripple is not evidence
of Gibbs causality or purely recurrent birth. The predeclared 18-row sparse
conservative correction oracle then achieves only `0.05135` median state-error
and `0.04164` median smooth-high-pass reduction, below its `0.15/0.20` gates.
The learned local correction is rejected, the strength-OOD test split remains
sealed, and no basis/gain/smoothing sweep is authorized.

D045 closes the next state-loss-only shared-face-impulse gate without a serious
training run. The 19,210,028-parameter wrapper uses the same full 250x100
training mesh, an antisymmetric shared interior-face decode, current-state-only
boundary heads, and raw conservative recurrence. Its loader validates the
frozen family, source-artifact, physical-geometry, and mesh-to-graph contracts;
it does not load or supervise cumulative reference face impulses. A corrected
execution smoke passes, but the four-pair tiny fit does not meet the joint gate.
After 800 updates its best mean state relative error/loss ratio is
`0.00166965/0.0730244`; after the single bounded 3,200-update exposure retry it
is `0.000955675/0.0254377`, passing the `0.001` error threshold but missing the
required `0.01` loss ratio. A precision replay does not explain the floor. No
serious shared-face run, divergence-active supervision row, or test-split
evaluation is authorized from this result. Algebraic balance against the
model's predicted boundary exchange is not evidence that the physical boundary
exchange is accurate.

D046 then tests the canonical divergence-active target without training on 24
frozen train/position-OOD-validation rows. Its corrected run fails only the
proof-level canonical/reference-state closure gate: `4.94497e-7` versus
`1e-8`. Reference closure (`1.23427e-12`), float32-shard closure
(`9.36905e-6`), independent field agreement (`1.61785e-8`), compatibility,
cycle divergence, norm, wall, provenance, and solver-status checks all pass.
The accepted reference cycle field contains only `0.1129--0.6691%` of interior
weighted energy. This supports practical identifiability but does not pass the
registered promotion rule. No divergence-active tiny fit, LSMR tolerance retry,
or test-split evaluation is authorized; an exact fixed-mesh projector would be
a separate numerical-method preflight.

D047's separately registered direct projector passes, but D048's supervised
canonical-face tiny fit then exposes severe decoded conditioning: roughly 10%
face error becomes `1001--1581x` relative cell-increment amplification and
0/4 admissible predictions. D049 independently confirms the underlying
norm-dependent algebra on the validated physical mesh: median low/mid/high
divergence gains are `0.34903/0.97641/2.25444`, while cycle gain is at most
`1.24e-10`. This supports a face-norm/discrete-divergence mismatch, not Gibbs
or branch attribution.

D050's frozen residual-to-face preflight is stopped. Its legal algebraic lift
reconstructs D044 to `5.37e-12` without future-reference data, but the stored
oracle closure metric was bound to intentional compatibility projection and
formally fails. The separate future-reference headroom readout would not
promote anyway: it removes essentially all budget defect but improves median
H60 state error only `7.41%` versus the required `15%`. Do not rerun D050 or
start a learned face/boundary row from this result.

D051 now closes the paired coarse-CFD row. Full 250x100 CFD is H60 state-error
matched to D044 (`0.00808583` versus `0.00834190`) but is not uniformly better:
it improves front position and vortex core while worsening shock strength and
thickness. Its `13.53` s median core time is descriptively much larger than the
PCNO timing estimates, but PCNO p95/median is `11.70`, so the strict timing gate
fails. Float32 CFD also misses the frozen own-balance tolerance. Report matched
error and descriptive implementation cost only; do not claim production-CFD
speedup or physical conservation from D051.


D052 closes frozen per-branch attribution on the six D013 validation cases.
All 48 teacher/rollout rows and 576 one-sided gain interventions are finite and
admissible, but spectral, pointwise, and differential each pass zero repeated
case gates. Spectral attenuation usually worsens ripple or state/front error;
the weak ripple improvement from differential attenuation is paired with
state/front regression. At call 60, raw rollout state error is `8.95x` its
teacher-forced counterpart while smooth high-pass error is only `1.095x`, so
recurrence accumulates error without selectively exploding the measured ripple
band. The result is `composite_or_unresolved`, not pure Gibbs or a single local
culprit, and it stops gain/basis/smoothing architecture tinkering.

D053 has now completed that exact decomposition. The corrected 360-row
artifact closes the identities to `4.93441e-15`, replays D052 within
`4.76837e-7`, and keeps every raw output admissible. At calls 30/60, median
full propagated shares are `0.88995/0.89043` with near-unit propagation gain,
while smooth-high-pass shares are only `0.15619/0.30733` with gains
`0.1889/0.4056`. Roughly 91% of full error energy is propagated, whereas
`94.75%/84.72%` of smooth-high-pass energy is fresh teacher-forced defect.
Recurrence therefore carries the large shock-local state error but does not
selectively amplify the smooth ripple; the one-step map regenerates that
ripple at later interaction states.

The frozen classification is `mixed_or_split` and routes no learned method.
Do not launch gain, basis, smoothing, local-corrector, generated-exposure,
seed-confirmation, or strength-OOD rows under the current contract. D044 remains
the serious residual baseline. A new learned Line-3 row requires a separately
authorized target/representation hypothesis with a zero-training falsifier.

D054 registers that falsifier without training. It asks whether D053's fresh
smooth-region high-pass defect is both strongly sparse under a truth-selected
20% node upper bound and captured by one bounded current-state-only four-hop
pressure-jump halo. The fixed late-call gates are 70% oracle capture and 50%
causal capture with at most 25% interior support, repeated on five of six cases
at calls 30 and 60. Only a joint pass can justify drafting one matched tiny-fit
contract for a shock-conditioned local detail target. No result from D054 by
itself is learned-method, rollout, flux, conservation, or OOD evidence.

D054 completes that gate with a split result. The truth-selected 20% oracle
captures `98.741%/92.536%` median late fresh-ripple energy, but the legal
four-hop current-shock halo captures only `18.751%/20.290%` and passes zero
cases. The defect is sparse and oscillatory, but much of it lies well away from
the detected shock; do not sweep the halo.

D055 is the single authorized follow-up: rank non-shock-halo interior nodes by
the high-pass amplitude of PCNO's own proposed scaled update and test whether
the top 20% of eligible nodes captures at least 50% of late fresh-ripple energy
on five of six cases. This score uses only current state and the frozen legal
proposal. Passing still authorizes only a tiny-fit contract, not training;
failure stops this sensor-gated local-detail route.

D055 passes: the frozen proposal's own high-pass amplitude captures
`91.224%/80.212%` median late fresh-ripple energy on all six cases while using
only `17.304%/16.866%` of interior nodes. This is a legal troubled-proposal
locator, not yet a correction. D056 is the next zero-training gate: within the
already fixed support, test whether a truth-informed, volume-balanced,
10%-update-capped state correction can reduce median state/high-pass error by
15%/20% without inadmissibility or shock/vortex smearing. Only that pass can
authorize one frozen-global tiny detail-head fit.

D056 rejects that fit. Its 24-row oracle is replay-valid, balanced to
`3.62e-17`, raw-admissible, and anti-smearing compliant, yet median late state
reduction is only `7.19%/7.01%`; call-60 smooth high-pass changes by `-3.59%`
and only two of six cases are jointly nonworse. The 10% cap never binds, so
raising it is not a supported remedy. The proposal sensor locates fresh ripple
energy, but a sparse post-hoc correction cannot repair the dominant state
error. Do not train the detail head or sweep its support. A later learned row
must change the global map and address both halves of D053's split mechanism.

D057 is the zero-training entry gate for that global-map hypothesis. It checks
whether clean state error, fresh smooth-region graph-high-pass error, and one
detached generated-state error have a shared full-model descent direction at
the frozen D044 checkpoint. Relative loss scaling is fixed by unit-normalizing
the three training gradients; no coefficient sweep is involved. Repeated
positive transfer to the six validation cases at calls 30 and 60 can authorize
only a short full-resolution continuation contract, never training or test
access by itself.

D057 fails that gate despite compatible training gradients. Clean and
generated objectives have `0.98776` gradient cosine, and the high-pass transfer
is positive on all 12 validation rows, but the joint direction passes only 4/6
cases at call 30 and 2/6 at call 60. The failure concentrates in the held-out
lower vortex-position group, so the four-pair direction is not a safe global
continuation.

D058 is the full-training-split, still-zero-step follow-up. It turns each of
the seven training vortex-position groups into one balanced state/recurrence
gradient task, adds one global high-pass task, and asks whether their fixed
minimum-norm convex combination transfers to both held-out boundary-position
groups. This is a constrained geometry-group preflight, not a new loss-weight
sweep or a downsampled model.

D058 closes with a valid negative result. Its minimum-norm solve has
`5.55e-17` Frank-Wolfe gap and every training-task cosine is at least
`0.563997`; 6/6 held-out cases pass all three objectives at call 30. At call
60 only 3/6 pass. High-pass and generated-state directions remain positive for
all six cases, but clean-state transfer is negative for every lower
boundary-nearest `y00` vortex and positive for every upper `y08` vortex.
Thus using the complete training split repairs D057's sampling weakness but
does not yield a safe static long-horizon OOD continuation direction. Stop this
joint-objective route without training or test access. This does not establish
that multiobjective learning or PCNO is impossible; the next Line-3 candidate
must introduce a new solver-facing representation or coupling and pass a new
predeclared headroom/cost gate.

D059 now tests the smallest distinct temporal-target hypothesis. The
full-resolution D044 architecture is trained only on four immutable stride-2
pairs, with no noise or generated exposure, to determine whether a direct
two-interval conservative residual can clear the same 100-fold tiny-fit
reduction and raw-admissible 30-call smoke. This is motivated by D053's fresh
per-call ripple injection and the established 1D fewer-call benefit, but it is
not yet a rollout claim. Passing this one sub-0.05-GPU-hour gate can authorize
only a separately frozen serious stride-2 contract.

D059 passes in 69.43 seconds. Its four-pair error falls from `0.012469` to
`0.001155`, loss falls by `102.27x`, and all states remain raw-admissible.
The one position-OOD smoke completes 30/30 calls without an intervention. The
stride-2 residual scale and fitted error are both approximately twice the
stride-1 tiny-fit values, while the relative fit reduction is unchanged; this
supports a harder but fit-compatible direct map, not improved rollout.

D060 completed its sole authorized cold serious stride-2 run under
D044's exact architecture, schedule, noise, split, and raw evaluator. The
selected epoch-34 checkpoint completes 24/24 scheduled H60 validation rollouts
with physical mean state error `0.00729369`, `12.56%` below D044, and improves
all 24 paired cases. It still fails promotion: six-case H60 D013 median smooth-
high-pass RMS is `1.0255x` D044 instead of `<=0.8x`, all-24 endpoint high-pass
energy is worse in every case, and front-centroid distance is `1.4351x` D044.
Strength, thickness, vortex-core, total-mismatch, direct/composed, completion,
and descriptive timing components pass. This isolates a recurrence-depth state-
error gain, not ripple control or a uniformly better shock solver. Stop without
training, test access, an extra seed, stride 4, or a method combination.

D061 is the one bounded frozen-artifact follow-up. It aligns the serious D044
and D060 raw validation rollouts on the six D013 cases, tests a fixed equal
blend and a predeclared truth-informed scalar convex oracle, and reruns the
state/ripple/front/vortex/physical-total hierarchy. It executes no checkpoint,
uses no GPU or test case, and cannot promote an ensemble as an autonomous or
efficient solver. Exact gates and stop rules are in
`MECHANISTIC_DIAGNOSTIC_TRACKER.md`.

D061 fails. The H60 oracle improves state error by only `8.14%` over the better
parent and high-pass RMS by `10.10%` over D060, versus required `10%/20%`; it
is jointly nonworse in 0/6 cases and passes the anti-smearing envelope in 0/6.
D060 owns lower state error on 6/6 cases while D044 owns lower high-pass RMS on
6/6, so scalar state averaging is a Pareto interpolation, not a shock-preserving
repair. The legal equal blend is also too costly and fails the front hierarchy.
Disagreement localizes error (`0.650` late Spearman; `57.81%` top-20% capture)
but does not make the correction realizable. Stop without multirate training.

D062 also fails and closes the explicit front-chart exception. Its exact
target-informed conservative warp and four strength coefficients complete all
12 frozen rows, remain raw-admissible, and preserve every row/component total
to `2.69e-15`. At H60, however, median per-case state error worsens by
`41.88%` and smooth high-pass RMS worsens by `828.08%`, despite a
`62.82%` front-curve MAE reduction. Joint nonworse is 0/6 and every
front/shock/vortex acceptance count misses 5/6. The phase-only row already
contains the failure: competing pressure-jump branches make the independent
row argmax switch by up to `10.54` cells, so target-informed scalar alignment
shears the 2D field even while improving its own coordinate error. Do not train
this chart, retry or smooth its extractor, access test, or start assimilation.
No further experiment is authorized; exact evidence and the bounded non-claim
are in the tracker.

The authoritative Line-4 handoff is
`line3_to_line4_handoff_s20260718_20260721b.json`. It authorizes the frozen
dynamic reference as training truth only for this family, exposes no validated
front candidate, and leaves transition training false. Exact evidence and stop
rules are in `RESEARCH_DIRECTION_DECISION.md`.

The conservative-coordinate FNO matrix, exact ADER cumulative-face-impulse
export, and first direct/joint flux-supervision screen are complete. The label
closes correctly, and gauge-canonical joint supervision can fit the identifiable
face field. At 64/16/16 stride-1 scale, however, it reduces active-flux MSE while
worsening held-out state error and failing every paired raw rollout earlier than
the state-loss-only conservative flux head. This exact face-value loss is
stopped before full scale; it is a label-valid decoded-state and recurrent-
stability failure, not an implementation or closure failure.

The matched short-unroll comparison is also complete. At 64/16/16 scale,
four-step autoregressive fine-tuning improved the state-loss-only conservative
flux head's test one-step relative L2 from `0.00461` to `0.00344`, mean raw
survival from `0.519` to `0.759`, and completion from 0/16 to 2/16. A matched
smooth training-only admissibility barrier produced essentially the same result
and did not reduce the 14 nonpositive terminations, so it failed its attribution
gate. Inference remained the exact conservative flux update without a limiter
or floor.

The promoted 384/64/64 confirmation used the 316,739-parameter FNO, 240,000
one-step optimizer updates, then 46,560 four-step recurrent updates. Its selected
test checkpoint reaches one-step relative L2 `0.001305`, mean raw survival
`0.97734`, and 57/64 completed 20-call rollouts at mean initial effective CFL
`3.84`. The seven remaining raw failures are all nonpositive proposals. This
passes the scale gate but misses the strict 90% completion gate by one
trajectory. A frozen-checkpoint extension then exposes the horizon limit: only
1/64 cases completes 50 raw calls (mean survival `0.566`), and 0/64 completes
100 calls. The 20-call row is therefore a strong short-horizon result, not a
medium-horizon or unconstrained-positivity solution.

The strict next-state/residual target control is now complete through full scale,
three initialization seeds, and 100-call evaluation. Direct next conservative
state fails its preregistered tiny-fit gate on both declared seeds, while the
information-equivalent conservative residual passes, isolating an output-
centering/identity-bypass optimization effect. The 316,419-parameter residual
FNO has three-seed mean one-step relative L2 `0.001136`; pooled completion is
192/192, 187/192, and 182/192 at 20, 50, and 100 raw calls without an inference
limiter, floor, or positive transform. All ten seed-case terminations at 100
calls are raw pressure failures.

The matched 316,806-parameter projected-residual ablation factors the increment
into a volume-zero spatial field plus one learned boundary budget and closes to
that budget at about `1e-7`. It improves pooled 100-call completion to 187/192,
but the mean state-error ratio is `1.016`; among common completers it is 9.8%
worse in state L2, 5.9% better in conserved-total error, and 4.8% worse in shock
MAE. Only the preregistered stability-noninferiority gate passes. Residual is the
strong fixed-setting accuracy baseline; projection remains a useful structural
ablation, not evidence of general target superiority or unconditional
positivity.

A matched 64/16/16 generated-state exposure pilot then tested whether the
projected residual's accuracy deficit was caused by seeing states that were too
close to the data manifold. The control used four-step BPTT; the intervention
first rolled eight detached model steps and supervised the following four. The
50 one-step histories match exactly and the recurrent optimizer-update counts
differ by less than 1%. Burn-in reduces 100-call final state L2 by 36%,
conserved-total error by 46%, and shock MAE by 6%; it wins the paired common-
endpoint state comparison on 14/16 cases. It also reduces 50-call state L2 by
23% and wins 15/16 paired cases. The preregistered gate nevertheless fails:
20-call shock MAE is 38% worse, and mean 100-call survival falls from `0.9650`
to `0.9394` even though both rows complete 14/16 cases. Burn-in rescues one
clean-control pressure failure but creates a different failure at call 58 and
moves another from call 50 to 47. This is a partial distribution-exposure
result, not a full-scale promotion of projected residual.

The same gate is now complete for plain residual. Generated burn-in improves
final state L2 by 4.6% at 20 calls and 32% at 50 calls; at 100 calls it lowers
the common-endpoint state error by 51.4%, wins 14/16 paired cases, and raises
completion from 15/16 to 16/16. It also lowers conserved-total error at every
horizon. The formal gate still fails because the legacy single-argmax shock
MAE is 45.6% worse at 20 calls. Focused replay shows that two dominant
contributions are front-strength rank changes, while one case has a genuine
spurious/displaced-front regression. This is strong evidence for later-state
exposure, but the formal clean-versus-generated gate remains failed.

The matched teacher-offset control is now complete. It uses the identical 8+4
windows and recurrent update count but starts supervision from the exact
reference state after the eight-step offset. Teacher offset is 10.9% worse than
clean at 20 calls and statistically indistinguishable at 50/100 calls; it
finishes 14/16 H100 cases versus 15/16 clean. Generated exposure, by contrast,
reduces common-endpoint state error relative to teacher by 13.9%, 33.9%, and
50.8% at 20/50/100 calls, wins 15/16 H100 cases, and finishes all 16. This
supports a fixed-setting causal claim that off-manifold generated states add
value beyond later physical-time sampling. D023 now shows that this is not a
large local PDE-map improvement on matched generated states. At prefix depth
eight, generated/teacher fixed-scale conservative errors are `0.992` to the
original next state and `1.017` to the same-state solver continuation; the
correction-to-trajectory cosine is only `0.074`. Clean is 2.6% closer to the
solver continuation than generated, and generated is 8.0% worse in the
shock-region continuation defect. The solver replay baseline is negligible
and no replay needs retry or fallback. Thus the preregistered result is mixed:
generated-state exposure improves closed-loop dynamics without making the
one-step map materially more solver-consistent on this state bank. Full-scale
promotion remains paused.

The frozen scale diagnostic now identifies the state-loss-only flux failure as
recurrent high-frequency growth, not deficient teacher-forced spectral fit.
A paired conservative Laplacian-flux probe then produces no material H50
stability gain: small coefficients worsen state and Nyquist-tail error, while
larger coefficients shorten survival. D025 is now complete and negative for
the tested global relative-interface parameterization. At 64/16/16 scale,
frame-zero weighting cuts selected test one-step relative L2 from `0.0154` to
`0.00791` and sharply improves initial shock/front error, but H50 completion
remains 0/16. A training-only admissibility barrier at weight `0.1` also gives
0/16 and does not improve mean survival. Four-step training fails only after a
generated state has already become inadmissible, while the raw FNO output is
still finite. This is a recurrent conditioning/admissibility failure, not an
obvious decoder implementation bug or a receptive-field limitation. Do not
promote this row to full scale or repair it with an inference limiter.

An identifiable boundary-exchange auxiliary has also completed its matched
projected-residual gate. RMS-normalized weights `0.01` and `0.1` reduce the
one-step boundary-exchange error by 19.8% and 56.6% and the H20 conserved-total
error by 36.7% and 72.5%. Both preserve 16/16 completion, but they worsen
one-step state error by 15.2% and 33.7%, H20 state error by 14.6% and 19.8%,
and shock-position error by 24.8% and 20.8%. This is a clean structural Pareto
result, not evidence that the auxiliary adds missing information: in 1D the net
exchange is already recoverable from the endpoint balance. Stop coefficient
search and keep plain residual as the accuracy baseline for stride/resolution
transfer.

The first strict stride-transfer gate is now complete. A separately trained
stride-2 residual FNO uses the same 316,419 parameters and beats repeated
stride-1 composition by 25.0%, 36.2%, and 48.6% in common-case state error at
physical frames 20, 50, and 100, while completing all 16 H100 rollouts. This is
direct evidence that a global fixed-step neural operator can learn a useful
larger-step propagator at effective-CFL maxima up to about 13.3. It is not a
claim of CFL-free or timestep-conditioned inference: direct frame-2 error is
1.216 times small-step composition, and a different model is trained per
stride.

Weight-only stride-1 to stride-2 continuation repairs that initial jump and
lowers final-target training floors by more than 40%. It also improves H100
state error and raises the minimum pressure from 0.00193 to 0.0452. The gain is
not uniform: H50 state and shock errors are 1.095 and 1.565 times the cold
stride-2 row. Preserve cold stride 2 as the large-step accuracy baseline and
continuation as a fit/stability tradeoff ablation. The conditional
total-exposure control is stopped by successive halving.

Frozen-checkpoint resolution transfer is now complete. Native-grid zero-shot
one-step errors at 128 and 512 cells are 5.5 to 8.3 times the corresponding
256-cell values, and the 512-cell H20 shock metric worsens by 2.10 times for
stride 1 and 2.99 times for cold stride 2. The mismatch persists across input
times. Along separately evolved paired native trajectories, update labels
differ by 28% to 58% of the coarse increment norm, so the current numerical
residual target is materially grid dependent. This is not an exact same-input
commutator test, but it is evidence against mesh invariance of the present
target contract rather than evidence of an FFT implementation failure.

The larger-step comparison does transfer. Cold stride 2 beats stride 1 at
H50/H100 on both off-grid resolutions; its H100 common-case state-error ratios
are 0.474 at 128 cells and 0.523 at 512 cells without lower common-case
completion. The result is a stable large-step advantage without native
resolution transfer.

The follow-up restriction-consistent gate is complete. One shared 316,419-
parameter residual FNO, trained with the same total sample presentations as one
single-grid row, stays within `1.296x` one-step and `1.418x` H20/H50/H100 state
error of all three 128/256/512-cell oracles. It never loses completion relative
to an oracle and passes the H20 shock and conservation gates. Relative to the
frozen native-nx256 checkpoint, its off-grid one-step error falls by 68.1% at
nx128 and 78.3% at nx512 on common restriction-consistent truth. This shows
that a shared FNO can learn one identifiable cell-average flow map across these
resolutions without an explicit cell-width channel.

It is not a native-solver-invariance result. On independently evolved native
coarse trajectories, the shared model's one-step error is 5.27 times its
restriction-consistent value at nx128 and 3.65 times at nx256; the native
targets differ by several percent even at frame zero. The row is classified as
`shared_restriction_operator_without_native_solver_equivalence`. It also loses
the same three pressure-limited cases by H100 on every resolution. Do not add
cell width or more modes merely to repair D029. The shared pressure-failure tail
is evidence for a common admissibility mechanism, but the former 1D tail-risk
intervention is deferred under the four-line program.

The bounded Line-1 stride frontier and its frozen follow-up are complete. With
the horizon labels corrected to physical frames, selected endpoint error is
best at stride 2 at H8, stride 4 at H16, and stride 8 at H32/H64/H96. At frame
32, truth-state one-call error increases across strides 1/2/4/8 while
accumulated on-policy error decreases, directly demonstrating the
harder-map/fewer-recurrent-calls tradeoff. The frozen D035 extension verifies
the same absolute-error ordering and the reverse per-physical-time ordering at
all 12 truth-state starts through frame 88. At H96, stride 8 is best on global
L1/L2, shock/smooth, and post-hoc translation-aligned error, while stride 2 is
best on explicit front-position MAE. The simple physical-descriptor router
fails its preregistered gate and is stopped. Stride 8 completes 192/192 H96
rollouts across three seeds. The same-hardware frozen benchmark also verifies
that the call-count gain survives direct wall-clock measurement against the
documented reference solver. D036 confirms the visible metric conflict: stride
8 has lower H96 global and smooth-region state error than stride 4, but higher
away-front derivative error, excess TV, and earlier crossing of diagnostic
ripple budgets across all three stride-8 seeds. D038 explains the coexistence:
stride 8 has a seed-consistent broadband one-call spectral deficit and a
rougher early rollout spectrum, while fewer calls later lower its total modal
error. The remaining H96 global spectral-shape ordering is not seed-stable,
although the localized D036 roughness ordering is, so do not describe the
ripple as global high-frequency blow-up. D039 completes the frozen synthesis:
stride 8 versus stride 4 crosses sustainably at H32 for endpoint error but only
at H56 for time-mean error; at H96 stride 8 wins global error, error-budget
reliability, and latency while stride 2 wins front-position and away-front
roughness metrics. The H8 and H32 aggregate endpoint winners are not separated
from their runners-up by paired case-bootstrap intervals. These are conditioned
operating-envelope results, not a universal optimal timestep or neural CFL
threshold. D031--D036, D038--D039, and the
artifact/provenance rules are summarized in `RESEARCH_DIRECTION_DECISION.md`
and `MECHANISTIC_DIAGNOSTIC_TRACKER.md`; no further 1D learned training is
authorized.

The strict 1D target-family screen is complete. Preserve its label,
fit/optimization, generalization/closure, and recurrent-stability
classifications as evidence for 2D design; do not restart it as a broad sweep.
`RESEARCH_DIRECTION_DECISION.md` owns the bounded remaining tests.

The old 1D rows labeled CPGNet used a generic directed residual head and are
deprecated. The corrected `cpg_interface` baseline reconstructs positive
directed interface states, forms one Rusanov flux per face, applies the exact
finite-volume update, and evaluates raw recurrence without a cell-state
limiter or hidden floor. Its CPU implementation gate passes; the benchmark GPU
run and parameter-matched controls are complete. The deep mp28 row is the frozen
strong reference; the controls support receptive-field depth rather than
parameter count as the main gain.

The flagship 2D diagnostic stage must measure ripple energy and shock-front
geometry using the common rollout artifact contract: `predicteds`, `targets`,
`pos`, `edges`, and `node_type` when available. The bump artifact still lacks
validated physical control volumes, face measures, normals, oriented face
connectivity, and a mesh-to-graph mapping, so it cannot support physical
conservation or flux claims. The closed shock--vortex finite-volume contract
supplies those quantities and cumulative accepted-substep face impulses on its
common comparison mesh. Its direct full-impulse labels are nevertheless
same-primary-solver labels; SharpClaw validates state only, and the 2D cycle
component remains discretization-specific. D013-2D and D014 remain gates
against a broad basis, smoothing, or structured-target sweep.

## Not Ported Or Active

The raw `cpggnspdes` training scripts are not vendored into this branch. The deprecated CPG-style pilot heads, CPGNet interface-latent probe code, state-drift/perturbation/time-alignment scripts, early smoke scripts, and raw-HDF5 PCNO adapter path were one-off research tools and are no longer active tracked code. Historical analyzers still recognize their saved row labels so old results cannot be mistaken for corrected CPGNet.

FNO, PCNO, and MPCNO baselines should use implementations already present in this repository or collaborator-provided preprocessing artifacts. Do not port baseline code from earlier data-assimilation experiment repositories.

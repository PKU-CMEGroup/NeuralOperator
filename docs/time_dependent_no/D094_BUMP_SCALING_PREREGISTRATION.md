# D094: Hundreds-Trajectory Data--Architecture--Optimization Scaling

Date: 2026-08-20
Status: B1-A/B1-B, the B1-C0/B1-C1 seed-0 `n={8,16,32,64,128,256}` PCNO/PCFNO ladder, owner-selected B1-C4, B1-C5-A evaluator repeatability, and the paired FP32 B1-C5-B fixed-map diagnostic are complete, locally retained, and rehashed; B1-C5-B resolves a common-input functional response but does not reproduce the autonomous early-help/tail-harm crossover
Scope: use the 300-trajectory supersonic-bump population for an immediate native-mesh calibration, while making a hundreds-trajectory PlanarDet population the required hard-benchmark target

## 1. Decision

The proposed `n in {1,2,3,5,7}` factorial sweep is retired as the main scaling experiment.

- D092/D093 remain bounded selected truth-input and free-rollout observations; they do not isolate optimization, recurrence, exposure, or differential-branch causes.
- No paper-level data-scaling claim will be based on a maximum of seven released training trajectories.
- The main exposure ladder must reach hundreds of genuinely independent simulator trajectories: `n in {8,16,32,64,128,256}`.
- Repeated temporal windows, duplicate microbatches, spatial patches, augmentations, and multiple checkpoints from one simulation do not count as additional trajectories.
- The existing 300-trajectory supersonic-bump bundle is the immediate population-scale calibration. It remains scientifically separate from PlanarDet.
- The hard-benchmark claim requires an official or independently qualified hundreds-trajectory PlanarDet population. A newly generated simplified detonation dataset cannot be reported as REALM PlanarDet.

The purpose is to locate when the active bottleneck moves among data coverage, architecture, capacity, and optimization. The purpose is not to force a predetermined conclusion that either architecture or scale always wins.

## 2. Exact correction to the three-versus-seven interpretation

The earlier plan used “presentations per active trajectory” as if duplicate presentations were additional passes through distinct samples. That interpretation is wrong for D093.

### 2.1 What one D093 optimizer step actually does

The trainer chooses one temporal window and reuses it for all seven micro-presentations accumulated before one optimizer update.

- With seven active trajectories, each condition appears once:

  `g_7 = (1/7) sum_{i=1}^7 g_i(w_s)`.

- With three active trajectories, the same three condition--window pairs are duplicated with counts `(2,2,3)` in some order:

  `g_3 = (1/7) sum_{i=1}^3 c_i g_i(w_s)`, where `c_i in {2,2,3}`.

The triply weighted condition rotates, so the three condition weights balance over three optimizer steps. The repeated backpropagations occur before `optimizer.step()`. They are therefore a weighted estimate of the same update, not two or three visits to the trajectory under successively updated parameters.

### 2.2 Consequence

Both exposure arms:

- make exactly one parameter update per step;
- use the same learning-rate schedule;
- enter detached two-call training at step 491;
- present one new common temporal window per optimizer step;
- cycle through all 48 two-call windows every 48 optimizer steps; and
- expose every active trajectory to that same temporal-window cycle.

Thus a three-trajectory arm does **not** receive `7/3` times as many distinct temporal-window passes per trajectory. It receives larger within-update weights on the same three condition--window pairs. Its effective condition diversity per update is three, not seven.

At step 850, both arms have completed about 7.5 two-call temporal-window cycles after the phase switch. Similar selected steps are therefore compatible with a transition controlled by the shared objective/LR schedule. They are not evidence that a small dataset somehow resists repeated-sample overfitting.

### 2.3 Other reasons the current comparison is insensitive to data count

1. The `n=3` maximin subset is unusually favorable to the sole open validation condition. It contains both `phi=0.8,T0=290 K` and `phi=1.2,T0=290 K`; the validation trajectory at `phi=1.0,T0=290 K` lies directly between them in released condition space.
2. Normalization for both arms was fitted on all seven released training trajectories, so the three-trajectory arm did not have only three-trajectory information exposure.
3. There is one initialization seed and one open validation trajectory.
4. Validation occurs every 50 steps and checkpoint eligibility begins at step 550, limiting onset resolution.
5. Online training loss is grouped normalized MSE, whereas selection uses decoded REALM NPE. Their difference is not a conventional train--validation generalization gap.

The correct conclusion is: D093 was not designed to determine whether three trajectories overfit sooner than seven. A hundreds-trajectory study needs a sampler in which additional trajectories create additional distinct condition--window pairs, not duplicate weights inside one update.

## 3. Claims under consideration

At most two primary claims will be tested.

### C1: Data--architecture--capacity interaction

For a fixed PDE family, representation, temporal objective, and evaluation contract, the architecture gap changes systematically with independent trajectory exposure. Architecture-specific inductive bias may dominate at low exposure; with increasing exposure, the bottleneck may move to optimization or model capacity.

Support requires a trajectory ladder reaching at least 256 training simulations, three initialization seeds, fixed-compute and fixed-exposure views, comparable train/validation metrics, and a high-exposure capacity intervention. A seven-trajectory curve cannot support C1.

### C2: Optimization-transition mechanism

The abrupt late validation degradation is governed primarily by an optimizer/objective/recurrence transition rather than by a universal fixed number of repeated trajectory passes.

Support requires checkpoint-resolved comparable train and validation metrics, a schedule-timing intervention, recurrent-gain/validity diagnostics, and replication across data counts. A coincidence of best checkpoints from one seed does not support C2 by itself.

### Explicit anti-claims

The study will not claim that:

- grid nodes, patches, or windows are independent trajectories;
- the bump and PlanarDet populations are samples from the same PDE family;
- PCFNO is the FNO/FFNO used in the REALM paper;
- a rasterized bump representation is equivalent to native-mesh PCNO;
- a dense two-parameter PlanarDet sweep establishes universal PDE-family generalization;
- parameter count alone explains an architecture difference;
- finite fixed-resolution tests prove resolution-independent operator learning; or
- proxy graph weights establish physical conservation.

## 4. Data-source audit

### 4.1 REALM PlanarDet

The public REALM release contains nine PlanarDet trajectories, split 7/1/1. The paper reports that each case uses a high-speed DeepFlame reacting-flow simulation and costs, on average:

- 5,645 core-hours to reach stable propagation; and
- 1,008 core-hours for snapshot collation;

for 6,653 core-hours per trajectory.

The public `datasets/cases/PlanarDet` directory contains a statistics file and a descriptive notebook, but no runnable DeepFlame case, mechanism bundle, solver pin, or generation launcher. The local NeuralOperator repository likewise contains acquisition, adaptation, training, evaluation, and visualization code, but no PlanarDet simulator.

There are also release-description inconsistencies that a generator qualification must resolve: public descriptions mention `840x400`, `840x440`, and a released `832x384` array. The pinned released artifact manifest, not prose, remains authoritative for existing data.

At the reported mean cost, a target population of 320 trajectories would require approximately:

`320 x 6,653 = 2,128,960 core-hours`,

before failed jobs, queueing, generator validation, or storage/postprocessing overhead. This is an HPC campaign, not an ordinary AutoDL training job.

### 4.2 Supersonic bump

The existing bundle has 300 training and 20 historical test trajectories on geometry-dependent unstructured meshes. It was copied from an external Trixi-based campaign; this repository does not contain the original 300-case generator.

The read-only shard audit closed 300 unique manifest entries, 300 unique geometry digests, 300 trajectory directories, and all 3,300 required nonempty files. The frozen source-manifest SHA-256 is `5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7`. The field-blind 256/44 development partition and nested ladder are recorded in `D094_BUMP_SCALING_SPLIT_MANIFEST.json`: file SHA-256 `feb404e295c104c2ae9e66d25bbb809737bdd5d7febb0094a9d4aa3146191ccc`, canonical payload `6e22a0bb754df158b7ea8838adda2ac36dd80ed554e300d84ba1c4478314709b`, partition digest `ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb`, and metadata digest `77188a65c38193c2de0ad775ddbb88c3936c0495fc67fe180f9a4f4e03f62b72`. Both `state_arrays_opened` and `historical_test_population_opened` are false.

It is immediately suitable for native-mesh PCNO versus PCFNO scaling because both models share the same nodes, graph, target, optimizer, and evaluator. It is **not** immediately suitable for a clean comparison with the current regular-grid REALM FFNO. A common-grid remap would introduce a new representation and shock-remapping contract and would no longer be a controlled comparison to the native D041/D044 results.

## 5. Required PlanarDet generation contract

Before generating even a pilot trajectory, obtain and freeze:

1. exact DeepFlame repository commit and build options;
2. the high-speed solver identity and numerical schemes;
3. the 9-species/19-reaction mechanism file and thermodynamic/transport inputs;
4. mesh construction and the distinction among full, cropped, and released grids;
5. boundary and symmetry conditions;
6. complete hot-spot geometry, placement, initialization, and restart policy;
7. physical parameter mapping from folder names to mixture composition;
8. timestep, stable-propagation criterion, snapshot cadence, and crop window;
9. raw-to-release field definitions, including cumulative `pMax`;
10. existing additional trajectories, if any, and available HPC allocation.

The preferred practical action is to request the authors' runnable PlanarDet case and ask whether a denser `(phi,T0)` campaign already exists. Reimplementing the case from prose is not a faithful reproduction.

## 6. Target population and split, conditional on generator qualification

Target total: 320 independent simulations.

| Split | Trajectories | Use |
|---|---:|---|
| development train pool | 256 | nested exposure ladder |
| open validation | 32 | checkpoint selection, uncertainty over conditions |
| sealed test | 32 | one final evaluation after claims freeze |

Primary exposure ladder: `n in {8,16,32,64,128,256}`. The released `n=7` result remains a legacy benchmark anchor and is not treated as a point on this scaling curve.

Parameter design must be fixed before fields are generated or inspected. If the official case varies only `(phi,T0)`, use a recorded low-discrepancy design over the official envelope and label the result as two-parameter condition coverage. Do not invent random hot-spot or chemistry variation and still call the data REALM PlanarDet. If the benchmark owners define additional random initial-condition factors, record them as separate axes and split by simulator seed as well as condition.

The open validation population must include both interpolation and blocked-condition regions. The test assignment is generated from identifiers before simulation, sealed, and never loaded during development.

## 7. Training-exposure contracts

### 7.1 Regular-grid PlanarDet

For every `n >= 8`:

- effective batch size is seven trajectory--window pairs;
- the seven trajectories in one optimizer update are distinct;
- each selected trajectory draws from its own shuffled legal-window queue;
- no condition--window pair is duplicated within an optimizer update;
- trajectory selection is balanced by a deterministic cyclic permutation;
- one optimizer/scheduler update follows accumulation of the seven losses divided by seven; and
- sampler order and digest are persisted.

Define window-equivalent exposure per trajectory as

`X(s,n) = 7s / (48n)`

for the 48 legal two-call PlanarDet windows. Report results against both optimizer steps `s` and `X`. Unlike the old D093 presentation count, this quantity is meaningful only after the no-duplicate sampler contract is satisfied.

### 7.2 Native-mesh bump

The varying bump graphs cannot be stacked into a heterogeneous seven-trajectory
tensor batch without a new padding/remapping contract. B1 therefore uses:

- one trajectory--window pair per optimizer/scheduler update (`batch_size=1`,
  `gradient_accumulation_steps=1`);
- a deterministic balanced cycle over active trajectories;
- one independently shuffled queue of all 79 legal one-step windows per
  trajectory, exhausted before that trajectory reshuffles;
- no within-update duplication or weighting;
- the same optimizer-step budget and learning-rate schedule for every `n`; and
- persisted subset, sampler, and per-epoch presentation-stream digests.

Define bump window-equivalent exposure per active trajectory as

`X_bump(s,n) = s / (79n)`.

Under this contract, a smaller subset really does revisit each trajectory more
often at fixed optimizer compute. Report every result against both `s` and
`X_bump`; neither view replaces the other.

Primary normalization is fitted only on the active training subset. Architectures at the same `n` share the exact normalizer. A separately labeled shared-256 normalizer sensitivity at `n=16` quantifies how much unsupervised population statistics help; it is not mixed into the main curve.

## 8. Experiment blocks

There are five blocks. A later block cannot proceed until its entry gate passes.

### B0: Correct instrumentation and phase diagnosis

Purpose: make C2 measurable and prevent the old sampler error from entering new runs.

Required work:

- replace raw presentation counts with optimizer steps, unique condition--window pairs, and window-equivalent exposure;
- retain model sentinels around steps 490, 550, 700--1,000, 1,250, 1,500, 1,750, 2,000, and later checkpoints;
- evaluate the same grouped MSE and REALM NPE on seen trajectories, omitted released trajectories, and open validation;
- record learning rate, gradient norm, clipping, transform margin, admissibility, boundedness, and empirical recurrent gain;
- replay a shifted or capped LR schedule and a one-call versus detached-two-call branch from a common checkpoint; and
- reject all sealed-test objects by construction.

Gate: deterministic synthetic tests and a short GPU prefix must agree with the registered numerical contract. Exact-resume replay is a separate gate before any interrupted calibration cell can be resumed.

### B1: Native-mesh bump-300 population calibration

Purpose: test trajectory scaling now on an existing hundreds-scale population without pretending it is PlanarDet.

1. Audit the 300-entry shard manifest and construct a field-blind stratified development partition: 256 train-pool trajectories and 44 open validation trajectories. Keep the 20 historical test trajectories out of selection.
2. Build nested `n={8,16,32,64,128,256}` subsets using only Mach number, geometry descriptors, and identifiers; do not use solution fields or validation outcomes.
3. Compare full PCNO and functional no-gradient PCFNO under identical native graphs, residual target, boundary policy, normalization, sampler, capacity, optimizer, and compute.
4. Use the batch-one balanced-queue contract in Section 7.2; do not reuse the seven-microbatch D093 sampler or attempt to stack different native graphs.
5. Run seed 0 at `n={16,64,256}` first. Expand to the full ladder and three seeds only after sampler, finite-training, and rollout gates pass.

The seed-0 pilot budget is frozen before its calibration runs. Seed 0 is
`20260718`. Each of the six PCNO/PCFNO cells at `n={16,64,256}` runs for 20
epochs of 256 optimizer steps, or 5,120 optimizer steps total, with batch size
one and no gradient accumulation. Every epoch evaluates the fixed four-window
bank on every active training trajectory and the fixed 176-pair open-validation
bank. Every fifth epoch evaluates five fixed open-validation rollouts through
all 79 calls and retains H20/H40/H60/H79 endpoints. Sentinels are retained every
256 steps. The optimizer is AdamW with the D041/B0 learning rate, weight decay,
gradient clipping, warmup-cosine schedule, causal nodal physical boundary
closure, residual target, and active-subset-only normalization. Runs do not
early-stop or access the historical test population.

This is an uninterrupted, unreplicated calibration pilot. Exact resume has not
yet been established for the scaling adapter, so an interrupted cell is
terminal rather than resumed. A completed seed-0 cell is usable for pilot
routing, but not for a confirmatory architecture-by-data claim.

#### B1-A: error-first long-schedule amendment (2026-08-21)

The reported remote 5,120-step pilot makes a denser data ladder premature if
its packets close: both families were reported still improving at the terminal
budget, `n=256` has seen only about 0.253 window-equivalent passes per
trajectory, and the historical
stop-on-physics selector can compare different valid-prefix populations. The
next gate therefore holds `n=256` fixed and tests schedule sufficiency before
spending compute on the full ladder.

- Compare full PCNO and functional PCFNO under two fresh, paired 20,480-step
  schedules. The prefix-tail arm exactly reproduces the original 5,120-step
  warmup-cosine schedule and then holds the registered minimum learning rate;
  the stretched arm resolves warmup and cosine decay across all 20,480 steps.
- Freeze 16 deterministic open-validation trajectories for checkpoint-selection
  rollouts. The remaining 28 open-validation trajectories stay outside rollout
  selection and remain available for a post-selection audit. All 44 continue to
  contribute the fixed one-step validation bank. The 20 historical test
  trajectories remain unopened.
- Continue every finite deployed conservative state through all 79 recurrent
  calls. Record the first call and count for each thermodynamic or outflow
  violation. Stop only on a nonfinite deployed state or nonfinite error metric.
  Finite physical violations are diagnostic and do not enter checkpoint rank.
- Rank numerically complete checkpoints by mean all-node relative L2 over every
  call and every selection trajectory, then H79 all-node relative L2, then the
  fixed one-step validation metric. A hard numerical failure is treated as an
  unevaluable, hence worse-than-finite, full-horizon score.
- Keep online one-step train error, post-epoch fixed-bank seen-train one-step
  error, post-epoch open-validation one-step error, and autonomous rollout
  error as separate metrics. The first diagnoses the optimization path; only
  the latter three are fixed-population comparisons, and rollout measures a
  different recurrent object from either one-step metric.
- Retain one full best checkpoint and one state-complete last checkpoint.
  These contain optimizer and scheduler state, but the D094 adapter marks them
  non-resumable until exact-resume exposure accounting is implemented.
  Retain model-only, non-resumable sentinels at optimizer steps
  `{256,1280,2560,3840,5120,7680,10240,15360,20480}`. This replaces the
  storage-infeasible policy of saving optimizer state every 256 steps.
- Use seed `20260718`, 256 presentations per epoch, 80 epochs, batch size one,
  the same frozen split, sampler, normalization, initialization, optimizer, and
  data stream within each architecture. This four-cell schedule gate would
  still yield unreplicated routing evidence and does not access the historical
  test set.

The winning schedule, if healthy, becomes the schedule for the later
`n={8,16,32,64,128,256}` fixed-compute sweep. No data-scaling conclusion is
drawn from this gate alone.

This block can support a gradient-path-by-data interaction within the bump family. It cannot establish a PCNO--FFNO interaction because no clean common FFNO representation currently exists.

Gate: a material interaction must reproduce in at least two of three seeds and improve free rollout without a structure, boundary, or validity regression.

#### B1-A retrieved result and B1-B outside-selection audit registration (2026-08-21)

The four B1-A cells completed all 20,480 optimizer steps with the registered
source-set digest `6c510fbdca8f50d2bfacd40239574e7ac4496bdb0ba575c5ce69bb744568fca5`,
the registered partition digest, no historical-test access, and no hard
numerical rollout failure. The retrieved metadata/source packets were rehashed
locally. Selected-checkpoint metrics are reconstructed from the unique
`metrics.jsonl` row at `summary.json::best_epoch`; they are not taken from the
runner-generated scalar receipt because that receipt combines a best-epoch
label with terminal scalars when best and terminal differ.

| Architecture | Schedule | Selected step | Online train one-step | Fixed seen one-step | Fixed validation one-step | Rollout all-call mean | H79 | Physical-admissibility rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PCNO | prefix-tail | 20,480 | 0.0119006 | 0.0124163 | 0.0126211 | 0.0948993 | 0.141995 | 1.000 |
| PCNO | stretched | 20,480 | 0.0111151 | 0.0112578 | 0.0114138 | 0.0525120 | 0.0791166 | 1.000 |
| PCFNO | prefix-tail | 20,480 | 0.0142998 | 0.0170696 | 0.0174813 | 0.0896706 | 0.129719 | 0.875 |
| PCFNO | stretched | 15,360 | 0.0134953 | 0.0151365 | 0.0154052 | 0.0824803 | 0.121960 | 0.875 |

The stretched/prefix selected all-call rollout ratios are `0.5533` for PCNO
and `0.9198` for PCFNO. PCNO/PCFNO is `1.0583` under prefix-tail but `0.6367`
under stretched, so this one seed exhibits a large schedule--architecture
interaction. It is not yet a replicated architecture result. No arm meets the
registered three-checkpoint one-step over-optimization definition. PCFNO
stretched instead shows a different event: from its selected step 15,360 to
the terminal step, online train one-step changes `0.0134953 -> 0.0127366`,
fixed seen changes `0.0151365 -> 0.0134270`, and fixed validation changes
`0.0154052 -> 0.0137305`, while all-call rollout changes
`0.0824803 -> 0.100822` and H79 changes `0.121960 -> 0.138182`. Thus both
fixed one-step metrics improve by about 11% while recurrent error worsens by
13--22%. This is one-step/recurrent-objective divergence, not classical
train--validation
separation and not yet a causal mechanism.

Before the data ladder is expanded, B1-B evaluates the four already-selected
checkpoints on the same 28 open-validation trajectories excluded from rollout
selection. This audit does not reselect checkpoints. Its schedule rule is
fixed before those trajectories are opened:

1. require numerical completion and zero hard failures for both architectures;
2. among complete schedules, minimize the geometric mean of the PCNO and PCFNO
   all-call rollout relative L2;
3. use the corresponding geometric-mean H79 error only as a tie-break; and
4. report physical admissibility, normal/boundary error, shock/front error,
   smooth-region high-pass error, and reconstructed-weight proxy-total error
   separately. Finite physical violations do not override a lower finite
   rollout error, and proxy totals do not establish physical conservation.

Only a numerically complete B1-B winner routes the seed-0
`n={8,16,32,64,128}` ladder; the winning n=256 B1-A cell supplies its paired
n=256 endpoint. The 28 audit cases remain outside checkpoint selection, and
the 20 historical test trajectories remain unopened.

#### B1-B result and routed ladder (2026-08-21)

B1-B completed on all 28 open-validation trajectories excluded from checkpoint
selection. All four selected checkpoints completed H79 with zero hard numerical
failure. The locally retrieved archive has SHA-256
`0cf1b16b6a91979849a9b6ebba75a5eb353c5d8b2537122568bee4b74a235cbf`;
all six files in `artifact_manifest.json` and all 23 evaluator source files
were rehashed successfully. The evaluator source-set digest is
`21b0bbd0141650bedd73d0e0e20c7e465f8782a22fee6c37d87884fa20ef864d`,
the checkpoint source-set digest remains
`6c510fbdca8f50d2bfacd40239574e7ac4496bdb0ba575c5ce69bb744568fca5`,
and the partition digest remains the registered value above.

| Architecture | Schedule | Audit all-call mean | Audit H79 | Final normal | Final boundary | Completion | Hard failures | Physical admissibility |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PCNO | prefix-tail | 0.0963097 | 0.148372 | 0.147696 | 0.188805 | 1.000 | 0 | 1.000 |
| PCNO | stretched | 0.0466426 | 0.0710526 | 0.0702815 | 0.110686 | 1.000 | 0 | 0.9643 |
| PCFNO | prefix-tail | 0.0972293 | 0.164575 | 0.163669 | 0.215982 | 1.000 | 0 | 0.9643 |
| PCFNO | stretched | 0.0878169 | 0.126438 | 0.125700 | 0.169155 | 1.000 | 0 | 1.000 |

The preregistered geometric-mean all-call score is `0.0967684` for
prefix-tail and `0.0640001` for stretched; the corresponding H79 scores are
`0.156264` and `0.0947825`. Stretched/prefix all-call ratios are `0.4843` for
PCNO and `0.9032` for PCFNO. Therefore stretched is the registered winner and
authorizes only that schedule for the fresh seed-0 ladder.

The result is not a uniform structure win. Relative to prefix-tail, stretched
improves front-centroid distance, front IoU, symmetric Chamfer distance,
smooth-region high-pass error, and the reconstructed-weight proxy-total error
for both architectures. PCNO shock-thickness log error is slightly worse
(`0.17692` versus `0.16593`), and PCFNO shock-strength log error is materially
worse (`0.17944` versus `0.10007`). These diagnostics do not override the
registered rollout-error ranking and the proxy total is not physical
conservation.

A post-hoc 10,000-draw paired trajectory percentile bootstrap, using one shared
resample stream with seed `20260821` and recorded only as a descriptive
robustness check, gives PCNO stretched/prefix all-call ratio `0.4843` with 95%
interval `[0.3775,0.6237]` and 28/28 trajectory wins; PCFNO gives ratio
`0.9032` with interval `[0.8597,0.9495]` and 21/28 wins. PCNO/PCFNO is
`0.9905` with interval `[0.7873,1.2146]` under prefix-tail but `0.5311`
with interval `[0.4585,0.6514]` under stretched, with PCNO winning 27/28
stretched cases.
These intervals use trajectories, not nodes or frames, as units. They were not
part of the preregistered schedule rule and do not replace seed replication.

The fresh paired `n={8,16,32,64,128}` ladder was then launched serially from
commit `066238b` using the stretched schedule, 20,480 optimizer steps per cell,
and the same seed, split, normalization, recurrence, finite-only rollout, and
error-first checkpoint selection contracts. The deployed Git archive has
SHA-256 `2a6793e5db81cbd372135a742ecc67d6b91b3e4064f0fd05e076b1144d09b2b8`,
the deployed v6 source-set digest is
`c9ecfd93f75f61a69a1333f2f25b0778f4436a33660b40bd3dbc38fc862a8e61`,
and the runner SHA-256 is
`43674d288e1a91fdbe05a6dfd9f574e75638b576e433fe9766a6b6e4307a2491`.
The B1-A and ladder source digests differ, but the bounded diff audit closes
that difference to line-ending materialization, fail-closed schedule and resume
guards, the selected-versus-terminal metric receipt, evaluator-only structure
diagnostics, and provenance text; it does not change fresh forward, objective,
optimizer, sampler, or base rollout-error semantics. This is why the
preregistered B1-A `n=256` reuse remains the paired endpoint.
Both architecture preflights passed. Initial PCNO `n=8` health closed through
step 1,280 with finite online-train, fixed-seen, fixed-validation, and
autonomous-rollout metrics, complete H79 rollout, and zero hard failures. This
is an operational health receipt, not a completed ladder result.

`evaluate_pcno_bump_scaling_ladder.py` is the registered post-completion
evaluator. It requires the exact ten fresh cells plus the two reused stretched
`n=256` endpoints, zero ladder/gate exit receipts, exact checkpoint/source/split
and nested-subset bindings, and matching boundary-policy digests. It records
selected and terminal training metrics separately, then evaluates every fixed
selected checkpoint on the same 28 outside-selection trajectories under the
finite-only H79 and structure contracts above. It cannot accept a historical
test input and does not reselect checkpoints.

#### B1-C: post-ladder analysis and replication route (registered 2026-08-21)

The bump calibration now tests two questions and no broader claim:

1. Does independent trajectory exposure change the full-PCNO versus
   no-gradient-PCFNO gap under matched native graphs, capacity, target,
   optimizer compute, and evaluation?
2. Does late degradation follow classical seen/held-out one-step separation,
   or do one-step metrics continue to improve while recurrent rollout worsens?

The explicit anti-claims are that one seed establishes an architecture law,
that a lower online training trace is comparable to fixed validation error,
that PCFNO is vanilla or paper-faithful FFNO, and that the 28 audit cases remain
an untouched final holdout after they selected the stretched schedule.

The next stages are:

| Stage | Required work | Decision role |
|---|---|---|
| B1-C0: completion audit | Let all ten fresh seed-0 cells finish; retrieve and rehash the exact cell matrix; combine it with the two bound `n=256` endpoints; run the registered 12-checkpoint outside-selection evaluator. | Fail closed on any missing cell, nonzero exit, source/checkpoint/split drift, hard numerical failure, or historical-test access. Do not interpret partial cells. |
| B1-C1: seed-0 surface | Plot the 80-row histories against optimizer step `s` and corrected exposure `X=s/(79n)`, using fixed seen only at checkpoints where it was actually evaluated; plot the selected fixed-compute outside-audit curve against `n`; retain selected and terminal rows separately. | Screen the shape of the data--architecture--optimization interaction; it remains single-seed development evidence. Do not interpolate a missing fixed-seen or model-state metric. |
| B1-C2: seed replication | If B1-C0 closes, propose the full six-count PCNO/PCFNO ladder for seeds 1 and 2 under the already selected stretched schedule: 24 new cells. | A positive, null, or reversed seed-0 curve is retained; the purpose is replication, not only confirmation. A material interaction must agree in at least two of three seeds. |
| B1-C3: compute/exposure control | Use the existing histories for exploratory matched-exposure slices. For future seeds, retain a count-specific model checkpoint at `s_X(n)=64n`, for common `X=64/79`, so fixed seen, fixed validation, rollout, and outside-audit metrics can be recomputed from one exact state. | Do not call interpolation between unmatched checkpoints a fixed-exposure result. A seed-0 replay is proposed only if the missing exact checkpoint would change the claim. |
| B1-C4: owner-selected bottleneck test | Run cold 40,960-step PCNO at `n={128,256}` under the exact contract below. Continue to 81,920 and 163,840 only after a separately reviewed joint one-step/rollout gate. Route PCFNO to an objective/recurrence diagnostic instead of generic longer training. | Distinguish under-optimization from recurrent-objective divergence and then from capacity. Do not diagnose capacity merely because adding data stops helping at one compute budget. |

Using the observed long-gate rates of about 31 minutes for PCNO and 22 minutes
for PCFNO, a conservative full 12-cell seed costs about 5.3 serial GPU-hours;
seeds 1 and 2 together cost about 10.7 GPU-hours before evaluation. At the
current roughly 1.15 GB per retained cell, 24 new cells can require about 28 GB.
B1-C2 therefore requires an actual post-ladder byte count, archive/retrieval
receipt, and free-space preflight before launch. No existing remote output is
deleted or compacted without a separate explicit decision. A count-specific
sentinel inventory should retain the required matched-exposure state without
duplicating unnecessary model-only checkpoints.

For every stage, the primary views are distinct:

- online train one-step is an optimization trace;
- fixed seen and fixed open-validation one-step use the same statistic and may
  support a classical generalization-gap statement;
- selection-cohort rollout traces diagnose checkpoint dynamics but are
  development metrics;
- the 28-case audit reports fixed checkpoints without reselection; and
- selected-checkpoint and terminal-checkpoint rollout are never merged.

#### B1-C4 owner correction and executable contract (2026-08-22)

Current explicit owner direction selects B1-C4 before seed replication. The
comparison is corrected as follows.

- D094 `n=256` PCNO has not beaten retained L3R-B1: H79 is `0.0791166`
  versus B1's selected `0.0332016`.
- At similar optimizer updates, D094 reaches H79 `0.0791166` at 20,480
  updates and 20,480 one-pair presentations, while the owner-provided B1
  comparison is H79 `0.08048` at 21,600 updates and 85,320 presentations.
  This is a roughly fourfold presentation-efficiency signal, not a SOTA or
  matched-evaluator result.
- Over D094 PCNO's final 5,120 updates, fixed validation improves 4.9%,
  all-call rollout improves 35.7%, and H79 improves 30.5%. PCNO is therefore
  still under-optimized under this exact schedule.
- PCFNO is qualitatively different: after its selected checkpoint, fixed
  validation improves 10.9% while all-call rollout worsens 22.2%. It is not
  admitted to the generic compute extension.

The B1 comparison uses a different validation/rollout cohort and historical
training contract. It is qualitative consistency evidence only. D044/B0 has
not been evaluated under the D094 evaluator. The retained D094 v6 source
snapshots and closeout artifact manifests rehash internally; the historical B1
v1 manifest does not retain its bound source copies and is not promoted to a
current-source compatibility receipt.

The authorized B1-C4 execution contract is exactly:

- architecture `PCNO`, differential branch `full`, initialization seed
  `20260718`, and trajectory counts `n={128,256}` only;
- cold initialization only: no checkpoint initialization or resume;
- 160 epochs of 256 one-pair optimizer updates, hence 40,960 updates and
  presentations per cell;
- the existing one-step residual target, balanced no-replacement queues,
  count-specific normalization, native graph, causal nodal physical boundary
  closure, optimizer, and error-first finite-only H79 selection contract;
- warmup-cosine decay stretched through step 40,960; consequently its
  20,480-step state is an exact within-run anchor but not the same learning-rate
  trajectory as the completed 20,480-step arm;
- model-only sentinels at `{8192,20480}` for `n=128` and `{16384,20480}` for
  `n=256`, retaining exact `s_X(n)=64n` and the old-budget anchor without
  duplicating every diagnostic checkpoint; and
- fixed-seen and recurrent rollout diagnostics every five epochs. The frozen
  continuation rows are steps `{38400,39680,40960}`.

An individual cell is eligible only to *propose* 81,920 steps when fixed open-
validation one-step error, all-call rollout error, and H79 error each strictly
decrease across all three frozen late rows, every rollout completes, and no
hard numerical failure occurs. Physical admissibility remains reported but is
not a checkpoint-selection or continuation criterion. The generated gate
receipt explicitly sets automatic continuation authorization to false. No
81,920/163,840 run, PCFNO/FFNO comparison, seed replication, capacity control,
or historical-test access follows automatically.

The post-completion B1-C4 outside-selection audit is frozen before execution.
It evaluates exactly four retained checkpoints per count on the same 28 open-
validation trajectories excluded from checkpoint selection: `{8192,20480,
38400(selected),40960(terminal)}` for `n=128` and `{16384,20480,
38400(selected),40960(terminal)}` for `n=256`. All-call, H79, completion,
finite failure, admissibility, boundary, shock/front, high-pass, and proxy-total
diagnostics are reported under finite-only recurrence. Outside cases never
reselect a checkpoint, and admissibility is not promoted ahead of rollout
error in ranking or interpretation.

Visualization is also frozen as diagnostic-only. The loss figure keeps online
train, fixed seen-train one-step, fixed open-validation one-step, 16-case
selection H79, and 28-case outside-audit H79 explicitly distinct. Trajectory
bundles use the first registered outside-selection key plus the largest paired
selected-checkpoint H79 disagreement among the remaining keys; the second is
post-hoc visualization selection and supports no population claim. For each
case, free recurrent pressure shows accumulated state error, while one-step
conservative density and energy residuals use exact reference `U(t)` at every
call and show predicted `U(t+1)-U(t)`, truth, and residual error with no
accumulated rollout drift. Scales are fixed across both counts and all calls.

Define the paired architecture curve as

    R_arch(n,s) = E_PCNO(n,s) / E_PCFNO(n,s),

where `E` is reported separately for fixed one-step, all-call rollout, and H79.
Plot raw per-seed and per-trajectory values before medians, bootstrap intervals,
or any log-data fit. A changing `R_arch` is an interaction signal, not by itself
a gradient-path mechanism.

Vanilla/paper-faithful FFNO is not inserted into this native-mesh bump matrix.
The controlled PlanarDet stage retains a matched-objective FFNO for architecture
attribution and a separate direct-state paper-faithful FFNO for benchmark
reproduction, only after B2 generator qualification and B3 population audit.

#### B1-C0/B1-C1 retrieved result (2026-08-21)

The ten fresh stretched-schedule cells completed at `2026-08-21T22:20:36+08:00`
with `ladder.exit=0`, 80 metric rows and one selected/terminal metric receipt per
cell, no hard-error signature, and no historical-test access. The registered
evaluator then combined them with the two B1-A `n=256` endpoints and evaluated
all 12 already-selected checkpoints on the 28 outside-selection trajectories.
All 12 complete H79 with zero hard numerical failures. The evaluator cannot
accept a historical-test input and did not reselect a checkpoint.

The exact retained bindings are:

- evaluator deployment archive SHA-256
  `5d8437029796cbbbb6097b163bbd9847487ee45cd9177e1e4010b43ad4511c91`
  from commit `1a2a859`;
- evaluator source-set digest
  `36178b2a6526d4da997c614c74090e2083cc85502ad0ac77e3b7ae39d061836b`;
- audit summary SHA-256
  `b1002e3f8f711efef8da2d753b6562bf39445bb4b9fc9b79c7ab6d1fac357b93`;
- retrieved fresh-metadata archive SHA-256
  `194d08d9e438eedfceb312a794893104cb1c3779073a4beb9731f1b17d3dd235`;
- retrieved audit archive SHA-256
  `cf0d1a84ddd71e8b7475e4734ed3b630e3b36ad8361d840847a13897e5a1912e`;
  and
- local B1-C1 analysis-manifest SHA-256
  `072a14ddc1a6b724cdb1c6c9410a22c7126898699875a98a335f2e9e5d75e801`.

The maintained invocation surface is
`scripts/time_dependent_no/analyze_pcno_bump_scaling_ladder.py`. It exists
separately from the B1-A schedule analyzer because this closeout must join ten
fresh cells with two bound B1-A cells, revalidate their paired controls, and
emit selected, terminal, outside-audit, and exact-exposure tables without
opening a test input.

All 14 declared evaluator artifacts, all 26 evaluator source/provenance files,
all ten fresh source snapshots/metric receipts/contracts, and both registered
checkpoint source-set identities rehash. PCNO and PCFNO share the exact
normalizer, initial full and non-differential parameter hashes, presentation
stream, fixed-seen bank, and fixed-validation bank at every `n`. The outer
detached-screen wrapper left a stale socket and did not emit its auxiliary
post-process exit file, so that wrapper receipt is not used as evidence; the
evaluator itself reached its terminal `status=complete` write, disappeared with
no GPU process, and its complete result/source manifests independently rehash.

The primary fixed-checkpoint outside-audit table is:

| `n` | PCNO all-call | PCFNO all-call | PCNO/PCFNO | Descriptive paired 95% interval | PCNO trajectory wins | PCNO/PCFNO physical admissibility |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0962884 | 0.0920867 | 1.0456 | [0.8996, 1.1834] | 18/28 | 0.250/0.857 |
| 16 | 0.0546169 | 0.0687441 | 0.7945 | [0.6624, 0.9636] | 23/28 | 0.750/1.000 |
| 32 | 0.0505600 | 0.0666006 | 0.7592 | [0.5758, 0.9860] | 25/28 | 0.857/1.000 |
| 64 | 0.0464832 | 0.0820257 | 0.5667 | [0.4650, 0.6901] | 23/28 | 0.929/0.964 |
| 128 | 0.0364494 | 0.0689236 | 0.5288 | [0.4815, 0.5822] | 28/28 | 1.000/1.000 |
| 256 | 0.0467030 | 0.0878169 | 0.5318 | [0.4581, 0.6546] | 27/28 | 0.893/1.000 |

The intervals are a post-hoc 10,000-draw paired trajectory bootstrap with seed
`20260821` and one shared resample stream across counts. They describe
within-seed case robustness; they do not replace initialization-seed
replication. These 28 trajectories already selected the stretched schedule and
are a development audit cohort, not untouched final evidence.

Four observations survive the metric-semantics checks.

1. **The architecture interaction is recurrent, not simply one-step.** At the
   common terminal step, PCNO/PCFNO fixed-validation one-step ratios stay in the
   narrow range `0.802--0.835` across all six counts, while terminal selection-
   cohort rollout ratios change from `1.167` at `n=8` to `0.521` at `n=256`.
   On the outside audit, PCNO is a near tie/slightly worse at `n=8` and is
   20--47% lower-error for `n>=16`. At `n=8`, PCNO therefore has better
   one-step fit but worse rollout and much lower admissibility; the one-step
   benefit does not yet survive recurrence.
2. **The common-budget high-`n` reversal is not evidence that data has stopped
   helping.** With every cell trained for at most 20,480 steps, the selected
   outside-audit checkpoints are best at `n=128`, then worsen by about 28% at
   `n=256`, where selected exposure falls from about `1.90` to `1.01/0.76`
   window-equivalent passes for PCNO/PCFNO. At exact common exposure
   `X=80/79`, however, increasing `n=128 -> 256` lowers the selection-cohort
   rollout by 66.0% for PCNO (`0.15436 -> 0.05251`) and 54.1% for PCFNO
   (`0.21980 -> 0.10082`). At exact `X=160/79`, increasing `n=8 -> 128`
   lowers rollout by 85.7% for PCNO and 79.4% for PCFNO. These are exact
   retained rollout rows, not interpolated states. The view changes both data
   count and optimizer compute to hold per-trajectory exposure fixed, so it is
   an interaction diagnostic rather than an isolated data-only effect.
3. **Classical one-step overfitting and recurrent degradation separate.** The
   terminal fixed-validation/fixed-seen gap shrinks from `20.9%/23.1%` for
   PCNO/PCFNO at `n=8` to roughly `1--2%` for `n>=64`. None of the 12 cells
   meets the registered three-checkpoint one-step over-optimization event.
   Nevertheless, PCFNO `n=256` improves fixed seen and validation one-step
   error by `11.3%/10.9%` after its selected step while rollout worsens 22.2%;
   the analogous PCFNO `n=128` rollout worsening is 12.6%. PCNO `n=128`
   rebounds 17.4% after selection while its fixed one-step metrics are nearly
   flat. These are recurrent-objective/checkpoint dynamics, not a suddenly
   widening conventional generalization gap.
4. **Checkpoint timing aligns more closely with optimizer phase than repeated
   passes.** Selected steps lie between 15,360 and 20,480 for every cell, while
   selected exposure ranges from `32.4` down to `1.01` passes for PCNO and
   `26.3` down to `0.76` for PCFNO. Thus similar best-step timing across data
   counts is compatible with the shared LR/objective phase and is not evidence
   that small subsets somehow avoid repeat-sample overfitting.

The structure result is useful but nonuniform. PCNO has lower H79 shock-strength
error, smooth-region high-pass error, and reconstructed-weight proxy-total
error, plus higher front IoU, at every `n`. Front-centroid distance and shock
thickness are mixed, and PCNO physical admissibility is lower at five of six
counts. The proxy total is not physical conservation. PCFNO is the exact-zero,
frozen differential-branch ablation: it preserves the 19,155,720-parameter
state-dict but trains 19,024,644 parameters, freezing 131,076 (`0.684%`). It is
not vanilla FNO or paper-faithful FFNO, so the changing ratio is an interaction
signal, not isolated gradient-path causality or an FFNO comparison.

The internal result-to-claim verdict is **partial, medium confidence** for C1
and C2, pending independent Codex review. The bounded observations above have
high confidence under this exact one-seed development contract. They do not
support a multi-seed scaling law, general architecture ranking, causal
differential-path mechanism, capacity bottleneck, physical conservation,
untouched-holdout result, or historical-test claim.

#### B1-C4 retrieved result (2026-08-22)

Both cold cells completed 40,960 optimizer steps with count-specific
normalization, 160 metric rows, complete finite H79 rollouts, zero hard
failures, and no historical-test access. Both selected step 38,400. Neither
cell passes any of the three frozen strict-decrease checks over steps
`{38400,39680,40960}`, so neither is eligible to propose 81,920 steps and no
automatic continuation is authorized.

Owner clarification (2026-08-22): this was a routing criterion for automatic
continuation, not a test of optimization, representation, or recurrent-map
convergence. Three late scalar rows cannot establish that internal
representations or deployment response have stopped changing after one-step
loss becomes nearly flat. An 81,920-step study therefore remains scientifically
open as a later representation-evolution probe, but it is not the current
priority and it requires a separately frozen scheduler and checkpoint-response
contract.

The primary fixed-checkpoint comparison is:

| `n` and checkpoint | Fixed seen one-step | Fixed validation one-step | Selection-cohort H79 | Outside-audit all-call | Outside-audit H79 | Outside admissibility |
|---|---:|---:|---:|---:|---:|---:|
| 128 selected, step 38,400 | 0.0107988 | 0.0109911 | 0.0399680 | 0.0293754 | 0.0389224 | 0.9643 |
| 128 terminal, step 40,960 | 0.0107622 | 0.0109583 | 0.0406254 | 0.0290868 | 0.0395505 | 0.9643 |
| 256 selected, step 38,400 | 0.0108548 | 0.0110188 | 0.0403593 | 0.0313774 | 0.0445500 | 0.9643 |
| 256 terminal, step 40,960 | 0.0108614 | 0.0110207 | 0.0434540 | 0.0315600 | 0.0464606 | 0.8929 |

Five bounded observations follow.

1. **The selection-cohort near tie does not transfer exactly to the outside
   audit.** At the selected checkpoint, `n=256` is only 0.25% worse in fixed
   validation one-step error and 0.98% worse in selection-cohort H79. On the 28
   outside trajectories it is 14.46% worse in H79, and `n=128` wins 22 of 28
   paired trajectories. This is recurrent generalization evidence under one
   seed, not a data-scaling law.
2. **The fixed-exposure and fixed-update slices answer different confounded
   questions.** At exact `X=64/79`, `n=256` step 16,384 has outside H79
   `0.155737` versus `0.263888` for `n=128` step 8,192, a 41.0% reduction, but
   it also has twice the optimizer updates and a later scheduler phase. At the
   same 20,480 updates, `n=256` has H79 `0.109682` versus `0.096438` for
   `n=128`, 13.7% worse, but only half as many passes. Neither slice isolates
   data count, compute, repetition, or schedule by itself.
3. **Late one-step and recurrent objectives decouple without a classical
   generalization gap.** Selected-to-terminal outside H79 worsens 1.61% for
   `n=128` and 4.29% for `n=256`, while fixed one-step changes are tiny. For
   `n=128`, outside all-call error improves 0.98% even as H79 worsens; endpoint
   selection and mean-horizon selection are therefore also distinct. The fixed
   seen--validation gap remains small, so this is not evidence of a suddenly
   widening conventional train--validation gap.
4. **More trajectories trade among recurrent observables rather than improving
   all of them.** At the selected checkpoint, `n=256` has better front-centroid,
   symmetric-Chamfer, and reconstructed-weight proxy-total means, but worse
   front IoU, shock-strength/thickness, smooth high-pass, and global H79. The
   proxy total is not physical conservation.
5. **Aggregate evaluation is more repeatable than individual case ranking.** A
   warning-bearing first attempt and its clean tensor-copy replacement differ
   by only `7.70e-5/1.86e-5` in selected aggregate H79 for `n=128/256`, but the
   maximum per-trajectory H79 difference across all audited checkpoints is
   `0.0273`, and the post-hoc largest-gap visualization case changes. This is
   consistent with recurrent amplification of CUDA/bfloat16 nondeterminism but
   does not yet identify the numerical mechanism. Population aggregates are
   retained; maximum-case and marginal admissibility rankings require a
   repeatability contract before scientific use.

Retained clean provenance is source commit `3baa6cd`, deployment-archive
SHA-256 `34ff20986067757eb7bbcaedc9f235c208d54c91eed45403f2aeea3c81106766`,
evaluator source digest
`e5923f1d54bbce9f41101829b37394b06d387da3d8cd911d4432238dfd9ecea5`,
provenance digest
`bb379962794d12f26451faff83f4977c73d61010d95118fe47636b8915d23208`,
clean summary SHA-256
`d48ab637704c49f647f7ea7ce00babaf32bda3690e41d9437ddbfe6d60d94cd1`,
and artifact-manifest SHA-256
`88cf197bcef98681162309b59aff76d4bc1f5e187eca070deae84d0cacc7e6ea`.
All 12 declared artifact members and all 28 retained Git-archive source members
rehash. The checkout-side verifier reports two Windows line-ending differences
in `euler2d.py` and `euler2d_metrics.py`; byte comparison to `git show
3baa6cd:<path>` closes all 28 source members, so this is a verifier portability
caveat rather than deployed-source drift.

The diagnostic visualization packet has manifest SHA-256
`934d70ad0c5329476d4708556ef4ce82c5ddfc881514ab137deac219838950b8`
and renderer SHA-256
`854eb0d931272f7d90ce013df16b95cf15c77025a48c74b086723fecf7ea6332`.
All 15 declared data/figure/movie records rehash. Each of the six H.264 movies
has 79 frames at 5 FPS, even `1782x380` dimensions, and `yuv420p` pixel format.
Trajectory 25 is the first frozen outside key; trajectory 83 is the clean
run's post-hoc largest selected-H79 disagreement and remains visualization-only
because that maximum-case ranking is not repeatable enough for a population
claim.

The B1-C4 registered route was:

1. do not automatically run 81,920 or 163,840 steps; the exact B1-C4
   continuation gate failed for both counts, without establishing convergence;
2. first bound evaluator repeatability on selected and terminal checkpoints,
   including repeated bfloat16 evaluation and a float32 or deterministic-
   reduction control, before using maximum-case or marginal-admissibility
   differences; this item is now complete as B1-C5-A;
3. on retained checkpoints, separately preregister and measure whether exact-
   input predictions and the response to common propagated inputs still change
   after scalar one-step loss is nearly flat; keep this distinct from hidden-
   feature similarity alone;
4. if longer-compute attribution is still desired later, preregister an
   81,920-step representation-evolution study whose schedule-aware matrix keeps
   fixed-update, fixed-exposure, scheduler-phase, and checkpoint-response
   questions separate rather than treating passes or updates as interchangeable;
5. freeze the resulting compute budget before paired PCFNO, seed replication,
   and any bump FFNO arm. A bump FFNO requires an explicit common-
   representation contract; paper-faithful FFNO remains a PlanarDet baseline;
   and
6. do not diagnose capacity from the present gate. A later capacity diagnosis
   requires a separately evidenced compute plateau while the fixed
   seen--validation gap remains small.

#### B1-C5: numerical floor and late deployed-map dynamics

This is an inference-only diagnostic on retained B1-C4 checkpoints. It creates
no new training checkpoint, does not reselect on the 28 outside trajectories,
and does not open the historical test population.

`B1-C5-A` first measures the evaluator floor. Run three fresh independent
processes with `amp=bf16` and one with `amp=none` on the same device and source
archive. The primary cells are selected and terminal checkpoints for
`n={128,256}`; sentinel cells emitted by the frozen B1-C4 evaluator are
secondary. The float32 arm is a precision control, not ground truth. The
analysis must verify identical checkpoint, data, split, outside-key,
boundary-policy, environment, and source-code bindings before comparing:

- aggregate outside all-call and H79 relative L2;
- every trajectory's H79 relative L2 and `n=128` versus `n=256` winner;
- selected-to-terminal direction within each count;
- physical-admissibility count, first violation cause, and first violation
  call; and
- the identity of any maximum-disagreement visualization case.

The selected aggregate count ordering is numerically robust only if all four
executions preserve it and the largest within-cell aggregate H79 range is at
most 25% of the smallest cross-count H79 gap. A selected-to-terminal direction
is robust only when its sign agrees in all four executions. A per-case winner
or maximum-case identity is usable only when it agrees in all four. An
admissibility event is stable only when its case/cause agrees and its first-call
range is at most one. Failure of one criterion bounds only that observable; it
does not invalidate population means or make float32 authoritative.

After `B1-C5-A` closes, `B1-C5-B` may compare retained step-20,480, selected,
and terminal maps in float32 on the same outside cases. For each count and
checkpoint it will evaluate both exact reference inputs and one common frozen
propagated-input path generated by that count's selected checkpoint. It will
report exact-input prediction defect, learned-residual drift from the selected
map, propagated-input learned-residual response, and selected-to-terminal
functional drift. The common-input comparison is primary; each checkpoint's
own rollout is not a representation comparison because both input path and map
change. Hidden-feature similarity is optional and cannot replace functional
map-response evidence. This stage is descriptive unless its own numerical
floor and directional gate are registered after `B1-C5-A`.

`B1-C5-A` stops before scientific interpretation if a bound input or
environment differs, any rollout has a hard numerical failure, or fewer than
three bfloat16 and one float32 executions complete. Neither B1-C5 stage
authorizes 81,920-step training, paired architectures, new seeds, or test
access.

#### B1-C5-A retrieved result (2026-08-23)

The registered matrix completed on one RTX 5090 under one source archive: three
fresh independent-process bfloat16 evaluations and one `amp=none` float32
control. All 448 primary case/checkpoint/execution rollouts reach H79 with zero
hard failures. No checkpoint was reselected and the historical test population
remained unopened. The 199-member remote retrieval manifest and all five
evaluator/analysis artifact manifests plus five source snapshots rehash locally
with zero mismatches. The retrieval contains 200 files including its own
manifest and `960,928,141` bytes.

The primary preregistered aggregate gate passes:

- selected bfloat16 H79 means are `0.03880899` for `n=128` and `0.04474342`
  for `n=256`; float32 gives `0.03957538/0.04528201`;
- every execution favors `n=128`. The four cross-count gaps are
  `0.00570664--0.00612945`. The largest within-cell four-execution H79 range is
  `0.00088932`, or `15.58%` of the smallest gap, below the frozen `25%`
  ceiling. Thus the selected aggregate `n=128 < n=256` ordering is numerically
  robust under this evaluator contract;
- the selected-to-terminal H79 direction is also stable in all four
  executions. Bfloat16 means worsen `1.98%` for `n=128` and `4.06%` for
  `n=256`; float32 worsens `1.71%/2.10%`;
- this is horizon-dependent, not uniform degradation. From selected to
  terminal, bfloat16 H20 improves `4.06%/5.15%` and all-call error improves
  `0.90%/0.11%`, while H79 worsens. Fixed-validation one-step error changes
  only `-0.30%/+0.02%`. Later updates therefore move the deployed response in
  a way that helps early recurrence but hurts the tail; they do not exhibit a
  sudden conventional seen--validation gap; and
- the formal four-execution result is consistent with the earlier clean
  B1-C4 audit on the exact same checkpoint, normalizer, split, data manifest,
  and 28 outside cases. It replaces single-execution values when stating the
  evaluator-repeatability conclusion, but does not turn one seed into a
  scaling law.

Local observables have narrower claim scope:

- 23 of 28 selected per-case count winners agree in all four executions:
  19 stable `n=128` wins and four stable `n=256` wins. Five winners are
  precision-sensitive;
- the maximum-disagreement case is not stable: trajectory 83 in three
  executions and trajectory 152 in one. Neither may be promoted as the unique
  worst mechanism case; and
- 104 of 112 case/count/checkpoint event rows are stable, but 102 are stable
  no-violation rows, only two are stable violation rows, and eight event rows
  are precision-sensitive. Admissible counts vary in three of four cells even
  though every rollout remains finite and completes. This supports treating
  strict admissibility as a secondary diagnostic rather than overriding
  rollout error for checkpoint selection; it is not a physical-conservation
  result.

Retained provenance is launcher source commit
`db3b4f7c512034a63cfe64739c6b52f12f286f01`, source-archive SHA-256
`893f33e5af25a8e3eeee4388688f222c84725b5f0f5d6244d610975fe414c52e`,
retrieval-manifest SHA-256
`0492c2fbdccf5e059361e545bbb861a675ceffb2548c173480299994a7dbd55b`,
analysis summary SHA-256
`e341e714fa2b69daf3e992ffa507cb42e798404fc4c96be9b9bfdc49183f5895`,
and analysis artifact-manifest SHA-256
`b267e4049a538e23f9ca9c0d2fa404057b7ea35e57243f52ae77df3b27390b84`.
The ignored result root is
`artifacts/time_dependent_no/d094_b1_c5_repeatability_20260823a`.

The reproducible visualization packet contains the H20--H79 crossover and a
four-panel aggregate/per-case/event summary in PDF and 300-DPI PNG. Its ignored
root is
`artifacts/time_dependent_no/d094_b1_c5_repeatability_visualizations_20260823a`;
manifest SHA-256 is
`4076e66ada24b3c229cf0c087bd9700d101ba94d666a2028a868df41481e0cca`
and renderer SHA-256 is
`494d3e153622f745b7f68625d8608b304585960357f10122d50ab377e880ff77`.
All four declared figure records rehash and both PNGs pass visual inspection.

#### B1-C5-B fixed-map response contract (registered 2026-08-23)

The owner authorized the exact inference-only continuation after reviewing
B1-C5-A. It creates no checkpoint, changes no normalizer, performs no
checkpoint selection, and does not open the 20 historical test trajectories.
The population remains the ordered 28 outside-selection development cases.

For each count separately, let `G_c` be the checkpoint-native deployed map,
including its frozen causal input and output boundary closure. Let
`F_c(x) = G_c(x) - x` be the denormalized learned conservative-state update.
The selected checkpoint first generates one autonomous path
`x^s_(t+1) = G_s(x^s_t)` from exact frame zero. The step-20,480, selected
step-38,400, and terminal step-40,960 checkpoints then receive two fixed input
sets at every call:

1. exact reference `U_t`, scored against `U_(t+1)`; and
2. the same count-specific selected path `x^s_t`, also scored against
   `U_(t+1)` for a one-call common-path response comparison.

Candidate outputs in item 2 never feed back. Therefore
`F_c(x)-F_s(x)=G_c(x)-G_s(x)` is a same-input functional-map difference. Each
count retains its own registered normalizer and component scale; functional
drift is compared only among checkpoints within the same count. Cross-count
map-drift magnitudes are descriptive, not a common-coordinate representation
comparison. Bump node weights remain reconstructed diagnostic proxies, not
physical control volumes.

The evaluator reports for every case, call, checkpoint, and input view:

- candidate and selected-map proxy-weighted, component-scaled state relative
  L2 against the same next reference state;
- exact-input prediction defect;
- pooled learned-residual/map-drift RMS from the selected map, and that drift
  relative to the selected learned-residual RMS;
- the signed candidate-minus-selected error change and its exact quadratic
  decomposition into selected error, error--drift cross term, and drift
  energy; and
- finiteness and Euler admissibility without using admissibility as the first
  selection priority.

The own numerical floor is paired and frozen before outcome access. Execute
the full matrix twice in fresh sequential processes, named `fp32_1` and
`fp32_2`, with no autocast on the same device, source archive, checkpoint
bytes, data/split, outside-key order, boundary policies, and runtime
environment. Within each execution the selected map is called again in a
two-input batch on its already generated selected path. The pooled replayed
selected-map drift RMS divided by pooled selected learned-residual RMS must be
at most `1e-4` in each execution; every output must be finite; every H79 path
must complete; and the quadratic identity must close to relative `1e-10`. A
physical violation is reported but is not a hard failure when the finite path
completes.

The common-input comparison is primary. For terminal versus selected at calls
20 and 79 and for both counts, define the signed effect as the mean
candidate state relative L2 minus the mean selected-map state relative L2 on
identical selected-path inputs and reference targets. A signed effect or
functional-drift magnitude is numerically resolved only when the two process
values have the same nonzero sign where applicable and

`(maximum - minimum) / smaller_absolute_magnitude <= 0.25`.

The fixed directional classification is:

- `resolved_common_input_functional_map_crossover` only when all four primary
  cells are resolved, terminal improves call 20, and terminal worsens call 79
  for both counts;
- `resolved_common_input_response_does_not_match_full_crossover` when all four
  cells clear the floor but that early-help/tail-harm sign pattern does not;
  this would route interpretation toward changed paths or repeated
  self-composition rather than erase functional map drift; and
- `numerically_unresolved` when any primary signed effect or drift magnitude
  fails its paired floor.

Early `1--20` and tail `61--79` windows, exact-input response, step-20,480
response, common-versus-exact drift ratios, per-case signs, and admissibility
are supporting diagnostics. Hidden activations are deliberately omitted: this
stage asks about deployed function values, not representation similarity.
Input/source/binding drift, any nonfinite proposal, an incomplete path, a
same-map replay failure, or an algebraic-closure failure stops scientific
interpretation. Neither outcome automatically authorizes 81,920-step training,
PCFNO/FFNO, another seed, or historical-test access.

#### B1-C5-B-A infrastructure stop (2026-08-23)

The first source archive passed its 14 remote CPU tests and entered the
separately labeled one-case full-grid engineering smoke. It generated model
calls on the first registered outside-development case, then stopped at the
first metric row because the retained checkpoint exposes `state_scale` with
shape `[1,1,4]` while the new metric guard required literal shape `[4]`. No
registered `fp32_1`/`fp32_2` execution started, no metric row or scientific
receipt was serialized, no outcome was interpreted, no training occurred, and
the historical test population remained unopened. The same smoke also exposed
a non-writable memory-map view warning before device transfer.

This is implementation provenance, not map-response evidence. The corrected
source accepts any four-value scale-buffer shape before canonical reshaping and
copies the reference slice before tensor conversion. The synthetic semantic
test now uses the retained `[1,1,4]` layout. A new source archive and remote run
root are required; the failed source identity is not overwritten. Its source
commit/archive SHA-256 are `e3671ab` and
`a9b375d833c3760786c789f95cef841c3e4bacab8a8e6b05c3dcb95c892d9c66`.
The retrieved preflight-log and stop-record SHA-256 values are
`a0d1c308cb396f3ec7c0731de98302417e6327b49fbdf933fbbd9d78f84b4317`
and `bb50dcd8669d7dd930aefa26055c673ad443e0fdee3bc2b6a84ea3a846012531`.

#### B1-C5-B retrieved result (2026-08-23)

The corrected source completed the registered `fp32_1` and `fp32_2`
executions in fresh sequential processes. Each execution contains 26,544
case/call/checkpoint/view rows. All candidate outputs are finite, every selected
path completes H79, all 28 outside-development cases are present, and no
checkpoint reselection, training, or historical-test access occurred. The
largest pooled same-map replay fraction is `9.02622e-5`, below `1e-4`; the
largest quadratic-closure residual is `6.54107e-16`, below `1e-10`.

All four primary terminal-versus-selected common-input effects are resolved,
but only the call-20 signs match the autonomous crossover:

| Count | Common-path call | terminal minus selected mean relative L2, `fp32_1/fp32_2` | terminal / selected error | Frozen expected sign |
| --- | ---: | ---: | ---: | --- |
| 128 | 20 | `-4.45288e-5/-4.45184e-5` | `0.998342/0.998343` | pass: terminal helps |
| 128 | 79 | `-5.14261e-5/-5.14272e-5` | `0.998701/0.998701` | fail: terminal helps rather than harms |
| 256 | 20 | `-8.40053e-5/-8.39981e-5` | `0.997063/0.997063` | pass: terminal helps |
| 256 | 79 | `-5.30254e-5/-5.30159e-5` | `0.998829/0.998829` | fail: terminal helps rather than harms |

The registered classification is therefore
`resolved_common_input_response_does_not_match_full_crossover`. This is not a
numerical-floor failure. Terminal-map drift on the common selected path is
small but resolved: at calls 20/79 it is `0.89%/2.12%` of the selected learned
residual for `n=128` and `0.79%/2.34%` for `n=256`. At call 79 its error--drift
cross term is favorable for both counts. The terminal map improves all 79
common-path call means for `n=128`; for `n=256` it improves 54/79 calls and
17/19 tail calls. At call 79, 24/28 `n=128` cases and 23/28 `n=256` cases
improve on common inputs.

This reverses the autonomous result from B1-C5-A. In its FP32 control,
terminal-versus-selected autonomous H20 changes by `-4.37%/-8.40%` for
`n=128/256`, while H79 changes by `+1.71%/+2.10%`; all-call error changes by
`-1.55%/-3.24%`. On the fixed selected path, the corresponding one-call
changes at calls 20/79 are only `-0.166%/-0.130%` for `n=128` and
`-0.294%/-0.117%` for `n=256`. Fourteen of 28
cases in each count have the decisive sign pattern: the terminal map helps at
call 79 on the selected path but the terminal checkpoint harms at H79 on its
own recurrent path. Endpoint admissibility is unchanged on the common path;
the autonomous `n=256` terminal path loses one admissible case. Thus the full
tail harm is not a pointwise defect of `G_terminal` on selected-path states. It
requires the changed recurrent path, repeated self-composition, or their
interaction.

The step-20,480 context further separates reference-state and deployment-state
fidelity. Relative to selection it is worse on exact reference inputs for all
79 calls in both counts (all-call ratios `1.11850/1.11999`), yet on the selected
path its call-79 ratios are `0.993368/0.999000`. Conversely, selected-to-terminal
exact-input all-call changes are `-0.308%/+0.037%`, closely matching B1-C4's
fixed-validation changes `-0.298%/+0.017%`. One-step reference fit is therefore
internally consistent but does not order self-composed H79 behavior.

The local result-to-claim verdict is **yes for the bounded functional
statement, pending independent Codex review**: under these retained maps,
population, and FP32 evaluator, autonomous tail harm is path/composition
dependent rather than terminal-map harm on the selected path. It does not
identify a hidden-representation, optimizer, data-count, capacity, gradient,
or convergence cause.

Retained provenance is source commit `b0be12c`, source-archive SHA-256
`1a77aee8ed593de07805b0d3eddfe01d465e2de4f99980d2c8857e2dd5198e6d`,
analysis-summary SHA-256
`45e3349c952173574f2219a17d0245a7aca7183cfd41a066c63c8f75bcbb74e4`,
analysis-artifact-manifest SHA-256
`c62f08d5a7414363080a5ebbaf3fab788698259f16c01d752269196fcb5c195b`,
and retrieval-manifest SHA-256
`d749ba95bca4fa084ca6a7d760a4b13adc5058e97650fb5f9ee185fab7b9b8c7`.
The retrieval manifest binds 115 files and `125,482,211` bytes with zero local
mismatches; all evaluator, analysis, and source-snapshot manifests close.
The original result manifest has one preserved operational mismatch:
`logs/finalize.log` was hashed while empty and then received 51 bytes of
finalizer output. The post-run retrieval manifest binds the final bytes; no
scientific artifact changed. A future launcher should exclude a live finalizer
log from its own inventory or finalize logging before hashing.

The minimum decisive continuation is an inference-only symmetric 2x2
map--path decomposition: evaluate selected and terminal maps on both selected
and terminal recurrent paths, then close map, path, and interaction terms. A
prospective result is that a harmful terminal-path term must exceed the
favorable selected-path map term at H79, especially in the 14 sign-flip cases.
Failure would route the explanation to a map--path interaction rather than a
simple path displacement. This result does not itself authorize that study,
81,920 steps, paired architectures, new seeds, or historical-test access.

### B2: Official PlanarDet generator reproduction

Purpose: establish that newly generated trajectories belong to the same benchmark.

- obtain the official case or an author-supplied additional-data bundle;
- reproduce at least three released conditions, including an interior and two corners;
- compare saved times, coordinate/crop hashes, field definitions, wave speed, front position, cell-size statistics, per-channel ranges, and cumulative-`pMax` semantics;
- quantify solver/numerical variability using repeated executions if the solver is nondeterministic; and
- freeze the generation source manifest and environment.

Gate: exact metadata/crop closure and preregistered tolerances for detonation speed, cell scale, field ranges, and snapshot cadence. Passing visual similarity alone is insufficient.

### B3: Generate and audit the 320-trajectory PlanarDet population

Purpose: create the independent population required for C1.

- run a 16-trajectory pilot to measure actual failure rate, wall time, storage, and postprocessing cost;
- review the pilot before authorizing the remaining campaign;
- generate train/validation/test identifiers from the frozen parameter design;
- validate every trajectory for finite fields, monotone time, solver completion, coordinate/crop identity, admissibility, detonation propagation, and nondegenerate cell structure;
- publish only small metadata/hash manifests in this repository; and
- keep raw fields, solver restarts, and sealed-test arrays outside git.

Gate: all 320 manifest entries close, failed simulations are rerun under the same frozen contract, and test arrays remain inaccessible to training/evaluation code.

### B4: PlanarDet architecture--data--compute surface

Primary factors:

- architectures: PCNO, PCFNO, FFNO;
- trajectories: `n={8,16,32,64,128,256}`;
- initialization seeds: `{0,1,2}`;
- compute checkpoints from one frozen training schedule; and
- active-only normalization.

Run order:

1. seed-0 screen at `n={16,64,256}` for all three architectures;
2. full ladder for seed 0 if healthy;
3. seeds 1 and 2 for every retained cell;
4. small/base/large capacity controls for PCNO and FFNO at `n=256`; and
5. one final sealed-test evaluation only after claims and selection rules are frozen and explicitly authorized.

The primary comparison uses the same temporal target and objective. A direct-versus-residual study is secondary and begins only after the data--architecture surface is complete; it is no longer a primary D094 claim.

## 9. Comparable metrics

At every diagnostic checkpoint, compute per trajectory:

- truth-input one-step grouped MSE in normalized coordinates;
- truth-input decoded NPE, total and by physical group;
- free-rollout horizon error and correlation;
- one-step residual magnitude, bias, cosine, and spectral error;
- detonation front position/speed, cumulative-`pMax` error, cell scale, and high-pass energy;
- finite, transform-domain, boundedness, admissibility, and monotonicity checks; and
- empirical recurrent gain.

For bump, use the maintained native-mesh rollout, shock/front, boundary, and proxy-total diagnostics, preserving the existing caveat that graph weights are not physical conservation measures.

Training and validation curves must use the same statistic when discussing overfitting. Online stochastic training loss may still be plotted as an optimization trace, but not as the y-axis partner of decoded validation NPE.

## 10. Statistical analysis and decision rules

The independent uncertainty units are initialization seed and held-out trajectory. Nodes and frames are not replicates.

- Show every seed and held-out trajectory point.
- Report paired architecture ratios within the same subset, seed, checkpoint, and evaluator.
- Report median and range over three seeds; bootstrap only over seed/trajectory units.
- Plot raw curves before fitting any log-data scaling law.
- Report both fixed optimizer compute and fixed window-equivalent exposure.

An over-optimization event requires three consecutive checkpoints where comparable held-out error is at least 25% above its earlier best while comparable seen-train error falls by at least 5%. A decode or boundedness failure is a separate, stronger instability event.

Interpretation table:

| Pattern | Supported interpretation |
|---|---|
| transition follows shifted LR timing | optimization schedule controls onset |
| transition aligns by `X(s,n)` under the corrected sampler | repetition/exposure contributes |
| seen and held-out errors separate smoothly as `n` falls | classical condition overfitting |
| seen error falls while recurrent gain/validity jumps | learned time-step map becomes unstable |
| architecture gap shrinks with `n`, then width helps at `n=256` | data-to-capacity bottleneck shift |
| architecture gap persists after exposure, capacity, and compute controls | inductive bias remains limiting |
| all gaps close under stable scaled training | data/recipe dominate in this regime, not universally |

## 11. Cost envelope

### PlanarDet data generation

- Reported mean: 6,653 core-hours per trajectory.
- Target 320: approximately 2.129 million core-hours.
- Processed released-size estimate from the pinned open bundle is on the order of 0.37 GB per trajectory, or roughly 118 GB for 320, excluding raw CFD fields and restart files.

### PlanarDet neural training

Using the observed 5,000-step times on one RTX 5090, the full six-count, three-seed base matrix is approximately:

| Architecture | Runs | Approximate GPU-hours |
|---|---:|---:|
| PCNO | 18 | 309--313 |
| FFNO | 18 | 34 |
| PCFNO | 18 | 22, projected |
| total | 54 | about 365--369 |

This excludes capacity controls, evaluator time, and failed runs. B4 is therefore staged; the seed-0 three-count screen precedes the full matrix.

The bump cost is measured by a native-graph smoke before queueing because its node counts and trainer differ from PlanarDet.

## 12. Immediate execution order

1. B1-C0 retrieval/rehash and the registered 12-checkpoint evaluator are
   complete; B1-C1 retains fixed-compute, exact corrected-exposure, selected,
   terminal, and outside-audit views separately.
2. B1-C4 execution, its four-checkpoint-per-count outside audit, and the
   diagnostic visualization packet are complete and rehashed. Both strict
   continuation gates fail; do not auto-launch 81,920 or 163,840 steps, and do
   not treat the routing result as evidence of optimization or representation
   convergence.
3. Close aggregate and per-case evaluator repeatability with repeated
   bfloat16 and float32 or deterministic-reduction controls before routing a
   maximum-case or marginal-admissibility mechanism claim.
4. Use retained checkpoints to test whether exact-input predictions and
   propagated-input response keep changing after scalar one-step loss is nearly
   flat.
5. Retain 81,920 steps as a later representation-evolution study, not the
   current priority. Before it runs, preregister fixed-update, fixed-exposure,
   scheduler-phase, and checkpoint-response controls separately. Freeze that
   budget before paired PCFNO, any representation-explicit bump FFNO, and
   initialization-seed replication; retain positive, null, and reversed seeds.
6. Keep the retired `n<=7` factorial sweep unlaunched and keep exact resume
   closed unless a separate replay/resume gate is implemented and tested.
7. In parallel through human coordination, request the official PlanarDet
   case, additional trajectories, and an HPC cost/allocation answer. Do not
   generate PlanarDet from the paper description alone.
8. Do not start the full PlanarDet architecture surface or its matched and
   paper-faithful FFNO arms until B2 and B3 close.
9. Keep the historical test population sealed until the three-seed development
   claims and final checkpoint-selection rules are frozen and separately
   authorized.

## 13. Minimum paper-level evidence

For C1:

- at least 256 training trajectories from one qualified hard PDE family;
- 32 or more open held-out trajectories;
- three architectures and three seeds;
- fixed-compute and fixed-exposure views;
- comparable seen/held-out metrics and long rollouts;
- a high-exposure capacity control; and
- no development access to the sealed test population.

For C2:

- corrected unique-exposure accounting;
- retained pre-transition, transition, and late checkpoints;
- schedule/objective interventions from a shared state;
- multiple data counts and seeds; and
- recurrence/validity evidence alongside loss.

Until these conditions are met, the strongest defensible D093 statement is descriptive: every selected seven-condition cell has lower truth-input error than its three-condition counterpart, only FFNO has a lower selected free-rollout sum, and the family ordering differs. This one-seed, one-open-validation pattern does not identify an exposure, architecture, optimization, or recurrence cause.

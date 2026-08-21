# D094: Hundreds-Trajectory Data--Architecture--Optimization Scaling

Date: 2026-08-20
Status: the six-cell seed-0 `n={16,64,256}` calibration pilot is reported complete remotely, but its packets are not retained locally and the report is not a local archival closeout; the error-first `n=256` long-schedule gate below is registered but not authorized to launch; the historical test population remains sealed
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

| Architecture | Schedule | Selected step | Fixed seen one-step | Fixed validation one-step | Rollout all-call mean | H79 | Physical-admissibility rate |
|---|---|---:|---:|---:|---:|---:|---:|
| PCNO | prefix-tail | 20,480 | 0.0124163 | 0.0126211 | 0.0948993 | 0.141995 | 1.000 |
| PCNO | stretched | 20,480 | 0.0112578 | 0.0114138 | 0.0525120 | 0.0791166 | 1.000 |
| PCFNO | prefix-tail | 20,480 | 0.0170696 | 0.0174813 | 0.0896706 | 0.129719 | 0.875 |
| PCFNO | stretched | 15,360 | 0.0151365 | 0.0154052 | 0.0824803 | 0.121960 | 0.875 |

The stretched/prefix selected all-call rollout ratios are `0.5533` for PCNO
and `0.9198` for PCFNO. PCNO/PCFNO is `1.0583` under prefix-tail but `0.6367`
under stretched, so this one seed exhibits a large schedule--architecture
interaction. It is not yet a replicated architecture result. No arm meets the
registered three-checkpoint one-step over-optimization definition. PCFNO
stretched instead shows a different event: from its selected step 15,360 to
the terminal step, fixed seen and validation one-step errors improve while
all-call rollout error worsens. This is one-step/recurrent-objective divergence,
not classical train--validation separation and not yet a causal mechanism.

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

1. Keep the retired `n<=7` factorial sweep unlaunched.
2. Treat the B1-A metadata/source packets as locally retrieved and rehashed;
   keep its single-seed result bounded to schedule routing.
3. Run the registered B1-B 28-trajectory outside-selection audit without
   opening the historical test population or changing checkpoints.
4. If B1-B closes numerically, use its winner for the seed-0
   `n={8,16,32,64,128}` fixed-compute ladder and reuse the corresponding B1-A
   n=256 endpoint. Do not infer scaling from the earlier strict-physical
   5,120-step pilot rollouts because they compare different valid-prefix
   populations.
5. Keep exact resume closed; no optimizer/scheduler replay is needed for B1-B
   or the fresh fixed-compute ladder.
6. In parallel through human coordination, request the official PlanarDet
   case, additional trajectories, and an HPC cost/allocation answer. Do not
   generate PlanarDet from the paper description alone.
7. Do not start the full PlanarDet training surface until B2 and B3 close.

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

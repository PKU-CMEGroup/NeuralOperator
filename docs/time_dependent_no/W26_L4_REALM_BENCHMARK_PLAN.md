# W26-L4 REALM Benchmark Audit And Experiment Plan

Status: A0 and D088 P1a/P1b are complete. The exact IgnitHIT train/validation
tree and reference replay passed under the recorded restricted-internal-research
disposition. The owner authorized P1c A3-smoke only on 2026-08-12 and selected
an alternate personal GPU workstation while AutoDL is occupied. Full training,
residual comparison, and sealed-test evaluation remain unauthorized.

Owner: the time-dependent neural-operator research line. This file is the
line-specific execution source of truth for W26-L4. The compact weekly tracker
records only stage status and links here; a second experiment-plan/tracker pair
would duplicate this contract.

## Decision

Do not generate more supersonic-bump frames by default. D087 already closes the
reference-backed H79 stability question, while H160/H320 bump continuation would
be truth-free survival evidence. The next benchmark-facing program is:

1. `IgnitHIT` as the first regular-grid REALM case;
2. `PlanarDet` only after the IgnitHIT baseline and metric-replay gates pass;
3. `ObstacleDet` as the first irregular candidate only after its released
   coordinate/connectivity/volume/boundary contract and feasible memory path are
   verified; and
4. no 3D case until both selected 2D stages pass.

This is a staged validation program, not a commitment to run the full REALM
corpus. REALM contains 11 reacting-flow cases, not 11 independent PDE families.

## Claim Map And Paper Storyline

At most two claims are active.

| Claim | Minimum convincing evidence | Explicit anti-claims |
| --- | --- | --- |
| C1, primary: a normalized-state residual parameterization improves long-horizon prediction over a matched direct-next-state FFNO on IgnitHIT. | Same released train/validation population; identical FFNO-M capacity, initialization family, preprocessing, recurrence exposure, optimizer schedule, presentation budget, precision, selection rule, and evaluator; three preregistered seeds after a faithful direct baseline; lower case-first paper-formula validation error and improved decoded structure/admissibility without a registered no-harm failure. | The residual is not a conservative PDE residual. A win does not establish PCNO superiority, conservation, general stability, or benchmark-wide superiority. |
| C2, supporting and conditional: the same target change remains useful on discontinuous PlanarDet. | C1 gate passes; a separate PlanarDet preregistration resolves `pMax`, grid/frame, and boundary semantics; matched three-seed comparison improves validation error and front/shock metrics without smoothing or peak bias. | IgnitHIT alone does not establish shock robustness. PlanarDet does not establish irregular-geometry or boundary-condition learning. |

The intended paper story is intentionally simple: first reproduce a strong
external direct-state baseline; then change only the predicted target and
recurrence algebra; finally ask whether that change improves both the benchmark
metric and the physical structures that the normalized aggregate can hide.
Architecture modification is conditional on a diagnosed failure, not a default
sweep.

## A0 Primary-Source And Artifact Bindings

### Paper and code

- Paper: `arXiv:2512.18595v2`, revised 2026-02-02, 52 pages. The retained local
  PDF has SHA-256
  `148971e6eef3782f1562b0eaa045ce8608ded673ffa13c0bfd99acbc39f5b137`.
- Official repository: `https://github.com/deepflame-ai/REALM`, pinned at commit
  `7d00523dbda7823efa03c20be36692c947a417b5`.
- The Git tree contains no `LICENSE` or `COPYING` file and the GitHub API reports
  no detected license. The README's license badge is not a license grant.
  Dataset cards also expose no license field. Redistribution and reuse terms are
  unresolved; public accessibility alone is insufficient. A1 may design an
  independent adapter from documented formats, but no source copy or trajectory
  acquisition advances until the owner accepts a documented license disposition
  or the authors publish/confirm terms.
- The repository README cites a different arXiv identifier than the retained
  paper. The paper identifier and exact code commit above are authoritative for
  this line.

The exact historical runtime-chain Git blobs at that commit are:

| Path | Git blob | Bytes | Role |
| --- | --- | ---: | --- |
| `XDEBench/train_xdebench_rollout.py` | `e14c51239faace509ebf618117bf1f25c24e06a1` | 4,844 | entry point, channel dimensions, seed |
| `XDEBench/runner/train_rollout.py` | `6f51e0141d50b52680e93b488976c8e63680a34f` | 11,479 | train, selection, recurrence, source metric |
| `XDEBench/models/xde_model.py` | `26836ac2299f19664696b305c536ac729a56b5d9` | 2,779 | Box-Cox/z-score encode and decode |
| `XDEBench/models/methods/ffno/ffno.py` | `a0de942b1826be2f422361b2dcebaf82e1f35f8b` | 4,164 | FFNO model |
| `XDEBench/utils/tools.py` | `2d3be00ed99a0b2fb39e43276209212ec5a74ed5` | 8,340 | loader, seed, coordinates |
| `XDEBench/data/dataset.py` | `34a373a20661ae528734c96d7099cd334095ffc0` | 6,200 | dataset helper |
| `XDEBench/evaluator/metrics.py` | `790fbd7ca0d414c5e88f2e099eb64faf25b0e7fa` | 5,131 | auxiliary image metrics |
| `tutorial/multi_gpu_launcher_rollout.py` | `2839c9bdeeda17df8e767808fcdf10f9ec005e47` | 6,919 | mutable example launcher, not an immutable experiment command |

No released checkpoint, training log, or IgnitHIT-specific immutable launch
command is present in the pinned tree. The numerical weight decay, training
iteration count, and original random seed for the paper's selected IgnitHIT
FFNO row are unresolved. The generic CLI defaults (`seed=0`, `weight_decay=0`,
`num_iterations=5000`, `loss_modes=mae`) and the checked-in detonation launcher
(`num_iterations=20000`) do not establish the paper run, whose methods specify
MSE and Adam with weight decay.

### Dataset revisions, released contracts, and acquisition cost

Only public API metadata and small source/statistics files were read during A0;
no trajectory array was downloaded.

| Case | Hugging Face repository and revision | Released shape / fields / frames | Split | Total bytes | Train + validation bytes | Conditional role |
| --- | --- | --- | --- | ---: | ---: | --- |
| IgnitHIT | `realm-bench/realm-bench-IgnitHIT@a0736b4d8c6c58a2688127e32addc30085e824c3` | `128 x 128`, 12 fields, 30 frames | 26 / 5 / 5 | 641,568,615 | 552,014,385 trajectory bytes; 8,634 metadata bytes | first case |
| PlanarDet | `realm-bench/realm-bench-PlanarDet@b084b6fc2e624e4ee5e44b88dd87d628bcbb9a4b` | `832 x 384`, 13 fields, 50 frames | 7 / 1 / 1 | 3,346,430,471 | 2,940,739,390 trajectory bytes; 20,077 metadata bytes | second case after C1 gate |
| ObstacleDet | `realm-bench/realm-bench-ObstacleDet@0350c42cb7b552f8c631ba76ac788236cb0f6499` | 310,224 nodes, 5 fields, 51 frames | 4 / 1 / 1 | 26,394,216,987 | 22,285,242,265 trajectory bytes; 26,593,067 metadata bytes | preferred irregular audit only |
| SupCavityFlame | `realm-bench/realm-bench-SupCavityFlame@a025fb37e01df6b4732870b70c867c8cf95c8803` | 298,848 nodes, 12 fields, 21 frames | 7 / 1 / 1 | 72,826,040,926 | 64,718,684,166 trajectory bytes; 2,168,363 metadata bytes | deferred on cost |

For fail-closed acquisition, the canonical Hugging Face manifest is UTF-8 lines
`path<TAB>size<TAB>oid<LF>` sorted by path, where `oid` is the LFS SHA-256 when
present and otherwise the repository object ID. The full/open-without-test
manifest SHA-256 pairs are IgnitHIT
`28c2d19f6ef78c65bf07b97795cda70920911c6f983a8b31f9fbad2ea73cefc3` /
`85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205`,
PlanarDet
`69bafb089ef3f4b9a40f65a93df195b50615955117baabffbb7ea822d6425304` /
`b53f8195f931ac46b112a76adc9c303b0c8b312cd8b054fb3fb0dbaaebbcc31e`,
ObstacleDet
`9d5757ecb8366861e9720aa1087be147d7132e17376993f1d25dfe97b4fcea32` /
`3467da239f2841e0a5f3a9f79983f4880b18230423bd63620a6c9f8c61a9ca49`,
and SupCavityFlame
`bd58fe105ea01dfd4a64d84dd8fb2e0a2e983024b69cac9159a4016ffc7ed5f6` /
`77d9a37194b28217ee3fc980d9293125c53e3e67047d1c1a589518817257c53d`.

The exact statistics-manifest blobs are IgnitHIT
`0d36e5df83d289726109fd903d1a326086b704ab`, PlanarDet
`c2318616379cd295ab41e5be66d716430f9a1b08`, ObstacleDet
`166bafae606e0ed170cfef2b753e90207453d667`, and SupCavityFlame
`2c069e1e13682e3e25fbf0ad51a334ac2e4f8d2e`.

IgnitHIT's released fields, in order, are `H`, `H2`, `H2O`, `H2O2`, `HO2`,
`O`, `O2`, `OH`, `T`, `rho`, `Ux`, and `Uy`. The 26/5/5 group names in the
official manifest are the only allowed split. A first acquisition must include
only the 26 train and five validation LFS objects plus metadata; the five test
objects (89,545,596 bytes) remain physically unfetched until a named A4 release.

The paper, README, official statistics, and case descriptions disagree in places
about native/released resolutions and frame counts. The pinned dataset revision,
per-file LFS object IDs, and decoded `data/data.npz` contract will govern actual
execution. In particular, the PlanarDet release is `832 x 384`, 50 frames, and
includes `pMax`; its physical state and history/recurrence meaning must be
resolved before P2. The irregular releases require an explicit audit of whether
coordinates, connectivity, cell volumes, boundary labels, and causal forcing
are actually present; a node count alone is not a PCNO geometry contract.

## Historical Paper Contract Versus Released Source

The paper-formula and released-source behaviors must be reported separately:

| Contract field | Paper-formula channel | Pinned released-source channel | W26-L4 disposition |
| --- | --- | --- | --- |
| species transform | Box-Cox with `lambda=0.1`, then train-stat z-score | same form; stats helper clamps species at `1e-40`, runtime encoder defaults to `1e-8` | reproduce both; primary matched runs use one preregistered consistent epsilon after actual-data audit |
| training target | one- or two-step autoregression, final-step-only grouped MSE | `totalTimeStep=2`, `te=ts+1`, `range(ts,te)`: exactly one learned call and one-step target | never label released behavior two-step; implement real one-step and real two-call modes explicitly |
| optimizer/history | Adam with weight decay, OneCycle; selected IgnitHIT max LR `1e-3`, batch 26 | Adam/AdamW switch, default weight decay zero; example launcher 20,000 iterations | unresolved fields block a paper-faithful claim; freeze a declared reconstruction after P0 author/license decision |
| selection | minimum validation error | minimum validation error, but test is also evaluated and logged on every validation improvement | our training code cannot load or evaluate test arrays; test remains sealed through selection |
| recurrence | autoregressive model output | normalized model output fed directly to the next call | bind stage tensors and require exact next-input equality |
| normalized prediction error | grouped transformed/normalized MSE averaged over calls and cases | grouped MSE summed over calls; no horizon division in `test()` | report both as separately named metrics; paper-formula average is primary |
| structural metric | decoded per-channel spatial Pearson correlation, averaged over channels/calls | not the training runner's selection metric | independently reproduce and test the paper formula |
| efficiency | synchronized model-only inference, three repeats, median | no immutable case-specific result harness | implement only after scientific metric parity |

The selected paper row is FFNO-M, width 128, four layers, 32 modes, about
8.9365 million parameters, batch 26, nominal two-step training, and max LR
`1e-3`. It reports train/validation/test normalized errors
`0.51785 / 1.8635 / 1.8739` and decoded correlation `0.97357`. These visible
paper outcomes make any numeric reproduction tolerance pilot-informed, not an
independent confirmatory test. Because the table-scale attribution between the
paper-average formula and released-source sum is unresolved, the value `1.8635`
is used only as a released-source compatibility gate below; it is not silently
applied to both metrics.

## Frozen First-Case Systems

### Common state and preprocessing

Let `U_t` be the 12-channel released IgnitHIT state and `Z_t = E(U_t)` its
transformed/normalized state. `E` applies Box-Cox only to the first eight
released species channels and then per-channel z-score statistics computed over
train trajectories, all train frames, and all spatial points. No validation or
test value contributes to the statistics. Decode with the exact inverse.

Before any training, A1/A2 must freeze:

- array axes, channel order, coordinate normalization, time cadence, units, and
  physical-domain extent from actual metadata;
- the Box-Cox clamp and inverse-domain policy;
- model/input/output dtype, autocast, TF32, CUDA/cuDNN/determinism flags,
  PyTorch/CUDA/device versions, all seeds, and package/source manifests;
- exact train and validation LFS object IDs and hashes; and
- whether omitted species can be reconstructed. Until this is authoritative,
  report released-species nonnegativity separately and do not call their partial
  sum a full species-simplex or elemental-conservation test.

### Matched model arms

The first matched pair uses the same FFNO-M implementation and differs only in
the output parameterization:

- Direct arm: `Z_hat_t = F_theta(Z_hat_(t-1), x)`.
- Residual arm: `Z_hat_t = Z_hat_(t-1) + R_theta(Z_hat_(t-1), x)`.

Here `x` is the current static coordinate tensor only; no future truth or
future-time field is an input.

The residual is therefore a transformed-normalized state increment. It is not a
conservative-variable increment, flux residual, or PDE residual. Both arms are
trained with the same grouped next-state MSE evaluated on the reconstructed
`Z_hat_t`; training the residual arm against an independently rescaled delta
would change the objective and is prohibited in the primary contrast.

For each seed, initialize both arms from the same full parameter tensors, use
identical minibatch/time-start order, optimizer,
scheduler, gradient clipping policy, precision, presentation and optimizer-step
budgets, checkpoint cadence, and validation selection. The output-head meanings
differ, so exact equality of their gradients is not a required pairing gate.
Parameter counts must match exactly.

### Training exposure

Two exposure modes are registered, but they cannot be pooled:

1. `one_call`: one input state, one learned call, loss on the next state. This is
   the only behavior established by the pinned trainer.
2. `two_call_final`: learned call 1 feeds learned call 2 without gradient through
   call 1, and loss is computed only at call 2. This implements the paper's
   stated final-step-only two-step scheme.

P1 first reproduces `one_call`. A true `two_call_final` baseline is a separately
named sensitivity run and must use both calls for both arms. It cannot be
introduced only for the residual model.

## Metrics And Aggregation

All trajectory summaries are case-first: compute per-trajectory curves and
summaries, then aggregate cases with equal weight. Never mix shorter accepted
prefixes into a time-varying cohort mean.

| Metric | Coordinate / definition | Truth support | Role |
| --- | --- | --- | --- |
| `realm_npe_mean` | per call, sum the MSE of each present group among (`chem`, `T`, `rho`, `u`, `p`) in `Z`; average over released calls, then cases | all 29 IgnitHIT calls | primary paper-formula compatibility |
| `realm_npe_sum_source` | same grouped per-call quantity summed over calls, then cases | all released calls | released-source compatibility only |
| `corr_decoded` | spatial Pearson correlation in decoded units per channel/call, then channel/call/case means; constant-field channels are reason-coded rather than coerced | all released calls | paper structural metric |
| group and channel errors | per-call transformed MSE plus decoded relative L2 by channel/group | all released calls | failure localization |
| admissibility | native finiteness; `rho > 0`, `T > 0`, each released species `>= 0`; complete-species sum only if omitted-species semantics close | every returned state | operational/physical-domain diagnostic, not accuracy |
| boundedness | componentwise decoded magnitude relative to a train-only envelope frozen before validation scoring | every returned state | operational diagnostic; not conservation |
| front structure | temperature/OH front position, thickness, area/length, strength, and high-physical-wavenumber error with thresholds frozen from train only before model training | released truth only | anti-smearing and phase diagnostic |
| boundary band | decoded error in fixed physical boundary bands versus matched interior; no causal boundary claim without released boundary/forcing inputs | released truth only | descriptive leakage diagnostic |
| efficiency | parameters, synchronized model-only FP32/BF16 time, peak CUDA allocation; three repeats, median | no truth required | secondary cost channel |

The exact front threshold, physical bands, spectra, train-envelope quantiles,
inclusive inequalities, and handling of inverse-Box-Cox domain violations are an
A1 preregistration obligation. They must be frozen after train-only schema
inspection and before any model result is computed. High-frequency reduction
without retained front position, strength, and thickness is not improvement.

Accuracy claims stop at released truth. Recurrent extension beyond frame 29 is
not planned for the first study; if later authorized, it is survival,
admissibility, and boundedness evidence only.

## Experiment Blocks, Gates, And Run Order

### P0 — source, license, and metadata audit (`A0`, current)

Current verdict: public provenance and case costs are bound well enough to plan,
but P0 is not a data-execution pass.

Go to A1 only if this plan is reviewed. Go to trajectory acquisition only if:

1. source/dataset license terms are published or the owner records an explicit
   restricted internal-research disposition;
2. the exact IgnitHIT revision and every selected train/validation LFS object ID
   are frozen;
3. test objects are excluded from the download allowlist;
4. paper-versus-source discrepancies remain separately named; and
5. no result is presented as exact paper reproduction while seed, weight decay,
   step budget, or exposure remain unresolved.

Stop on a revision change, missing manifest object, checksum mismatch, license
rejection, or any requirement to access the test split for training/selection.

### P1a — local contract implementation (`A1`, next request)

No dataset array, checkpoint, GPU, remote host, or sealed population is used.
Implement an independent manifest/adapter/metric surface and synthetic tests.
Exact proposed files:

- `docs/time_dependent_no/W26_L4_REALM_PREREGISTRATION.md`;
- `utility/time_dependent_no/realm_benchmark.py`;
- `scripts/time_dependent_no/audit_realm_benchmark.py`; and
- `tests/time_dependent_no/test_realm_benchmark.py`.

Focused CPU gates:

1. Box-Cox and inverse round trip in float64, including zero/clamp and inverse
   domain behavior;
2. train-only mean/std axes and zero-variance policy;
3. exact channel/group slicing and paper-average versus source-sum metric
   identities;
4. direct and zero-residual recurrence identities, and two-call final-step
   detachment semantics;
5. no future truth in either recurrence;
6. exact split allowlist with a hard failure on any test-path open;
7. canonical source/data/runtime manifest hashing and LFS object validation;
8. case-first aggregation, missing/constant-channel reason codes, and native
   nonfiniteness propagation; and
9. synthetic front, boundary-band, admissibility, boundedness, and spectrum
   invariants plus `--help` and Ruff checks.

P1a passes only when all focused tests pass, every unresolved scientific field
is explicit, and the preregistration has no stable D-series ID. A bounded
noncollision search currently finds no `D088`, but allocation is deferred until
the A1 source/population/metric review. The proposed request is “allocate D088
to W26-L4 only if that review passes,” not an allocation by this plan.

### P1b — train/validation acquisition and replay (`explicit download approval`)

Download only IgnitHIT metadata plus train/validation objects at the pinned
revision. The allowlisted trajectory payload is exactly 552,014,385 bytes; test
payloads remain absent. Produce a local artifact manifest, schema report,
train-stat report, and decoded-reference diagnostic report. Do not commit data,
arrays, statistics derived from private local paths, or machine paths.

Required real-data gates before GPU training:

- all objects and array axes match the frozen manifest;
- every raw/decoded train and validation state is finite;
- species, density, temperature, coordinates, time cadence, and units are
  reconciled or explicitly unresolved;
- encode/decode closure and independently computed metrics meet preregistered
  float64/FP32 tolerances;
- no test object exists locally; and
- the baseline command/config is frozen without reading validation outcomes
  beyond schema/metric-replay fixtures.

### P1c — GPU smoke and faithful direct baseline (`A3`, conditional)

First run a synthetic-shape forward/backward memory probe, then one train-batch
and one validation-trajectory real-data smoke. Freeze the measured step time,
peak memory, feasible microbatch/accumulation policy, and projected full cost.
Changing effective batch 26 requires explicit review.

The authorized smoke is attempt
`d088_realm_ignithit_p1c_personalgpu_20260812a`: FP32 on exactly one visible RTX
5060 Ti, seed `20260812`, microbatch 1 with 26 ordered one-case accumulations,
exactly one Adam step over train frame 0 to 1, then an H29 rollout from frame 0
for first validation group `phi=_t_15_3_t`. It has a 900-second hard cap, loads
no checkpoint, writes no checkpoint, and cannot access a test object. The exact
architecture, runtime flags, optimizer, gates, and anti-claims live in the D088
preregistration. The SSH alias remains private.

The faithful direct baseline uses FFNO-M and the frozen reconstructed contract.
Because the original seed/history are unresolved and the paper outcome is
visible, this is a reproduction attempt, not an independent confirmatory result.
The pilot-informed go gate is:

- finite completion and exact manifest/runtime closure;
- parameter count within 0.5% of 8.9365M, or a documented source-exact reason;
- released-source-sum validation error no worse than
  `1.25 * 1.8635 = 2.329375`, while the separately named paper-formula average
  is also reported;
- no systematic decoded inverse failure; and
- no more than the preregistered 12 GPU-hour hard cap for one full seed.

If the metric exceeds the gate, debug metric/exposure/config provenance once.
Do not tune on the residual arm or test split. A second failed exact attempt
stops P1 and returns a failed-reproduction result rather than beginning a sweep.

### P2 — matched residual comparison (`A3`, conditional)

Run direct and residual arms for seeds `0`, `1`, and `2`, paired as specified.
The full registered matrix is six training runs. The one-seed direct baseline
from P1 may count as seed 0 only if its final manifest exactly matches P2.

Primary success requires all of:

1. residual median `realm_npe_mean` improves over direct by at least 10%;
2. the paired seed effect has the same sign in at least two of three seeds;
3. median decoded correlation does not decrease by more than 0.005;
4. no registered front-position, front-thickness, admissibility, boundedness, or
   boundary-band metric regresses by more than 10%; and
5. presentation, parameter, optimizer-step, runtime, selection, and recurrence
   parity gates pass.

The 10%/0.005/10% cutoffs are prospective operational choices but are informed
by the visible benchmark table. Continuous per-call/per-case curves and paired
effects remain primary evidence; passing a cutoff is not an independent
statistical confirmation. With five validation trajectories, report all paired
case effects and seed effects; do not overstate asymptotic significance.

Stop the matrix after two completed paired seeds if both show a residual primary
regression over 10%, any unresolved recurrence mismatch, repeated nonfinite
training, or a no-harm failure over 25%. One failed seed is diagnosable, not a
license to change hyperparameters mid-matrix.

### P3 — PlanarDet and irregular continuation (`new preregistration`)

PlanarDet is next only if P2 passes or yields a sharply motivated transfer
question. Before download or training, resolve `pMax` as a Markov state/history
variable, exact released units/cadence, front/shock metrics, and its one-case
validation limitation. Acquire train/validation only under a new explicit
download approval. The test case remains unfetched.

ObstacleDet is preferred over SupCavityFlame for the first irregular audit
because its full public payload is about 26.4 GB rather than 72.8 GB, but it is
not yet a run candidate. Proceed only if metadata provides a reproducible graph
or point-cloud contract and the selected architecture is scientifically fair.
PCNO requires audited coordinates and quadrature/volume meaning; constructing
an arbitrary k-nearest-neighbor graph cannot support mesh or conservation
claims. A DeepONet comparison must preserve its time-conditioned mechanism and
cannot be pooled as the same autoregressive factor contrast.

## Compute, Storage, And Artifact Budget

| Stage | Resource class | Hard budget / expected artifact |
| --- | --- | --- |
| P0 | CPU/network metadata only, XS | current plan; no trajectory bytes |
| P1a | CPU synthetic, XS | four reviewed source/doc files; focused tests only |
| P1b | network/storage, S | 552,014,385 IgnitHIT train/validation bytes plus 8,634 metadata bytes; ignored manifest/schema reports |
| P1c smoke | one owner-selected personal RTX 5060 Ti, XS | at most 0.25 GPU-hour; runtime/memory manifest |
| P1c full baseline | resource selected in a later authorization, M | at most 12 GPU-hours for one seed; best/last checkpoints and compact histories |
| P2 | resource selected in a later authorization, L | at most six 12-hour runs, but staged 2 + 2 + 2 with stop review after each paired seed |
| PlanarDet | network/storage then GPU, L | at least 2.94 GB train/validation; cost re-estimated from smoke before authorization |
| irregular | network/storage/GPU, XL | deferred; ObstacleDet train/validation is about 22.3 GB before runtime artifacts |

Every executed run retains a canonical config, source/data/runtime manifest,
environment snapshot, stdout/stderr, compact per-call/per-case metrics,
checkpoint-selection history, best and last checkpoints, and a result summary.
Raw data, checkpoints, arrays, media, and logs remain ignored under
`artifacts/time_dependent_no/`. Do not retain redundant intermediate
checkpoints after a result is manifest-verified and the owner authorizes cleanup.

## Risks, Alternatives, And Decision Rules

- **License ambiguity:** block source copying/data acquisition or record a
  narrowly scoped owner disposition after author clarification; never infer a
  grant from a badge.
- **Historical reproducibility gap:** keep paper-formula, released-source, and
  our reconstructed contract as three named objects. Do not tune until a table
  value is matched accidentally.
- **Metric leakage:** test files remain physically absent through development,
  tuning, and selection. A later A4 test release is one frozen evaluation.
- **Residual target ambiguity:** the primary increment lives in `Z`, not in
  conserved reactive variables. If the research claim requires conservation,
  choose a dataset with an authoritative complete state/volume contract rather
  than relabel this experiment.
- **Boundary claim unavailable:** IgnitHIT can support boundary-band error, but
  not a causal boundary-information/enforcement claim unless the release binds
  deployable boundary conditions and forcing. Route C2 boundary claims to
  W26-L3 otherwise.
- **Baseline too costly:** preserve effective batch/exposure with reviewed
  accumulation or stop; do not silently shrink the model/budget and call it
  paper-faithful.
- **No residual win:** publish the matched negative mechanism result if valid.
  Diagnose fresh versus propagated error and structure channels before changing
  architecture.
- **Strong residual win:** confirm the paired three-seed result, then PlanarDet.
  Do not skip directly to 3D or a broad architecture sweep.

## Execution Tracker

| Stage | Status | Exit product | Next authorization |
| --- | --- | --- | --- |
| L4-P0 public audit | `COMPLETE FOR PLANNING` | this source/data/metric feasibility plan | restricted disposition recorded for P1b only |
| L4-P1a contract implementation | `COMPLETE; D088` | preregistration, adapter/evaluator, manifest audit, CPU tests | none |
| L4-P1b IgnitHIT train/validation acquisition/replay | `COMPLETE; D088; ALL GATES PASS` | exact sealed-safe open tree, train statistics, and reference replay | none |
| L4-P1c FFNO-M GPU smoke | `AUTHORIZED; IMPLEMENTATION IN PROGRESS; D088` | source/runtime/memory manifest and smoke verdict | close this exact capped personal-GPU attempt |
| L4-P1c full direct FFNO baseline | `BLOCKED ON SMOKE AND SEPARATE AUTHORIZATION` | one-seed reproduction verdict and measured full cost | named A3 full run with explicit resource |
| L4-P2 residual comparison | `BLOCKED ON BASELINE` | six-run matched result-to-claim packet | named A3 matrix with explicit resource |
| L4-P3 PlanarDet | `BLOCKED ON P2 AND NEW AUDIT` | discontinuous-case preregistration | separate download/training approval |
| L4-P4 irregular | `DEFERRED` | geometry/feasibility decision | separate A0/A1 first |

## Current Authorized Action

The owner has authorized **W26-L4-P1c A3-smoke only**, capped at `0.25` GPU-hour
on the selected personal GPU workstation. The authorized source surface is
exactly:

- update `docs/time_dependent_no/W26_L4_REALM_PREREGISTRATION.md`;
- add `utility/time_dependent_no/realm_ffno.py` for an independent FFNO-M
  reconstruction without copying official source;
- add `scripts/time_dependent_no/smoke_realm_ignithit_ffno.py`; and
- add `tests/time_dependent_no/test_realm_ffno.py`.

First require CPU shape/gradient/parameter-count tests. Then run one synthetic
forward/backward memory probe, at most one optimizer step on one registered real
train batch, and one registered validation-trajectory rollout. Freeze source,
data, runtime, initialization, precision, step-time, peak-memory, and feasible
microbatch/accumulation manifests; preserve effective batch 26 unless a later
explicit review changes it. Stop if the model count is outside 0.5% of 8.9365M
without a source-exact explanation, any tensor/recurrence/normalizer binding
fails, any state is nonfinite, or the 0.25-hour cap is reached.

This authorization does not include persistent training, a second attempt,
hyperparameter changes, checkpoint writing, a residual arm, or test access.

This request does not authorize copying the official implementation, accessing
test objects, persistent training, checkpoint selection, a full baseline seed,
the residual arm, or a scientific performance claim.

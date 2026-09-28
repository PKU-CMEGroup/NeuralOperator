# Implementation Plan: Shared Response Assay And One New PDE

Updated: 2026-09-12

Scientific contract: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
Delivery: [PROJECT_PLAN.md](PROJECT_PLAN.md). Earlier plan bytes are retained
at Git `624c9c0d2b611172ccb6c5afcf1c06af2c6400ec`. Reuse code under new
identities; do not reinterpret closed sources/results as a new study.

## September 13 Pre-Sprint Checkpoint

The owner paused implementation and experiments for local and AutoDL cleanup.
[README.md](README.md) now owns the inspected code/dependency classification;
[HANDOFF.md](HANDOFF.md) records recovery status and the current pause. Another
agent is using AutoDL; a possible move to zw-gpu does not itself authorize
migration or a launch.
Keep frozen source paths and hashes intact. The next infrastructure changes,
if adopted, should extract only the shared SSH execution primitive and exact
Kolmogorov array-hash encoding, with parity tests under a new source identity.
The recipes below remain scientific contracts, not an instruction to launch.

## September 8 Coverage And Diagnostic Stage

The primary integrator owns two separately invocable successors and their
focused tests. Existing Clean, C-recovery and solver sources remain unchanged.
The old entry points bind closed identities and cannot execute these new roles.

```text
python -m scripts.time_dependent_no.evaluate_kolmogorov_clean_rollout --parent <C-packet> --parent-source <C-source> --clean <Clean-packet> --clean-source <Clean-source> --output <fresh-diagnostic> --device cuda
python -m scripts.time_dependent_no.extend_kolmogorov_training_population --parent <C-packet> --parent-source <C-source> --output <fresh-extension>
```

The evaluator reuses the exact population loader, PCNO adapter and checkpoint
replay; only restricted predictions are recurrent feedback. The generator reuses
the frozen native initial-law, case-generation and refinement functions. Its
index references C and new train-only paths, with no validation reassignment.
Audit checkpoint/data/source binding, recurrence versus teacher forcing,
normalization, failure censoring, Fourier Parseval accounting, solver replay,
peak/late continuation coverage, parent nonmutation and overwrite refusal.

These two preparation jobs produce no new trained model. One ignored attempt-specific
transport owns the source-only payload, existing-AutoDL guard, separate launch
receipts and watchdogs; it must not trigger training implicitly. Keep storage
cleanup decisions separate from numerical results. The tracker owns test,
payload, capacity and launch status.

### Fixed-Update Coverage Fit

The expanded data are now qualified. The primary integrator owns
`fit_kolmogorov_coverage.py` and its focused test. This separate entry point is
needed because the closed trainer binds the original 8/4 split and adaptive
stopping rule. Reuse its PCNO, LR schedule, epoch sampler, checkpoint/RNG and
replay helpers unchanged. Do not edit the closed source to run the new identity.

```text
python -m scripts.time_dependent_no.fit_kolmogorov_coverage --parent <C-packet> --parent-source <C-source> --extension <24-path-packet> --extension-source <extension-source> --clean <Clean-packet> --clean-source <Clean-source> --output <fresh> --phase validate
python -m scripts.time_dependent_no.fit_kolmogorov_coverage --parent <C-packet> --parent-source <C-source> --extension <24-path-packet> --extension-source <extension-source> --clean <Clean-packet> --clean-source <Clean-source> --output <fresh> --phase fit --device cuda
```

The first invocation verifies full manifests, canonical states, original
float32 bytes and normalization without constructing a model. The second
uses the experiment plan's fixed budget, actual solver successors as labels,
separate original/added training metrics, and unchanged validation. Tests cover
split/index correctness, full-pair coverage, old-evaluator metric parity,
sampler continuation, exact stopping, replay, nonmutation and failure packets.
Reject outputs overlapping input packets before creating directories. Check
deadlines after computation, serialization and terminal replay, not only before
operations. Preserve every already-validated input binding if loading fails;
report unchecked evidence as unknown, never stable. Save all-pair scalar metrics
and six field sentinels, with the late-improvement interval endpoints explicit.
The CPU fixture is API-only and cannot be selected by the scientific CLI.
Review and exact source-only payload approval precede upload/launch. No new
solver generation, cleanup, resource purchase or other agent's resource is used.

### Expanded-Model Rollout

The primary integrator owns `evaluate_kolmogorov_coverage_rollout.py` and its
focused test. A separate entry point is needed because the closed evaluator
pins the eight-path checkpoint/schema. Reuse its recurrence, physical/spectral
diagnostics, summaries and sentinel replay unchanged; do not monkeypatch its
globals or modify any of the 25 B-bound source files.

```text
python -m scripts.time_dependent_no.evaluate_kolmogorov_coverage_rollout --parent <C-packet> --parent-source <C-source> --coverage <B-packet> --coverage-source <B-source> --output <fresh-validation> --phase validate
python -m scripts.time_dependent_no.evaluate_kolmogorov_coverage_rollout --parent <C-packet> --parent-source <C-source> --coverage <B-packet> --coverage-source <B-source> --output <fresh-rollout> --phase rollout --device cuda
```

Validate B's immutable checkpoint and C's original data/normalizer, not the
union state-store hash. Explicitly test the 36-to-12 teacher mapping, exact
reference labels, checkpoint buffer, no-feedback truth, failure censoring,
decoder failure finalization, source/input nonmutation and output isolation.
The validate phase performs data/metric checks without constructing a model.
The run-local transport packages 27 source files; review and exact payload
approval precede deployment. No new general orchestration layer is needed.

### Common-Input Response Diagnostic

The primary integrator owns `evaluate_kolmogorov_common_response.py` and its
focused test. The closed rollout entry points cannot run this cross-model bank;
reuse their packet validators, adapter, replay and Fourier metrics, and the
existing `path_conditioned_tube_metrics` without changing any closed source.

```text
python -m scripts.time_dependent_no.evaluate_kolmogorov_common_response --help
```

The entry point has separate `validate` and `assay` phases. Bind C, both fits,
both rollout manifests and actual snapshot arrays. Validate exact seed/role/time
mapping, train-only RMS calibration, identical recipient inputs, zero-direction
handling, raw/restricted attribution and replay. Save arrays plus per-query
metrics, complete-accounting and failure receipts. Reject overlapping output
paths and optimized Python. Do not label path-recovery error a displaced-solver
defect. Safeguard successor B checks the full realized float32
displacement vector, cooperative deadlines between expensive stages, and actual
construction/forward invocations even when subsequent work fails. Its packet
test independently reconstructs metrics from saved arrays. Successor C adds
deadline-aware sentinel orchestration without editing the closed replay helper;
real-decoder expiry and numerical-parity tests cover that boundary. Keep both
reviewed A/B source archives unchanged. Same-scope code-only review approval
remains distinct from scientific deployment. The owner subsequently approved
the exact C launch, now completed, retrieved and independently recomputed.
The tracker owns tests, reviews, receipts and the proposed next solver-response
qualification. Do not reuse this completed identity or expand its recipe.

### Train-Only Solver-Response Pilot

Add `screen_kolmogorov_common_solver.py` and its focused synthetic test. The
closed screens generate prescribed directions or clean trajectories; neither
can consume the frozen model-error bank with shared learned outputs. This
entry point owns the new 53-call pilot and has separate `validate`/`assay`
phases. Reuse the immutable reference stepper, its budgeted wrapper and the
paired-response metric; do not modify their closed source files.

Validate the exact C manifest/result, selected train-only bank/output arrays,
original FP64 population blocks, configurations and source hashes. Keep raw,
projected and lifted inputs distinct. Test role/time/donor mapping, shared clean
solves, label provenance, projection floors, all four refinement levels,
53-call accounting, zero directions, deadline/failure packets and output
isolation. Save every completed solver answer before later work; final hashes
and completion gates must not turn an incomplete run into a qualified result.
The primary integrator owns implementation; Codex agents may perform bounded
code-only reviews/tests. The tracker owns review and launch status.

### Paired Gaussian Bank And Adaptation

The owner accepted the paired recovery/relabeling successor. Implement its
training-only bank first in `generate_kolmogorov_paired_bank.py`, with a focused
synthetic test. The closed common-input pilot cannot generate new Gaussian
queries or read the 32-path union; preserve its source and reuse its immutable
reference, numerical configurations, hashes and budget helpers.

```text
python -m scripts.time_dependent_no.generate_kolmogorov_paired_bank --population <C-packet> --extension <24-path-packet> --common <common-response-C-packet> --qualification <53-call-packet> --output <fresh> --phase validate
```

`generate` is a separate explicit phase. Stream selected training blocks rather
than loading validation or the full state store. Store raw FP32 clean/displaced
inputs and FP64 A labels, with full refined answers only at declared sentinels.
Save completed answers before later work; bind source, role/step/noise identity,
input blocks, actual call counts and final hashes. Validate-only makes no solver
or model call. Test Gaussian covariance scaling without per-draw normalization,
antithetic pairing, label identity, shared clean solves, sentinel selection,
all 1,657 calls, no validation reads, output isolation and incomplete packets.
Native Codex code-only reviews precede scientific execution. The primary
integrator owns the runner and test; the tracker owns exact payload approval.

The bank is complete and independently audited; fine-grid qualification remains
sampled at the declared sentinels. The September 12 adaptation runner is
implemented in `fit_kolmogorov_paired.py`, with a focused synthetic test. It
reuses the audited PCNO, sampler and replay primitives without changing closed
fit sources. The experiment plan owns the three row views and matched budget;
the tracker owns verification receipts and deployment status.

```text
python -m scripts.time_dependent_no.fit_kolmogorov_paired --population <C-packet> --extension <24-path-packet> --bank <paired-bank-packet> --parent <clean32-packet> --output <fresh> --phase validate --arm clean_continuation
```

- Validate all 608 training blocks and saved bank inputs/A labels, not just the
  subset used to generate the bank. Do not decode validation states, the mixed
  teacher archive or refined B/C/D answers. Validation hashes the parent
  checkpoint without deserializing it or constructing a model.
- Resource/fit strictly load only the parent's model state, preserving its
  trained head and normalizer. Adam, RNG and both samplers start fresh. Keep
  raw FP32 inputs unchanged and explicitly cast A targets to FP32.
- Use sequential half-weight backward passes and one optimizer step. Preserve
  signed row IDs and both sampler states; count an update only after device
  synchronization succeeds. The shared PCNO output restriction stays deployed.
- Retain partial work on failure, but promote a terminal only after training,
  train-only replay, deadline and final source/input checks pass. Serialized
  model state must match exactly; repeated inference uses the inherited Clean
  per-input relative RMS tolerance `1e-6`, not bitwise GPU output equality.

The synthetic suite covers all three arms, full-budget sampler tapes, matched
gradient/Adam updates, forbidden reads and failure receipts. Small actual-PCNO
fixtures also exercise strict initialization, fitting and terminal replay.
These are implementation checks, not corrective efficacy or GPU readiness.

The separate `resource/dynamics` payload is now frozen: 16 updates, no retained
checkpoint and no continuation into `fit`. Its source-only transport is owned
by the ignored resource attempt; the tracker records tests, capacity checks and
the pending exact cleanup/upload approval. Check runtime/memory and freeze the exact
scientific payload before the three fixed 4,096-update fits. All phases use
fresh output directories. Never combine generation, training and rollout into
an automatically expanding job; diagnostic predictions precede new rollout.

## Immediate Work

| Order | Owner / surface | Deliverable |
| --- | --- | --- |
| I1 | Diagnostic implementer: path_conditioned_tube.py and existing test | Complete: paired solver-response metrics; legacy outputs preserved. |
| I2 | Solver implementer: screen_kolmogorov_response_readiness.py and focused test | Complete: source-bound A and B response screens, not population qualification. |
| I3 | Independent reviewer | Trace targets, response, geometry, recurrence and claims before scientific execution. |
| I4 | Primary integrator, owner approved September 6 | Minimal periodic scalar PCNO adapter and bounded CPU/GPU resource smoke. |
| I5 | Training/evaluation integrator | Reuse audited mechanisms under fresh identities; smoke, tiny fit, paired training and prediction freeze. |
| I6 | Paper integrator and numerical reviewer | Recompute tables from raw rows; inspect figures/PDF and claim-to-evidence mapping. |

## Shared Response API

Add `paired_solver_response_metrics` to the existing branch-local diagnostic
module. Inputs: clean/displaced state, model outputs at both, trusted successors
at both, node weights and component scales. This function reads no dataset and
generates no labels.

Return clean defect, recovery/dynamics errors, learned/trusted responses,
response defect, finite-amplitude gains, signed alignments and algebra closure.
Validate real finite aligned arrays, weights/scales and nonmutation; use None
for zero denominators. Callers bind solver/state/query/model identities and
common random tapes. Norms alone do not identify normal dynamics.

Two present uses justify this extension: the new common-bank PDE experiment
and the missing retrospective NACA paired-bank comparison. Legacy path-only
metrics/aggregation remain unchanged and do not become solver-relative.

## Local Pilot

Invocation from repository root:

```text
python -m scripts.time_dependent_no.screen_kolmogorov_response_readiness --output <fresh-B-output-directory>
```

A new entry point is needed because old M1 runners bind different closed
populations/contracts. Reuse `kolmogorov_reference.py` unchanged. The small
screen is frozen in the experiment plan, not a production data generator.
The current source is B (N128/N256); A's exact source is preserved in its
separate verified ignored archive. Do not reuse either completed output path.
Test separately labeled N16/N32 fixtures before scientific execution. Save
protocol/source/runtime/measurement hashes to a fresh ignored packet and refuse
overwrite. No old scientific arrays or checkpoint are needed.

The new longer-time entry point is
`screen_kolmogorov_trajectory_readiness.py`, owned by I2, with its focused test.
It is separate because its 512-step development trajectories and durable block
output have a different contract from the closed one-step screens. Invocation:

```text
python -m scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness --output <fresh-directory>
```

The current run uses an ignored source-frozen hidden launcher. Preserve its
snapshot and live output. Read the final manifest/exit receipt and reconcile
all blocks, queries and numerical gates before interpreting the result; do not
rerun into its existing path. Exact status and ETA live in the tracker.

## Periodic PCNO Adapter And Smoke

`utility/time_dependent_no/pcno_kolmogorov.py` implements the fixed periodic
scalar adapter; its focused test owns seam, forcing, scaling, recurrence and
restriction checks. The maintained smoke runner has one present purpose:
verify optimization and measure full-grid cost before freezing the campaign.

```text
python -m scripts.time_dependent_no.smoke_pcno_kolmogorov --device cpu --output <fresh-directory>
python -m scripts.time_dependent_no.smoke_pcno_kolmogorov --device cuda --output <fresh-directory>
```

CPU: N16, width 8, 100 synthetic updates. CUDA: N128 and N256, width 64,
four PCNO layers, 12 Fourier modes per axis, batch 1, 11 updates and 20 timed
inference calls each. These are resource presets, not a frozen scientific
architecture. All data are analytic synthetic arrays, not solver trajectories.
The initial/final loss and gradient checks fail into retained failure packets.

`fit_kolmogorov_debug.py` and its focused test own the subsequent solver-labelled
engineering fit. They are separate from the frozen synthetic smoke: the new
runner must validate its parent packet, retain a checkpoint and prediction
arrays, and report in-sample recurrence without a generalization claim.
Its exact data and optimization contract lives in the experiment plan.

`generate_kolmogorov_population.py` and its focused test own the fresh
training/development packet. Reuse the frozen trajectory screen's initial-law,
stepping, block-writing and comparison functions without changing that closed
screen. A separate entry point is needed for immutable train/development roles
and clean-only, maximum-palinstrophy numerical spotchecks. Invocation:

```text
python -m scripts.time_dependent_no.generate_kolmogorov_population --output <fresh-directory>
```

I2 implements and I3 independently audits the fixed population contract before
execution. This is not a generic data-generation framework; all twelve seed
roles, numerical limits and budget are fixed in the experiment plan. That N128
packet completed but failed its early-transient spatial checks; preserve it.

The same maintained generator/test now own the N256 B successor, instead of
adding another near-duplicate generator. Before changing either file, verify
the complete A source snapshot against A's manifest. A remains replayable from
its ignored `cm_next_kf_pop_20260906a_launch/source/` snapshot; do not claim its
old source hashes match the revised checkout. B receives a new identity and
parent binding and reuses unchanged peak-screen query/array helpers. Invocation:

```text
python -m scripts.time_dependent_no.generate_kolmogorov_population --parent <peak-screen-packet> --output <fresh-B-directory>
```

Audit all twelve roles, the full native horizon, peak/fixed-anchor deduplication,
two continuation anchors on each selected seed, independent fine feedback,
partial arrays, non-clobbering and gate coverage before launch. Keep the trusted
reference, longer-time and peak-screen sources unchanged. Exact B compute and
storage limits are in the experiment plan; no automatic population upload.

September 7 C recovery reuses this generator/test after verifying B's complete
archived source closure. The current invocation is:

```text
python -m scripts.time_dependent_no.generate_kolmogorov_population --parent <B-packet> --parent-source <B-launch/source> --output <fresh-C-directory>
```

The pinned independent B audit must be beside that source directory. Test
strict parent/source validation, saved-answer and deterministic replay,
byte-preserved complete cases, partial-prefix restart, peak reconstruction,
full recomputation of the partial query, elapsed-budget accounting and honest
failure receipts on synthetic fixtures. The scientific replay runs only on
the approved target before continuation. Keep reference, longer-time and peak
helpers unchanged. One ignored, study-specific transport in
`cm_next_kf_pop_20260907c_transport/` binds the exact payload, excludes the
other agent's resource, checks capacity and hashes, and owns the Linux
watchdog/retrieval receipts; it is not a new maintained deployment framework.

The bounded successor `screen_kolmogorov_peak_refinement.py` and its focused
test are owned by I2/I3. A separate entry point preserves the closed sources
and retains full query/continuation arrays, which the older scalar-row helper
does not save. Reuse unchanged reference and trajectory helpers; do not add a
general solver-sweep interface. Invocation:

```text
python -m scripts.time_dependent_no.screen_kolmogorov_peak_refinement --parent <population-packet> --output <fresh-directory>
```

Before the local N256/N512 screen, test native initial-state generation, peak
selection/deduplication, same-input refinement, independent full-fine recurrence,
recomputed gates, source/parent hashes and honest partial-output handling on
small synthetic fixtures. The fixed scientific contract is in the experiment
plan. The AutoDL debug fit is now complete and audited; preserve its exact
source closure and payload for replay.

For Kolmogorov, the solver/reference tests match retained spatial/temporal
sources at the September 6 audit. That does not qualify new parameters or
populations. The implemented state contract is:

- full scalar vorticity, fixed zero mean velocity and explicit cos(4y) feature;
- periodic (x,y) ordering, wraparound graph and minimum-image gradient weights;
  raw coordinate differences are wrong at seam-crossing edges;
- train-only normalization, residual identity and correct recurrent feedback;
- declared shared mean/band restriction, with raw/restricted outputs retained;
- explicit projection accounting for float32 solver queries;
- no downsampled fine-state Markov assumption; and
- static geometry reuse; CPU fit and GPU resource results are recorded in the
  tracker. The planned tiny fit is finite-grid optimization debugging only;
  a PDE-accuracy claim still requires qualified trajectory data.

Do not force N64 to save compute: choose numerical resolution from sensitivity
checks and physical regime from measured dynamics. The older low-viscosity
N256 result does not qualify the new candidate settings.

For SU2, a NACA SA restart writer is not an SST/incompressible writer with
renamed fields. Pin mesh/config/binary, evolved fields and complete BDF2 state;
qualify convergence and clean/displaced restart first.

## Training And Code-To-Intent Audit

The full-population Clean trainer and focused test are implemented in the
existing resource entry point, owned by I5/I3; the completed debug fitter is
unchanged. Validate the
completed recovery manifest/source binding before loading exactly 4,096 training and
2,048 development transitions, fit scaling only on training inputs, and retain
optimizer/RNG state, terminal checkpoint and durable training/development logs.
The actual N256 batch-eight resource phase in `fit_kolmogorov_clean.py` is
complete and audited. Its synthetic tests cover split/index mapping,
train-only scaling, integrity rejection and failure receipts. Preserve its
approved source snapshot, then extend that same entry point and test for the
full fit; no separate generic trainer or resource weights are used.
The [first full Clean pilot](EXPERIMENT_PLAN.md) now freezes sampling, optimizer,
schedule, competence targets and the single extension rule. The owner-approved
exact payload completed and passed result/checkpoint audits on September 8.
Tests cover exact epoch coverage,
LR endpoints, pooled metric/band
definitions, strict extension decisions, checkpoint replay and preserved
RNG/optimizer state; the tracker owns final-byte verification and approval.
The accepted terminal checkpoint is the deployed
model; development one-step scores monitor readiness, not desired rollout failure.
Avoid duplicating the data or saving every teacher-forced field prediction:
retain the model, reproducible evaluator, per-pair metrics and small replay
sentinels, then budget the rollout arrays explicitly.

The full phase is selected explicitly; omitting `--phase` retains the closed
resource runner's behavior:

```text
python -m scripts.time_dependent_no.fit_kolmogorov_clean --phase clean --parent <qualified-C-packet> --parent-source <C-source-root> --output <fresh-directory> --device cuda
```

One ignored attempt-specific transport in `cm_next_kf_clean_20260907a_transport/`
prepares the exact source closure, checks the existing AutoDL instance, and
owns the bounded launch/retrieval receipts. This is not a new deployment API.
The exact payload approval and launch receipts are recorded in the tracker.
Do not repeat the closed launch command or modify its source/checkpoint evidence.

NACA/Bump implementations supply reference equations, not drop-in generic
trainers. Preserve field/history semantics when porting curriculum, EMA,
noise targets and refinement. An independent reviewer checks:

1. same paired input arrays, correct clean versus displaced successor targets;
2. normalization, loss weights, initialization/data order and budget;
3. prefix history, stop-gradient, terminal target, warmup and EMA deployment;
4. corrector composition, identity parity and corrected-state feedback;
5. refinement scheduler, random tapes and model-call cost;
6. train-only calibration and trajectory role separation; and
7. source/input/runtime/checkpoint/evaluator/final hash bindings.

Tests are necessary but not a scientific audit. Run narrow tests, affected
regression and Ruff; preserve failed attempts under unique identities. No
generic registry, dashboard, framework or speculative CLI. Add only files with
an owner and present invocation.

## Access And Handoff

The September 6 owner decision authorizes the dedicated Kolmogorov solver and
existing AutoDL instance. Remote runs use that resource with exact receipts
after code/numerical checks. No new cloud purchase or protected reveal is
implied. Another agent owns a separate GPU resource: do not connect to or modify it.
The AutoDL transport must reject a changed destination before connecting
and check GPU/process occupancy before launching. Keep datasets, checkpoints, private machine details and unpublished
paper out of Git publication. Report exact checks, artifacts, unmeasured
quantities and next decision. Protocol and run receipts stay out of the paper.

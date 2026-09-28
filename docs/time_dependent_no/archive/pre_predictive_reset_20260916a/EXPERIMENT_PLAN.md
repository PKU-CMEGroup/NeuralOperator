# Experiment Plan: Identify Mechanisms Before Rollout Outcomes

Updated: 2026-09-11

New study: `CM_NEXT_20260906A`. This does not reopen or amend closed NACA,
Bump or M1 identities. Previous plan bytes are recoverable at Git
`624c9c0d2b611172ccb6c5afcf1c06af2c6400ec`; original contracts still govern
completed studies. Scope/deadline: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Claim Map And Story

| Block | Question | Paper output | Priority |
| --- | --- | --- | --- |
| R: readiness | Can the actual PCNO state be evolved and perturbed by a trusted, affordable reference? | Application definition and numerical credibility. | Must |
| M: mechanisms | What does each method change on identical inputs? | Common-response figure and rollout/cost table. | Must |
| P: prediction | Do frozen diagnostics predict unseen rollouts beyond clean error? | Prospective comparison, including negative results. | Must |
| H: design | Can a dynamics buffer plus recovery preserve useful response and correct harmful error? | One method/deletion comparison. | Conditional |

ODE, NACA and Bump remain supporting evidence. No new ODE or broad architecture
sweep. Benchmark selection alone is not the contribution.

## September 8 Priority: Qualify Clean Generalization

The owner approved the following bounded successor after inspecting Clean's
training/validation one-step gap. The mentor supports Kolmogorov as the PDE
family; its specific regime is still provisional. Passing the old pooled 2%
engineering gate does not establish the paper's starting point. Require small,
comparable one-step errors on independent training and validation trajectories,
including the early transient. A rollout gap is a separate measurement.

1. **Existing-checkpoint diagnostic** (`CM_NEXT_KF_CLEAN_ROLLOUT_20260908A`):
   evaluate the unchanged `terminal_049152.pt` from all eight training and four
   existing validation initial states, through the original 512 steps. Training
   starts test self-composition on supervised paths. Validation results remain
   confounded by the known clean generalization gap. No retraining or selection
   of a worse checkpoint is permitted. Report teacher-forced and recurrent error
   on matching windows, fixed-training-scale error, persistence, spectra,
   energy/enstrophy and mean vorticity. Keep raw output and the common numerical
   restriction distinct; Fourier bands are not labelled tangent/normal.
   Save sparse raw/input/prediction/truth snapshots at steps
   `0/1/8/32/64/128/256/512` and every per-step metric. A path stops individually
   on nonfinite output, a numerical/model error, or RMS above `1e6` times the
   fixed training scale. Record the failed step; do not reset, clip, or impute
   the remaining horizon. This amplitude guard is not an accuracy threshold.
   Compute cap: 30 minutes, with a separate transport allowance for finalization.
2. **Train-only population extension** (`CM_NEXT_KF_TRAIN32_POP_20260908A`):
   add exactly 24 independent phase draws, seeds `2026090801`--`2026090824`, to
   C's original eight training trajectories. C's four validation paths remain
   validation. The new index references C read-only instead of copying it.
   Keep N256, viscosity `.01`, drag `.1`, forcing amplitude `1`, wavenumber `4`,
   the six-phase initial law and perturbation RMS `.25` times laminar RMS,
   macro-step `.05`, and 512 transitions unchanged. Before generation, replay
   C's training-611 step-zero base/half-step/fine answers twice, requiring
   relative discrepancy at most `1e-10` and exact same-runtime repeatability.
   Every new path receives the existing fixed-anchor and maximum-palinstrophy
   N256/N512 clean checks. New seeds 0801/0802 also receive independently
   recurrent eight-step refinement checks at the peak and step 256. Preserve
   the existing `1e-3/1e-3/2e-3` numerical limits, every failed path, and partial
   packets. No seed replacement. Compute cap: eight hours; require 9 GiB free
   storage before generation. Sampled clean qualification is not displaced-input
   qualification or a uniform numerical guarantee.
3. **One-step coverage comparison** (`CM_NEXT_KF_CLEAN32_MATCHED_20260909B`):
   the expanded data passed audit. Fit the 32-path union from fresh seed 17;
   preserve C's four validation paths, the PCNO configuration and common
   numerical restriction. Keep the original eight-path input RMS normalizer
   (`4.063412684964112` before float32 storage), not the expanded population's
   RMS. Use batch eight, the same Adam/loss/precision/zero-head initialization,
   and exactly the original terminal's 49,152-update learning-rate history:
   512-update warmup, cosine through update 32,768, then constant `1e-4`.
   This gives 24 complete data passes versus the original run's 96. Do not stop
   early, choose a best checkpoint, or extend according to validation outcomes.
   Evaluate all pairs at update zero and every 4,096 updates; report the
   original eight training paths, added 24, full training union, and unchanged
   validation separately, with every trajectory and input bands
   `[0,64)/[64,256)/[256,512)`. Retain all-pair raw/restricted squared-norm
   metrics, persistence and fixed-scale MSE, plus six input/target/raw/restricted
   field sentinels for replay; do not retain every prediction field at every
   evaluation. Compare absolute errors on the
   same original/validation paths, not just the gap between changing pooled
   populations. Report improvement over the last 8,192 updates as an
   optimization diagnostic. Any longer fit is a separate, newly approved
   comparison; a gap reduced by worsening training error is not success.
   Retain atomic latest and the fixed terminal with replay checks. Cap this
   fit at five hours from source/input validation through terminal replay;
   final provenance checks are timed separately. Require 2 GiB free storage
   before model construction; no automatic deletion. The September 9 identity
   fixes prelaunch failure-handling/provenance findings in the unlaunched prior
   sources, without changing the scientific recipe. The owner approved these
   narrow revisions, regression tests, re-review and conditional launch without
   another approval round; freeze the exact revised payload and launch only
   after review and fresh resource/runtime checks pass.
   This job does not run rollouts or corrective arms.
   Completion and the old 2% engineering gate are not paper qualification.

September 9 status: all three jobs are complete and audited. The expanded fit
reduces validation one-step error to 0.4808% versus 0.2571% training, with a
remaining early-transient gap. Its rollout is not yet measured. The tracker
owns the full comparison and recommendation for a separate matched-start
rollout; the frozen contracts above are unchanged.

### September 9: Expanded-Model Matched-Start Rollout

The owner accepted this next step and emphasized avoiding data-scarce,
severely overfit baselines. `CM_NEXT_KF_CLEAN32_ROLLOUT_20260909A` evaluates
only B's fixed `terminal_049152.pt` on C's original eight training and four
validation initial states. Preserve all 512 transitions, original normalization,
shared mean-zero/dealias restriction, snapshots, horizon summaries, diagnostics
and `1e6` amplitude guard from the September 8 rollout contract. Compare the
same paths and windows, retaining failures without resetting, clipping or
survivor-only pooling. The first-step and early-time errors remain visible.

Bind B's audited final manifest/checkpoint and C's original state-store identity.
Select teacher metrics from B positions `0--7,32--35`, explicitly matching
C's seed/role and reference norms; do not treat B positions `8--11` as validation.
Reuse the six saved one-step sentinels for checkpoint replay; two are added
training inputs, not extra rollout starts. No added trajectory arrays, new
solver labels, optimizer steps, correction, geometry fitting or protected
population are involved. Validate-only creates no model. Retain source/input
checks and failure packets for both validation and evaluation.

Use the existing AutoDL resource, a 30-minute work cap plus separate finalization
allowance, and at least 1 GiB free storage. Freeze and review the exact new
source-only payload before requesting upload/launch approval. No cleanup,
training extension or correction matrix follows automatically. This is an
exploratory matched comparison, not prospective confirmation: whether improved
coverage also resolves rollout failure is an open outcome, not a selection rule.

The diagnostic and population jobs have separate invocations and permission
receipts; neither automatically starts training or the correction matrix. Exact
payload approval and current resource/storage checks precede launch. Current
status, including the storage constraint, belongs in the experiment tracker.

If clean coverage resolves rollout difficulty too, retain that outcome. If the
physical regime is unsuitable, define and qualify a separate setting rather
than alter this comparison after observing its outcomes. Do not silently discard
the transient, change the initial-condition law, or tune for Clean blowup.
Parameter-based local geometry remains a possible offline diagnostic, not an
already measured manifold or an implemented corrector.

September 10 repair successor: the owner approved the five review-requested
safeguards, focused tests, Astra re-review and conditional launch under
`CM_NEXT_KF_CLEAN32_ROLLOUT_20260910B`. Changes are limited to optimized-
interpreter rejection, exact captured upload/checkpoint bytes, symmetric output
isolation and separate transport/packet failure reporting. The checkpoint,
original twelve starts, normalization, recurrence, diagnostics and budgets above
are unchanged. After exact prompt approval, re-review requested two further
transport-only safeguards: twelve-path completion validation and retrieval
independent of post-launch live-file edits. No additional scientific-evaluator
changes were requested; no upload or launch occurred. Preserve both reviewed
bundles; the tracker owns exact source, review and proposed repair scope.
No correction matrix follows this repair.

September 10 closeout: transport-only R2 passed review and the unchanged
scientific evaluation completed. The matched H32 validation comparison reverses
the one-step ranking: 2.4484% to 0.6963% one-step, but 35.2606% to 74.8571%
rollout. All twelve expanded-model paths hit the amplitude guard at steps
84--108; H128/H512 are censored. Exact verification and limitations belong in
the tracker. The bounded common-input response/forcing diagnostic below is now
completed and independently recomputed. It separates learned responses, not
geometry or causality; no corrective comparison or further fit follows automatically.

### September 10: Common-Input Response Diagnostic

The owner accepted the next diagnostic. `CM_NEXT_KF_COMMON_RESPONSE_20260910C`
is the sentinel-deadline successor to undeployed A/B, with the same scientific recipe:
one solver-free, retrospective assay of the two frozen Clean terminals,
not corrective training or prospective confirmation. The primary integrator
owns its implementation. Existing fit, population and rollout bytes stay fixed.

Use all original eight training and four open validation paths. At saved output
steps **1, 8, 32, 64**, recover the actual recurrent input at step `k-1` from
each model's snapshot and the matching reference pair `(u[k-1], u[k])` from C.
Both recipient models receive each donor's identical input. Keep donor,
recipient, time and role separate; no path or anchor is selected by its error.

Two input views:

- **Natural:** the saved recurrent input, unchanged.
- **Matched RMS:** the same displacement direction rescaled to one common RMS.
  Set its physical RMS to the expanded terminal's pooled clean one-step error
  RMS over all **32 training paths and 512 transitions**, using saved scalar
  squared errors only. Validation is excluded from calibration. Freeze this
  rule before new inference; report the resulting scale in training-state units
  and the actual float32 displacement. Zero directions are unresolved and are
  not replaced by invented noise. There is no optimal-scale search. Reject
  float32 relative RMS error or full-vector rounding error above `1e-4`;
  the latter is RMS(realized minus intended displacement) / requested RMS.

For each recipient, report clean forcing `d = Psi(u)-u_next`, learned response
`r = Psi(x)-Psi(u)`, secant gain, signed forcing-response alignment and
`||d+r||`. Check the vector and squared-norm identities. Report raw and
numerically restricted outputs separately, with their Fourier-band energies;
Fourier bands are not tangent/normal directions. Replay each model's own
natural snapshots and its saved terminal sentinels before interpreting results.
Retain query arrays and outputs so metrics can be recomputed without a model.

Pre-inference interpretation rules: larger gain for the expanded model on
both donors at matched RMS implicates the learned response on this bank.
A difference only at natural amplitudes points instead to amplitude-dependent
behavior; a difference only on one donor points to direction dependence.
Similar gains require examining forcing and alignment, not declaring that the
framework failed or that the responses explain the rollout ranking. These are
diagnostic distinctions, not a causal decomposition of the full rollout.
Without displaced solver successors, do not report trusted response, response
defect, true normal stability, measured OOD or a calibrated stability tube.

Validation constructs the bank and checks evidence without model inference.
Synthetic tests and independent code review precede the one existing-AutoDL
assay; exact payload approval and fresh capacity checks precede deployment.
Use the existing cooperative 30-minute work cap and 1 GiB minimum free storage.
Check between expensive stages; final provenance is separate. Deployment also
needs an outer process timeout for stalled individual operations. No new
solver labels, optimization, geometry fit, protected reads or cleanup. The
next decision is a qualified paired-response/correction recipe, not an
automatically launched matrix. The tracker owns actual verification/status.

Completion routing: C finished September 10 at 22:24:33 China time. The tracker
owns the full stratified result and qualifications. Its next proposed decision
is a bounded train-only displaced-solver qualification before choosing whether
recovery should suppress the measured response or relabeling should reproduce it.
This does not authorize new solver labels, training or protected access.

### September 11: Train-Only Solver-Response Qualification

The owner accepted the proposed bounded qualification after C's common-input
result. `CM_NEXT_KF_COMMON_SOLVER_20260911A` asks whether the observed learned
responses differ from numerically resolved dynamics on those same inputs.
It is a numerical pilot, not corrective training, all-eight confirmation,
validation-population evidence or a new rollout.

Select the first two original training IDs by ordinal (`2026090611/12`), output
steps **8/32** (input steps **7/31**), and **both** matched-RMS donors from C.
No error-based path selection, natural-amplitude extension or scale tuning:
four clean anchors and eight displaced inputs at the existing train-only RMS.
Keep the original float32 model inputs and saved raw/restricted model outputs.
Project each solver input once at N256 to canonical mean-zero float64, record
the change, and lift that same polynomial to N512. Do not renormalize after
projection. The declared reference includes this input projection; it is not
a training-manifold projector or a newly deployed model correction.

| Numerical level | Grid | dt_max | CFL |
| --- | --- | --- | --- |
| A, original recipe | 256 | .002 | .4 |
| B, coarse temporal refinement | 256 | .001 | .2 |
| C, spatial refinement | 512 | .001 | .2 |
| D, fine temporal refinement | 512 | .0005 | .1 |

All levels retain the parent physical parameters and macro step `.05`. Advance
all twelve states at all four levels: **48 calls**. Add four original-FP64
clean-input/archived-FP64-successor replays under A, plus one repeat of the
first projected clean input under A: **53 calls total**. Share clean solves
between donors and recipients. No checkpoint loads or model calls are needed.
The work cap is 90 minutes, with final provenance separate; exact transport
must also impose an outer timeout. No automatic expansion or retry.

Before scientific execution, test and review the implementation. Apply these
newly frozen gates per query, not only in aggregate:

- input projection and displacement-vector change / actual displacement RMS
  at most `1e-4`; a collapsed direction is unresolved;
- lift/restriction round trip at most `1e-11` relative L2;
- original-FP64 replay at most `1e-10` relative L2, and exact same-process repeat;
- A/B and C/D response differences / displacement RMS at most `1e-3`;
- B/C response difference / displacement RMS at most `1e-2`;
- discarded full-fine D response / displacement RMS at most `1e-2`; and
- B/C state discrepancy and discarded full-fine D state at most `1e-3`
  relative L2, on clean and displaced states.

Recompute both trusted endpoints under every numerical map; never substitute
the archived successor for a freshly evolved clean endpoint in a response.
Retain native A and finest restricted D results, paired learned/trusted response,
response defect, recovery versus dynamics-target error, Fourier and physical
diagnostics. Keep the archived-clean-target discrepancy separate. The summed
successive A/B/C/D response differences are an empirical refinement-sensitivity
indicator, not a rigorous continuum-error bound. Response defects comparable
to that indicator, or recipient-defect gaps comparable to twice it, remain
unresolved even if the absolute gates pass.

Qualification covers only these four anchors/eight displacements. Failure
closes this attempt without changing thresholds or rejecting a method family.
The result informs the later recovery/relabeling decision; it does not select
a corrector automatically. No protected data, further population generation,
paper edit, cleanup or new compute resource is included. The tracker owns
implementation, tests, review, exact payload and launch status.

### September 11: Paired Gaussian Recovery/Relabeling Successor

The owner accepted the proposed matched comparison after the solver-response
pilot. This is a bounded adaptation study, not the full three-seed corrective
matrix or a prospective C2 test. Preserve the completed pilot and both Clean
models. First prepare and qualify the common training bank; training follows
only a reviewed bank and exact source/payload freeze.

**Question:** with the same clean supervision, displaced inputs and optimization
budget, what changes when the displaced target is `S(Pu)` versus `S(Px)`?
The anti-claim is that any improvement merely comes from more clean optimization
or repeated exposure to selected reference states.

Bank identity: `CM_NEXT_KF_PAIRED_BANK_20260911A`.

- Use all 32 existing training trajectories; validation/protected arrays are
  excluded. Select input steps `15 + 32*j`, `j=0,...,15`: 512 clean anchors.
  This is fixed time-bin sampling, not error-selected anchors.
- At each anchor draw one real white Gaussian field with PCG64 and SeedSequence
  `[2026091101, trajectory_seed, input_step]`. Project it to the canonical
  mean-zero rectangular 2/3 band. At N256 this projection has rank 29,240;
  multiply by `sigma * 256/sqrt(29240)` so its **expected** spatial mean square
  is `sigma^2`. Inherit `sigma=0.0107071767634698` physical RMS from the previous
  training-only calibration (`0.00263502` in training-state units).
  Do not normalize individual draws, clip, resample or search scales.
- Store both antithetic inputs `float32(u32 +/- eta)`, giving 1,024 displaced
  rows. Record realized RMS, full-vector rounding error, antithetic discrepancy
  and canonical projection change. This is band-limited Gaussian augmentation,
  not measured normal noise or the previously examined model-error directions.
- Every training label uses the same original N256 map A: clean `S(Pu32)` and
  displaced `S(Px32)`. Keep FP64 solver answers; the learner casts targets to
  FP32. Never replace the clean target by a model prediction or silently mix
  refined D labels into sentinel rows. There are 1,536 A endpoint calls.
- Numerical sentinels are seeds `2026090611/12,2026090801/02` at input steps
  `15/239/495`: twelve anchors, both signs. Add B/C/D at every sentinel (108
  calls), original-FP64 clean-input replays (12), and one exact same-process
  repeat of the first A clean solve: **1,657 calls total**. Reuse the previous
  A/B/C/D definitions and all ten per-query numerical limits. Retain full fine
  answers, not only restricted states.
- At every anchor require finite canonical solver inputs/outputs, relative
  full-vector FP32 rounding error at most `1e-4`, the input projection/direction
  limits, and clean A versus archived FP64 successor RMS / actual displacement
  RMS at most `1e-3`. Fine-grid qualification remains sampled: passing twelve
  sentinels is not a uniform certificate for all bank labels. Preserve every
  failed draw/answer; no threshold or noise-law adjustment under this identity.
- CPU work cap: three hours, separate final provenance and an outer launcher
  timeout; require 2 GiB free before generation. Estimated output is under
  1.5 GiB before metadata/compression variation. No automatic cleanup, retry,
  model construction, training, rollout or protected access in the bank job.

The first adaptation comparison has three arms:

| Arm | First branch | Second branch on the same signed bank row |
| --- | --- | --- |
| Clean continuation | Original clean pair | `u32 -> S(Pu32)` |
| Paired recovery | Original clean pair | `x32 -> S(Pu32)` |
| Dynamics relabeling | Original clean pair | `x32 -> S(Px32)` |

The control is anchor-reweighted clean continuation, not uniform continuation.
All start from the identical audited clean32 terminal; do not zero its trained
head. Use seed 17, fresh Adam with betas `(0.9,0.999)`, epsilon `1e-8`, zero
weight decay, constant LR `1e-4`, and exactly 4,096 updates. Each update uses
batch eight from all 16,384 clean pairs and
batch eight from the signed bank, with loss weights `0.5/0.5`. Divide both MSEs
by the inherited **state** scale squared, not the noise scale squared. Preserve
FP32/no-AMP/no-TF32/no-clipping semantics. Use independent fixed without-replacement
sampler streams (seeds 17 and 1701), replayed across arms; keep duplicate signed
row IDs in the control. Thus each clean pair is sampled twice and each signed
bank row 32 times. Use sequential backward passes to retain batch-eight memory
usage. No validation-based selection, early stopping or automatic extension.
This is a single-seed matched adaptation comparison conditional on one parent
checkpoint; it does not estimate training-seed variability.

The separate resource check uses `resource/dynamics` for 16 updates; all three
arms have the same computation and tensor shapes. Require an idle approved
GPU with 12 GiB free, 8 GiB effective host-memory headroom and 1 GiB free disk.
Its outer watchdog is 30 minutes plus 60 seconds termination grace; the unchanged
trainer's internal work cap is three hours. Retain timing, memory and sampler
receipts, but no checkpoint, validation prediction or rollout. No resource
updates carry into scientific fitting and no fit launches automatically.

Predictions before new outcomes: recovery should reduce learned response and
clean-successor error on its augmentation law; relabeling should better preserve
trusted response and reduce response defect. Neither implies a universal rollout
winner. Test transfer on the separately frozen model-error probes and retain
clean forcing, response defect and signed alignment separately. Record failures
and clean no-harm violations rather than selecting a preferred endpoint.
Qualified bank production precedes the trainer/resource-smoke freeze. After
training, freeze common-bank diagnostics and a ranking prediction before the
new rollout evaluation; old validation outcomes are not fresh confirmation.
The Gaussian bank alone cannot establish corrective benefit or C2. Explicit
correctors remain in the later representative comparison, not this target-only
contrast. The tracker owns actual implementation/review/deployment status.

## R. Readiness And Candidate Decision

Selected route, approved September 6: periodic 2D forced Navier--Stokes/Kolmogorov
flow with the retained full-state solver and existing AutoDL instance.
Square-cylinder wake flow is parked, not a second production campaign.
Close numerical/population/model readiness by September 7.

The official SU2 [square-cylinder configuration](https://raw.githubusercontent.com/su2code/SU2/master/TestCases/unsteady/square_cylinder/turb_square.cfg)
is a short RANS--SST BDF2 regression restart, not a production trajectory.
The [von Karman tutorial](https://su2code.github.io/tutorials/Inc_Von_Karman/)
describes periodic shedding. Pin current public resource bytes before any use.

Required checks:

- Complete state/history/forcing/boundary/mean-mode contract; restart equality
  and repeated execution.
- Time-step and spatial sensitivity on clean AND displaced states; numerical
  error floor smaller than the response/intervention differences interpreted.
- Independent trajectories with immutable role assignments.
- Competent clean PCNO with an informative rollout gap; no deliberately weak
  training, manufactured blowup or mandatory tiny global PCA rank.
- Measured solver-label, training, inference, memory and storage costs.

### Immediate pilot: CM_NEXT_KF_R0_20260906A

Purpose: engineering readiness and preliminary finite-grid response/cost.
No old population, model, checkpoint or protected outcome is used.

| Item | Fixed local screen |
| --- | --- |
| State | Full mean-zero 2/3-band vorticity on periodic square; zero mean velocity fixed. |
| Physics/numerics | N64, L=2*pi, viscosities {0.01,0.005}, drag .1, forcing amplitude 1, forcing mode 4; macro step .05, maximum substep .002. |
| Fresh initial states | Seeds 2026090601/02; laminar state plus random-phase modes (1,1),(1,2),(2,1),(2,2),(3,1),(1,3), perturbation RMS .25 of laminar RMS. |
| Anchors | One anchor after four macro steps for each viscosity/seed. |
| Displacements | Low mode (1,1), high retained mode (18,17), both signs, RMS .01 of anchor RMS, canonicalized explicitly; not called tangent/normal. |
| Temporal check | Same input, base versus half maximum substep. |
| Spatial check | Same lifted input, N64 versus N128 using half substep for both; report restricted discrepancy AND discarded fine-state content. |
| Other checks | Fresh-stepper restart, exact repeat, raw rejection/projection residual, trusted response, energy/enstrophy, timing/substeps. |

These are candidate configurations, not qualified coherent/chaotic regimes or
a final population. Diagnostic probe magnitude is not the training-noise
prescription. Record protocol, source hashes, runtime, per-query measurements
and final output hashes in a fresh ignored packet; refuse overwrite.

The pilot cannot qualify stationarity, a data manifold, long-time complexity,
a population law, PCNO readiness or the old low-viscosity N256 setting. A small
restricted spatial discrepancy alone does not establish closure. If numerical
differences obscure a response, refine/reject the setting rather than label it
trusted. Long-time structure and resolution checks precede production freeze.

### Refinement and model readiness

September 6 successor `CM_NEXT_KF_R0_20260906B` repeats the physical recipe at
N128/N256 under a new identity. A's exact source bytes are preserved separately;
the B packet binds that archive and A's manifest. B's anchors are regenerated
on N128, so only the within-B lifted-input comparison is exactly paired.
The four cases completed in 543.66 seconds. Maximum high-probe spatial response
discrepancy divided by input RMS is 0.001337 at viscosity .01 and 0.02998 at
.005. This is a short-anchor refinement result, not a population qualification.

The periodic PCNO adapter now has tested forcing features, periodic gradients
and attributable output restriction. Its synthetic CPU fit and full-grid GPU
smoke are engineering checks, not PDE fits. The remaining September 7 gate is
long-time clean/displaced resolution evidence, an explicit fresh population
law and an affordable measured training/storage budget. See the tracker for
source-bound outputs; do not generate production data into an almost-full disk.

### Bounded longer-time development screen

Completed identity: `CM_NEXT_KF_LONG_20260906A`. Use only viscosity .01 at N128,
seeds 2026090603/04 under the same six-mode initial law, 512 macro steps of .05
(T=25.6). These are fresh development trajectories, not training or confirmation.
Keep viscosity .005 conditional until the first route is qualified.

At steps 0, 64, 256 and 512, repeat the fixed clean/signed-low/signed-high assay
with temporal refinement and N128/N256 same-input comparison. Halve both the
maximum substep and CFL factor for refined steppers (.002/.4 to .001/.2), so
temporal refinement still operates when advection or diffusion limits the step.
This is a new-C numerical choice; completed A/B sources remain unchanged. At step 256,
also compare eight independently composed clean steps. Retain both discarded
fine-state and discarded fine-response content. No probe is labelled normal.

Freeze engineering limits before execution: clean spatial relative error
1e-3; spatial response discrepancy/input RMS 1e-2; temporal response discrepancy/
input RMS 1e-3; discarded fine-state fraction 1e-3; eight-step restricted clean
error 2e-3. Report each gate and observed value, not only an overall pass.
These tolerances do not promise resolution of smaller intervention differences.

Save canonical float64 trajectory blocks and incremental diagnostics; require
512 MiB free local space and stop at a 90-minute wall budget with an honest
incomplete receipt. B's measured timings suggest about 40 minutes serial work,
but later CFL costs are unknown. Describe variability, physical budgets and
spectral redistribution; do not certify stationarity, chaos, manifold dimension
or production-population readiness from this screen alone.

### Solver-labelled debug fit

After the complete longer-time packet passes its audit, use only seed
2026090603, states 0--32, for a 32-pair engineering fit. This repurposes those
development states for debugging; they cannot later be called untouched
validation or prospective evidence. No existing protected data are used.

Fix the current PCNO resource preset (width 64, four layers, 12 modes/axis),
zero-initialize its residual output head, fit one RMS scale from the 32 inputs,
and run 256 Adam updates at 1e-3, batch 8, seed 17 and sequential batches.
Keep shared numerical restriction explicit. Measure actual-batch memory/cost,
teacher-forced errors and a 32-step in-sample recurrence. Retain the terminal
checkpoint, inputs/targets, raw/restricted predictions and loss logs for
independent recomputation. This is not a method comparison or a held-out PDE
accuracy result, and it does not calibrate production recovery noise.

Before production, freeze a fresh trajectory-disjoint population and establish
clean-model competence/convergence on development data. Do not choose a weak
training budget to manufacture rollout instability.

### Actual-batch resource check

`CM_NEXT_KF_CLEAN_RESOURCE_20260907A` measures the proposed training path on
the completed, hash-bound N256 population C. Use width 64, four layers,
12 modes/axis, projection hidden width 128, batch eight, float32 without AMP or TF32,
zero residual output head and Adam at 1e-3. Run only 16 seeded, shuffled
training updates with train-RMS-normalized MSE; exclude the first three from
steady-state timing. Fit scale from the 4,096 training inputs in float64;
keep one float32 state store on CPU and transfer each batch to the GPU.
Development states are loaded for memory accounting, never used in updates.
Record data-loading cost, finite losses/gradients, memory and batch-one/eight
inference costs. No checkpoint, rollout ranking or competence claim comes
from this probe. The approved probe completed and passed independent audit:
0.198686 seconds per warm inclusive update and 6.57/9.30 GiB peak GPU
allocated/reserved memory. Exact evidence belongs in the experiment tracker.

### First Full Clean Pilot: Frozen Recipe

`CM_NEXT_KF_CLEAN_20260907A` completed and passed the frozen one-step/stopping
gates after its single extension; the September 8 audits are in the tracker.
Its recipe below remains unchanged. It uses the
qualified C packet and the same N256 PCNO architecture,
float32 precision, output restriction and train-only scale as the resource
probe. Start fresh at seed 17 with a zero residual output head; no resource
weights survive. Use all 4,096 training transitions, batch eight, with one
seeded without-replacement permutation per epoch. Preserve the sampler and
all relevant RNG state in checkpoints.

Fix 32,768 updates (64 epochs) of train-RMS-normalized one-step MSE, Adam
(betas .9/.999, epsilon 1e-8, no weight decay), no AMP/TF32 and no gradient
clipping; abort on nonfinite loss or gradient. For update u=1,...,512, use
learning rate 1e-3*u/512. Thereafter use cosine decay from 1e-3 to 1e-4,
with phase (u-512)/(32768-512), reaching 1e-4 at update 32,768.

Evaluate every training/development pair before training and every 4,096
updates. Report pooled relative L2, train-scale-normalized MSE, each development
trajectory, and input-step bands [0,64), [64,256), [256,512). Persistence here
predicts the current input at the next time, not the initial state forever;
also retain zero-predictor controls. The engineering competence criteria are:

- pooled development relative L2 at most .02; and
- in each of the four development trajectories and all three time bands,
  sqrt(sum learned squared error / sum one-step persistence squared error)
  at most .5, using identical input/target pairs.

These are declared readiness targets, not consequences of the theory. Do not
replace pooled norms with means of per-state ratios. Undefined zero-denominator
comparisons or nonfinite evaluation results are unresolved/failed checks,
not epsilon-adjusted passes.

Use one fixed extension rule, unrelated to rollout outcomes. Let E(u) be the
pooled development relative L2 at update u. If
1-E(32768)/E(24576) > .05, continue for exactly 16,384 updates at 1e-4,
retaining optimizer and sampler state. Otherwise do not extend automatically.
Record the 32,768 terminal state and, if triggered, the 49,152 terminal state;
never choose weights by rollout quality. After extension, apply the same
five-percent plateau check to E(40960) and E(49152). Competence failure or a
still-improving terminal pilot calls for diagnosis, not promotion of a weak or
unsettled baseline. No second automatic extension is authorized by this plan.

Deploy only the accepted terminal checkpoint. Freeze the common comparison
budget after this pilot, then evaluate its rollout gap and design calibrated
displacements. Checkpoints retain model/optimizer/schedule/RNG identity;
per-pair metrics and small replay sentinels make evaluation auditable without
duplicating every teacher-forced field. Full rollout storage is budgeted
separately. Preserve the completed resource source archive when extending its
entry point and test. Source review, CPU tests and exact new-payload approval
still precede scientific deployment.

Checkpoint retention is bounded: one atomic rolling latest checkpoint, the
base terminal and the optional extended terminal. CPU tests verify exact
optimizer/sampler/RNG continuation; terminal raw/restricted sentinel replay
uses a relative-RMS tolerance of 1e-6. Bitwise CUDA replay is not claimed.
The external safety cap is five hours plus sixty seconds of termination grace;
an interrupted or incomplete packet cannot qualify the pilot.

Measured-cost projection: allow roughly 2--2.5 hours for the base fit and
one-step evaluations, plus about an hour if the extension triggers. This is
an estimate, not a guaranteed runtime or a model-competence prediction.

### Fresh training/development population

September 6 result: this exact packet completed but failed both spatial limits
during the early transient on all twelve trajectories. Keep the contract and
all data unchanged; it is not a qualified production population. A focused
native-N256/N512 early-transient comparison precedes any replacement campaign.

Freeze `CM_NEXT_KF_POP_20260906A` before generation. Keep the same finite-time
six-mode phase law, viscosity .01, N128 and all other longer-time solver
parameters. Retain states 0--512 at spacing .05; no burn-in or stationary-law
claim. Assign seeds 2026090611--2026090618 to training (4,096 transitions),
and 2026090621--2026090624 to development (2,048 transitions). No readiness
seed is included. Diagnostic and confirmation populations remain separately
unfrozen and ungenerated; this packet does not authorize their reveal.

For every trajectory, check clean inputs at steps 0, 64, 256, 512 and the
maximum-palinstrophy retained state (earliest maximum; deduplicate anchors).
Use the same-input base/half-step/fine-grid comparison, halving both dt_max
and CFL. Preserve clean spatial and discarded-fine-state limits of 1e-3;
report temporal error separately. On seeds 2026090611 and 2026090621 also
compare eight independently composed refined coarse/fine steps from step 256,
with endpoint limit 2e-3. Keep every failed seed and the full packet; do not
silently replace seeds, relax limits or truncate the horizon after seeing them.
These spotchecks are not a uniform numerical guarantee over all retained states.

Measured development costs suggest 19--63 minutes for trajectory generation,
plus roughly 7--15 minutes for checks. Reserve 90 minutes and at least 1.5 GiB
free output space; keep partial blocks and an honest incomplete receipt on
budget expiration. Expected compressed trajectory storage is about 745 MiB.

The original Clean-budget proposal is now governed by the frozen first full
Clean pilot above, following the successful actual-batch N256 resource check.
The pilot's update count is not yet the final comparison budget, and no
checkpoint is selected for desired rollout failure. Actual displaced-input
laws still need separate numerical checks before any solver-relabeling claim.

### Focused native-resolution refinement

Freeze `CM_NEXT_KF_PEAK_20260906A` before execution. This is a numerically
selected stress screen, not a fresh validation population. Select training seed
2026090617 and development seed 2026090621: each had its role's largest N128
peak discarded-state fraction. Regenerate the unchanged initial law natively
at N256 through step 64; do not lift the failed N128 trajectories.

Query steps 16, 24, 64 and the earliest native maximum-palinstrophy state,
deduplicating anchors. Compare identical inputs using N256 base (.002/.4),
N256 refined (.001/.2), and N512 refined (.001/.2) maximum-step/CFL settings.
Keep both clean spatial and discarded-fine-state limits at 1e-3. Report temporal
error without introducing a new threshold. From seed 2026090617's native peak,
also compose eight refined N256/N512 steps independently, without resetting
the fine state, retaining the existing 2e-3 endpoint limit. No displaced-state
qualification or full-population qualification is implied by this screen.

Retain native trajectory blocks, every queried input and base/refined/fine
successor, and every coarse/full-fine continuation state for independent
recomputation. Freeze sources and parent population manifest before execution.
Require 512 MiB free storage and a 90-minute compute deadline checked each
substep, with retained partial output. The hidden launcher allows five more
minutes for shutdown/hash work before terminating only its own child process.
Estimate 45--65 minutes; N512 cost is unmeasured, so update the estimate
from the first completed fine query. Neither a pass nor a failure authorizes
relaxing the previous limits or changing the population after seeing outcomes.

### Full N256 population qualification

The selected early-window screen passed independent array/source audit. Freeze
`CM_NEXT_KF_POP_20260906B` for the next local run; this replaces neither the
failed N128 result nor its archived source. Regenerate all twelve unchanged
seed/role assignments natively at N256 through step 512, using the same initial
law, physics, macro step and base integration settings. No seed replacement,
burn-in, horizon truncation, confirmation access or model selection is allowed.

For each trajectory, compare N256 base/refined and N512 refined successors
from identical clean inputs at steps 0, 16, 24, 64, 256, 512 and the earliest
native maximum-palinstrophy state; deduplicate anchors. On seeds 2026090617
and 2026090621, independently compose eight refined coarse/fine steps from
both the native peak and step 256 (deduplicate if equal). Keep clean spatial
and discarded-state limits at 1e-3, and every continuation endpoint at 2e-3.
Temporal error remains a separately reported diagnostic. Retain full query and
continuation arrays, native trajectory blocks and all failed/incomplete cases.

The passed two-seed screen is the source-bound parent; its manifest is
`da4991f1e1d7c5f5b93f551aaf77d38f73c41dc08a61a878b53893ca358edfe6`.
This is sampled numerical qualification of the declared finite-time population,
not a uniform, displaced-input, stationary-law or learned-model guarantee.
Future calibrated displacement laws still require their own checks.

Plan 4.5--6 hours locally, provisionally extrapolated from measured native and
fine-grid costs. Require 6 GiB free output space, an eight-hour compute deadline
checked each substep, and five minutes of external shutdown/hash grace. Retain
partial output on expiration; do not extend the budget after seeing numerical
outcomes. This run does not upload the resulting population or launch a model.

### Restart-audited AutoDL recovery

B closed incomplete after local standby; preserve its packet and exact source
archive. Owner-approved successor `CM_NEXT_KF_POP_20260907C` recovers only the
missing work on the existing AutoDL instance. The B manifest and independent
audit hashes are pinned in the tracker and generator. No seeds, roles, physics,
anchors, horizons or numerical thresholds change.

Before continuation, compare the Linux runtime against saved B answers:

- Regenerate seed 2026090611's initial state.
- Replay base/refined successors at seed 611 step 0 and seed 616 step 16,
  twice each.
- Restart seed 616 at stored step 8 and advance eight native steps twice,
  comparing every successor with the retained prefix.
- Replay seed 611's N512 fine successor at its native peak, step 14, twice.

This is 26 solver calls. Every saved-answer comparison must have relative L2
error at most 1e-10; repeated outputs must be bitwise identical within the
new runtime. Cross-platform bitwise equality is not required. Retain replay
arrays and a started/completed/failed receipt; fail closed before continuation
if any check fails.

Copy the five complete cases byte-for-byte into a fresh C packet, never using
hardlinks. Preserve seed 616's native prefix through step 16 and complete
step-0 query. Recompute its entire partial step-16 query in Linux, then resume
native evolution at step 17. Find its peak over both the retained prefix and
new continuation. Generate the six unstarted cases in the original order.
Keep inherited source/timing provenance separate from new execution; all B
files remain unchanged. C receives fresh global receipts and a complete hash
inventory rather than relabelled B receipts.

The new eight-hour elapsed budget includes replay, copying and generation.
Require 6 GiB free after payload upload. A Linux eight-hour-five-minute
watchdog allows a further 60 seconds before force-stopping its own process
group. Retain partial evidence on expiration. Estimate completion from actual
Linux throughput after launch, not the interrupted Windows elapsed time.
Full qualification still requires all twelve trajectories and every declared
gate, independently audited. No new model training is part of this recovery.

### Numerical restriction must be explicit

The solver advances canonical states. `advance_projected(x)` is `S(Px)`,
not evolution of unchanged x. Either represent the complete canonical state
or declare a shared numerical projection in every PCNO arm. Retain raw output
and projection displacement; this baseline is PCNO with a stated restriction,
not correction-free vanilla PCNO. Float32-to-float64 conversion does not restore
canonicality. Downsampled fine trajectories are not silently treated as a closed
coarse Markov flow.

## M. Common Inputs And Different Targets

For clean u, displaced x=u+delta, trusted S and complete deployed Psi:

```text
b = Psi(u) - S(u)
R_model = Psi(x) - Psi(u)
R_solver = S(x) - S(u)
recovery_error = ||Psi(x) - S(u)||
dynamics_error = ||Psi(x) - S(x)||
response_defect = ||R_model - R_solver||
```

Use declared node-weighted, channel-scaled RMS. Report clean forcing,
learned/trusted response, response defect, secant gains, signed alignments
and exact algebra checks. Zero denominators are unresolved, not hidden with
epsilon. Short multi-step assays compose the complete learned and reference
maps independently. Stochastic methods use common random tapes for paired
responses and separately report across-tape variation.

Geometry is fitted only from clean training data. Defect does not define ID,
OOD or the tube. Local PCA remains an empirical residual subspace unless
held-out reconstruction and neighborhood stability justify stronger language.
Known translation symmetry gives legitimate-response controls where applicable;
phase-aligned error accompanies, never replaces, raw trajectory error.
For the selected fixed forcing cos(4y), continuous x translations preserve the
problem; arbitrary y translations do not. Do not label both as symmetry probes.

### Population, bank and scale

Use trajectory-disjoint training, development/calibration, diagnostic and
fresh ID confirmation roles. Each regime has its own common ID train/evaluation
law; no exogenous OOD claim. Freeze seeds/counts, burn-in, sampling and horizon
after readiness and before production generation. Do not assume stationarity.

Recovery and relabeling share identical displaced arrays and clean weights;
only S(u) versus S(x) changes. Calibrate one training-noise RMS from train-only
pilot errors, with no optimal-scale search; freeze a small diagnostic scale
range around it. RMS matching does not match direction or error law.

A common bank includes canonical Gaussian, physically meaningful and pilot-model
error directions. Donor and recipient models are separate. A method-owned
encountered-state bank is a second explanatory view, not a replacement for
common inputs. Preserve direction/scale/tail summaries rather than pool unlike
query laws into one number.

### Representative matrix

Three broad baseline families: target augmentation, model-prefix exposure and
deployed correction. Use three paired seeds for stochastic training.

| Arm | Intended contrast | Expected effect / risk |
| --- | --- | --- |
| Clean / Clean-EMA | Same supervised PCNO and budget; isolate EMA. | Establish clean error, response and rollout gap. |
| Paired recovery | Displaced inputs with clean-future targets. | Lower recovery error; may suppress legitimate response. |
| Dynamics relabeling | Same inputs with displaced solver successors. | Lower response defect; need not return to clean path. |
| Curriculum EMA pushforward | Warmup/ramped exposure, detached EMA prefixes of depth 1--3. | Better prefix-error behavior; no assumed universal contraction. |
| Empirical projection | One train-only deterministic projector on Clean, with identity ablation. | Reduce representation residual; may damage clean state, phase or recurrent tails. |
| PCNO PDE-Refiner adaptation | Audited conditional refinement semantics. | Recover useful low-amplitude information; report stochasticity and cost, not an asserted exact projection. |

Freeze exact projector, architecture, optimizer, schedule, update count and
checkpoint rule after tiny fit/resource smoke, before comparison. Default is
terminal checkpoints without rollout selection. Match initialization/data order
where interfaces permit and disclose exceptions. Report forward/backward calls,
GPU-hours, labels and inference latency; do not claim equal compute for
extra-prefix or iterative methods. No second weak exposure variant or exhaustive
method reproduction is required.

## P. Prospective Test

Freeze checkpoint/source/query hashes, diagnostics, prediction rule and metric
choices before fresh long-rollout outcomes. Compare a clean-error-only ranking
with a fixed response-informed ranking. Prefer a few predeclared observables;
do not fit a flexible predictor to the reveal. If data cannot justify a numeric
rule, freeze qualitative pairwise predictions with ties/abstentions.

Primary endpoints: horizon-averaged state error, forecast horizon at a frozen
development-calibrated threshold, failure incidence and tails. Secondary:
endpoint error, phase/shape, physical budgets, spectra and reference proximity.
For incompressible periodic flow use energy/enstrophy and mean-mode diagnostics;
shock/positivity/boundary leakage are not applicable. Genuine chaotic
decorrelation can remain on-attractor; compare trusted perturbation growth
without switching from trajectory prediction to statistics after seeing failure.

Report pairwise prediction accuracy/rank association beyond clean error. The
trajectory is the sampling unit: paired comparisons, trajectory-clustered
intervals, seed variation separately disclosed. Neither frames nor method
pairs are independent. Three seeds do not establish precise tail probabilities.
Keep every registered failure in the analysis.

Freeze numerical, practical tie and no-harm thresholds on development.
Uncertain/no incremental value means C2 is unsupported. Already-opened
NACA/Bump results cannot validate this claim. Existing protected populations
remain excluded; a new protected confirmation reveal needs its exact freeze
and named owner decision.

## H. Conditional Buffer-Plus-Recovery Design

Go/no-go September 9. Proceed only if the paired assays reveal a useful
response distinction and the core matrix is on schedule.

Learn faithful dynamics in a supported neighborhood; train a residual corrector
to preserve clean/legitimate nearby states and recover harmful excursions.
Fix correction inputs/schedule offline: no online solver, defect threshold or
OOD detector. Check overlapping neighborhoods for inconsistent targets first.
Compare recovery-only, relabel-only and combined versions, accounting for labels,
training and inference. Perform a focused prior-art check before claiming novelty.

Success needs the predicted response tradeoff AND rollout benefit, not just
extra data/calls. Cut this block or the second regime if it delays the core PDE
evidence or paper.

## Run Order And Failure Interpretation

1. Analytic/CPU tests and independent code-to-intent review.
2. New local solver pilot; establish measured readiness limits.
3. Use the owner-approved dedicated Kolmogorov solver and existing AutoDL.
4. Close long-time readiness, genuine tiny PDE fit, resource/storage budget;
   then freeze state/population/model.
5. Freeze comparison and launch bounded training with durable incremental logs.
6. Common-bank assay and prediction freeze.
7. Named confirmation reveal, independent raw-result audit, science freeze Sep 10.
8. Verified paper integration and final review by Sep 13.

Solver failure rejects that setting, not the family. Implementation defects
pause execution for repair under a new source identity. Negative scientific
results are reported. No post-outcome matrix expansion to rescue a claim.

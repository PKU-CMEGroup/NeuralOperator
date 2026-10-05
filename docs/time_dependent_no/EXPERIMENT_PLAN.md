# Experiment Planning: Version 2

Updated: 2026-10-04

The fixed-data and REC/DYN studies, including their twelve-case follow-up, are
complete. On October 4 the owner approved the next first stage: a REC/DYN
midpoint and a controlled four-step training comparison, with response and
geometric diagnostics. The concrete protocol below governs this stage. Longer
training horizons, Jacobian-matching training, PDE-Refiner and the second PDE
remain subsequent review stages. The [paper plan](../../paper/PAPER_PLAN.md)
is locked and must not be modified without explicit permission.

## October 4 first stage: refreshed inputs and temporal gradients

**Questions.** Does refreshing inputs with the current model help beyond
fixed model-generated inputs? What do gradients through preceding predictions
add under the same objective? Does an intermediate REC/DYN target improve
rollout? Interpret these as open comparisons, without presuming attached
training is harmful.

**Five training runs.** All start from the unchanged original pretrained PCNO,
use fresh Adam at 1e-4 for 4,096 updates, retain the original normalization and
deployed restrictions, and save checkpoints at 1,024, 2,048 and 4,096 updates.
Only the fixed terminal enters the primary comparison.

| Run | Input to prediction step j | Target and gradient |
| --- | --- | --- |
| Four-step clean control | Ground-truth state at j-1 | Ground-truth state at j |
| Four-step fixed-input recovery | Frozen pretrained rollout at j-1 | Ground-truth state at j; input constant |
| Four-step detached | Current model rollout at j-1 | Same target; detach preceding predictions |
| Four-step attached | Current model rollout at j-1 | Same target; differentiate the entire four-step computation |
| Mixed-length midpoint | Existing fixed mixed-length displaced input | Half REC plus half DYN squared loss |

The four sequence runs use batches of four complete windows, the same shuffled
sample sequence (seed 1701), and the mean loss over all four predicted steps.
There is no optimizer update inside a window. Detachment preserves the forward
rollout and changes only differentiation at a shared parameter value. Training
will subsequently generate different inputs for the detached and attached models.
The clean control uses the same windows and weights. It controls the change in
time sampling and objective relative to the older clean continuation.

Select 32 windows per training trajectory: the window starting at zero and one
start from each of 31 equal-width strata over starts 1--508, using seed
2026100401. This covers onset and the full eligible interval without crossing
trajectory boundaries. Freeze the pretrained model's four input states for each
of these 1,024 windows once. No validation states enter training or sampling.
The sequence comparison is distinct from the earlier mixture of 1/2/4/8-step
endpoints; its existing REC model is context, not the fixed-input control.

The midpoint reuses exactly the original mixed-length inputs, both target
arrays, batch sizes (eight per branch), clean/displaced weights (half each),
sampler seeds 17/1701 and 4,096-update budget. It trains toward the arithmetic
mean of the REC and DYN labels. For fixed inputs this has exactly the gradient
of the equally weighted squared losses; record the target-separation constant
when reporting the latter loss. Existing REC and DYN endpoints are unchanged.
No additional solver labels are needed for this training run.

**Measurements and figures.** Reuse the original eight training and four
validation initial conditions through step 32. Retain the unchanged pretrained
model, original clean continuation and the two mixed-length endpoints. Report
per-step rollout error, clean and own-input dynamics errors, independent-probe
errors, signed error terms and physical observables. Evaluate all new maps on
the same saved pretrained rollout inputs for REC/DYN target comparisons.
Counts across initial conditions are not independent training replicates.

At initialization and the saved checkpoints, use fixed training windows to
measure the direction and magnitude of immediate-loss and future-loss gradients
at each intermediate state. Evaluate both temporal-gradient choices at each
shared checkpoint. A negative cosine is a diagnostic, not proof that attached
training is inferior. Save loss by prediction step, parameter-gradient norms,
sample indices, wall time, forward-call counts and peak allocated GPU memory.

Add time-resolved empirical geometry using only training states at each input
time: nearest-state distance and residuals to centered PCA subspaces of ranks 8
and 16. For training-case evaluation, exclude that case's entire trajectory
from the geometric representation. Evaluate validation ground-truth states as
the coverage baseline. These are finite-sample measurements, not exact manifold
distances. PCA-parallel and PCA-orthogonal responses must retain those labels;
they are not established tangent and normal dynamics.

At input steps 1, 4, 8, 16 and 31 of the fixed pretrained rollouts, compare
responses along the actual error and its PCA-parallel/orthogonal components.
Use actual-size errors for the finite-response comparison, and centered
differences at RMS amplitudes 0.01s and 0.005s for the small-response comparison.
Numerical solver refinement and agreement between these amplitudes qualify
interpretation as a local derivative. Store unresolved directions and numerical
checks explicitly. No uniform stability bound follows from these samples.

**Execution boundary.** CPU analytic-gradient, sampling, target and geometry
tests precede a GPU resource check. Use the selected personal workstation, one
training job at a time, batch four for all sequence runs and batch eight per
midpoint branch. Resource checks use disposable model/optimizer copies and never
become training checkpoints. Per-run training allowance is six hours; check the
warmed estimate before continuing. Preserve immutable input and code identities.
Stop and record a failed stage instead of silently changing batch size, horizons,
loss weights or failed cases. Generate vector figures and numerical summaries
after each completed evaluation for owner review. No broader sweep is queued.

**October 1 priority decision:** retain the fixed-data experiment to establish
the discrepancy between clean prediction and rollout quality. Reserve data
scaling for a later comparison if time permits; it is not a prerequisite for
the mechanistic study. Further clean training can reduce clean training and
validation errors while worsening rollout. In the two-case pilot, this statement
uses clean RMS pooled over inputs 0--31 and rollout RMS over outputs 1--32, with
one trajectory from each population. Keep the time-resolved curves: improvement
is not uniform at every input, and the training onset average is slightly worse.
This comparison varies training on fixed data, not the quantity of clean data.

The owner subsequently removed the independent clean probe from Experiment 1's
completion requirements. Its authorized training continues for later comparisons.
The next selected study is the four-way comparison of recovery/dynamics targets
and Gaussian/model-generated displacements, specified below. Its detailed
sampling and training protocol remains under owner review.

## Approved question structure

Section 4.2 should parallel the development of Sections 2--3:

| Paper argument | Scientific question | Evidence role |
| --- | --- | --- |
| 2.1: One-step prediction and rollout | Where does the discrepancy between clean accuracy and rollout accuracy arise? | Relate clean local error, local error on reached states, and accumulated trajectory error over time. |
| 2.2: What clean supervision leaves uncontrolled | Why can further clean training improve one-step prediction while worsening rollout? | Examine whether improvement on ground-truth states accompanies unfavorable behavior on reached states. |
| 3.1: Corrective training | Do input distributions and targets produce their intended effects? | Connect the behavior actually learned under recovery, dynamics, and response supervision to rollout outcomes. |
| 3.1: Repeated prediction | What do temporal gradients add beyond exposure to model-generated inputs? | Establish their effect in the studied setting and examine gradient disagreement as an explanation. |
| 3.2: Correction during prediction | What does the prediction-time correction accomplish? | Distinguish improved proximity or individual-state plausibility from accurate evolution and rollout. |

## Experiment 1: approved question and evidence requirements

**Question:** Does further clean training worsen rollout because accuracy
deteriorates on states reached during prediction, despite improved accuracy on
ground-truth states?

This is a hypothesis to test. First assess the reported clean-continuation
discrepancy; any reuse of historical outcomes requires source verification.

| Approved evidence requirement | What it would establish |
| --- | --- |
| Establish the clean-error/rollout discrepancy over a common time interval, comparing continuation with the unchanged pretrained model. Check clean competence on both training and validation ground-truth states. | Identify where the phenomenon occurs and whether poor ordinary clean generalization already explains it. |
| Examine local true-flow defects at each model's own reached states alongside its accumulated trajectory error, retaining the evolution over time. | Determine whether deteriorating rollout accompanies loss of local accuracy, beyond what clean-state errors reveal. |

Supporting evidence would combine maintained clean competence with worse rollout
and increased local defects on reached states. These observations alone do not establish exact
manifold distance or a causal explanation of the rollout difference.

If local defects remain small despite worsening rollout, examine error
propagation and dynamical sensitivity. If clean accuracy is already poor over
the relevant interval, the intended explanation is not isolated there.

Different models' own rollouts generally visit different states. To claim that
continuation worsened local prediction at a given input, compare maps on common
states or provide an equivalent control. The common-input evaluation below
provides this control. Local-defect
magnitude alone also does not quantify its contribution to trajectory error:
propagation and the directions of the error terms can reinforce or cancel.

The owner approved these requirements as the first experiment on October 1,
selected the existing pair, and then approved proceeding with the proposed
measurements and visualizations. After reviewing the pilot figures, the owner
also authorized training a reusable independent clean probe on October 1.

### Protocol: implementation and pilot review

Compare the unchanged clean-trained model A with its clean continuation B.
Keep their clean training data, architecture and normalization fixed.
The separately trained clean probe P is reserved for later comparisons;
Experiment 1 uses direct numerical next-state labels and does not depend on P.

Start A and B from the same initial conditions. At each input time n, collect
the corresponding ground-truth state and both predicted states. Obtain trusted
next-state labels by restarting the numerical flow map from each predicted
state, without inserting these labels into either rollout. The same label can
evaluate A and B on that input, and can also be reused by P later.
Qualification of the numerical restart and
handling of numerical failures are specified under the initial pilot below.

Evaluate A and B on all three input sources: ground-truth states, A's rollout
states and B's rollout states. This common-input comparison distinguishes a
change in the learned map from a change in the states it visits. Also retain
each autonomous trajectory's accumulated error. A and B's clean errors must be
reported separately on training and validation trajectories.

Keep time explicit. Aggregation is pooled RMS over spatial points and
trajectories separately at each time, normalized by one fixed clean-training
scale shared by all curves. Do not normalize by model-generated state amplitude.
Keep training and validation populations separate and retain paired per-trajectory
values. Input step n yields a local next-state error for step n+1; plotting labels
must align it correctly with accumulated trajectory error.

The main comparison shows clean errors, accumulated rollout errors, and
own-rollout local defects. A common-input comparison supports interpretation
of model versus input effects. Existing working figures that mark the probe
as missing predate the owner's decision to reserve it for later comparisons.

**Approved setup choice:** the owner selected the existing pretrained/clean-
continuation pair on October 1. Do not retrain or replace that pair.

Use the locally present pair from kf_fixed_data_20260916a: the unchanged Clean32
checkpoint in input/train/checkpoint.pt and its continuation in
fit_clean_continuation/terminal.pt. Saved receipts describe the same PCNO
architecture and 32 clean training trajectories, with 49,152 original training
updates and 4,096 additional clean updates. The pilot's authenticated loaders
recompute checkpoint, data and required source hashes before evaluation.

Propose the original eight training and four validation rollout cases, beginning
from their original initial conditions, through output step 32 (time 1.6 at
the saved step size 0.05). Evaluate local defects at inputs 0--31. These are
previously examined development cases, not a new independent confirmation.
Keep all cases, and define numerical failure reporting before execution. This
interval contains the reported discrepancy; clean competence must be shown
throughout it rather than assumed from the early steps. Report later clean
generalization differences even if they weaken the intended explanation.

The available reference arrays cover 32 training and four validation trajectories
through step 512. Propose clean-state evaluations of A and B on all these
trajectories over inputs 0--31, with the original eight training rollout cases
identifiable for like-for-like comparison. Preserve the time-resolved curves;
do not use a time-pooled clean score to justify competence at every step.

Saved rollout metrics are dense, but state snapshots are only dense through
step 8, followed by selected later steps. Replaying the two frozen models to
save every state through step 32 is therefore required. Existing assays did
not supply trusted displaced-state next states. At most 768 primary reached-state
labels cover the two models, 12 cases, and 32 inputs; shared inputs can reuse
labels. Numerical qualification needs a separate evaluation budget. These are
new diagnostic evaluations, not already available results.
The same labels support all common-input model and probe evaluations.

### Reusable independent probe for later comparisons

P tests whether reached states also challenge a separately trained clean model.
Before interpreting this diagnostic, calibrate P on ground-truth states over
the same time interval. Its errors are not automatically manifold distances.

No independently initialized clean probe was identified in the selective asset
check, including the original Clean8 and Clean32 runs (both initialized with
seed 17). The owner subsequently authorized new clean probe training and its
reuse in later experiments. Initialize P from scratch with seed 29; use the
same 32 clean training trajectories, PCNO configuration and stored normalization
as A. Preserve the original single-branch clean recipe: batch 8, Adam, zero final
head, and exactly 49,152 updates with the original warmup/cosine/constant learning
rate schedule. Seed 29 also controls the clean-transition permutations.

Choose the fixed terminal checkpoint regardless of diagnostic results. Do not
load A/B weights, train on rollout states, or access validation data during
training. Record data/source hashes, initialization identity, optimizer/sampler
state and the checkpoint identity so the frozen probe can be reused. A short
resource phase verifies finite updates and estimates runtime before the full
run starts from the same fresh initialization; resource weights are discarded.
The 16-update resource check measured about 0.794 seconds/update (10.84 hours
for the fixed schedule) and 6.57 GiB peak allocated GPU memory. Use a 14-hour
session limit on the selected workstation; this adjusts the historical time
allowance without changing the scientific recipe. A, B and their recovery
descendant cannot serve as P. Shared data and architecture mean independent
initialization does not imply statistically independent prediction errors.

Run `python -m scripts.time_dependent_no.fit_kolmogorov_clean_probe --input`
with the authenticated training packet, a new `--output`, `--phase resource|fit`
and `--device cuda` on the selected workstation. This is one probe training run,
not authorization for a hyperparameter search or additional study populations.

### Initial pilot

Use the first original training case (2026090611) and first validation case
(2026090621), selected before inspecting the new measurements, through step 32.
The pilot establishes measurement correctness and presents actual figures for
owner review; it is not the full population comparison. Clean competence is
evaluated on these same two cases for the pilot. The full study retains the
all-32-training/four-validation clean evaluation described above.

Reuse the authenticated model loaders and the original fixed-grid solver
(256 square grid, step 0.05, viscosity 0.01, drag 0.1). Keep the model's inputs
and deployed outputs unchanged. Numerical restart labels advance the FP64
canonical projection of each FP32 input; report that projection change.
At input steps 0, 7, 15 and 31, halve both the solver's maximum substep and CFL
number. Compare clean restarts with archived next states at the same steps.
These sampled differences indicate numerical sensitivity, not uniform bounds.
Retain physical diagnostics and the signed local-error/propagation decomposition
for interpreting any surprising result. Nonfinite states or failed numerical
evaluation produce an explicit failed run; do not omit cases from averages.

Run `python -m scripts.time_dependent_no.evaluate_kolmogorov_local_defects`
with `--run-root`, a new `--output`, and `--device cuda` on the selected
workstation. Default scope is the two-case pilot. Render its saved metrics with
`python -m scripts.time_dependent_no.plot_kolmogorov_local_defects --input`
and `--output`. These two entry points own the measurements and figures;
per-case arrays and metadata stay in ignored artifacts.

The owner found the first figures unclear. The revised primary display fixes
one evaluating predictor per column and uses three consistently colored curves
for its errors on ground-truth, pretrained-rollout and continued-rollout inputs.
Every one-step prediction is compared with the true next state from that same
input. Plot these errors at the input-state time n*0.05, with the target at
(n+1)*0.05 explained explicitly. A separate figure shows accumulated rollout
errors. This changes the presentation, not the saved measurements.

## Experiment 2: targets and displacement sources

On October 1 the owner authorized preparing and queuing this comparison after
the independent clean probe, and selected equal sampling of rollout lengths
1, 2, 4 and 8 for the mixed condition. There are six new training runs, with
the unchanged pretrained model and existing clean continuation as baselines.
For each displacement condition, REC and DYN use identical displaced training
inputs; only their target changes:

| Displacement source | Recovery target at x = u + eta | Dynamics target at x = u + eta |
| --- | --- | --- |
| Gaussian | Phi(u) | Phi(x) |
| Native one-step error | Phi(u) | Phi(x) |
| Native errors from 1, 2, 4 and 8 steps, equally sampled | Phi(u) | Phi(x) |

Use the existing pretrained model as the common frozen displacement generator F.
The independent probe is a separate evaluator and is not needed to generate
these inputs. Attached/detached multistep training is reserved for a later
comparison of refreshing generated inputs and propagating temporal gradients.

**Owner's displacement decision:** sample model-generated errors at different
times and after one or several prediction steps, retaining their actual sizes.
Starting from a ground-truth training state u_n, a k-step prediction gives
x = F^k(u_n) and eta = x - u_(n+k). The clean state associated with x is
u_(n+k), so REC targets u_(n+k+1) and DYN targets Phi(x). Do not normalize or
rescale eta to a prescribed amplitude. The starting-time and rollout-length
sampling determine the distribution of generated errors.

Gaussian displacements should have roughly the same order of magnitude as the
errors encountered during deployment. Calibrate their scale from the frozen
model's training-rollout errors before comparing interventions; exact matching
of individual norms or the full amplitude distribution is not required.
Use the same Gaussian inputs for the two target choices.

### First queued comparison

Use 32 clean centers per training trajectory (1,024 per condition). Draw one
input time from each of 32 equal-width strata over steps 8--511, using seed
2026100102. All three conditions use these same centers and clean next states.
For center u_t, the one-step input is F(u_(t-1)); the mixed input is
F^k(u_(t-k)), with eight occurrences of each k in {1,2,4,8} per trajectory,
randomly assigned across the time strata. This controls the ground-truth time
distribution across exposure conditions. The clean loss branch still samples
all 512 transitions on all 32 training trajectories.

Calibrate one Gaussian ensemble scale to the pooled RMS of these mixed native
errors, using only training data. Draw independent mean-zero, band-limited
Gaussian fields with seed 2026100103. Their expected squared RMS equals the
squared calibrated scale; do not normalize each draw. Retain and report the
realized error-magnitude distributions. This single Gaussian condition is a
broad scale comparison, not an exact magnitude-matched direction experiment.

Freeze all displaced inputs before training. REC uses archived clean next
states; DYN uses the existing FP64 Fourier/RK4 solver restarted from each exact
FP32 input's canonical projection. Record all projection, clean-restart and
label-rounding discrepancies, with sampled halved-step/CFL refinement across
the set. Abort on failed numerical qualification rather than omitting rows or
silently changing amplitudes. Reuse labels when the mixed input has k=1 and is
identical to the one-step input. There are 2,816 distinct primary displaced
solver labels, 1,024 clean-restart checks and at most 60 refined labels.

All six models start from the original pretrained checkpoint with fresh Adam,
learning rate 1e-4, 4,096 updates, batch eight per branch and equal clean/displaced
loss weights. Use the existing clean sampler seed 17 and displaced sampler
seed 1701, giving identical sample tapes across all six runs. Keep the original
architecture, deployed output restrictions, fixed normalizer and FP32 precision.
Save periodic recoverable checkpoints and use the fixed terminal checkpoint;
do not select by validation or rollout results. This is one controlled training
replicate, not a multi-seed comparison. Each condition uses a finite set of
displaced inputs, including Gaussian recovery; it is not online fresh-noise
training.

The entry point is
`python -m scripts.time_dependent_no.run_kolmogorov_displacement_comparison
--input TRAIN_PACKET --output NEW_DIRECTORY --after-probe PROBE_FIT --device cuda`.
This new runner is necessary because the historical generator rescales native
directions and pools displacement types in one training set. The runner reuses
the existing authenticated model/data loader, solver and two-branch optimizer
update. Synthetic CPU tests precede deployment. A frozen source copy on the
selected workstation waits for successful probe completion and a verified
terminal checkpoint, then prepares inputs and trains the six models serially.
It stops on failure; no automatic retuning or additional conditions are queued.
The preparation allowance is four hours and each training allowance three hours.

On October 2 the owner authorized retrieval, careful analysis, autonomous
rollouts and evaluation with the independent probe. The first diagnostic
comparison retains Experiment 1's first training and first validation initial
conditions through step 32. Evaluate the two existing baselines and all six
corrective terminals; wait for any unfinished training without selecting an
intermediate checkpoint. Pending methods remain explicit in partial figures.

Compare accumulated rollout error, each model's clean one-step error, its local
true-flow defect on its own trajectory, and the independent probe's defect on
that same trajectory. The model and probe use the identical numerical next-state
label for each reached input. Calibrate the probe on ground-truth states over
the same time interval. Keep the signed error decomposition and energy/enstrophy
checks: a large dynamics defect can be a useful correction. Probe errors are
diagnostics, not exact manifold distances. Use the fixed training scale without
a percentage multiplier, and plot local errors at the input-state time.

`evaluate_kolmogorov_corrective_rollouts.py` owns this two-case evaluation and
can wait for the fixed terminals with `--watch`. It calls
`plot_kolmogorov_corrective_rollouts.py` after each completed method, producing
separate rollout, model-defect and probe-defect figures, supporting alignment
and physical diagnostics, a training-exposure plot and a numerical table.
No new validation population, protected test set, attached multistep training
or data-scaling study is opened. The manuscript and locked paper plan stay fixed.

## October 2 follow-up: target behavior and the remaining controls

The owner approved proceeding with the paired recovery/dynamics diagnostics,
closing the identified training/sampling/evaluation gaps, and producing figures
for Section 4.2. This authorization includes the bounded follow-up below and
the original eight training/four validation evaluation cases. The paper plan
remains locked. Existing checkpoints and result records remain immutable.

**Questions and figures.** Evaluate every original map on the same pretrained
rollout inputs, then each REC/DYN pair on both of its generated paths. Plot
recovery error, dynamics error, target separation and signed cancellation versus
input time. Own-path recovery error is the next rollout error; do not count it
as independent evidence. Retain clean/probe calibration and physical observables.
Report individual paths as well as population RMS; path dispersion is not
training-seed uncertainty. Export vector PDF, PNG previews, numerical arrays,
and captions identifying the predictor, input generator, target and averaging.

**Training duration.** Resume all six original optimizers/samplers from update
4096 to 8192, preserving checkpoints at 4096, 6144 and 8192. This tests budget
sensitivity; it is not an assertion of convergence. The old trainer overwrote
intermediate `latest.pt` files, so earlier intermediate weights are unavailable.
Use fixed-terminal comparisons, not validation-selected checkpoints.

**Exact clean control.** Train from the original pretrained checkpoint with
the identical two-branch sampler, replacing the displaced branch inputs by
their associated ground-truth states. Run through 8192, saving 4096 and 8192.
This controls the selected-center distribution and training duration.

**Early-time coverage.** Prepare 32 one-step inputs per training trajectory:
all input times 1--7, nine times drawn without replacement from 8--31, and one
time from each of sixteen strata over 32--511. Keep 1024 inputs, native error
magnitudes, the original frozen generator, optimizer, branch weights and sample
tapes. Train REC, DYN and the exact clean-center control for 4096 updates each.
Compare these with the original-center models at 4096, separating coverage from
the duration extension. Input time zero is clean and has no preceding rollout
error. Use seed 2026100201 for the new center selection.

**Gaussian amplitude control.** At each original clean center, reuse the
original Gaussian direction and set its ensemble RMS to that row's native
mixed-error magnitude. Do not rescale the native errors. Train REC/DYN for 4096
updates with shared new solver labels. This separates the amplitude profile
from the original constant-scale Gaussian comparison. It does not make the
Gaussian distribution identical to the native distribution.

**Second initialization.** Use the existing independent clean model (seed 29)
as a new pretrained generator, keeping the original 1024 centers. Generate its
native one-step inputs; train REC, DYN and the exact clean-center control for
4096 updates. Compare with the corresponding original-initialization runs.
For this family, the original pretrained model is the independent clean probe;
do not call the seed-29 ancestor an independent evaluator of its descendants.
This is a replication of the one-step-exposure question, not all method rankings.

The new runs retain fresh Adam at 1e-4, batch eight per branch, weights 0.5/0.5,
sampler seeds 17/1701 and the fixed clean data. Duration extensions restore Adam,
sampler and random states rather than restarting optimization. Save immutable
intermediate snapshots for the new runs. No automatic loss reweighting,
learning-rate search or validation checkpoint selection is introduced.

**Evaluation sequence.** First complete the target diagnostics on the two saved
pilot cases. Extend the original models and follow-up terminals to the original
twelve cases through 32 steps. Reuse qualified existing fields/solver labels;
new paths receive the same projection and temporal-refinement checks. Record
training-only target errors by generating length and time interval. Mixed-length
loss concentration is reported explicitly; no claim that mixing lengths itself
is necessary is made. Cross-trajectory tests assess transfer from frozen training
inputs; refreshing model-generated inputs remains the later multistep study.
Probe defects are measurements of predictive competence, not manifold distances.

**Execution and cost.** The selected personal workstation runs one GPU job at a
time. Six 4096-update extensions, one 8192-update clean control, and eight new
4096-update runs require 65536 updates in total (approximately 29 GPU-hours at
the preceding measured speed), plus preparation/evaluation. Keep per-run wall
limits and qualified solver labels. Stop a failed stage and record its reason;
never drop failed trajectories or silently alter the comparison.

## Historical completed-study index

The following routing and interpretations record the September 28 programme.
They are historical evidence, not the version 2 experiment queue. References to
paper roles describe that earlier programme. The
[experiment tracker](EXPERIMENT_TRACKER.md) owns exact results, contracts,
artifact identities, and execution limitations. Historical archive references
may be absent in the current worktree; do not restore owner deletions implicitly.

### Evidence Routing

| Paper role | Retained study or run |
| --- | --- |
| Clean competence and the recovery/composition discrepancy | `kf_fixed_data_20260916a`; independent confirmation below |
| Response range, direction, clean-error cost and eventual violation of the physical envelope | `kf_reached_response_20260917a`; retrospective analysis with sampled ground-truth qualification |
| Exposure versus temporal gradients | `kf_unroll_pair_20260918a`; matched detached/full K=4 |
| Recovery versus faithful dynamics targets | `kf_target_pair_20260918a`; identical displaced inputs, different targets and information costs |
| Next-state refinement and conditional generation | `kf_refiner_20260919c/evaluation`, `kf_acdm_20260920b/evaluation_v2`; fixed-PCNO adaptations with recorded acquisition and physical-fidelity limits |
| Response prior, amplitude cap and architecture restriction | `kf_remaining_coverage_20260921a`, `kf_spectral_control_20260921b`; selected components, not whole-family verdicts |
| Response-preserving design, forecasts and fitting-order repeats | `kf_bias_design_20260922b`, `kf_bias_forecast_20260922a`, `kf_bias_replication_20260922a`; three adaptation pairs from one parent |
| Independent population and frozen prospective controls | `kf_confirmation64_20260923c`; nine frozen maps and 64 evaluation-only trajectories |

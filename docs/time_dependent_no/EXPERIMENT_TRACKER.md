# Active Experiment Tracker

This is a dated evidence ledger. Historical "next", "active" and approval
language below records its original scope; it is not a live queue.
[PROJECT_PLAN.md](PROJECT_PLAN.md) and [HANDOFF.md](HANDOFF.md) own the current
scientific plan and checkpoint.

## October 5: Midpoint analyzed; interrupted four-step queue resumed

`KF_MULTISTEP_20261004A` completed the midpoint's 4,096 updates and original
twelve-case H32 evaluation before a plotting-helper signature mismatch stopped
the queue. No four-step condition had started. A separate source snapshot fixes
the call; the original failure and sources are retained. Regression verification
passes, actual deployed plotting succeeds, and `clean4` is training. Only the
four previously authorized unstarted conditions resume; the existing midpoint
checkpoint and evaluations are unchanged. The bounded collector has restarted.

The ignored run root's `ANALYSIS.md` and `paper_figures/index.html` contain the
partial analysis and seven figure pairs. Midpoint validation H8/H32 are
0.035414/0.410677, versus REC 0.039603/0.493623 and DYN 0.044302/0.444201.
Midpoint beats both endpoints on all four validation cases at H8, but beats DYN
on only two at H32. On identical displaced training inputs its REC/DYN errors
fall between the endpoints. Along autonomous trajectories, REC has smaller
local/probe defects despite larger rollout error. One training run per condition
does not establish a universal target ranking. Sequence conclusions are pending.

All 72 saved trajectory files reproduce the six completed models' metrics
exactly, and all 4,096 clean/displaced minibatch index pairs match across the
midpoint and endpoint runs. Two focused plot/statistic tests pass. Midpoint
common-input and training-target measurements are complete; geometry and sampled
responses remain scheduled after the sequence runs. The paper plan and manuscript
are unchanged. Resume/source/verification receipts remain under the ignored run
root, alongside the existing bounded collector.

## October 4: Midpoint and controlled four-step study launched

Run `KF_MULTISTEP_20261004A` implements the owner's approved first stage in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): the existing mixed-length REC/DYN
midpoint and four sequence controls separating clean inputs, frozen pretrained
inputs, refreshed detached inputs and full temporal differentiation. Five fresh
4,096-update continuations are queued on the selected workstation, after
disposable memory/timing checks. Each completed model receives the original
twelve-case, 32-step rollout/probe evaluation; shared-input targets, empirical
geometry and sampled response diagnostics follow. No new population is used.

The related CPU suite passed 69 tests; the final focused suite passed all 16
tests after independent review. The review's resource-memory, shared-probe,
input-provenance and empty-direction issues were fixed. The deployed snapshot
is isolated from prior runs; source hashes and launch receipts are under
`artifacts/time_dependent_no/kf_multistep_20261004a/`. The bounded collector
retrieves completed figures and measurements to `results/`, checks transfer
hashes and reproduces field metrics at completion. Training outcomes are pending.
The locked paper plan and nine checked manuscript/plan files are unchanged.

## October 4: Corrective follow-up completed and analyzed

Run `KF_CORRECTIVE_FOLLOWUP_20261002A` completed all 15 jobs, 65536 new updates,
25 model/checkpoint evaluations and 61 paired target requests. Each evaluation
uses the original eight training/four validation trajectories through step 32.
Elapsed queue time was 37.51 hours, including 30.60 hours in training stages with
checkpoint writing and fixed-input measurements. No outcome-based checkpoint
selection or additional experiment was introduced.

`ANALYSIS.md` under
`artifacts/time_dependent_no/kf_corrective_followup_20261002a/` contains the
numerical table, controlled comparisons, interpretation and proposed figure roles
for Section 4.2. The clean continuation lowers clean error pooled over inputs
0--31 and worsens H8/H32 rollout on all twelve cases. Against the exact clean control, one-step
REC/DYN reduce validation H32 error by 3.8%/3.3%, mixed-length REC/DYN by
73.3%/76.0%, and Gaussian REC/DYN by 71.9%/69.3%. Those H32 improvements have the
same sign on all twelve cases. Duration, early coverage, amplitude-profile and
initialization controls qualify these findings; their details belong in the
report, not a universal method ranking.

All 162 retrieved metadata/metric/log files match the final snapshot. Training
logs have finite losses, contiguous prescribed updates and identical paired
sample tapes. The independent full-field check reproduces metrics exactly from
300 saved trajectory files on the workstation. The paired signed identity closes
within 9.8e-15 and its triangle inequality passes. Nine focused CPU tests pass.
`analysis_verification.json` and `final_field_verification.json` record the checks.

The regenerated gallery has 29 PDF/PNG pairs. New principal figures explicitly
connect autonomous rollout to model/probe defects and contrast target learning
on training inputs with transfer to common rollout inputs. Three CSV tables and
captions accompany them; PDFs/captions are exported into the ignored manuscript
figure directory while preserving manual edits. The automatic metric collector
has finished. A separate full raw-field/checkpoint transfer is in progress;
`complete_retrieval.json` will record its verified completion. The deployed
training source, locked paper plan and manuscript text are unchanged.

## October 3: Paired target figures verified; broader evaluation running

The saved two-case pilot now has 20 predictor/input-trajectory combinations for
paired recovery and dynamics measurements. All original maps are queried on
the pretrained paths, and each REC/DYN pair is queried on both of its paths.
Own-path recovery and dynamics metrics reproduce the previous saved measurements
exactly; the signed squared-error identity has maximum residual 2.7e-15, and
the target-separation triangle inequality passes. Eleven PDF/PNG figure pairs
cover common inputs, cross-evaluation, the signed defect effect, and sampling
and loss concentration. They do not yet include the follow-up training results.

Open `paper_figures/index.html` under
`artifacts/time_dependent_no/kf_corrective_followup_20261002a/`. The
initial archive and all 30 members were hash-verified; the archive is retained
for provenance. A local rendering pass improves the logarithmic axes and labels
without changing scientific arrays or the frozen deployed source. PDFs and
captions are exported to the ignored manuscript figure directory; approved
manuscript text and the locked paper plan remain unchanged.

The pretrained and clean-continuation models have completed evaluation on the
original eight training and four validation trajectories. The other original
models are being evaluated before the training controls. A run-specific local
collector polls the queue, verifies retrieved data, renders completed comparisons
and exports figures while preserving manual changes to paper assets. Its status
is `sync_status.json` under the follow-up artifact root. Follow-up training
outcomes and the final full-field verification remain pending.

## October 2: Authorized corrective-study follow-up

The owner approved proceeding with the paired target experiments and addressing
the identified weaknesses, with figures for Section 4.2. The bounded protocol is
in EXPERIMENT_PLAN.md. Run `KF_CORRECTIVE_FOLLOWUP_20261002A` was launched on the
selected idle workstation after eight synthetic CPU tests passed. These cover
target accounting, early-time sampling, Gaussian amplitude control, exact Adam
and sampler resumption, all three training target branches, and population RMS.
The Windows sandbox prevented pytest temporary-directory access; the authorized
run outside the sandbox passed all eight tests. Import and source-hash checks
passed on the execution resource. Launch is verified, not experiment completion.

The first stage reuses existing qualified pilot fields and solver labels for
paired recovery/dynamics measurements. Later stages expand to the original
eight training/four validation trajectories and execute the training controls.
Existing checkpoints/results and the locked paper plan are unchanged.
Local source/check receipts are under
`artifacts/time_dependent_no/kf_corrective_followup_20261002a/`.

Additional reductions motivating the controls: eight-step inputs account for
97.4011% of the original mixed squared displacement magnitude and 80.1665% /
85.8784% of the final mixed REC/DYN displaced-target MSE, despite equal counts
of generating lengths. Mixed/Gaussian median input magnitudes are 0.00254703 /
0.10724009 training scales, while both pooled RMS values are about 0.10724.
The 51 original centers before step 32 contribute 53.4882% of mixed displacement
energy; scarce sample count does not imply negligible loss influence. The
original run still excludes displaced centers at steps 1--7. These reductions
use the authenticated preparation manifest and training-investigation arrays;
they are not gradient-contribution measurements.

## October 2: Similar-Curve Training Investigation

The owner requested investigating the weak one-step-exposure effect, similar
REC/DYN error norms, probe coverage and the enstrophy diagnostic. The completed
investigation evaluates all 1,024 stored training pairs, queries all eight maps
on the same two saved pretrained rollout paths, and rolls the independently
trained clean probe forward on those two initial conditions. It uses frozen
checkpoints and existing solver labels; no new training or population is added.

At validation input 4, finite response gains on identical inputs are
4.418/4.539/4.185/4.129 for pretrained, clean continuation, one-step REC and
one-step DYN, versus 1.023 for the true flow map. Their similar curves accompany
similar excess amplification and radial error spectra. On their stored training
inputs, one-step REC/DYN reduce the respective target RMS by about 22%/16%, but
their remaining errors exceed the pooled REC/DYN target separation. Mixed
REC/DYN have distinct target-specific errors and responses. Exact values,
definitions and interpretation are in the existing comparison's `ANALYSIS.md`.

The corrective centers omit inputs 0--7; only 51/1,024 precede step 32. The clean
branch includes the onset. One-step inputs preserve their small native scale;
state and direction coverage also matter because excess response is observed
within their maximum amplitude. The older clean continuation is not an exact
zero-displacement control for the selected corrective centers. These are design
limitations, not a detected implementation deviation from the approved recipe.

The independent clean model has validation H32 error 2.6973 versus 1.8143 for
the original pretrained model; its gain on the common input above is 4.948.
Thus excessive response appears beyond the original checkpoint family, while
identical error envelopes and general ineffectiveness of one-step training are
not established. The probe was already evaluated on every REC/DYN rollout;
the figure title and legend now explicitly identify its input trajectories.
Enstrophy is one half of mean squared vorticity, compared at the same time to
ground truth; it measures amplitude, not spatial or trajectory fidelity.

Run-specific reproduction scripts `investigate_training.py`,
`check_probe_rollout.py` and `plot_training_investigation.py` live under the
existing `analysis_20261002a/` artifact root, with invocation instructions.
They preserve this bounded inference audit and its figures. Results and hashes
are in `training_investigation/`; `index.html` contains the three new figures.
Checkpoint/source/metric hashes and independent-probe calibration checks pass;
the probe calibration and saved rollout metrics reproduce exactly. Batched
pretrained inference differs from saved single-input inference by at most
4.48e-6 RMS / s. The focused plotting test passes after running outside the
Windows sandbox's temporary-directory restriction. The manuscript and locked
paper plan are unchanged. Follow-up designs require owner review.

## October 2: Six-Method Comparison Completed and Retrieved

The owner requested completing the remaining approved experiments before
interpreting the full comparison. Both Gaussian models have now joined the
four native-error models at their fixed 4,096-update terminals. Evaluation
`evaluation_20261002a` completed both original baselines and all six corrective
models on the same first training and validation cases through step 32.
The independently initialized clean probe completed 49,152 updates and supplies
the diagnostic predictions. No training protocol or evaluation population changed.

The complete archive, checkpoints, logs and sixteen trajectory field files are
retrieved under the existing comparison's `analysis_20261002a/results/` directory.
The archive SHA-256 is
`d756c1ee8ad072f1760f7319c95b5fd548c12a582aa59a559e027fd5fdca083a`.
Independent local verification checked the archive and 40 individual artifact
hashes, all six training/evaluation checkpoint bindings, fixed update counts,
finite training logs and identical clean/displaced sample indices across runs.
Recomputing rollout and defect metrics from the saved fields reproduced them
exactly. The signed error identity closed within 2.3e-15; the largest sampled
solver refinement discrepancy remained 1.11e-4 RMS / s. All sixteen rows of
the summary CSV reproduce the saved metrics.

All six PDF/PNG figures were regenerated and visually checked, including the
Gaussian columns. Open `results/evaluation/figures/index.html` for rollout error,
per-step model and probe defects, signed alignment, enstrophy and input exposure.
`evaluation_verification.json` contains the final receipt; `retrieval_status.json`
records completion of the bounded retrieval helper. `ANALYSIS.md` labels its
earlier native-only interpretation as a historical snapshot. Interpretation of
the complete batch is deferred to owner discussion. The manuscript and locked
paper plan are unchanged; broader evaluation remains pending review.

## October 2: Native Corrective Rollouts and Independent Clean Probe

The owner authorized retrieving results, analyzing them and generating rollouts
with probe diagnostics. The probe completed its fixed 49,152 updates; the four
native-error models completed 4,096 updates each. Their checkpoint hashes and
training identities were verified, and all four logs have identical clean and
displaced sample indices. Gaussian recovery is training; Gaussian dynamics is
queued. The evaluation watches for their fixed terminals, without changing the
training recipe or selecting intermediate checkpoints.

The completed evaluation covers both original baselines and the four native
models on the same first training and first validation cases through step 32
(time 1.6). All numbers below are pooled spatial/time RMS divided by the fixed
training normalization scale, without percentage conversion.

| Model | Training H32 | Validation H32 | Probe defect on validation rollout |
|---|---:|---:|---:|
| Pretrained | 1.6393 | 1.8143 | 1.4166 |
| Clean continuation | 1.6721 | 1.8320 | 1.4375 |
| Native one-step recovery | 1.6370 | 1.7988 | 1.4290 |
| Native one-step dynamics | 1.6268 | 1.8121 | 1.4642 |
| Native mixed recovery | 0.6327 | 0.5624 | 0.2758 |
| Native mixed dynamics | 0.3707 | 0.4209 | 0.3657 |

Mixed exposure improves rollout substantially despite larger clean errors.
One-step-only exposure gives modest H8 gains and little H32 benefit. Mixed
recovery has smaller own/probe defects than mixed dynamics on both cases, yet
larger H32 trajectory errors. Defects are evaluated on each model's own path;
this is not a comparison on identical inputs. Probe error on ground-truth
inputs is 0.0072/0.0182 for the training/validation cases. The reached-state gap
measures loss of clean-trained prediction competence, not exact manifold
distance. Mixed-model terminal energy/enstrophy ratios are within about 2% of
ground truth; substantial trajectory errors nevertheless remain.

The pending Gaussian pair uses pooled-RMS calibration. The native mixed
distribution is strongly skewed: median displacement 0.00255 versus pooled
RMS 0.10724, with the largest 11/1,024 rows contributing 55.6% of squared
displacement. Gaussian displacements concentrate near 0.10724. The resulting
comparison concerns different magnitude distributions as well as directions.
No calibration or training condition was changed after observing this.

Artifacts are under
`artifacts/time_dependent_no/kf_displacement_comparison_20261001a/analysis_20261002a/`:
`ANALYSIS.md` records interpretation and limits;
`results/evaluation/figures/index.html` displays rollout, own-defect, probe,
signed-alignment, enstrophy and exposure figures; per-case arrays retain reached
states and solver/probe next states. The bounded `retrieve_when_complete.py`
retrieves final Gaussian results, rerenders local figures and exits. Its
`retrieval_status.json` is the live status; the analysis report records the
native-model snapshot only.

New maintained code comprises `evaluate_kolmogorov_corrective_rollouts.py`,
`plot_kolmogorov_corrective_rollouts.py` and their focused evaluation test file.
These provide the common trajectory/probe measurements and reproducible figures
for the approved comparison. Seventeen CPU tests passed before deployment;
five passed again after a shared-axis clipping fix. Baseline rollout and
own-defect arrays reproduce Experiment 1 exactly. The largest sampled solver
refinement difference is 1.11e-4 RMS / s, projection change below 2.0e-7, and
signed error-identity residual below 2.3e-15. Numerical, source and artifact
receipts are in the analysis directory. The manuscript and locked plan are
unchanged. Broader case evaluation still requires owner review.

The owner's follow-up requested an implementation audit of the nearly overlapping
one-step-exposure curves. The run-specific `audit_one_step_implementation.py`
under the analysis directory replays saved input/target construction, initial
optimizer updates and full autonomous rollouts independently of the evaluator's
loading/rollout helpers. All 1,024 center/REC target indices agree; seven selected
rows reproduce native inputs and solver labels exactly; both one-step models'
first losses/gradients reproduce their logs. All four native terminals show
4,096 optimizer steps and changed weights, and reproduce their two full saved
rollouts exactly. Thirty-seven focused CPU tests pass. No implementation error
was found in these checks. `audit_one_step_implementation.json` records the audit.
The refreshed per-step defect figure includes the pretrained baseline. REC's
validation defect falls from 0.03147 to 0.01877 at input 4, but is 0.63658 versus
0.64053 at input 8. Similar error norms conceal distinct predicted fields.
The registered displaced-input centers begin at step 8; only 51/1,024 are before
step 32, although the clean branch covers the full interval. This coverage limit
and the small one-step displacement range qualify the scientific interpretation.
No sampling or training settings were changed during the audit.

## October 1: Six Corrective-Training Runs Queued

The owner authorized preparing this comparison and queuing it after the
independent clean probe. The owner selected equal sampling of native rollout
lengths 1, 2, 4 and 8 for the mixed condition, alongside one-step-only errors
and Gaussian errors. Each condition has a recovery/dynamics target pair.
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) records the complete fixed protocol.

Run `kf_displacement_comparison_20261001a` uses an isolated source copy on the
selected workstation. Launch verification confirmed a live queue process in
`waiting_for_probe` state, zero completed corrective training runs, and the
probe still running. No new comparison results are available at this checkpoint.
The queue requires successful probe completion and a verified terminal, then
prepares the shared inputs/labels and trains the six models serially. It opens
only the existing authenticated training packet; evaluation is a later stage.

The new runner and its focused test file are the only added maintained code.
The old generator explicitly rescales native directions and therefore cannot
implement the owner's requested exposure conditions unchanged. Model/data
loading, solver dynamics and the two-branch optimizer update are reused.
Twenty focused CPU tests passed, including native magnitudes and time indexing,
Gaussian ensemble calibration, solver/recovery label semantics, identical
sample tapes and initialization across all six runs, checkpoint reload,
corrupted-input rejection and failed-probe queue blocking. Windows sandbox
permissions blocked pytest's temporary directories; the passing verification
ran outside the sandbox. Ruff was unavailable in the local Python environment;
the Python imports and deployed command-line import check passed.

Seventeen frozen source files were transferred and rehashed successfully.
The source archive SHA-256 is
`9d15edcf96a59ecf58b075ac974edfd6ea8246df9d800c8015533e1d20a83264`.
Source/protocol, launch and queue receipts are retained under
`artifacts/time_dependent_no/kf_displacement_comparison_20261001a/`.
The manuscript, locked paper plan, existing models and running probe are unchanged.

## October 1: Rollout and One-Step Defect Animations

The owner requested spatial animations of the same two pilot cases. The existing
plotter now supports `--animate --common-input-dir <prediction-fields> --mp4`
with its usual `--input` and `--output` arguments. Outputs are in
`artifacts/time_dependent_no/kf_local_defects_20261001a/animations`: a standalone
`trajectory_viewer.html`, plus rollout and own-defect MP4s for each case.
The viewer separates accumulated rollout errors, each model's own-input defects,
and defects at identical ground-truth or reached inputs. Signed fields use the
fixed training scale without a percentage multiplier. A/B share color limits;
fixed limits are the default, with an explicitly labeled per-frame option.

The run-specific `export_prediction_fields.py` retains full common-input
predictions in `prediction_fields_20261001a/` so future rendering needs no model
calls. The 51-second frozen-model export reproduces the preceding similarity
audit's error norms exactly. It generates no new rollouts or solver labels.
Arrays retain their original `(x,y)` order; rendering places y upwards.

Twelve focused CPU tests passed. Rendered own-defect and rollout RMS values
match saved metrics within 4.44e-16. Source and output hashes match
`animation_manifest.json`; videos contain all 33 state frames or 32 transitions
at four frames per second. Headless-browser checks passed for case selection,
time scrubbing, common-input selection, color scaling and playback, without
JavaScript errors. Independent review checked definitions, indices and orientation.
The interactive file is about 95 MB; the four MP4s are about 1.8--2.4 MB each.
The locked paper plan and manuscript remain unchanged.

## October 1: Direct Comparison of Prediction Fields

The owner requested a direct check of the nearly overlapping A/B error curves.
The audit re-evaluated both frozen models on identical saved ground-truth,
pretrained-rollout and continuation-rollout inputs for the same two pilot cases
and all 32 input steps. No training, solver calls or new rollouts were performed.
The run-specific script is `audit_similarity.py` under
`artifacts/time_dependent_no/kf_local_defects_20261001a`; its `--run-root`,
`--pilot`, `--output` and `--device` arguments reproduce the audit. Results,
metrics and `prediction_similarity.pdf/png` are in `similarity_20261001a/`.

There is no accidental checkpoint, model-object or tensor-storage reuse. All 34
parameter tensors changed; the parameter difference has L2 norm 0.019701 times
the pretrained parameter norm. At reached inputs 7--31 on both saved rollout
paths, the prediction difference is 0.0225--0.0421 times the mean defect norm
for the training case and 0.0209--0.0328 for validation. Error-field cosine is
at least 0.999456 and 0.999582, respectively. Early small error fields are less
aligned; the full time curves are retained rather than asserting uniform error
alignment from onset.

Subtracting the common input also confirms similar learned increments. Across
inputs 1--31 on both rollout paths, their relative difference is at most 0.0250
(training) and 0.0204 (validation), normalized by the mean increment norm.
Finite responses `Psi(x)-Psi(u_n)` also agree closely: median relative differences
are 0.0214 and 0.0206. These are sampled finite responses, not Jacobians or
uniform stability bounds. The shared checkpoint and training history are a
plausible explanation, not a demonstrated causal attribution. Clean supervision
failing to determine off-data behavior does not require two related models to
disagree there. The independent probe will help assess whether similarity extends
to a separate initialization.

The synthetic self-test distinguishes identical error magnitudes with opposite
error fields. Independent source reviews found no indexing or measurement issue.
The audit completed in 48 seconds and passed replay checks: maximum error-norm
change is 3.45e-7 and maximum own-output RMS change is 4.47e-6, both divided by
the fixed training scale. Retrieved artifact hashes match the receipt. Source
SHA256 is `fdfdfe07f6136da5ecdfd80cc4788951d5e73aa3bb25385930d29c62a204e794`.
The locked paper plan and manuscript remain unchanged.

## October 1: Figures Clarified and Independent Clean Probe Started

The owner requested clearer curves and authorized training an independent clean
probe for reuse. The existing pilot data are unchanged. Primary figures are now
`trajectory_one_step_errors` (one fixed predictor per column, three input
trajectories per panel) and `trajectory_rollout_errors` (accumulated error
separately). One-step errors use the input-state time, with the next-state
target explained. Earlier figures remain as prior outputs.

Run `kf_clean_probe_20261001a` started at 06:24 UTC on the selected workstation.
Fresh initialization/sampling seed 29 replaces seed 17; all 32 training
trajectories, normalization, PCNO architecture, clean objective, Adam settings
and 49,152-update learning-rate schedule are retained. No pretrained weights,
validation data or rollout states are used. Terminal selection is fixed.

Thirteen CPU tests passed, including interrupted-training recovery. The resource
phase completed 16 finite updates, retained no weights, and measured
0.794 seconds/update and 6.57 GiB peak allocated GPU memory. Its 10.84-hour
estimate exceeded the historical five-hour allowance, so only the wall-time
limit was changed to 14 hours before launch. The resource and fit source
capsules are both retained. Launch verification observed 37 finite updates,
a recoverable checkpoint, and the same initialization fingerprint as the resource
phase. This records training in progress, not a completed or calibrated probe.
Receipts are under `artifacts/time_dependent_no/kf_clean_probe_20261001a`.

## October 1: First Version 2 Pilot Completed

`kf_local_defects_20261001a` evaluates the existing pretrained/clean-continuation
pair on the first original training trajectory (2026090611) and validation
trajectory (2026090621), through step 32, without training. Selection preceded
the new diagnostics. Runtime was 303 seconds on the selected workstation.
This is the pilot review stage; the remaining ten cases and independent clean
probe have not been evaluated. No qualifying independent probe was found in
the selectively inspected saved checkpoints.

The main artifacts are three PDF/PNG figures in
`artifacts/time_dependent_no/kf_local_defects_20261001a/figures`: clean versus
rollout errors, common-input defects, and percentage changes after continuation.
All absolute errors use one fixed training scale, rather than state-relative
normalization; local errors are plotted at their output time. Saved per-case
states and numerical next states, time-resolved metrics, and `analysis.json`
are under `pilot/`. `analyze_pilot.py` reproduces the reductions and replay checks.

On the validation trajectory, continuation improves clean one-step error at all
32 inputs, but increases true-flow defects at 29/32 common inputs on either
rollout path. Its accumulated error at step 32 is larger. The training case has
improved clean error at 26/32 inputs; its first-eight-input clean RMS is slightly
worse. Own-rollout local defects are not uniformly worse: the full-horizon mean
square is slightly lower for continuation on the training case. Common-input
comparisons therefore matter, and defect magnitude alone does not order rollout
quality. Both models overpredict kinetic energy and enstrophy at step 32 on
both cases; their values are retained in `analysis.json`.

The 23 local CPU tests passed. All available old snapshots through step 32
replay exactly for both maps and both cases. The signed error identity closes
within 1.78e-15 in normalized mean-square units. Solver restarts explicitly
project FP32 inputs into the FP64 canonical state space; maximum projection
RMS/train-scale is 1.96e-7, and sampled clean restart/archive disagreement is
at most 7.18e-8. Halving both maximum substep and CFL at inputs 0, 7, 15 and 31
gives maximum reached-state label change 1.10e-4 in the same units. At sampled
inputs 7, 15 and 31, twice that case/step-specific change is at most 3.28% of
the observed common-input error-norm gap. These rankings survive that temporal
refinement; this is not a bound against the continuous PDE or an unsampled-state
guarantee. The workstation environment lacked pytest; local tests and the
workstation evaluator import/replay checks are distinct validations.

Independent Codex reviews checked artifact identity, measurement algebra,
plot aggregation and scientific interpretation. The pilot supports loss of
accuracy at reached states, but does not yet establish manifold departure or
causal attribution of the entire rollout difference. The paper plan and
manuscript remain unchanged. Await owner review before broader evaluation.

## September 23: Independent 64-Trajectory Confirmation Completed

`kf_confirmation64_20260923c` completed in 4.69 workstation hours with no fitting
or protected-population access. All 64 reference identities and nine maps match
the frozen scope below. Forecasts, controls, tolerances and the trajectory-level
uncertainty rule precede all six adapted-map autonomous evaluations.

An independent saved-array analysis verifies 35 source files, 40 frozen inputs,
384 forecast packets and 7,854 rollout snapshots. It recomputes field error,
mean, kinetic energy and enstrophy from saved fields, reduces all horizons from
per-step sums, and reproduces the fixed forecast controls and joint trajectory
bootstrap. No model or solver call is used by this reduction. The six sampled
reference restarts pass exact base replay; maximum temporal/spatial relative
differences are 7.51e-9/4.00e-5 and discarded fine-grid content is 5.24e-5.
These are sampled numerical checks, not a uniform H128 reference certificate.

All errors below are percent relative L2. Clean inputs are 0--7. An incomplete
H128 population has no unconditional pooled error; the final column retains
every completion/guard outcome.

| Map | Clean | H8 | H32 | H128 complete |
| --- | ---: | ---: | ---: | ---: |
| Clean continuation | 0.0913 | 6.3310 | 75.73 | 0/64 |
| Online recovery | 0.0957 | 0.6276 | 59.50 | 0/64 |
| Bank DYN | 0.3752 | 1.4381 | 20.23 | 64/64 |
| Seed 17 clean-only | 0.1297 | 0.7841 | 42.57 | 37/64 |
| Seed 17 preserving | 0.3379 | 1.2378 | 20.10 | 62/64 |
| Seed 18 clean-only | 0.1168 | 0.6071 | 40.68 | 42/64 |
| Seed 18 preserving | 0.7870 | 3.8207 | 25.19 | 64/64 |
| Seed 19 clean-only | 0.3286 | 1.7435 | 43.05 | 34/64 |
| Seed 19 preserving | 0.3235 | 1.1378 | 20.72 | 62/64 |

Recovery beats continuation at H8 and H32 on all 64 paths despite slightly worse
clean error. Corresponding onset training errors are 0.0823/0.0848%; the later
clean generalization gap remains, so this eligibility claim stays at onset.
Response preservation beats its matched clean-only fit at H32 on all 64 paths
in each pair. Per-fit pooled error ratios are 0.4722/0.6191/0.4813. The predeclared
mean log ratio averages fitting pairs within trajectory, then trajectories:
-0.662706, with 95% bootstrap interval [-0.695270, -0.630637]. Exponentiating gives
0.515454 [0.498940, 0.532253]. The bootstrap resamples 64 trajectory identities
jointly across maps (10,000 draws, analysis seed 2026092317), conditional on these
three fits and their one parent; it does not treat 192 cells as independent.

Qualified preserving H8 predictions are 1.2382/3.8174/1.1406%, versus observed
1.2378/3.8207/1.1378%. Relative point discrepancies are 0.028/0.086/0.245%, and
all 64 paths per map pass the local checks and fall within their empirical ranges.
The ranges are not confidence intervals or uniform stability bounds. Identity
transport gives 30.80/7.07/42.16% discrepancies; fixed DYN-calibrated clean error
gives 0.83/25.17/3.29%. Scalar calibration already predicts the preserving fits'
change signs; full response adds conditional precision. These controls are
prospective here, unlike their first development-population comparison.

All three pooled clean-only forecasts fail qualification; their point errors
are 272/393/70%. Their broad ranges contain outcomes without validating those
predictions. The full parent-relative screen correctly recommends preserving
seeds 17/19 and predicts seed-18 harm. It abstains on the misleading clean-only
seed 19, but also on beneficial clean-only seeds 17/18. There is no universally
successful method selector or prediction of what fitting will learn.

At H128 DYN/preserving seed 18 complete all paths with 93.24/89.54% field error
and 1558.9/167.1% energy NRMSE. Preserving seed 17 guards on identities ending
2338/2342 at steps 119/110; seed 19 guards on 2329/2338 at steps 120/119.
The old four-development-path universal loss against DYN does not transfer.
The stable conclusion is poor sustained physical/path fidelity, not that the
parent wins every long-horizon comparison.

Reproduction in the retained run directory: `python analyze_confirmation.py`
reduces the frozen arrays; `python plot_confirmation.py` produces the paper panel
from the bound analysis. These two ignored artifact scripts exist to reproduce
this result and figure, not as new experiment entry points. Analysis SHA-256:
`ae4b359dd0655233c71348a675740d31eb1e73e5ad0a6566de0a0a7bfdd3420b`.
`verification.json` and `figure_manifest.json` bind inputs and outputs. Sections
4--5 and Appendix B now distinguish this confirmation from representative-method
development results. Next finish supporting-case methods and manuscript
consolidation; no new fit or broader population is selected.

## September 23: Fixed-Model Confirmation Expanded To 64 Independent Trajectories

The owner selected 64 fresh evaluation-only ID trajectories while keeping all
32 training trajectories and nine fitted maps fixed. `kf_confirmation64_20260923c`
is launched; the latest check finds reference generation running, with no model
outcome yet available. All 64 local and remote CPU tests passed, and all nine
checkpoint loads passed before generation. Seeds 2026092301--2026092364 were
frozen before generation and checked against the 40 recorded prior population
identities. Historical protected populations remain closed.

The scope SHA-256 is
`0f2780e08b4bca3f90061e67d72f072ee5cc43415e748dd551c26c36ba224901`.
The isolated packet binds 35 source files, checkpoints and the unchanged
training manifest. Its queue freezes independently reduced forecasts and both
simpler controls before adapted autonomous outcomes. Whole trajectories are
the uncertainty units; all three fitting pairs and guarded outcomes remain in
the accounting. Experiment Plan Block 4 owns the scientific specification.

The two preceding packets stopped during preflight: an omitted test file,
then the seed-17 pair's older fit-source validator. No reference population or
model outcome was generated by either. The corrected loader retains the
original validator and checks that inference sources are unchanged; no fit or
checkpoint was modified. Receipts remain in the ignored run directories.

## September 23: Simpler Forecast Controls Bound The Added Predictive Claim

This is an additional **retrospective** reduction of the saved forecasts in
`kf_bias_forecast_20260922a` and `kf_bias_replication_20260922a`. No model,
solver, new trajectory or protected-population access is involved. All six
fits and both existing forecast populations are retained.

The identity-response control keeps `G(v_n)-v_(n+1)` but replaces the candidate
Jacobian by identity. The saved identity displacement matches independent
FP32 cumulative forcing exactly in all 72 seed-path packets. The scalar control
multiplies clean inputs-0--7 error by 3.6330056115889438, calibrated from the
**original DYN** development H8/clean ratio, without using candidate outcomes.
Neither control inherits the full calculation's local qualification.

Entries below are percent relative discrepancy of predicted H8 error, not
rollout error. Full-response clean-only predictions remain unqualified.

| Fit seed | Arm | Full response | Identity response | Calibrated clean |
| --- | --- | ---: | ---: | ---: |
| 17 | Clean-only | 13.681 | 85.056 | 32.927 |
| 17 | Preserving | 0.009 | 25.361 | 0.025 |
| 18 | Clean-only | 19.594 | 157.767 | 14.382 |
| 18 | Preserving | 0.069 | 6.368 | 25.448 |
| 19 | Clean-only | 1.947 | 18.501 | 29.636 |
| 19 | Preserving | 0.270 | 41.962 | 8.460 |

The full forecast improves numerical precision across all three qualified
preserving fits. Scalar clean calibration nevertheless already recovers each
fit's parent-relative direction and is nearly exact for seed 17. The result
supports conditional quantitative information from measured response, not
uniquely correct intervention selection. Unqualified forecasts need not beat
the simpler controls. Section 5 states this distinction and Appendix B gives
the six-map table. These controls are specified before the proposed fresh-ID
confirmation in Experiment Plan Block 4; no confirmation is claimed here.

Reproduction: run `forecast_controls.py` in the replication artifact directory
from the repository root. Its SHA-256 is
`0a3cdbc037824844f2906c1747f9d55ec86f3a465cf93960fd41de33369b22bc`;
`forecast_controls.json` is
`a8770a5771bfb69396a54b8fd593a86b11aa4ff987abcf4b930c2667ece7dcb3`.
The reduction checks each NPZ hash, full-forecast replay, common-population
clean metrics and identity recurrence. Training forecast clean metrics use
eight paths, distinct from the 32-path teacher summary; development uses the
same four in both. Original forecast freezes, summary, figures and outcomes
are unchanged.

## September 23: Design Replication Separates Robust Response Protection From Variable Clean Forcing

`kf_bias_replication_20260922a` is complete and retrieved. Four new fixed
4,096-update fits (paired seeds 18/19), their nonrecurrent assays and unchanged
H8 forecasts finished before autonomous evaluation. Independent saved-array
reduction on the workstation froze the numerical predictions; the unchanged
H8/H32/H128 evaluator then ran all four maps. Local retrieval independently
rechecked that freeze and reduced the assays, rollout errors and physical fields.
No solver labels, trajectories, loss weights, model sources or protected roles
were added. No worker from the study remains active.

The table reports pooled errors on the same four open development paths.
Seed 17 is the previously completed pair, retained without alteration.

| Fitting seed | Map | Clean onset (%) | H8 predicted (%) | H8 observed (%) | H32 (%) | H128 (%) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| -- | Original DYN | 0.37838 | known | 1.37466 | 19.93781 | 75.65897 |
| 17 | Clean-only | 0.12908 | 0.79482, unqualified | 0.69917 | 42.45828 | incomplete: 1/4 |
| 17 | Preserving | 0.34264 | 1.24441, qualified | 1.24452 | 19.31814 | 98.97956 |
| 18 | Clean-only | 0.11692 | 0.59336, unqualified | 0.49615 | 40.54001 | incomplete: 2/4 |
| 18 | Preserving | 0.78918 | 3.84308, qualified | 3.84575 | 25.09448 | 85.74086 |
| 19 | Clean-only | 0.32816 | 1.72732, unqualified | 1.69433 | 43.01698 | incomplete: 0/4 |
| 19 | Preserving | 0.32107 | 1.07836, qualified | 1.07546 | 20.65980 | 94.61733 |

Response retention generalizes to unused probe times/draws in all three fits:
development signed-vector change is 7.93--10.06% versus 179.44--452.35% for
clean-only. Preservation wins H32 on every one of the twelve matched paths in
each pair. Development H32 is 21.6908 +/- 3.0230% versus 42.0051 +/- 1.2992%
(mean/sample SD across fitting seeds). The path sets are reused; these are not
36 independent population samples. Clean-only wins H8 in seeds 17/18, while
preservation wins in seed 19. Seed 18's preserving clean error is 0.77280% on
training and 0.78918% on development: the deterioration is not a large ordinary
generalization gap. Response retention does not ensure clean-bias reduction.

The preserving forecasts pass unchanged checks on training and development for
all three seeds. Development relative point discrepancies are 0.0091/0.0694/
0.2696%; all outcomes lie within the frozen empirical ranges. Seed 18 predicts
harm relative to DYN, while seed 19's upper range passes the predeclared
0.95*DYN H8 screen. This is a conditional fixed-map forecast, not prediction
of fitting or a selected best seed. All clean-only forecasts fail their local
checks (development remainder ratios 0.2771/0.2642/0.2028); their subsequent
outcome comparisons remain exploratory. No qualified ranking versus clean-only
is established. Empirical tolerances are not confidence intervals or bounds.

The stronger duration claim does not repeat: preserving development 10% crossings
are steps 17, 12--13 and 16--19, versus 11--13 for every clean-only fit.
All preserving fits complete H128 and lose to DYN on every path. Development
H128 is 93.1126 +/- 6.7464%, with energy NRMSE 34.09--99.94% versus DYN's
16.60%. Clean-only reaches the amplitude guard in six training and nine
development seed-path evaluations; no surviving-path H128 average is used.
Thus the repeated result is intermediate composition protected against further
clean-only adaptation, with variable onset forcing and no sustained gain over DYN.

Verification: all 164 fit/assay/forecast files and 64 outcome files pass transport
and internal hashes. Four terminal checkpoint identities, finite parameters,
exact 4,096-update tapes, matched within-pair samples, objective arithmetic,
source closure and bank/model/input bindings pass. All clean pairs have two
visits per stream; each response input has 32 visits. Saved signed vectors,
clean bands, recurrence closure, nonlinear qualification, rollout snapshots,
physical scalars and horizon/censor accounting independently reduce. The same
34 scientific sources retain the 62 passing local/remote CPU tests; no additional
test run or scientific implementation change was needed during this closeout.

Numerical freeze SHA-256:
`ed50ec3486d8b3fdf493bfd74746907d869d84876617acefcd6f450dd345e000`.
Fit/assay/forecast transport SHA-256:
`372e109637f935bfbcfb6d28130d45b9045f5275d28309cea62020f24b092592`.
Outcome transport SHA-256:
`705f887fab2a5ce05f3234ad100c68e7b09d425b4afa145a5e92fb1fe4adcfb7`.
The ignored run's `analyze_replication.py`, per-seed reductions,
`replication_analysis.json` and `closeout_verification.json` retain reproduction
and exact bindings. `evaluate_repeats.py` invokes only the fixed outcome evaluator;
`paper_results.py` reproduces the consolidated table inputs and replacement figure.
These are run-local evidence producers; no maintained model/evaluator was changed.

Section 5 now reports all three fits in one table and one forecast/outcome figure;
the previously fixed seed-17 field example remains. Abstract, introduction,
conclusion and Appendix B reflect both the replicated benefit and failed onset/
duration expectations. The 45-page PDF has no LaTeX/box warnings or unresolved
references and embeds all fonts; changed pages 1,22,24,42 were visually checked.
PDF SHA-256: `e70390813235ac4f6d25bec367461c989c9c087af9967811329b48230dd0d1fd`.
Prior manuscript/PDFs are preserved. Next agree the minimal untouched-ID
confirmation scope; do not tune the completed recipe or reopen method coverage.

## September 22: Paired Design Replication Launched; Paper Evidence Completed

The owner's continuation selects `kf_bias_replication_20260922a`: two additional
paired fitting seeds (18/19), conditional on the same bank-DYN checkpoint and
fixed training/response inputs. The experiment plan records the recipe and
qualitative expectations before these fits. There is no new data, solver label,
loss-weight selection, architecture or protected-population access. The worker
performs four fixed 4,096-update fits, existing nonrecurrent assays and unchanged
H8 forecasts, then stops before new autonomous outcomes. Forecast reduction and
numerical freeze precede the subsequent authorized outcome evaluation.

Launch occurred September 22 at 22:06 China time. At 22:19, seed-18 clean-only
fitting was running at update 403 with finite loss and gradients. Both 16-update
resource checks passed: teacher replay error is exactly zero, peak allocations
are 6.602/12.218 GiB, and warmed fit estimates are 1.793/3.582 hours for clean-only/
preserving. Estimated completion including four assays/forecasts is around
September 23 10:00 China time. This is running work, not replicated evidence.
The local and remote focused suites both pass 62 checks. The new seed test first
failed on the absent seed argument, then verified changed sampling order,
within-pair matching, common parent and checkpoint/identity round trips. A
Windows sandbox temporary-directory failure is setup evidence, not a model result.

The 34-file source packet changes only the response-preserving fitter and its
test relative to the first forecast source packet. Default seed-17 semantics,
objective, model/evaluator and numerical forecast stay fixed. Source archive
SHA-256: `8b0a1602fc671e3287083284436cb4a0ce831b85e0bd6b8b11dfc0d8eeab6ed7`.
Plan SHA-256: `78105ec2975f3d25d17ce82eeb533b0cca7c791e4d9ee8e60c396385bb21cb11`.
The historical probe-bank compatibility check authenticates its original source
and unchanged probe/metric definitions; it does not modify the bank. Exact
launch/resource/progress receipts remain in the ignored run directory.

Separately, `paper_evidence.py` reproduces two manuscript additions using only
completed saved fields. All 243 reference-query packets and 64 states across
eight restart comparisons rehash and independently reduce. Maximum temporal,
restricted spatial and discarded-state relative L2 discrepancies are
1.49421e-8, 1.27300e-4 and 1.51465e-4. Eight-step endpoint discrepancy is at most
2.11513e-5. The six-phase initial law reconstructs to 4.43e-16; reduction
differences from recorded scalars are at most 1.41e-16. These sampled checks
do not establish full-H128 continuum convergence. No new solver/model calls.

The new common-scale field display uses the first stored development trajectory
at steps 8/32/128, with no outcome-based sorting or color clipping. It retains
clean-only's missing H128 field and distinguishes instantaneous from pooled
errors. Section 5 includes the figure; Appendix B now specifies the initial
law, reference solver/checks, parent selection and hardware. Abstract/introduction
status prose now reflects the achieved conditional quantitative prediction.
Fitting/population confirmation remains red. The 44-page PDF builds without
LaTeX/box warnings or unresolved references and embeds all fonts; changed pages
were visually checked. Its SHA-256 is
`7940cf66197662e0419ede5ad18701af8269607b14c3ac0f3b6f6f7010c3b405`.
The run's paper-build receipts preserve earlier PDFs and bind actual inputs.

## September 22: Quantitative H8 Prediction And Horizon-Dependent Design Outcome

The bounded `kf_bias_forecast_20260922a` study is complete. It reuses both
unchanged terminals from `kf_bias_design_20260922b`, the saved bank-DYN path,
eight fixed training paths and the four already-open development paths. There
are no new fits, labels, population access or deployed operations. The new
entry point applies the existing forced-tangent routine around the fixed DYN
states using the full candidate Jacobian, retaining map-change forcing and
separate clean-reference forcing. Nonlinear signed probes only check validity.
Affine/nonlinear and input-binding tests plus inherited checks give 61 passes
locally and remotely. All 31 prior fit sources remain unchanged.

Forecasts were independently recomputed from saved vectors and frozen before
the outcome launch. The freeze SHA-256 is
`aed7213ab1d442d2d4e66bf31e1a8ae7b0121bc77816f05c5cfc1890701b189c`;
the outcome launch binds it. The known bank-REC control predicts development
H8 error 4.12132% versus the already-observed 4.15906%, with maximum local
remainder/linear ratio 0.03430 and a qualified empirical range.

| Map / development | H8 prediction (%) | Empirical range (%) | Locally qualified | H8 observed (%) | H32 observed (%) | H128 observed (%) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Original bank DYN | known baseline | -- | -- | 1.37466 | 19.93781 | 75.65897 |
| Clean-only adaptation | 0.79482 | 0.26074--1.32890 | no | 0.69917 | 42.45828 | incomplete: 1/4 paths |
| Response-preserving | 1.24441 | 1.11997--1.36885 | yes | 1.24452 | 19.31814 | 98.97956 |

The preserving arm's maximum development remainder ratio is 0.01187. Its
training H8 prediction/observation is 1.12156/1.12142%. The clean-only ratios
reach 0.27713 on development and 0.97147 on training; its training prediction
3.58991% substantially exceeds the observed 1.03022%. The failed checks were
retained, and the paired comparison was explicitly labeled exploratory before
opening outcomes. The preserving arm retains its separately qualified
prospective numerical test. The tolerance is empirical, not a statistical
confidence interval or a uniform stability bound; no qualified ranking of
the two arms or guaranteed 5% improvement over DYN was selected.

Clean-only wins H8 on all four development paths. Preserving wins H32 over
clean-only on every training/development path (12/12). First development 10%
crossings are steps 11,12,12,13 for clean-only, versus 17 on all four preserving
paths; these correspond to times 0.55--0.65 and 0.85. Preserving improves DYN
H8 on 7/8 training and 4/4 development paths, but its small pooled development
H32 advantage is not consistent across paths or populations: it wins only 2/4
development and 1/8 training paths against DYN; training pooled H32 error is
13.44330% versus DYN's 11.59397%.

At H128 preserving is worse than DYN on every path. Development energy,
enstrophy and palinstrophy NRMSEs are 99.9444/61.4125/903.2199%, versus DYN's
16.6009/11.7519/98.8080%. Clean-only hits the amplitude guard on three training
and three development paths; the remaining paths do not supply a population
H128 score. The guard is a numerical stopping threshold, not an accuracy or
physical-fidelity criterion. Both outcome evaluations took about 1.3 minutes,
including diagnostics, on the unchanged approved runtime.

Scientific interpretation: preserving measured response protects intermediate
composition against clean-only adaptation, at an onset-accuracy cost. A checked
directional model accurately predicts the preserving map's onset effect.
Neither the finite response bank nor an H8 approximation secures later fidelity.
This is a conditional quantitative result on a fixed fitted map and reused
paths, not prediction of what training will learn or independent confirmation.

The local packet checks cover 55 forecast and 35 outcome members, with shared
test receipts left identical. `forecast_analysis.json`, `prediction_freeze.json`
and `outcome_analysis.json` own the reductions, chronology and limitations;
`figure_manifest.json` binds the compact display. The script is run as
`python -m scripts.time_dependent_no.evaluate_kolmogorov_adaptation_forecast`;
the isolated launcher records exact arguments. Section 5 now has one
intervention/forecast/outcome narrative, Appendix B the numerical checks, and
the conclusion the achieved scope. Next scope paired fitting replication and
untouched-population confirmation; no penalty/horizon search or new family is
selected. Protected roles remain closed.

## September 22: Response Retention Acquired; Clean Improvement Is Nonuniform

Both 4,096-update fits and all nonrecurrent assays in ignored
`kf_bias_design_20260922b` are complete and retrieved. The two packets contain
56 hash-verified files (55 newly extracted, one identical adapter already local).
All 31 frozen sources still match the workspace. Terminal identities, initial
DYN checkpoint, all three sample tapes and loss arithmetic verify: each clean
pair appears twice in each stream, and the preserving arm uses every response-bank
input 32 times. Actual fitting takes 1.7969/3.5856 hours and 6.6016/12.2182 GiB peak
allocated memory for clean-only/preserving. There are no new solver labels,
deployment calls or parameters; both arms inherit the DYN checkpoint's solver
information. No new autonomous rollout or protected population was accessed.

The original worker stopped at `target_probe_clean_only`: the old bank's
provenance included an evaluator file subsequently extended for model loading.
The training bank and both fits were unaffected. A run-local adapter verifies
the old bank's source archive separately from the new fitted-map source archive;
probe, population, prediction and metric definitions are unchanged. It resumes
only the five missing assays under `assays_resume/`, preserving the failed
attempt. No maintained source or checkpoint was changed. All resumed stages
exit zero, and the GPU is idle after completion. This was an infrastructure
compatibility issue, not a failed corrective mechanism.

`analyze.py` independently recomputes target-bank response/forcing/REC/DYN errors
and alignment from saved fields, clean-error pooling and time-resolved errors,
common signed/odd/even response identities and exact common-input equality.
Target metrics reproduce exactly; maximum common-metric discrepancy is
2.85e-14. `analysis.json` binds the analysis/helper source and input receipts.

Percent clean onset errors (inputs 0--7):

| Map | Training, 32 paths | Development, 4 paths |
| --- | ---: | ---: |
| Original bank DYN | 0.33756 | 0.37838 |
| Clean-only adaptation | 0.11797 | 0.12908 |
| Response-preserving adaptation | 0.30110 | 0.34264 |

Both adapted maps improve this pooled error on every training/development path.
Development improvement is 65.89% for clean-only and 9.44% for preserving.
The onset gaps remain small. Later clean gaps remain explicit in analysis.json;
these onset results do not establish uniform competence throughout H128.

For unused development probes at inputs 6,10,14, define response change as
sqrt(sum ||R_G-R_D||^2 / sum ||R_D||^2), retaining full signed vectors. The
four groups below each contain 24 signed inputs; percentages are neither gains
nor confidence intervals. The probe bank has disjoint fitting times/Gaussian
draws but uses the already-open development population and the same parent
direction donor.

| Direction | RMS / training scale | Clean-only response change | Preserving response change |
| --- | ---: | ---: | ---: |
| Gaussian | 0.01 | 184.24% | 9.05% |
| Gaussian | 0.1 | 180.48% | 8.82% |
| Parent error | 0.01 | 452.35% | 8.88% |
| Parent error | 0.1 | 357.32% | 7.93% |

Preservation transfers beyond fitting-bank inputs, including every development
path in these groups. Clean-only parent-error response gains rise from
0.682/0.549 to 2.987/1.957; preserving keeps them at 0.689/0.553. This is a
measured finite-response result, not a uniform stability or manifold claim.

The central limitation is input-dependent forcing. Preserving improves
development clean errors at inputs 0--2 but worsens 3--7. At training-bank
centers (4,8,12,16), pooled clean error rises from 0.42677% to 0.52679%, while
clean-only reaches 0.25913%. At common development input 4, preserving clean
error rises from 0.22971% to 0.27893%. Its positive parent-error MSE is also
above DYN at all three common onset probes 4,5,6. Thus a reduced pooled clean
average cannot be substituted for reduced forcing along each step.

The paired-target measurements also depend on amplitude and direction. On
unused development Gaussian probes at amplitude 0.01, clean-only has lower
REC/DYN error than preserving; at amplitude 0.1 preserving is better. Native
parent-error probes favor preserving over clean-only at both amplitudes.
These are one-call target errors, not autonomous rankings or universal method
preferences. Preserving DYN's learned response can also retain its defects.

Paper interpretation: the test exposes clean fine-tuning's change to an acquired
response and demonstrates a transferable response-retention lever, with a
nonuniform clean-forcing cost. Section 5 contains the objective and compact
acquisition table; Appendix B records cost. The H8 quantitative forecast and
new rollout comparison remain open. Do not claim the preserving map improves
DYN self-composition, select another penalty, or quietly change the target
horizon. Next prepare the bounded forcing-aware H8 forecast from the frozen
maps; if it fails qualification, name that limitation before any exploratory
outcome. The selected comparison roster stays closed; independent confirmation
retains its separate population boundary. The updated 41-page PDF compiles
without LaTeX warnings or unresolved references; changed pages 22,23,38 were
visually checked. Build inputs, PDF hash and the preserved prior installed PDF
are bound in `paper_build_analysis/verification.json` under this attempt.

Fit packet SHA256:
`0e041e4fe6f49f75de8ebd4de7cd0d2fc0586b6dba1bc6f9d9d8e2bd25d4ce0f`.
Assay packet SHA256:
`4725b788e8ddc412d3c27a7c120eef8d25f45782d3c8c3acd16a07c7f6d5f121`.
The source and launch-plan hashes remain those recorded in the entry below.

## September 22: Comparison Closed; Response-Preserving Design Started

The owner closes the selected representative-method comparison and chooses a
matched clean-only versus response-preserving adaptation from the same bank-DYN
terminal. Block 3 of EXPERIMENT_PLAN.md owns the fixed recipe. This is an attempt
to reduce state-dependent clean forcing while retaining an acquired finite
response, not a claim that this tradeoff has already been resolved. The first
worker stops after fitting and nonrecurrent assays. A numerical H8 prediction
must precede the new autonomous outcomes; H32/H128, threshold crossing and
physical fidelity remain required limits, and independent confirmation is open.

The preliminary saved-training-query analysis is retained under ignored
`kf_bias_design_20260922a/analyze_bias.py` and `bias_analysis.json`. It uses only
bank-DYN center outputs on 32 training paths at inputs 4,8,12,16. Bank, checkpoint,
query-array and frozen-source bindings pass; clean-forcing metrics and the
mean/variance identity are recomputed. A fixed mean bias accounts for 1.9758%
of forcing energy, but leave-trajectory-out subtraction gives MSE ratio
0.9967406 and leave-time-out ratio 0.9990274. The PCNO itself already used these
paths; this only cross-validates the additive field. The small benefit does not
justify a constant-correction rollout and does not reject state-dependent bias
reduction. No new autonomous outcome or protected population enters this check.

One maintained fitter, `fit_kolmogorov_response_preserving.py`, and one focused
test are added, with a narrow new-recipe branch in the existing evaluator.
The response penalty differentiates through both G(x) and G(u); the archived
DYN response is fixed. The two arms share the original two clean sample tapes,
fresh optimizer and 4,096-update terminal. The preserving arm adds one eight-pair
response batch per update with unit penalty coefficient. Both deploy one PCNO
call. Fitting FLOPs differ and are reported, not claimed matched.

The 46-test CPU dependency suite passes locally and on the workstation.
The optional local Ruff check was unavailable because that module is not installed;
no dependency installation was added. Git whitespace checks pass. The
first remote preflight in attempt a stopped immediately because the launcher
omitted the existing isolated pytest dependency path; no fit ran. Attempt b,
`kf_bias_design_20260922b`, restores that path without changing scientific source
or recipe. Its 31-file source archive is verified before launch. Both 16-update
resource phases complete, and frozen teacher center/displaced queries replay
bit-for-bit in the original GPU runtime. Retrieved resource receipts, artifacts,
the 29-file scientific source closure and the initial checkpoint all verify.

| Arm | Estimated fit hours | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Clean-only | 1.7929 | 6.6016 | 7.1465 |
| Response-preserving | 3.5813 | 12.2182 | 12.5273 |

Clean-only fitting is running at this checkpoint; neither fixed terminal nor a
new rollout result is available. The queued worker includes training-bank,
unused-probe and common clean/response assays, then stops. There is no automatic
weight sweep, continuation or protected-population access.

Attempt-b source archive SHA256:
`d73773f071ee685cf27b5ad26dbf45f4f55c677a57479fd29409c4f2d4f9bc62`.
Plan SHA256:
`92f2f227b3975879a10baf874ae717c687a28a774e914a86eef77c0565766f13`.
The ignored scientific-plan snapshot, launch/resource receipts and CPU XMLs
retain operational evidence; machine-specific details stay outside tracked docs.

Section 4 now closes the comparison scope; Appendix B consolidates recorded
fit/memory/call and separate solver-bank costs. Section 5 states the selected
response-preserving objective and its matched control, with the quantitative
forecast/outcome gaps still red. The 41-page draft builds without warnings or
unresolved references, with changed pages inspected. Source/PDF hashes and
preserved prior PDF copies are in attempt b's `paper_build/verification.json`.
No completed predictive-design result is claimed by this manuscript update.

## September 21: Architectural Result Retrieved; Selected Comparisons Complete

`kf_spectral_control_20260921b` completed its unchanged 4,096-update recipe,
common assay and twelve H128 rollouts. The 402,510,712-byte results archive
is retrieved; all 41 files verify (37 new, four already present and identical).
The 32-file source archive and 30-file scientific source closure agree with
the local implementation. All process exits are zero. Checkpoint/manifest
identities, every clean sample pair against continuation, teacher-error sums,
common input equality, response algebra, pooled errors and saved-field physical
reductions are independently checked. Three original GPU training probes
replay bit-for-bit. No protected population was used.

The trained restriction is active: all 1,263 channel matrices have original
norm above one and effective norm at most 1.000000477. This verifies the
implemented component, not a complete-map Lipschitz bound. Residual additions
and graph derivatives remain. The arm is a PCNO component adaptation inspired
by McCabe et al., not a reproduction or rejection of ReFNO.

Percent errors on the four open development paths:

| Map | Clean onset | H8 | H32 | H128 |
| --- | ---: | ---: | ---: | ---: |
| Clean continuation | 0.09087 | 6.1494 | 76.1225 | incomplete |
| Online recovery | 0.09517 | 0.7526 | 60.9974 | incomplete |
| Shell prior | 0.08037 | 5.9320 | 68.8438 | 163.9019 |
| Channel norm cap | 0.46284 | 3.5109 | 73.0125 | 212.0748 |

Channel-cap training onset error is 0.45722%, close to development's 0.46284%.
This arm increases approximation error rather than primarily opening a
training--development gap. Later clean errors at inputs 16--31 are
3.4904/3.8084% (train/development). The fixed-budget result does not determine
optimal constrained approximation or optimization.

At development inputs 4/5/6, parent-error response gains are
1.71584/3.09346/1.93996 versus continuation's 2.83272/4.85515/2.46599.
Gaussian gains are 1.24835/1.23506/1.19982 versus 1.23601/1.25281/1.22997;
they are not uniformly reduced even on these probes. Clean forcing RMS is
4.83--5.03 times continuation's. At input 31, the Gaussian gain is lower
(1.05538 versus 1.09002), but positive-target MSE is 12.93 times larger.
This separates a sampled response benefit from total corrective accuracy.
The probes concern shared parent-error directions, not this arm's own errors.

All twelve autonomous paths complete H128 inside the coarse all-time RMS
ball. Six training and two development paths nevertheless violate the
tighter time-dependent envelope; development violations begin at 89 and 96.
The development first 10% error crossings are 8/9/9/9.
Development energy/enstrophy/palinstrophy NRMSE is 0.3298/0.1961/16.5290%
through H8, 9.2275/42.5550/1614.04% through H32, and
247.93/331.85/9453.12% through H128. Forcing-preserving spatial translations
at the synchronized endpoint leave 94.53--99.82% of per-path squared error.
The fields show pronounced fine-scale distortion; this does not uniquely
identify aliasing, nor does it measure distance to a true solution manifold.

Fit cost is 5.8764 hours and 6.7609 GiB peak allocated memory. The assay and
rollouts take 85.80 and 45.59 minutes. The literal implementation recomputes
exact SVD norms each forward call; the cost is implementation-specific.
Frozen effective matrices can be materialized once. There is still one
network transition per physical step and no added solver labels or parameters.

The analysis first checked CPU replay with a chosen relative RMS tolerance
of 1e-5. The largest displaced probe measured 1.0361e-5; matching the original
paired batch size gives 1.0395e-5, so that change does not remove the discrepancy.
The other two probes differ by 2.90e-6 and 3.16e-6. These CPU differences and
the original failed threshold are retained in `analysis.json`; they are
not relabeled as a passed portability check. Replaying the same inputs with
the original frozen GPU source/runtime gives exact array equality for all
three (`replay_check.json`). The scientific tables use that original runtime.

Run `python artifacts/time_dependent_no/kf_spectral_control_20260921b/analyze.py`
from the repository root to regenerate `analysis.json` and `fields.png`.
The script reuses the preceding shell analysis's pure reductions and records
that helper's hash; it adds the trained-cap and replay checks needed for this
comparison. Archive SHA-256:
`d6b901f6022577d70eb17c14cd35d0306b0f30d3faadd6b38f2363560a4ef10d`.
Terminal SHA-256:
`29abd538c535b1147983a31336902e9eda5dbfc144b8c68990b17995b23448a4`.

This completes the selected representative comparisons at their declared
adaptation and population scope, including the explicit stationary-data and
deployment exceptions. Section 4 adds one shared-table row and one response
paragraph; Appendix B owns recipe, physical and cost detail. The compact
story separates acquisition, response transfer and fidelity. No new main
figure, fit, tuning sweep or protected-data access is selected. Intervention
design, quantitative prediction and independent-confirmation scope now await
the owner's discussion. Attempt a remains an infrastructure-only stop.

The manuscript now has twelve measured maps in the shared table. The 39-page
PDF compiles without unresolved citations/references or box warnings; pages
13, 14, 20, 21, 35 and 36 were visually checked. Source hashes stayed fixed
during compilation and the previous installed PDF is preserved. The run's
`paper_build/verification.json` binds these checks to PDF SHA-256
`536b0d50c84d3e5246ae7057ff5dbd1724ac05b6c9730c5a67c28a8bb46091e0`.

## September 21: Shell Prior Retrieved; Architectural Comparison Running

The complete `kf_remaining_coverage_20260921a` archive is retrieved. All
63 members verify (43 newly extracted, 20 already present and identical),
including the shell terminal, full clean assay, common-response arrays and
twelve H128 rollouts. All processes exited successfully. Source closure,
checkpoint identity, artifact hashes, all 4,096 clean sample pairs against
continuation, teacher sums, response algebra and saved-field error/physical
reductions verify independently. No protected population was used.

Percent errors on the four open development paths:

| Map | Clean onset | H8 | H32 | H128 |
| --- | ---: | ---: | ---: | ---: |
| Clean continuation | 0.09087 | 6.1494 | 76.1225 | incomplete |
| Online recovery | 0.09517 | 0.7526 | 60.9974 | incomplete |
| Online recovery + RMS cap | 0.09517 | 0.7526 | 60.9974 | 332.7542 |
| Shell prior | 0.08037 | 5.9320 | 68.8438 | 163.9019 |

The shell objective is acquired on 64 fresh inputs: mean relative squared
error to 0.5x falls from 0.0773524 to 0.00107227; mean output/input RMS falls
from 0.751125 to 0.502742. The before measurement is the common parent,
not a shell assay of clean continuation. This is sampled target acquisition,
not a uniform contraction bound. The norm ratio from the origin must not be
equated with the antithetic near-reference response gain.

On identical common inputs, pooled Gaussian gains at steps 4/5/6 are
1.23638/1.25299/1.22965, differing from continuation by less than 0.031%.
Parent-error gains are 2.84637/4.87644/2.40283 versus continuation's
2.83272/4.85515/2.46599 and recovery's 1.41948/3.14460/1.95004.
The shell prior's clean onset errors are 0.07106/0.08037% (train/development);
its first development 10% pointwise crossings are 7/8/8/8. Thus the strong
change on synthetic shell inputs does not transfer into substantial early
response improvement. These probes use parent-error directions, not the
shell model's own reached displacements.

All twelve shell rollouts remain inside both RMS envelopes through H128,
and below the trained shell's inner radius throughout. Development physical
NRMSE through H128 is 271.56/145.08/4573.97% for energy/enstrophy/palinstrophy.
Even through H8, palinstrophy NRMSE is 59.83%. At step 128, independent
forcing-preserving spatial translations leave 85.55--99.67% of per-path
development squared error. This alignment searches all grid x shifts and
quarter-period y shifts; it does not optimize time or search neighboring
solutions. The implementation's (x,y) axis convention was checked with exact
synthetic allowed/forbidden shifts. The saved-field images show fine-scale
distortion; neither this nor the diagnostic identifies aliasing as its cause.

Interpretation for the paper: successful acquisition on a prescribed region,
response on early displaced inputs, finite evolution, and physical/path
accuracy are separate observations. The shell prior supplies a useful
contrast with local recovery and the explicit coarse cap. It does not
demonstrate manifold retention or a universal ordering. Section 4 receives
one shared-table row and one interpretive paragraph; Appendix B receives
the recipe and cost. No separate main figure or design study is added.

The shell fit took 2.7012 hours and 6.60 GiB peak device allocation;
assay/rollout took 811.33/78.88 seconds. Its run-scoped `analyze_shell.py`
regenerates `shell_analysis.json` and `shell_fields.png` from the frozen
artifacts, preserving the evidence behind the manuscript's new row. Invoke from
the repository root with
`python artifacts/time_dependent_no/kf_remaining_coverage_20260921a/analyze_shell.py`;
this independently checks the evaluator's reductions without rerunning a model.
Results archive SHA-256:
`7b2f90c1394a2eecfbcc69d47d566f978ebff8628cc423b8c2b9d7173a8894d7`.
Shell terminal SHA-256:
`63157149514b18919c95265ea6448911f8e12026cd5c56079a797bed783d6ada`.

Architectural attempt `kf_spectral_control_20260921a` completed its finite
16-update resource trial, but stopped before the full fit because its 5.86-hour
estimate exceeded a four-hour allowance. Its resource logs and failure record
are retained; no mechanism conclusion follows. The isolated successor
`kf_spectral_control_20260921b` runs the same objective, model, sample tapes,
optimizer and 4,096 updates with an eight-hour fit allowance and two hours per
evaluation. Only runtime guards changed; three focused CPU tests pass again.
It is training; full acquisition, common assays and rollouts are pending.
Finish that comparison before the owner discussion of intervention design
and quantitative prediction. No additional data or protected scope was opened.

The updated manuscript remains 38 pages, with eleven measured maps in the
shared table. It compiles without unresolved citations/references or box
warnings; pages 13, 18, 20, 34 and 35 were visually checked. Source hashes
remained fixed during the build, and the previous installed PDF is preserved.
The run's `paper_build_shell/verification.json` binds the checks to PDF SHA-256
`cb68fa32a8d4f10ae1c27f769761b0f6b0befb4f2ad1103de61680bdbbe51721`.
The task-created test temporary directory and analysis bytecode were removed.

## September 21: Remaining Coverage, Physical Cap And Reference-Law Scope

Owner sequencing is coverage first, then discussion of intervention design and
quantitative prediction. No blend, switch or new framework-guided method is
selected. Two bounded runs use the existing fixed32 training and four open
development trajectories; only source was uploaded to the selected workstation.

Completed and independently reduced in `kf_remaining_coverage_20260921a`:
online recovery followed by projection onto the qualified all-time RMS ball
(25.712973861328997 times 1.00001). All twelve paths finish H128. Development
clean onset/H8/H32/H128 percentages are 0.09517/0.7526/60.9974/332.7542.
First cap activation is 55--67 on training paths and 57--65 on development,
well after development's 10% error crossings at steps 10--12. All common-response
vectors are bitwise unchanged; all pre-activation rollout diagnostics match the
uncapped map. Independent saved-field reductions verify squared error,
energy/enstrophy and the pooled horizon totals. This enforces a coarse ball,
not the tighter time-dependent envelope, data-law proximity, or path accuracy.
Section 4 now contains one table row and the interpretation; no new main figure.

The same run is fitting an MNO-style shell prior: unchanged clean exposure plus
canonical radially uniform shell inputs outside the qualified ball, synthetic
half-input targets, and the fixed 4,096-update rule. Its relative shell-loss
normalization is a labeled PCNO adaptation, not the source Sobolev recipe.
The resource trial passes (2.37155 seconds/update, estimated 2.70 hours,
7,090,510,848 allocated bytes). Fresh-shell acquisition, common assays and
rollouts are queued after fitting; no trained-map result is claimed yet.

`kf_spectral_control_20260921a` is queued after this run releases the GPU.
It caps each dense complex Fourier-mode and pointwise channel-matrix norm at
one, retaining residual connections and graph derivatives. Parent mode-matrix
median norms 1.73--2.17 show this restriction is nonredundant. Matched clean
sample tapes, optimizer and 4,096-update budget expose clean-bias and acquisition
cost. This is the normalization component adapted to PCNO, not full ReFNO or a
whole-map contraction certificate. Both fits retain their pre-outcome qualitative
expectations and unfavorable-outcome interpretations in the ignored run plans.

Training-only stationary-law applicability is also checked. Sample every eighth
state; blocks [256,320), [320,384), [384,448), [448,513). Enstrophy means are
5.0622/5.1944/5.5002/5.7337 and palinstrophy means
111.3403/114.6759/120.0698/125.7039. Last-minus-first paired trajectory intervals
are [0.1944,1.1486] and [4.4384,24.2888], giving ensemble changes 13.26% and
12.90%. Energy's paired interval [-0.0487,0.0156] includes zero. These moments
do not justify an invariant-law interpretation of the fixed transient pool.
DySLIM/Thermalizer remain in the mechanism discussion without an empirical
ranking here. No new stationary dataset or transient-objective adaptation is
selected. Section 4 and Appendix B record this population limitation.

Verification: 30 dissipative/common-evaluator CPU tests and three architecture
tests pass locally and remotely. The two source archives contain 33 and 32 files.
The cap analysis verifies all 31 evaluator source hashes, both input manifests,
all assay/rollout artifacts and the shared REC checkpoint. Exact evidence:

- `kf_remaining_coverage_20260921a/cap_analysis.json`, `analyze_envelope.py`,
  `cap_fields.png`, and `assay_envelope/` / `rollout_envelope/`.
- Cap assay/rollout result SHA-256:
  `2348cea91d470189a2595fb396e5f58f0cf52cbc15cd49a97b1c189049678b8e` /
  `b0f55dc925233bc03742a5a0c7add58debb7422f85f925e56444915646b34698`.
- `applicability.py`, `applicability.json`, `training_statistics.npz` in the
  same run. Training array hash remains
  `9e204083b0e9978668dd1ee13ef15886245f4833a291a9eb7268aaf9df7f709c`.
- Coverage source archive:
  `69d1c232db7da2e5fb06e04ae74d853536e0035f5d96b51a930a85956beaf4a0`;
  architecture source archive:
  `57f4f7cc12d42521b4cbd5f162fedf10551bc482132d5d8a1e36274a8a8103fd`.

The updated 38-page manuscript compiles without unresolved references/citations
or box warnings. Changed table/text pages 13, 18, 20 and the reference-law appendix
were visually checked. PDF SHA-256:
`4c5987cbfaa422381c3f1ec390b5d25b87bb65dcfae4c59771784782c2bddef7`.
The build receipt and checked renders are in the run's `paper_build/` directory.
Task-created local pytest temporary directories were removed after validation;
frozen source/run records and pre-existing work remain intact. No commit was made.

## September 20: Fixed Refiner Comparison Separates Successor Refinement From Feedback Stability

Completed, retrieved and independently reduced:
`kf_refiner_20260919c/evaluation`. This evaluates the existing 16,384-update
EMA terminal with four calls per physical step. No fit, extra labels or protected
population is added. All 57 transported files, 39 frozen source files and exact
checkpoint/plan bindings verify; eight training query pairs replay across three
tapes. All 25 focused CPU checks pass locally and remotely. Independent saved-
field reductions verify clean moments, response accounting, identical physical
inputs against five controls, rollout/scalar/spectral/phase snapshots and all
horizon sums. Both evaluation stages completed and the GPU was checked idle.

Onset clean error is 0.92054% training and 0.89707% development. The development
three-field mean gives 0.72628%; within-sample spread accounts for 34.4532% of
sampled squared error. These are one-fit sample moments, not fitting replication
or an ensemble rollout. Development H8 is 5.8652% (draw range 5.7392--6.0426%).
H32 is already explosive: mean 552723.53%, range 94460.52--1321504.01%.
All 36 sampled paths reach the existing numerical amplitude guard (RMS/training
scale > 1e6), at steps 35--50 on training and 33--42 on development. All H32
paths are still scored under that loose guard; no H128 aggregate is available.
Development first 10% crossings occur at steps 8--9. The independent physical
envelope is violated earlier than the numerical guard, at steps 26--35.

On development Gaussian probes 4/5/6, shared-noise odd gains are
0.9663/0.9697/0.9720, versus continuation's 1.2360/1.2528/1.2300, but positive
clean-target MSE is 3.64/4.42/6.06 times larger because clean forcing increases.
Native-parent gains improve relative to continuation at these inputs but remain
1.1813/1.6137/1.0564. Positive-input error improves at 5/6, then the input-31
gain reaches 2.1364 and MSE is 3.35 times continuation's. The benefit is limited
in direction and amplitude; these measurements are not uniform stability bounds.
Through H8, energy/enstrophy NRMSE is only 0.4075/0.3982%, while palinstrophy
NRMSE is 36.46%. Step-8 forcing-preserving phase alignment removes no error.

Interpretation: internal successor refinement and recovery of the recurrent
conditioning state are distinct tasks. The earlier chain audit and present
rollout jointly expose a limitation of this fixed adaptation, without isolating
its optimization, representation or schedule as the cause. This contrasts with
ACDM's bounded but overdamped failure. Clean versus online recovery remains the
strongest clean-error/rollout reversal; neither stochastic recipe replaces it.

Section 4 adds one shared-table row and compact interpretation; Appendix B
records acquisition, sampling, physics and cost. Assay/rollout took
355.01/236.34 s; median attempted-update time including diagnostics is 0.1491 s.
The ignored evaluation analysis.md/json and inspection figures own details.
Close this comparison without automatic tuning. Response-prior/applicability,
useful quantitative design and independent confirmation remain open.

Bindings: terminal
`49a07a49069f3f113a0fea3df2fb9b0817e7d3215beb442894d232dfedfb5d08`;
retrieval archive
`2f832b2f227f071a56334ef4005bd6274198c7760c50544f5e25a243e500ec73`;
independent reduction producer
`f80129fb96d601b0de4180a6b712faf88165cdbfdefb7536307c294823d5b92a`.

## September 20: Conditional Generation Attenuates Response But Overdamps Rollout

Completed and retrieved: `kf_acdm_20260920b/evaluation_v2`. The fixed 8,192-update
terminal is unchanged. All 57 transported files, 33 source files, plan/launcher/
checkpoint bindings and completed-process receipts verify. The eight acquisition
queries replay exactly before the assay. Independent local reductions verify
teacher-field SSE and empirical sampling decomposition, common signed inputs
and response accounting, sparse rollout fields/physical scalars/spectra/phase,
all recorded horizon sums and paired reference norms. The seven focused CPU
checks passed before remote evaluation. No new fitting, solver labels or protected
population was used; both stages are terminal and no job remains running.

Onset clean error (inputs 0..7) is 8.23548% training and 8.24933% development.
The three-field sample mean still has 7.70901% development error; within-sample
spread contributes 12.6722% of mean sampled squared error. This is an exact
three-realization decomposition, not a population bias estimate or three fit
seeds. Input-255 spread is larger (48.45% on development). The acquisition's
9.5380% score used different times and remains separately identified.

On identical unused development Gaussian inputs at 4/5/6, common-noise odd gains
are 0.2754/0.2804/0.2864 versus continuation's 1.2360/1.2528/1.2300. Positive
clean-target MSE is nevertheless 313.55/305.65/288.77 times larger, dominated
by clean forcing. Native-parent target-MSE ratios are 142.54/6.072/1.005 at
4/5/6 and 0.6044 at the large step-31 displacement. Lower gain is not lower
total target error, a uniform bound, or trusted displaced-state fidelity.

Development H8/H32/H128 is 31.3987/69.1665/83.2056%, averaging three per-draw
pooled L2 scores. Draw ranges are 30.9516--31.9022%, 68.7747--69.6153%, and
82.9075--83.4242%. All 36 path/draw combinations complete H128 within both
coarse envelopes, but all cross 10% pointwise error at step 2 (time 0.1).
H128 energy/enstrophy/palinstrophy NRMSE is 79.01/70.35/83.06%; fields show
excessive amplitude decay relative to the reference. Synchronized-time allowed
grid translations leave 89.68% of step-128 squared error. Spectral L1 differences
at steps 8/32/128, normalized by reference enstrophy, are 61.80/89.30/73.42%.
Finite rollout therefore supplies neither useful path nor measured physical
fidelity here. It does not establish proximity to a manifold or invariant law.

Assay/rollout wall time is 26.62/56.88 minutes; median diagnostic-inclusive
transition time is 0.733 s, with 20 network calls. This is one fitted PCNO
adaptation on reused paths, not a paper-faithful ACDM reproduction or family
ranking. The outcome explains a clean-bias/response tradeoff retrospectively;
no numerical horizon or rank was predicted. Keep the design obligation open.

The ignored attempt's updated `analysis.md`, `evaluation_v2/analysis.json`,
verification log and fixed-first-case field/physical inspection figures own
detail. The two reduction/plot producers are offline reproducibility tools,
not new scientific runtime dependencies. One shared-table row and compact
Section 4 interpretation replace the pending ACDM slot; Appendix B records
sampling and implementation. No additional main-text figure, fit or sweep.
Evaluation archive SHA256:
`e84a8ba1b3dd5e1e2ebcb8eb030028e084d6fe8a9b591b9d0038ef0d29baf5d1`.
Terminal SHA256:
`bf52115c6b15be69b898a34ee49597575bfcf313391faef228cdbac4f42763bc`.

## September 20: ACDM Acquisition Retrieved; Fixed-Map Evaluation Running

`kf_acdm_20260920b` completed normally at 8,192 updates in 3.99297 hours with
6.65375 GiB peak allocated memory. All 18 retrieved files and 18 frozen sources
verify, as do both complete sampling streams, the terminal identity, all update
rows and all four recorded 128-query probes. `analyze_acquisition.py` reproduces
the comparisons directly from the saved per-query errors and bound deterministic
teacher arrays. Terminal SHA256:
`bf52115c6b15be69b898a34ee49597575bfcf313391faef228cdbac4f42763bc`.
Retrieval archive SHA256:
`f788d73741e4e1ed7d4e009aace07a8448dec889121497cbee3a6200e273be8e`.

On identical training queries (32 paths, input steps 0,7,31,255), mean three-tape
ACDM relative L2 is 9.53801%, with individual values 9.56452/9.55997/9.48954%.
Clean continuation, online recovery and conditional Refiner are 0.22827%,
0.24213%, and 2.34112% on those same queries. ACDM improves persistence's
15.0066% error by 36.44% relative but sacrifices much of the available clean
accuracy. Its mean probe sequence is 12.7690 -> 11.2232 -> 9.5846 -> 9.5380%;
the last quarter improves only 0.4865% relative. This is not a convergence
certificate, severe training/validation overfitting, or an autonomous ranking.

The completed fit establishes a substantial reference-state cost in this PCNO
adaptation. It does not determine whether the model handles displacement better
or retains useful flow structure. Similar three-tape aggregate errors also do
not establish small field-level sampling spread. The ignored `analysis.md` and
`acquisition_analysis.json` own the per-time results, acquisition interpretation,
source bindings and comparison limits. Appendix B includes the clean-query
comparison and measured cost; Section 4's rollout row remains pending.

The already-planned evaluation is now running in `evaluation_v2/`: clean input
steps 0..7,31,255 on all 32 training/four open-development paths, common-noise
Gaussian/native-parent response, then H128 on eight training/four development
paths with three sampling realizations. It reuses the unchanged sampler and
existing FP32 recurrence/scoring helpers; spectra and forcing-preserving phase
alignment are measured on saved fields. Seven focused local and remote CPU
checks pass. Exact generation replay precedes new assays. No fitting, solver
labels or protected population is added. New outcomes are not yet reported.

The first evaluation package stopped before data access because an unchanged
high-noise FP32 roundtrip test had an overly tight fixed tolerance: one error
was 2.258e-6 against 2e-6. The focused successor suite includes a bound accounting
for cancellation and FP64 verification. No model or scientific measurement
changed. The original test/launch evidence is retained, separately from results.

## September 20: Conditional-Generation Comparison Implemented And Fitting

`kf_acdm_20260920b` advances the separate ACDM-style main-case comparison.
The question is whether noisy conditioning plus iterative successor generation
changes response and sustains useful flow fidelity after path accuracy is lost.
The expectation is recorded before outcomes; lower sensitivity, plausible fields,
clean bias and sampling variation are separate possibilities. Following current
owner direction, a poor clean score or unsuccessful local forecast does not
automatically veto finite-rollout evaluation.

The one-state PCNO adaptation retains joint epsilon/Huber training, 20 DDPM
levels, the released linear schedule and repeated noisy-condition replacement.
The known forcing stays exact and only the final physical successor receives
the common restriction. The official source is pinned at commit
`123e71b8d8f0f8cdb53b6bf29201b050c6446299`; the ignored plan binds three source
snapshots and discloses the PCNO, history and initialization adaptations.
The fixed 8,192-update fit uses the existing authenticated 32-path train packet,
fresh Adam and the fixed terminal. No new labels or population are introduced.

All 18 remote CPU tests pass, including the real small-fit/checkpoint path and
exact randomized-update replay. Seven local checks also pass; local temporary-
directory ACL failures prevented the remaining fixture-based checks there, so
the complete focused suite ran remotely. The 16-update GPU resource stage is
complete and independently verified: 18 source bindings, public reference
hashes, launch/recipe hashes, artifact hashes and both exact pair streams pass.
Parent replay's maximum relative RMS discrepancy is 1.462e-7, below its 1e-6
tolerance. This is a tolerance-qualified replay, not bitwise equality.

Measured update time is 1.5913 seconds and peak allocated memory 6.654 GiB.
Update-only fitting projects to 3.62 hours; scaling the measured full-transition
probe gives approximately four hours including the four acquisition probes.
The resource model is discarded. The fixed full fit is now running, with a
six-hour cap. Resource-probe errors are not acquired-map or rollout evidence.
Resource result SHA256:
`2e5badaf6890fa538478d91f8f240fecc7190998cf508abac836db9f699cf216`.
Source archive SHA256:
`2a3279c2839190b9b94e9673a3a2b379f72e229b744d912e5e659cd1bd3be117`.
The ignored attempt's `resource_verification.json` and `verify_resource.py`
own the independent check; source and launch receipts are retained.

Infrastructure predecessor `kf_acdm_20260920a` stopped in test discovery before
training because the launcher named an unsupplied baseline test. The successor
removes that unchanged-baseline test from the launcher; all new tests remain.
Its source archive and scientific plan are byte-identical. This is not a
scientific failure or a recipe revision.

Next: retrieve the terminal and evaluate three sampling realizations on the
already-open eight training/four development paths, with common-noise response
assays, H128 accounting and time-resolved physical/spectral fidelity. No protected
outcome is opened. Section 3 and Appendix B now describe the implemented mechanism;
the shared Section 4 row retains its result TODO. The manuscript builds to 35
pages with no unresolved references or box warnings. New results are pending.

## September 20: Fixed Prefix-Switch Design Check Closed Without A Candidate

`kf_prefix_switch_20260920a` tests the single bounded revision to the preceding
blend: online REC for steps 1--8, then bank-DYN. The schedule was fixed before
the diagnostic, motivated by fewer deployed calls and no repeated map mixing.
It is a composition of two acquired maps, not an isolated target contrast.
The first eight training paths, checkpoints and clean population stayed fixed.
No fitting, solver, development or protected-population access occurred.

Along the observed DYN path, a first variation transports the REC-minus-DYN
step-8 displacement to step 16. Signed finite-amplitude probes and separately
propagated remainders do not change the forecast states. The frozen selection
rule rejects the candidate: maximum remainder/linear-response ratio is 1.2460
against 0.10; only 3/8 paths pass throughout. The linear surrogate endpoint
error is 22.2202% versus actual DYN 5.5871%, and the curvature diagnostic makes
the empirical range uninformative. **None of these surrogate quantities is a
measured hybrid rollout error. No hybrid rollout followed.**

The switch itself needs a qualification: REC improves step-8 endpoint error on
7/8 paths, but pooled REC/DYN errors are 1.9320%/1.8290%. The eighth path also
accounts for 92.96% of surrogate squared error. Earlier cumulative/development
advantages cannot be substituted for this endpoint/cohort. The result limits
this local forecast rule; it does not establish failure of scheduled correction,
manifold escape, or an instability of true dynamics. Do not drop a path, relax
the screen or sweep switch times after this result.

All 22 retrieved files, 31 frozen source files, prior checkpoint/population
bindings and saved step-8 states pass checks. FP64 reductions independently
reproduce the errors, all 128 signed-probe ratios and selection decision.
Thirty-nine focused CPU tests pass locally and remotely. The forecast stage
took 58.34 seconds and 3.500 GiB peak allocated memory; DYN anchor replay is
bitwise exact and the JVP qualification passes. Result SHA256:
`a700a9aae6c66f342572f952864074b31573ee00c241f0992cff77db4cf08980`.
The ignored attempt's `plan.md`, `analysis.md`, `analysis.json` and `analyze.py`
own the rationale, complete bindings and reproducible reduction.

The bounded revision is closed. Keep this feasibility evidence in planning
records, with no new main-text panel. Section 5 now explains why finite-amplitude
validity is part of a quantitative decision and retains its unresolved outcome
test. Appendix B records the verified conditional-Refiner acquisition limit;
Section 3 briefly distinguishes memory/state closure from correction. Remaining
representative comparisons, a useful quantitative design and independent
confirmation remain open. Continue those paper obligations instead of another
blend/switch forecasting branch.

## September 20: Fixed Refiner Continuation Retrieved And Acquisition Review Closed

`kf_refiner_20260919c` finished normally at update 16,384 on September 19.
All 16 retrieved files, checkpoint and inherited 20-file source binding match.
The 8,192 new update rows and both complete 16,384-update sample streams were
independently checked, including exact restart-probe agreement. Eleven focused
CPU tests passed. The continuation cost 13,315.03 seconds (3.699 hours), bringing
the two fits to 7.395 hours; peak allocated memory was 6.969 GiB.

The same 128 training queries give full-map relative L2
2.34003/2.33984/2.34349% across the three fixed noise tapes. Their mean is
2.34112%, versus 2.48831% at 8,192 updates: 5.915% relative improvement.
Matched clean continuation and online recovery remain 0.22827% and 0.24213%.
The gap is present at all four inspected training times; no development gap or
new-data requirement is inferred. Continuation probes at 10,240/12,288/14,336/
16,384 updates give 2.22960/2.22239/2.23972/2.34112%. The fixed terminal is
retained; the smaller intermediate score is not checkpoint selection or a
convergence certificate.

The repeated train-only reverse-chain diagnostic exactly replays all 128
seed-101 acquisition errors. Its decoded state error is
4.42746 -> 2.47930 -> 2.34596 -> 2.34003% through levels 3/2/1/0; candidate
deviation is 1.00015/1.64694/8.13183/89.15023 times the respective training
noise RMS. The final oracle-noised reconstruction error is 0.01892%, but its
candidate already contains the true increment with tiny noise. Final-level
velocity MSE remains 89.75% of the zero-velocity comparator's. The audit took
52.02 seconds and used no development state, solver label or physical rollout.

Close the unchanged recipe at its registered review stop. The internal chain
improves its coarse estimate but has not resolved the clean-fidelity and
candidate-response concern. A larger clean error alone is not evidence of worse
rollout; no autonomous Refiner ranking or family rejection is established.
Any bounded revision needs a concrete mechanism addressing this mismatch, rather
than another automatic fit extension. Main-case representative coverage and the
quantitative design/confirmation obligation remain open. Keep this acquisition
history in reproducibility/supporting material rather than adding a main-text
campaign panel. Existing NACA/Bump Refiner evidence remains separate.

Bindings: terminal
`49a07a49069f3f113a0fea3df2fb9b0817e7d3215beb442894d232dfedfb5d08`;
fit result `827f545548732a4b2d13fb9bc5488bd4e5f626cf2d55cd71bba31350a80651a0`;
chain result `18da03eb6c328cdfc50f14a69b00670eab85b3732d453a74b1ba0026fba92389`.
The ignored attempt's `analysis.md`, `acquisition_analysis.json` and two retained
audit producers own matched-query reductions, replay and exact source bindings.

## September 19: Refiner Pilot Retrieved; Blend Sensitivity Closed Without A Candidate

`kf_refiner_20260919b` completed 8,192 updates in 13,308.25 seconds (3.697
hours), with 6.968 GiB peak allocated memory. All seven retrieved files, the
20-file source binding, checkpoint and 8,192 update rows pass verification.
Independent matched-query reductions use only the same 128 training pairs:
full-map Refiner relative L2 is 2.4921/2.4913/2.4815% across three noise tapes,
versus 0.22827% clean continuation and 0.24213% online recovery. These are
acquisition queries, not onset-band or development scores.

The fixed-terminal chain audit exactly replays all 128 seed-101 errors.
Decoded-state error decreases 4.8013 -> 2.6923 -> 2.5010 -> 2.4921% through
reverse levels 3/2/1/0. Candidate deviation, in units of the corresponding
training sigma, has RMS 1.0002/1.7479/8.9445/96.1360. Oracle-noised candidates
have RMS near one. The internal mismatch is an acquisition/finite-range
limitation, not evidence against the published method family. No Refiner
autonomous or development evaluation exists. Its ignored analysis.md records
the matched cohorts, exact replay and one diagnostic reduction-shape repair.

The falling acquisition curve motivates one unchanged-recipe continuation,
`kf_refiner_20260919c`, to a fixed terminal at update 16,384. Eleven focused
CPU tests and the restored terminal's three-tape acquisition replay pass;
training is running at approximately 1.58 seconds/update. Model, EMA, optimizer,
samplers and RNGs are restored. The five-hour cap and terminal acquisition
review prevent an automatic fit extension; no new data or changed target.

`kf_design_sensitivity_20260919a` completes the registered train-only DYN-to-
online-REC first-variation check. All 17 retrieved files and source bindings
match; eight baseline paths replay bitwise; saved state/sensitivity vectors
are independently reduced. Thirty-four tests pass locally and remotely.
At the fixed H16 endpoint, the surrogate optimum is alpha=0.18137, predicting
5.1111% versus DYN's observed 5.5871% and online REC's 39.9795%. The worst
signed local remainder/linear ratios at alpha 0.01/0.03/0.1 are
0.10155/0.28704/0.83455, so none passes the declared 0.10 sampled screen.
The marginal 1% failure is not a universal cutoff: its predicted relative gain
is under 1%, while the useful surrogate optimum lies beyond the supported
range and doubles inference calls. H32 sensitivity also disfavors a positive
coefficient. No strength was selected and no candidate rollout followed.

The clean-query optimum is 0.96825; omitting Jacobian propagation gives an
H16 optimum of -0.01150. Direction and time ordering matter, but these surrogate
differences do not close quantitative application prediction. Corrected plan
rationale: the eight DYN training 10% crossings are steps 20--22, not 15--18;
H16 remains the preselected earlier endpoint. The original frozen plan is
retained. This check took 124.70 seconds and 3.860 GiB allocated memory.

Bindings: Refiner terminal
`a6494b4b0f20dd84f0e1b95f05623bfb2a6c6f14af9a2ef76a205db1fb78a263`;
chain result `728aae6dbd0cdbf1f08a060a22bc0db5f7409ba947373c972955449f54b52e71`;
blend result `8b36ba38f595a06efd346da2a3de85414441813334a2b83c51c8dbe5cb7064ef`.
Attempt-local analyses own detailed reductions. These findings update planning;
they add no main-text figure, acquired-method ranking or protected confirmation.

## September 19: Conditional Refiner Acquisition Started; Design Feasibility Limit

`kf_refiner_20260919b` is running a fixed-data, train-only 8,192-update
conditional Refiner acquisition pilot. It uses the inherited PCNO body/scale,
seven input channels, four-level v-prediction, EMA .995 and four reverse calls.
All 47 focused CPU tests pass, including the FP32 oracle, parent transplant,
checkpoint/EMA/RNG continuation, training-only probes and tiny end-to-end runs.
The resource phase completes 16 updates: 1.58251 seconds/update, 12,963.93
seconds projected training time (3.60 hours), 6.97 GiB peak allocated memory,
and zero parent-to-initial-level-3 replay discrepancy. Probe overhead is extra.
The worker stops before development or autonomous evaluation. Resource success
and an incomplete fit do not establish denoising acquisition or method efficacy.

The initial `kf_refiner_20260919a` preserved two test-environment failures, then
passed 46 checks and began fitting. An added FP32 oracle exposed inconsistent
rounding of the last noise level (maximum synthetic reconstruction error
3.64203e-5); that incomplete fit was stopped at update 153. The corrected fresh
attempt forms the signal/noise coefficients before casting in both training and
sampling. Prior code/logs and the stop receipt remain. No autonomous result was
opened or compared. Test dependencies are isolated; model/data sources are
unchanged apart from the recorded scheduler fix.

Retained current packet: ignored `kf_refiner_20260919b/`, including frozen plan,
source archive, launcher, CPU-test and resource receipts; no new data upload.
Its 20-file source archive SHA256 is
`a5ba2f2eaf052b01d02dc56cb2dd384c849e65c11c5b23564875f1330a95daad`.
The frozen scientific recipe is a PCNO adaptation of the released Refiner,
not a reproduction of its U-Net configuration. Acquisition is measured at each
level and through the complete stochastic map before comparative interpretation.

Alongside it, `kf_target_pair_20260918a/audit_design_lever.py` rehashes the
completed parent/DYN common-input vectors and computes the exact affine blend
loss without any new model call. On 24 training parent-native input-4/5/6
queries, the unconstrained recovery-MSE optimum has alpha=1.0673129; its
admissible convex optimum is the DYN endpoint. Recovery-target MSE is 4.958%
of the parent's, but clean query MSE is 8.976 times larger. On the twelve reused
development queries the corresponding ratios are 3.291% and 8.958. These are
query metrics, not the full clean onset bands or new rollout outcomes.

The retained `design_lever_feasibility.json` therefore does not select an
interior strength or establish a quantitative useful-horizon forecast. It
identifies the remaining composition/direction/alignment requirement. No
strength sweep or candidate rollout followed. The draft estimate and its
scientific dependencies live in [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md).

## September 19: Matched REC/DYN Targets Retrieved And Closed

`kf_target_pair_20260918a` completed its 14 bank/resource/fit/assay stages,
then two fixed-terminal autonomous evaluations after a bound interpretation.
All 122 retrieved files and 29 frozen/current source files match. Identical
sample tapes visit every clean pair twice and every bank input 32 times.
Independent reductions verify both targets, saved response vectors, teacher
bands, rollout errors, spectral energy/enstrophy and failure accounting.
Target metric discrepancy is zero; squared-identity closure is 5.55e-17.

| Map | Train clean onset (%) | Dev clean onset (%) | Dev H8 (%) | Dev H32 (%) | Dev H128 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bank REC | 0.83470 | 0.85727 | 4.1591 | 21.9536 | 73.2598 |
| Bank DYN | 0.33756 | 0.37838 | 1.3747 | 19.9378 | 75.6590 |

On unused development native-direction inputs at amplitude 0.1, REC/DYN
recovery-target RMS/scale is 0.05528/0.06156 and dynamics-target RMS/scale
0.06550/0.05676. All four development and eight unused training paths agree
with this target-dependent ranking. Finite-response gains are 0.462/0.549,
trusted 0.661, parent 3.007. On Gaussian inputs DYN suppresses excessive
response more than REC, so faithful dynamics is not synonymous with retaining
large gain. At smaller inputs clean bias can dominate either target loss.

The recorded qualitative H8 prediction favors DYN over REC and passes on all
twelve paths; its magnitude was not predicted. DYN's H32 advantage has 2/4
development and 8/8 training wins. REC has lower pooled H128 error in both
populations, with 2/4 development and 8/8 training wins. All 24 paths complete
H128 without violating the coarse physical envelope. Development H128 energy/
enstrophy NRMSE is 3.98/5.00% REC and 16.60/11.75% DYN, despite large path error.
Post-hoc same-path time/allowed-grid-shift matching leaves 65.12/60.83% of
step-128 squared error. No exact manifold retention or tangent/normal claim.

Solver qualification passes: maximum clean restart relative error 2.65e-8;
largest sampled spatial-response difference 0.008257 times displacement.
There are 1,024 displaced training labels, 1,584 total solver calls including
probes/restarts/refinements, and 78.07 minutes bank generation. Fits take
6,478.80/6,482.69 seconds with 6.602 GiB peak allocated memory; both deployed
maps retain one PCNO call. One paired fit seed and reused paths remain development
evidence. No scientific source changed after the prior 40 passing CPU tests.

Bindings: assay_analysis.json
`66fd992d3992f2503a92590282a7d207c8b938e5677a59fc9cfcdad001294a0c`;
rollout_analysis.json
`3d0c194879d8e31a3fa2fd1a274393b1ddfc0c71d041ea6b2cf65ed2958a43f0`;
pre_rollout_interpretation.md
`447ad74580d9ee63665725f2e2c7141e2a48ed3c99a9d9f6f2fd7696695bf90e`.
The ignored analysis.md and figure_manifest.json bind the detailed comparison,
saved-field check and shared replacement panel. Section 4 includes the result;
Section 5 still explicitly lacks quantitative design confirmation. The active
plan selects Refiner next, alongside bounded prediction feasibility, then the
remaining distinct coverage and proportionate confirmation. No new fit launched.

## September 18: Onset REC/DYN Target Contrast Launched

`kf_target_pair_20260918a` freezes identical displaced FP32 inputs for REC and
DYN while retaining all 32 clean training paths in both objectives. Training
bank: 128 centers, 1,024 signed Gaussian/native-direction inputs at steps
4/8/12/16 and RMS amplitudes 0.01/0.1 training scales. Separate unused probes:
36 centers, 288 inputs at steps 6/10/14, including the four already-open
development paths. No protected population or new clean trajectory is accessed.
The experiment plan and ignored attempt plan own the recipe and expectations.

Two new bounded entry points reuse existing solver, model, replay, sampler and
optimizer-update code. Forty focused CPU tests pass, covering exact target
semantics, matching inputs/sample tapes, fresh parent restarts, numerical
refinement, train/probe exclusion, saved-vector metric checks, failure receipts
and the older unroll/evaluator contracts. Windows pytest directory permissions
required a temporary repo-local fixture within the sandbox; this was a test
environment issue. No scientific outcome follows from that issue.

The 29-file source archive, scientific plan, input manifests and launcher were
hashed before upload. The launcher rechecked them, verified a free GPU and
available disk, and started one durable worker. Its 14 bounded stages are bank
qualification/generation, disposable resource checks, paired fixed fits, and
training/unused/common-input assays. It stops before autonomous evaluation.
The initial live receipt showed training-bank generation running with no
stderr. Qualification or execution failure stops this attempt; no automatic
rerun, tolerance relaxation or fit selection is queued. No REC/DYN result yet.

The manuscript was shortened by removing the superseded three-map overview;
the artifact remains intact. The 32-page draft builds with no unresolved
references/citations or box warnings. New target results will join the shared
comparison and response/bias panels after verification; Section 5's quantitative
application prediction remains open.

## September 18: K=4 Gradient Contrast Retrieved And Closed

`kf_unroll_pair_20260918a` completed both fixed terminal fits, assays and
subsequent autonomous evaluations. All source/checkpoint/input bindings and
the identical 4,096-by-4 sequence-index tapes pass. The 16/12/32-file fit,
assay and rollout transport packets were rehashed before extraction. All
96 response rows per map were independently reduced from saved fields; shared
inputs are byte-identical across the two new and three contextual maps.
Maximum metric discrepancy is 3.38e-14 and signed closure is 1.43e-14.
Teacher bands and rollout scalar/snapshot reductions, including censoring, pass.

| Map | Train clean onset (%) | Dev. clean onset (%) | Dev. H8 (%) | Dev. H32 (%) |
| --- | ---: | ---: | ---: | ---: |
| Detached K=4 | 0.16691 | 0.16583 | 1.1694 | 59.7540 |
| Full K=4 | 0.20268 | 0.20160 | 1.7871 | 62.6150 |

Only these two arms isolate the added temporal-gradient paths. The older
independent-pair continuation/recovery fits remain contextual controls.
Full/detached positive clean-target MSE ratios on common development
parent-native inputs at steps 4/5/6 are 1.3529/1.3238/1.1179. Clean forcing
and signed alignment contribute alongside odd response. Assay-informed H8
predictions were saved and bound before either new autonomous evaluation:
detached below full and both below clean continuation. Both pass. These are
qualitative predictions on reused development paths, not quantitative design
or independent confirmation.

Detachment improves H8 on all twelve paired paths and H32 on 11/12 (3/4
development). Recovery still wins H8 on all twelve paths. Detached has a
small pooled H32 advantage over recovery, with only 2/4 development and 4/8
training wins. Development H32 energy/enstrophy NRMSE is 18.90%/27.83%
detached versus 21.04%/30.18% full and 25.58%/35.74% recovery. Twenty-three
of 24 new map--path rollouts hit the amplitude guard before H128. The sole
completed detached training path has H128 relative L2 1.2904e4; no
complete-population H128 aggregate is reported. Small onset train/development
gaps persist; later clean transient gaps remain qualified.

Actual fit times are 6,491.43/6,487.05 seconds and peak allocated GPU memory
3.5492/11.9310 GiB (detached/full). No solver labels or deployed model calls
were added. The two fits' initial GPU forward losses differ by at most
1.006e-7, distinct from the smaller discarded-resource difference below;
analytic CPU gradient checks, not bitwise GPU equality, verify semantics.
The original assay/rollout numerical function ASTs are unchanged. Relative
to the intervening reached-response source, only checkpoint dispatch changed.
No source edits followed the previously passing 40 focused CPU tests.

Bindings: assay_analysis.json SHA256
`64e0efd29f4a990b17b82279c64319a0526595db72df28dd2f2df3d934e10377`;
rollout_analysis.json
`4e84a019a965671cff74337ad83d42c2ea59345aff3a9333816b0e75969fe815`;
pre_rollout_interpretation.md
`fa926e7977bad4624f75bfd78db5624bbe3d72053a7d2d05be0628acbb311c4a`;
unroll_comparison.pdf
`491b5adf14fc8ffc7c6b352b6db652ee57b013d7fd9731ed8712483a694131cf`.
The ignored attempt retains exact manifests, checkpoint/array artifacts,
analysis producers, cost records and launch receipts. Section 4 incorporates
the comparison. No protected population was opened. Next: the selected
same-input REC-bank/DYN-bank target contrast, not a gradient/depth sweep.

## September 18: Matched K=4 Gradient Pair Launched After Resource Checks

`kf_unroll_pair_20260918a` tests native-prefix exposure versus full temporal
credit assignment on the unchanged fixed32 population. Both arms restart the
same parent with fresh Adam at 1e-4, four sequence starts per update, four
averaged successor losses and 4,096 updates. The deployed transition remains
one-state restricted PCNO. Equal state-prediction counts do not make the older
independent-pair continuation an exact sequence control. This attempt isolates
the gradient path between its own two arms; it is not a new algorithm claim.

Forty focused CPU tests passed, including analytic full/detached derivatives,
matching forward objectives, boundary-safe sequence sampling, parent restart,
new checkpoint validation and existing recovery behavior. No core PCNO or
recovery-fitter change was needed. A new bounded unroll entry point and its
focused tests serve this comparison; the existing evaluator handles both
checkpoint recipes.

Both 16-update resource passes completed. Warmed estimates are 1.800 hours per
fit; peak allocated memory is 3.549 GiB detached and 11.931 GiB full. The resource
update tapes, source closure and parent replay match. First-update GPU losses
are not bitwise equal: maximum absolute/relative differences are 1.82e-8/4.44e-6.
The stencil uses CUDA index_add/scatter reductions; retain this numerical
qualification rather than claim exact GPU forward equality. The analytic CPU
check verifies the intended gradient distinction. The two resource passes are
discarded and both fits restart the original parent.

The detached fit has started. The durable worker runs both fits and then their
clean/common-input assays, stopping before autonomous evaluation. Resource
measurements and partial training logs are not scientific outcomes or a ranking.
No new dataset, solver label or protected population enters the attempt.
Pre-fit expectations are qualitative; the paper's quantitative design obligation
remains open. Source/archive/launcher hashes agree with the retrieved launch
receipt; exact runtime details remain in the ignored attempt.

Bindings: plan.json SHA256
`69b825b0de314054d5a92140f4c750c3f48540e68151870e88a7a907fedce963`;
source.tar.gz SHA256
`fee996d5564a607a9cdebc77c75360206a95f3a66a3269f1982cfabdca59c3d1`.
The attempt retains plan.md, resource_verification.json, both resource results
and update logs, and the durable launch receipt. Next: interpret completed
assays before rollouts and prepare the matched REC/DYN target contrast.

## September 17: Reached-State Response Separates Range, Direction And Forcing

`kf_reached_response_20260917a` completes the plan's first diagnostic block.
The three frozen maps reproduce all retained anchors on 36 paths bit for bit.
The same 900 signed input-pair rows are used by each map at steps 4/8/12/16/31,
with Gaussian, parent-error and recovery-error directions at common RMS
0.01/0.03/0.1 training scales and both native donor magnitudes. This is
retrospective evidence on eight training and four already-open development
paths, with one paired optimization seed; no fit, solver label or protected
population was added.

The ratios below are recovery/continuation mean positive clean-successor
**query MSE** on identical recovery-error inputs, not rollout-error ratios.

| Development input step | RMS 0.01 | Native recovery magnitude | Mean native RMS / scale |
| --- | ---: | ---: | ---: |
| 4 | 1.0231 | 1.0352 | 0.0064 |
| 8 | 0.3583 | 0.6656 | 0.0415 |
| 12 | 0.3959 | 0.9141 | 0.5119 |
| 16 | 0.5352 | 0.9695 | 1.0186 |
| 31 | 0.9752 | 0.9977 | 2.6053 |

At steps 8/12/16 every evaluated path improves at both listed amplitudes;
training shows the same range limitation. Step 4 is a directional exception:
the parent-error ratio at RMS 0.01 is 0.3054, but the recovery-error ratio is
1.0231. Attenuation is not contraction: step-8 common-amplitude recovery gains
remain above one. Late clean forcing can offset reduced odd response: at input
31 the Gaussian odd-response MSE falls from 1.1532e-4 to 9.6530e-5, but clean
forcing rises from 3.0921e-4 to 3.3085e-4. The net positive-target ratio is
1.0077. This qualifies the interpretation of the earlier near-unit late score;
it does not replace that historical assay's different draw law and metric.

An independent PDE/Galerkin envelope gives an all-time vorticity RMS bound of
25.71297, or 6.32793 training scales. All 32 training and four development
reference paths satisfy the time-indexed envelope within relative allowance
1e-5. This sampled check is not a uniform time-integrator certificate. Recovery
crosses 10% relative path error at steps 9--12 across the twelve paths, the
time-indexed envelope at 39--51, and the all-time bound at 55--67. Thus eventual
ideal-family departure has evidence beyond finite-bank distance, while early
normal escape versus path drift remains unresolved. A coarse physical cap
would first act much later than onset accuracy loss.

All 34 retrieved files, source/checkpoint/data bindings and shared input hashes
pass. Independent FP64 reduction of 90 retained vector rows has zero discrepancy
from stored scalar metrics; maximum algebraic closure residual is 2.53e-15.
The evaluator/fitter CPU suite passed 33 tests; historical numerical functions
are unchanged. Six remote stages took 7.89 wall minutes including process
overhead. No successor fit was launched in this block.

The ignored attempt retains analysis.md, analyze.py, analysis.json,
response_summary.csv, response_fidelity.{pdf,png}, all dense replay/response
arrays and retrieval receipts. The report and figure are incorporated into
Section 4. SHA256 bindings:

- analysis.json: `14705bc782a7bc3a61e4c40b24f7bbcf871f807237358f672dd65c1d95286f22`
- response_fidelity.pdf: `6ba764d8143978a56438c9acd2f6764238bb5ab5b583fbc982d60edf072681a0`
- retrieval_manifest.json: `22f4e8ff86d7c7c55eba7c1f59c7a66588374902e23317a536f7121f6fb8a443`

Next are the plan's matched REC-bank/DYN-bank and detached/full-gradient K=4
contrasts, emphasizing early reached directions, finite range and clean-bias
cost. Stronger noise and a new corrector remain hypotheses. Quantitative
prospective design, independent confirmation and broader representative
coverage remain open.

## September 17: Fixed-Data Recovery Improves Onset, Not Long-Horizon Fidelity

`kf_fixed_data_20260916a` completed both 4,096-update fits, all three assays and
all three autonomous evaluations. Remote/local output hashes, frozen sources,
input manifests and checkpoints match. Every logged sample index is identical
between fits; all 32 training paths and exact clean successor labels stay fixed.
Fit times are 108.00 minutes for continuation and 108.70 for recovery, with the
same 6.602 GiB peak CUDA allocation. No new solver labels or protected paths enter.

Development values below are pooled relative L2 percentages on the same four
open paths. Clean onset uses inputs 0--7; autonomous H8/H32 pool predicted states
through the horizon. Original-eight training outcomes show the same ordering.

| Map | Clean onset | Autonomous H8 | Autonomous H32 | H128 |
| --- | ---: | ---: | ---: | --- |
| Untouched parent | 0.09835 | 5.77168 | 74.94221 | All four censored |
| Clean continuation | 0.09087 | 6.14938 | 76.12248 | All four censored |
| Recovery | 0.09517 | 0.75263 | 60.99738 | All four censored |

Recovery improves H8 and H32 on all twelve paired paths versus both controls,
despite worse clean accuracy than continuation in every reported band. Its
development squared-error ratios versus continuation are 0.0149795 and 0.642091.
Continuation improves clean onset error yet worsens both rollout horizons on
every path relative to the parent. Training/development onset errors remain
close; the later 16--31 clean gap is retained rather than hidden.

Unused Gaussian paired-MSE ratios (recovery/continuation) on development anchors
4/5/6/31 are 0.61983/0.62796/0.66393/1.00276. Native-positive ratios on identical
parent-error inputs are 0.26294/0.42284/0.63034/0.99693. All early sampled paths
improve. The dominant early accounting change is reduced odd-response energy;
clean error, even response and signed alignment remain included. Late Gaussian
acquisition fails narrowly on the development aggregate, and native benefit is
negligible at the large input-31 displacements. These scores target the clean
successor and are not trusted displaced-state defect measurements.

The pre-training directional H8/H32 hypotheses pass. A separate assay-informed
expectation was frozen before rollout launch, with stronger H8 confidence,
weaker H32 confidence and no percentage or H128 prediction. This confirms the
directional effect, not a quantitative magnitude/strength design forecast or
independent confirmation. One paired model seed and reused development paths
remain the scope.

H32 is still poor: recovery's development endpoint error is 149.44%. All 36
paths hit the declared numerical amplitude guard before H128 (parent steps
84--107, continuation 85--110, recovery 99--111). No H128 population aggregate
is reported; two recovery training paths hit the guard earlier than controls.
Energy/enstrophy prediction errors improve at H8/H32, but remain large at H32.
Near-zero mean vorticity is imposed by the shared restriction. The scientific
result is a large onset benefit with an explicit failure to sustain fidelity.

Independent FP64 reduction verifies 288 signed response rows and identical
input bytes, with maximum closure residual 3.45e-15. All 36 rollout snapshot
files are checked; pooled errors, censoring and saved physical diagnostics are
recomputed. Separate energy recomputation agrees within 4.2e-16 relative.
The ignored attempt retains `analysis.md`, `pre_rollout_interpretation.md`,
`assay_analysis.json`, `rollout_analysis.json`, and `figures/overview.{png,pdf}`.
Assay-analysis SHA256 is
`73eca709d264b2af8cc5022ea3200b91dd229804bcf825098c3c60956520b0e1`;
rollout-analysis SHA256 is
`a58d4b3d372658f6dc495179173f0fb0a1b6781e64f99c3bdd1cf5e97b9a568b`.
The numerical recipe and raw outputs remain unchanged. No successor experiment
is launched; the next paper decision is a bounded quantitative design test
addressing the correction's useful range and clean-dynamics cost.

## September 16: Fixed-Data Upload And Study Explicitly Approved

The owner explicitly approved the fixed 32 training trajectories, four already-open
development trajectories and checkpoint/replay metadata (4.61 GB compressed),
excluding protected test data and manuscript files. This resolves the automatic
transfer-review block recorded below. The approved transfer is in progress;
this approval does not change the fixed recipe or open protected populations.
Remote archive verification and the full-packet resource check precede the two
fresh-parent fits. The launcher stops after the three common-input assays;
new autonomous rollouts follow assay interpretation, as specified in the plan.

Connection resets interrupted the transfer. The retained 2,523,844,608-byte
training-archive prefix was independently hashed on both ends and matched.
The attempt-local transfer helper permits three resumed attempts and invokes
the unchanged launcher only after transfer success. This is transport recovery,
not a scientific restart; no fit has started at this checkpoint. The ignored
deployment receipt owns exact live status and local invocation details.

## September 16: Fixed-Data Comparison Prepared; Broader Upload Pending

The owner clarified that data are fixed and sufficiently rich to identify
nearby reference behavior; severe clean one-step overfitting must not confound
the thesis. The owner authorized planning and execution with full access to
the selected workstation. The next Clean8/Clean32 comparison and further
baseline-forecast development are withdrawn. Existing results remain unchanged.

Independent reduction of the source-bound Clean32 terminal teacher arrays and
previous rollout records gives the following onset evidence. Values are pooled
relative L2 percentages, not mean per-step errors:

| Population | Clean one-step, inputs 0..7 | Autonomous H8 | Autonomous step-8 endpoint |
| --- | --- | --- | --- |
| Original eight training paths | 0.08415% | 5.68487% | 13.63227% |
| All 32 training paths | 0.08545% | Not evaluated on all 32 | Not evaluated on all 32 |
| Four open development paths | 0.09835% | 5.75790% | 14.06171% |

This supports an onset discrepancy shared by training and development, without
severe clean generalization failure. It does not establish uniform neighborhood
identifiability or uniformly matched H32 clean competence: inputs 16..31 retain
0.4860% training versus 1.0564% development clean error. Preserve start 0 and
report that later limitation. The ignored attempt's `eligibility_audit.json`
has SHA256 `d173009ecfba171cb6ef421a256100e7757fbdc8ec51eb3d4ace1cb6a0ac42d5`.

The new comparison holds all 32 paths, exact successor labels, parent and
normalizer fixed. It selects clean continuation versus online Gaussian recovery,
two batches of eight, 4,096 matched updates, one paired seed, expected noise
RMS/training scale 0.01. The experiment plan records pre-outcome hypotheses for
unused Gaussian acquisition, native-error transfer and autonomous benefit.
No new solver labels or protected populations enter this comparison.

The narrow trainer/evaluator and focused tests are implemented; 25 combined CPU
tests pass. Separate assay and rollout phases preserve response-before-outcome
inspection. The old solver-bank trainer's label/time contract is unchanged.
Exact repacking verified 684 blocks and 18,468 source states; all 36 ordered
FP32 paths match the original fit's state-store SHA256
`b9e3cfa138ad2103b50b29ac422897a759bb7f34deaa4f1feff055fc266e2552`.
Training and already-open development are separate packets. Their compressed
total is 4,608,894,056 bytes; manifests and archive hashes are retained under
ignored `artifacts/time_dependent_no/kf_fixed_data_20260916a/`.

The 23-file code bundle reached the verified owner-selected workstation.
Automatic approval review initially rejected code plus unpublished-plan upload;
strict host-key and prior approved-checkpoint checks verified the destination,
and a reduced code-only upload succeeded. The plan stays local. Automatic
review separately rejected the broader private data upload, requiring specific
payload approval despite the owner's full-resource-access instruction. That
bounded question is pending; the broader arrays were not transferred.

Unblocked GPU qualification used only the previously approved 33-state training
prefix and parent. Sixteen disposable updates passed parent replay, finite
optimization and memory checks: 6.60 GiB peak allocated, 7.02 GiB reserved,
1.5862 seconds per warm update, suggesting 6,497 seconds (1.80 hours) per full
fit before evaluation. This resource check is not the fixed-dataset fit, does
not validate its sampling law, and retained no checkpoint. The full-packet
resource run, both fits, common assays and new rollouts have not started.

## September 16: Finite-Amplitude Response Explains The Tangent Transition

The owner-authorized continuation `kf_response_transition_20260916a` completed
on the same selected workstation, using only the previously uploaded Clean32
checkpoint, training seed `2026090611`, and immutable first-pilot arrays.
Inputs 4, 5 and 6 explain output discrepancies at steps 5, 6 and 7. Both saved
forecast-error and actual-error directions received identical absolute RMS
amplitudes: seven fixed values plus both native error magnitudes. The attempt's
`plan.json` records expectations before these new measurements; actual-error
directions and anchor selection still make the study post-hoc.

At the actual encountered displacements:

| Input -> output | Actual error RMS / training scale | Remainder / linear response | Local derivative gain | Finite odd-response gain |
| --- | --- | --- | --- | --- |
| 4 -> 5 | 0.0056985 | 0.021574 | 1.6441 | 1.6345 |
| 5 -> 6 | 0.0100130 | 0.254716 | 6.7266 | 5.3998 |
| 6 -> 7 | 0.0542926 | 0.797802 | 9.5979 | 3.6968 |

The response is strongly amplitude dependent; at input 6 its direction also
changes (linear/odd cosine 0.66395 at the actual magnitude). The odd-minus-linear
response is 0.79805 times the linear norm, versus 0.03391 for the even component.
Thus the large departure at this anchor is mainly a change in the signed
response, rather than an even bias. These are learned-map diagnostics; reduced
gain remains above one and does not establish contraction or physical fidelity.

For d=z-e, the saved-array decomposition is
d_next = J*d - R(e) + stored-forcing/clean-forward offset + closure.
All norms below use RMS / training scale:

| Input | Propagated mismatch J*d | New remainder R(e) | Net next mismatch | Signed projection fractions: propagation / -remainder |
| --- | --- | --- | --- | --- |
| 4 | 0.000146425 | 0.000202123 | 0.000256733 | 0.352738 / 0.647264 |
| 5 | 0.00190126 | 0.0171560 | 0.0183565 | 0.068623 / 0.931377 |
| 6 | 0.187672 | 0.415730 | 0.594875 | 0.305567 / 0.694433 |

The transition at input 5 is dominated by the newly omitted nonlinear term;
inherited mismatch becomes substantial at input 6. Propagation and negative
remainder reinforce, with cosines 0.0611, 0.5981 and 0.9345. The fractions are
signed vector projections, not causal percentages. Propagation gains on the
existing mismatch are 6.399, 7.406 and 10.224.

Common-amplitude curves nearly coincide at inputs 4/5. At input 6, evaluating
the forecast direction at the actual magnitude reduces remainder/linear from
0.88607 to 0.83416; changing to the actual direction reduces it further to
0.79780. Both amplitude and direction contribute. The actual-direction result
shows that the breakdown is not merely an artifact of an oversized forecast.

Exact saved-state replay is bitwise equal at all three anchors. Closure norms
relative to the net mismatch are 0.0011680, 9.51e-7 and 6.52e-7; the first
transition carries the largest relative execution residual. Clean offsets are
3--4e-7 in training-scale RMS. Tiny-amplitude response differences retain FP32
noise despite FP64 recentering and summaries. No uniform validity radius follows.

The existing runner gained a `response_transition` phase and the utility gained
a saved-error decomposition; 41 focused CPU tests passed. The run took 24.04
seconds in the evaluator, 25.87 seconds including startup, 3.63 GiB peak allocated
GPU memory and 141 model forwards (backward work additional). All 13 staged
sources, original input and parent artifact hashes were checked unchanged;
all ten receipt-listed local files match. Result SHA256:
`509e41e4865aad4b4fe5d885d200440c2b3611ec5d015daec265caa34785c64f`;
raw-array SHA256:
`ffa3d949e5ee160f9b92db99c15035c2bccbf4666a2badb3aa15f070242252f8`.
The ignored attempt directory retains receipts, raw vectors, `summarize.py`,
verified derived metrics and the PNG/PDF figure. No solver, fitting, new path,
protected outcome or corrective intervention was used.

Scientific consequence: local derivatives explain early error growth but need
finite-amplitude qualification before quantitative corrective design. This
diagnoses forecast failure, not the physical cause of rollout inaccuracy or
the best intervention. A paired model contrast remains a possible next step;
further elaboration of the forecast is not automatically the paper's priority.

## September 16: Clean32 Tangent Pilot Completed; Short-Window Validity Only

After explicit owner approval of the exact bounded transfer and diagnostic,
attempt `kf_tangent_20260916a` completed on the selected workstation. Scope:
Clean32 terminal 49,152; original training seed `2026090611`; clean states
`0..32`; complete restricted deployed map. No fitting, solver query, new
population, protected reveal or corrective intervention was performed.

The 13 source files and five staged input files remained unchanged. All five
retrieved output hashes match the post-run remote integrity record. The packet
is under ignored `artifacts/time_dependent_no/kf_tangent_20260916a/`; the main
result is `retrieved/pilot_clean32_2026090611/result.json`, SHA256
`f783e86f35a201851e6ec4acbe589f9d483993878bc61402e96a6deef102e7b6`.
The forecast was saved and hashed before nonlinear rollout. Its SHA256 is
`6006a254fac1273c2b4cf9c35f147a7bc24dbb9b6457299501f4d9390402decf`.
Known historical outcomes still make this calibration, not prospective evidence.

GPU training-sentinel replay passed the existing 1e-6 relative RMS tolerance
(maximum restricted-output discrepancy 1.46e-7). The initial directional JVP
checks gave relative errors 3.48e-5 and 1.23e-4 at perturbations of 0.01 and 0.001
training-scale RMS. The evaluator took 14.19 seconds, about 16.20 seconds including
process startup, with 3.47 GiB peak allocated GPU memory. Its 85 model forward
calls exclude backward-pass work and should not be reported as equal inference cost.

Relative vector mismatch below means norm(predicted error minus actual error)
divided by norm(actual error); this is not the rollout's relative solution error.
Step one is exact in ideal arithmetic and excluded from skill scoring.

| Step | Relative vector mismatch | Forecast/actual error cosine | Prefix skill over cumulative forcing, steps 2--n |
| --- | --- | --- | --- |
| 4 | 0.4015% | 0.999992 | 0.999979 |
| 5 | 2.5640% | 0.999687 | 0.999418 |
| 6 | 33.8103% | 0.980755 | 0.889652 |
| 7 | 295.2075% | 0.540373 | -7.11670 |
| 8 | 1801.8618% | -0.360608 | -249.28289 |

At H32, forecast RMS/training scale is 9.13e24 versus actual 2.6984; full-horizon
skill is overwhelmingly negative. The nonlinear rollout is finite. Neither a
finite rollout nor correct small-amplitude derivatives rescue this linear
long-horizon forecast. Saved-array recomputation verifies the observed error,
identity-response cumulative forcing and reported norms.

The frozen signed probes at input 7 already have linear/odd response cosines
-0.526 and -0.540 at half/full forecast displacement. Remainders are about
1.06 times the linear response norm. Inputs 15/31 use enormous forecast
displacements: retain them as failed-extrapolation stress, not local physical
response evidence.

A separately recorded post-hoc numerical qualification on the same inputs
checked GPU grad/no-grad kernel differences and finite differences in the
forecast directions at inputs 0,4,5,6,7. Clean-output discrepancies are at most
3.78e-7 training-scale RMS. At inputs 5--7, finite differences with displacement
RMS 1e-4 of training scale match JVPs within 0.07%; at the full forecast
displacements, plus-remainder/linear norms are 0.258, 0.886 and 1.057.
This supports a finite-amplitude limitation rather than an evident JVP
implementation failure. It is not a complete actual-path error decomposition,
a physical-versus-learned response attribution, or a uniform stability bound.

Scientific consequence: the short-window response model adds substantial
information beyond clean forcing, but its valid amplitude range is decisive.
Resolve the transition near inputs 5--6 before extending methods or claiming
quantitative corrective design. No intervention is selected by this result.

## September 16: First Predictive Diagnostic Implemented; Upload Pending

The owner authorized proceeding with code in the chosen direction and selected
the personal workstation. The bounded first action is an offline clean-path
forced tangent calibration on Clean32, training seed `2026090611`, states
`0..32`, as recorded in the experiment plan. Existing outcomes make it
calibration evidence; a source/forecast freeze does not make it prospective.

New utility `forced_tangent.py` and evaluator
`evaluate_kolmogorov_tangent_forecast.py` preserve the complete deployed map,
ordered JVP propagation, cumulative-forcing control, separate nonlinear rollout,
signed finite-amplitude probes, and explicit failures. Their two focused CPU
test files pass together: 33 tests. These are implementation checks, not a
trained-model scientific result or CUDA derivative qualification.

The ignored `kf_tangent_20260916a/` packet contains the preparation script,
bound training inputs, a 13-file source bundle and staging receipt. Parent
manifests, fit-bound sources, selected reference blocks/states and checkpoint
bytes were checked before staging. Replay inputs retain training sentinels only.
The new evaluator also passed local real-input hash/shape/identity validation
without checkpoint execution. No historical packet or model source was edited.

Read-only workstation inspection found an idle GPU and a CUDA-enabled PyTorch
runtime. Automatic approval review then rejected the private payload upload,
requiring explicit approval for that transfer. A bounded owner question is
pending. No upload, GPU diagnostic, solver query, optimization, protected reveal
or external review occurred. GPU derivative validity and scientific outcomes
remain unestablished.

## September 16: Predictive Paper Plan Reset

The owner authorized documentation cleanup and a careful project plan before
actual work resumes. The current completion goal is a substantive PDE case
that diagnoses a response error, predicts a quantitative application benefit
and limit, chooses an intervention, and independently tests that prediction.
The old calendar, mandatory matrix and prepared adaptation queue are superseded.
Kolmogorov remains the leading candidate; the exact case and comparison,
replacement deadline and compute envelope are not yet agreed.

The [pre-reset snapshot](archive/pre_predictive_reset_20260916a/INDEX.md)
preserves preceding working-tree plans and navigation with their SHA256 values.
All earlier entries in this ledger remain unchanged. The ignored paper plan
was updated separately; manuscript LaTeX and scientific artifacts were not edited.

September 15--16 context inspection checked current source/manifest bindings
and selected result records. It did not rerun scientific analyses or rehash
every binary. Legacy C2 prospective ranking remains unestablished; the paired
fitter is prepared, with no local resource-payload launch receipt at inspection.
This documentation task runs no experiment, model, solver, remote cleanup or
external review and grants no protected-population access.

## September 13: Shared AutoDL Cleanup E Complete

The owner requested continuation while the other agent was active. E retained
the exact 79-file selection and restored B's admission of one frozen existing
GPU-job identity, using D's native local-checksum implementation. Every process
remained subject to the unchanged cleanup-scope conflict checks; new or changed
GPU identities would abort. No old attempt or allowlist was rewritten.

Independent review, 147 synthetic tests, native checksum interop and fresh
preflight passed. The guarded deletion completed at 16:36 China time. The
receipt and 160-event journal record 79 removals: 77 intermediate checkpoints
and two duplicates. Fresh read-only reconciliation at 16:37 finds exactly those
79 paths absent, all 3,202 retained file signatures and six links unchanged,
and the same other-agent GPU job still active.

The selected allocation was 18,700,922,880 bytes; observed net recovery was
18,700,894,208 bytes (17.42 GiB), leaving 19,662,733,312 bytes (18.31 GiB) free
on the work volume. The system partition separately has only 566,915,072 bytes
free. Shared caches and the other agent's tree were excluded.

The ignored `autodl_cleanup_20260913e/CLOSEOUT.json` binds the final evidence;
its plan maps every removal to a retained local recovery copy. Keep all partial
and complete backups, older attempt receipts and imported dependencies. The
original full cleanup plan is now stale and must not be resumed. No experiment,
compute migration, process interruption, commit or paper upload occurred.

## September 13: Cleanup C Safe Stop; Faster D Prepared Only

Owner-approved C used the same 79 backed-up targets and passed 97 synthetic
tests, independent review and fresh live preflight. Its deletion invocation
finished local recovery hashing, then stopped at the initial remote GPU-status
query timeout, before journal creation or any unlink. Reconciliation at 14:20
China time confirms all 79 targets, 3,202 retained file signatures and six links
unchanged. Another agent now has an active GPU job. No space was freed;
961,839,104 work-volume bytes remain free.

The ignored `autodl_cleanup_20260913c/CLOSEOUT.json` binds the exact evidence.
C exited itself; the attempted local stop helper failed before any signal.
Do not retry C or resume the original interrupted worker.

Slow Windows-mounted checksum reads dominated local verification. Prepared D
uses the existing native Windows interpreter for those reads only; the remote
deletion code is byte-identical to C. Its 129 synthetic tests and tiny real
native-interpreter check pass. D has no plan, preflight or launch. Coordinate
a fresh cleanup window before freezing another attempt. This is infrastructure
work, not a scientific result; no experiment or compute migration occurred.

## September 13: Shared AutoDL Cleanup Stopped Before Deletion

The owner authorized continued cleanup while another agent uses AutoDL.
Successor `artifacts/time_dependent_no/autodl_cleanup_20260913b/` selects 79
fully backed-up files: 77 intermediate checkpoints and two duplicates, totaling
18,700,772,590 logical bytes (17.42 GiB). Incomplete backups, final checkpoints,
datasets, the other agent's tree and shared caches are excluded.

Fresh local hashes, independent source review, 84 synthetic tests and live
preflight passed. The deletion invocation stopped after remote target hashing
when a new GPU job appeared, before journal creation or any unlink call.
Read-only reconciliation at 02:45 China time confirms all 79 targets, 3,202
retained file signatures and six retained links unchanged. Both old/new remote
cleanup journals are absent; no space was freed. Work-volume free space remains
961,839,104 bytes. This is an infrastructure safety stop, not a scientific result.

The packet's `CLOSEOUT.json` binds source, plan, tests, failure receipts and
reconciliation. Preserve both cleanup packets and their recovery copies. Do not
retry B or resume the old worker. A new exact attempt requires fresh ownership
checks and coordination of a no-new-job-start window; existing training can
continue. No experiment, migration, zw-gpu access, commit or paper upload was
performed. All 469 scientific sources and two checked paper sources remain
byte-identical to the pre-cleanup baseline.

## September 13: Local Pre-Sprint Audit And Pause

The owner paused experiments for local code/evidence inspection and cleanup.
[HANDOFF.md](HANDOFF.md) owns this checkpoint and [README.md](README.md) the
code/paper/retention map. Older forward queues below are historical context.

The initial local-only inspection found WSL stopped and the backup interrupted:
77/226 final-name checkpoint copies have expected sizes, plus one partial;
no backup-verification, deletion-start or completion receipt exists. Remote
state was not queried. Preserve the complete operation and dependency chain;
do not rerun its completion worker blindly. No new paired-resource probe or
adaptation fit launched.

The pre-cleanup README/handoff were archived byte-for-byte. The local cleanup
packet `artifacts/time_dependent_no/local_cleanup_20260913a/` records removal
of 87 regenerable cache files after verifying a recovery ZIP, and a reversible
same-volume archive of one generated transport-test tree (2,867 files,
517 directories, 22 links). Its before/after content-and-link tree hashes agree.
Sibling test XMLs, source and receipts remain in place. No scientific source,
checkpoint, dataset, result or paper file was removed; inaccessible scratch
was left alone. See the packet for inventories, exclusions and recovery paths.

## September 6 Successor: CM_NEXT_20260906A

Owner direction: plan and proceed toward a complete paper September 13; full
draft September 11 for review. The revised project/experiment/implementation
plans supersede older forward queues below. Existing result entries are
unchanged evidence.

| Item | Scope | Current state |
| --- | --- | --- |
| Common paired solver-response assay | Additive metrics and synthetic tests; no model/data access. | Complete; independent math/legacy-parity audit and 59 focused tests pass. |
| CM_NEXT_KF_R0_20260906A | New local N64/N128 solver-only screen; fresh states, two viscosities, no M1 retry. | Complete in 74.98 s; packet independently verified; N64 high-band spatial response remains unqualified. |
| CM_NEXT_KF_R0_20260906B | Same physical recipe at N128/N256, newly evolved anchors. | Complete in 543.66 s; short-anchor refinement, not population qualification. |
| Periodic PCNO / synthetic smoke | Explicit forcing, periodic gradient and shared numerical restriction. | 113 combined CPU tests pass; CPU fit and N128/N256 AutoDL smoke completed and audited. |
| CM_NEXT_KF_LONG_20260906A | Two fresh development trajectories at viscosity .01, with fixed refinement queries. | Complete, exit 0 in 848.89 s; full packet independently audited, all five declared limits met. |
| CM_NEXT_KF_DEBUG_FIT_20260906A | Seed 2026090603 first 32 pairs; fixed 256-update engineering fit. | Complete, exit 0; independent provenance/array/metric and five-input CPU checkpoint audits pass. Engineering evidence only. |
| CM_NEXT_KF_POP_20260906A | Eight fresh training and four development trajectories; frozen finite-time law. | Complete packet, failed spatial qualification. All twelve trajectories retained; no production promotion. |
| CM_NEXT_KF_PEAK_20260906A | Two native N256 early-window trajectories with N512 clean-query and independent-composition checks. | Complete in 24m41s; independent full-array audit passes all three declared limits. Not full-population qualification. |
| CM_NEXT_KF_POP_20260906B | Twelve unchanged roles at N256, with fuller numerical checks. | Closed incomplete after local standby exhausted elapsed budget: five complete training paths, one partial, no development paths. Packet audit passes; population is not qualified. |
| CM_NEXT_KF_POP_20260907C | Preserve B and recover its missing work on the existing AutoDL instance after cross-runtime replay. | Completed, retrieved and independently audited: all twelve trajectories and all three declared clean numerical gates pass. Physical/persistence analysis complete; model and displaced-input readiness remain open. |
| CM_NEXT_KF_CLEAN_RESOURCE_20260907A | Sixteen actual N256 batch-eight updates; resource evidence only. | Complete, retrieved and independently audited. Original source archive preserved; no checkpoint retained. |
| CM_NEXT_KF_CLEAN_20260907A | Fresh full-population Clean fit; fixed one-step competence/convergence checks. | Completed, retrieved and independently audited September 8. Terminal 49,152 passes the original one-step/engineering gates; the separately identified rollout is below. |
| CM_NEXT_KF_CLEAN_ROLLOUT_20260908A | Existing-checkpoint recurrence from all eight training and four validation initial conditions. | Completed at 15:17:56 China time, exit 0. All 45 files retrieved; local metric/state/provenance recomputation passes. All twelve paths reach the amplitude guard; H512 remains censored. |
| CM_NEXT_KF_TRAIN32_POP_20260908A | Add 24 independent training paths under the unchanged physical/initial law; preserve C's validation roles. | Completed at 19:16:23 China time September 8, exit 0; all 738 files retrieved. Independent full-array audit passes all three registered numerical gates. The index now has 32 training and four unchanged validation trajectories; no new model training. |
| CM_NEXT_KF_CLEAN32_MATCHED_20260908A | Fixed 49,152-update Clean fit on the audited 32-path union; original normalizer and unchanged validation. | Never launched. After owner-approved reviewer substitution, static review found blocking implementation/contract issues. The exact reviewed payload is preserved and superseded by the source-hardening identity below. |
| CM_NEXT_KF_CLEAN32_MATCHED_20260909A | Same scientific fit; output-isolation, deadline and failure-provenance fixes. | Owner-approved re-review completed September 9. One MAJOR decoder-exception gap and two MINOR provenance/test findings remain; four synthetic probes reproduce the two bugs. The 190-test suite and data validation remain valid but do not close these gaps. Exact payload preserved; no upload or scientific fit. |
| CM_NEXT_KF_CLEAN32_MATCHED_20260909B | Same scientific fit; decoder finalization, role-specific evidence and regression-test repairs. | Completed 49,152 updates September 9, exit 0; full retrieval, metric/history audit and fresh CPU checkpoint replay pass. Validation one-step error falls 1.6163% to 0.4808%; all-32 training is 0.2571%, original-eight training 0.2626%. Clean generalization materially improves; remaining gap is early-time concentrated. No new rollout evidence. |
| CM_NEXT_KF_CLEAN32_ROLLOUT_20260909A | B's fixed terminal on the same original eight training/four validation starts; unchanged 512-step horizon and guard. | Closed unlaunched after five review-required safeguards. Frozen source ZIP and all review/evidence records are preserved. Owner-approved repair successor is CM_NEXT_KF_CLEAN32_ROLLOUT_20260910B below; A supplies no rollout result. |
| CM_NEXT_KF_CLEAN32_ROLLOUT_20260910B | Owner-approved narrow safeguard successor; identical scientific experiment, separate identity. | Transport R2 passes 60 tests and Astra review. Launched at 16:19:24 and completed at 16:21:01 China time September 10, exit 0; all 45 files retrieved and local recomputation passes. All twelve paths hit the amplitude guard at steps 84--108. Matched H32 validation one-step improves 2.4484% to 0.6963%, but rollout worsens 35.2606% to 74.8571%. H128/H512 are censored. |
| CM_NEXT_KF_COMMON_RESPONSE_20260910A | Owner-accepted common-input diagnostic of both frozen maps. | Approved exact Astra review returned REVISE with three required safeguards. Undeployed A's source, validation, prompt and raw review are retained. |
| CM_NEXT_KF_COMMON_RESPONSE_20260910B | Narrow safeguard successor; scientific recipe unchanged. | Approved Astra re-review accepted direction/accounting repairs but required deadline checks inside sentinel replay. B's source/review/validation are retained; never deployed. |
| CM_NEXT_KF_COMMON_RESPONSE_20260910C | Sentinel-deadline repair only; closed helper and scientific recipe unchanged. | Owner-approved launch completed September 10 at 22:24:33 China time, exit 0. All 45 files retrieved; all 672 diagnostic rows independently recomputed. Early matched clean32-donor validation gain is 2.6075 versus 9.4406 for clean8/clean32, despite smaller clean forcing for clean32. This is direction-, amplitude- and time-dependent learned response, not a displaced-solver defect or causal rollout attribution. |
| AutoDL storage | Owner-directed preservation-first cleanup before further experiments; another agent is active. | E completed the exact backed-up 79-file subset; 17.42 GiB recovered and 18.31 GiB work-volume space free. System partition remains tight (541 MiB). Keep recovery copies; no old cleanup or experiment worker restarts automatically. |
| CM_NEXT_KF_COMMON_SOLVER_20260911A | Owner-accepted train-only solver-response qualification: first two training IDs, two early anchors, both fixed matched-RMS donors; 53 solver calls. | Completed September 11 at 10:43:31 China time, exit 0. All 12 retrieved files and 128 paired metric rows independently audited; all numerical gates pass. Early clean32-donor solver gain is 0.5552 versus learned 2.4444/7.9888. Two-path diagnostic only; no model inference or training. |
| CM_NEXT_KF_PAIRED_BANK_20260911A | First stage of the owner-accepted recovery/relabeling comparison: 512 training anchors, 1,024 signed Gaussian inputs, shared clean and displaced solver targets. | Completed September 12 at 01:07:15 China time, exit 0 in 64.25 min. All 520 files retrieved; independent identity, label, numerical and final-provenance audit passes. All-row A-response gain is 0.3166--0.3341. Fine-grid qualification covers twelve sentinel anchors, not the entire bank. No adaptation training or new rollout yet. |
| CM_NEXT_KF_PAIRED_ADAPT_20260912A | Matched single-seed clean continuation, recovery and dynamics relabeling from the clean32 terminal. | Trainer implemented; scientific/failure-handling review, 82 focused tests, 196 dependency tests and full train-only loader validation pass. No resource smoke, real-model inference, adaptation fit or new rollout yet. |
| CM_NEXT_KF_PAIRED_ADAPT_20260912A_RESOURCE_DYNAMICS | Separate 16-update resource check; no retained checkpoint or automatic fit. | Exact 22-file payload frozen; independent review and 127 transport tests pass. Preflight stopped on insufficient disk space. Owner-directed cleanup now precedes further work; nothing deployed. |
| New PCNO main comparison | Three paired seeds, common-input targets and explicit correctors. | Planned; no corrective-arm or comparison-matrix training launched. |
| Prospective confirmation | New ID population and prediction freeze; old protected roles excluded. | Not opened; exact reveal decision remains required. |

The local screen is not population qualification, a learned-model result or a
new primary-PDE selection.

### September 12: Preservation-First AutoDL Cleanup

The owner requested machine cleanup before further experiments. A fresh
inventory confirms 917.3 MiB work-volume free space and no active experiment.
Most occupancy is historical data/output, not cache. Do not classify an old
run as disposable solely by its age or assume that metadata hashes constitute
a backup. Independent audit found no intermediate checkpoint bytes in the
named local D094 metadata archives.

The ignored `autodl_cleanup_20260912a` operation freezes 350 exact files:
226 intermediate checkpoints from three closed Bump runs, two existing backed-up
duplicates, and 122 aged pip HTTP-cache entries. The 226 checkpoint backups
(34,747,021,174 bytes) are now transferring locally; no remote deletion has
occurred. Retain every best/last checkpoint, all 26 final sentinels, all other
run files and symlinks, datasets and the other agent's resource. All 1,161
current input bindings passed the initial hash check and must pass again after
cleanup. No checkpoint is deserialized and no protected array is decoded.

Independent cleanup review and **32 synthetic tests pass**. The repairs forbid
optimized local Python and enforce the completion wait's deadline even when
the backup exits during the final sleep. Neither path can bypass verification.
Deletion requires completed, freshly rehashed local recovery files, unchanged
remote hashes/inodes, no process references and an idle GPU. It uses only exact
nonrecursive file removal with a durable intent/completion journal; no automatic
deletion retry or experiment launch is included. `plan.json`, backup/authorization
and review receipts are the operation's source of truth. Plan SHA256:
`341f28fc78ea91d884e4af685788c3ef98b4a3b69a1b987a34c29e7377b9d28d`.

The local `preserve` process and separate `finish` worker are active. The latter
waits at most 9,000 seconds for the exact PID/start-time/argument identity, then
requires the completed backup receipt and fresh checks before deletion. Missing
or incomplete backups, PID reuse, timeout or a failed check stop it without
retry. Keep the local machine awake and connected. Do not edit the frozen
cleanup source or start another copy of either action. `startup_verification.json`
records their identities; `completion_finished.json`, `delete_receipt.json` and
`remote_events.jsonl` must be inspected when they exist. A
`completion_failed.json` instead requires diagnosis; never assume that cleanup
or disk recovery completed. No experiment launch is chained to this worker.

### September 12: Resource Payload And Storage Gate

The resource-only `dynamics` invocation is frozen without changing any of the
21 scientific source files. Its launcher enables GPU 0, retains the paired
batch-eight recipe, and uses a 30-minute outer timeout plus 60 seconds kill
grace. No checkpoint, scientific continuation or new rollout is included.
The experiment plan owns the fixed capacity requirements.

Independent review and **127 synthetic transport tests pass**. Review closed
two narrow issues: use the payload-pinned interpreter rather than a legacy
helper's interpreter, and retain a failed-packet receipt for malformed/truncated
sampler archives. Preflight programs travel over stdin to avoid large shell
argument limits. Approval, immutable capture, unique launch, all 1,161 bindings,
completion counters, sampler/history checks and checkout-independent retrieval
are tested. No scientific source or payload bytes changed in those repairs.

The read-only preflight stopped before upload on the disk gate. Current free
space is **961,839,104 bytes (917.3 MiB)** versus **1 GiB** required. A separate
diagnostic finds an idle RTX 5090, 10,139,197,440 bytes conservative effective
host headroom, matching Python/NumPy/PyTorch/CUDA and exact sampler prefixes.
It does not bypass the failed disk gate or qualify launch.

One possible remedy is the unselected `latest_checkpoint.pt` from the completed
clean32 fit: 133,484,852 bytes, with an exact locally retained backup. Fresh
local and remote hashes match its parent manifest, and no open-file holder was
found. The selected `terminal_049152.pt` is not this file and remains required.
At preparation close no remote file was removed. The ignored resource attempt's
`launch_request.md` asked for this exact single-file removal and conditional
upload/launch only after a fresh full preflight passed. The subsequent owner
request broadened cleanup under the separate operation above, not this frozen
experiment payload or its scientific recipe.

Exact source-only upload: **22 files / 503,085 bytes**. Payload manifest SHA256:
`85d0b3a592fee3cd457baeb37867f472a7b0ce0f607b87f4174d98dd76eb13da`.
The ignored `cm_next_kf_paired_resource_20260912a` attempt retains the source ZIP,
launcher, transport/test/review receipts, initial disk failure and read-only
capacity/backup diagnostics. Cleanup now precedes further work; no source upload
or GPU resource process has started. Disposable local test fixtures were removed;
their XML receipts and all real research artifacts are retained.

### September 12: Matched Adaptation Implementation

`fit_kolmogorov_paired.py` implements the frozen three-arm recipe without
changing closed fit sources. It retains the trained parent head and normalizer,
uses fresh Adam and two matched samplers, and applies the shared PCNO output
restriction. Recovery and relabeling read identical saved raw FP32 inputs;
only their A-target column differs. No fine-grid target is used for training.

Scientific and implementation reviews pass. Required repairs now count updates
only after device synchronization, check the deadline before terminal saving,
retain failure receipts if ancillary serialization fails, and promote a terminal
only after replay and final provenance succeed. Checkpoint/model identities stay
exact; repeated inference inherits Clean's per-input relative RMS limit `1e-6`.

**82 focused tests and 196 dependency tests pass.** Coverage includes full-budget
sampler tapes, matched gradients/Adam updates, forbidden reads, failure paths
and all three arms with an actual small N16 PCNO. The first dependency invocation
had 113 Windows temporary-directory access errors and no assertion failures;
the unchanged suite passed in a fresh directory outside the sandbox. Its failed
XML is retained, not counted as a scientific failure or passing test run.

The actual `validate` phase completed at **13:03:46 China time**, exit 0 in
**228.13 s** including final provenance. It checked all 32 training paths,
16,384 clean pairs, 1,024 signed bank rows and 21 source files. Its 1,161 input
bindings include **608 clean blocks**, 96 more than the bank-generation subset.
Decoded training stores occupy 5,108,670,464 bytes; the validator does not
write a second dataset copy. Every final source/input check passes. Model
construction, checkpoint deserialization, forwards, optimizer steps and solver
calls are all zero; validation/protected arrays and mixed teacher data are
excluded. Independent source/JSON review verifies the packet hashes, role/block
partition and zero-call accounting; it does not independently decode the arrays.

Source / test / validation-result / validation-manifest SHA256:

- `8f844fc902dd55d85def8c23676ace3afa4fb04c7fe2ee1306c83b52a76de8a2`
- `835aa1d4a1141759802285e6a97297f15373bbb9326f89c992b90c21f85ff523`
- `2a06e61312523dd49468d499e9a8dc6078119919f5881a4df4a0679b946d3867`
- `bf3ddd0329b2fdc35349e6e4a99cfed7c17a20922c0d7cbdbf95b36f1ad40267`

The ignored `cm_next_kf_paired_adapt_20260912a` attempt retains test XML,
`validate_r1` and `implementation_verification.json`. Disposable successful-test
scratch was removed; receipts and research artifacts remain intact. This closes
implementation/data-readiness only. Next is a separately frozen 16-update GPU
resource smoke, then exact scientific payload approval for three fresh
4,096-update fits. No remote access, upload, real-model inference, scientific
optimization, new rollout or paper edit occurred. No corrective ranking is yet
supported; freeze diagnostic predictions before evaluating new rollouts.

### September 12: Paired Gaussian-Bank Closeout

The exact approved CPU-only job completed all 512 anchors, 1,024 signed inputs
and 1,657 solver calls in **3,854.93 s**, including final provenance. Source and
inputs remained unchanged. All **520 files / 1,328,820,320 bytes** were retrieved;
transport integrity, completion and bank usability checks pass. The terminal
result, not the progress file's retained `running` label, owns completion.

The independent auditor checks the frozen source against the live checkout,
552 parent input bindings, deterministic Gaussian identities, every saved raw
input and A target, call accounting, registered gates and final hashes. A
separate direct-vector-norm calculation agrees on all-row target statistics;
a coefficient-space Parseval check agrees on 192 sentinel response/state
quantities. Neither local check reintegrates the PDE or calls a learned model.

| Quantity | Minimum | Mean | Maximum |
| --- | ---: | ---: | ---: |
| Realized input RMS, all 1,024 signs | 0.01058239 | 0.01070783 | 0.01081590 |
| A response gain | 0.316609 | 0.325971 | 0.334104 |
| Recovery/dynamics target separation, physical RMS | 0.00338332 | 0.00349043 | 0.00360027 |
| Target separation / inherited train-state scale | 0.00083263 | 0.00085899 | 0.00088602 |

Here A response gain is `RMS(A_j - A_0) / RMS(raw_j - raw_0)`.
Recovery uses the clean A successor; relabeling keeps the contracted but
nonzero A response to the same saved Gaussian input. These are sampled secants
of the declared coarse solver map, **not identified manifold-normal gains**.
They establish a resolved target contrast, not which objective improves PCNO
rollout or a causal explanation of the earlier ranking reversal.
Antithetic signs are not independent replicates. A null training comparison
would remain conditional on this augmentation law, budget and single seed,
not close either corrective-method family.

| Registered check | Maximum | Limit |
| --- | ---: | ---: |
| FP32 rounding / intended displacement | 2.39494e-5 | 1e-4 |
| Projection / displacement | 2.53660e-5 | 1e-4 |
| Projected direction change | 1.78245e-5 | 1e-4 |
| Lift roundtrip relative L2 | 4.20930e-16 | 1e-11 |
| Native replay relative L2 | 2.66423e-16 | 1e-10 |
| Coarse temporal response / displacement | 1.54461e-5 | 1e-3 |
| Fine temporal response / displacement | 5.45587e-8 | 1e-3 |
| Spatial response / displacement | 0.00835453 | 0.01 |
| Discarded fine response / displacement | 0.00688310 | 0.01 |
| Spatial state relative L2 | 5.32683e-5 | 1e-3 |
| Discarded fine state relative L2 | 6.71540e-5 | 1e-3 |
| Clean A versus archived successor / displacement | 5.35501e-6 | 1e-3 |

Input/clean-label checks cover the whole bank; A/B/C/D refinement and native
replay cover **twelve anchors / 24 signed queries**. The largest spatial-response
ratio uses 83.5% of its limit. Summed response-refinement sensitivity reaches
2.56% of the A response on sentinels; this is not a certified PDE error bound
or uniform qualification of all bank labels. A labels remain the learner
targets; no fine-grid answer is substituted and no noise scale is changed.

The first native-Windows audit stopped on an exact intended-FP64-noise hash,
despite matching saved FP32 inputs. The original and independent FFT formulas
agree locally; the unchanged reconstruction matches remote bytes under the
existing Linux/WSL runtime. The **unchanged auditor then passed all 512 anchors
and final hashes in WSL**, without relaxing identities, tolerances or gates.
The failed receipt is preserved; platform-dependent FP64 reconstruction is an
audit-portability issue, not a failed solver qualification or a bank repair.

The ignored attempt retains `analyze_bank.py`, `analysis_bank.json`, the native
failure, review and execution receipts. Final manifest / result / analysis SHA256:

- `58adafa766985007556676f78440a1e0ada85793ecd61f898a6c868deb7adeed`
- `93f4c2378b7a57a99aa4e6272a2ebf184eadb985df1c716c37d61ff0a6866c1d`
- `c2e9e3744b21d6a0590091a8628c4ac8dc175349b2ffec7c9d1baf72b7ccc87e`

At bank closeout, the next step was the frozen single-seed, three-arm adaptation:
anchor-reweighted clean continuation, paired recovery and paired dynamics
relabeling. Source review and synthetic sampler/gradient prechecks were complete,
but the trainer was not yet implemented; the later implementation entry above
supersedes that status. The implementation plan records
the model-only checkpoint load, train-only loader and matched-budget details.
Review, resource smoke and exact payload freeze precede scientific training;
diagnostic prediction freeze precedes its separate rollout evaluation. No model
inference, optimization, new rollout, protected access, cleanup or paper change
occurred in this bank closeout. Do not relaunch the completed identity.

### September 11: Paired Gaussian-Bank Preparation

The owner accepted the matched recovery/relabeling successor. Its experiment-
plan amendment fixes the 32-path bank and the later single-seed, three-arm
adaptation, including an anchor-reweighted clean-continuation control. Only
the bank generator is implemented at this stage; no corrective model is fit.

Native Codex review found and closed one compatibility issue: population C's
immutable manifest lacks a status field. Its completed result is now required,
while other manifest schemas still require status. The parent is unchanged.
The 61 focused tests and 121 closed dependency tests pass. End-to-end synthetic
generation uses a 33-call N16 fixture; production accounting is 1,657 calls.
Ruff and `git diff --check` pass. Current generator/test SHA256:

- `14e26637f1942d47c6404f75fa31746cac0818573a0ba56ea748986d1133b24a`
- `13d2910e5ce9b3d0449c9178dbc5e6f166c8ff628a763ae8802650dbc62239d8`

The first actual validate-only run stopped after 432 anchors on a Windows
progress-file replacement PermissionError, with zero solver/model calls and
no numerical gate failure. Its partial packet remains intact. The unchanged
validator completed in a fresh directory outside the sandbox in 538.91 s:
512 anchors, 1,024 signed inputs, 552 input bindings, twelve source files and
stable final provenance. An independent NumPy reconstruction matches every
saved signed FP32 input exactly. Realized Gaussian RMS is
`0.0105823865--0.0108159010`; ensemble mean-square / prescribed variance is
`1.0001369999`. Maximum rounding and projection ratios are `2.39494e-5` and
`2.53660e-5`, both below the frozen `1e-4` limits. No scale was adjusted.

Successful validate-only result / manifest SHA256:

- `5b80914ef4fc4d4df81b2f8a772cc525ee0f1ec5573ec63f2354154a6e601d05`
- `24f740f8c544423e3095d3cf570460564988e05e3cfd0223fabecb50eb649c1c`

The ignored attempt directory retains validation packets, test XMLs and
`input_validation_review.json`. Disposable synthetic fixtures were removed;
test reports and research evidence are preserved. The source-only transport
also passes independent review and 47 exact-byte tests. Review closed a
per-anchor filename-alias gap; a fixed 900-second SSH read timeout accommodates
the larger parent hash pass. The NTFS-backed test run was interrupted as an
engineering check; the complete final suite passed on Linux temporary storage.

Local payload verification and read-only AutoDL preflight pass: all 552 input
bindings match, the frozen Python/NumPy runtime is unchanged, the GPU is idle
and 2.1346 GiB is free. Transport uses the existing configured WSL environment;
native Python lacks its SSH dependency. No package installation or remote
cleanup was performed. The exact upload contains twelve Python files plus one
launcher, **210,937 bytes**; no data, checkpoint or paper is uploaded. Payload
manifest SHA256:

`0aa620da33db96b6b80a67de7e603c8ba8a25964b0761375ffab14f1258763f1`

The ignored attempt owns `payload_for_approval.json`, source ZIP, launcher,
`code_review.json`, `transport_verification.json` and the preflight receipt.
At preparation close, exact owner upload/launch approval was still pending;
none was inferred from preparation. A completed validate-only packet is not a
qualified label bank. The prior pilot's timings suggest roughly 1.5--2 h for generation,
subject to late-state cost variation. The work cap is three hours; the outer
watchdog allows 11,700 seconds plus a 60-second kill grace. No automatic retry,
training or scientific expansion is attached to this launch.

The owner's following reply approved that exact upload and launch. All 229
recorded tests, review receipts, validation result/manifest and payload bytes
were reverified. Fresh pre-upload and post-upload checks passed: all 552 input
bindings, unchanged Python 3.12.2 / NumPy 2.5.0 / Linux, no conflicting compute
process and 2.1344 GiB free. The CPU-only job launched at **00:02:59 China time
September 12**. Initial authenticated status showed seven completed solver
calls, zero completed anchors, a live launcher and no reported error; the first
anchor includes the expensive B/C/D refinement sentinel. Expected finish is
**01:35--02:05 China time September 12**, not a completion guarantee. The
healthy job is left running without continuous monitoring.

The ignored attempt retains `deployment_approval.json`, `launch_receipt.json`
and `initial_status.json`. Approval / launch receipt SHA256:

- `a306aae520ae4910631addc772c2bf734a40bebb0f1538ba6d7f8294771ed80f`
- `a99558cae24b75501526aec6738e642d61fcfc7f47d01cb2938967fabdb250c6`

The September 12 closeout above supersedes this initial running status and ETA.
It records complete retrieval and independent audit before the separate
training stage. No model inference, training, new rollout, protected access,
cleanup or paper change occurred. Do not relaunch this identity.

### September 11: Train-Only Solver-Response Qualification

The owner accepted the bounded next diagnostic. The experiment plan owns the
fixed four-anchor/eight-displacement selection, 53 solver calls and numerical
limits; no scale tuning, model inference, correction training or protected
population is included. A native Codex protocol audit and a separate static
implementation review pass. All **49 focused tests** and **82 solver/metric/
budget dependency tests** pass. The first focused test invocation lacked its
scratch parent; the unchanged rerun passed. Both receipts are retained.

Actual-input validation completed in **8.22 s**, with zero solver/model calls.
All 15 read-input bindings, ten source files and seven validation-packet files
were independently rehashed. Maximum input projection/displacement ratio is
`2.7933e-5`, displacement-vector change is `2.3022e-5`, and lift/restriction
roundtrip is `4.8099e-16`; these passed the frozen input gates. Solver-response
qualification had not yet run at this validation stage.

The ignored attempt root `cm_next_kf_common_solver_20260911a` retains the
validation, test XML, native review and source-only payload. It contains ten
Python sources plus the CPU-only launcher: **154,288 uncompressed bytes**.
Source ZIP SHA256 is
`7d63cf3de085c3fe094d5c7e4bc0a4468727a3f1ec344d0141e73792363b77c7`;
payload manifest SHA256 is
`170d31115767203447f7b9933e31a720678f23734a80b482c5e6ee5fde1b1529`.
All **74 transport tests** and an independent static transport review pass after
metadata/source-containment repairs. Read-only AutoDL preflight passed September
11 at **03:17:48 China time**: no conflicting compute process, matching Python/
NumPy/Linux runtime, and both manifests/results plus all twelve selected members
verified. Free storage was 2.23 GiB; raw cgroup headroom 5.43 GiB and conservative
effective host-memory headroom 10.70 GiB. No cache manipulation or cleanup was
performed. These live checks were repeated immediately before launch.

`code_review.json` and `transport_verification.json` bind the final code, tests,
payload and preflight receipts. The owner approved the exact upload/launch in
the following reply; `deployment_approval.json` preserves that scope. All 205
test receipts and final source/payload bytes were reverified before deployment.
Fresh pre-upload and post-upload checks passed, and the single CPU assay launched
at **10:28:17 China time September 11** and completed at **10:43:31**, exit 0,
in **913.59 s**. All **53 solver calls**, four anchors and **128 paired metric
rows** completed. All **12 files / 101,831,076 bytes** were retrieved and their
hashes verified. The original runtime was retained: Python 3.12.2, NumPy 2.5.0,
Linux. No model inference, correction training, protected access, cleanup or
paper change occurred. This attempt is closed; do not relaunch it.

Independent local recomputation from saved arrays passes, using a separate
Fourier implementation and scalar formulas, without importing the scientific
solver, model or metric code. It binds the source ZIP, launch/retrieval receipts,
parent manifests and selected inputs; checks all 53 call identities, all 128
rows, raw/deployed outputs, full-fine physical diagnostics and Fourier energies;
and rechecks input hashes after analysis. Numerical comparisons use
`rtol=2e-10, atol=2e-12`; gate decisions use the unchanged frozen limits.
Independent code-only analyzer review passes after tightening completion
consistency: input-geometry/native-replay hard stops must have passed, whereas
later refinement failures remain legitimate completed negative results.
Synthetic mathematics, five early-stop rejection cases and six later-negative
cases pass. The first analysis receipt is preserved as
`analysis_before_validator_hardening.json`; the final rerun reproduces every
numerical result exactly. Terminal `result.json` and exit status own completion;
`progress.json` retains the last pre-finalization running snapshot.

| Numerical check | Largest recomputed value | Frozen limit |
| --- | --- | --- |
| Input projection / displacement | 2.7933e-5 | 1e-4 |
| Displacement-vector change / displacement | 2.3022e-5 | 1e-4 |
| Lift/restriction relative L2 | 4.8099e-16 | 1e-11 |
| Original-FP64 replay relative L2 | 2.6429e-16 | 1e-10 |
| A/B response difference / displacement | 6.0708e-5 | 1e-3 |
| C/D response difference / displacement | 2.3585e-7 | 1e-3 |
| B/C response difference / displacement | 9.4277e-4 | 1e-2 |
| Discarded full-fine response / displacement | 8.3899e-4 | 1e-2 |
| B/C state relative L2 | 2.1757e-5 | 1e-3 |
| Discarded full-fine state relative L2 | 2.8368e-5 | 1e-3 |

All eight query-level qualifications pass, and the same-process repeat is exact.
The largest summed response-refinement indicator / displacement is `9.4621e-4`.
Every response defect exceeds its indicator, and every recipient-defect gap
exceeds twice that indicator (minimum ratio 111.0). This is numerical sensitivity
evidence, not a certified continuum-error bound.

The following are arithmetic means over the **two ordinal training paths**,
using the **deployed restricted model maps** and finest restricted solver map D.
The displacement RMS is unchanged at `0.00263502` in training-state units.
Response gain is output-response RMS / input-displacement RMS; response-defect
gain is the RMS of learned minus trusted response / that same displacement RMS.

| Output step | Direction donor | Solver gain | clean8 gain | clean32 gain | clean8 response-defect gain | clean32 response-defect gain |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | clean8 | 1.0640 | 1.2577 | 1.8051 | 0.6356 | 1.3058 |
| 8 | clean32 | 0.5552 | 2.4444 | 7.9888 | 2.3427 | 8.1411 |
| 32 | clean8 | 1.0170 | 1.2698 | 1.6677 | 0.8442 | 1.3040 |
| 32 | clean32 | 0.9661 | 1.1971 | 1.5354 | 0.8034 | 1.2061 |

1. **The large early response is not faithful PDE amplification on these
   inputs.** For the step-8 clean32 donor, solver gains are 0.5506/0.5597 on
   the two paths, while clean32 gains are 8.0062/7.9715. Its learned/trusted
   response cosines are -0.2211/-0.2630. Its mean response-defect gain is
   **247.50% larger** than clean8's. Mean displaced-dynamics error is
   **2.1716% versus 0.6916%** in training-state RMS units, a **214.02% increase**.
2. **Not all displaced response should be suppressed.** The step-8 clean8
   donor has solver gains 1.0178/1.1101. At step 32 the responses are closer to
   unit gain. Even the early contracting response is nonzero: recovery and
   dynamics relabeling impose different targets, whose value must be tested.
   These are sampled finite-amplitude responses, not identified normal dynamics.
3. **Keep the local clean-error comparison honest.** At step 8 the mean clean
   defects are **0.3133%/0.3406%** for clean8/clean32; at step 32 they are
   **0.6251%/0.9289%**. Thus clean32 is worse on average on this fixed two-path
   subset, unlike the broader clean-error comparison. This pilot identifies
   response error; it is not a new aggregate clean-error/rollout ranking reversal.
   clean32 has larger response defects on all eight paired queries, but two
   trajectory IDs are not independent model-training seeds.
4. **The error channel is structured, not an identified manifold normal.**
   At step 8 with the clean32 donor, 99.9797%/99.8873% of clean32 response-defect
   energy lies in `12 < max(|kx|,|ky|) <= 85`. Across all queries, raw/deployed
   response gains differ by at most `5.7383e-4`; amplification survives the
   existing numerical restriction. This locates a correction candidate but
   does not establish its architectural cause or justify filtering this entire
   band, which also contains physical response.

**Next proposed decision:** freeze a paired recovery-versus-dynamics-relabeling
pilot on identical training-only displaced inputs, with equal clean-data and
optimization budgets. Retain clean forcing, response defect and signed alignment
as separate readouts, then test held-out rollout effects under a separate freeze.
Use the response structure to motivate a targeted explicit corrector control;
do not assume it beats relabeling or that global suppression is safe. The current
four-anchor qualification does not certify an expanded label bank. No bank
expansion, corrective training, prospective reveal or paper claim is authorized
by this diagnostic result alone.

The ignored attempt retains `analyze.py` and `analysis.json`: the former is run
locally with no arguments to create an exclusive report, or `--self-check` for
synthetic mathematics only. It is kept to reproduce the independent audit and
all per-path/raw/deployed tables, not as a new general-purpose analysis API.
The final analyzer is `9ce12b6adb421244812bccbf3de0a360877f79cec052d80e970cd64dac68ba6b`;
its report is `670d896ed1aeed966278874bb30ee3a729f8f8a30f98a77154753c6b88906a0f`.
All ten current scientific sources still match the deployed snapshot. No test
scratch, scientific evidence, unrelated work or remote resource was removed.

- Result: `5b61f36b7b5a8e811b37474d2a7b5b1577ce98e5d082b8c8270d57de358d2fd0`;
- manifest: `1620aafac5a8d27e8690dbe2c901a27b09942afabe7eaeca289a03b39e297051`;
- retrieval inventory: `4b074338446b77c7bfe350c720a76ee6e640610691647ec5527c50c7c8c5ebba`.

### September 10: Common-Input Response Result (C)

The owner approved the exact C source-only upload and launch. The local-only
transport passed **62 synthetic tests** and an independent Codex static review;
the retained 57 scientific tests and both passing scientific reviews remain
unchanged. Fresh resource, runtime, source and input-provenance checks passed.
C launched at **22:22:51 China time**, completed at **22:24:33**, and exited zero:
**100.63 s**, **386 forwards**, two model constructions, **672 rows**. All 45
files were retrieved. There were no solver calls, optimization steps, new
rollouts, geometry fits, corrections or protected reads. Do not relaunch C.

The primary integrator independently recomputed the main response metrics,
Fourier energies and physical diagnostics from all saved arrays, without a
model: all 672 rows pass (`rtol=2e-10`, `atol=1e-11`). Sentinel and own-donor
natural-snapshot replays have zero error. All numerical input arrays equal
the pre-inference validation bank bit-for-bit. Full NPZ files differ only in
**36 ZIP creator-platform bytes** (Linux versus Windows); changing those bytes
in memory reproduces the validation archives exactly. This is not full-archive
byte equality and does not change a query or its calibration.

The matched displacement RMS is **0.00263502** in the fixed training-state
scale, calibrated from all 16,384 expanded-training pairs and no validation
pairs. The table reports arithmetic means over the **four open validation
paths**, for the **deployed restricted map**, not raw PCNO. Output step `k`
uses the reference input at `k-1`.

| Output step | Matched direction donor | clean8 response gain | clean32 response gain | clean32 larger, paired paths |
| --- | --- | --- | --- | --- |
| 8 | clean8 | 1.1660 | 1.3642 | 4/4 |
| 8 | clean32 | 2.6075 | 9.4406 | 4/4 |
| 32 | clean8 | 1.1233 | 1.4205 | 4/4 |
| 32 | clean32 | 1.0805 | 1.2182 | 4/4 |
| 64 | clean8 | 1.0204 | 1.0200 | 1/4 |
| 64 | clean32 | 1.0538 | 1.0559 | 3/4 |

1. **Clean accuracy and displaced-input response separate.** At step 8 on
   clean32-donor directions, clean32 reduces mean clean forcing from 0.0178073
   to 0.00437226 (**75.45% lower**) but increases mean response gain by
   **262.06%**. Mean error against the clean successor rises from 0.0191628 to
   0.0252724 (**31.88% higher**, all four validation paths). These errors are
   RMS divided by the fixed training-state scale, not target-relative L2.
   The gain ordering also holds on all eight original training paths
   (means 2.5195 versus 8.3039); it is not confined to held-out inputs.
2. **Gain alone is not a replacement score.** For the other donor at step 8,
   and both matched donors at steps 32/64, clean32 has smaller total
   path-recovery error on every validation path. Its smaller clean forcing
   outweighs the response difference. Keep forcing, response and their signed
   alignment together. Matched gains nearly coincide at step 64, whereas
   natural-amplitude gains remain different: do not pool away amplitude/time.
3. **The response channel is measurable, not yet physically classified.**
   For the step-8 matched clean32-donor validation bank, 98.92% of input energy
   and 99.82% of clean32 response energy lie at
   `12 < max(|kx|,|ky|) <= 85`. Raw/restricted mean gains are almost equal
   (9.4413/9.4406): the amplification survives the existing numerical filter.
   These Fourier bands are not proven normal directions. Raw clean errors
   differ substantially; the table is not a raw-PCNO performance claim.

This single-training-seed, retrospective assay distinguishes learned responses
on identical inputs. It does **not** measure true displaced-state dynamics,
response defect, manifold drift, a stability tube or prospective C2 performance.
The early clean train/validation gap remains a limitation. Gain above one is
not by itself unphysical, and this assay does not causally explain the complete
rollout reversal.

**Next decision:** qualify trusted solver continuations on a bounded train-only
subset of the same early common directions, before comparing recovery with
dynamics relabeling. The solver response must distinguish physical amplification
from model response error before suppression is called useful. Freeze that
recipe separately; no new solver labels or correction matrix are launched here.

Ignored `cm_next_kf_common_response_20260910c/analysis.json` owns all 48 stratified
groups, pathwise gain ranges/SDs, raw/restricted band accounting, independent
recomputation commands and receipt hashes. Path SD is not model-seed uncertainty.
The local transport/test and analysis receipt are retained for reproducing and
auditing this one attempt. Cleanup removed only 2,657 regenerable local test
files (1,061,300 bytes) and 11 test links; source and passing XML remain. No
scientific packet or remote evidence was removed. Paper, Git index and other
agent's resource are unchanged.

- Result: `7721c52862154292376e1c061b3bc9a5be7ff332f410de542989037f310605ff`;
- manifest: `0a5d4cb2563f6d873a0c315bc6e0469ad71b8cd68796616cfedf74190a5e033f`;
- launch receipt: `8f07192454550c0b1433626c0c9db20bab2fffa50772f4f06f8bc74030e8d476`;
- retrieval verification: `a4091bb4c404974dca4e750becd21679f487f1724ab1aecaed5414f1aa7576e2`;
- local analysis: `9324ccc7929a9d809d66a4e15d61c83ba4b9761eea235d2d7996f53c12ef67a8`.

### September 10: Sentinel-Deadline Re-Review And Repair

This is the preserved review-only checkpoint; subsequent deployment and results
are recorded above.

The owner approved B's exact 105,416-byte prompt for GPT-6 Astra xhigh.
Review completed at 20:21:31 China time, exit zero, with no tool calls. It
accepted the float32-vector and execution-accounting repairs, and found the
visible scientific logic consistent with the protocol. The verdict is still
**REVISE**: the immutable sentinel helper can finish decoding or transfer after
expiry and then begin the next expensive stage. An outer caller check misses
that boundary. No scientific job was launched.

The primary integrator reproduced both cases: one model invocation occurred
after expiry. `CM_NEXT_KF_COMMON_RESPONSE_20260910C` adds deadline-aware sentinel
orchestration in the new evaluator, preserving the closed helper's equations,
zero-denominator rules and `1e-6` tolerance. The test fixture now uses real
synthetic sentinel files. All 57 focused tests pass without warnings, including
decode/transfer/forward expiry, six closed-helper parity cases, and the complete
672-row/386-call packet test. Scoped lint and diff checks pass. The 127 unchanged
dependency tests were not rerun; their earlier results remain historical evidence.

Actual-data validate-only passes in 124.49 s, including 9.55 s final provenance,
with zero model/solver calls. Separate hashing/comparison verifies all five input
bindings, 31 current sources and byte-identical bank metadata/files versus B:
96 natural queries, 72 matched queries and 24 unresolved zero directions.
Only the evaluator/test changed; all other 29 sources remain byte-identical.
The evaluator's other function bodies are unchanged; only `run` routes to the
new replay function. This validation does not test real CUDA inference.

Ignored `cm_next_kf_common_response_20260910c/verification.json` preserves the
pre-review receipts. The owner subsequently approved the **31,202-byte** exact
prompt and further same-scope, code-only Codex review rounds without per-round
consent. Its 13 excerpts and 31 source files were rehashed before transmission;
no data arrays, checkpoints, scientific outcomes, manuscript, hosts or credentials
were sent. Astra returned **PASS, no required revisions**, at 21:51:54 China
time September 10, OS exit zero and no tool calls. A complementary native Codex
audit also returned PASS for query pairing, train-only calibration, population
mapping, metric semantics and 168-query/672-row/386-call accounting. Both reviews
are static. Optional extra test coverage was deferred; source bytes did not change.

`code_review_astra_completion.json` and `protocol_review_codex.md` retain those
reviews. `payload_for_approval.json` freezes 31 source files plus `run.sh`
(643,262 upload bytes), reusing the five input packets. Local Bash syntax and
command/environment/hash checks pass. The work/outer/grace limits remain
1,800/2,100/60 seconds. `payload_verification.json` owns the final local checks;
the 57 tests were rehashed, not rerun in this review-only turn. Exact upload/launch
approval, transport verification and fresh resource/runtime/provenance checks
remain. No AutoDL access, scientific inference, solver call, protected read,
paper edit, commit or push occurred.

The preceding repair's cleanup removed only two completed scratch directories (346 regenerable
synthetic files, 20,578,905 bytes); both reproducer packets were moved and
rehash-verified unchanged. XMLs, reviewed A/B archives and C's bank remain.

- B raw re-review: `4440e149ec17e1501c1836087c000df4a7a4fb960b9ca610f17abe03ea7a004c`;
- C narrow prompt: `40e7939295733085c1a3fe5c011b7fcd7ac81e781dcf026064365de7bd4e1ba3`;
- C source archive: `dc11e3d86036aba6cbda6152ad3781579c8c19e700245ccc1e31ee30f4bce328`;
- C raw passing review: `b049fa88db9803faedd2f6540c0dd2a3693e82bee38739af275748438b454598`;
- C review receipt: `dd6037110dad99a3d0f918ac1073a8b7e904850ae878b1c81db9997b7465cb7b`;
- C proposed payload: `ae41abe3eca352cc681eaafa960bfe75b2b93c7e47312bdaef0902c675c62a9f`;
- C validate result: `114bde9027ed5823373b54094263f4c7a744855dd6f9c8468ce764657ddfaa8a`;
- C validate manifest: `6b0d8a187122c49a1e402b83ef0d1495e54746d3d9c9399e49d9a52d09eedd9f`.

### September 10: Common-Input Review And Safeguard Repair (B)

The owner approved A's exact prompt for GPT-6 Astra xhigh. The static review
completed with exit zero and no tool calls, but returned **REVISE**, not approval
to deploy. It identified float32 direction distortion missed by an RMS-only
check, missing intermediate deadline checks, and undercounted model work in
failure packets. The primary integrator reproduced six failing regressions
before repairing the evaluator/test under `CM_NEXT_KF_COMMON_RESPONSE_20260910B`.
The same `1e-4` numerical tolerance now also bounds full-vector rounding error.
Unsigned-safe snapshot ordering and saved-array metric reconstruction address
the two optional comments. The other 29 source files remain unchanged.

Verification in ignored `cm_next_kf_common_response_20260910b`:

- 48 focused CPU tests pass without warnings. The combined 175-test run has
  174 passes and one Windows `PermissionError` replacing `progress.json.tmp`;
  that test passes on a fresh-directory retry. Retain both outcomes. The failed
  packet correctly preserves 145 actual model invocations. This is a synthetic
  infrastructure failure, not a scientific result. Lint and diff checks pass.
- Actual-data validate-only completes in 216.39 s, including 8.37 s final
  provenance. All five input bindings and 31 sources pass; no model construction,
  inference or solver call. Separate recomputation confirms all 12 bank files
  and their arrays are byte-identical to A, with unchanged calibration and
  96 natural / 72 matched queries; 24 zero directions remain unresolved.
  Maximum relative full-vector rounding error is `2.65e-5`, below `1e-4`;
  maximum relative RMS error remains `1.87e-7`.
- Cleanup removed only four completed scratch directories (3,886 regenerable
  synthetic files, 165,548,691 bytes), after containment/link checks. The 26-file
  failed packet was moved to `infrastructure_failure_r1` and rehashed unchanged.
  All XMLs, source archives, review records and validated banks are retained.

The revised **105,416-byte** `CODE_REVIEW_PROMPT.md` contains code/tests and
11 narrow excerpts from eight files, including the previously omitted loader
and deadline helper. It excludes scientific outcomes, data arrays, checkpoints,
paper, private hosts and credentials. The owner subsequently approved this
exact B prompt; its re-review and C successor are recorded above. Exact payload
approval and fresh capacity checks remain before deployment. No AutoDL job,
upload, remote cleanup,
protected read, paper edit, commit or push occurred in this repair turn.

Exact receipts remain in the ignored A/B folders; B's `verification.json` owns
the consolidated hashes. Key bindings:

- A raw review: `a5366b70d7916ca6c908b178dbb8de7ca1d10c785d4d06c7327be86a9a50a043`;
- B review prompt: `eb70145eb7d8ec2e0e47e042751257550b9a1e52729fd7a0930af1c3941a9d2a`;
- B source archive: `d61aaeedf8c925ffae442e96b8c5108148d8cc7d247104c3f46f463f0ee8c720`;
- B validate result: `d88de0faf022f3135b0a98a6a6bd79a0537e2cf2a8e2f5e890e4e325bf00c944`;
- B validate manifest: `60679287c89e0ff8841b00f09a15fe773267593373b9c8c0575c57ea027ed455`.

### September 10: Common-Input Diagnostic Preparation (A)

The owner accepted the proposed diagnostic. The experiment plan now owns the
bounded protocol, and the implementation plan names the separate entry point
and test. The primary integrator added only those two maintained code files;
all 27 closed rollout/fit-bound sources remain unchanged. No correction matrix,
solver-response qualification, protected reveal or paper edit is implied.

Verification in ignored `cm_next_kf_common_response_20260910a`:

- 38 focused synthetic CPU tests pass in 10.28 s. They cover identical recipient
  inputs, natural-input preservation, RMS/direction control, train-only calibration,
  zero directions, signed alignment, snapshot/checkpoint replay guards, output
  isolation and failed/partial packet handling. One test-only tensor-to-scalar
  warning remains. The first invocation had 10 passes and 28 setup errors because
  the scratch parent did not exist; its XML is retained, not counted as a science
  failure. The corrected invocation used a fresh scratch path.
- 127 unchanged dependency tests pass in 112.34 s: path metrics, both rollout
  evaluators/validators and the periodic adapter. Scoped Ruff E/F/I (E501
  excluded) and `git diff --check` pass.
- Actual-data validate-only passes in 76.70 s, including 6.16 s final provenance.
  All five input bindings and 31 source files pass. It creates no model and makes
  no solver call. The 15-file bank packet is 63,925,333 bytes.
- The bank has 96 natural and 72 matched-RMS queries. The 24 zero directions at
  the initial input remain unresolved for rescaling. Calibration uses 16,384
  training pairs and zero validation pairs: physical RMS `0.0107071767634698`,
  or `0.002635020767694896` of the fixed training-state scale.
- Separate primary-integrator recomputation rehashes the packet/current sources,
  matches all reference anchors to C, checks both donors' natural inputs, and
  recomputes training-only calibration and rescaling. Maximum matched-RMS relative
  rounding error is `1.87e-7`. This is not an independent external review.

No scientific response result exists yet. The planned assay has 672 raw/restricted
metric rows and, for this validated bank, 386 model calls including the two
terminal sentinel batches. It is retrospective and cannot establish prospective
ranking, true normal dynamics or off-state solver fidelity.

The exact 82,461-byte `CODE_REVIEW_PROMPT.md` was subsequently approved and sent
to GPT-6 Astra xhigh; see the repair entry above for its verdict and successor.
It contains the new evaluator/test, scoped protocol and narrow dependency
excerpts from eight files; no numerical outcomes, arrays, checkpoints, paper,
private host information or credentials. A was not uploaded or deployed.
The cleanup pass removed only the two new test-scratch directories (3,312
regenerable synthetic files, 124,618,597 bytes), after absolute containment and
link checks. All XML reports, the validated bank and prior research evidence
remain. No commit or push occurred.

Bindings:

- review prompt: `fc36a2a5a2b4a8ebf768ca93f3dd91ececae842dd3bf544447b6180a2013205d`;
- validate result: `93ebb6ebac10f2dd4433b216f921fcb4facd34be0e078a768613b9ddf7a02953`;
- validate manifest: `b3376d801157b008d170703eca876788fc93e7ed9ef3384dee215ce5f86de09e`;
- focused test XML: `a253c83b74d678a5fa690d5e0e5601673422ff77675a0c971eb5c9bf1453c82d`;
- dependency XML: `66e2f0be3f909dc1c27d88979274c9cfa6eef6bbf71fb6d0c0858b3dd6e1b90f`.

### September 8: Owner-Directed Clean Generalization Study

The owner clarified that the paper needs small, comparable training and
validation one-step errors, followed by an independently measured rollout gap.
The mentor supports Kolmogorov flow but leaves the specific regime provisional.
The approved next study separates clean coverage from physical-regime tuning.
The experiment plan owns the exact preparation-job and fixed-update coverage
contracts. Neither the new 32-trajectory fit nor the correction matrix is launched.

Reaggregation of the immutable `teacher_049152.npz` (SHA256
`0fa0a43b5e30a2f3ab063563e9aa743fa01551c1a4b52775d805b55bd04bb78a`):

| Input steps | Training one-step relative L2 | Validation one-step relative L2 |
| --- | ---: | ---: |
| 0--63 | 0.253495% | 2.389059% |
| 64--255 | 0.169494% | 0.427885% |
| 256--511 | 0.105104% | 0.239168% |

These are pooled squared-norm ratios, not averages of per-frame relative errors.
The early band carries 97.575% of validation squared error. Sparse clean-path
coverage is a hypothesis, not an established cause. Passing the closed 2%
engineering gate is not the paper-oriented clean-generalization qualification.

Implementation preserves every existing solver/population/trainer source and
checkpoint. All 13 new CPU tests and the 112-test combined synthetic suite
(new jobs, Clean, population and adapter tests) pass outside the Windows
sandbox. Two earlier invocations encountered temporary-directory ACL errors;
the successful runs used fresh explicit scratch directories. These fixtures
are not scientific evidence. Ruff E/F/I passes for the four new maintained
files; both launchers pass `bash -n`. The primary integrator checked the
code-to-contract mapping; this is not an independent reviewer sign-off.

The ignored attempt folder
`artifacts/time_dependent_no/cm_next_kf_coverage_20260908a_transport/` owns the
exact source-only payload: 23 files, 418,602 uncompressed bytes, plus two
separately invocable launchers. Both actual dependency closures are covered.
The archive, current source and launcher bytes agree. Approval-manifest SHA256:
`7a57489d04829597b9670ebbf0db45be9ca3c930203468230cb19877b5bb189e`;
source-archive SHA256:
`7f861bc81b6f0260f4f60aa0a2e9c8c15dd76f09f860c3660e7be14eab6e3348`.
No paper, docs, private context, dataset or checkpoint is in the upload.
The launcher requires approval bound to the exact payload and named phase.
Neither phase trains a new model or launches the other phase automatically.

The preparation-only AutoDL preflight verified C/Clean manifests and Clean
exit 0. The RTX 5090 was idle; the conservative memory guard passed. Diagnostic
preflight passed; population preflight correctly rejected 4,735,967,232 free
bytes (4.41 GiB). A read-only audit of the old
`d094_b1_c2_seed_replication_20260824b` run verifies all 72 remote `.pt` copies
against their exact contents in the retained local tar and the original
retrieval manifest. These copies total 5,522,361,080 bytes; deleting only them
would give about 9.55 GiB free before new outputs. The proposal leaves all
1,628 other remote files and every current C/Clean artifact unchanged.
The complete local recovery archive is 5,609,922,560 bytes, SHA256
`215a4adea5f850a9c1a79c7fe934fbcb77c22a305dd151cb42ff123c52fa9c92`.
Exact targets and backup/remote hashes are in `storage_review.json` (SHA256
`f72a50db5953709fc986396b09c10a4aa3ee999d87111bbe94d38a4a08d505d5`).
The owner subsequently approved that exact payload, both jobs and the exact
72-copy deletion proposal. `owner_approval.json` binds this affirmative reply
at SHA256 `2b7cfacfaa237cbe6061e6a682f07a7a12b752263879a88bf00be7719170a1f0`.
After a fresh full local-archive hash and remote path/hash/process checks,
only those 72 remote copies were deleted. All 1,628 non-target files and the
current C/Clean manifest/checkpoint bindings remained unchanged; the complete
local recovery archive is retained. Post-cleanup free space was 9.55363 GiB.
The deletion receipt is `cleanup_receipt.json`, SHA256
`969eed8781e39d90033a658623bf579975f5cbf0328dc1c075b760775737f76a`.
The original review remains an immutable proposal, not the deletion receipt.

#### Completed Existing-Checkpoint Diagnostic

All 23 approved source files were uploaded and verified. The unchanged
diagnostic launched at 15:15:13 China time and finished at 15:17:56, exit 0,
in 161.549 seconds. Its complete retrieval contains 45 files / 111,914,908
bytes. A separate CPU-only audit checks all packet hashes, current source,
the original C float32 state-store hash, all 5,307 retained per-step rows,
84 noninitial snapshots against the solver data, teacher arrays, metric
aggregation, physical/Fourier identities and censoring. All four frozen
checkpoint-replay batches match exactly for raw and restricted outputs.
This is a primary-integrator recomputation audit, not a second reviewer.
Cross-runtime spectral comparison uses band norms normalized by total error
norm to handle tiny FFT tails; maximum discrepancy is `1.11e-16`. The initial
over-strict relative-tail check and its audit-only adjustment changed neither
the producer nor any scientific threshold.

The following pooled relative L2 errors use matching input/target windows and
all trajectories in each role; they are not averages of per-frame ratios.

| First transitions | Training one-step | Training rollout | Validation one-step | Validation rollout |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.224895% | 26.2935% | 2.448379% | 35.2606% |
| 128 | 0.246810% | 127.9085% | 2.180576% | 124.6723% |

At H128 the rollout/one-step ratios are 518.25 for training starts and 57.17
for validation starts. Fixed-training-scale RMSE is 1.92565 / 1.91045,
respectively. All eight training-start and four validation-start paths reach
the registered `1e6` prediction-RMS/train-scale guard, at steps 420--509 and
409--471 respectively. These are finite-prefix amplitude failures, not
completed 512-step trajectories. H512 errors remain null; no failed path is
omitted or reset. Exit 0 means successful diagnostic execution, not stable
rollout.

The training-start result directly separates self-composition failure from
ordinary held-out one-step error: even the supervised clean paths have small
one-step error and poor recurrence. It does not qualify geometry, identify
normal versus tangential causes, establish correction necessity, or repair
the training/validation clean-error gap. This is one trained model; keep the
coverage study and later paired-response comparison separate.

Exact diagnostic bindings:

- launch receipt: `a03cf45d892cd2600b91061ab314ec515489647a96ec3cee9faf0e83af719fbe`;
- final artifact manifest: `43fef61913e62650169bc8d0a298eb89046f44783c09f492bd89eb75719f985b`;
- result: `2683ef021192229f2dd4a5c6b982de98e581ad70e64835ac84cf408a7bbcc576`;
- separate `diagnostic_audit.json`: `055737e4c2f2294d86d67ec2ccdde68ccd895fa1d2be79096cce84a6e5a8fe55`.

#### Completed Train-Only Extension

After the diagnostic exited and released the resource, the separately approved
population job launched at 15:20:23 China time with 9.45 GiB free and passing
memory/source/process checks. Launch-receipt SHA256 is
`81429ba0d903b3714e0de1b4de8d26610ff33b10a567c343e0267507f7e6a9a9`.
The job finished at 19:16:23 China time September 8, exit 0, after 14,158.513
seconds (3h56m). All 738 files / 7,021,872,295 bytes were retrieved. The separate
CPU-only audit passed at 21:36:48: all source/parent/payload bindings, 456 state
blocks, 12,312 finite native states, full-path diagnostics and peak anchors,
162 clean queries, 32 independently recurrent continuation steps and their four
eight-step endpoints agree. All six saved-answer replay checks pass (maximum
`2.342e-16`), with exact same-runtime repeats. This is independent numerical
recomputation by the primary integrator, not a second-reviewer sign-off.

| Maximum relative L2 discrepancy | Original C packet | New 24 paths | Registered limit |
| --- | ---: | ---: | ---: |
| Clean N256/N512 spatial comparison | 6.1702e-5 | 1.2730e-4 | 1e-3 |
| Discarded fine-state component | 8.1522e-5 | 1.5147e-4 | 1e-3 |
| Independent eight-step continuation endpoint | 2.1151e-5 | 1.6123e-5 | 2e-3 |

The two new spatial maxima occur at seed 2026090815, step 17. They exceed C's
maxima but remain below 16% of their limits. The largest temporal discrepancy
is `1.4942e-8` (descriptive, not a new gate); the largest discrepancy across
all continuation steps is `9.1456e-5`. Maximum mean-zero/dealiasing outside-band
fraction across audited arrays is `2.489e-16`. These are sampled clean-state
checks, not full-horizon cross-resolution or displaced-input qualification.

The read-only index now references 32 training paths / 16,384 transitions and
the same four validation paths / 2,048 transitions. No role changed. As a
descriptive coverage check, compare each validation initial state with its
nearest training **initial** state, using raw-vorticity RMS distance divided by
that validation state's RMS:

| Validation seed | Original 8 training starts | Expanded 32 training starts | Distance reduction |
| --- | ---: | ---: | ---: |
| 2026090621 | 0.30037 | 0.24181 | 19.50% |
| 2026090622 | 0.25059 | 0.20255 | 19.17% |
| 2026090623 | 0.28188 | 0.23328 | 17.24% |
| 2026090624 | 0.31397 | 0.21945 | 30.11% |

This measures closer initial-state coverage, not coverage of later trajectories,
a learned stability tube, or a causal explanation of the original one-step
gap. No seed or setting was selected using these distances. New-path peak
palinstrophy occurs at step `17.54 +/- 3.39` (range 14--26); final/initial
enstrophy is `0.09294 +/- 0.01883` (mean +/- sample SD across 24 paths). The
strong transient remains; no stationarity or chaos claim is qualified.

Training-input RMS over steps 0--511 changes from `4.063412685` for the original
eight paths to `4.191256370` for their 32-path union (+3.146%). No model or
normalizer was changed. Before the next fit, explicitly freeze normalization,
matched-update exposure and stopping, and distinguish any longer optimization
extension from the data-count comparison. The current trainer hard-codes the
8/4 split and requires a reviewed extension before use with 32 paths. The data
stage is complete; comparable clean generalization and corrective efficacy
remain untested by this packet. No new model training, protected access or
paper upload occurred.

Exact population bindings, in the existing ignored transport folder:

- final artifact manifest: `a0f446389160974f29ad0cec07ce9ac752442b5b7d78f118a8a7e83ffd01a0cc`;
- result: `b5aed8a70cc436b0c7d4dafdaca9555111a8dc307e51954c23e7f5428beac377`;
- retrieval receipt: `a0b91c6c8e83deca2ac9d164eb79da0dbe3d3fc35c6d9674ee23c61c16f7f307`;
- `population_audit.json`: `8553e0619f463c2d10cd05ada457020af0930766c6ee26aef8c53016ffacbdc0`.

The slow local downloader was resumed with four read-only SFTP streams, reusing
136 verified files. Its single interrupted 12,058,624-byte partial copy is
preserved outside the scientific packet inventory. No remote data were changed
or deleted during retrieval. The ignored `retrieve_population.py` and
`audit_population.py` retain this packet's retrieval and recomputation logic;
the existing cleanup/status and diagnostic helpers remain unchanged. These
are packet-specific evidence helpers, not maintained experiment APIs. The
approved transport and scientific source payload remain byte-identical.

#### Fixed-Update 32-Path Fit: Implementation And Review

The September 8 source snapshot below was not launched. Its review-driven
September 9 successor is recorded at the end of this subsection; the scientific
recipe is unchanged.

The owner requested implementation and careful checks after data qualification.
The new `fit_kolmogorov_coverage.py` keeps all closed source/checkpoint bytes
unchanged. The experiment plan owns its fixed 49,152-update contract; the
implementation plan owns the validate-only and CUDA-fit invocations. This is
24 union-data passes versus the baseline's 96, with identical update/LR history
and the original eight-path normalizer. Original-eight, added-24, full-training
and unchanged-validation metrics are separate. No adaptive extension,
best-checkpoint selection, rollout, solver call or corrective arm is implemented.

Verification: 38 focused new tests and the 150-test combined synthetic suite
pass (218.59 seconds for the combined run); Ruff and Git whitespace checks pass.
Tests include old-evaluator parity, actual successor labels, role/index coverage,
normalizer leakage, deterministic sampling/checkpoint continuation, terminal
replay, corrupted inputs, nonfinite/budget/storage failure and parent nonmutation.
One earlier sandbox test attempt hit Windows temporary-directory ACL errors;
fresh scoped directories outside that sandbox resolved the infrastructure issue.
This is primary-integrator testing, not independent code-review sign-off.

The new loader also passed on the complete retrieved scientific packets in
361.45 seconds, without constructing a model or calling a solver. Its
`[36,513,256,256]` float32 store is 4,841,275,392 bytes; original-state hash and
normalizer reproduce the baseline exactly. All input/source evidence remains
stable. The descriptive union RMS is `4.191256370`, but the model scale stays
`4.063412684964112` before float32 storage. Maximum added-state canonical
residual is `3.30e-16`. This verifies data handling, not new model performance.

The ignored attempt folder is
`artifacts/time_dependent_no/cm_next_kf_clean32_20260908a_transport/`.
Its data-validation manifest SHA256 is
`d7e341e444db433458594e4006879f11f989eb458add8b6efe68d817048efb50`;
union float32 store SHA256 is
`b9e3cfa138ad2103b50b29ac422897a759bb7f34deaa4f1feff055fc266e2552`.
The source-only archive contains 25 files / 479,963 uncompressed bytes, with
no paper, docs, private context, dataset or checkpoint. Archive SHA256 is
`4ddb75a39fa365ac8d7824f469f0ec470cb6d2951700ccc3217cc627749f97db`.
`CODE_REVIEW_PROMPT.md` contains the exact new code/tests, fixed contract and
selected unchanged helper excerpts; SHA256 is
`16dd44e52b97051bbef9fc2e79f2949901fb1f678933d42ef14c0a197e432b2b`.
The experiment-bridge review step must not send that private material without
explicit owner approval. No external review, upload or scientific fit has run.

At 22:23 China time, read-only AutoDL preflight confirmed an idle RTX 5090 and
2.91 GiB free storage, but its initial 12 GiB conservative host-memory guard
rejected the available 9.39 GiB. The system-profile check then measured the
unchanged loader using external Windows process counters (250 ms sampling).
The repeat validation passed with the identical union state hash; no production
code was instrumented and no model, solver or network call ran in that profile.

| Host-memory basis | GiB | Status |
| --- | ---: | --- |
| Loader peak working set | 4.77 | Measured locally. |
| Loader peak committed memory | 6.24 | Measured locally. |
| New CUDA-fit host RSS | 6.34 | Estimate: old GPU-run peak RSS plus the added float32 state-store bytes. |
| Revised conservative headroom guard | 8.00 | 1.66 GiB above the estimated fit peak; not a guaranteed upper bound. |

The subsequent read-only preflight passed: GPU idle, 2.91 GiB free disk, about
9.39 GiB conservative headroom, and the identical baseline Python/NumPy/Torch/
CUDA runtime. No cache clearing, remote deletion or process stop occurred.
Retain the initial unapproved transport/manifest in
`profile_output/preprofile_transport.zip`; the scientific source archive is
unchanged. The revised exact payload-manifest SHA256 is
`4b3d684c079b861284c049a9464d5c68f91952176d2d3c2920384ec955e8bca5`.
The old manifest `56a9435f...` is superseded, not approved. The launcher repeats
capacity/runtime checks immediately before launch and rejects an existing run.

Profiling artifact/change inventory (within the ignored attempt folder):

| Surface | Change | Purpose |
| --- | --- | --- |
| `profile_output/memory_profile.json` | Generated | External process counters and sampling/exit receipt. |
| `profile_output/data_validation/` | Generated | Repeat unchanged-loader validation and its source/input manifest. |
| `profile_output/preprofile_transport.zip` | Generated | Preserve the initial unapproved guard/manifest exactly. |
| `transport.py`, `payload_for_approval.json` | Updated | Evidence-backed 12-to-8 GiB guard and memory-provenance binding only. |

No source instrumentation needs reverting. The baseline's measured training/
evaluation times project roughly 3.5--4 hours for the new fit, with its five-hour
cap. Independent review and exact payload approval remain pending; no upload
or scientific fit occurred. Any review-driven source change requires a new
payload freeze and approval, not reuse of these hashes.
Final cleanup removed only this turn's three synthetic-test scratch directories
(78,742,870 bytes, reproducible from the recorded test invocations). All
scientific validation/profile packets, closed sources, source archives and
pre-existing work remain. The only new maintained files are the bounded
coverage runner and its test; the ignored attempt folder owns its transport,
review, validation and capacity evidence. No commit, push or paper edit occurred.

September 9 continuation: the owner approved the exact prompt and current
`4b3d684c...` payload, conditional on review passing unchanged and fresh resource
checks. `owner_approval.json` records this approval; neither the prompt nor
payload was edited. All 25 source hashes reverify. Fresh read-only AutoDL
preflight passes with an idle RTX 5090, 2.91 GiB free disk, about 9.39 GiB
conservative host-memory headroom and the unchanged baseline runtime. An initial
sandbox WSL access denial was resolved by the approved host execution path.
Both GPT-5.4 xhigh review calls timed out after 300 seconds without a verdict;
their raw responses are preserved in `code_review_attempt1_response.json` and
`code_review_attempt2_response.json`. A same-prompt, same-model CLI recovery
then exited with code 1: GPT-5.4 is unsupported when using this Codex account.
It used read-only mode, disabled project-document loading and saved its events
and stderr; it produced no review. `review_gate.json` binds these receipts and
the exact session IDs. Model unavailability is established for that CLI call,
not proven as the cause of the earlier transport timeouts. At that point the
review condition remained unmet and substitution required owner permission.

September 9, approved substitute review and local corrections: the owner
authorized an available Codex reviewer with the same exact prompt. GPT-6 Astra
xhigh completed the static review (exit 0); the recorded event stream contains
no tool calls. It reported four MAJOR and three MINOR findings, not a pass.
`code_review_astra_final.md`, its event log and `owner_review_substitution.json`
remain in the September 8 attempt folder. No source upload or fit followed.

The successor fixes output/input-packet overlap before directory creation,
deadline expiry inside the last operation, and incomplete provenance after
loading failures. It also preserves writable failure packets after source/setup
errors and names the two-interval diagnostic with explicit endpoints. The
retention finding exposed ambiguous prompt wording: the intended storage plan
is all-pair scalar metrics plus six input/target/raw/restricted field sentinels,
not every prediction field (9 GiB per evaluation). The experiment plan now
states that distinction. Existing closed helper tests already verify Adam/RNG
restoration; the new integration test also replays a fresh PCNO and crosses
sampler reshuffles. No scientific hyperparameter, label or stopping rule changed.

All 59 focused tests pass (148.55 seconds), as do all 190 tests selected from the
runner's source-bound dependency set (262.03 seconds). Controlled-clock tests
expire inside final batches, serialization, optimizer/checkpoint operations and
replay without directly raising a synthetic timeout. Loading-failure tests catch
parent mutation and distinguish unchecked evidence from stable evidence. Ruff
and Git whitespace checks pass. An initial focused invocation failed to create
its missing scratch-parent directory; creating that directory resolved setup.

Full retrieved-data validation passed again in 446.30 seconds, including 31.60
seconds of final provenance checks. All three input bindings and all 25 current
sources pass; no model was constructed and no solver was called. The original
and union state hashes, original normalizer and descriptive union RMS match the
earlier validation exactly. This is data/implementation evidence, not a new
model-performance result or a passed independent re-review.

The new ignored attempt folder is
`artifacts/time_dependent_no/cm_next_kf_clean32_20260909a_transport/`.
It contains the exact review prompt, source-only archive, bounded transport,
test reports, validation packet and `verification.json`. The transport differs
only in run identity; all 23 closed dependency files remain byte-identical.

- Payload manifest SHA256:
  `d66f4cb7580409a062fb2e5d3daad3c6f871fcef287abe16f05233d53f6ccf0e`.
- Archive: 25 files / 491,644 uncompressed bytes; SHA256
  `ab250875c9800ebdb3a19cd3f302103557fcaa988613a43b67fec17c16a92377`.
- Exact revised review prompt SHA256:
  `0c197066b161844f3c00209a312c630295620be3e3fc3298b567437f700d3118`.
- Data-validation manifest SHA256:
  `0b2c5a0bb4f12b3ed5d9e3e6e4aa94ae808eaa87e1ae8f6f6dd47a31e27533cb`.

Next: obtain approval for this revised private code-review prompt and exact
payload. Re-review, then upload/launch only if it passes unchanged and fresh
AutoDL resource/runtime checks pass. Do not reuse the old conditional approval
for changed source. No paper, dataset, checkpoint, credentials or private
resource details are included in the review prompt; the source upload excludes
paper/docs/data/checkpoints and private context. Leave the other agent's resource
untouched. No commit, push or paper edit occurred.
Scoped cleanup removed only three reproducible synthetic-test scratch directories
(133,588,320 bytes); test reports and all scientific/review evidence remain.

September 9, exact re-review outcome: the owner approved the revised prompt and
conditional payload launch. Fresh source/archive/test-report/validation bindings
pass. AutoDL preflight at 11:37 China time passes with an idle RTX 5090, 2.91 GiB
free disk, 9.39 GiB conservative memory headroom and the baseline runtime. The
single GPT-6 Astra xhigh review completed at 11:45 with a matching final event
and no tool calls; the detached process's OS exit code was not captured.

The review did not pass: archive decoder exceptions can bypass finalization
(MAJOR); path-keyed provenance checks conflate aliased input roles (MINOR);
direct output-norm and finite-loss/nonfinite-gradient tests are missing (MINOR).
It found no blocking issue in scientific pairing, update accounting or metric
formulas. These findings do not show corrupt scientific data or a failed model.
Four primary-integrator synthetic CPU probes confirm the first two defects in
31.28 seconds. Their passing assertions reproduce unwanted behavior, not
readiness. No approved source, test, archive or recipe was modified.

The ignored attempt's `code_review.json` binds the exact approval, final review
(SHA256 `152e8ba36d84599b8871c6fcd60465b679d5d9059a45c113f45146b4be1fe74d`),
event log, preflight and `review_failure_probe.py`/`.xml`. The probe and its
synthetic scratch are retained to support the focused repair decision. The
previous `verification.json` remains the earlier local-validation snapshot;
it is not a post-review sign-off. No upload or scientific launch occurred.
The owner then approved the narrow decoder/provenance/test-only repair,
re-review and conditional launch without another approval round. The successor
is `CM_NEXT_KF_CLEAN32_MATCHED_20260909B`; data, model, training budget and
protected-access rules remain fixed. Its exact source, test, review and launch
receipts are kept separately from the failed A review. Repairs are complete:
decoder exceptions now enter failure finalization, validated evidence is keyed
by role, and nine new regression cases cover decoder failures/mutation, aliased
roles, output norms and nonfinite backward gradients. All 68 focused tests pass
in 152.54 seconds. Independent static and payload audits pass; only the runner
and its test changed among the 25 source files (498,898 bytes).

The exact B payload manifest is
`11fbeaef4926099f5acef6f60c671b28a73d23a35014aedebb1149f658e0df4a`;
the archive is `2e3bcdf8e12a48cdf4ccb2df96553cf26ab3817314a2c6e7d46295d9f0a1678a`.
The locally saved revised review prompt is bound by
`9c49316d65dd1056f2c1ca65f4488e3bdd7f552d7894150af6024e612c906944`.
The permission checker rejected the GPT-6 Astra xhigh call before process
creation: it requires explicit approval of this exact prompt and destination,
despite the broader repair/re-review authorization. No code was transmitted;
no workaround or duplicate retry was attempted. The ignored B `code_review.json`
records the approval scope and rejection separately from a scientific verdict.
AutoDL preflight at 12:29 China time passes: idle GPU, 2.91 GiB free disk,
9.39 GiB conservative memory headroom and unchanged baseline runtime.

Final local checks pass: 199 source-bound tests in 373.78 seconds; full-data
validate-only in 418.80 seconds (332.51 work, 86.29 final provenance checks).
The 36-path float32 store and original eight-path normalizer reproduce the
previous fingerprints exactly. All three input bindings and all 25 source
bindings are stable. The validation manifest is
`3b159288a602d91d956d558257ef99d54044d4b429d4907c8ac48690edcea3b5`.
The ignored B `verification.json` binds test reports, data checks, source ZIP,
prompt and preflight. Ruff and whitespace checks pass. No new maintained script
was added; reproducibility receipts and synthetic scratch remain local.
The owner subsequently approved the exact B prompt/destination and conditional
launch. GPT-6 Astra xhigh reviewed that unchanged prompt from 16:26:54 to
16:32:13 China time, with OS exit 0, a matching terminal event/final response,
no tool calls and no required revisions. The final review SHA256 is
`905b777ffff83d92cd724f756e84a17e11d015554babf8c6b29b172676595fb0`.
The ignored `code_review_completion.json` binds the passing review to the
approval, prompt, payload and earlier local verification; the earlier permission
rejection and verification receipts remain unchanged historical evidence.

Fresh before/after-upload resource/runtime checks pass: idle RTX 5090,
2.91 GiB free disk and 9.43 GiB conservative host-memory headroom. All 25 source
files and the launcher were uploaded and remotely hash-verified. B launched
at **16:35:09 China time September 9**, launcher PID `120197`. At 16:35:55 the
launcher is alive, with no error output or exit code; input loading/validation
precedes GPU use. Startup audit completed at **16:43:57**: all 917 logged updates
are contiguous with finite loss and gradient norm, and the warmup schedule
matches. The initial 18,432-pair evaluation (2,304 forward calls) and initial
checkpoint rehash correctly; remote source, state-store, original normalizer
and recipe identities match the approved local bindings. Data loading took
117.35 seconds and the initial evaluation 193.50 seconds. Updates cost about
0.20 seconds; host RSS/peak so far is 6.22 GiB. The 16:42 GPU snapshot shows
7,695 MiB used and 98% utilization, with no error output. The ignored
`startup_resource_check.json` binds these read-only checks to the review and
launch receipts. The startup estimate was 20:05--20:35 China time; it is
superseded by the completed result below.

#### September 9 B Completion And Matched Clean Comparison

B completed all **49,152 updates** at **20:03:56 China time**, exit 0, in
12,525.74 seconds (3 h 28 min 46 s), including 9.31 seconds of final provenance
checks. The launcher has exited and the GPU is idle. Peak process RSS was
6.36 GiB; peak allocated/reserved GPU memory was 6.57/6.90 GiB. No training
extension or checkpoint selection was performed.

All 26 retained files (376,782,777 bytes) were downloaded. Local hashes match
an unchanged before/after remote inventory and the completed manifest; all
25 source bindings and three input bindings pass. Independent NumPy reductions
reproduce 1,924 summaries across 239,616 evaluation-pair records, including
the saved field-sentinel norms. Original/validation input, target and
persistence norms match the baseline exactly. A separate audit reconstructs
all 24 seed-17 sampler permutations and all 49,152 learning rates; the rates
match the baseline exactly. All updates and Adam states are finite, and latest
and terminal checkpoint state trees agree. Fresh CPU inference from the fixed
terminal reproduces all six sentinels within the registered `1e-6` relative-RMS
limit (maximum raw/restricted discrepancies `6.29e-7/6.67e-7`). CUDA RNG resume
and optimizer continuation were not tested locally.

Local evidence remains under the ignored
`artifacts/time_dependent_no/cm_next_kf_clean32_20260909b_transport/`:

- completed manifest: `437f826a273afc331ce5a1441ffd57f230b28bb27d94bc198080dca64f45fa76`;
- result: `69eb5b04ce8182086e65b098bf001cc9d7404452677830b7ce1cde81a32df1b7`;
- terminal checkpoint: `6f8b106eb3758ac4cdb08ece4313969835bdc9c77afdaffab3f055810c9d0dbf`;
- retrieval verification: `e3d19e0ccb81ebe2756d215dd4384e458e7a7ff7a6c68103925167ede646891f`;
- independent metric audit: `caf797479d06a60583ba1a15910ad8e69e4c2083632dfbbc76741c8195f5b779`;
- completion/checkpoint audit: `254ac00d9fc7595ecad2abc3687eaabaa2e3e3af649f1742e3ea5efc83180f6d`.

`retrieve_result.py` is a download-only helper for this frozen transport's
missing retrieval action; `audit_metrics.py` independently reduces its saved
arrays. Both are run-local evidence tools, invoked directly with Python, not
new maintained experiment entry points. The retrieval/completion receipts bind
their outputs without rewriting prelaunch or source-bound evidence.

All errors below are **pooled next-state relative L2 percentages on clean
reference inputs**, not recurrent rollout errors. Both models use the same
mean-zero/dealias restriction, normalizer, seed, update count and LR history.

| Evaluation population | Original eight-path fit | Expanded 32-path fit |
| --- | ---: | ---: |
| Same original eight training paths | 0.201864% | 0.262575% |
| Added 24 training paths | -- | 0.255403% |
| All 32 training paths | -- | 0.257105% |
| Same four validation paths | 1.616301% | 0.480783% |

Validation error improves **70.25%**, while original-eight training error
worsens **30.07%**. Thus the smaller gap is not merely a consequence of worse
training fit. Every one of the 2,048 matched validation pairs improves; all
four validation trajectories improve by 67.84--71.38%. Validation SSE and
fixed-scale MSE fall 91.15%. The validation/training ratio falls from 8.007 to
1.870 using all 32 training paths, or 1.831 on the same original eight paths.

| Input-step band | Original validation | Expanded validation | Expanded all-32 training |
| --- | ---: | ---: | ---: |
| `[0,64)` | 2.389059% | 0.700985% | 0.356566% |
| `[64,256)` | 0.427885% | 0.176425% | 0.152628% |
| `[256,512)` | 0.239168% | 0.113125% | 0.097923% |

The remaining discrepancy is concentrated in the early transient: its first
64 steps contribute 94.94% of validation SSE but 44.66% of target energy.
Validation/training error ratios are 1.97, 1.16 and 1.16 across these bands.
Original-eight training error worsens 40.62% in the early band, but improves
8.03% and 14.70% in the middle and late bands. Do not discard the early window.
Over the last 8,192 updates, validation improves 4.32% and training 5.76%; over
the last 4,096, validation improves only 0.127% and training worsens 0.833%.
Keep the prescribed terminal; these are optimization diagnostics, not a basis
for retrospective checkpoint selection or automatic extension.

**Interpretation and next decision.** Additional same-law trajectory coverage
materially addresses the previous clean generalization deficiency at a fixed
optimizer-update budget. This is one paired training seed, with 24 versus 96 data passes,
not replicated uncertainty, matched epochs or proof of optimization convergence.
The four validation paths remain development evidence, not a fresh prospective
test. The main metrics describe the **restricted transition, not raw PCNO**:
raw validation error is 1.6782% for B versus 2.0914% for the baseline, whereas
restricted validation error is 0.4808% versus 1.6163%. This shared numerical
restriction is not a measured projection onto a data manifold.

No new rollout, displaced-response or geometry assay was run. The earlier
eight-path model's blow-up cannot be attributed to this checkpoint. Recommend
a separately frozen, evaluation-only rollout of B on the same original eight
training and four validation initial states, retaining the original 512-step
horizon and amplitude guard. Compare its teacher-forced and recurrent errors
on matching windows before corrective arms or primary-case qualification.
If coverage also resolves rollout failure, retain that finding. This analysis
does not authorize or launch that successor, a solver job, corrective arm or
training extension. No protected read, cleanup, paper edit, commit or push
occurred during retrieval and analysis.

#### September 9 Expanded-Model Rollout Preparation

The owner accepted the matched-start rollout next step and explicitly excluded
a data-scarce, severely overfit baseline as the intended central example.
Coverage adequacy is assessed through clean training/validation fidelity,
including early times, not a trajectory-count label. The new contract is
`CM_NEXT_KF_CLEAN32_ROLLOUT_20260909A` in the experiment plan. The completed
8-path and 32-path fits and the old rollout remain unchanged evidence.

The separate `evaluate_kolmogorov_coverage_rollout.py` entry point reuses the
closed evaluator's recurrence, metrics, snapshots, horizon summaries and
amplitude guard. It binds B's fixed terminal and C's original state store;
teacher indices `0--7,32--35` are checked against all 6,144 original reference
transitions. Only `next_state` is recurrent feedback. No additional training
trajectory arrays, solver calls, optimizer steps, correction or geometry assay
are included. The new test file owns this wrapper's identity/mapping and
failure-handling checks; no general orchestration layer or core API was added.

The first 23 focused tests passed. Independent review then identified two
failure-receipt gaps: paths started before an I/O failure could be labelled
unstarted, and `pickle.UnpicklingError` bypassed finalization. Both are fixed
and covered by regressions. All **26 final focused tests** pass in 160.92 s;
the **36 unchanged rollout/adapter tests** pass in 13.31 s. The initial
sandboxed dependency invocation had a temporary-directory permission failure;
its report is retained, and the fresh-scratch rerun passed. Ruff and whitespace
checks pass. Full-data validate-only completed in **116.41 s** (105.52 work,
10.89 final provenance), with no model construction, checkpoint deserialization,
rollout or solver call. Original state/normalizer fingerprints reproduce
exactly. Independent code-to-intent and packet/source/test audits pass.

The ignored run-local transport provides preparation, guarded upload/launch,
status and bounded retrieval, including failed packets. It requires the exact
owner approval and a passed prompt/payload-bound external review. Its syntax,
five generated remote-call snippets, valid synthetic approval and nine invalid
approval cases pass static checks. All 25 B-bound sources remain byte-identical;
only the new evaluator and test extend the payload to **27 files, 534,791 bytes**.
The exact private prompt contains five selected code files plus the narrow
contract/dependency descriptions. No dataset arrays, checkpoint tensors,
numerical results, credentials, host addresses, local context or manuscript
are included; it has not been transmitted.

Current evidence under ignored
`artifacts/time_dependent_no/cm_next_kf_clean32_rollout_20260909a_transport/`:

- payload manifest: `c3a9a8fe6d7c400b46e54329ce715cd8ffb936559e8994b60cbb7934f494126c`;
- source archive: `aca3036d1086f6ac0db51ba53fac354cd790741110a65a5353cb2f1daa60ab3b`;
- exact review prompt: `ba514d40fce7c222b2d27b4e5290cea35114563c93b2b682b01aca48a04c4f95`;
- full-data validation manifest: `294b7d79c384100009b4baebd9a20ba8bf62a21e88c34f48deffce6326ff70c2`;
- local verification receipt: `b0df48c762451dda0031ef7750ea0f9f535dcd9b1412a494309b4fe8d084f998`.

Read-only AutoDL preflight at **22:55 China time September 9** passes: idle
GPU, 2.56 GiB free disk, 11.07 GiB conservative host-memory headroom, unchanged
B runtime and pinned input hashes, and no process using the new attempt root.
Refresh these checks immediately before upload/launch. The work cap remains
30 minutes, with a 35-minute external timeout and 60-second kill grace; these
are limits, not measured full-horizon runtime estimates. The old rollout's
161.55-second duration included early path failures and is not a reliable ETA.

**Decision requested at preparation:** approve the exact prompt for GPT-5.4 xhigh review
and the exact source-only payload for upload/launch only if that unchanged
payload passes review and fresh preflight. Local checks are not a substitute
for that gate. No prompt transmission, upload, launch, cleanup, protected read,
paper edit, commit or push occurred. Keep the fixed terminal and all outcomes;
no training extension, regime tuning or corrective matrix follows automatically.

#### September 9 Approved Review: Model Unavailable

The owner replied "Yes, please proceed" to that exact review and conditional
unchanged-payload launch request. `owner_approval.json` records the scope; a
fresh independent audit confirms all 27 live/archived sources and the 26+36
test receipts remain unchanged. No tests were unnecessarily rerun.

At **23:57:53--23:58:01 China time September 9**, the exact prompt was submitted
to GPT-5.4 xhigh with user configuration disabled, project-document loading
disabled and instructions forbidding reviewer tools or additional context.
The service returned HTTP 400: GPT-5.4 is not supported for this Codex ChatGPT
account. The captured OS exit code is 1, terminal event `turn.failed`, and tool
calls zero. There is no review verdict or required-revision count. This is a
reviewer availability failure, not a code or scientific failure.

The ignored attempt retains `code_review_launch.json`, raw
`code_review_events.jsonl`, `code_review_stderr.log`, actual
`code_review_process.json`, and the hash-bound `code_review_completion.json`.
The one-use local `launch_review.ps1` records the invocation and is not part of
the uploaded source payload. Earlier frozen receipts remain untouched.

No model substitution, AutoDL access, upload, rollout launch, cleanup, protected
read, paper edit, commit or push occurred in this continuation. Request owner
approval for a supported reviewer (the preceding B review used GPT-6 Astra
xhigh) and only the necessary review-destination metadata rebinding. Preserve
the rejected attempt and all scientific source bytes. A passing new review
and fresh resource/runtime checks must still precede any launch.

#### September 10 Approved Reviewer Substitution

The owner approved GPT-6 Astra xhigh, review-destination rebinding and the
single unchanged scientific rollout conditional on review/preflight passing.
Thirteen exact earlier artifacts are preserved and rehashed in
`review_gpt54_rejected/`, including all dependencies of the failed-review
receipt. The active transport changes only the expected reviewer and completion
receipt name. Its payload manifest changes only `transport_sha256`; the prompt
changes only matching reviewer bindings and a final blank line. All 27 live/ZIP
source files (534,791 bytes), the ZIP itself and `run.sh` are unchanged.

Current bindings in the same ignored transport directory:

- payload: `7e2753ca3936a7d2987c64eb492b58ea2651df13a535a7e1c204382b7e0c2789`;
- transport: `342d5100abe1b0977cab931431e69aa0dc48bafa1d2c66527a9ddf22a04fdcc7`;
- review prompt: `aa335c0108efcdadbcadf08520cd99d2aa713d3bfbb312d2c87dbf5c35fedbf4`;
- scientific ZIP remains `aca3036d1086f6ac0db51ba53fac354cd790741110a65a5353cb2f1daa60ab3b`.

Local checks verify all five embedded prompt files, transport syntax, one valid
synthetic review receipt and eight invalid receipts rejected. These are guard
tests, not an external review verdict. The retained 26+36 tests still bind the
unchanged scientific sources. At 10:25 China time, read-only AutoDL preflight
passes: idle GPU, 2.56 GiB free disk, 11.05 GiB conservative memory headroom,
unchanged runtime and pinned inputs, no process using the new root.

The exact replacement static review ran **10:24:42--10:32:42 China time**,
OS exit 0, terminal event `turn.completed`, zero reviewer tool calls. Its verdict
is **REVISE: five required revisions**, not a passed launch gate. Raw events,
final response, stderr, actual process completion and hash-bound disposition
are retained under separate `code_review_astra_*` names.

Required revisions and root checks:

1. Guard optimized Python explicitly: transport safety assertions can disappear.
   A synthetic optimized guard accepts invalid approval/review fixtures; no live
   approval bypass was attempted, and optimized execution was not observed.
2. Upload captured, hash-verified archive/launcher bytes instead of reopening
   mutable live paths after preflight. The code permits this race; no actual
   intervening edit or unapproved upload was observed.
3. Hash captured checkpoint bytes and deserialize that same buffer. The current
   path is reopened after population validation; no checkpoint corruption or
   malicious deserialization was observed or performed.
4. Reject source/output overlap in both directions before creating output.
   The asymmetric predicate reproduces on synthetic paths without file writes;
   the frozen actual output root is separate from both input source roots.
5. Preserve successful transport verification separately if final scientific
   packet parsing/validation fails. Current retrieval can abort before writing
   its receipt; no scientific packet exists for this unlaunched attempt.

The reviewer found no mismatch in the 0--7/32--35 mapping, original normalizer,
`next_state` recurrence or validate-only design. These findings concern safeguards
and failure handling, not a numerical result. Active-path write-failure retention
and a direct 36-entry mapping fixture are optional coverage suggestions.

No scientific code changed after review. Nothing was uploaded or launched; no
training, solver, correction, protected read, cleanup, paper edit, commit, push
or other-resource action occurred. The current approval covers reviewer-binding
changes only. Request narrow repairs, focused tests, a new source freeze,
re-review and conditional launch; preserve this failed-review identity.

#### September 10 Narrow Repair Successor

The owner approved the five repairs, focused tests, Astra xhigh re-review and
automatic launch only after review and fresh preflight pass. The repaired
identity is `CM_NEXT_KF_CLEAN32_ROLLOUT_20260910B`. A's rejected packet, prompt,
review, source ZIP and permission history remain untouched.

The existing coverage-rollout entry point now rejects optimized execution,
checks source/output overlap in both directions, and hashes a captured
checkpoint buffer before deserializing those same bytes. Its numerical rollout
helper, checkpoint, twelve starts, normalization, horizons and diagnostics are
unchanged. Fixtures now use a separate archived fit-source tree.

The private transport rejects optimized local/remote interpreters, uploads only
captured and hash-verified source ZIP/launcher bytes, and records successful
transfer separately from malformed or incomplete scientific packets. The private
`test_transport.py` runs through pytest with fake SFTP and synthetic bytes only;
it is retained as transport regression evidence and is not uploaded. No generic
new infrastructure or scientific method was added.

Verification:

- Before the overlap fix, two below-source cases fail and the two above-source
  cases pass. The initial sandbox attempt failed during scratch setup, before
  the tests; both receipts are retained.
- Final source-bound scientific/dependency suite: **68 passed**, 277.525 s,
  exit 0. Private transport/guard suite: **29 passed**, 2.317 s, exit 0.
- Lint passes with the existing pytest fixture-import `F811` exception; files
  are formatted. Final scientific tests ran after the formatter adjustment.
- Actual-data validate-only completes in 166.679 s, including 10.798 s final
  provenance checks. All original 6,144 pairs match B indices 0--7/32--35;
  source/input stability passes, and no model or checkpoint deserialization
  occurs. The model scale remains 4.063412666320801.
- All 27 live/archived sources match (538,019 bytes); all 25 closed fit-bound
  sources are unchanged. All six prompt-embedded files match current code.
- Fresh read-only AutoDL preflight at 13:31 China time passes: idle GPU,
  2.56 GiB free disk, 11.05 GiB conservative memory headroom, matching runtime/
  pinned inputs and no process using the new root. Receipt:
  `preflight_20260910T053104.json`. Refresh before any upload or launch.

Exact current bindings under ignored
`artifacts/time_dependent_no/cm_next_kf_clean32_rollout_20260910b_transport/`:

- payload: `ebd2f689f42a062577c957867e2023f2b282f628fbaf25c7bbf2b88ee55dccd3`;
- source ZIP: `cef85e77a95d33b211bd9f0d0fa12c83e6452e798047f8330641b63a9c2e678a`;
- transport: `8bc7910b2c514bf13b2de8577fb8f0566f1470cbb660cdac10f8d0a5481f71ab`;
- six-file review prompt: `cf65a35803e98d2cfde939e0243b0c7ca4f0fe87702def3d7f12bd12efa468da`;
- final scientific test XML: `7368c80c1a96c72e1340a91119d2fecc15fcb5f81258c05aeb7df3b585302dec`;
- transport test XML: `0210b61e65eeaae7f1e0657c9b8af2a09125a28c7ace74a4c578dd1b20faee1e`;
- validation manifest: `86ec2e7b668c56df4a19e4fbba8b94225542402edca4937722c5b8b16cda69bb`.

The external review command was rejected **before transmission** by the
environment's approval checker. It acknowledged general repair/re-review
authorization but required explicit consent for this exact prepared six-file
private prompt and GPT-6 Astra xhigh destination. This is a permission rejection,
not a reviewer finding, OS process failure or scientific result.
`code_review_permission_rejection.json` records the reason; that rejected
invocation produced no reviewer process, OS exit code or verdict. No indirect
submission was attempted.

The owner then explicitly approved the linked six-file prompt for GPT-6 Astra
xhigh and conditional source-only launch. `owner_approval_workflow.json`
preserves the earlier workflow receipt; `owner_approval.json` records the exact
consent. The unchanged reviewer launcher ran from 15:15:56 to 15:22:41 China
time on September 10, with OS exit 0, `turn.completed`, zero tool calls and an
empty stderr log. Verdict: **REVISE, two required transport revisions**.

1. Before accepting evaluation completion, validate all twelve selected paths,
   started/unstarted accounting and allowed terminal outcomes. A completed path
   must have 512 steps; legitimate amplitude/nonfinite censoring remains valid.
   The current synthetic completed packet omits cases yet is accepted.
2. Status/retrieval must authenticate the frozen payload against the launch
   receipt, not require unchanged live source/ZIP/launcher files. Otherwise the
   supported captured-byte upload can succeed after local edits but subsequent
   retrieval is blocked. Keep live verification before launch and all transfer
   hash/inventory checks.

The primary integrator confirmed both issues by source/test inspection; no new
runtime tests were run this turn. The reviewer accepted repairs 1--4 and found
no additional scientific-evaluator revisions. Review scope was static only,
not numerical validation. Exact raw output, actual process exit and bound
verdict are retained in `code_review_astra_{events,process,final,completion}`
files. Final-review SHA256:
`b8709142c94471c56f8352589ec57a389a56e217a7032b0a3b1c515c27e08c97`.
Completion-receipt SHA256:
`eabf2d29b134afc8f255590d5e4a2c263fae888fb75aa77c6328c52097a34b69`.

Fresh read-only preflight (`preflight_20260910T071755.json`) again passes: idle
GPU, 2.56 GiB free disk, 11.05 GiB conservative host-memory headroom, matching
runtime/input hashes and no process using the attempt root. All 27 live/ZIP
sources still match the frozen payload and all 25 fit-bound sources are
unchanged. The required review gate does not pass, so no upload or scientific
launch occurred. Preserve this reviewed bundle; request the two transport-only
repairs, focused tests and re-review of the same six-file scope before launch.
No training, solver, correction study, cleanup, protected read, paper edit,
commit, push or other-resource action occurred.

##### Transport Revision R2

The owner approved both transport-only fixes, focused tests and re-review of the
same six-file scope by GPT-6 Astra xhigh, with launch only after review and fresh
preflight pass. R2 is a separate local transport revision of the same unlaunched
scientific run, not a different experiment. The original B review bundle is
unchanged. The new ignored folder is
`artifacts/time_dependent_no/cm_next_kf_clean32_rollout_20260910b_transport_r2/`.

`validate_completed_paths` checks the exact twelve paths, horizon, step/row and
started/unstarted accounting, terminal outcomes and aggregate completion flags.
It preserves valid amplitude/nonfinite censoring. `launched_payload` authenticates
captured manifest bytes against the launch receipt for status/retrieval without
depending on mutable live source, ZIP or launcher files. Prelaunch live-file
verification and all transfer hash/inventory checks remain in place.

The original transport fails 25 new regressions while 31 controls pass. The
repaired final suite passes **60 tests**, 5.959 s, exit 0, including first-step
nonfinite and last-step amplitude censoring. The initial sandbox invocation had
scratch ACL errors; successful tests used a fresh workspace scratch directory
outside that restriction. Ruff E/F/I passes with E501 excluded for retained long
embedded command strings. All six embedded prompt files match their source.
Retention correction: `repro_tests.xml` does exist and records five passes and
51 scratch-setup errors. The reviewed `verification.json` incorrectly marked
that initial XML unavailable; it is preserved unchanged rather than rebinding
the passed-review/launch chain. The separate final 60-test passing receipt is
unaffected.
The previous 68-test scientific suite and full-data validation were rehashed,
not rerun: all 27 scientific files, the source ZIP and remote launcher are
byte-identical to the reviewed B bundle.

R2 bindings:

- transport: `1b118856ddb456abc459f1383ca96b11d885f8fd2f310b3d1b9745803c24d8b1`;
- tests: `dd4e8df1945babeebc611224a4aa962cef5626da7de0ed22a21ee64776d392cf`;
- final test XML: `e33782c01b720e12558334b8dc63ec85297be0b3cdadf579c5fdfae45bc2833c`;
- payload: `2f22fc552a7161fb570c56a5348026564d99707b68097f6f3b2253118baf8469`;
- six-file prompt: `8346a210a2e74617aec82d2e224fa6030762ac2468a5ccbfd2d7db985866fde2`.

The same-scope Astra review completed at 16:17:13 China time, exit 0,
`turn.completed`, zero tool calls: **PASS, zero required revisions**. The bound
completion receipt is `1355b51d6a5b86f1c20469fef305067e75819e25f1ff07980084136a2f397c7f`.
Fresh checks before and after upload passed: matching runtime/input hashes,
idle GPU, 2.56 GiB disk and 11.05 GiB conservative host-memory headroom. Every
uploaded source and the unchanged launcher matched the approved bytes.

##### Completed Matched-Start Rollout

The one evaluation launched at 16:19:24 and completed at 16:21:01 China time
September 10, exit 0, in 95.966 s (92.765 s work plus 3.201 s final provenance).
All 45 files / 73,701,589 bytes were retrieved with matching before/after remote
inventories. Transport integrity, packet validation and source/input stability
pass. All six checkpoint sentinel inputs reproduce raw and restricted outputs
exactly. Exit 0 means completed diagnostic execution, not successful rollouts.

The retained `audit_rollout.py` is run with native local Python. It adapts the
old eight-path-model audit for the 36-index teacher mapping and completed or
censored outcomes; it makes no model, solver or network calls. Its first
recomputation passes: all 1,136 per-step rows, 60 noninitial snapshots, original
reference-state blocks/float32 store, per-case summaries, physical/Fourier
identities and censoring are checked. The old 45-file diagnostic packet is
also rehashed for the matched comparison. Maximum normalized Fourier-band
recomputation difference is `2.22e-16`. This is a primary-integrator audit, not
another independent reviewer. Its FFT tolerance is inherited from the prior
audit; no new tolerance adjustment was needed here.

All errors below are pooled relative L2 over the **same first 32 transitions**
and all paths in each role, not averages of frame-wise ratios. One-step errors
use reference inputs; rollout errors use autonomous `next_state` feedback.

| Metric, first 32 transitions | Original 8-path model | Expanded 32-path model |
| --- | ---: | ---: |
| Original-eight training one-step | 0.224895% | 0.331144% |
| Original-eight training rollout | 26.2935% | 73.2452% |
| Same-four validation one-step | 2.448379% | 0.696336% |
| Same-four validation rollout | 35.2606% | 74.8571% |

Validation one-step improves **71.56%**, while validation rollout is **2.123x
worse**. Each of the four validation paths has lower matched-window one-step
error and higher rollout error; all twelve paths have worse H32 rollout.
Training H32 one-step also worsens, so do not describe its H32 comparison as a
one-step ranking reversal. At the initial transition, both roles improve and
the expanded model's training/validation errors are 0.06983%/0.07284%.

All eight training paths reach the amplitude guard at steps **87--108**, and
all four validation paths at **84--99**, versus **420--509**/**409--471** for
the original model. The pooled median guard step falls from **435.5 to 94**.
No path completes H128 or H512 under the expanded model; those rollout errors
remain null. No path is dropped, reset, extended or imputed.

Interpretation: this fixed, single-model-seed comparison supplies a validation
one-step/rollout ranking reversal after broader clean coverage. The full-window
one-step gap remains the previously audited 0.2571% all-32 training versus
0.4808% validation, with an early-transient residual gap. This does not establish
that more data generally harms stability, that coverage is fully sufficient,
or that manifold drift/normal amplification causes this outcome. The matched
update budget gives different effective passes through each training set.
The shared numerical restriction is present in both models; raw PCNO and a
learned corrector have not been compared here.

Next decision: a small, explicitly specified common-input response/forcing
diagnostic, then representative matched corrective interventions if qualified.
Keep measurements separate from causal or prospective claims. No new training,
solver-response bank, correction matrix or protected reveal follows automatically.

Final R2 evidence bindings:

- result: `9c270eed51459841d2bf382d5d837ef2fd4b210df2867080e98e403d2c5553e8`;
- manifest: `63ed02808bd7ce68126e685c7e51e9ce18156aae606ce1c695bfacd57883cd9f`;
- launch receipt: `7742b57afc426ba112f2622dfefbc761bd0df0a7172fabe8e24198b0f5babda1`;
- retrieval receipt: `e05ae371d460f53ab505fea4506d411a7b6527f693d0a3ab3bd5372518d1e92d`;
- audit source: `ede156a355f7aa1f14852b48c0aba35fbe5c1fa71d4079a1e02f7bebc5e84599`;
- recomputation audit: `b2d3c349707a78fdedb752a7664cd1f0576ead09026ea6c4dbba06d718fcadec`.

No scientific source was edited, no new model trained, and no solver, protected
population, paper, Git commit/push, remote cleanup or other resource was touched.
The job has exited. Preserve all reviewed bundles and result evidence.
The final local cleanup removed only this turn's five pytest scratch directories
(1,509 synthetic files, 8,061,072 bytes). They can be regenerated by the tests;
all XML reports, review records, source archives and scientific evidence remain.

### September 7: B Closeout And C Recovery

B ended with `incomplete_budget`, exit 1, at 08:00 China time. Its packet has
160 files (1,438,029,724 bytes): five complete training trajectories, training
seed 2026090616 through step 16, and six unstarted roles. The 2,582 retained
states provide 2,576 training transitions and no development transitions.
All 159 payload hashes, twelve frozen source bindings, retained states and
available query metrics passed independent audit. No required eight-step
continuation is present, so the full-population gates remain unresolved.

Across 35 complete queries, maximum clean spatial discrepancy is 5.532e-5
and discarded fine-state fraction is 6.819e-5 (both limits 1e-3). The largest
of 36 available temporal discrepancies, including the partial query, is
5.813e-9. These are partial numerical results, not population qualification
or evidence about learned rollout behavior.

Windows power events identify lid-triggered standby from 22:55 September 6
until 08:00 September 7, apart from brief transitions. The elapsed-budget
clock counted this interval; the external Windows wait did not enforce an
awake-independent deadline. This is infrastructure incompleteness, not a
failed numerical threshold. The original 4.5--6 hour estimate was also
optimistic; do not reuse it as the recovery ETA.

Evidence: B manifest
`27ad081d6b2875c9d613b2978ce5d0aee0f698e3ba380775d26c0dccce281815`;
`cm_next_kf_pop_20260906b_launch/independent_population_audit.json` SHA256
`e31afb7cf11024f53f566bdab6bdda7cb52a51f00e89a05652c91454c3fa7d14`;
timing receipt `timing_audit_20260907.json` SHA256
`270b10bcbc017172aba0cfa827457dd7b01d4f9df83263517b5ffe6766613568`.
These paths are relative to `artifacts/time_dependent_no/`, with the timing
receipt beside the audit. B's static interpretation text overstates coverage;
its structured status and independently verified inventory govern. Preserve
the original bytes, including that limitation.

C keeps all seeds, roles, physics, horizons and numerical limits unchanged.
Its restart gate and preservation contract are in the experiment plan.
The September 7 owner continuation approves this recovery on the existing
AutoDL instance, not new model training or protected-population access.
The read-only resource check found no GPU job or same-root process, a 25-core
CPU quota and 9.57 GiB free. Another agent's resource remains untouched.

C implementation is frozen: generator SHA256
`3b5dd8263ea336b8304bd186028f77ab3fdfe76ca68d0ebdc2a132714e2f4c06`,
test SHA256
`96de4b04c5aa421683b888f445295dc485d8ee5d2e1c86bbabeee87350dc3e71`.
All 13 final-byte synthetic tests passed in 48.13 seconds; Ruff and independent
code-to-contract review pass. The transport passes Python and generated-snippet
compilation plus focused Ruff checks. No scientific replay was run locally.

The exact upload is 185 files, 1,438,379,407 bytes: current source closure,
complete B packet, archived B sources and pinned independent audit. It excludes
the paper, credentials, private context and protected populations. Manifest
SHA256 is
`8ccfd03462890f7ec78c5daca834a8adbf1ec8095362e0453caad7de3c8ff41a`.
Local operational evidence is in
`artifacts/time_dependent_no/cm_next_kf_pop_20260907c_transport/`:
`payload_for_approval.json`, `source_snapshot.zip`, `code_review.json` and
the read-only preflight receipt. These artifacts preserve exact deployment
and review evidence; no additional maintained runner was added.

The first upload attempt was rejected before execution because prior
exact-payload approval did not cover this new private-source/data payload.
The owner then explicitly approved the named 185-file payload, existing AutoDL
destination and restart-gated launch with "Yes, please proceed." Source,
transport and payload-manifest hashes were rechecked unchanged before retry.
Upload completed with all 185 remote hashes equal to the approved manifest.
The scoped CPU-only process launched at 09:23:39 China time on September 7.
The new launch check found no other GPU/same-root job, 8.23 GiB free disk and
5.93 GiB cgroup memory headroom. Sources and numerical settings are unchanged.

The runtime gate completed all 26 calls in 113.91 seconds. Maximum saved-answer
relative L2 discrepancy is 1.132e-15 against the frozen 1e-10 limit; native
initial-state discrepancy is 2.130e-16. All within-runtime repeat checks pass.
The 144 inherited complete-case files were separately rehashed byte-identical.
At the 09:26 health check, seed 616 had advanced to retained step 24; its
partial step-16 query had been recomputed wholly in Linux. The six not-yet-run
case files still contain B's inherited `incomplete_budget` placeholders; they
are not new C failures. Use origin metadata and actual execution progress.

Operational receipts in the same transport directory:
`upload_receipt.json` SHA256
`f3226ae77488aabeef9f542df69e7da0962c0fb27ed2cb01ed2d16bd45ace300`;
`launch_receipt.json` SHA256
`8ba6a2c7aa5d94f09bf08c25a0766c1093b312a2a593b5f0b70a99862d831d1e`;
`launch_health_20260907T012654Z.json` SHA256
`166941227a98616b4ee406c7ffaedba29b009a9572def47a189d6a6968dcef8c`.
An independent local receipt audit verified all 26 call identities, parent
reference hashes/substep counts, repeat output hashes, both restart feedback
chains, twelve source bindings and the inherited file map. Replay JSON SHA256:
`a18e7aa304be7100a1ff2581a0a7e80ce34d9eeb7d90b47de0b02711033b4604`.
This launch-stage review did not decode the remote replay arrays.

At approximately 09:30, the process remained healthy and seed 616 had retained
step 224. Provisional finish is **11:00--11:30 China time September 7**,
excluding retrieval and final audit. Linux median seconds/RK4 substep are
.0263064 (N256) and .136132 (N512). The retained-work estimate of 112,634
N256 and 16,377 N512 substeps gives 5,192 seconds central post-gate compute;
using upper rates, a 25% allowance and five minutes of bookkeeping gives
7,502 seconds. This is a planning range, not a confidence bound; shared-host
slowdown can extend it. The eight-hour compute cap remains unchanged.

Next action after completion: retrieve the exit/result/manifest and all new
arrays, verify both inherited and generated evidence, then evaluate every
unchanged numerical gate. Only qualified data can proceed to the actual N256
batch-eight resource check and a separately frozen competent Clean recipe.
No model training or manuscript change occurred. Full population qualification
still requires completion, retrieval and independent array/threshold audit.

### Completed C: Clean Population Qualified, Model Question Still Open

C finished at 10:43:50 China time September 7, exit 0, after 4,806.26 seconds
(80m06s) of new execution including replay/copying. This excludes inherited B
case timings. The 409-file packet is 3,628,090,961 bytes. All 415 retrieved
files, including operational receipts/logs, passed checksum checks; 147 were
reused from byte-identical local B evidence. Root separately rehashed every
retrieved file and checked twelve live source/launch/result/manifest bindings.

Independent array audit verified all 408 manifest-listed payloads twice,
6,156 states in 228 blocks, 81 clean queries, 32 continuation steps (four
endpoints), and all 26 replay arrays. The five completed B cases and seed
616's seventeen-state prefix are byte-identical; its interrupted query was
wholly recomputed in Linux. No partial/error rows remain. Coverage is eight
training trajectories / 4,096 transitions and four development trajectories /
2,048 transitions, each through step 512 (T=25.6, N256).

| Independently recomputed quantity | Maximum | Frozen limit | Location |
| --- | ---: | ---: | --- |
| Clean spatial relative L2 | 6.1702e-5 | 1e-3 | Training seed 617, step 16 |
| Discarded fine-state fraction | 8.1522e-5 | 1e-3 | Training seed 617, step 16 |
| Eight-step endpoint relative L2 | 2.1151e-5 | 2e-3 | Development seed 621, anchor 16 |
| Clean temporal relative L2 | 1.4260e-8 | Diagnostic only | Development seed 622, step 64 |

The largest intermediate continuation discrepancy is 6.3786e-5 at seed 617,
anchor 15, continuation step 3; the frozen gate concerns endpoints. All recorded
metrics reconcile independently. This qualifies the sampled clean finite-time
population under the declared checks, not every retained state or displaced input.

Packet: `artifacts/time_dependent_no/cm_next_kf_pop_20260907c_transport/retrieved/packet/`.
Final manifest SHA256:
`9ae1c66f2d7b8a2cdb105d7d57c71ee7811e64cc7f17203db7bcb51441c57f9c`;
result SHA256:
`824bd16c259030abe054a352d0ac1aa7d336a89749d6d5ef228550e451b4291d`.
The transport directory owns `independent_population_audit.py` and its JSON
receipt (SHA256
`c4e42a5609dd0c7dd317bcd9a94b1badedde2db83db3fdc92ade2b941e34fbc5`),
plus `independent_population_analysis.py` and its JSON receipt (SHA256
`bd51287a6aebb24c058c91127b2707c9fd710211223df1b464aa840b23d74528`).
These ignored one-study scripts preserve reproducible verification. Run the
auditor directly after retrieval; run the analysis with
`--manifest <the verified C manifest hash above>`. Neither advances a solver
or loads a learned model. Physical diagnostics independently agree to 7.81e-16
relative discrepancy; an analytic Fourier-mode formula check also passed.

Physical interpretation: mean energy falls from 4.45328 to 0.52819 (seed SD
0.10296), and mean enstrophy from 62.86982 to 5.19826 (SD 1.38234). Palinstrophy
initially rises from 971.40 to peaks of 1,405.70--1,617.62 at steps 14--22.
This is substantial finite-time evolution with early spatial-gradient growth;
it does not establish chaos or stationarity of a later regime.

Persistence predicts the initial state throughout. Its endpoint relative L2
errors at horizons 1/32/128/512 have seed means .11981/1.78830/3.56740/3.33616.
At horizon 512 the seed SD is .58874; the global space-time relative norm is
2.64196, not a mean-per-time rollout score. With initial rather than target
RMS normalization, endpoint error is .92977 +/- .02083. Late amplitude decay
therefore amplifies target-relative persistence error; a zero predictor has
relative error 1. Persistence failure alone does not certify a hard learned
benchmark. Retain zero/persistence controls and amplitude-sensitive diagnostics.

Mean adjacent reference-state change is 19.52%, 9.34% and 5.46% in the first
64, next 192 and last 256 transitions. These are reference changes, not
learned one-step errors. Development seeds 621/623 finish above the observed
training energy range despite sharing the same initial law; they are not
exogenous-OOD initial conditions. An empirical training-range boundary is not
a physical admissibility boundary.

Result-to-claim, in-session independent Codex review: **yes** for the frozen
sampled clean numerical-readiness claim; **partial** for suitability as a
mechanism-separating paper case. Nonlinear-manifold geometry, Clean-PCNO
instability, correction rankings and prospective diagnostic value remain
untested. Unpublished results were not sent to an external AI service.

Next: measure actual N256 batch-eight cost, freeze a competent Clean recipe,
and evaluate its one-step/rollout gap with the stated controls and time-window
diagnostics. Qualify the actual displaced-input law before relabeling/off-path
response claims. The 256-update, 32-pair debug fit is not this baseline.
Retrieval/analysis launched no new training or solver campaign, opened no
protected population, and made no paper edit or remote cleanup.

### Actual-Batch Resource Check: Completed And Retrieved

`CM_NEXT_KF_CLEAN_RESOURCE_20260907A` is implemented in
`scripts/time_dependent_no/fit_kolmogorov_clean.py`, with its focused test.
It is a fixed sixteen-update measurement, not the full Clean trainer or a
competence result. The experiment plan owns its architecture and timing recipe.
The completed debug and population sources are unchanged.

Verification: all 53 focused CPU tests pass (10 new resource tests plus 43
adapter/debug/smoke regressions), Ruff passes, and independent code/transport
review passes. Root also loaded the actual C packet locally without running a
model: all artifact/source bindings and 6,156 canonical states pass. The loader
keeps one 1,613,758,464-byte float32 store; scale is fitted from the 4,096
original float64 training inputs only: `4.063412684964112`.
Its float32 store SHA256 is
`94e2316b5afa467eb315f215116ea17279bd11cdec2c8a66db938a799d027120`.
The local validation/load took 105.02 seconds; this is not GPU throughput.

Read-only AutoDL preflight passes: RTX 5090 GPU idle, 5,208,391,680 bytes free
disk, no process referencing the completed C or proposed resource root.
Conservative RAM headroom is 11,255,500,800 bytes after crediting only inactive
file cache minus dirty/writeback/shared pages. Raw cgroup headroom is smaller;
this estimate is not guaranteed reclaimability. No cache manipulation, deletion,
new rental or access to the other agent's machine occurred.

The ignored attempt directory
`artifacts/time_dependent_no/cm_next_kf_clean_resource_20260907a_transport/`
contains the one-attempt transport, preflight receipt, source snapshot,
`run.sh` and exact `payload_for_approval.json`. Invoke the transport through
the existing WSL environment with `launch`, then `status`/`retrieve`; it
requires unchanged payload/source bytes and fresh capacity checks.
The payload is **12 Python source/test files, 175,154 bytes**, reusing C data
and its source root read-only. It excludes datasets, checkpoints, documentation,
paper and private context. The wrapper has a 900-second timeout and 60-second
termination grace; this is a safety cap, not a measured ETA.

Frozen SHA256 bindings:

- runner: `fc89f14300f49e859b2c0214b0edbb184147e42f383b87e281da17e08c36a912`;
- focused test: `6d0d63b1e99a75fc87520faa8d488f5f1d3cfd041da4505697c9c4ef3362eb00`;
- transport: `49304aad6ce3edb188d3009a1fa31b466829b3cde48a920c5bc37a80a480ee09`;
- exact payload manifest: `0f24388ba98201e2f1c93d1b1b5250caff7609ca43120282a6c903c406aa0eaf`;
- source archive: `356420023c0313cb2132f417d05d080e5887bf7beaffe241106bcae851939543`.

The initial launch request was blocked pending exact private-source approval.
The owner then explicitly approved the named payload and bounded probe. The
unchanged twelve files were uploaded and rehashed, with fresh capacity and
process guards before launch. The frozen approval file's original pending
label is preparation history, not current authorization status; its bytes
remain unchanged.

The probe launched at 15:40:37 China time September 7 and finished at 15:41:55,
exit 0. Runner elapsed time was 76.44 seconds, including parent validation and
final integrity checks. All sixteen optimizer updates and both inference
measurements completed with finite losses, gradients and outputs. Nine result
and operational files (78,875 bytes) were retrieved and independently rehashed; every live,
archived, uploaded and result source binding agrees. The full float32 data-store
hash and float64 train-input RMS exactly match the prior local loader audit.
An independent in-session Codex reviewer reproduced the permutation, timing
summaries and all receipt bindings. No checkpoint or predictions were retained,
so this verifies logged finiteness, not tensor-level loss recomputation.

| Measurement | Observed value |
| --- | ---: |
| Data validation/loading | 37.5851 s |
| Warm update, compute only | 0.197647 s |
| Warm update, gather/transfer included | 0.198686 s |
| Warm inclusive update range, 13 observations | 0.198275--0.199017 s |
| Batch-eight inference, compute / inclusive | 0.081631 / 0.082198 s |
| Batch-one inference, compute / inclusive | 0.010426 / 0.010576 s |
| Peak GPU allocated / reserved | 6.5695 / 9.3027 GiB |
| Peak process host RSS | 3.2172 GiB |
| Parameters / state-dictionary tensor bytes | 10,298,053 / 50,894,040 |

The first three training updates are excluded only from warm timing; peak
memory includes model construction, cold Fourier cache and Adam state. Each
inference figure is a ten-call median after three warmups. The 128 training
indices come from the fixed without-replacement permutation. Batch losses span
0.001899--0.059183 and gradient norms 0.002265--0.156241; different batches
have different state amplitudes, so this is not a convergence curve.

At measured throughput, 32,768 updates cost approximately 1.8085 hours.
One full train/development teacher pass is approximately 63.13 seconds;
nine such passes plus loading bring the estimate to 1.98 hours before
checkpointing and other unmeasured overhead. Allow roughly 2--2.5 hours for
the proposed full pilot, not a guaranteed deadline. A 16,384-update extension
would add about 0.90 hours of update work. These are extrapolations from a
short run, not measured long-run performance or costs for corrective arms.

Packet: the attempt transport directory's `retrieved/packet/`.
Final artifact manifest SHA256:
`c268e008838f94db49f22c66bcc465eb77af967d70b67aa5eba83505b6d21089`;
result SHA256:
`b1001e94e3544396b3bf52b15a86c03cf014552b80750d94b1c230b03311e4f4`;
retrieval verification SHA256:
`8f0644f14fcbbf0a9dc9c28894b29889ed062ce5854b6fb65dd08713b4562997`.

Conclusion: the actual N256 batch-eight PCNO path is resource-feasible on the
approved instance. It is not a competent Clean model or rollout comparison.
No checkpoint was retained, no development prediction or protected outcome
was evaluated, and no solver call or paper change occurred. This probe has
exited; do not rerun its launch command. The full Clean successor below has
its own identity and source snapshot; it does not amend this completed probe.

### Full Clean Pilot: Completed And Audited

`CM_NEXT_KF_CLEAN_20260907A` implements the frozen recipe in the existing
entry point with `--phase clean`; the default resource behavior is unchanged.
Independent AST comparison against the preserved resource ZIP confirms all
thirteen pre-existing resource/loader functions and classes are unchanged.
The completed debug fitter, C population, reference solver and paper are untouched.

Verification: 58 focused synthetic CPU tests pass in 35.43 seconds, and 43
adapter/debug/smoke regressions pass in 16.04 seconds. Ruff passes on the runner,
test and transport; independent final-byte code-to-contract review passes.
Tests cover complete shuffled epochs, schedule boundaries, pooled errors and
all twelve development/time-band cells, undefined denominators, the strict
five-percent extension boundary, terminal replay, exact CPU optimizer/sampler/RNG
continuation, both tiny full-run branches, and honest nonfinite/provenance
failures. These are engineering tests, not a trained PDE result.

The run retains one rolling checkpoint and at most two fixed terminals,
per-pair float64 squared errors, and four raw/restricted replay sentinels per
evaluation. It does not retain every prediction or select weights by rollout.
Raw/restricted terminal replay tolerance is 1e-6 relative RMS; bitwise CUDA
continuation is not claimed. An execution can complete but still fail to
qualify the Clean baseline; interrupted final packets remain incomplete.

Read-only AutoDL preflight at 19:34 China time September 7 passes: GPU idle,
5,208,084,480 bytes free disk and estimated conservative RAM headroom
11,314,577,408 bytes. No cache manipulation, remote write, new purchase or
access to the other agent's resource occurred. Repeat capacity checks at launch.

The exact new upload is **12 Python source/test files, 228,199 bytes**, reusing
the qualified C packet and archived C sources read-only. It excludes datasets,
checkpoints, paper, documentation, private context and protected populations.
The ignored `artifacts/time_dependent_no/cm_next_kf_clean_20260907a_transport/`
holds the one-attempt transport, source archive, launcher, approval manifest,
preflight and code-review receipts. All twelve source files and five generated
remote code blocks compile offline; archive/current-source hashes agree.

Frozen SHA256 bindings:

- runner: `12355c814911a2fd01e42b3e34f68fcee8200110cf8fa227afd4fe9e1f067e4b`;
- focused test: `55bf317396cc82320eb493216803e2665e6784d919817ed7f3f774eb420d199c`;
- transport: `e086c759628ab5a159325c8662ad24b17666c92a1b66501390e66389ff7d4aac`;
- exact payload manifest: `798efb9961a225b9e94a0234ecdb34a1bac064b3b550c275dca2187b12a4d92b`;
- source archive: `24caba0df98feb9b51d75ea567aca111aa56011efdaf46e6bb224697e3b6550e`.

The owner explicitly approved this exact payload and bounded pilot in the
next reply. The unchanged twelve files were uploaded and rehashed, with fresh
capacity/occupancy checks before upload and launch. The process launched at
21:26:10 China time September 7. Both checks found an idle GPU and no scoped
conflicting process; prelaunch free disk was 5,207,818,240 bytes and conservative
RAM headroom was 11,315,027,968 bytes. An independent receipt audit passes.
The original preparation/review files' pending and not-launched labels remain
unchanged history, not current authorization or execution status.

Operational receipt SHA256 values in the same ignored transport directory:

- `owner_approval.json`: `0646dc2ef6601c7eda901faafd0a86bf21dca6524c674d16522259acc7750ab6`;
- `upload_receipt.json`: `8a1000d8e3974058b9078766ef723079a8047a3ca7d6672dfdbbf478f4878910`;
- `launch_receipt.json`: `52fa7a5f44d3f046527ca5778351435e5b69ef0e3ec7b97d7b39281fc81d8c8d`.

The first health check at 21:28:16 found the initial evaluation active, GPU
utilization 98%, all twelve remote source hashes unchanged, and no traceback
or exit receipt. The 21:31:53 check confirms training at update 1,037, with
finite recent losses/gradient norms and a last-100-update median of 0.199868
seconds. GPU utilization is 99%, using 7,695 of 32,607 MiB. The complete initial
teacher pass took 64.617 seconds. These are health/timing observations, not
a competence or convergence result; the initial zero-head evaluation is
essentially the one-step persistence control.

Independent local health audit passes: all 871 captured updates (168--1,038)
reproduce the seeded sampler, epoch cursor and learning-rate schedule. Data,
train-only scale, float32 state-store hash and checkpoint identity match the
frozen inputs. Initial raw-error sums equal persistence, and restricted-output
ratios differ by less than 1e-6. The status snapshot at update 1,037 and later
log tail through 1,038 were read sequentially; `progress.json` still refers to
the update-zero checkpoint, not a stalled optimizer.

Health receipt `launch_health_20260907T133153Z.json` SHA256:
`66ef291fa45687e25a49d29f964f1b36f7629b8878053247b13b625ec7d40d85`.
The launch-time estimate was 23:30 September 7--00:00 September 8 China time,
or approximately 00:30--01:00 with the extension. The actual completion is below.
The five-hour watchdog was a safety cap, not a convergence criterion.

#### September 8 Result And Interpretation

The run finished at 00:25:22 China time September 8, exit 0, after 49,152 updates
and 13 full teacher-forced evaluations. Runtime was 10,750.37 seconds (2.986 h).
All 27 retrieved files (471,774,955 bytes), including the 22-file result packet,
passed independent hash/provenance checks. All 409 C-parent files and both
twelve-file source inventories remain unchanged. The complete update ledger
reproduces 96 seeded epochs and 393,216 training-pair uses, with finite logged
losses/gradients and the exact declared LR schedule. No development pair entered
an optimizer update.

| Evaluation | Train pooled relative L2 | Development pooled relative L2 | Worst of 12 development/persistence ratios |
| --- | ---: | ---: | ---: |
| One-step persistence | 14.43897% | 13.98773% | 1 |
| Update 32,768 | 0.23246% | 1.74935% | 0.14720 |
| Accepted update 49,152 | 0.20186% | 1.61630% | 0.13622 |

The 24,576-to-32,768 development improvement was 5.42196%, so the frozen
rule required the single 16,384-update extension even though competence already
passed. The final 40,960-to-49,152 improvement is 4.11801%, below the 5% boundary.
The accepted terminal meets both readiness targets: development error <=2%
and every trajectory/time-band learned/persistence ratio <=0.5. This is a
declared engineering stopping decision, not proof of optimization convergence.
The extension reduced development relative L2 by 7.606% versus the base terminal;
no rollout outcome selected the weights.

Independent metric recomputation covers all thirteen NPZ snapshots and 650
aggregate groups; 10,193 scalar comparisons exactly match the archived summaries.
All 416 retained sentinel squared norms agree to relative discrepancy <=3.30e-16.
All three checkpoint model/Adam/sampler/RNG states pass inspection; the latest
checkpoint and accepted terminal have identical decoded states. Eight independent
CPU batch-one sentinel predictions from the two terminals reproduce saved GPU
predictions within 5.996e-7 relative RMS, below the 1e-6 replay tolerance. No
new rollout, optimization or solver call was needed for these audits.

Interpretation and limitations:

- The early input-step band [0,64) carries 97.575% of development learned squared
  error, versus 12.5% of pairs and 44.661% of target squared norm. Pooled early,
  middle and late errors are 2.3891%, 0.4279% and 0.2392%; their learned/persistence
  ratios are 0.1292, 0.0403 and 0.0345. Early dynamics are the main remaining
  weakness even after normalization by persistence. All four early-band errors
  exceed 2%, but 2% was the pooled criterion, not a per-band cutoff.
- The development/train error ratio is 8.007. The four development-trajectory
  errors span 1.5688--1.6802%, so the result is not driven by a single bad path;
  it is still one training seed and four reused development paths, not independent
  confirmation or evidence that more training/data cannot help.
- Qualification attaches to the complete restricted map. Raw development error
  is 2.09144%, versus 1.61630% after the shared mean-zero rectangular 2/3-band
  restriction: 22.718% lower relative L2 and 40.275% lower squared error. The
  loss is applied after restriction, so the removed output component is not
  directly penalized. This is expected numerical-state restriction, not a learned
  data-manifold projection or an independently trained raw-PCNO comparison.
  Keep it identical and explicitly attributable in every future arm.
- Independent in-session result-to-claim review: competent Clean engineering
  gate supported, with high confidence; representative correction-necessity
  case only partially ready; prospective predictive-ranking claim untested.
  No rollout accuracy/instability, manifold drift or off-trace response control
  has been established. Untested claims are not falsified claims.

Recommended next step, not executed by this retrieval/analysis task: freeze the
accepted terminal and its training budget, then evaluate its full declared
development rollouts with the same numerical restriction and one-step controls.
Retain early/middle/late errors, spectral/physical diagnostics and raw versus
restricted outputs. An informative rollout gap and separately qualified displaced
queries precede the corrective comparison. Do not weaken Clean or select a
checkpoint to manufacture failure. The paper and other agent's resource remain
untouched.

Evidence in `cm_next_kf_clean_20260907a_transport/`:

- final packet manifest: `a6ffc5512a3431095dbc588acf2f86b0b070ef647cd3ddccf8e8e2739d83fcd4`;
- final result: `48b58dab1967228779f74c7a85977cc0e88fe0d58aea3c627f416442973cb349`;
- retrieval verification: `b6b1f9a5c093557cb80c7e5a4a665779b5fe38d108affe957921f6ba010328dc`;
- accepted checkpoint `terminal_049152.pt`: `56469747abafdfd34dbedb3cf1c843a54f5ec8dfeaf1feda383756060eeec454`;
- `independent_training_audit.json`: `30d85fc66003987952a6ce98f82f77a3233e670181b0e66112d3061364e53819`;
- `independent_metric_audit.json`: `0f1711d4317e389185819afa89ca45eb622e18e818151d6086fa63da63317e96`;
- `checkpoint_replay_audit.json`: `6fbe788326137a357bf47a1615cf057849c238b5d03ea60fb09c63f246301b7c`.

### Completed A Screen

Packet: `artifacts/time_dependent_no/cm_next_kf_r0_20260906a/`.
Artifact-manifest SHA256:
`d2101f544b9e2d55e52165a9a1d98d24fbc02ba481ea7dff71f23429e286a464`.
At A closeout, independent audit verified the exact three-file inventory, both output hashes,
all three source before/after/live hashes, four completed cases, twenty query
rows, and equality of incremental and final case records. The reference solver
source remains unchanged at
`6c3c1938318deb52c8873243692bd06daed4f9958e010556fee3c8666fe3dbda`.
Runtime: local CPU, Python 3.13.12, NumPy 2.4.6; 74.983 seconds. All four exact
restart/repeat and raw-noncanonical-rejection checks pass.

| Diagnostic | Viscosity .01 | Viscosity .005 |
| --- | ---: | ---: |
| Clean spatial state discrepancy, relative L2 | .000566--.000710 | .005824--.006464 |
| Low-probe spatial response discrepancy / input RMS | .001793--.001887 | .02295--.02951 |
| High-probe spatial response discrepancy / input RMS | .3541--.3693 | .5675--.5786 |
| Maximum high-probe temporal response discrepancy / input RMS | 8.039e-6 | 8.549e-5 |
| High-probe discarded fine-state fraction | .00470--.00483 | .01140--.01200 |

Spatial response compares coarse and restricted fine responses from identical
lifted queries, with half time step on both grids. Temporal response compares
base and half time steps on the same grid. These are vector-response differences,
not merely differences between gain magnitudes. Fine-restricted gain is not the
full fine-state gain: discarded response content is material.

Conclusion: restart and local time integration are usable; N64 is not spatially
qualified for the selected high-retained-band response assays. Do not promote
this to a training population or interpret band directions as tangent/normal.
This motivated the now-completed B refinement below. A's exact three source
files are preserved in `cm_next_kf_r0_20260906a_source_snapshot.zip`, adjacent
to the A packet, with CRC/entry hashes independently verified. Archive SHA256:
`44b2b94ab922bb743ca44539a4007bd77fcfb37c8f86a4129d90f19cbcec9e33`.

Verification: 81 combined CPU tests pass (59 response, 13 retained reference,
9 pilot); Ruff and `git diff --check` pass. Independent random-formula checks
match weighted algebra on 100 cases, and legacy metric dictionaries equal
HEAD on the same cases. Review caught silent complex-to-real coercion in the
new API; it is fixed with 27 regression cases including imaginary NaNs.
During A, no learned model, old scientific dataset, remote machine or protected
outcome was accessed. Only fresh solver states were generated. The paper plan was
updated and locally archived; LaTeX sources and PDF were not changed or rebuilt.

### Completed B Refinement And PCNO Engineering Checks

B packet: `artifacts/time_dependent_no/cm_next_kf_r0_20260906b/`; manifest
`03aef5dd3055a99b8ac5eceed6c32ad3d4091bac0d06bd7a929636128d19e216`.
All artifacts, three source bindings, four cases, twenty rows and A parent
bindings independently verify. Runtime 543.66 seconds. Evolved anchors differ
from A; only within-run lifted-input comparisons are exactly paired.

| N128/N256 diagnostic | Viscosity .01 | Viscosity .005 |
| --- | ---: | ---: |
| Clean spatial state discrepancy, relative L2 | 4.34e-6--8.02e-6 | 4.75e-4--5.99e-4 |
| Low-probe spatial response discrepancy / input RMS | 2.30e-5--3.67e-5 | .00350--.00359 |
| High-probe spatial response discrepancy / input RMS | .001104--.001337 | .02312--.02998 |
| Maximum high-probe temporal response discrepancy / input RMS | 1.1724e-5 | 3.0385e-5 |
| High-probe discarded fine-state fraction | 4.23e-5--4.81e-5 | 8.80e-4--9.56e-4 |

Use viscosity .01/N128 for the bounded longer-time development screen; do not
promote these T=.2 anchors to a stationary regime or production population.

Periodic adapter verification: 28 focused tests; smoke runner: four tests;
combined affected suite: 113 passes. Independent review checked geometry,
forcing, normalization, response algebra, recurrence and fail-closed packets.
The 100-update CPU synthetic fit reduced MSE from .0101402 to 1.7990e-6
(99.9823%). Packet `cm_next_kf_pcno_smoke_20260906a_cpu`, manifest
`16c65b4ef93e28e0ad08e9281a8412e1b8aa13c61f7439d4e813332d8d5c46e8`.
Logs/metrics reconcile; final tensors were not retained for independent MSE
recomputation. This is optimization plumbing, not PDE/generalization evidence.

AutoDL received exactly eight source files (103,255 bytes), no data or paper.
The detached synthetic CUDA run exited zero; all sources and downloaded
artifacts verify. Its result is under
`cm_next_kf_pcno_smoke_20260906a_cuda_transport/retrieved/packet/`, manifest
`4f1fcbd7a28694ede18f32f2ce72bdd7a90bfd3606b544ef98b6ff7ff25fde07`.

| Grid | Median update | Median inference | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| 128 x 128 | 24.38 ms | 5.68 ms | .434 GiB |
| 256 x 256 | 33.68 ms | 10.42 ms | 1.235 GiB |

Preset: batch 1, width 64, four layers, 12 modes/axis, 10,298,053 parameters,
FP32/no TF32, 11 updates (first three excluded from timing median), 20 timed
inferences, cached geometry. Short synthetic throughput is not a production
ETA. The run body took 2.706 seconds, excluding transport/import overhead.

Storage preflight found only about 2.6 GiB persistent working space free.
No existing data were removed; production requires a measured storage plan or
an exact approved cleanup. Protected populations remain unopened. Only two
owned disposable pytest directories were removed; fixtures are reproducible.

### Completed Longer-Time Screen

Output: `artifacts/time_dependent_no/cm_next_kf_long_20260906a/`; private
launcher/immutable source snapshot/exit receipt are in the sibling
`cm_next_kf_long_20260906a_launch/` directory. No old population is read.
The source-bound hidden local process started at 08:17:11 UTC September 6 and
exited zero at 08:31:22 UTC, earlier than its initial planning estimate.
The packet has 42 payload files plus manifest, including 34 nonoverlapping
float64 trajectory blocks. Independent audit verifies all 1,026 states,
per-state/progress/checkpoint hashes, source bindings, complete coverage and
query/anchor alignment. Manifest SHA256:
`ebbfa9bf70eb4df892ce9d14e785d1c2f4e54e6c87e06da7d661424f39e8a939`.

| Frozen check | Observed maximum | Limit |
| --- | ---: | ---: |
| Clean spatial relative error | 7.3493e-4 | 1e-3 |
| Spatial response discrepancy / displacement RMS | 3.1346e-3 | 1e-2 |
| Temporal response discrepancy / displacement RMS | 6.2117e-6 | 1e-3 |
| Discarded fine-state fraction | 9.6908e-4 | 1e-3 |
| Independent eight-step endpoint relative error | 3.9826e-4 | 2e-3 |

All five declared checks pass; the discarded-state margin is narrow. Full
discarded fine-response reaches 5.8449e-3/input RMS and full-fine response
mismatch reaches 6.6324e-3/input RMS (reported, not additional gates).
Across steps 256--512, one-step persistence medians are 6.1%/7.8% and 64-step
medians approximately 123%. The fields evolve substantially, but energy drops
from 4.45 to .409/.697: this is a finite-time transient, not a demonstrated
stationary or chaotic regime.

Proceed to the clean debug fit and a fresh finite-time population/Clean
baseline. Both readiness seeds remain development/debug only. These checks do
not qualify all future inputs: Gaussian noise near the retained-band edge and
model-error queries need their own numerical checks after calibration.

The ten new runner tests and independent code-to-intent audit pass. Main also
reran the retained reference plus new runner: 23 passes; the prior affected
suite remains 113 passes (123 distinct tests across both scopes). Checks cover
real CFL refinement, independently evolved fine trajectories, retained anchor
hashes, full/discarded fine responses, complete gate counts, deadline and partial
block retention. The reference source is unchanged.

Frozen runner SHA256:
`0cff649a4d80bd9140d819e284697fdeace5f622e4bbd0161f0bac17975bfa02`;
test `575fb8805c7b13d4100af77b2f855c58451c948e07fa3fc5a1ebaa4895151f3e`.
The [experiment plan](EXPERIMENT_PLAN.md) owns the numerical limits. In
particular, discarded fine-response is reported without a frozen threshold;
passing the listed engineering limits cannot certify all response directions.

### Debug Fit, Fresh Population And Storage

The solver-labelled debug fitter is implemented and independently reviewed.
Eleven focused fixture tests pass; a separate checkpoint replay reproduces
every saved synthetic teacher prediction and all 32 recurrent outputs exactly.
The main fit/adapter/smoke/response suite has 102 passes. Runner SHA256:
`ea5a8e24bc7edf8f0d2b3a16a5d81aa140369199f167ed39bde14ad0d656e214`;
test `4303ff53df47c8229e7fd6c21d285c39b71febc7fe5b871486c99a64ece9678e`.

Auto-review blocked its private upload pending exact-payload approval. No debug
payload was uploaded and no real PDE model fit was launched. The local approval
manifest binds 58 files, 132,342,353 bytes: twelve source/dependency files, the
complete already-open longer-time packet, and its three frozen sources. It
excludes paper, credentials and protected populations. Approval-manifest SHA256:
`07340e576c75ddcd38a273bdfa73c5e790818c34841532c37375aeca6cb678e0`.
The payload and private destination are recorded in the ignored debug transport
directory; do not bypass this exact upload boundary through another launcher.

Fresh local population identity `CM_NEXT_KF_POP_20260906A` has immutable roles
and checks in the experiment plan. Nine focused tests and an independent
19-test/helper review pass; the latter independently recomputes fixture state
hashes, palinstrophy, peak selection, gate counts and incomplete-budget handling.
Runner SHA256 `e479ad5868262c17a064ed847619cb37de22abc2c6ace0b34513636e870da8c2`;
test `9accc504cb9e8408b0f70b5416236957fe9d25a82de66bc58176b8d6fb4a5d26`.
The source-frozen local launch is separate from the blocked upload and makes no
network/model/old-population access. Its output and launch receipts are under
`cm_next_kf_pop_20260906a` and `cm_next_kf_pop_20260906a_launch` respectively.
Generation completed in 2,235.79 seconds (37.26 minutes), ending at 17:45:27
China time. Exit 1 is the intended numerical-gate rejection, not an interrupted
runner. Independent audit verifies all 242 payload files, ten frozen/current
sources, 204 blocks, 6,156 finite canonical float64 states, 60 clean queries,
16 late rows, role assignments and all progress/checkpoint/state hashes.
Energy, enstrophy and palinstrophy were independently recomputed by FFT.
Manifest SHA256 `10103ba067a5e07cbe6bd1a2faa7f36c70392170da8400e1e4503a0d0c21ed5b`;
result `a37b20b0dcac4764776008288bdc511ed7c3873a07b813399d9ec726dd79e6ba`.

| Fresh-population check | Maximum | Limit | Result |
| --- | ---: | ---: | --- |
| Clean spatial relative L2 | .00600631 | .001 | Fail |
| Discarded fine-state fraction | .00719622 | .001 | Fail |
| Eight-step endpoint relative L2 | .00026071 | .002 | Pass |

All twelve peak-palinstrophy checks fail both spatial limits at steps 14--22
(time .70--1.10). At step 64, two clean-spatial and six discarded-state checks
also fail; checks at 0, 256 and 512 pass. Temporal discrepancy is below 6.15e-8.
This localizes a spatial-resolution/retained-band problem. The prior LONG
packet did not query its early peaks; its sparse-check pass remains true but
cannot qualify this full window. N256 is not yet qualified either. Next is a
bounded early-transient refinement check, not a full replacement population.
The final affected CPU suite passes all 143 tests (18.93 seconds); the new
runners/tests pass Ruff and `git diff --check` is clean. No real model fit has
been substituted locally for the blocked AutoDL action.

Owner-approved cleanup removed exactly 37 remote checkpoint copies from sixteen
old open-role smoke/tiny-fit runs, after preserving 7,774,712,833 checkpoint
bytes locally. The independent AFTER audit verifies all 105 backup files,
all 37 remote absences, 68 unchanged remote metadata files and the 39-event
journal. Work-volume free space is 10,560,585,728 bytes (9.84 GiB). Checkpoints
are recoverable from the ignored preservation directory. BEFORE SHA256:
`9f078f08c84a07518c484bbea06a6936a4e4d151088ac0dc1f37c457bddea87a`.
Receipts are in the existing synthetic-CUDA transport directory. Fourteen
system-volume B5 duplicates and the other agent's GPU resource remain untouched.

The owner's continuation was interpreted as approval of the preceding exact
debug-payload question, followed by a warning that another agent uses a separate GPU resource.
Auto-review nevertheless rejected the upload again on September 6 because
the trusted messages did not explicitly name this payload/destination. No
upload or fit occurred; request explicit exact-payload approval, not an indirect
launcher or local substitute. Local SSH configuration distinguishes
that resource from the selected AutoDL endpoint; endpoint identity and fresh GPU/process
checks must both precede the fit. Six local endpoint-guard checks, five synthetic
occupancy cases and remote-block syntax checks pass. The ignored transport
checks idle GPU/process state before upload and again before launch; it rejects
changed/excluded endpoints before connecting. No changes to the scientific payload
are needed. Its targets remain the declared finite-grid map; the fit cannot
repair or overturn the new population's spatial-qualification failure.

The native-resolution stress screen uses seeds 2026090617/0621 selected by the
largest within-role N128 peak discarded-state error, not model outcomes.
Its fixed contract and 90-minute cap are in the experiment plan; original
population bytes and thresholds remain unchanged. This is a local CPU action
independent of the blocked upload and does not touch the other agent's resource.

The owner's next explicit continuation accepted the exact upload scope. A
pre-upload process-directory permission failure was diagnosed without changing
the payload. The repair removed unrelated cwd-symlink inspection while retaining
fail-closed managed-command-line, GPU and fresh-attempt checks. Failed transport
bytes/log remain preserved. All 58 files then uploaded and rehashed, and the
debug fit launched at 11:46:10 UTC (19:46:10 China time) on an idle RTX 5090.
The terminal update was logged with finite loss/gradient. Retrieval succeeded
after one transient SSH banner failure: exit 0, completed result, all eleven
retained files rehashed. Independent prediction/checkpoint audit passed.

### Solver-Labelled Debug Fit: Complete

The audit verifies all 58 approved payload files, twelve source files and their
ZIP snapshot, the complete longer-time parent packet and three frozen parent
sources, all twelve saved array digests, 256 sequential update logs and all
34 Adam parameter states at step 256. Input-only RMS is 9.775870595829275 in
float64; the model stores 9.775870323181152 in float32. Saved normalized inputs
and targets match exactly. Five fixed teacher/feedback inputs replay from the
terminal checkpoint on CPU with maximum normalized RMS disagreement 2.534e-7
against saved GPU outputs (audit tolerance 2e-5). Historical pre-update losses
were checked for finite values/order, not reconstructed by training replay.

| Independently recomputed debug metric | Value |
| --- | ---: |
| Initial one-step global relative L2 | 18.4529% |
| Final one-step global relative L2 | 1.78745% |
| 32-step restricted-feedback global relative L2 | 32.3876% |
| Constant-persistence global relative L2 | 122.947% |
| Final rollout-step relative L2 | 53.4171% |
| Final persistence-step relative L2 | 184.468% |

These global metrics pool squared error and target energy over all 32 times;
they are not the arithmetic mean of per-step relative errors. Teacher/rollout
projection RMS is 0.537911/0.492156 (about 5.50%/5.03% of training RMS).
Raw proposals are saved along **restricted feedback**, not an independently
propagated raw rollout. The baseline includes shared numerical mean/band
restriction; no correction-free vanilla-PCNO claim follows.

On the RTX 5090, median batch-8 update time was 51.57 ms, with 1.75 GiB peak
allocated and 1.91 GiB peak reserved memory. Run-body time was 16.285 seconds,
excluding transfer, imports and parent verification. These N128 debug timings
do not fix N256 actual-batch training cost. The fit establishes finite-grid
optimization and in-sample composition only: no convergence, held-out PDE
accuracy, intervention ranking or production noise-scale claim.

The retrieved packet is under
`artifacts/time_dependent_no/cm_next_kf_debug_fit_20260906a_transport/retrieved/packet/`.
Manifest SHA256 `6126b0650fc93f02c569ac7a951acb35bd2c16fcfbe5e03c5f0adc3ce77ab8ee`;
result `c9234b9ab862c808002d315f33432700bd82ff217914c8a4265ab86e21a9a5a2`;
checkpoint `175178a76fedfa688f188029d7e80d65b664ed4cb071b6b566bfbcd88b44a211`;
independent audit `acfec3d3b91a8720f16b3a26cb935d8224a3cc341ff9e487202e21c90e8e59ec`.
The ignored transport directory retains the audit script/receipt for replay;
its source hash is recorded inside the receipt. No new remote or solver call
was used for this audit.

### Native N256/N512 Stress Screen: Complete

The new runner and focused test passed nine fixture tests, a separate 19-test
helper regression and the integrator's 32-test reference/trajectory/peak suite.
Ruff and diff checks pass. Independent code-to-intent review verified native
generation, metadata-only selection, identical query inputs, full-fine
independent recurrence and per-call partial evidence retention. Runner SHA256:
`cde2e942da8a05398c57ca4f944e6db2ecc8b78c9d9916aa7caf478130ca0fe4`;
test `bf6f1c6797cd66ff84471c3362c3379e10f929eb8acf845367ee92c4a8a6740a`.

The source-frozen local process started September 6 at 10:27:33 UTC and finished
at 10:52:14 UTC (18:52:14 China time), exit 0, in 1,480.718 seconds. Output is
`artifacts/time_dependent_no/cm_next_kf_peak_20260906a/`; the sibling
`cm_next_kf_peak_20260906a_launch/` retains the hidden launcher, source snapshot,
launch/process/exit receipts. Independent audit verifies all 35 payloads,
ten frozen/current sources, fifteen parent metadata files, 130 native states
in twelve blocks, seven clean queries and eight continuation rows. The earliest
native peaks are steps 15/16 for seeds 0617/0621; fixed/peak anchors deduplicate
to four/three queries. Independent FFT calculations reproduce all structure
and error metrics. Every fine continuation input equals the preceding full
fine output bitwise; discarded content confirms no intermediate reset.

| Recomputed N256/N512 check | Maximum | Limit |
| --- | ---: | ---: |
| Clean spatial relative L2 | 6.17017e-5 | 1e-3 |
| Discarded fine-state fraction | 8.15222e-5 | 1e-3 |
| Eight-step endpoint relative L2 | 9.56930e-6 | 2e-3 |
| Temporal relative L2 | 4.01500e-9 | Diagnostic only |

Manifest SHA256 `da4991f1e1d7c5f5b93f551aaf77d38f73c41dc08a61a878b53893ca358edfe6`;
result `a8dc60a662b11ea8fdab0cadc1b5fd165efd06af85093dd4aa135ee77a0a1ac7`;
independent audit `c441df3615371995eed662f577b2ba01de0990fe0c49510b531dcd0ee8958c92`.
The audit script/receipt live in the launch directory and call no solver/model.
Native macro steps averaged 2.954 seconds; fine queries averaged 60.078 seconds.
This supports the next full N256 population check only, not a qualified full
population or displaced-input law. The numerical reference remains unchanged.
The exact launch-time helper is preserved as `launch_at_start.py` at its receipt
hash; the live helper has only a status-display correction separating started
from completed query counts. The frozen scientific sources are unchanged.

The N256 B successor reuses the maintained population generator/test after
independently verifying all ten A source-snapshot files against both A receipts.
The archived A sources remain replayable; A is no longer claimed compatible
with the revised live generator/test. B uses a new identity, the passed PEAK
parent, the unchanged twelve role assignments and the fuller checks in the
experiment plan. Measured-cost forecast: 4.5--6 hours, eight-hour compute cap,
five-minute shutdown grace and 6 GiB free-space requirement. No B population
upload or model training is implied by the local generation contract.

B's implementation and independent code-to-intent review pass. The author and
reviewer each passed all eighteen focused tests; the integrator's reference,
trajectory, peak and population suite passed all fifty tests. Ruff and diff
checks pass. Generator SHA256
`a00a4b24c010015848d8bd57d31b717ef72f1b09f64ba21f97a189c12fabd8b0`;
test `03f1428349c3432db2fd984ed2b793d90853ea2ac992b794c8411d14a27c0b91`.
The hidden local worker launched September 6 at 11:59:38 UTC (19:59:38 China
time) from twelve frozen sources. Output is
`artifacts/time_dependent_no/cm_next_kf_pop_20260906b/`; the sibling `_launch/`
directory retains launch/worker/source receipts. The 20:10 China-time health
check found the first trajectory at step 64 with four completed clean numerical
queries and no failed-process receipt.
Provisional completion is 00:30--02:00 China time September 7, subject to later
CFL/query cost; the eight-hour compute cap and five-minute grace are unchanged.
No continuous monitoring is needed while progress is healthy.

Post-verification cleanup removed only four disposable local test-fixture
directories: 1,287 regenerable files, 10,273,843 bytes. Literal containment,
reparse-point and inactive-pytest checks passed; all scientific packets,
source snapshots, launchers and remote recovery copies remain intact.

The private manuscript's design section now defines the paired recovery and
dynamics errors and offline/prospective distinction. Local mathematical review,
compilation and visual inspection of pages 11--13 pass. The 34-page working
PDF is not a final submission; no new numerical result or citation was added.
No undefined references/citations or overfull boxes remain in the compile log;
pre-existing underfull table warnings and a Perl locale warning remain.

Updated: 2026-09-06

Status: current block and authorization tracker. Historical evidence remains in
`MECHANISTIC_DIAGNOSTIC_TRACKER.md`. This file does not authorize execution.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `PLAN_REVIEW` | Design exists and awaits owner review. |
| `IN_PROGRESS` | A named, bounded work scope is authorized and active. |
| `IN_PROGRESS_OPEN_ONLY` | Only open-role work is authorized; the row states whether it is active or complete, and protected reveal is not authorized. |
| `PLANNED_NOT_AUTHORIZED` | Scientifically selected; no implementation or execution approval. |
| `BLOCKED_BY_PREREQUISITE` | A named readiness condition is absent. |
| `EVIDENCE_ONLY` | Historical evidence, not a prospective active block. |
| `PARKED` | Preserved outside the critical path. |
| `COMPLETE` | The exact block and required packet are complete. |

Authorization is separate from scientific status.

## Active Blocks

| Block | Claim role | Status | Authorization | Next admissible action |
| --- | --- | --- | --- | --- |
| T0 documentation and paper skeleton | C1/C2 framing | `COMPLETE` | Documentation/manuscript edits only. | Owner inspects active files; fix any scientific-language objection. |
| R0 NACA first screen | C2 prerequisite | `COMPLETE` with negative delayed-failure decision | The frozen NACA screen completed and passed independent packet audit. | Preserve the exact R0 identity and verdict; use it only as immutable parent evidence. |
| [NACA corrective successor](B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md) | Primary fixed-PCNO PDE case | `COMPLETE` at open-development scope | Development evaluation, replay, final presentation packet, portable freeze, manuscript, and exact mounted-checkout WSL verification are complete; protected reveal is not authorized. | Owner inspects the evidence and separately decides whether to reveal prospective outcomes. |
| [NACA corrective extension](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md) | Missing B4 mechanism contrasts | `COMPLETE` at open-development scope | All pilot, bank, source, smoke, training, common-evaluator, retrieval, packet-audit, corrected-visualization, and claim-bounded manuscript gates closed under `B3B4_NACA_CM_EXT_20260902A`; protected roles are closed. | Owner inspects the manuscript. Freeze a separate evaluation-only mechanism assay before claiming identical-bank trusted-response fidelity. |
| B1 exact ODE | C1 boundary and measurement calibration | `COMPLETE` | Closed under the frozen B1 contract. | Preserve the verified packet and use only as supporting calibration. |
| B2 learned ODE | Supporting mechanism realization | `COMPLETE` | Closed under the frozen three-seed/four-arm B2 contract. | Freeze the implementation and packets; continue paper writing without inferring ODE-to-PDE ranking transfer. |
| [B2-NL fixed-support nonlinear pilot](B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md) | Historical response-regime stress | `COMPLETE` | Closed under the frozen local-CPU contract; no PDE or remote work occurred. | Preserve the packet unchanged. Its fixed-radius support is excluded from the active draft and may be visualized only as an internal diagnostic. |
| [B2-GN Gaussian normal-noise successor](B2_GN_GAUSSIAN_NORMAL_NOISE_PREREGISTRATION.md) | Supporting scale-response study | `COMPLETE` | Closed under the audited normal-only local-CPU contract. No PDE, GPU, remote, or sealed work occurred. | Preserve the packet and use its scale-response results only as supporting ODE evidence. No rollout ranking is mechanism-qualified. |
| B3 primary-PDE diagnosis/freeze | Main C2 diagnosis | `COMPLETE` | NACA successor open-population diagnosis, signed predictions, audit, and source-bound freeze are closed. | Preserve the verified packet and its claim boundary. |
| B4 primary-PDE interventions/reveal | Main C2 test | `COMPLETE` at open-development scope | Parent and extension experiments, corrected figures, and claim-bounded manuscript integration are complete on open development roles. Prospective and sealed evidence remain unopened and unestablished. | Owner inspects the evidence and separately decides whether any protected reveal is warranted. |
| [B5 Supersonic-Bump comparison](B5_BUMP_SOLVER_FREE_COMPARISON.md) | Secondary transfer/no-harm evidence | `COMPLETE`: one-seed open-development pilot | Five training runs and all 224 rollouts completed; retrieved metadata, terminal checkpoints and saved replay passed independent audit. No intervention passes the full efficacy/no-harm rule. | Discuss benchmark representativeness and next scope with the owner; no automatic confirmation run or new PDE. Historical test remains sealed. |

## Candidate Readiness

| Candidate | Verified live state | Missing gate | Status |
| --- | --- | --- | --- |
| SU2 Unsteady NACA0012 | Stage 0 through audited R0 evaluation completed. Every rollout is finite and severely wrong late, but all three fail the old early-accuracy threshold (`0.2669/0.2368/0.2458 >= 0.15`). | The delayed-failure phenotype is absent; state-space drift and ripple cause remain unmeasured. | Preserve `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`; later corrective results belong only to the distinct successor/extension identities. |
| Turbulent square cylinder | Public SU2 case material exists. | A pinned, reproducible production configuration and all downstream gates. | Unselected candidate; owner/mentor choice and a new contract are required. |
| Laminar von Karman cylinder | Official unsteady SU2 tutorial route exists. | R0 population, PCNO, and transverse-failure qualification. | Cheap solver/control fallback, not assumed to be the hero case. |
| Shock-vortex / supersonic bump | B5 solver-free Bump pilot completed; results below. | No trusted arbitrary-state restart; only stored trajectories supply training targets. | Bump supplies bounded comparison evidence, not a qualified general-PDE showcase. Next scope awaits owner discussion; historical test stays sealed. |

NACA0012 has completed a valid negative R0 delayed-failure screen. Its distinct
successor and extension now provide completed open-development evidence for the
narrower fixed-PCNO question of how corrective mechanisms change recurrent
response, path, structure, and rollout. The visible ripples are not assumed to
be manifold-normal. Prospective diagnostic selection remains untested.

## Historical Evidence In The Current Story

| Evidence | Use |
| --- | --- |
| D094 | Retrospective one-step/rollout discordance and self-composition motivation; never prospective C2 evidence. |
| M1 reference packets | Finite-grid Kolmogorov parent evidence only; old residual-FNO model stage and failed population attempts do not authorize continuation. |
| P0 native-coarse restart | Explains why the old shock coarse map is not a trusted displaced-state reference. |
| P1 path-conditioned tube | Solver-free vocabulary and synthetic plumbing only; not `Phi(u+eta)` fidelity. |

## Owner Decisions Still Required

- [x] Select SU2 Unsteady NACA0012 as the first R0 screen and conditional
      primary case after the bounded candidate review (2026-08-31).
- [x] Rewrite the NACA R0 solver/state/failure/population/resource contract and
      freeze the public-resource manifest (2026-08-31).
- [x] Approve the exact B1 ODE system and compact B2 arm set (2026-08-30).
- [x] Approve the separate B2-NL nonlinear response-regime stress test and
      local CPU execution after independent audit (2026-08-30).
- [x] Approve the exact B2-GN Gaussian covariance and scale-sweep contract
      (normal-only, 2026-08-30);
      report both standard deviation and variance, and do not use a long-rollout
      outcome to select the displayed scale.
- [x] Issue the complete NACA R0 verdict and do not promote it as the primary
      PDE (`R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`, 2026-09-01).
- [x] Select NACA as the primary case for the distinct corrective successor and
      approve open-population implementation/execution (2026-09-01).
- [x] Approve the B3 representative intervention set and signed predictions in
      `B3B4_NACA_CM_20260901A` (2026-09-01).
- [x] Approve successor implementation, resource smoke, and bounded open-role
      training/evaluation (2026-09-01).
- [x] Approve `B3B4_NACA_CM_EXT_20260902A` implementation and train/development
      execution, with the full relabeling bank gated by the train-only SU2
      pilot (2026-09-02).
- [x] Select the Supersonic Bump for bounded solver-free B5 secondary transfer
      and authorize local Stage-0A/B contract, analyzer, and synthetic-test work
      only (2026-09-05).
- [x] Select AutoDL, complete the B5 train-only audit, and authorize the solver-free
      comparison under `B5_BUMP_SOLVER_FREE_COMPARISON_20260905A`.
- [ ] Discuss the mentor's concern about NACA/Bump representativeness and decide
      the next experimental scope after reviewing the completed comparison.
- [ ] Decide whether to approve the prospective reveal; the portable prediction
      packet is closed, but its existence is not reveal authorization.
- [ ] Approve any sealed/test access as a still later, separate decision.

## B5 Bump Comparison Result And Paper Comparison

The one-seed pilot completed on September 5 at 22:42 China time, after 96
minutes. Source is `4fb8dc2`; the completed final-manifest SHA256 is
`50a3b4811211265c76b6374e9f42cd09096cfb4c697b48e512dafaf53535e579`.
All 44 output files and 30 source files rehashed remotely. The 39 local output
files include all five terminal checkpoints and nine saved rollout/reference
arrays; five rolling optimizer-resume checkpoints remain hash-verified on
AutoDL. Local metadata/source/checkpoint audits, all 224 per-case metric
recomputations, and saved-array CRC/shape/finiteness, equal-node error and
admissibility checks pass. No model rerun, solver call or protected read was
needed. The analysis packet is in ignored
`artifacts/time_dependent_no/b5_bump_comparison_20260905a/retrieval_20260906a/`.

All arms use the same 28 development trajectories and 79-step horizon. Errors
are all-node, proxy-weighted, component-scaled conservative-state relative L2;
only the training loss masks to normal nodes. One-step averages eight fixed
clean windows per case. AUC is the arithmetic mean over 79 errors. Ratios in
the table are medians of paired case ratios to Clean, not ratios of means or
medians. Invalid means at least one inadmissible state at any rollout step.

| System | Mean one-step (%) | Mean H79 | Paired H79 ratio | Paired AUC ratio | Joint wins / 28 | Invalid cases / 28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 1.165 | 0.08920 | 1.000 | 1.000 | -- | 0 |
| Clean-EMA | 1.163 | 0.07769 | 0.833 | 0.822 | 25 | 0 |
| IID recovery | 1.182 | 0.07711 | 1.189 | 1.207 | 9 | 0 |
| Curriculum/EMA prefix | 1.184 | 0.03889 | 0.578 | 0.611 | 28 | 1 |
| Explicit prefix corrector | 1.115 | 2016.25 | 0.424 | 0.506 | 26 | 6 |
| PDE-Refiner | 1.408 | 0.08741 | 1.251 | 1.217 | 7 | 1 |
| Mapped PCA rank 7 | 14.569 | 0.18066 | 2.567 | 3.859 | 1 | 25 |
| Mapped PCA rank 32 | 12.584 | 0.17285 | 2.462 | 3.606 | 1 | 20 |

The central findings are:

1. Curriculum gives the strongest consistent global-error improvement. Its
   clean one-step error is worse on every case, yet H79 and AUC improve on
   every case. Against Clean-EMA it still wins jointly on 26/28, with paired
   H79/AUC ratios 0.752/0.793. It nevertheless produces negative pressure in
   case 156 at calls 78--79 (terminal minimum -0.01643) and increases the
   paired median shock-strength log-error ratio to 1.322.
2. Explicit correction has good typical error but catastrophic tails. Cases
   184 and 49 reach H79 errors 56,269.77 and 184.39; inadmissibility begins at
   calls 17 and 18. Case 184 has slightly better clean-window error than Clean
   (ratio 0.976), yet a 213,325-times larger H79 error. Its low median H79
   (0.03182) and small median clean identity error (0.001569 normalized RMS)
   cannot certify safe composition. All trajectories are finite numerically.
3. IID recovery improves mean H79 by 13.5% and reduces the maximum from
   0.31251 to 0.11457, but worsens typical paired H79/AUC by 18.9%/20.7%.
   There are 18 joint losses and only nine joint wins. Its rough-error energy
   decreases, consistent with a smoothing tradeoff, not proof of normal
   contraction or manifold return.
4. Mapped PCA already damages clean inputs: median identity errors are
   0.30666/0.25847 normalized RMS for ranks 7/32. Rank 32 reduces this damage
   on all cases but remains inaccurate. Rank 7 has 25 first-step-invalid
   cases; both ranks have admissible final states on every case, so endpoint
   positivity alone would hide their intermediate violations. Donor mismatch,
   remapping, affine-rank restriction and admissibility remain confounded.
5. Refiner's favorable NACA ranking does not transfer to this tested
   implementation/schedule/calibration: only seven joint wins, one invalid
   case and a 3.245-times paired smooth-region high-pass error energy.
   This is one training seed and one keyed stochastic rollout per case,
   not a replicated refiner or method-family rejection.

No intervention passes the frozen combined efficacy/no-harm rule. Clean-EMA
passes the global-error, clean-error and admissibility conditions; its sole
failure is the shock-strength proxy (paired ratio 1.298). The absolute median
shock-strength log errors are 0.04466 for Clean, 0.06013 for EMA, and 0.07184
for curriculum. This distinguishes a modest structure tradeoff from the
explicit corrector's catastrophic failures. The threshold is not retuned.

Training took 15.22/20.14/15.55/22.74/14.03 minutes for Clean/IID/curriculum/
explicit/refiner; explicit correction additionally needs its Clean base.
Mean deployment time per 79 steps is about 0.589 s for one-call learned
systems, 1.161 s for explicit composition, 2.257 s for Refiner and 0.632 s
for PCA, excluding PCA donor mapping and basis setup. Explicit deployed
parameters total 38,311,440, not the correction-only count in `training.json`.
Matched seeds/initializations/streams do not imply bitwise paired CUDA
execution: Clean and curriculum differ even during clean-only warmup.
Finite, decreasing training histories are not convergence certificates.

### Comparison With The Current Manuscript

The current manuscript contains NACA, not Bump. Rehashing its 12 evaluation
files and recomputing 89,856 raw rows reproduces its table to rounding.
Its numerical conclusions remain valid at their declared scope:

- NACA path correction ranks first at every seed; Bump's mapped affine PCA
  is harmful. Importantly, these are different constructions. NACA uses PCA
  to select a nearest piecewise-linear path segment, then interpolates full
  training snapshots. Bump projects onto a mapped donor's affine subspace.
  The amendment's phrase "NACA's shared-mesh global path PCA" is imprecise;
  this clarification does not rewrite its frozen bytes.
- NACA Refiner ranks second at every seed and sampler tape; the tested Bump
  Refiner worsens typical paired error. This demonstrates non-transfer of
  these deployments, not that PDE complexity alone determines the ranking.
- NACA curriculum is seed-sensitive; Bump curriculum gives consistent
  within-pilot error gains but still violates physics/structure conditions.
- NACA paired recovery is robustly helpful; Bump IID recovery trades tail
  suppression for typical-case harm. The noise laws, targets' sampling,
  state histories and optimization budgets are not matched across cases.

Do not compare raw error magnitudes: NACA uses equal-node train-field-scale
RMS, trapezoidal H1--208 AUC and late H174--208 error; Bump uses relative L2,
arithmetic H1--79 AUC and H79. NACA has three model seeds and three refiner
tapes, versus one Bump seed/tape. Its eight temporal anchors share one periodic
trajectory; Bump uses distinct geometries/conditions. The new explicit
prefix-error corrector has no matched NACA deployment.

### Claim And Representativeness Assessment

Independent in-session audits support bounded one-step/rollout discordance
and non-transfer of the tested rankings. They do not establish a correction
meeting the full Bump rule, broad PDE representativeness, measured manifold
drift/normal contraction, or a prospective quantitative diagnostic-to-ranking
rule. No unpublished content was sent to an external reviewer.

The mentor's concern is well founded. NACA is an unusually favorable periodic
path example, not a broad trajectory population. Bump adds shocks and varied
meshes, but Clean stays finite and admissible (mean H79 8.92%), and there is
no trusted displaced-state solver for the targeted mechanistic comparisons.
This does not prove that Clean stays near a manifold. Both cases are useful
examples and limitations, not yet the decisive representative-PDE validation.
The reverified POD spectra show a longer Bump linear tail (median rank 58 for
99.9%, while seven modes already capture about 90.6%), not intrinsic dimension
58. NACA reaches 99.9% at six model-coordinate or seven area-weighted modes.
These differently sampled/weighted temporal spectra do not isolate nonlinear
manifold complexity or explain all mapped-projector damage.

The owner will discuss the next benchmark after this analysis. Preserve all
results, leave the manuscript unchanged, and do not automatically start
confirmation seeds, a new corrector, a new PDE, or protected evaluation.

## Attempt E Development Result

The evaluation-only retry for `B3B4_NACA_CM_EXT_20260902A` completed under
final-manifest SHA256
`15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.
It reused the 18 Attempt-D training packets without mutation. Source tests,
GPU smoke, evaluation, raw-table recomputation, NPZ CRC/shape/finiteness, and
local/remote hash-parity checks passed. All values below are development-only:
each seed is the median over the same eight anchors, followed by the descriptive
median `[minimum, maximum]` over seeds 17, 29, and 43. The range is not a
confidence interval.

| Primary deployment | Rollout AUC | Late-window error |
| --- | ---: | ---: |
| `PATH_PROJECTION`, corrected | 0.01453 [0.01144, 0.01519] | 0.02477 [0.01950, 0.02530] |
| `PCNO_PDEREFINER_K3_VPRED`, EMA sampler-median | 0.64862 [0.63807, 0.69877] | 0.97480 [0.94732, 1.04325] |
| `DETACHED_PUSHFORWARD` | 1.01607 [0.83671, 1.36423] | 1.81917 [1.39480, 3.77995] |
| `PAIRED_RECOVERY` | 1.06803 [0.97535, 1.15693] | 1.69931 [1.54134, 1.93490] |
| `DYNAMICS_RELABEL` | 1.09442 [1.01955, 1.32361] | 1.71306 [1.67811, 2.95696] |
| `CURRICULUM_EMA_PUSHFORWARD_K13`, EMA | 1.15916 [0.94547, 5.29715] | 2.03362 [1.51489, 22.24117] |
| `CLEAN_EMA`, EMA | 1.26954 [1.08723, 289.58284] | 1.87478 [1.68988, 838.24174] |
| inherited `CLEAN` | 1.28764 [1.02310, 1.37085] | 2.21296 [1.55229, 2.21709] |
| `MP_PDE_PUSHFORWARD_M01` | 1.64431 [1.05054, 2.41758] | 4.92573 [1.75293, 8.13172] |

The result supports three development-case conclusions. First, one-step
ranking does not determine rollout ranking: `CLEAN_EMA` seed 43 has essentially
the same horizon-one error as inherited clean but 283-times larger AUC, whereas
PDE-Refiner has 5.36-times larger central horizon-one error than clean but about
half its AUC. Second, path projection ranks first and PDE-Refiner second on both
primary metrics at every seed; paired recovery is the safest one-call learned
intervention. Third, literal MP-PDE and curriculum exposure are heterogeneous,
so this experiment does not reject pushforward or multistep training as a
family.

The mechanism evidence is narrower. Recovery preserves clean fidelity and
reduces one-prefix response without validity harm. Relabeling lowers the same
gain only with large clean-path and optimization harm. The evaluator did not
run the trained recovery and relabeling models on identical bank inputs against
both stored targets, so trusted displaced-response fidelity remains untested.
The one-prefix gain is not a stability certificate: it misses the coherent late
threshold failure of `CLEAN_EMA` seed 43. PDE-Refiner improves each reverse step
near the clean path but is not a monotone projection at late reached states.
All numerical states are finite, yet six deployment/seed variants develop
nonpositive density; finite and physically admissible are therefore distinct.
Prospective and sealed populations remain unopened.

## Execution Log

- 2026-09-05: The solver-free B5 pilot launched at 21:06 China time from
  producer commit `4fb8dc2fe091228948ec0624a8ce5c7037950e6c`, source-archive
  SHA256 `89b2ec32525e2979139982d5bd5a73ce4a1afa70241b046292761194249585d7`.
  The replacement smoke passed all 24 remote CPU tests, all five training
  objectives on the 23,359-node mesh, and four inference variants. The
  training-only 19,345-to-23,359-node PCA check passed with normalized
  idempotence error `9.54e-7`. Local receipt verification closes 16 smoke
  outputs and 30 exact Git-source hashes; final smoke manifest is
  `c1c492201820a677bce8dae9130adbfad1975be080a62b1f8a663f6552540d06`.
  Raw dataset arrays were not redownloaded: their training-only access and
  digest checks are recorded by the remote runner.
  At the initial 21:18 check, Clean had completed 44/64 epochs with finite
  loss/gradients; about 5 GiB remained on the output filesystem. The automatic
  queue trains all five arms, then evaluates eight deployed systems on the
  same 28 development trajectories for 79 steps. Estimated completion is
  23:00 September 5 to 00:00 September 6 China time, not a deadline guarantee.
  No intervention outcome is available yet. Receipts and initial log are in
  ignored `artifacts/time_dependent_no/b5_bump_comparison_20260905a/retry1/`;
  verify final source, checkpoint, data-access and output bindings on retrieval.

- 2026-09-05: The first B5 comparison smoke (`5ce08f7`) passed all 23 remote
  synthetic tests, then stopped before scientific array access because source
  collection treated PyTorch's virtual `_classes.py` filename as repository
  source. The attempt is retained as infrastructure-only. The collector now
  ignores relative virtual filenames, with a regression test; all 24 focused
  CPU tests pass. A new source snapshot and smoke attempt are required.

- 2026-09-05: B5 Stage 0B completed successfully on AutoDL in 229 seconds.
  The retrieved packet independently closes at final-manifest SHA256
  `479eaa3c28fd7118dc291cb779a0d27b96977e6cad7a00b60e358567505a9bf3`.
  All 32 training trajectories exceed rank 7 at 99.9% variance; median is 58,
  range 49--62. Rank 7 captures median 90.6%, so this is a long spectral tail,
  not proof of high intrinsic dimension. The owner then requests the solver-free
  intervention comparison. Its new amendment freezes five training runs and
  eight deployed systems, including a shared Clean-EMA control and mapped PCA
  ranks 7/32. Twenty-three focused synthetic CPU tests and Ruff pass before
  full-mesh GPU smoke. Generated launch/results belong under the ignored
  `b5_bump_comparison_20260905a` artifact directory.

- 2026-09-05: owner selected the Supersonic Bump for a bounded second-PDE
  transfer/no-harm study and clarified that NACA's projection ranking is a
  consequence of its special nearly linear low-rank path, not a universal
  conclusion. `B5_BUMP_SOLVER_FREE_TRANSFER_20260905A` freezes train-only
  per-trajectory native-mesh temporal POD as a diagnostic and declares four
  intended solver-free systems. Only Stage 0A/B local implementation and
  synthetic tests are currently executable. Stage 0C and Stage 1 require
  identity/algorithm amendments; remote execution has no named resource and
  the historical test remains sealed.

- 2026-09-04: a presentation-only v3 derivative rerendered the extension's
  exact replay (SHA256
  `79886d205f21ed64ae9727a0a70ad3692ae1962bad221fbb488800da618878ec`)
  as unfiltered piecewise-linear native-node fields on a deterministic quad
  triangulation. Scientific arrays are unchanged: the checkerboard in the old
  flat-cell view was a rendering artifact. The presentation-contract and
  render-manifest canonical SHA256 values are respectively
  `35efce1ac222312ec3bc8a3db8905995152c086a0fb791c3273c77d17c9136c0`
  and `4562919f14b24e9b925a9b0bfaf0c599700af78ac06cfcd8cbeb7ccb90b1d0f2`;
  the corrected horizon-208 PDF is
  `54c87cf2995eacb224b9346e133e88c03be24eba750d9a0572bd3e24deb3d5bf`.
  A separate train-only packet, `naca_dataset_pca_20260904b`, analyzes the 238
  train-current states in 72,880 normalized model coordinates. The first
  `2/4/6` PCs explain `96.1294/99.8007/99.9809%` of variance; the complete
  BDF2 view is nearly identical, while area weighting needs seven PCs for
  `99.9%`. This supports strong low-rank, path-like variance concentration,
  not an intrinsic manifold dimension. Its final-manifest SHA256 is
  `e83cd7aaeedcc24b2cefc1683da222f6730af7ed3bb702e470650766750916f7`
  and summary canonical SHA256 is
  `f536320d111587e0988d2b582114231782ca4d76b06d4a3fbe88444e44f9d9ff`.
  Both figures are integrated in the private 34-page manuscript (SHA256
  `966c2aad4b02d01c198555008e4db87d3ec583390461d05df4a782758c9e5f95`),
  which has no undefined references/citations, overfull boxes, Type-3 fonts,
  or unembedded fonts. Prospective and sealed populations remain unopened.

- 2026-09-03: the corrected extension presentation derivative
  `naca_corrective_extension_visualization_results_20260903b` passed hash,
  format, and direct visual audit. Its presentation-receipt canonical SHA256 is
  `7be542ef9ead7d8f7835686450795aaf32b944fe87a6a4f0eb42c398bd208b62`;
  its remote parent receipt is
  `07360658a8f8e085515bb69550e2e6fada17a9f21f503cbac7513bc81cb6ee53`,
  render-manifest canonical SHA256 is
  `12b3054ffbdebf047e8e63d4957217889697d0ad40902a2cd2d71f204d8857a2`,
  and exact replay SHA256 is
  `79886d205f21ed64ae9727a0a70ad3692ae1962bad221fbb488800da618878ec`.
  The two exact-snapshot PDFs are integrated in the clean 33-page manuscript;
  its SHA256 is
  `d7ed11861fec76148131f8b81881f0a2feb67d069630d7d90b98dc736b4eece6`.
  The three animations are independent qualitative rerolls. Scientific replay
  and evaluation are unchanged, and every protected-access flag is false.

- 2026-09-03: Attempt E closed the extension's evaluation-only retry and the
  exact result packet above. The local metadata retrieval contains all 126
  non-checkpoint JSON files from the 18 training packets; their hashes,
  self-hashes, source bindings, and final-manifest identities verify. No
  checkpoint was duplicated locally. Any new common-bank response assay
  requires a frozen evaluation-only contract before execution.

- 2026-09-02: the extension trainer/evaluator source closure and final
  code-to-intent audit passed. A first audit correctly blocked execution because
  the trainer used relabel-receipt semantics without binding that module's
  bytes; the repaired closure now includes and live-rehashes it. The trainer
  file/source-set SHA256 values are
  `0d79bfa9d5312770d81574d6d30020d3a708a3332a77b367a5625f4a609ad852`
  and `9a2911d11e19d9d3a1f0dba4866d8964fe412380b1d0a46bae45dacb51694f8a`;
  the evaluator file/source-set values are
  `f9409fb5edcba78f69636e62e6329b64df06b7b63f8a4e2725bab57623cdfa8f`
  and `46670860fb9eb1e1271449dc6cb86350f5163eaf954837122c1c3d6380e6b1e4`.
  The minimal 37-file deployment/test archive is `292,836` bytes at SHA256
  `3ece0808eb82b53940353479dc4d0d86a4abc2d31eed61cd460237ad986dd6ea`;
  a clean extraction reproduces both source-set digests exactly.
  The final integrated extension partition passes `75` tests; separate R0 and
  parent compatibility partitions pass `45` with four skips and `11`
  respectively, and Ruff check/format-check pass. The evaluator recomputes all
  extension deployments plus inherited `CLEAN`, `DETACHED_PUSHFORWARD`, and
  `PATH_PROJECTION`; no scalar result is spliced from the parent evaluator. No
  extension GPU training or development evaluation has occurred.

- 2026-09-02: the complete train-only paired displaced-BDF2 bank closed with
  all `476/476` fixed-SU2 calls successful, no redraw or dropped case, every
  per-field realized-scale ratio inside its frozen bounds, and no protected
  access. The final-manifest and compact-NPZ SHA256 values are respectively
  `9202d9738ce2549918eb548b3ca2cd78541c8b892e9a6959883b4f4ec9e04293`
  and `04bb27b633930874890f2bd34d8acd0c60dcc9c1e520aac1b395979eb607fc53`.
  An independent streaming pass reverified all `478` registered files. This
  unlocks paired training; it is not a learned-model result.

- 2026-09-02: corrected pilot R1 completed all `11/11` registered train-only
  SU2 calls and passed replay, convergence, admissibility, repeatability,
  auxiliary-invariance, displacement-scale, zero-control, response-separation,
  and authority-immutability gates. Its final-manifest and pilot-receipt SHA256
  values are respectively
  `f55dbab47943d34f0a34ccc40f54bab2f3676faa9bee28f29b5d7a10868b5603`
  and `a5a388c4718b1ce8a78ce6cf36482a7289b035fed84bc8c7b0e54c9695d00aa4`.
  The earlier configuration-only attempt is preserved under final-manifest
  SHA256 `6b1ab039a47d91e42325fdc53f97f230c3387fde315df1920a6a9ccf4f55e8b3`;
  it produced no numerical SU2 result and is not scientific evidence.

- 2026-09-02: `B3B4_NACA_CM_EXT_20260902A` was frozen as a distinct
  train/development-only extension. It preserves the closed five-arm parent and
  registers literal `MP_PDE_PUSHFORWARD_M01`, separate
  `CURRICULUM_EMA_PUSHFORWARD_K13`, paired frozen-bank `PAIRED_RECOVERY` versus
  `DYNAMICS_RELABEL`, and `PCNO_PDEREFINER_K3_VPRED`, with `CLEAN_EMA` as the
  EMA control. Focused CPU tests and independent code-to-intent audit precede
  execution. The then-pending 476-input paired bank was gated on the eleven-call
  train-only SU2 restart pilot passing every registered replay, convergence,
  admissibility, determinism, scale, and response-separation gate. No solver
  case could be silently dropped. Solver labels are offline only; no prospective,
  sealed, online-solver, or online-defect-trigger access is authorized. This
  registration is not an implementation or result claim.

- 2026-09-02: `B3B4_NACA_CM_20260901A` reached local open-development
  closeout. The development evaluation remains bound by final-manifest SHA256
  `09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
  Its exact archived evaluator is
  `development_source_snapshot/evaluate_pcno_naca0012_successor.py` at SHA256
  `2630a7548de796ff1f11a23ed395ae454f2df87c1d6fed0d2d706c4769a7de8c`;
  the snapshot manifest is
  `2ca73afd4b44be7af0af3b3ee876b71b82ed4b393daf362231b0a215e62f6ce6`,
  and the development source-set identity is
  `df7949e9a1eb81dcb0af4e95e07a14d23c9f4abd205eac8f950dcbe5e453ff13`.
  Independent verification closed the declared inventory, hashes, source/data/
  checkpoint bindings, metric cardinalities, identity parity, and protected-role
  flags. Every evaluated rollout is finite. `PATH_PROJECTION` improves both
  primary metrics in all 24 paired anchor-seed comparisons and improves the
  pressure and graph-Dirichlet diagnostics; its zero post-correction path
  residual is true by construction and is not outcome evidence.
  `IID_RECOVERY` helps relative to `CLEAN`; `ERROR_SUBSPACE_RECOVERY` improves
  `CLEAN` but not `IID_RECOVERY`. The registered `DETACHED_PUSHFORWARD`
  mechanism prediction is falsified and its rollout behavior is seed-unstable.
  These are open-development learned-map/corrector findings, not evidence that
  visible flow features are physically off-manifold or a trusted SU2 response
  assay at arbitrary displaced states.

  The immutable replay `development_replay_v2_81646368` contains
  `rollout_replay.npz` at SHA256
  `8def4971abb669339f9a8ea5d5b44f8a6d0ddf792f0745cb893c03c460e73bd3`
  and `replay_manifest.json` at SHA256
  `f509b21c5fb448d2aa56a5ac135bdb0d85f337f3be7873b49c94ee3838a8120c`.
  Historical `development_visualization_v2_81646368` remains valid provenance:
  raw manifest SHA256
  `bc070b378ebe006e1fbaaf7796980b7a86fb133b1b9d5d35f15c9660702fe49b`,
  canonical payload
  `2887f7fbf79489ff674e97f2bcac0d4397f4f70a24b18236a881e8b497c46daf`,
  and producer
  `816463686ff7e53df8787a5cc31dc3cac8d731d40858820b2d6f986935a640bd`.
  Its earlier `corrective_tradeoffs.pdf` and
  `density_signed_error_h208.pdf` hashes are
  `4a996fca73cbc527c97a92b819c0a2272a8b9c5a355c2ef5eedf8b7680b52e72`
  and `dedca6a1cde51aa4957a531dd2d0b7ed76dbbe497eeffffcb188c2a7e442b0f8`.

  Final presentation-only packet `development_visualization_v3_e2eccdbf` has
  exactly 14 files and 11 rendered outputs. Its raw manifest SHA256 is
  `ea27126baf53c0621c7c55ae865f0bb65f55f40b9f8e551e499153bfe050d309`,
  canonical payload is
  `f6bbbacf5c400a6ec4d173b7e543285b970b1b61d763baa046a2a111220da776`,
  and current renderer is
  `e2eccdbf2619a1966bb939f53a0d38cb472cf4a18f5e1e31b0ac390323ece867`.
  It explicitly pins the historical producer above and contains byte-identical
  replay files. Final `corrective_tradeoffs.pdf` and
  `density_signed_error_h208.pdf` have SHA256
  `0e4304fab66c04dbae2a4c08ffd9fe09e4fdf6bbf0547bcae37a33b02419f572`
  and `b24332f34534efb0615522d71854885247a7b6e0f10d11f070a34b0544c3aacd`.
  All eight PDFs contain no Type-3 fonts; all three MP4 files validate. Static
  figures use verified development evaluation, while animations are qualitative.

  The canonical portable `prospective_freeze` contains exactly three files.
  Its raw final-manifest, freeze-file, source-manifest, canonical freeze-payload,
  and evaluator source-set SHA256 values are respectively
  `eef941053e549aaa4b5e2fd080ba0fbd73eed4bda4095a3d941f76d4e607217f`,
  `afaa206c6a6a449e48d41a090adbb055f2fe9824255ba7ae57f86af7b030fa6a`,
  `6aafec1880dbeee0a0c8762fcab341f8520d1fbfb26afa2655303935c2620a1c`,
  `e0f56dbe219fbaade339db24eb1907c055e7d77c8d8d95297dba4549aafbfdfb`,
  and `1cd504a1482e07100ac8ab63818ee919da089b5d396fa5c9bc350b36104a4f62`.
  Its current evaluator is
  `fbe8e61328fdbeec5ca9a28855d9f47f20d5ec2fbed46f53ffe975b8a26d29a0`.
  Inherited/readable ACLs and false prospective/sealed flags are verified.

  Two earlier freezes are preserved only as superseded provenance. The
  owner-only `prospective_freeze_private_acl_c5efa3f4` has raw final-manifest,
  freeze-file, source-manifest, payload, and source-set SHA256 values
  `c5efa3f435eb8ed5831c69c09675afcc30cb35622df4a2931c3ba4e7b1e7a28c`,
  `028e11e26659093e7f245a6e0810b245ebd33460b8e5d450322fe9e263557a17`,
  `f448118ad02e07aaa5a512dfbc7493edcb0ac7a11cce2c959daecae016e3e081`,
  `8d0c2845d33916481ea23379a0a1c28f0b5cecdb1338002818327e8c86eba8c3`,
  and `d343f10736b60047808def780b35fa6fbcb737b394083959e4767d285b6e29f9`.
  Pre-hardening `prospective_freeze_stale_pre_hardening_09b470be` has raw
  final-manifest, freeze-file, source-manifest, payload, and source-set SHA256
  values
  `09b470bef70f6362b52d9249eed27de9c5380985e63903c66fa7958f43eeca09`,
  `8307f9e66a3139ea439949f9093705ab3a5daadaeba566a822b97e4f0e034488`,
  `6c952c672e593d338d4debd2fe796d6eb449b2a704bc005c21387b95c00902d4`,
  `f8921371b0d8e2cc9b3f8fcc2238a2d04ff8a32e214890ce143e371cf9b35934`,
  and `bb66ec8ba904ab71bcb9c0a65181c9809ec232014456e60861106afdf708f8a2`.

  The integrated `paper/main.pdf` has 35 pages and SHA256
  `79f64c09030a85160777bea72af896b7f98d141bc081c893398807c6993f6c66`.
  It has no undefined references/citations, overfull boxes, `VERIFY` markers,
  Type-3 fonts, or unembedded fonts. Windows 68 focused tests and Ruff
  check/format-check pass with the same explicit `E402` exception described
  below for the deliberate provenance-first imports.

  Exact mounted-current-checkout WSL Ubuntu verification also passes under
  Python `3.10.12`, Torch `2.8.0+cpu`, pytest `9.0.2`, and Ruff `0.12.12`.
  Pre/post SHA256 identities match for the evaluator
  (`fbe8e61328fdbeec5ca9a28855d9f47f20d5ec2fbed46f53ffe975b8a26d29a0`),
  visualizer
  (`e2eccdbf2619a1966bb939f53a0d38cb472cf4a18f5e1e31b0ac390323ece867`),
  evaluator test
  (`75e75a09622edd3c57f5bf5d27852a1a8bcbb9dbccd60b5704145387953907f4`),
  visualization test
  (`a7d2bdfa36a7a44045fe41427a2f92df694df1959da05322bfae4b2a0ddc0cd3`),
  and successor utility
  (`8f788a25b25895fa2ad48dfc437ef051b051d3548ad2d340ef0e7370a3180cca`).
  Ruff format-check passes for the four source/test files. Ruff check passes
  with explicit `--ignore E402`; those imports follow deliberate source-byte
  provenance capture. All 68 focused tests pass in `20.46 s`. Both exact WSL
  temporary directories were deleted and their absence verified.

  An earlier AutoDL exact-four-file attempt verified those four file hashes but
  could not run checks because its dependency closure was incomplete. It is not
  a Linux test pass; its temporary root was deleted. No protected population,
  data, or checkpoint was opened. No prospective authorization artifact exists;
  prospective and sealed populations remain unopened, and C2 prospective
  generalization is unclaimed.

- 2026-09-01: the frozen NACA PCNO R0 screen completed on one RTX 5090 under
  CUDA 12.8. A full-resolution forward/backward/optimizer smoke passed, followed
  by exact sequential seeds `17/29/43`, each completing 100 epochs with best
  checkpoint at epoch 100, and the unchanged development-only evaluator. The
  dataset final-manifest SHA256 is
  `1cb5fd2d751bdf1ca27cde5e1a1457973a525c19a98dbadd7cc3738c3efe2e4b`;
  training final-manifest SHA256 values are
  `5fc86d129ca456aed1a8b8870020f93d5e0bf3e3d41881ec5ea3ef87418f3071`,
  `3078accaf90c7c0c2cd653171ca6f938ec42bfe1e01b23bb2f22f8711699d020`,
  and `0ab746f1ca30f72471146165ee6bbfc2dd20f7978e52bc4d62b3cc12680c8e50`.
  The result archive is `900,044,171` bytes with SHA256
  `7eb53ec50a454630f5461c82593322830d9f995029809c359ff89585d671ef17`;
  evaluation manifest SHA256 is
  `37e43b315c9160b4d3525b4b6d7991881faacdae5629235361c64e837046725a`,
  and canonical result payload SHA256 is
  `11aac10b5fb7781eeb64a0e3fb5d09ef8ce30f02e271a20de0cfdaa64d9b3630`.
  Independent audit rehashed all 32 declared files and 21 JSON self-hashes,
  verified the live source, dataset, contract, replay, smoke, authorization,
  device, seed, completion, and access bindings, and recomputed the raw grids.
  Clean next-state relative L2 is `0.001181/0.001233/0.001190`; early
  train-state-scale window error is `0.2669/0.2368/0.2458` versus persistence
  `0.3364`; late error is `1.8564/1.7023/1.7319` versus persistence `0.3365`.
  All runs remain finite and all satisfy the severe-late gate, but none passes
  the strict early `<0.15` gate, so the frozen decision is
  `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`. Native replay also misses
  the strict quarter-margin in uniform-node and near-body/wake views. The tiny
  persistence error at terminal step 208 is a near-period alias; registered
  window metrics control interpretation. No horizon extension, prospective or
  sealed access, corrector run, or post-outcome retuning occurred.

- 2026-08-31: before any PCNO population materialization or model outcome, the
  single-attractor baseline was frozen in
  `R0_NACA_PCNO_BASELINE_CONTRACT.json`, SHA256
  `94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87`.
  Independent audit recomputed the 32 phase anchors, source crossings, maximum
  phase-rounding error, 238 train/development transitions, complete BDF2 and
  208-step containment, six-period ratio, and 16-channel input. It also closed
  windowed non-aliased gates, persistence advantage, failed-rollout accounting,
  qualified shoelace measures, five-state pressure-only force diagnostics,
  train/development-only access, precision/reproducibility, and the paired
  replay-margin rule. Prospective/sealed states remain unopened; no dataset,
  checkpoint, training, or scientific model result was produced.

- 2026-08-31: before scientific analysis, the fixed-system attractor pilot was
  frozen and then clarified in `R0_NACA_PHASE_PILOT_CONTRACT.json` (current
  SHA256 `f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5`).
  It uses only solver states/history, requires eight stationary recurrent
  cycles, freezes extension/resource-cap outcomes, and treats 32 phase anchors
  as finite quadrature on one attractor rather than independent trajectories.
  No PCNO artifact or outcome may enter this decision.
- 2026-08-31: the audited trajectory runner completed output indices
  499--1999: 1,501 native states, `3,326,514,699` bytes, and ordered-state
  aggregate SHA256 `9adbcfd53c4812009d3c5e5081730b709a2f63f48237760390e2c8559ab200ef`.
  The run took `5631.17 s` (`0.2665` written states/s); the short one-step
  throughput above is therefore not used as a production-rate estimate.
  Receipt SHA256 is
  `3d857d6ddca4dca111ecc716b2c85ff23f727a4fafc97d39f24db340d2b2672b`.
- 2026-08-31: the unchanged audited analyzer returned
  `PHASE_ONLY_SUPPORTED` on the full trajectory. The CL clock gives period
  `34.6543` steps; 31 complete cycles remain after the selected burn boundary.
  Uniform-node recurrence has median/q90 component-balanced errors
  `0.003845/0.004076`; near-body/wake has `0.003827/0.004038`. The required
  physical-area secondary view fails (`0.110995/0.146430`, q90
  cycle-to-template `1.255897`) because its farfield-weighted orbit amplitudes
  are extremely small; it did not qualify or rescue the decision. Analysis
  SHA256 is
  `b14ef6d66c5e520633ea94954ff3b243a47c3185743de43a5ff10d1c7f74069f`.
  A preceding sandbox-only temp-directory permission failure is preserved as
  `INVALID_ARTIFACT` receipt SHA256
  `e5f4196fa98cfcb17b7ccc8e2874ef5e5ce436e643ce40c8d16c43988c2f2dc4`;
  it consumed no packet data and is infrastructure evidence only.
- 2026-08-31: after 47 focused tests, Ruff, and three independent audits, final
  native cases `su2_naca0012_native_replay_20260831_c` and `_d` both completed
  execution/evaluation under the pinned SU2 `v8.5.0` identity. Their outputs
  are byte-identical (`a39f0b4a...7fab`); the five evolved fields have
  component-balanced relative L2 discrepancy `4.3264e-7` with physical-area
  weights, `5.9447e-4` with uniform-node weights, and `8.9081e-4` in the fixed
  near-body/wake box against canonical target 499; observed local throughput is
  `1.92`--`1.96` transitions/s.
  The comparison packet revalidates both receipts and reports zero
  native-to-native numerical error. The model-relative quarter-margin,
  arbitrary-state restart, population, support, and PCNO gates remain open; no
  corrective-mechanism outcome is implied. Qualification addendum
  `R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json` records that the canonical
  target has no producer-bound integrated coefficients, freezes the common
  target/replay surface functional and auxiliary-field scope, and projects
  storage without rewriting the immutable C/D receipts; its SHA256 is
  `8269354d4a021b921f6c1d99c4f0dd416896b62c23fb5f9beb77fab0f0e960d6`.
- 2026-08-31: the preparation-only NACA replay harness passed 14 focused tests,
  Ruff, and an independent fail-closed audit. The regenerated ignored preflight
  stages only histories 497/498, keeps canonical target 499 external, freezes a
  distinct output stem and non-compact restart schema, and records all native
  solver claims as false. No SU2 executable or scientific evaluator was run.
- 2026-08-31: owner authorized an honest SU2 attempt after asking whether any
  time-dependent benchmark was clearly better. The bounded review found none
  with a better combined scientific fit and completion risk, so Unsteady
  NACA0012 was frozen as the first R0 screen and conditional primary case.
  Stage 0 independently bound and validated the public license, configuration,
  mesh, and restart triplet 497/498/499, including native binary layout, mesh
  coordinates, complete RANS--SA evolved state, and BDF2 history. No native
  solver replay, dataset generation, PCNO training, or PDE result occurred.
  The next milestone is the non-destructive native replay harness and its
  independent code-to-intent audit.
- 2026-08-30: final ODE evidence synthesis reverified every canonical B1/B2,
  landscape, nonlinear, Gaussian, and derived-diagnostic packet against current
  sources. The final-paper ODE core is the exact separation results, learned
  affine target comparison, and affine defect landscape. The Gaussian study
  contributes only its registered density--fidelity association and failed
  all-seed scale gate. Fixed-support nonlinear rollouts, nonlinear diagnostic
  plots, Gaussian planar atlases, and state-only proxy negative controls remain
  documentation or visualization assets rather than paper evidence. No new ODE
  run, post-outcome retuning, optimal-scale claim, rollout ranking, or PDE
  transfer claim is opened.

- 2026-08-30: ODE result-to-claim closeout found no claim-critical additional
  arm. CLEAN, RECOVERY, DYN-RELABEL, and `CLEAN+C_0` already isolate clean-only
  supervision, embedded return, trusted displaced-state supervision, and a
  separately attributable operational projection; the exact study also covers
  finite correction strength and the recovery--relabeling crossover. kNN and
  learned samplers would duplicate approximate projection in this known-circle
  geometry, while pushforward and the relabeling--recovery hybrid are reserved
  for the primary PDE study. The ODE implementation and verified packets are
  frozen; visualization scripts are retained for later aesthetic revisions.

- 2026-08-30: B2-GN closed under canonical packet
  `corrective_ode_gaussian_normal_noise_20260830a`, manifest
  `8902430e479caae18e877407cfbc40c8ddd231bb55dc310bd765990a5af46172`.
  Independent verification closes 72 outputs, 51 checkpoints, six exact current
  sources, the audited parent, and the exact forcing-sign inheritance. Six
  focused tests and Ruff pass. The fixed-radius density--fidelity signatures
  pass for RECOVERY (median Spearman `1.0`, `17/18` positive) and DYN (median
  `1.0`, `53/54` positive). No scale passes the all-seed response gate: most
  misses are the preregistered cross-regime clean-error ratio, and seed 29 also
  exceeds the `2e-3` DYN-M clean-error ceiling at `sigma_train=0.005` and
  `0.02`. All rollout rows therefore remain descriptive. At the largest scale,
  `115/576000` Gaussian draws exceed the declared `|r|=0.15` tube, close to the
  theoretical tail, while none exceeds the `0.5` safety radius. No threshold,
  scale, architecture, or outcome was changed after reveal.

- 2026-08-30: owner authorized B2-GN with normal-coordinate Gaussian input
  corruption only. The frozen standard deviations are
  `{0.005,0.01,0.02,0.04}`; clean/noisy rows are mixed 64/64 with paired
  fresh-per-update draws, and recurrent forcing remains a separate Rademacher
  assay. Implementation and the bounded local-CPU run may proceed only after
  focused tests and an independent code-to-intent `AUDIT_PASS`.

- 2026-08-30: palette-corrected nonlinear diagnostics were regenerated under
  successor packet `corrective_ode_nonlinear_diagnostics_20260830c`. It uses
  the manuscript's `magma`, reference, trusted-path, and categorical arm
  colors while preserving the exact parent computations. The prior `...b`
  packet remains immutable historical provenance.

- 2026-08-30: the parent-bound visualization packet
  `corrective_ode_nonlinear_diagnostics_20260830b` was generated after visual
  review and independently verifies 12 outputs, two current sources, all 15
  checkpoints, and nine learned/reference trajectory digests under manifest
  `942459c972a4935fe89dd95bf055dac78cb3552db1ba5cb864886a0ab9dd36db`.
  Its three regime atlases show trusted-flow defect, clean-return error, output
  distance to the clean set, actual fixed support, and learned/trusted paths;
  its trajectory diagnostic shows retention and visited-state defects. It is
  explicitly retrospective, non-Gaussian, and excluded from active paper
  evidence.

- 2026-08-30: owner clarified that fixed radii near `0.15` are not the intended
  analogue of PDE noise augmentation and directed a Gaussian-input successor
  with an explicit perturbation-scale hyperparameter. B2-NL remains a valid frozen
  record of its exact deterministic-support contract, but it is no longer
  active draft evidence. The missing nonlinear defect landscapes, trajectories,
  and visited-state defects are recoverable from its verified checkpoints as
  a separate read-only diagnostic packet. A new training run requires B2-GN;
  old samples will not be relabelled as Gaussian.

- 2026-08-30: B2-NL closed under canonical packet
  `corrective_ode_nonlinear_stress_20260830a`, which independently verifies 31
  outputs and five current sources under manifest
  `8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40`.
  Three final audits returned `AUDIT_PASS`; nine focused tests and Ruff passed.
  The frozen scientific classification is `response_qualification_failed`:
  one seed's expansive DYN clean error was `2.421e-3` against `2e-3`, its
  cross-regime clean-error ratio was `4.207` against `2`, and two mixed-regime
  log-gains were `-0.525` and `-0.529` outside the `|lambda| <= 0.5` band.
  All signed target effects and all nine normalized in-tube DYN fidelity gates
  passed. Rollout descriptors matched P3--P5, including C/M/E residence
  `1.000/0.998/0.229` at `sigma=5e-3`, but remain descriptive by contract.
  No thresholds, architecture, or training settings were changed after reveal.

- 2026-08-30: owner authorized proceeding to the harder learned-ODE setting
  after inspecting the completed defect landscapes. B2-NL was preregistered
  under a separate identity; its exact source freeze remains pending the
  independent audit, while the verified B1/B2 packets remain current-source
  compatible. The test varies trusted clean-manifold normal response across
  contractive, mixed, and expansive regimes and separately scores robustness
  to the clean path and fidelity to a same-forcing trusted path. No PDE,
  remote, dataset, or sealed action is included.

- 2026-08-30: generated the requested learned-ODE defect-landscape
  visualization before any harder setting. The read-only derived renderer
  verified the canonical B1/B2 parent, reconstructed its training bank, and
  evaluated all nine checkpoints on a common phase--radius grid. The final
  packet corrective_ode_landscapes_20260830e binds parent manifest
  54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99
  and has manifest
  256e175d8d77e352a2fb545e85ce4470e603a9af06cfb8370e1df03562f6ff0a.
  Independent code-to-intent and visual audits passed. The two heatmap rows
  deliberately separate trusted-flow defect from clean-return error; neither
  is an ID/OOD label or online rule.
- 2026-08-30: tightened the manuscript around the corrective-mechanism thesis.
  Execution protocol, packet mechanics, readiness gates, placeholder PDE
  prose, and manuscript TODOs were removed from the rendered paper and retained
  in project Markdown. The ODE section now motivates each assay, defines both
  defect landscapes, explains why noiseless labels do not imply exact fitted
  response coefficients, interprets the reached trajectories, and states the
  ODE-to-PDE limitation explicitly.
- 2026-08-30: B1/B2 closed. Independent code-to-intent review returned
  `AUDIT_PASS`; 13 focused tests passed. Canonical packet
  `corrective_ode_study_20260830d` verified 23 output hashes and three current
  source hashes under manifest
  `54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99`.
  Every exact identity/inequality and every frozen learned mechanism check
  passed. CLEAN had the smallest mean clean one-step error among learned
  predictors but one of three seeds exited the tube catastrophically;
  RECOVERY learned return, DYN-RELABEL learned trusted displaced response, and
  `CLEAN+C_0` separated explicit retention from normal-dynamics fidelity. The
  continuous-phase metric is explicitly a lifted-coordinate path error, with
  chordal state-space error reported separately. The result is supporting ODE
  evidence only; no PDE action was taken.
- 2026-08-30: owner authorized completion of the ODE part and corresponding
  manuscript sections. B1/B2 were frozen before implementation: exact
  scenarios, learned data/target contracts, architecture, seeds, updates,
  checkpoints, metrics, signed predictions, null-ID-rollout handling, and
  artifact verification are now explicit. This authorization is local and
  ODE-only; primary-PDE selection remains deferred.
- 2026-08-30: owner clarified the two-axis thesis: clean one-step supervision
  probes on-reference evolution, while accurate rollout also requires a
  transverse retention/recovery property supplied by an embedded or operational
  corrective mechanism; neither small clean error nor tube retention alone is
  sufficient. The on-reference/tangent observation is background, not the
  novelty claim.
- 2026-08-30: primary-PDE selection was deferred pending the owner's mentor
  discussion. Shock-vortex was reclassified provisionally as a tangential/
  no-harm control. Kolmogorov, balanced rotating shallow water, and one of
  Lambda--Omega/Cahn--Hilliard remain literature candidates only. No candidate-
  specific implementation or scientific execution is authorized.
- 2026-08-30: owner accepted the plan and officially started the project. R0
  source/artifact audit and contract drafting began. No solver, test, dataset,
  model, checkpoint, training, remote, prospective, or sealed action was
  performed. Candidate-specific implementation remains gated by the mentor/owner
  PDE decision and review of the rewritten candidate-specific R0 contract.
- 2026-08-30: documentation and manuscript structure only. Exact prior plans
  and paper sources were archived. Live candidate source/result manifests were
  checked. No solver, model, checkpoint, scientific array, training, remote,
  download, or sealed/test action was performed.

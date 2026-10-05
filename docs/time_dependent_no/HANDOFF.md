# Handoff: Corrective Rollouts and Independent Probe

Updated: 2026-10-05

## Current checkpoint

Section 4.2 now includes the REC/DYN comparison, the equal-weight target
mixture, and the controlled four-step study. Working figures show target
learning, autonomous rollout and local defects. Appendix C.4 gives the sequence
protocol and sampled gradients. The draft compiles and its new pages have been
checked visually; the approved text through Section 4.1 and locked paper plan
are unchanged. Existing references to the removed Section 5 remain in the
locked text and need a later authorized revision.

All five 4,096-update training runs in `KF_MULTISTEP_20261004A` and the ten
models' original twelve-case H32 evaluations are complete. The midpoint improves
pooled validation H8/H32 over REC and DYN, but beats DYN on only two of four
validation cases at H32. In the four-step comparison, frozen inputs give the
lowest pooled error; refreshing inputs worsens H32 on all twelve cases.
Detachment beats full differentiation on all twelve cases at H8 and H32.
The 192 sampled intermediate-state gradient cosines are positive, sometimes
nearly orthogonal: opposition between the immediate loss and the sum of later
losses is not observed on these four onset windows. These results use one
training run per condition and do not give universal rankings.

The ignored run root is `artifacts/time_dependent_no/kf_multistep_20261004a/`.
`ANALYSIS.md` owns the numerical interpretation; `paper_figures/index.html`
contains the working figures. `all_model_field_verification.json` reproduces
metrics exactly from all 120 saved trajectory files. The midpoint and sequence
pairing receipts verify every training minibatch; independent review checked
source identities and all sixteen gradient-file hashes. The original queue's
plotting failure and separate repaired source snapshot remain preserved.

Common-input and empirical-geometry measurements are complete; the final
solver response diagnostics are running. The bounded `sync_results.py`
collector retrieves and verifies their results. Geometric residuals need the
validation ground-truth coverage baseline and cannot be read as exact manifold
distances. Their main-text role remains for discussion with the owner, as does
the remaining mechanistic evidence. Reported peak GPU allocation includes
common offline diagnostics and does not isolate optimization memory.
Review this stage before extending horizons or launching more method families.

`KF_CORRECTIVE_FOLLOWUP_20261002A` completed its 15 training jobs (65536 new
updates), 25 model/checkpoint evaluations and 61 paired target requests on the
selected workstation. All evaluations use the original eight training and four
validation trajectories through step 32. No experiment remains queued in this
authorized batch. Elapsed queue time was 37.51 hours. The protocol remains in
the October 2 follow-up section of [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

The completed numerical analysis is `ANALYSIS.md` under
`artifacts/time_dependent_no/kf_corrective_followup_20261002a/`.
Open `paper_figures/index.html` for 29 verified PDF/PNG figure pairs, captions,
per-trajectory measurements and paired comparison tables. The gallery now leads
with autonomous clean/reached-input/probe diagnostics and training-target
specialization versus transfer. PDFs and captions are exported to the ignored
`paper/figures/kf_corrective_followup_20261002a/` directory. The report identifies
main-text candidates and supporting appendix controls; it does not change the
locked paper plan. The agreed main-text selection is now included in the draft
as described above.

The clean-error/rollout discrepancy persists across all twelve cases. One-step
REC/DYN give modest H32 improvements, while mixed-length and Gaussian exposure
give much larger improvements. Longer training helps the latter conditions;
more early one-step inputs and another initialization do not remove the large
one-step-condition failure. REC/DYN learn their respective training targets, but
the target ranking can change on generated inputs. Local/probe defects do not
uniformly rank autonomous rollout accuracy. Read the report for numbers, case
counts, qualifications and the distinction between resolved controls and still
open scientific questions.

The automatic metric/figure collector finished successfully. All 162 retrieved
metadata/metric/log files match their final snapshot; all 300 saved trajectory
files on the workstation reproduce rollout and defect metrics exactly. The
paired target identity closes within 9.8e-15. Nine focused CPU tests pass,
including a check of population and horizon aggregation. Receipts are
`analysis_verification.json`, `final_field_verification.json` and the figure
manifest. The frozen deployed source and the analysis renderer have separate
hashes; preserve both identities.

The optional full local copy of raw trajectory fields and 16 evaluated
follow-up checkpoints is in progress through the run-specific
`sync_results.py --full` command. `complete_retrieval.json` records its success;
`completed_fields_manifest.json` owns member hashes. This archival transfer does
not block the verified numerical analysis. Machine-specific details stay in
ignored artifacts. The existing runner and plotter remain the maintained entry
points under `scripts/time_dependent_no/`.

The October 2 follow-up is complete. The October 4 stage now tests refreshed and
attached/detached four-step training. Longer horizons, broader comparison design
and the second PDE remain future stages for owner review.
The present results do not establish exact manifold retention,
uniform stability, universal target rankings or fully converged training.

The owner manually revised and locked the [paper plan](../../paper/PAPER_PLAN.md)
on October 1. Do not modify it without the owner's explicit permission, including
formatting, corrections, status notes or added TODOs. Read the live file as the
authority for the intended paper; earlier assistant drafts are superseded.

The completed comparison pairs recovery and dynamics targets under three input
conditions (six training runs): Gaussian, native one-step errors, and equally
sampled native errors from 1, 2, 4 and 8 prediction steps.
Use the existing pretrained model as the common frozen generator, sampling
errors at different times and after one or several prediction steps without
rescaling their magnitudes. Gaussian noise should be of the same order as
deployment errors; exact amplitude-distribution matching is not required.
The concrete training protocol is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
The probe completed 49,152 updates, and all six corrective models completed
their fixed 4,096 updates. On October 2 the owner requested completion of the
remaining approved experiments, deferring interpretation until all are done.
Evaluation `evaluation_20261002a` is now complete for the two original baselines
and all six corrective terminals, using the same first training and validation
initial conditions and 32-step horizon as Experiment 1. All checkpoints, logs,
rollout fields and probe measurements have been retrieved and verified.
No intermediate checkpoint selection, additional training conditions or new
populations were introduced.

The final local check verified the complete archive and 40 individual artifact
hashes. Recomputing rollout and defect metrics from all sixteen saved trajectory
field files reproduced the saved values exactly; the signed error identity
closed within 2.3e-15. All six training logs have identical sample indices.
The earlier implementation verification passed 37 focused CPU tests.
Local checkpoints, source receipts and the complete figures are under
`artifacts/time_dependent_no/kf_displacement_comparison_20261001a/analysis_20261002a/`.
Open `results/evaluation/figures/index.html` for all six PDF/PNG figures and
`summary.csv` for the numerical table. `evaluation_verification.json` records
the final checks. The bounded retrieval helper has finished; its status is
`completed`. The owner's subsequent questions prompted a completed training
investigation: all stored training inputs, common queries on the pretrained
pilot rollouts, and an autonomous rollout of the independently trained clean
probe on the same two cases. No new training or solver labels were generated.
`ANALYSIS.md` now leads with this investigation and retains earlier snapshots.
Open `training_investigation/index.html` for the new training-loss, spectrum and
common-response figures. One-step REC/DYN learn modest training-loss reductions
but preserve the large response on reached inputs; the independent clean model
also fails, with a different full error curve. Onset coverage and an exact
selected-center clean control are concrete design issues for the next owner
discussion. The newly approved follow-up above now addresses these issues.
Attached/detached multistep training is a later comparison. The independent
probe's fixed terminal is used in these diagnostics and remains reusable; it
is no longer a requirement for Experiment 1.
The first study's completed probe TODO has been replaced by the remaining
mechanistic questions in the current manuscript edit.

The owner approved writing the first Kolmogorov result alongside the ongoing
experiments, using existing figures until a final presentation pass. Section
4.2 now introduces the problem and the pretrained/clean-continuation pilot,
with measurement definitions, a numerical summary, the three-measurement
figure, and interpretation through the framework. Appendix C contains the
necessary setup and common-input comparison. Red TODOs identify broader
evaluation, independent-probe calibration/results, and later figure redesign.
The draft reports the two-case scope and leaves geometric departure and a full
causal explanation open. Approved text through Section 4.1 and the locked plan
are unchanged. This writing step does not expand experiment authorization.

After reviewing the first result and its animations, the owner accepted its
role: lower clean training and validation errors can accompany worse rollout.
The pilot comparison pools the first 32 transitions of one training and one
validation trajectory; it does not assert lower clean error at every time.
Data scaling is reserved for later if time permits. Continue prioritizing the
mechanistic study, with the next experiment design subject to owner review.
This priority decision leaves the locked paper plan unchanged.

The owner has approved the overall experimental question structure paralleling
Sections 2--3 and the first experiment's question and evidence requirements.
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) records the protocol. The owner selected
the existing pretrained/clean-continuation pair and approved proceeding, with
visualizations as the main artifacts. Do not replace or retrain that pair.
The first training and first validation cases through step 32 are the pilot
review scope before expanding to all twelve.
After reviewing the first figures, the owner requested clearer trajectory-based
visualizations and authorized training a reusable independent clean probe.
The probe uses fresh seed 29 and the original fixed 49,152-update clean recipe;
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) records its scope. Existing A/B models
and diagnostic data remain fixed.
The two-case pilot is now complete. Review the PDF/PNG figures under
`artifacts/time_dependent_no/kf_local_defects_20261001a/figures` before broader
evaluation. [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) records the results
and numerical checks. The validation case shows improved clean error at all
32 inputs but worse common-input defects at most reached states. Own-rollout
defects do not give a uniform ranking. The revised primary figures are
`trajectory_one_step_errors` and `trajectory_rollout_errors` (PDF/PNG): the first
fixes a predictor per column and compares three input trajectories; the second
shows accumulated error separately. Local errors now use input-state time.

The owner subsequently requested a direct test of the overlapping A/B error
curves. The completed audit confirms closely aligned prediction errors and
learned increments on identical reached inputs, with distinct checkpoints and
model storage. See the tracker and `similarity_20261001a/prediction_similarity`
under the existing pilot artifact root. This used only frozen-model inference
on the two saved pilot cases; the authorized follow-up above now expands
evaluation to the original twelve cases.

The requested spatial animations are now in the same artifact root's
`animations/` directory. `trajectory_viewer.html` provides playback and time
scrubbing for rollout fields, own-input defects and identical-input defects;
four smaller MP4s provide rollout/own-defect previews for the two cases.
Saved common-input fields, numerical replay checks and browser verification are
recorded in the tracker. Review these before choosing the next experiment stage.

Probe run `kf_clean_probe_20261001a` started on the selected workstation at
06:24 UTC on October 1. Launch verification observed finite updates and a
recoverable initial checkpoint. The measured estimate is about 11 hours, with
a 14-hour session limit and an unchanged fixed update count. Local resource and
launch receipts are under `artifacts/time_dependent_no/kf_clean_probe_20261001a`.
The trainer saves `latest.pt` every 1,024 updates and a fixed `terminal.pt` at
49,152. Its completed checkpoint has now been retrieved and hash-verified;
the evaluator verifies identity and reload, then measures clean competence over
the same pilot time interval. It is not tuned or selected using separation
between rollout curves. Full twelve-case evaluation is included in the
authorized follow-up above.
Start with Section 4.2's
mechanistic study. Leave the comprehensive comparison for Sections 4.2--4.3 until
later, following the locked plan's sequence. Version 2 is due before October 8,
with October 7 as the completion day.

## Required owner review

The bounded October 2 follow-up above is explicitly authorized. For additional
stages, present a concrete, reviewable proposal for owner review. The working sequence is scientific questions
and evidence requirements; experimental design and measurement definitions;
implementation and verification; pilot results and broader execution; then
interpretation and follow-up decisions. Approval covers its stated scope, not
unreviewed later stages. Do not launch experiments or begin implementation while
the scientific questions and design remain under discussion.

Scientific concerns or ambiguities in the locked plan should be raised in the
discussion, not silently edited into the plan or treated as established findings.
Record approved experimental decisions in the existing experiment documents when
their stage is reached; do not create a parallel paper plan or revive old queues.

## Manuscript checkpoint

The direction is a rigorous comparison that explains corrective behavior and
provides practical guidance. The locked structure is Introduction, Framework
and measurements, Corrective methods, Numerical experiments, and Discussion and
conclusions. The plan omits the old standalone intervention/forecast study.
Any new method belongs in the methods section; proposing one is optional.

Experiment implementation now follows the approved plan. The manuscript preserves
the approved abstract and Sections 1--4.1 and now drafts the first version 2
Kolmogorov result in Section 4.2. Later comparisons and the synthesis remain
unfinished. The plan proposes revisions to Sections 2--3, retains the
Section 4.1 illustration, and defers the Introduction and abstract until the
remaining material is settled. Preserve manual edits during future revisions.

The appendix retains the proofs and signed-response identity supporting
Sections 2--3, the reproducibility details for Section 4.1, and the new pilot's
Kolmogorov setup and common-input measurements. Superseded experimental material
remains outside the manuscript. The previously recorded
private version 1 archive was absent in the October 1 local check. An earlier
plan snapshot was present under the appendix-consolidation build directory.
Preserve existing deletions. This remains a mentor-facing
checkpoint, not a submission-readiness claim.

Kolmogorov and a second, unselected PDE are the intended empirical coverage.
Method coverage, diagnostics, comparison scales and implementation choices remain
open. No experiments or implementations are authorized by the paper-plan rewrite.
Existing workstation access persists without an implicit launch instruction.
Venue is undecided. The current paper plan takes precedence over older schedules
and experiment queues.

## Evidence and manuscript pointers

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) owns current experiment planning and
  its approval status, followed by the historical completed-study index.
- [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) owns exact results, source and
  artifact bindings, failed qualifications and historical attempts.
- Twelve-map development coverage and nine-map, 64-trajectory confirmation
  have different scopes. Stationary-distribution methods remain conceptual.
- H8 forecasts, H32 adaptation comparisons and H128 limits remain historical
  evidence; they are removed from the active manuscript and do not prescribe
  the new study.
- The current manuscript is `paper/main.tex`; current PDF is `paper/main.pdf`.
  The first Kolmogorov draft's build log is
  `paper/build/kf_pilot_20261001a/compile.log`. Three existing `sec:design`
  references in locked Sections 1--3 remain unresolved; new references resolve.
  The locked [paper plan](../../paper/PAPER_PLAN.md) owns the paper's current
  structure and next-step sequence. Earlier build receipts describe dated snapshots.
- [LITERATURE_AUDIT.md](../../paper/LITERATURE_AUDIT.md) owns source verification.
  Keep established response matching and local forecasting distinct from the
  paper's contribution through controlled comparisons and useful diagnostics.

## Preservation and authority

Keep all manuscript sources, scientific artifacts, source snapshots and original
receipts. The private `paper/` directory stays ignored and uncommitted.
Private context, historical protected populations and external-review approval
boundaries remain in force. Do not revive old experiment or cleanup workers.

The documentation state before this compaction is retained at local Git commit
669f7fc. The completed experiment-plan archive is indexed in
[history/README.md](history/README.md); the preceding private paper plan is
`paper/archive/PAPER_PLAN_before_cleanup_20260928.md`.
The [README](README.md) owns code and retention navigation.
Current human direction takes precedence over historical recommendations.

The September 28 cleanup preserves all pending research Python files unchanged.
Its bounded CPU verification exercised 727 cases: 709 passed, 5 failed and 13
had setup errors; it is not a fully passing suite. Logs and remaining limitations
are retained in `artifacts/time_dependent_no/local_cleanup_20260928a/verification.json`.

# Time-Dependent Neural Operators

Updated: 2026-10-05

This page owns navigation, code ownership and preservation boundaries for
`time-dependent-no`. Following the September 30 mentor discussion, the
[version 2 paper plan](../../paper/PAPER_PLAN.md) was manually revised and locked
by the owner on October 1. Do not modify it without explicit permission.
Experiment planning now begins with Section 4.2's mechanistic questions; the
owner must review each stage before work advances, as recorded in
[HANDOFF.md](HANDOFF.md). The first existing-pair pilot has clearer trajectory
figures for owner review. The owner also authorized the reusable independent
clean probe and six-run corrective-training comparison. All six corrective runs,
the probe, and the two-case rollout/probe evaluations are complete and retrieved.
On October 2 the owner requested finishing this batch before interpretation;
the complete figure gallery and verification receipts are linked in the handoff.
The follow-up training investigation is also complete: common-input response
measurements, training losses and the independent clean model's rollout are
linked there, with design limitations recorded for owner review.
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) now also records the owner-approved
October 2 follow-up: paired target diagnostics, training and sampling controls,
and the original twelve-case evaluation. That queue is complete: 15 training jobs,
25 evaluated models/checkpoints, a numerical analysis and 29 figure pairs.
Its entry points are `run_kolmogorov_corrective_followup.py` and
`plot_kolmogorov_corrective_followup.py`; the handoff owns status and artifact
navigation. The locked paper plan is unchanged.
The owner-approved October 4 stage adds a REC/DYN midpoint and four controlled
sequence-training runs. `run_kolmogorov_multistep_study.py` owns the fixed queue,
`diagnose_kolmogorov_multistep.py` its offline measurements, and
`plot_kolmogorov_multistep_study.py` its vector figures and comparison tables.
See the experiment plan for the protocol and the handoff for live status.
All five training runs and ten models' twelve-case evaluations are complete.
The midpoint and four-step comparisons are analyzed and incorporated into
Section 4.2 with working figures; Appendix C.4 records the protocol and sampled
gradients. Common-input and geometry measurements are complete, and final solver
response diagnostics are running. The handoff links the results and receipts.
The target is a
second draft before October 8, completed by October 7. The manuscript preserves
approved text through Section 4.1 and now includes the Kolmogorov clean-training,
REC/DYN, target-mixture and four-step comparisons in Section 4.2, with working
figures and supporting appendix details. Other planned
revisions remain pending; venue remains open.

## Start Here

Read repository `AGENTS.md`, the [research decision](RESEARCH_DIRECTION_DECISION.md),
[handoff](HANDOFF.md), and the [mechanistic index](MECHANISTIC_DIAGNOSTIC_TRACKER.md)
when historical routing is needed, then this page. Read ignored
`LOCAL_CONTEXT.md` privately; inspect live Git status before editing and verify
source/artifact manifests before using an exact result.

| Question | Source of truth |
| --- | --- |
| Scientific scope and claim boundaries | [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) |
| Current checkpoint and next action | [HANDOFF.md](HANDOFF.md) |
| Completion criteria and mentor milestones | [PROJECT_PLAN.md](PROJECT_PLAN.md) |
| Manuscript claims, figures and section work | [paper/PAPER_PLAN.md](../../paper/PAPER_PLAN.md), local and ignored |
| Experiment planning and historical studies | [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) |
| Exact results, contracts and execution records | [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) |
| Historical IDs and bounded lookup | [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md), [history index](history/README.md) |
| Code dependencies and reproducibility | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| Immediate editorial sequence | [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) |
| Public-source literature checks | [paper/LITERATURE_AUDIT.md](../../paper/LITERATURE_AUDIT.md), local and ignored |

The paper compares corrective methods, explains their effects on autonomous
rollout, and seeks practical guidance with little problem-specific engineering.
The plan retains the dynamical-system illustration and calls for Kolmogorov plus
a second PDE, still to be selected. Comparison protocols and scale/distribution
coverage are deferred to experiment planning. The current paper plan supersedes
older editorial schedules; historical queues and prepared payloads are not new
authorization.

## Code Ownership And Dependencies

- Reusable project code: `utility/time_dependent_no/`.
- Experiment entry points and figure producers: `scripts/time_dependent_no/`.
- Focused tests: `tests/time_dependent_no/`.
- Leave unrelated examples and core APIs unchanged unless the task requires them.
- Kolmogorov utilities include `pcno_kolmogorov.py`,
  `kolmogorov_reference.py` and `path_conditioned_tube.py`; fitters, evaluators
  and population generators remain under the project script directory.
- `run_kolmogorov_displacement_comparison.py` owns the completed native-error and
  Gaussian REC/DYN comparison. It preserves native magnitudes, qualifies shared
  solver labels and reuses the existing two-branch training update. The older
  rescaled-direction generator remains unchanged for historical reproduction.
- `evaluate_kolmogorov_corrective_rollouts.py` and
  `plot_kolmogorov_corrective_rollouts.py` own the new comparison's autonomous
  rollouts, independent-probe measurements and working figures. They retain
  fixed terminal selection and label unfinished methods as pending.
- ODE study/landscape scripts, NACA PCA and corrective evaluators/visualizers,
  and Bump corrective scripts retain their supporting-case roles.

Historical code can remain an active dependency. In particular:

- Paired fitting imports clean-training sampler/helpers and verifies source
  hashes and function-origin paths.
- Bank generation imports common-solver helpers that depend on readiness code.
- `run_corrective_ode_gaussian_normal_noise.py` imports
  `run_corrective_ode_nonlinear_stress.py`.
- NACA successor/extension code imports R0 dataset, encoding, evaluator and
  trainer helpers; transport code imports earlier helpers and a Bump
  `inspect.sh`; B5 reuses Bump scaling helpers.
- `fit_kolmogorov_debug.py` and `smoke_pcno_kolmogorov.py` retain fixture and
  qualification roles. Closed intervention scripts reproduce closed checks.

Check imports and frozen source contracts before moving or deleting old code.
Absence from the current figure include graph does not make a scientific
plotter or retained figure disposable.

## Retention And Access

Preserve datasets, checkpoints, frozen source bundles, results, manifests,
launch/retrieval receipts, original drafts and historical records. Interrupted
backups and local recovery copies remain protected; an inaccessible path is
unknown, not empty. Do not change ACLs or follow reparse points during cleanup.

Historical protected populations stay closed. The completed 64-trajectory
confirmation used fresh evaluation-only draws; it does not open other test sets.
Keep private machine details in ignored local context. Do not commit raw data,
checkpoints, large logs or private manuscript material. External AI review and
public release retain their content-based approval boundaries.

Use [history/README.md](history/README.md) for superseded plans. Their dated
instructions preserve provenance and do not resume experiments.

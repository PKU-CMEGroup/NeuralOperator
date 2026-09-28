# Time-Dependent Neural Operators

Updated: 2026-09-28

This page owns navigation, code ownership and preservation boundaries for
`time-dependent-no`. The completed studies now support manuscript consolidation:
the abstract and Sections 1--3 are locked; Section 4 is next. New research waits
for the mentor discussion. Venue selection remains open.

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
| Completed studies and their paper roles | [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) |
| Exact results, contracts and execution records | [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) |
| Historical IDs and bounded lookup | [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md), [history index](history/README.md) |
| Code dependencies and reproducibility | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| Immediate editorial sequence | [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) |
| Public-source literature checks | [paper/LITERATURE_AUDIT.md](../../paper/LITERATURE_AUDIT.md), local and ignored |

The paper studies corrective mechanisms for autonomous rollouts from
in-distribution initial conditions with fixed, sufficiently rich training data.
Kolmogorov is the main PDE case; ODE, NACA and Bump evidence is used selectively.
The completed-study index distinguishes development comparisons from independent
confirmation. Historical queues and prepared payloads are not new authorization.

## Code Ownership And Dependencies

- Reusable project code: `utility/time_dependent_no/`.
- Experiment entry points and figure producers: `scripts/time_dependent_no/`.
- Focused tests: `tests/time_dependent_no/`.
- Leave unrelated examples and core APIs unchanged unless the task requires them.
- Kolmogorov utilities include `pcno_kolmogorov.py`,
  `kolmogorov_reference.py` and `path_conditioned_tube.py`; fitters, evaluators
  and population generators remain under the project script directory.
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

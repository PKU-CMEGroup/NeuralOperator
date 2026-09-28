# Time-Dependent Neural Operators

Updated: 2026-09-24

This is the onboarding and code map for the `time-dependent-no` branch.
The owner has authorized coding and the fixed-data corrective comparison in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), on the selected workstation configured
in ignored `LOCAL_CONTEXT.md`. Manuscript structure and the source survey are
complete. The first scientific draft is ready for mentor review as of September
24, with the selected empirical programme closed. The target is Journal of
Computational Physics; the paper plan owns revision priorities and the delivery
schedule retains October 8 as the polishing target.
[HANDOFF.md](HANDOFF.md) owns the current checkpoint;
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) owns dated execution evidence.
The current comparison follows that explicit authorization; old run directories do
not resume historical queues.

## Read Order And Scope

Read repository `AGENTS.md`, [research decision](RESEARCH_DIRECTION_DECISION.md),
[handoff](HANDOFF.md), [experiment index](MECHANISTIC_DIAGNOSTIC_TRACKER.md),
then this page; read ignored `LOCAL_CONTEXT.md` privately and inspect live Git
status and source/artifact manifests before editing or running anything.

The empirical setting fixes sufficiently rich data to identify nearby reference
behavior; severe clean one-step overfitting is not the intended explanation.
The framework separates clean accuracy, error propagation and corrective bias.
Retention/recovery is not sufficient without accurate on-path
dynamics, and projection is not a universal winner. The study concerns
self-composition from ID initial conditions, not exogenous OOD evaluation or a
deployable OOD detector. Solver-relative diagnostics are offline instruments.

Current objective: diagnose, predict quantitatively, choose an intervention
and verify its effect. Keep the current fixed-data Kolmogorov case; new
comparisons follow the scientific plan, not an old prepared queue.
Selected representative-method coverage and the three paired response-preserving
design fits and independent 64-trajectory confirmation are complete. H32
protection against clean-only adaptation transfers to every matched path, while
onset benefit varies; qualified H8 forecasts track both harm and improvement.
H128 fidelity remains poor, including guarded outcomes. The handoff owns the
verified closeout; the paper plan owns the remaining manuscript consolidation.
The completed studies hold all 32 clean training paths fixed: clean/online
recovery, detached/full K=4 gradients, and same-input REC/DYN targets. The
handoff records their different response, bias and long-horizon limits and the
conditional Refiner acquisition limit and the closed fixed-blend and prefix-switch
feasibility checks, neither of which selected a candidate rollout.
The Refiner continuation is closed at its fixed terminal;
the handoff owns the acquisition decision. The paper must cover all representative methods
identified in the literature review; its coverage table is in the experiment
plan. Completed tangent pilots are supporting
diagnostics; neither a data-size contrast nor a forecasting project is next.
ODEs remain supporting laboratories; NACA and Bump enter the paper selectively.
Older M1, REALM, D-series and W26 queues do not restart implicitly.
Historical protected prospective/test populations remain closed; the newly
authorized 64 trajectories are evaluation-only draws from the unchanged ID law.

## Sources Of Truth

| Question | Maintained document |
| --- | --- |
| Thesis and scientific scope | [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) |
| Scope, work packages and completion criteria | [PROJECT_PLAN.md](PROJECT_PLAN.md) |
| Experiment targets and comparisons | [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) |
| Reusable implementation and proportionate future checks | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| Current operational checkpoint | [HANDOFF.md](HANDOFF.md) |
| Draft targets and weekly mentor-feedback schedule | [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) |
| Exact results, attempt IDs and receipts | [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) |
| Historical routing / inactive work | [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md), [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md) |
| Theory / literature | [theory and taxonomy](CORRECTIVE_MECHANISMS_THEORY_AND_TAXONOMY.md), ignored [literature ledger](../../paper/LITERATURE_AUDIT.md) |
| Manuscript | ignored [paper plan](../../paper/PAPER_PLAN.md), [main.tex](../../paper/main.tex) |

Keep execution narratives in the tracker, not duplicated across this README,
the handoff and every planning document. Frozen contracts and old source
snapshots are evidence; preserve their exact bytes.

## Code Classification

The September 13 top-level inventory has 214 entry scripts, 72 utility modules
and 183 tests. Those counts do not measure active scope. The untracked
Kolmogorov additions are real research code, not disposable outputs.

| Class | Concrete scope | Treatment |
| --- | --- | --- |
| Retained candidate-case implementation | `pcno_kolmogorov.py`, `kolmogorov_reference.py`, `path_conditioned_tube.py`; Kolmogorov population, fit, rollout, response, solver and paired-bank entry points and tests | Preserve the current source-bound dependency closure; availability does not select a run. |
| Direct manuscript support | Exact/Gaussian ODE study and landscape producers; NACA PCA, successor and corrective-extension pipelines | Keep code, tests, canonical results and presentation derivatives. |
| Supporting / historical reproduction | B5 Bump; older M1, Euler/CPG, shock-vortex, REALM, D-series and W26 work | Park by role, not by deleting dated filenames. Check imports and frozen manifests before physical relocation. |
| Reusable infrastructure | Runtime/rollout, metric and artifact helpers; existing reference solvers and adapters | Reuse only after checking semantics. Different hash encodings, loaders and JSON policies are not interchangeable. |
| Run-local orchestration | Ignored transport scripts, reviews, source bundles, launch/retrieval receipts | Freeze existing attempts. Reuse only as needed; do not rewrite historical receipts. |
| Disposable generated state | Python/pytest/Ruff caches and verified synthetic test outputs | Remove only after exact inventory and ownership checks; retain test XMLs and meaningful failure evidence. |

### Retained Kolmogorov Dependency Boundaries

All names below are under `scripts/time_dependent_no/`, unless a utility is
identified explicitly:

- Data/reference: `generate_kolmogorov_population.py`,
  `extend_kolmogorov_training_population.py`, and the
  `screen_kolmogorov_*` qualification scripts.
- Learner: `fit_kolmogorov_clean.py`, `fit_kolmogorov_coverage.py`,
  `fit_kolmogorov_paired.py`, with utility `pcno_kolmogorov.py`.
- Fixed-data comparison: `fit_kolmogorov_recovery.py` uses exact archived clean
  labels for continuation and online recovery; `evaluate_kolmogorov_recovery.py`
  separates clean/response assays from H128 autonomous evaluation. Its `replay`
  and `reached_response` phases verify frozen trajectories and compare shared
  amplitudes along Gaussian and native-error directions without refitting.
- Temporal-gradient contrast: `fit_kolmogorov_unroll.py` uses matched K=4
  sequence losses with detached/full temporal gradients; the same evaluator
  validates and scores its terminal checkpoints.
- Target contrast: `generate_kolmogorov_target_bank.py` qualifies identical
  REC/DYN inputs and solver labels; `fit_kolmogorov_targets.py` changes only the
  displaced target in matched fits. The same evaluator's `target_assay` scores
  both targets and trusted response before autonomous outcomes.
- Conditional refinement: `fit_kolmogorov_refiner.py --phase resource|fit`
  runs train-only acquisition using `pcno_kolmogorov_refiner.py`. It checks
  four denoising levels and full EMA transitions. Acquisition evidence is
  separate from the completed stochastic comparison; paired evaluation shares
  noise across common inputs.
- Conditional generation: `fit_kolmogorov_acdm.py --phase resource|fit` uses
  `pcno_kolmogorov_acdm.py` for joint noisy-condition training and 20-call DDPM
  transitions. `evaluate_kolmogorov_acdm.py --phase assay|rollout` measures
  clean sample spread, response with shared noise, and sampled autonomous paths,
  reusing the existing deterministic scoring helpers. The frozen plans separate
  acquisition from evaluation; the handoff owns live status.
- Diagnosis: `evaluate_kolmogorov_clean_rollout.py`,
  `evaluate_kolmogorov_coverage_rollout.py`,
  `evaluate_kolmogorov_common_response.py`,
  `screen_kolmogorov_common_solver.py`, and
  `generate_kolmogorov_paired_bank.py`.
- Predictive calibration: `evaluate_kolmogorov_tangent_forecast.py` uses utility
  `forced_tangent.py` for clean-path JVP propagation, signed probes and saved-error
  decomposition. Its `response_transition` phase compares directions at matched
  amplitudes. The completed pilots expose short-window linear validity and
  subsequent finite-amplitude breakdown; scope and evidence live in the plan
  and tracker.
- Intervention design: `evaluate_kolmogorov_intervention.py` and
  `evaluate_kolmogorov_prefix_switch.py` use utility `intervention_sensitivity.py`
  for map-mixture and initial-displacement responses along a frozen baseline.
  Run as Python modules; the latter separates `--phase forecast` from a
  hash-bound `--phase outcome`. Retain them to reproduce the closed design
  checks; they did not select a candidate or authorize a schedule sweep.
- Current design fit: `fit_kolmogorov_response_preserving.py` trains matched
  clean-only/response-preserving copies of bank DYN, using archived training
  responses. Its terminal recipes load through the existing recovery evaluator;
  no extra network or teacher is used at deployment. The initial worker runs
  fitting and nonrecurrent diagnostics before autonomous outcomes. `--seed`
  selects the paired sampling order without changing the fitting recipe.
- `evaluate_kolmogorov_adaptation_forecast.py` predicts H8 with full candidate
  Jacobians on a fixed saved DYN path, retaining signed remainder checks.
  Its nonlinear probes do not update the predicted trajectory. Reuse the fixed
  forecast and qualification for the paired repeats; do not extend its horizon.
- `fit_kolmogorov_debug.py` and `smoke_pcno_kolmogorov.py` retain qualification
  and fixture roles; do not delete them merely because their runs are complete.

Some scripts currently serve as libraries. Paired fitting imports Clean's
sampler/helpers and verifies source hashes **and function-origin paths**.
Bank generation imports common-solver helpers, which depend on earlier
readiness code. Moving these files would change the frozen implementation.

The SSH stdin/EOF execution primitive and identical Kolmogorov array hash
have multiple callers, but extracting them is not a pending scientific task.
If a selected change later requires extraction, preserve byte-level parity
and the old replay closure. Do not add a general experiment framework.

### Paper Producers And Tests

| Live manuscript material | Producers to keep |
| --- | --- |
| ODE table and defect landscape | `run_corrective_ode_study.py`, `plot_corrective_ode_landscapes.py` |
| Gaussian ODE appendix | `run_corrective_ode_gaussian_normal_noise.py` and its imported `run_corrective_ode_nonlinear_stress.py` parent |
| NACA low-dimensional structure | `visualize_naca0012_dataset_pca.py` and the R0 dataset/encoding helpers |
| NACA comparison and figures | `evaluate_pcno_naca0012_successor.py`, `evaluate_pcno_naca0012_corrective_extension.py`, corresponding visualizers/trainers and utilities |
| Bump supporting comparison in Appendix B | `run_pcno_bump_corrective.py`, `pcno_bump_corrective.py`, B5 contracts and packets |

Keep the raw-input XY/nonlinear ODE plotters and NACA rollout visualizer for
later aesthetics, as requested by the owner. Absence from the current figure
include graph is not a deletion criterion. The handoff records the outstanding
ODE figure-byte provenance check.

For presentation edits, begin with the landscape, NACA PCA and NACA
corrective-extension visualization tests. For ODE target/dynamics changes add
the exact-study, nonlinear-stress and Gaussian-normal-noise tests. NACA pipeline
tests remain shared coverage for encoding, FP32 recurrence and immutable reads;
Bump scaling tests cover helpers reused by B5. For Kolmogorov changes, run the
matching entry-point tests and their imported fixture/helper dependencies.

Use synthetic CPU fixtures and an OS temporary directory for scratch. Do not
put disposable pytest working trees beside canonical result packets. A passing
test does not authorize data, checkpoint, solver, GPU or protected-role access.

## Artifact Retention

The September 13 inventory and cleanup receipts are under ignored
`artifacts/time_dependent_no/local_cleanup_20260913a/`. This is metadata-only
inventory plus cleanup evidence, not full scientific revalidation. Permission-
denied directories are unknown, not empty; reparse points were not followed.

Preserve:

- active Kolmogorov inputs, checkpoints, banks, source snapshots and manifests;
- exact/Gaussian ODE evidence; NACA PCA and both relevant presentation versions;
- B5 and historical claim-bearing packets, including failed/incomplete attempts;
- all local checkpoint recovery copies from earlier remote cleanups;
- source, review, launch and retrieval receipts, and protected-role access state.

In particular, `autodl_cleanup_20260912a/` contains an interrupted backup, not
trash. B/C stopped before deletion; successor E completed the exact 79-file
subset. Its closeout and plan bind the removals and retained recovery paths.
Use the [dated closeout](archive/pre_predictive_reset_20260916a/HANDOFF.md)
and exact retained receipts before any separately authorized cleanup; never resume the original
worker against its now-stale full plan. Keep all local recovery copies after
their remote duplicates are removed. The older CUDA-smoke transport also
contains recovered July checkpoints. The current transport imports earlier
helpers and an older Bump `inspect.sh`; those directories are dependencies.

Deduplicate a result only after checking complete member coverage and an
independently recorded archive hash. Do not substitute metadata-only tarballs
for checkpoint bytes. Do not change ACLs or remove inaccessible scratch merely
because its name contains `pytest`.

## Privacy And Recovery

Keep private paths, hosts, credentials, datasets and reviews in ignored local
storage. Do not commit or push `paper/`. Do not send unpublished material to
external AI tools without explicit approval of the exact prompt/scope.

The pre-cleanup README and handoff are preserved byte-for-byte in
[archive/pre_sprint_20260913a/](archive/pre_sprint_20260913a/).
Resolve their old relative links against `docs/time_dependent_no/`.
Use [history/README.md](history/README.md) for earlier recovery anchors.

The [September 16 planning snapshot](archive/pre_predictive_reset_20260916a/INDEX.md)
preserves the preceding working-tree plans and cleanup checkpoint exactly.
Its expired dates and forward queues are historical evidence. The current
project plan owns completion; the delivery schedule owns current draft targets.

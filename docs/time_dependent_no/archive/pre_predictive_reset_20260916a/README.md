# Time-Dependent Neural Operators

Updated: 2026-09-13

This is the onboarding and code map for the `time-dependent-no` branch.
The owner has paused experiments for a local pre-sprint audit and cleanup.
[HANDOFF.md](HANDOFF.md) owns the current checkpoint;
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) owns dated execution evidence.
Neither this page nor an old run directory is authorization to launch work.

## Read Order And Scope

Read repository `AGENTS.md`, [research decision](RESEARCH_DIRECTION_DECISION.md),
[handoff](HANDOFF.md), [experiment index](MECHANISTIC_DIAGNOSTIC_TRACKER.md),
then this page; read ignored `LOCAL_CONTEXT.md` privately and inspect live Git
status and source/artifact manifests before editing or running anything.

The framework separates on-reference accuracy from displaced-input response.
Clean one-step supervision does not identify the latter without additional
assumptions. Retention/recovery is not sufficient without accurate on-path
dynamics, and projection is not a universal winner. The study concerns
self-composition from ID initial conditions, not exogenous OOD evaluation or a
deployable OOD detector. Solver-relative diagnostics are offline instruments.

Current research: fixed-PCNO Kolmogorov flow under `CM_NEXT_20260906A`.
ODEs are supporting mechanism laboratories; NACA and Bump are supporting PDE
comparisons. Older M1, REALM, D-series and W26 queues do not restart implicitly.
Protected prospective/test populations remain closed.

## Sources Of Truth

| Question | Maintained document |
| --- | --- |
| Thesis and scientific scope | [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) |
| Claims and deliverables | [PROJECT_PLAN.md](PROJECT_PLAN.md) |
| Experiment targets and comparisons | [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) |
| Implementation contracts and next extraction boundaries | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| Current operational checkpoint | [HANDOFF.md](HANDOFF.md) |
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
| Active scientific implementation | `pcno_kolmogorov.py`, `kolmogorov_reference.py`, `path_conditioned_tube.py`; Kolmogorov population, fit, rollout, response, solver and paired-bank entry points and tests | Preserve the current source-bound dependency closure. |
| Direct manuscript support | Exact/Gaussian ODE study and landscape producers; NACA PCA, successor and corrective-extension pipelines | Keep code, tests, canonical results and presentation derivatives. |
| Supporting / historical reproduction | B5 Bump; older M1, Euler/CPG, shock-vortex, REALM, D-series and W26 work | Park by role, not by deleting dated filenames. Check imports and frozen manifests before physical relocation. |
| Reusable infrastructure | Runtime/rollout, metric and artifact helpers; existing reference solvers and adapters | Reuse only after checking semantics. Different hash encodings, loaders and JSON policies are not interchangeable. |
| Run-local orchestration | Ignored transport scripts, reviews, source bundles, launch/retrieval receipts | Freeze existing attempts. Extract small primitives for new attempts; do not rewrite historical receipts. |
| Disposable generated state | Python/pytest/Ruff caches and verified synthetic test outputs | Remove only after exact inventory and ownership checks; retain test XMLs and meaningful failure evidence. |

### Active Kolmogorov Dependency Boundaries

All names below are under `scripts/time_dependent_no/`, unless a utility is
identified explicitly:

- Data/reference: `generate_kolmogorov_population.py`,
  `extend_kolmogorov_training_population.py`, and the
  `screen_kolmogorov_*` qualification scripts.
- Learner: `fit_kolmogorov_clean.py`, `fit_kolmogorov_coverage.py`,
  `fit_kolmogorov_paired.py`, with utility `pcno_kolmogorov.py`.
- Diagnosis: `evaluate_kolmogorov_clean_rollout.py`,
  `evaluate_kolmogorov_coverage_rollout.py`,
  `evaluate_kolmogorov_common_response.py`,
  `screen_kolmogorov_common_solver.py`, and
  `generate_kolmogorov_paired_bank.py`.
- `fit_kolmogorov_debug.py` and `smoke_pcno_kolmogorov.py` retain qualification
  and fixture roles; do not delete them merely because their runs are complete.

Some scripts currently serve as libraries. Paired fitting imports Clean's
sampler/helpers and verifies source hashes **and function-origin paths**.
Bank generation imports common-solver helpers, which depend on earlier
readiness code. Moving these files would change the frozen implementation.

Two bounded future extractions have multiple real callers: the SSH
stdin/EOF execution primitive, and the identical Kolmogorov
dtype/shape/buffer array hash. Preserve byte-level parity with tests. Keep
scientific recipes, role access, approval schemas and complete-packet validators
run-local. Do not build a general experiment framework during the final sprint.

### Paper Producers And Tests

| Live manuscript material | Producers to keep |
| --- | --- |
| ODE table and defect landscape | `run_corrective_ode_study.py`, `plot_corrective_ode_landscapes.py` |
| Gaussian ODE appendix | `run_corrective_ode_gaussian_normal_noise.py` and its imported `run_corrective_ode_nonlinear_stress.py` parent |
| NACA low-dimensional structure | `visualize_naca0012_dataset_pca.py` and the R0 dataset/encoding helpers |
| NACA comparison and figures | `evaluate_pcno_naca0012_successor.py`, `evaluate_pcno_naca0012_corrective_extension.py`, corresponding visualizers/trainers and utilities |
| Bump supporting comparison, not yet included in active LaTeX | `run_pcno_bump_corrective.py`, `pcno_bump_corrective.py`, B5 contracts and packets |

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
Use the [handoff](HANDOFF.md) before further cleanup; never resume the original
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

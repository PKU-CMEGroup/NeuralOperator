# Time-Dependent Neural Operators

Updated: 2026-09-05

This directory is the onboarding surface for the `time-dependent-no` branch.
The current project studies corrective mechanisms for stable and accurate
self-composition from in-distribution initial conditions. It is not an online
OOD-detection project, architecture zoo, or generic repository-cleanup branch.

## Required Read Order

1. repository-root `AGENTS.md`;
2. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md);
3. [HANDOFF.md](HANDOFF.md);
4. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md);
5. this README;
6. ignored `LOCAL_CONTEXT.md` privately, if present; and
7. live Git status plus current source/artifact manifests.

Current explicit owner direction outranks repository snapshots. No document
implicitly authorizes execution.

## Active Planning

| Document | Role |
| --- | --- |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Master objective, claims, deliverables, critical path, gates, and risks. |
| [paper/PAPER_PLAN.md](../../paper/PAPER_PLAN.md) | Active manuscript narrative, section order, figures, and writing guardrails. |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Supporting ODE laboratories and the mandatory fixed-PCNO PDE case study. |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Minimal code, test, query-bank, manifest, and freeze/reveal architecture. |
| [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) | Current block and authorization status. |
| [R0_PRIMARY_PDE_READINESS_CONTRACT.md](R0_PRIMARY_PDE_READINESS_CONTRACT.md) | Frozen SU2 Unsteady NACA0012 readiness gates and state/replay contract used by the closed first screen. |
| [R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json](R0_SU2_NACA_NATIVE_REPLAY_QUALIFICATION.json) | Immutable C/D replay hashes, field roles, force-history availability, and storage projection. |
| [R0_NACA_PHASE_PILOT_CONTRACT.json](R0_NACA_PHASE_PILOT_CONTRACT.json) | Frozen solver-only stationarity, recurrence, extension, and phase-population decision rule used by the completed pilot. |
| [R0_NACA_PCNO_BASELINE_CONTRACT.json](R0_NACA_PCNO_BASELINE_CONTRACT.json) | Frozen content-disjoint phase population, 208-step horizon, complete-BDF2 PCNO representation, training, and R0 decision rule. |
| [B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md) | Frozen successor question, exact five-arm comparison, signed predictions, and protected-access gates. |
| [B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json](B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json) | Source-bound populations, learned arms, seeds, evaluation grid, packet schemas, and protected-role state. |
| [B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md](B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md) | Frozen extension question, mechanism matrix, train-only SU2 relabeling pilot, and completed open-development gates. |
| [B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json](B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json) | Frozen extension identity, schedules, paired bank, PDE-Refiner contract, and role protection. |
| [B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md](B5_BUMP_SOLVER_FREE_TRANSFER_PREREGISTRATION.md) | Active bounded Supersonic-Bump transfer/no-harm question, Stage-0 gates, four-arm pilot, and signed predictions. |
| [B5_BUMP_SOLVER_FREE_TRANSFER_CONTRACT.json](B5_BUMP_SOLVER_FREE_TRANSFER_CONTRACT.json) | B5 split, state, reducibility, phenotype, intervention, promotion, and access contract. |
| [PARKED_EXPERIMENTS.md](PARKED_EXPERIMENTS.md) | Preserved but inactive work and re-entry rules. |
| [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) | Surrounding calendar; the parent and extension are closed on development, while protected reveals remain separate decisions. |

## Scientific Sources And Evidence

- [CORRECTIVE_MECHANISMS_THEORY_AND_TAXONOMY.md](CORRECTIVE_MECHANISMS_THEORY_AND_TAXONOMY.md):
  deterministic theorem ladder and framework taxonomy; its proof line has
  completed audit;
- [paper/LITERATURE_AUDIT.md](../../paper/LITERATURE_AUDIT.md): maintained
  cutoff-bounded source ledger, refreshed and source-verified through
  2026-09-01;
- [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md):
  compact historical run-ID/topic routing;
- [M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md](M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md)
  and [tracker](M1_KOLMOGOROV_INFORMATION_COMPARISON_TRACKER.md): current M1
  reference/population provenance;
- [R0_SU2_NACA_RESOURCE_MANIFEST.json](R0_SU2_NACA_RESOURCE_MANIFEST.json):
  frozen public-resource, solver-version, mesh, restart, and claim-boundary
  contract for the closed R0 screen;
- the NACA successor preregistration and JSON contract above: immutable
  authority for the closed five-arm parent;
- the NACA corrective-extension preregistration and JSON contract above:
  frozen authority and provenance for the completed train/development work;
  neither authorizes a protected reveal; and
- the B5 bump preregistration and JSON contract above: current authority for
  local Stage-0A/B work and the bounded secondary transfer design, not a remote
  launch or historical-test opening.

Historical records and preregistrations remain tracked for reproducibility,
but they are not active planning surfaces. Use the compact tracker for bounded
claim summaries, open archived detail only when needed, and verify live
artifacts before relying on exact historical evidence. Historical
recommendations are not an active queue.

## Current Project In One Screen

Two claims:

1. clean one-step data do not identify deployment-relevant off-trace response
   over a sufficiently rich equivalence class; and
2. frozen response/forcing/drift diagnostics should prospectively predict
   rollout ranking reversals and intervention value beyond clean error.

The explanatory split is on-reference fidelity plus transverse
retention/recovery. Clean one-step error probes the first; corrective mechanisms
target the second. Neither axis alone is sufficient, and the familiar
tangent-dynamics interpretation of clean training is not the novelty claim.

Evidence hierarchy:

1. theory establishes the limitation and framework;
2. exact and learned ODEs provide supporting mechanism and measurement checks;
3. a mandatory, end-to-end fixed-PCNO case study on one qualified complex PDE
   compares representative embedded and operational corrective mechanisms;
4. frozen diagnostics and predicted rankings remain reserved for a separately
   authorized prospective long-horizon reveal in that case study; and
5. the selected Supersonic-Bump B5 study is secondary and bounded.

The active manuscript follows eight sections: introduction; one-step
versus rollout; corrective-mechanism framework; methods and prospective
predictions; framework-guided corrector design; compact ODE calibration;
primary fixed-PCNO PDE case study; and limitations/conclusion. The ODE
calibration provides verified supporting evidence. The first NACA R0 model
screen remains a verified negative promotion result. The distinct five-arm
NACA successor supplies immutable parent evidence. Its completed
`B3B4_NACA_CM_EXT_20260902A` extension adds source-faithful and curriculum
model-prefix exposure, a paired recovery--relabeling contrast, and a learned
iterative PCNO corrector. The open-development result is verified; prospective
and sealed evidence remain unopened.

PCNO is fixed only to remove an empirical architecture confound; the theory is
map-agnostic. Recovery and dynamics relabeling are neutral competitors with
different targets. Offline drift scores are diagnostics, never deployed OOD
detectors. The literature taxonomy is cutoff-bounded and comprehensive across
declared method families; the empirical implementations are representative
rather than exhaustive.

SU2 Unsteady NACA0012 completed the first R0 screen and is rejected as the
primary correction-necessity case under that frozen identity. Its pinned
solver/resource/phase evidence, complete-BDF2 population, dataset,
full-resolution smoke, three-seed PCNO
training, and development-only evaluation all have verified packets. Clean
next-state relative L2 is about `0.12%`; all three rollouts are finite and
severely wrong late, but all fail the frozen early-accuracy threshold. The
exact verdict is `R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE`. Under R0,
NACA may be used only as supplementary one-step/rollout-discrepancy and
structure evidence; it does not identify the transverse cause or authorize a
corrector study.

The separate `B3B4_NACA_CM_20260901A` successor now uses NACA as the primary
fixed-PCNO case for a narrower question. Its exact deployed comparison is
`CLEAN`, `IID_RECOVERY`, `ERROR_SUBSPACE_RECOVERY`,
`DETACHED_PUSHFORWARD`, and `PATH_PROJECTION`. All twelve learned-arm/seed
training packets and the development-only result packet are locally verified.
The final-manifest SHA256 is
`09e3ae7e5896039e2230073c1ead5e5f9808c0cc9b47d0b15ff337b26b3e0ed1`.
The immutable replay `development_replay_v2_81646368` and historical v2
visualization remain valid provenance. The final presentation packet
`development_visualization_v3_e2eccdbf`, portable `prospective_freeze`, and
then-current 35-page parent manuscript are closed with all protected-access flags false;
static figures use verified development results and the three seed animations
are qualitative only. Windows and exact mounted-checkout WSL focused tests,
Ruff, and local packet/manuscript checks pass. Prospective and sealed roles
remain unopened and require separate owner authorization. Exact artifact
hashes and the claim-bounded outcome are in
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md). Other PDEs remain conditional
controls or fallbacks and do not activate automatically.

The completed extension preserves the five-arm parent byte-for-byte. Its exact
learned mechanisms are `MP_PDE_PUSHFORWARD_M01`,
`CURRICULUM_EMA_PUSHFORWARD_K13`, `PAIRED_RECOVERY`, `DYNAMICS_RELABEL`,
and `PCNO_PDEREFINER_K3_VPRED`, with `CLEAN_EMA` as the EMA control and the
closed parent arms as evidence. The corrected pilot, 476-input bank,
full-resolution smokes, all 18 training packets, and common development
evaluation passed their audits. Attempt E closes under final-manifest SHA256
`15475da78b4182754fe959eadc175434ffca18ef1558a45ea06cbf9c93ca7370`.
Path projection is the strongest operational reference, PDE-Refiner the
strongest learned corrector, and paired recovery the safest one-call learned
arm; neither new pushforward protocol is robust across seeds. The result does
not validate trained-model fidelity to trusted displaced SU2 response.
Prospective and sealed roles require separate owner decisions. SU2 labels are
offline training/diagnostic information only, never an online solver or defect
trigger.

The extension presentation derivative
`naca_corrective_extension_visualization_results_20260903b` is parented to the
verified remote Attempt-E visualization receipt and exact replay. Its static
panels use exact evaluator snapshots; its three seed animations are independent
qualitative rerolls. The correction changes only color normalization and
layout, not the scientific replay or evaluation. Exact identities are in
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md).

The selected B5 study tests transfer beyond NACA's special nearly linear,
low-rank path geometry. It does not treat NACA's projection ranking as
universal: the bump uses distinct native meshes, so raw global PCA is not a
legitimate deployable correction without a new remapping representation. B5
Stage-0A/B local identity/reducibility and synthetic implementation are
authorized; Stage 0C, its four Stage-1 systems, and the conditional PDE-Refiner
port remain behind the amendments in the source-of-truth preregistration. No
remote resource or dataset-scale launch is selected, and the historical test
remains sealed.

## Current M1 Readiness

The spatial and temporal M1 packets independently rehash; together they qualify
the registered finite-grid N256 reference on the frozen trajectories. The fresh
population is not qualified:

- R2-POP launch 0 ended at the eight-hour cap before a packet; and
- R1 exited with confirmed code `1` after `22307.064 s` and produced no result,
  artifact manifest, series, or output directory.

R1 has no recorded infrastructure classification and no scientific conclusion.
Q2 is closed, no retry is implicit, and the old M1 model plan fixed residual
FNO rather than the current PCNO backbone. Any future PCNO study needs a new
model/source identity and explicit approval.

## Maintained-Code Navigation

Obtain the live inventory from the checkout:

```powershell
rg --files utility/time_dependent_no
rg --files scripts/time_dependent_no
rg --files tests/time_dependent_no
```

Presence means maintained or recoverable implementation, not current selection
or authorization.

### Reusable utilities

`utility/time_dependent_no/` contains branch-local support for:

- errors, losses, metrics, rollout, and artifact/source contracts;
- 1D and Euler reference/data utilities;
- PCNO residual/state/boundary adapters;
- Kolmogorov reference stepping;
- strict SU2 NACA resource, mesh, and native-restart interface auditing plus
  isolated replay preparation, pinned native execution/evaluation, two-run
  comparison, multi-step trajectory generation, and solver-only phase
  analysis, together with complete-BDF2 PCNO dataset/training/evaluation
  support for the closed R0 screen, plus the closed successor's recovery,
  detached-pushforward, train-path projection, corrected recurrence, and
  source/packet validation in `pcno_naca0012_successor.py`, plus the completed
  extension's EMA/pushforward/refiner contracts in
  `pcno_naca0012_corrective_extension.py` and fixed-SU2 displaced-restart
  interface in `su2_naca0012_relabel.py`;
- path-conditioned, response, structure, resolution, and correction
  diagnostics;
- dynamic finite-volume geometry and physical metrics; and
- historical REALM adapters and benchmark-specific metrics, which are not an
  active application route.

Keep new reusable code here until it has multiple real callers or is stable
enough for a core API. Do not change unrelated `pcno/`, `baselines/`, or example
surfaces for this project.

### Entry points

`scripts/time_dependent_no/` contains historical and maintained training,
evaluation, analysis, visualization, reference-generation, and provenance
entry points. Review the exact source, data, output, population, and
authorization contract before executing any script. Old W26/D-series scripts
remain for reproducibility even when their forward work is parked.

The closed five-arm successor entry points are
`calibrate_pcno_naca0012_successor.py`,
`train_pcno_naca0012_successor.py`,
`evaluate_pcno_naca0012_successor.py`, and
`visualize_pcno_naca0012_successor.py`. The evaluator implements three distinct
modes: open-development evaluation, a data-free prospective prediction freeze,
and authorization-first prospective reveal. The reveal requires the exact
source-bound freeze and authorization schema and consumes a one-shot access
receipt before the first protected scientific read. No sealed reveal mode is
implemented.

The corrective-extension entry points are
`run_su2_naca0012_relabel_pilot.py`,
`generate_su2_naca0012_paired_bank.py`,
`train_pcno_naca0012_corrective_extension.py`,
`evaluate_pcno_naca0012_corrective_extension.py`, and
`visualize_pcno_naca0012_corrective_extension.py`. All five have completed
their registered train/development or presentation roles. The visualizer derives exact static
panels from evaluator snapshots and labels independent recurrent animations as
qualitative rather than quantitative reproduction.

The ADER generator remains fail-closed and requires an explicit reviewed run
flag. Do not invoke historical scripts based on filename alone.

### Tests

`tests/time_dependent_no/` contains synthetic and small CPU fixtures for the
same categories. Run the narrowest relevant test first. A passing synthetic
test is plumbing evidence, not authorization for a scientific array,
checkpoint, solver, GPU, or remote run.

## Data And Physical-Semantics Boundaries

### SU2 Unsteady NACA0012: closed parents and completed development extension

The Stage-0 audit binds the public license, configuration, fixed mesh, and
restart histories 497/498 with target 499. It validates the native binary
layout, coordinate row order, and evolved state
`[Density, Momentum_x, Momentum_y, Energy, Nu_Tilde]`. These checks establish
interface completeness only. The independently audited native runner keeps
target 499 external, and the final C/D pair establishes pinned executable
compatibility, a small observed evolved-state replay error, bitwise repeat
determinism, and local throughput. The native output has two extra derived
velocity fields that are excluded by exact name from canonical metrics but
remain covered by whole-file hashes. Original producer-integrated coefficients
are unavailable from the canonical target; the qualification addendum retains
the replay's pressure-plus-skin-friction proxy, while the baseline contract
defines a distinct five-state pressure-only proxy. Auxiliary wall discrepancies
remain diagnostics. The completed solver-only
phase analysis supports a single-attractor population freeze: period `34.6543`
steps and the uniform-node plus near-body/wake recurrence views pass, while the
required physical-area secondary view fails and remains disclosed. The
32-anchor population, content-disjoint role blocks, 208-step horizon,
complete-BDF2 baseline, metrics, and access rules passed independent audit at
SHA256 `94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87`.
The bounded dataset, full-grid smoke, three-seed training, evaluation, and
result archive subsequently completed and passed independent packet/raw-metric
audit. Every seed fails the early `<0.15` train-state-scale gate despite severe
late degradation, so NACA is not the primary hero case. The native replay
quarter-margin also fails uniform-node and near-body/wake views. Do not extend
the horizon or implement primary-case correctors under this closed identity;
prospective/sealed values remain unopened.

The separate five-arm successor does not change this verdict. It freezes four
learned arms--`CLEAN`, `IID_RECOVERY`, `ERROR_SUBSPACE_RECOVERY`, and
`DETACHED_PUSHFORWARD`--at seeds `17/29/43`, plus the operational
`PATH_PROJECTION` deployment on each clean predictor. All twelve learned
training packets, the calibration packet, authorizations, resource smokes, and
the development evaluation are locally verified. Every evaluated rollout is
finite. `PATH_PROJECTION` improves both primary metrics on all 24 paired
anchor-seed comparisons and improves pressure and graph-Dirichlet diagnostics;
its zero path residual is construction-only. `IID_RECOVERY` helps,
`ERROR_SUBSPACE_RECOVERY` improves `CLEAN` but not IID recovery, and the
registered `DETACHED_PUSHFORWARD` mechanism is falsified with seed-unstable
rollout effects. These findings make no physical off-manifold or arbitrary
displaced-state SU2 claim. Prospective and sealed values remain unopened.

That five-arm result is parent evidence for `B3B4_NACA_CM_EXT_20260902A`.
Under its separate preregistration and JSON contract, the extension completed
implementation, the corrected pilot, paired bank, full-resolution smokes, 18
training packets, and the common development evaluation. The verified outcome
and limitations are centralized in
[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md). It closes the planned
open-development comparison, not the prospective C2 claim.

### Supersonic bump

The graph bundle does not expose a validated inverse to the original DG state
or audited finite-volume faces/volumes. Bump integral summaries are diagnostic
proxies, not physical conservation or arbitrary-state solver restart. The
active `B5_BUMP_SOLVER_FREE_TRANSFER_20260905A` therefore uses stored clean
trajectories only and makes no trusted displaced-state-dynamics claim.
Per-trajectory POD is a train-only reducibility diagnostic, not a deployed
projector. Its local Stage-0A/B scope is authorized under the linked B5 contract;
remote execution remains unselected and the historical test is sealed.

### Dynamic shock-vortex

The dynamic finite-volume family has audited geometry and boundary accounting,
but its native coarse transition failed the registered factor-of-four solver-
bias qualification as a proxy for the fine-evolve/restrict target. Do not call
it a trusted displaced-state target under that registration.

### Kolmogorov inactive fallback

The finite-grid reference has qualified spatial/temporal parent packets. Fresh
population qualification, PCNO model mesh/resource closure, and a new fixed-
PCNO representative case-study contract remain missing.

### Shock-vortex control candidate, not a primary selection

The data/PCNO path is mature, but the fine solver does not expose a qualified
arbitrary-state restart in the PCNO state representation. The native coarse
alternative failed its prior bias gate, and the current PCNO wrapper/trainer do
not match the retained D074 source manifest. Its current planning role is a
possible tangential-dynamics/no-harm control. A restart audit is not on the
immediate path without a new owner-selected role and readiness contract.

## Experiment And Artifact Discipline

- Bind source, data/split/population, model/checkpoint, normalizer/feature map,
  evaluator, result, and final artifact hashes separately.
- A changed contract receives a new identity.
- Keep failed/incomplete attempts and their raw receipts.
- Use open development populations; sealed/test access requires a named
  decision.
- Use synthetic CPU fixtures before dataset-scale or GPU work.
- Before any experiment, audit that the implementation matches the owner's
  intended mathematical target, information source, deployment composition,
  recurrent feedback state, and identity/no-correction ablation. Ask for
  clarification if a scientific instruction is ambiguous.
- Do not commit raw data, checkpoints, rollout arrays, large logs, credentials,
  private hosts, or local machine paths.
- Packet integrity is distinct from current-checkout compatibility.
- Artifact deletion requires a refreshed inventory and reference check.
- Documentation edits may invalidate legacy provenance hashes without
  invalidating archived result packets. Do not rewrite old identities.

## Privacy And External Review

Private paths, host details, credentials, datasets, and machine-specific
locations belong only in ignored local context. The unpublished manuscript and
private mentor material must not be sent to Gemini or another external tool
without the owner's explicit approval of the exact prompt/scope.

## Recovery

Tracked local history files remain available for bounded provenance lookup but
are not active documentation. Verify their hashes and scope before relying on
them. Large generated artifacts remain under ignored storage and depend on
their own manifests rather than this README or Git history.

The exact pre-refinement 2026-08-29 active plans are preserved under
`archive/plans_20260829_pre_refinement/`. They are historical snapshots and do
not enter the active read order.

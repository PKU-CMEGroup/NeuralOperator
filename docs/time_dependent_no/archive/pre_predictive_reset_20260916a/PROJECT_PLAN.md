# Project Plan: Mechanism Identification And Paper Completion

Updated: 2026-09-07

Owner target: complete the paper by the end of next week. Working deadline:
**Sunday September 13**, with a **complete manuscript Friday September 11**
for review. This is a delivery target, not a promise of positive findings.
Pre-revision tracked plans remain recoverable at Git
`624c9c0d2b611172ccb6c5afcf1c06af2c6400ec`; closed contracts and artifacts
are unchanged.

## Objective And Evidence

Explain the clean-trace/displaced-response gap and demonstrate a practical
sequence: **measure -> predict -> intervene -> verify**, with PCNO fixed.

| Claim | Minimum evidence | Boundary |
| --- | --- | --- |
| C1: conditional response non-identifiability | Audited theory, exact examples, separate tube and path requirements. | A rich hypothesis class is required; not every model fails away from data. |
| C2: prospective diagnostic value | Qualified common-input solver assays, frozen predictions, fresh ID confirmation and improvement beyond clean error alone. | Retrospective explanation and training-bank fit do not establish C2. |

ODE work is complete supporting calibration. NACA is a completed favorable
low-rank path example; Bump is a completed transfer/no-harm contrast. Neither
alone resolves the mentor's representativeness concern. Preserve their
results and limitations as supporting cases, not an assumed final centerpiece.
Exact outcomes/hashes live in [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md).

The NACA identical-input trained recovery/relabel assay remains missing.
Its existing bank can support retrospective calibration, not a prospective
claim. Do not reopen ODE training or expand the old Bump comparison.

## New Primary-PDE Route

Selected route, approved September 6: periodic 2D forced Navier--Stokes/Kolmogorov
flow with the retained dedicated solver and existing AutoDL instance. Establish
state, solver and measurement readiness; do not assume chaos, PCNO blowup or a
tiny linear PCA rank.

SU2 availability removes installation work, not a new case's mesh, evolved
state, restart or convergence obligations. Square-cylinder wake flow is the
parked SU2 alternative; do not launch two full PDE campaigns.
Close the selected case's readiness decision by September 7.

The old M1 population failure is infrastructure evidence, not a PDE rejection.
Only its verified solver may be reused under a new PCNO study identity.
A small new local solver screen is in scope now; old protected data stay closed.

## Required And Conditional Work

1. **Readiness and baseline:** complete restartable state, clean/displaced
   numerical checks, independent trajectories, measured cost and competent PCNO.
2. **Mechanism comparison:** paired recovery/relabeling, curriculum exposure,
   empirical projection and learned refinement, with clean/EMA controls.
3. **Prospective comparison:** freeze diagnostics and predictions before fresh
   ID confirmation; report tails, physical diagnostics and cost.
4. **Conditional design:** one dynamics-buffer/recovery hybrid with deletion
   controls, only if response evidence and implementation readiness support it
   by September 9. Admit it then only if implementation is already audited and
   measured runtime fits the September 10 result freeze. A second regime is
   also conditional, not a second campaign.

Architecture recommendations remain hypotheses unless supported by a controlled
ablation. No universal ranking, online OOD detector, exogenous OOD benchmark,
architecture zoo, REALM revival or unqualified solver label.

## Milestones

| Date | Required output / decision |
| --- | --- |
| Sep 6 | Current plans; common-response CPU tests; independently reviewed local solver pilot. |
| Sep 7 | Select one PDE; close state/solver/population contract; tiny fit and measured PCNO resource smoke; freeze affordable matrix. |
| Sep 8 | Core paired training underway; methods and diagnostic text drafted in parallel. |
| Sep 9 | Common-bank assays and trained-model prediction freeze; hybrid/second-regime go/no-go. |
| Sep 10 | Permitted fresh confirmation, paired statistics, failure analysis and scientific-content freeze. |
| Sep 11 | Complete serious manuscript, main figures and appendices. |
| Sep 12--13 | Independent numerical/claim audit, revisions, compiled and visually checked PDF. |

The [weekly calendar](WEEKLY_RESEARCH_PLAN.md) owns delivery timing; the
[experiment plan](EXPERIMENT_PLAN.md) owns scientific choices and the
[implementation plan](IMPLEMENTATION_PLAN.md) owns code/test responsibilities.

## Budget And Stop Rules

Core target: one primary regime, three paired seeds, existing PCNO backbone.
The provisional **72 GPU-hour core planning envelope** is a cap to test, not
observed runtime, a new rental authorization or a guarantee of completion.
Measure seconds/update, evaluation cost, solver seconds/label and memory at
the smoke; reduce optional work first if the matrix does not fit. Reserve the
final two days for writing and audit.

If no candidate is ready by September 7, report the blocker and revise scope
with the owner. Do not repeatedly search for a desired winner. If C2 fails,
report it; if the new application remains unfinished, deliver a progress
manuscript without calling the scientific application complete.

September 7 progress: the shared response assay, N64/N128 and N128/N256 solver
screens, periodic PCNO adapter and synthetic CPU/GPU smokes are complete;
the longer-time development packet now passes all five declared numerical
checks at sparse anchors. The fresh eight-train/four-development N128 packet
completed but failed both spatial limits in the early transient; native
N256/N512 stress refinement passed independent full-array audit. The full N256
population B closed incomplete after local standby; its partial packet is
audited and preserved. Recovery C launched on existing AutoDL after explicit
payload approval and hash verification. All twelve trajectories completed and
were retrieved; independent full-array audit passes the unchanged sampled
clean numerical contract.
The approved AutoDL debug fit completed
and passed independent prediction/checkpoint audit; preservation-first cleanup
is complete.
Clean-model competence and actual displaced-input qualification remain open; the tracker
owns live execution status, checks and hashes. Leave the other agent's resource untouched.

## Authorization And Preservation

The owner approved the dedicated solver and existing AutoDL instance on
September 6. Use exact source/input/command/output receipts and numerical/code
checks before training; no additional instance purchase is implied. Existing
protected populations stay closed. A prospective/sealed
reveal needs the exact freeze and named owner decision. Planning alone does
not override these boundaries.

Keep old attempts immutable, preserve unrelated work, and keep private context,
datasets, checkpoints and the unpublished paper out of Git publication.

# Codex Instructions For The time-dependent-no Branch

This branch is for the summer 2026 time-dependent neural-operator project in the PKU-CME NeuralOperator codebase.

## Scope

The current project is an insight paper on corrective mechanisms for long-horizon neural-operator prediction, with PCNO/Kolmogorov flow as the main empirical study. Earlier Euler, aerodynamic, and other studies remain historical evidence. The current phase is manuscript consolidation; `docs/time_dependent_no/HANDOFF.md` identifies the next authorized task.

Do not treat this as a generic cleanup branch. Keep unrelated NeuralOperator examples and core APIs unchanged unless the current task explicitly requires touching them.

## Context Loading

At the start of a coding session in this branch:

1. Read this file.
2. Read `docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md`.
3. Read `docs/time_dependent_no/HANDOFF.md`.
4. Read the compact `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md`
   only when the task needs experiment history or diagnostic routing.
5. Read `docs/time_dependent_no/README.md` for onboarding and maintained-code
   navigation.
6. When exact historical evidence is needed, search by run ID and read only the
   matching bounded section of
   `docs/time_dependent_no/history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md`.
   Do not load the full archive into routine context.
7. If `LOCAL_CONTEXT.md` exists, read it. It is private local context and must
   not be committed or quoted.
8. Inspect `git status --short --branch` before editing.

## Documentation Governance

- Current explicit human direction takes precedence over repository planning
  snapshots. Reconcile the documents afterward when the direction changes.
- Historical results, gates, and Codex recommendations are evidence, not
  permanent experiment authorization or prohibition.
- A failed gate closes the exact registered attempt. It does not reject an
  entire method family unless the evidence supports that claim and the current
  human direction adopts it.
- Standing privacy, sealed-population, destructive-action, and claim-validity
  boundaries persist until explicitly changed by the human owner.
- Preserve exact historical language in the archive. Keep active README,
  handoff, decision, and experiment-index files compact and current.
- Keep the private `paper/` directory ignored and uncommitted. Do not send
  unpublished material to external AI tools without approval of the exact scope.

## Implementation Discipline

- Keep reusable code under `utility/time_dependent_no/` until it is stable enough to promote into core `pcno/`, `baselines/`, or `utility/` APIs.
- Keep experiment entry points under `scripts/time_dependent_no/`.
- Keep tests under `tests/time_dependent_no/`.
- Do not commit raw datasets, checkpoints, generated rollout arrays, large logs, credentials, private hostnames, or local machine paths.
- Use synthetic fixtures and CPU tests before any authorized dataset-scale run.
- For numerical work, report rollout errors and physical diagnostics appropriate
  to the PDE, including conservation, positivity, shocks, or boundary leakage where applicable.

## Infrastructure Rule

Use the execution resource explicitly selected by the owner. The current
selection is the personal workstation described in ignored `LOCAL_CONTEXT.md`;
retained AutoDL workflows do not authorize a run there.

Committed docs may contain templates and placeholder commands. Machine-specific paths, SSH details, credentials, and dataset locations belong in `LOCAL_CONTEXT.md` or `docs/time_dependent_no/PRIVATE_*.md`, which are ignored by git.

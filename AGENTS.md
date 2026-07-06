# Codex Instructions For The time-dependent-no Branch

This branch is for the summer 2026 time-dependent neural-operator project in the PKU-CME NeuralOperator codebase.

## Scope

The branch owns hard time-dependent PDE work: CPG-style 2D Euler benchmark reproduction, FNO/PCNO/MPCNO failure analysis, structure-aware neural-operator methods, and data-assimilation experiments after open-loop failure modes are diagnosed.

Do not treat this as a generic cleanup branch. Keep unrelated NeuralOperator examples and core APIs unchanged unless the current task explicitly requires touching them.

## Context Loading

At the start of a coding session in this branch:

1. Read this file.
2. Read `docs/time_dependent_no/README.md`.
3. Read `docs/time_dependent_no/HANDOFF.md`.
4. Read `docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md` when the task concerns experiment history or current diagnostic status.
5. If `LOCAL_CONTEXT.md` exists, read it. It is private local context and must not be committed or quoted.
6. Inspect `git status --short --branch` before editing.

## Implementation Discipline

- Keep reusable code under `utility/time_dependent_no/` until it is stable enough to promote into core `pcno/`, `baselines/`, or `utility/` APIs.
- Keep experiment entry points under `scripts/time_dependent_no/`.
- Keep tests under `tests/time_dependent_no/`.
- Do not commit raw datasets, checkpoints, generated rollout arrays, large logs, credentials, private hostnames, or local machine paths.
- Use synthetic fixtures and CPU tests before launching dataset-scale AutoDL runs.
- Report both paper-compatible rollout errors and structure diagnostics: shock, conservation, positivity, and boundary leakage.

## Infrastructure Rule

AutoDL is the active GPU environment for this project.

Committed docs may contain templates and placeholder commands. Machine-specific paths, SSH details, credentials, and dataset locations belong in `LOCAL_CONTEXT.md` or `docs/time_dependent_no/PRIVATE_*.md`, which are ignored by git.

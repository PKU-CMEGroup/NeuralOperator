# Implementation Boundaries and Retained Code

Updated: 2026-09-28

## Current phase

The current task is manuscript consolidation from existing evidence.
The abstract and Sections 1--3 are locked; Section 4 is next under the
[paper plan](../../paper/PAPER_PLAN.md). No new scientific implementation,
experiment, evaluation population or theory programme is selected before
the mentor discussion.

Existing access to the owner-selected workstation persists, but is not an
instruction to run a job. The [project plan](PROJECT_PLAN.md) owns scope;
[HANDOFF.md](HANDOFF.md) owns the checkpoint.

## Preserve and reuse existing implementation

[README.md](README.md) owns the code map. [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
routes completed recipes; [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) owns
their dated results and checks. Historical test counts are not fresh validation.

- Preserve the complete deployed flow map, checkpoint loading, normalization,
  shared numerical restrictions and stochastic-input semantics.
- Keep input/target pairing, temporal gradient paths, sampling roles and
  failed/guarded outcomes explicit when interpreting a comparison.
- Preserve source/data/checkpoint/solver bindings and original attempt identities.
  Changed scientific recipes require new identities.
- Some entry scripts serve as libraries; frozen checks include dependency hashes
  and function-origin paths. Do not relocate or refactor them for cosmetic cleanup.
- Keep ODE, NACA and Bump producers, tests, canonical evidence and presentation
  derivatives. Absence from the current manuscript is not a deletion criterion.

## Verification for an authorized change

Use the narrowest meaningful check for the changed behavior. For manuscript
edits, inspect the changed source and relevant compiled layout. For a later
scientific code change, use focused synthetic CPU tests before a separately
selected real-data check. Broaden verification only for a concrete dependency
risk; distinguish newly executed checks from historical receipts.

Keep reusable branch code under utility/time_dependent_no/, entry points under
scripts/time_dependent_no/, and tests under tests/time_dependent_no/. Reuse
existing conventions and helpers when their scientific semantics match.
A new script needs a current purpose and invocation; no generic experiment
framework, broad refactor, extra review loop or migration is required.

## Access and historical recovery

Solvers and ground truth remain offline instruments. Protected populations,
private context, unpublished manuscript and external-review material retain
their standing boundaries. Preserve other work, recovery copies and immutable
packets. Cleanup and publication require their own applicable authorization.

The preceding implementation descriptions are retained at Git commit 669f7fc.
They describe completed attempts rather than a current execution queue.

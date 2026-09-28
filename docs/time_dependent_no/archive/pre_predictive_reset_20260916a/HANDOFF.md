# Handoff: Pre-Sprint Cleanup

Updated: 2026-09-13

## Current Owner Direction

Pause experiments for preservation-first local and AutoDL cleanup before the
final sprint. Another agent is using AutoDL: protect its processes, files and
shared caches. A possible move to zw-gpu is not a compute migration or launch
instruction. Do not resume the old experiment queue automatically.

The current scientific contract remains
[RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) and
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The [README](README.md) now owns the
code/paper/retention map. [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) retains
dated results and exact attempt history.

## Scientific Checkpoint

- Active line: `CM_NEXT_20260906A`, fixed-PCNO Kolmogorov flow.
- Clean8/Clean32, matched rollout, common-response and qualified paired-bank
  evidence are retained. Their scope and numerical qualifications remain as
  recorded in the tracker; this cleanup did not recompute scientific results.
- The paired clean-continuation/recovery/dynamics implementation and train-only
  validation are prepared. The September 12 resource-only payload has no local
  launch receipt. No resource probe or adaptation is started by this cleanup.
- ODEs and NACA supply current manuscript results. Bump is retained supporting
  evidence but has no section/figure/table in the active LaTeX yet.
- The thesis remains two-axis: on-reference accuracy does not identify
  displaced-input response; retention/recovery alone does not ensure accuracy.
  Prospective predictive claims are not established by a retrospective fit.
- Sealed/prospective populations stay closed. No online OOD detector, online
  solver correction or exogenous OOD evaluation is added.

## Local Inspection Findings

The current branch is `time-dependent-no`, with existing uncommitted planning,
path-conditioned diagnostic and Kolmogorov changes. Preserve them. The source
baseline in the cleanup packet covers all 469 branch-local Python files.

1. **No scientific source deletion is justified by age or filename.**
   Current trainers import older entry-point helpers, and some check their
   exact source hashes and origin paths. Historical tests also cover shared
   current behavior.
2. **Infrastructure extraction should be small and prospective.**
   SSH stdin/EOF execution and the identical Kolmogorov array hash have multiple
   callers. A future extraction needs parity tests and a new source identity;
   do not rewrite frozen attempts or unify unlike JSON/hash semantics.
3. **Generated scratch is the disposable class.**
   Keep XMLs and diagnostic failure records. Access-denied scratch is not empty
   and is not deleted without an inventory.
4. **Presentation versions are not all redundant.**
   The NACA roughness figure matches the September 3b packet, whereas the H208
   comparison matches September 4a. Keep both.

Current figure checks also found a missing ODE presentation bridge: the paper's
`ode_defect_landscapes.pdf` has SHA prefix `0204561a`, while the canonical
landscape PDF matches its manifest at `d240abf4`. The canonical runner and
plotter still match their recorded hashes. No derivative receipt was located
in the bounded local search. Preserve both PDFs; reconcile their provenance
before the final paper freeze. This is not evidence of numerical disagreement.

## AutoDL Cleanup: Selected Subset Complete

Successor `autodl_cleanup_20260913e/` completed on September 13. The separate
read-only inventory at 16:37 China time confirms the exact selected removals.

- Removed 79 files: 77 intermediate checkpoints and two duplicates. Their
  local recovery copies were freshly hash-verified in both preflight and
  deletion; all copies remain retained.
- Removed allocation: 18,700,922,880 bytes. Observed net work-volume recovery:
  18,700,894,208 bytes (17.42 GiB), measured on the live shared filesystem.
- Work-volume free space: 19,662,733,312 bytes (18.31 GiB).
- All 3,202 retained file signatures and six retained links are unchanged;
  26 final checkpoint sentinels remain protected.
- The same other-agent GPU job was present before, after and at reconciliation.
  No training process was stopped or modified.

E reused D's native local-checksum implementation and B's exact existing-job
admission policy. The remote code, per-file identity checks, scoped process
conflict checks, low-priority I/O and durable journal stayed unchanged.
Independent source/plan review, 147 synthetic tests, native interop and fresh
preflight passed. The completion receipt and 160-event journal record all 79
ordered intent/removal pairs.

The ignored E packet's `CLOSEOUT.json` binds the evidence; `plan.json` maps
every removed remote file to its retained local recovery copy. Keep the whole
packet and its A/B/C/D dependencies. B and C remain failed attempts with zero
deletions; D was never launched. Do not retry any completed or failed identity,
or resume the original interrupted worker against its now-stale full plan.

The 149 incomplete/missing checkpoint backups and the local partial copy were
not cleanup candidates. Final checkpoints, datasets, shared caches, the other
agent's tree and the July checkpoint recovery archive were also left intact.
The current transport still depends on older Bump helpers; retain that chain.

The separate system partition remains tight: 566,915,072 bytes (541 MiB) free
at reconciliation. Work-volume cleanup did not resolve that constraint.
No experiment, resource probe, zw-gpu access, commit or paper upload occurred.

## Cleanup Record And Next Steps

The pre-cleanup README and handoff were archived exactly under
`docs/time_dependent_no/archive/pre_sprint_20260913a/`. Their SHA256 values are:

- README: `bf3ddabf5ecde0d024b414763a16b5b4c95a3751db466366bc6b59eeed2a93a2`;
- HANDOFF: `7ea39e70f5197d849e18e79e8c6736c1164a8249afd7b4f920c9e80db180f584`.

The ignored `local_cleanup_20260913a/` packet under the artifact root owns
the before-inventory, source baseline, exact cache plan and cleanup receipt.
It records exclusions and recoverability; do not treat it as a blanket deletion
allowlist. The README maps retained code and evidence.

Completed hygiene: 87 cache files removed after verifying every recovery-ZIP
member; one generated transport-test tree (2,867 files, 517 directories and
22 links) archived by same-volume rename. Its before/after tree hash matches.
The synthetic tree can be restored to its original path; its absolute Linux
links are intentionally not rewritten. Test XMLs and local scientific evidence
remain in place. No numerical experiments or broad test suite were run.

Next, inspect this compact handoff and code map with the owner. System-partition
headroom and the ODE figure binding still need separate reconciliation before
the corresponding runtime/paper freeze. Any infrastructure refactor should preserve
the old replay closure and migrate new work only. No new experiment, broad source
relocation, commit or paper upload is implicit in the audit.

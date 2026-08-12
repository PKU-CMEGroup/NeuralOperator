# Time-Dependent Neural Operators

Updated: 2026-08-12

This directory is the onboarding surface for the summer 2026 time-dependent
neural-operator work on branch `time-dependent-no`. It points to authority,
data contracts, maintained code, and recovery records. It is not an experiment
queue and does not replace current human direction.

## Navigation

Read active context in this order:

1. Repository-root `AGENTS.md` for branch rules.
2. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current scientific verdict, claim boundaries, and owner constraints.
3. [HANDOFF.md](HANDOFF.md) for the current workspace and unresolved decisions.
4. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   compact experiment-ID and topic routing.
5. [WEEKLY_RESEARCH_PLAN.md](WEEKLY_RESEARCH_PLAN.md) for the five coordinated
   research lines and prospective gates.
6. [W26_L4_REALM_BENCHMARK_PLAN.md](W26_L4_REALM_BENCHMARK_PLAN.md) for the
   current REALM source/data audit, staged benchmark contract, and next
   authorization boundary.
7. [CODEX_KICKSTART_PROMPTS.md](CODEX_KICKSTART_PROMPTS.md) for read-only A0
   kickoff prompts.
8. Read one bounded section of the
   [2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
   or [evidence archive](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
   when exact historical contracts, results, or hashes are required.

The [archive guide](history/README.md) explains snapshot authority and link
resolution. The frozen D072 [experiment plan](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_PLAN.md)
and [execution tracker](history/d072_refine_logs_frozen_2026-08-03/EXPERIMENT_TRACKER.md)
are preregistration history, not a live queue.

Private paths, host details, credentials, and machine-specific dataset locations
belong in ignored `LOCAL_CONTEXT.md`. Generated reports, arrays, figures,
checkpoints, and large logs belong under ignored `artifacts/time_dependent_no/`.

## Data And Provenance

### Supersonic bump

The folder historically labeled `forward_300` is the CPG supersonic-bump
bundle: 300 training trajectories, 20 test trajectories, 80 HDF5 frames, and
roughly 19k--23k graph nodes per trajectory.

- [CPG_EULER_DATASET_CONTRACT.md](CPG_EULER_DATASET_CONTRACT.md) defines the
  live HDF5 schema and reader convention.
- [BUMP_300_DATASET_AUDIT.md](BUMP_300_DATASET_AUDIT.md) records raw-bundle and
  solver-lineage evidence.
- [CPGGNSPDES_REFERENCE_AUDIT.md](CPGGNSPDES_REFERENCE_AUDIT.md) records the
  bounded public-reference audit.

The HDF5 bundle does not expose validated finite-volume control volumes,
oriented physical faces, measures, normals, or accepted-substep impulses.
Equal-node and reconstructed-weight totals are diagnostic proxies. They do not
establish physical conservation.

### Dynamic shock-vortex

The separate Mach-1.1 shock-vortex family has audited finite-volume geometry,
boundary accounting, and cumulative accepted-substep impulses. Physical
conservation diagnostics are meaningful only on that frozen solver/data
contract. State-residual PCNO recurrence is not conservative by construction.

Its family-local cell types are `0=interior`, `1=touches y-symmetry`,
`2=touches x-extrapolation`, and `3=touches both`; the combined corner type
takes precedence. These meanings must not be pooled with bump vertex labels.

Common-source resolution comparisons use regenerated geometry and conservative
restriction from the same evolved reference. Image resizing or unrelated
coarse solves do not satisfy that contract.

## Compact Evidence-Family Summary

| Family | Maintained conclusion | Claim limit |
| --- | --- | --- |
| 1D residual FNO | Useful fixed-family residual flow-map baselines and corrected evaluation exist. | Larger-step behavior is horizon- and metric-dependent, not universal stability. |
| CPGNet | Released-code evaluation shows that message reach matters materially to the reported 1D gain and causal boundary training improves the bounded local 2D release-bundle result. | The oracle gap remains; legal-boundary evidence is not paper-faithful parity or exact DG replay. |
| Bump residual PCNO | D041 remains the historical comparator; later training, projection, boundary-field, resampling, orientation, and D087 stability-forensics results provide bounded mechanism evidence. D087 separates accurate, admissible, bounded, and finite horizons and finds propagated-input response dominating D019's realized late error. | No later arm is an exact-contract D041 replacement; D087 is not a causal training/architecture or general-stability result; proxy totals are not conservation; transformed-input failures are not an independent rotated PDE benchmark. |
| Dynamic residual PCNO | D044/D060 support a useful baseline lineage. Resolution and pathway studies isolate persistent large-scale mesh defect plus locally cancelling shock/vortex error; bounded adaptive correction evidence exists. | No resolution invariance, general safe correction, benchmark-wide boundary improvement, or learned-solver claim follows. |
| Latent/assimilation | The attempted latent forecast lacked sufficient capacity; assimilation concepts are recorded. | Assimilation is reserved until an open-loop mechanism and target claim justify it. |

Across families, distinguish propagated state error, fresh exact-input defect,
front or shock-position error, smooth high-pass error, admissibility, boundary
leakage, and conservation where defined. A model can remain finite after error
saturates or after becoming inadmissible; neither fact alone establishes stable
or physically valid rollout.

The active research program asks five linked questions: long-horizon stability;
shock representation and the differential pathway; boundary information and
finite propagation; REALM benchmark validation; and cross-resolution
correlation/correction. Exact ladders and authorization stages live only in the
[weekly plan](WEEKLY_RESEARCH_PLAN.md).

## Maintained-Code Navigation

This is a category map, not a hand-maintained exhaustive manifest. Obtain the
exact current inventory from the checkout:

```powershell
rg --files utility/time_dependent_no
rg --files scripts/time_dependent_no
rg --files tests/time_dependent_no
```

Presence in these directories records an implementation or reproducibility
surface. It does not mean an experiment is selected, authorized, or running.

### Reusable utilities

`utility/time_dependent_no/` contains branch-local reusable code:

- common errors, losses, finite-volume helpers, and metrics;
- 1D Euler data, targets, models, and solver support;
- CPG release, mesh-contract, reach, and bump-state utilities;
- residual-PCNO artifact, runtime, rollout, and Euler-state adapters;
- boundary-field construction and intervention support;
- dynamic shock-vortex reference, geometry, family, and metric contracts;
- resolution-transfer, ripple, residual-structure, and pathway diagnostics; and
- bounded defect-correction and local-correctability components.

Keep new reusable code here until it is stable enough for a core API. Promote an
abstraction only when it removes current complexity or has multiple real callers.
Core `pcno/`, `baselines/`, and unrelated examples remain outside this
branch-specific inventory.

### Experiment entry points

`scripts/time_dependent_no/` contains maintained entry points grouped by role:

- 1D dataset generation, target-ladder training, evaluation, runtime, and plots;
- CPG release provenance, mesh/reach audits, legal-boundary training, and plots;
- bump shard preparation, residual-PCNO training/evaluation, boundary protocols,
  rollout decomposition, and geometry diagnostics;
- dynamic shock-vortex reference/family generation and baseline evaluation;
- common-source resolution rollout and pathway analysis;
- boundary-field, node-type, admissibility, and finite-propagation probes;
- long-horizon stability event, recurrence-feedback, and fresh/propagated
  diagnostic evaluation;
- REALM benchmark, IgnitHIT normalization, FFNO, and domain-compatible output
  contracts; and
- native residual-correction, response-controller, local-channel, and
  visualization tools retained for reproducibility.

Review an entry point's arguments, source binding, population, and output path
before execution. The ADER generator is configuration-driven: invoking it with
`--help` starts its default multiprocessing generation job, so do not use it as
a harmless help probe.

### Tests

`tests/time_dependent_no/` mirrors the maintained categories with synthetic or
small CPU fixtures. It covers data and mesh contracts, 1D targets, residual
rollout, artifact snapshots, boundary fields and interventions, dynamic-FV
geometry, resolution transfer, structure/pathway diagnostics, correction
controllers, long-horizon event/decomposition contracts, and visualization
payloads.

Run the narrowest relevant test first. When the environment requires a writable
pytest temporary directory, place `--basetemp` inside an ignored workspace path
and clean only the exact resolved cache paths afterward.

## Experiment And Artifact Discipline

- New run identities bind source, checkpoint, normalizer, split, population,
  recurrence, boundary policy, precision, and metric definitions.
- A changed scientific contract receives a new identity; historical outcomes
  are immutable.
- Use open populations for development. Strength-OOD and test populations stay
  sealed until explicitly opened by the owner.
- Use synthetic CPU fixtures before dataset-scale or AutoDL execution.
- Do not commit raw datasets, checkpoints, generated rollouts, large logs,
  credentials, private hostnames, or local machine paths.
- Every retained generated package needs a manifest, hashes where required, and
  a clear link to its scientific or recovery role.
- Artifact deletion requires a refreshed inventory and reference check. Ignore
  status, age, or a smoke-like name alone does not make an output disposable.

## Source-Snapshot Semantics

New PCNO training runs use `pcno_euler2d_source_snapshot_v5`, implemented in
`utility/time_dependent_no/pcno_artifacts.py`. Continuation compares the current
checkout against the schema-specific executable/scientific file set recorded by
the run.

| Schema | Historical meaning |
| --- | --- |
| v2 | Binds the then-current decision and tracker inside the strict source file set together with the original residual-PCNO sources. |
| v3 | Uses the original executable/scientific source set and records the decision/tracker separately as provenance. |
| v4 | Expands the executable set to the package and support modules needed for exact continuation. |
| v5 | Adds the boundary-field utility while retaining separate provenance-document hashes. |

For v3-v5, later edits to the active decision or tracker do not by themselves
invalidate executable continuation. Their copied provenance and manifest
digests remain part of the archived run record. Historical schemas keep their
registered inventories and must never be silently reinterpreted as v5.

Changing a v4/v5-bound executable source during cleanup requires an explicit v6
design plus v2-v5 compatibility tests. Archived snapshot integrity and
compatibility with the current checkout are different checks. The legacy
`fixture` wording in `utility/time_dependent_no/__init__.py` therefore remains
until the v6 gate is approved.

## Recovery

- [Archive guide](history/README.md)
- [Decision state through 2026-08-11](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
- [Evidence ledger through 2026-08-11](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
- [Frozen D072 records](history/d072_refine_logs_frozen_2026-08-03/)
- [Corrected 1D baselines](SECTION_1_2_CORRECTED_BASELINES.md)
- [Boundary-field derivation package](BOUNDARY_FIELD_DERIVATION_PACKAGE.md)
- [Boundary-field prior-art audit](BOUNDARY_FIELD_PRIOR_ART_AUDIT.md)

Commit `5646bfb` preserves the full active decision and tracker immediately
before compaction. Commit `ebf210a` preserves the weekly plan and prompts;
`3e646ac` preserves the first isolated-scaffolding cleanup. Large generated
artifacts remain outside Git and depend on their manifests rather than commit
history for recovery.

# Time-Dependent Neural Operators

Updated: 2026-08-21

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
6. The [W26-L4 section of the weekly plan](WEEKLY_RESEARCH_PLAN.md#w26-l4-realm-benchmark-and-paper-level-validation)
   for the current PlanarDet problem-discovery contract and authorization
   boundary.
7. [D093_W26_L4_PLANARDET_SCALING_RECORD.md](D093_W26_L4_PLANARDET_SCALING_RECORD.md)
   for the closed PlanarDet architecture/exposure comparison.
8. [W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md)
   for the closed reference-free H320 bump recurrence result.
9. [D094_BUMP_SCALING_PREREGISTRATION.md](D094_BUMP_SCALING_PREREGISTRATION.md)
   for the retained B1-A/B1-B schedule audit and running stretched-schedule
   seed-0 trajectory ladder.
10. Read one bounded section of the
   [2026-08-11 decision archive](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
   or [evidence archive](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
   when exact historical contracts, results, or hashes are required.

Current W26-L4 execution state: PlanarDet A1--A3, D092-R1 training, open-
validation evaluation, P0b, corrected G0b, and the G1 one-call chemistry/density
pulse study are complete. The PCNO's released-code H49 validation sum is
`88.13821`, 7.01x the REALM paper's FFNO validation value `12.577`. P0b is
near-null; G0b localizes strong chemistry/density recurrence sensitivity; G1
shows that the partner response is immediately bidirectional but materially
persistent only from chemistry to density at calls 12 and 32. D093 then reuses
the PCNO-7 result anchor and adds PCFNO/FFNO at three and seven unique
supervised conditions. Each seven-condition cell has lower selected truth-input
error than its three-condition counterpart, but only FFNO has a lower selected
free-rollout sum; FFNO-7 is best at `1.36466/32.53910`.
This is one seed and one open trajectory under a shared residual contract, not
clean data scaling, an architecture cause, paper-faithful FFNO reproduction,
physical causal graph, correction method, or sealed ranking; the released test
remains absent. Use the handoff, D093 record, and weekly plan for exact bindings
and claim boundaries.

Current D094 execution state: B1-A and the preregistered 28-trajectory B1-B
outside-selection audit are locally retained and rehashed. B1-B selects the
stretched schedule under its paired error-first rule. The fresh paired seed-0
`n={8,16,32,64,128}` PCNO/PCFNO ladder is running serially on AutoDL; the
winning B1-A `n=256` endpoints will be reused. Initial health is verified, but
the ladder has no completed result yet and should not be interpreted from
partial checkpoints. The historical test population remains sealed.

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
| Bump residual PCNO | D041 remains the historical comparator; later training, projection, boundary-field, resampling, orientation, and D087 stability-forensics results provide bounded mechanism evidence. D087 separates accurate, admissible, bounded, and finite horizons and finds propagated-input response dominating D019's realized late error. The separately registered H320 study finds later reference-free failure events for active-gradient PCNO than PCFNO on the 30 open cases. | No later arm is an exact-contract D041 replacement; H80--H320 has no truth and is not accuracy, physical-validity, conservation, asymptotic-stability, or causal gradient evidence; proxy totals are not conservation; transformed-input failures are not an independent rotated PDE benchmark. |
| Dynamic residual PCNO | D044/D060 support a useful baseline lineage. Resolution and pathway studies isolate persistent large-scale mesh defect plus locally cancelling shock/vortex error; bounded adaptive correction evidence exists. | No resolution invariance, general safe correction, benchmark-wide boundary improvement, or learned-solver claim follows. |
| REALM PlanarDet | D092-R1 provides one completed tuned residual-PCNO anchor with a 14.45x free/truth H49 error gap. P0b is near-null; G0b/G1 localize a checkpoint-specific chemistry-to-density persistence asymmetry. In D093, each seven-condition cell has lower selected truth-input error than its three-condition counterpart, while only FFNO has a lower selected free-rollout sum. | One seed and one validation trajectory; the low-condition arm retains seven-condition normalization and D092/D093 bind distinct source inventories. No clean data scaling, paper-faithful FFNO, sealed test, physical causal graph, architecture cause, seed robustness, promoted oracle intervention, conservation, or REALM-wide result. |
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
- REALM PlanarDet open-manifest/data adapters, train-only normalization,
  residual-PCNO runtime/provenance, and detonation-structure diagnostics;
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
- REALM benchmark, IgnitHIT normalization, FFNO, regular-grid PCNO, and
  domain-compatible output contracts;
- pinned PlanarDet acquisition/audit, full-grid GPU smoke, exact-resume
  residual-PCNO training, shared fresh/teacher/free evaluation, causal
  cumulative-`pMax` projection, recurrent group-feedback and one-call pulse
  diagnosis, and result-bound visualization; and
- native residual-correction, response-controller, local-channel, and
  visualization tools retained for reproducibility.

Review an entry point's arguments, source binding, population, and output path
before execution. The ADER generator is configuration-driven but now fails
closed: `--help` is side-effect-free, a no-argument invocation is rejected, and
artifact generation requires explicit `--run` after reviewing its `CONFIG`.

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

Euler2D residual-PCNO runs use `pcno_euler2d_source_snapshot_v2` through `v6`,
implemented in `utility/time_dependent_no/pcno_artifacts.py`; v6 is the latest
registered schema for that family. REALM PlanarDet instead uses
`realm_planardet_pcno_source_snapshot_v1`. D092 and D093 share that schema name
but bind distinct inventories and payload digests. Continuation compares the
current checkout against the schema-specific executable/scientific file set
recorded by the run.

| Schema | Historical meaning |
| --- | --- |
| v2 | Binds the then-current decision and tracker inside the strict source file set together with the original residual-PCNO sources. |
| v3 | Uses the original executable/scientific source set and records the decision/tracker separately as provenance. |
| v4 | Expands the executable set to the package and support modules needed for exact continuation. |
| v5 | Adds the boundary-field utility while retaining separate provenance-document hashes. |
| v6 | Retains the v5 base inventory and adds a sorted, unique, run-specific `extra_source_files` registry to the strict source set. |

For v3-v6, later edits to the active decision or tracker do not by themselves
invalidate executable continuation. Their copied provenance and manifest
digests remain part of the archived run record. Historical schemas keep their
registered inventories and must never be silently reinterpreted as v6.

V6 closes the governance gap created when a v5-bound core source changed after
v5 was registered. V5 keeps its frozen inventory; an archived v5 run is
compatible with the current checkout only when every recorded hash still
matches. The v2--v6 compatibility tests exercise schema-specific inventory and
verification branches, not byte equality between historical snapshots and
current source.
Archived snapshot integrity and current-checkout continuation compatibility are
different checks.

Source-set digests are byte-level identities. A Git archive can materialize LF
bytes while a clean Windows worktree materializes selected files with CRLF, so
their source-set digests may differ even when both derive from the same commit.
For a remote run, bind and verify the exact deployment-archive hash first, then
use the source-set digest generated inside that deployed tree; never substitute
a digest recomputed from a byte-distinct local materialization.

## Recovery

- [Archive guide](history/README.md)
- [Decision state through 2026-08-11](history/RESEARCH_DIRECTION_DECISION_through_2026-08-11.md)
- [Evidence ledger through 2026-08-11](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-11.md)
- [Frozen D072 records](history/d072_refine_logs_frozen_2026-08-03/)
- [Corrected 1D baselines](SECTION_1_2_CORRECTED_BASELINES.md)
- [Boundary-field derivation package](BOUNDARY_FIELD_DERIVATION_PACKAGE.md)
- [Boundary-field prior-art audit](BOUNDARY_FIELD_PRIOR_ART_AUDIT.md)
- [H320 reference-free rollout record](W26_L1_H320_REFERENCE_FREE_ROLLOUT_RECORD.md)
- [D094 bump-scaling preregistration](D094_BUMP_SCALING_PREREGISTRATION.md)
- [Hash-bound W26-L5 derivation source](../../DERIVATION_PACKAGE.md)

Commit `5646bfb` preserves the full active decision and tracker immediately
before compaction. Commit `ebf210a` preserves the weekly plan and prompts;
`3e646ac` preserves the first isolated-scaffolding cleanup. Large generated
artifacts remain outside Git and depend on their manifests rather than commit
history for recovery.

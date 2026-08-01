# Time-Dependent Neural Operators

Updated: 2026-08-01

This directory is the onboarding and maintained-code inventory for the summer
2026 time-dependent neural-operator work on branch time-dependent-no. It is not
an experiment queue or a substitute for current human direction.

## Navigation

Read the compact active context in this order:

1. [RESEARCH_DIRECTION_DECISION.md](RESEARCH_DIRECTION_DECISION.md) for the
   current scientific state, claim boundaries, and standing owner constraints.
2. [HANDOFF.md](HANDOFF.md) for the current workspace and next human review.
3. [MECHANISTIC_DIAGNOSTIC_TRACKER.md](MECHANISTIC_DIAGNOSTIC_TRACKER.md) for
   experiment-ID and topic routing.
4. One bounded section of the
   [historical evidence ledger](history/MECHANISTIC_DIAGNOSTIC_TRACKER_through_2026-08-01.md)
   when exact contracts, metrics, hashes, or prior outcomes are needed.

The byte-preserved prior decision surface is
[archived here](history/RESEARCH_DIRECTION_DECISION_through_2026-08-01.md).
Historical forward-looking language records what was proposed at that time; it
does not constrain a newer explicit owner request.

Private paths, remote hosts, credentials, and machine-specific dataset
locations belong in ignored LOCAL_CONTEXT.md. Generated reports, arrays,
figures, checkpoints, and large logs belong under ignored
artifacts/time_dependent_no/.

## Data And Provenance

The folder historically labeled forward_300 is the CPG supersonic-bump bundle:
300 training trajectories, 20 test trajectories, 80 HDF5 frames, and roughly
19k--23k graph nodes per trajectory.

- [CPG_EULER_DATASET_CONTRACT.md](CPG_EULER_DATASET_CONTRACT.md) defines the
  live HDF5 schema and reader convention.
- [BUMP_300_DATASET_AUDIT.md](BUMP_300_DATASET_AUDIT.md) records raw-bundle and
  solver-lineage evidence.
- [CPGGNSPDES_REFERENCE_AUDIT.md](CPGGNSPDES_REFERENCE_AUDIT.md) records the
  bounded public-reference audit.
- [SECTION_1_2_CORRECTED_BASELINES.md](SECTION_1_2_CORRECTED_BASELINES.md)
  preserves corrected historical 1D labels and results.

The bump HDF5 does not expose validated finite-volume control volumes, oriented
physical faces, measures, normals, or accepted-substep impulses. Equal-node or
reconstructed-weight totals are diagnostic proxies, not physical conservation.

The separate Mach-1.1 shock-vortex family has audited finite-volume geometry,
boundary accounting, and cumulative accepted-substep impulses. Those fields
support physical outcome diagnostics only on that frozen solver and data
contract. State-residual PCNO recurrence is not conservative by construction.

## Maintained Code Inventory

Presence here records a maintained implementation or reproducibility surface.
It does not imply that an experiment is selected or currently running.

### Reusable utilities

Shared and 1D finite-volume support:

- utility/time_dependent_no/__init__.py
- utility/time_dependent_no/errors.py
- utility/time_dependent_no/fv.py
- utility/time_dependent_no/euler1d.py
- utility/time_dependent_no/euler1d_data.py
- utility/time_dependent_no/euler1d_models.py
- utility/time_dependent_no/euler1d_targets.py

CPG/bump contracts and diagnostics:

- utility/time_dependent_no/euler2d.py
- utility/time_dependent_no/euler2d_synthetic.py
- utility/time_dependent_no/euler2d_metrics.py
- utility/time_dependent_no/euler2d_fixture.py
- utility/time_dependent_no/cpg_release.py
- utility/time_dependent_no/cpg_mesh_contract.py
- utility/time_dependent_no/cpg_reach.py

Residual-PCNO and dynamic finite-volume support:

- utility/time_dependent_no/pcno_artifacts.py
- utility/time_dependent_no/pcno_euler2d.py
- utility/time_dependent_no/pcno_rollout.py
- utility/time_dependent_no/pcno_runtime.py
- utility/time_dependent_no/pcno_fv_geometry.py
- utility/time_dependent_no/pcno_ripple_diagnostics.py
- utility/time_dependent_no/pcno_resolution_transfer.py
- utility/time_dependent_no/shock_vortex_fv.py
- utility/time_dependent_no/shock_vortex_coarse_cfd.py
- utility/time_dependent_no/shock_vortex_metrics.py
- utility/time_dependent_no/shock_vortex_family.py
- utility/time_dependent_no/fv_impulse_diagnostics.py

### Entry points

Frozen 1D Euler generation, training, evaluation, and reporting:

- scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py
- scripts/time_dependent_no/train_euler1d_target_ladder.py
- scripts/time_dependent_no/analyze_euler1d_target_ladder.py
- scripts/time_dependent_no/evaluate_euler1d_flow_map_frontier.py
- scripts/time_dependent_no/benchmark_euler1d_flow_map_runtime.py
- scripts/time_dependent_no/generate_euler1d_flow_map_ood.py
- scripts/time_dependent_no/visualize_euler1d_flow_map_frontier.py

The ADER generator is configuration-driven rather than a safe argparse help
surface. Invoking it with --help starts its default multiprocessing generation
job. Review its configuration and output path before deliberate execution.

CPG release, mesh, reach, legal-boundary, and visualization:

- scripts/time_dependent_no/audit_cpg_release_provenance.py
- scripts/time_dependent_no/audit_cpg_mesh_contract.py
- scripts/time_dependent_no/evaluate_cpg_release.py
- scripts/time_dependent_no/diagnose_cpg_characteristic_reach.py
- scripts/time_dependent_no/train_cpg_legal_boundary.py
- scripts/time_dependent_no/visualize_official_cpg_rollout.py

Bump residual-PCNO preparation, evaluation, and diagnostics:

- scripts/time_dependent_no/prepare_pcno_euler2d_shards.py
- scripts/time_dependent_no/train_pcno_euler2d_residual.py
- scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py
- scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_protocol.py
- scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_splice.py
- scripts/time_dependent_no/decompose_pcno_euler2d_rollout_error.py
- scripts/time_dependent_no/rollout_pcno_preprocessed.py

Dynamic shock-vortex reference, family, PCNO, and resolution tools:

- scripts/time_dependent_no/generate_euler2d_shock_vortex_reference.py
- scripts/time_dependent_no/generate_euler2d_shock_vortex_pyro_reference.py
- scripts/time_dependent_no/generate_euler2d_shock_vortex_sharpclaw_reference.py
- scripts/time_dependent_no/audit_euler2d_shock_vortex_convergence.py
- scripts/time_dependent_no/build_euler2d_shock_vortex_family.py
- scripts/time_dependent_no/generate_euler2d_shock_vortex_family_case.py
- scripts/time_dependent_no/generate_pcno_shock_vortex_multires_reference.py
- scripts/time_dependent_no/prepare_pcno_shock_vortex_shards.py
- scripts/time_dependent_no/evaluate_pcno_shock_vortex_baseline.py
- scripts/time_dependent_no/evaluate_pcno_resolution_transfer.py
- scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py
- scripts/time_dependent_no/visualize_pcno_resolution_rollout.py

### Tests

Shared support:

- tests/time_dependent_no/__init__.py
- tests/time_dependent_no/_pcno_test_support.py

CPG/release:

- tests/time_dependent_no/test_cpg_euler_data.py
- tests/time_dependent_no/test_cpg_mesh_contract.py
- tests/time_dependent_no/test_cpg_reach.py
- tests/time_dependent_no/test_cpg_release.py
- tests/time_dependent_no/test_train_cpg_legal_boundary.py
- tests/time_dependent_no/test_visualize_official_cpg_rollout.py

1D Euler:

- tests/time_dependent_no/test_euler1d_flow_map_frontier.py
- tests/time_dependent_no/test_euler1d_flow_map_visualization.py
- tests/time_dependent_no/test_euler1d_solver_targets.py
- tests/time_dependent_no/test_euler1d_training_noise.py

Generic 2D/PCNO:

- tests/time_dependent_no/test_euler_fixture.py
- tests/time_dependent_no/test_euler_metrics.py
- tests/time_dependent_no/test_fv_impulse_diagnostics.py
- tests/time_dependent_no/test_pcno_artifacts.py
- tests/time_dependent_no/test_pcno_euler2d_residual.py
- tests/time_dependent_no/test_pcno_euler2d_boundary_splice.py
- tests/time_dependent_no/test_pcno_euler2d_multistep_training.py
- tests/time_dependent_no/test_pcno_fv_geometry.py
- tests/time_dependent_no/test_pcno_ripple_diagnostics.py
- tests/time_dependent_no/test_pcno_rollout.py
- tests/time_dependent_no/test_pcno_rollout_error_decomposition.py
- tests/time_dependent_no/test_pcno_runtime.py

Dynamic shock-vortex:

- tests/time_dependent_no/test_pcno_shock_vortex_baseline.py
- tests/time_dependent_no/test_pcno_resolution_transfer.py
- tests/time_dependent_no/test_shock_vortex_coarse_cfd.py
- tests/time_dependent_no/test_shock_vortex_family.py
- tests/time_dependent_no/test_shock_vortex_fv.py
- tests/time_dependent_no/test_shock_vortex_sharpclaw_adapter.py
- tests/time_dependent_no/test_visualize_pcno_resolution_rollout.py

## Source-Snapshot And Recovery Notes

New PCNO runs use source-snapshot schema v3. Executable/scientific source bytes
form the continuation compatibility set. The active decision and experiment
index are still copied and hashed into each run as provenance, but later
documentation edits do not invalidate continuation.

Historical v2 snapshots retain their original strict equality semantics,
including the then-current decision and tracker bytes. They are not silently
reinterpreted as v3.

Key recovery anchors:

- ae1f402: D063 evidence record;
- e2070f6: reusable PCNO infrastructure and the full active documentation
  immediately before documentation compaction;
- ce5d6a2: pre-infrastructure-cleanup source and retired diagnostic entry
  points; and
- 31e5765: earlier expanded documentation provenance.

The compact index records experiment-specific source and artifact pointers.
Core pcno/, baselines/, and unrelated examples remain outside this
branch-specific inventory.

# Time-Dependent Neural Operators

Updated: 2026-07-26

This directory records the summer 2026 time-dependent neural-operator work on
branch `time-dependent-no`.

## Authority And Navigation

Read project state in this order:

1. [`RESEARCH_DIRECTION_DECISION.md`](RESEARCH_DIRECTION_DECISION.md) is the
   authoritative scientific decision and claim boundary.
2. [`HANDOFF.md`](HANDOFF.md) is the derived operational snapshot.
3. [`MECHANISTIC_DIAGNOSTIC_TRACKER.md`](MECHANISTIC_DIAGNOSTIC_TRACKER.md) is
   the evidence ledger for exact run IDs, gates, and artifact provenance.
4. This README is non-authoritative onboarding and the sole documentation
   inventory of active code.

If these pages disagree, follow the earlier item. Private paths, hosts, and
credentials belong only in ignored `LOCAL_CONTEXT.md`.

## Final Four-Line Status

The overall result-to-claim verdict is `partial`, with high confidence. The
campaign supports a useful fixed-family residual PCNO baseline and a bounded
mechanistic account. It does not support a generally improved, shock-stable
geometry-aware neural operator.

| Line | Final status | Supported scope | Explicit non-claim |
| --- | --- | --- | --- |
| 1: large-step flow maps | Closed | On the frozen 1D Euler contracts, the useful stride is horizon- and metric-dependent: a harder one-call map can win after fewer recurrent compositions. | No universal optimal stride, learned CFL limit, timestep-conditioned solver, or native-grid transfer follows. |
| 2: CPGNet validity and mechanism | Closed | In the corrected 1D study, message reach matters more than width alone; the learned interface coordinates are functional controls rather than verified physical traces. Legal-boundary training improves the release-bundle result without closing the oracle-boundary gap. | This is not a paper-table reproduction, an implicit scheme, a general learned solver, or proof of physical interface states or conservation. |
| 3: geometry-aware 2D rollout | Closed without method promotion | The one-seed D044 conservative-coordinate residual PCNO is a useful raw-rollout baseline on the frozen Mach-1.1 shock-vortex family. D060 reduces recurrent state error, while the combined branch, decoder, recurrence, geometry-transfer, and front-identity evidence remains composite. | No promoted ripple-stable solver, facewise-conservative learned method, broad PCNO/MPCNO conclusion, seed-robust result, strength-OOD result, or test result follows. |
| 4: latent forecasting and assimilation | Stopped before forecast training | The tested smooth and fixed-Haar representations isolate a decoder-capacity limitation; discontinuous regularity helps but does not pass the reconstruction and front hierarchy. | No latent transition, autonomous recurrence, geometry transfer, neural-operator, or data-assimilation claim was tested. |

## Authorized Work

The active queue is report-only and must use frozen artifacts. It may:

- place D044 and D060 state, high-pass, front, and geometry-stratum curves
  together;
- present D053 propagated-versus-fresh error shares beside the D052 branch
  falsification;
- present D048/D049 discrete-decoder amplification beside D062 front-identity
  failure;
- tabulate objective, sample presentations, selected epoch, intervention,
  population, evidence grade, and non-claim for each reported result; and
- retain failed implementation attempts only as provenance, not as scientific
  trials.

This queue does not authorize checkpoint execution, a changed threshold, GPU
work, a sealed-split read, a new oracle, model training, transition learning, or
data assimilation.

## Data And Provenance Boundaries

The folder once labeled `forward_300` is the CPG supersonic-bump bundle: 300
training trajectories, 20 test trajectories, 80 saved HDF5 frames, and roughly
19k-23k graph nodes per trajectory. Use
[`CPG_EULER_DATASET_CONTRACT.md`](CPG_EULER_DATASET_CONTRACT.md) for the live
schema, [`BUMP_300_DATASET_AUDIT.md`](BUMP_300_DATASET_AUDIT.md) for frozen
bundle provenance, and
[`CPGGNSPDES_REFERENCE_AUDIT.md`](CPGGNSPDES_REFERENCE_AUDIT.md) for the public
reference-code audit.

The bump bundle does not establish paper dataset/checkpoint identity or expose
validated control volumes, face measures, normals, oriented physical faces, or
accepted-substep reference impulses. Equal-node or reconstructed-weight sums
remain diagnostic proxies; they cannot support physical 2D conservation or
reference-flux claims. The released evaluator's next-reference boundary
injection is oracle-assisted and must not be presented as autonomous rollout.

The separate Mach-1.1 shock-vortex family has an audited finite-volume mapping,
physical volumes and oriented faces, boundary accounting, and cumulative
accepted-substep impulses. Those fields support physical diagnostics only on
that frozen family and time contract. D044 and D060 predict conservative-state
residuals; neither recurrence is a face-flux conservative neural solver.
Strength-OOD and test splits remain sealed.

Historical corrected 1D row labels, final results, and the frozen-run analyzer
contract remain in
[`SECTION_1_2_CORRECTED_BASELINES.md`](SECTION_1_2_CORRECTED_BASELINES.md).
Generated reports, arrays, figures, checkpoints, and large logs belong under the
ignored `artifacts/time_dependent_no/` tree.

## Active Code Inventory

This section is the sole documentation inventory of active code. Presence here
records a maintained implementation or reproducibility surface; it does not
authorize a new experiment. Historical one-off probes and failed-method
scaffolds are recoverable from commit `729091b`. The pruned active surface is
commit `cf6cbe1`.

### Reusable utilities

Shared and 1D finite-volume support:

- `utility/time_dependent_no/__init__.py`
- `utility/time_dependent_no/errors.py`
- `utility/time_dependent_no/fv.py`
- `utility/time_dependent_no/euler1d.py`
- `utility/time_dependent_no/euler1d_data.py`
- `utility/time_dependent_no/euler1d_models.py`
- `utility/time_dependent_no/euler1d_targets.py`

CPG/bump contracts and diagnostics:

- `utility/time_dependent_no/euler2d.py`
- `utility/time_dependent_no/euler2d_synthetic.py`
- `utility/time_dependent_no/euler2d_metrics.py`
- `utility/time_dependent_no/euler2d_fixture.py`
- `utility/time_dependent_no/cpg_release.py`
- `utility/time_dependent_no/cpg_mesh_contract.py`
- `utility/time_dependent_no/cpg_reach.py`

Residual-PCNO and dynamic finite-volume support:

- `utility/time_dependent_no/pcno_euler2d.py`
- `utility/time_dependent_no/pcno_fv_geometry.py`
- `utility/time_dependent_no/pcno_ripple_diagnostics.py`
- `utility/time_dependent_no/shock_vortex_fv.py`
- `utility/time_dependent_no/shock_vortex_coarse_cfd.py`
- `utility/time_dependent_no/shock_vortex_metrics.py`
- `utility/time_dependent_no/shock_vortex_family.py`
- `utility/time_dependent_no/fv_impulse_diagnostics.py`

### Entry points

Frozen 1D Euler generation, training, evaluation, and reporting:

- `scripts/time_dependent_no/euler1d_weno_hllc_ader_dataset.py`
- `scripts/time_dependent_no/train_euler1d_target_ladder.py`
- `scripts/time_dependent_no/analyze_euler1d_target_ladder.py`
- `scripts/time_dependent_no/evaluate_euler1d_flow_map_frontier.py`
- `scripts/time_dependent_no/benchmark_euler1d_flow_map_runtime.py`
- `scripts/time_dependent_no/generate_euler1d_flow_map_ood.py`
- `scripts/time_dependent_no/visualize_euler1d_flow_map_frontier.py`
- `scripts/time_dependent_no/launch_euler1d_large_step_frontier.sh`

`euler1d_weno_hllc_ader_dataset.py` is a configuration-driven generator, not an
argparse help surface. Do not invoke it with `--help`: that starts its default
multiprocessing dataset job. Run it only as a deliberate generation task after
reviewing its configuration and output path.

Frozen CPG release, mesh, reach, legal-boundary, and visualization surfaces:

- `scripts/time_dependent_no/audit_cpg_release_provenance.py`
- `scripts/time_dependent_no/audit_cpg_mesh_contract.py`
- `scripts/time_dependent_no/evaluate_cpg_release.py`
- `scripts/time_dependent_no/diagnose_cpg_characteristic_reach.py`
- `scripts/time_dependent_no/train_cpg_legal_boundary.py`
- `scripts/time_dependent_no/visualize_official_cpg_rollout.py`

Bump residual-PCNO preparation, evaluation, and frozen diagnostics:

- `scripts/time_dependent_no/prepare_pcno_euler2d_shards.py`
- `scripts/time_dependent_no/train_pcno_euler2d_residual.py`
- `scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py`
- `scripts/time_dependent_no/diagnose_pcno_euler2d_ripples.py`
- `scripts/time_dependent_no/rollout_pcno_preprocessed.py`

Dynamic shock-vortex reference, family, residual-PCNO, and coarse-CFD surfaces:

- `scripts/time_dependent_no/generate_euler2d_shock_vortex_reference.py`
- `scripts/time_dependent_no/generate_euler2d_shock_vortex_pyro_reference.py`
- `scripts/time_dependent_no/generate_euler2d_shock_vortex_sharpclaw_reference.py`
- `scripts/time_dependent_no/audit_euler2d_shock_vortex_convergence.py`
- `scripts/time_dependent_no/build_euler2d_shock_vortex_family.py`
- `scripts/time_dependent_no/generate_euler2d_shock_vortex_family_case.py`
- `scripts/time_dependent_no/prepare_pcno_shock_vortex_shards.py`
- `scripts/time_dependent_no/evaluate_pcno_shock_vortex_baseline.py`
- `scripts/time_dependent_no/benchmark_pcno_shock_vortex_coarse_cfd.py`

### Tests

Active CPU and synthetic-fixture tests live under `tests/time_dependent_no/`:

- CPG/release: `test_cpg_euler_data.py`, `test_cpg_mesh_contract.py`,
  `test_cpg_reach.py`, `test_cpg_release.py`,
  `test_train_cpg_legal_boundary.py`, and
  `test_visualize_official_cpg_rollout.py`.
- 1D Euler: `test_euler1d_flow_map_frontier.py`,
  `test_euler1d_flow_map_visualization.py`, `test_euler1d_solver_targets.py`, and
  `test_euler1d_training_noise.py`.
- Generic 2D/PCNO: `test_euler_fixture.py`, `test_euler_metrics.py`,
  `test_fv_impulse_diagnostics.py`, `test_pcno_euler2d_residual.py`,
  `test_pcno_fv_geometry.py`, and `test_pcno_ripple_diagnostics.py`.
- Dynamic shock-vortex: `test_pcno_shock_vortex_baseline.py`,
  `test_shock_vortex_coarse_cfd.py`, `test_shock_vortex_family.py`,
  `test_shock_vortex_fv.py`, and `test_shock_vortex_sharpclaw_adapter.py`.

The CPG and PCNO evaluation files above are frozen reproducibility surfaces, not
an active experiment queue. Core `pcno/`, `baselines/`, and unrelated examples
remain outside this branch-specific inventory.

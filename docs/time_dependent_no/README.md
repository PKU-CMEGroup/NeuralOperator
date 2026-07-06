# Time-Dependent Neural Operators

This branch contains the summer 2026 time-dependent neural-operator work inside the PKU-CME `NeuralOperator` codebase. The active benchmark is the CPG-style 2D Euler supersonic bump dataset, with CPGNet and PCNO rollout diagnostics used to understand time-dependent neural-operator failure modes.

The tracked code is intentionally lean. Historical one-off probes and scaffold scripts were removed from the active tree after their conclusions were recorded in `MECHANISTIC_DIAGNOSTIC_TRACKER.md`; recover them from git history if an old result must be reproduced exactly.

## Active Code

Reusable branch utilities live in `utility/time_dependent_no/`:

- `euler2d.py`: CPG HDF5 schema inspection, primitive/conservative conversion, node-type helpers, and graph-frame materialization.
- `euler2d_synthetic.py`: deterministic CPG-style synthetic fixture for CPU tests.
- `euler2d_metrics.py`: rollout, positivity, conservation, shock-proxy, boundary, and compact-summary diagnostics.
- `euler2d_fixture.py`: end-to-end no-model fixture diagnostics.
- `errors.py`: small NumPy error helpers used by diagnostics.

Active command-line entry points live in `scripts/time_dependent_no/`:

```bash
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
python scripts/time_dependent_no/diagnose_cpg_rollout_mechanisms.py --help
python scripts/time_dependent_no/visualize_official_cpg_rollout.py --help
python scripts/time_dependent_no/visualize_cpg_shock_overlays.py --help
python scripts/time_dependent_no/rollout_pcno_preprocessed.py --help
```

`rollout_pcno_preprocessed.py` is the corrected PCNO evaluation path. It assumes the collaborator-compatible preprocessing contract: HDF5 trajectories are converted to per-trajectory arrays, reconstructed into the PCNO Euler `.npz` format, and then rolled out. The retired raw-HDF5 PCNO adapter path should not be used for checkpoint evaluation.

Tests live in `tests/time_dependent_no/`.

## Current Dataset State

The copied dataset folder label was `forward_300`, but extracted files identify the supersonic bump dataset:

- 300 train trajectories and 20 test trajectories;
- 80 HDF5 time steps per trajectory;
- roughly 19k to 23k nodes per trajectory;
- expected CPG keys are present;
- extracted test cases contain `Bump.jl`, `Bump.msh`, `Bump.inp`, `Mach.txt`, `params.txt`, and VTU snapshots.

See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md` and `docs/time_dependent_no/CPG_EULER_DATASET_CONTRACT.md` for stable schema facts. Exact AutoDL paths remain in ignored local context.

## Current Diagnostic Readout

CPGNet learns accurate teacher-forced one-step updates but fails in autoregressive rollout, mainly around shock-local phase, shape, amplitude, and stability rather than simple one-step underfitting. PCNO with the corrected preprocessing path initially follows the solution better visually but develops Fourier-style ripples that can trigger rollout crash.

Next diagnostics should measure ripple energy and shock-front geometry directly, using the common rollout artifact contract: `predicteds`, `targets`, `pos`, `edges`, and `node_type` when available.

## Not Ported Or Active

The raw `cpggnspdes` training scripts are not vendored into this branch. The CPGNet interface-latent probe code, state-drift/perturbation/time-alignment scripts, early smoke scripts, and raw-HDF5 PCNO adapter path were one-off research tools and are no longer active tracked code.

FNO, PCNO, and MPCNO baselines should use implementations already present in this repository or collaborator-provided preprocessing artifacts. Do not port baseline code from earlier data-assimilation experiment repositories.

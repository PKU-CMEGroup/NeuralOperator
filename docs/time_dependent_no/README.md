# Time-Dependent Neural Operators

This branch contains the summer 2026 time-dependent neural-operator work inside the PKU-CME `NeuralOperator` codebase.

The immediate target is the CPG-style 2D Euler benchmark from the Structure-Preserving Graph Neural Solver work. The first objective is not to add a new method. The first objective is to reproduce the data contract, establish FNO/PCNO/MPCNO baseline interfaces, and build diagnostics that explain forecast failure.

## Current Code

Reusable branch utilities live in `utility/time_dependent_no/`:

- `euler2d.py`: CPG HDF5 schema inspection, primitive/conservative conversion, node-type helpers, and graph-frame materialization.
- `euler2d_synthetic.py`: deterministic CPG-style synthetic fixture for CPU smoke tests.
- `euler2d_metrics.py`: rollout, positivity, conservation, shock-proxy, boundary, and compact-summary diagnostics.
- `euler2d_fixture.py`: end-to-end no-model fixture diagnostics.
- `errors.py`: small NumPy error helpers used by the diagnostics.

Thin command-line entry points live in `scripts/time_dependent_no/`:

```bash
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
python scripts/time_dependent_no/inspect_cpg_euler_dataset.py /path/to/train.h5 --metadata-only
```

Tests live in `tests/time_dependent_no/`.

## Current Dataset State

The first copied real dataset has been inspected on AutoDL. Its folder label is `forward_300`, but the extracted files indicate it is the supersonic bump dataset:

- 300 train trajectories and 20 test trajectories;
- 80 HDF5 time steps per trajectory;
- roughly 19k to 23k nodes per trajectory;
- expected CPG keys are present;
- extracted test cases contain `Bump.jl`, `Bump.msh`, `Bump.inp`, `Mach.txt`, `params.txt`, and VTU snapshots.

See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md` for the schema audit. Exact AutoDL paths remain in the ignored `LOCAL_CONTEXT.md`.

## What Is Not Ported

The raw `cpggnspdes` training scripts are not vendored into this branch. They are compact research scripts with hard-coded paths, hard-coded CUDA use, and shutdown side effects. This branch keeps their benchmark contract and model mechanism as reference documentation, then reimplements needed pieces cleanly.

FNO, PCNO, and MPCNO baselines should use the implementations already present in this repository. Do not port baseline code from earlier data-assimilation experiment repositories.

## First Milestone

Completed:

1. Create branch-local context and docs.
2. Port NumPy-first CPG Euler data and diagnostic utilities.
3. Verify synthetic fixture tests locally.
4. Inspect the first copied real dataset schema on AutoDL.

Next:

1. Add a real-data frame smoke script.
2. Choose PCNO/MPCNO adapter-first versus FNO remeshing-sanity-first.
3. Run the smallest no-training data adapter gate before launching any GPU job.

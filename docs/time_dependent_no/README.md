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

## What Is Not Ported

The raw `cpggnspdes` training scripts are not vendored into this branch. They are compact research scripts with hard-coded paths, hard-coded CUDA use, and shutdown side effects. This branch keeps their benchmark contract and model mechanism as reference documentation, then reimplements needed pieces cleanly.

FNO, PCNO, and MPCNO baselines should use the implementations already present in this repository. Do not port baseline code from earlier data-assimilation experiment repositories.

## First Milestone

1. Inspect the real CPG Euler `train.h5` and `test.h5` on AutoDL.
2. Confirm schema, trajectory count, node counts, node types, and whether mesh weights or geometric factors are available.
3. Run synthetic fixture tests locally.
4. Build the first PCNO/MPCNO/FNO data adapters only after the real dataset schema is verified.


# Status

Date: 2026-06-23
Branch: `time-dependent-no`

## Current Milestone

M0: establish branch context, safe documentation split, CPG Euler data utilities, and CPU diagnostic smoke tests.

## Current State

- Branch created from PKU-CME `NeuralOperator` main.
- Reusable CPG Euler data/state/diagnostic utilities are staged under `utility/time_dependent_no/`.
- Thin inspection and fixture scripts are under `scripts/time_dependent_no/`.
- Tests are under `tests/time_dependent_no/`.
- Raw `cpggnspdes` scripts are intentionally not vendored.

## Next Small Step

Run local CPU tests:

```bash
python -m pytest tests/time_dependent_no
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
```

Then inspect the real AutoDL dataset with:

```bash
python scripts/time_dependent_no/inspect_cpg_euler_dataset.py /path/to/train.h5 --max-trajectories 2 --output artifacts/time_dependent_no/train_schema.json
```

Use the real AutoDL dataset path from private `LOCAL_CONTEXT.md`, not from committed docs.

## Open Questions

- Exact AutoDL path for the 200GB CPG Euler dataset.
- Whether public HDF5 files include mesh weights, cell areas, normals, edge lengths, or only graph connectivity.
- Whether the first baseline gate should start with PCNO/MPCNO data adapters or FNO remeshing sanity.

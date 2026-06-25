# Status

Date: 2026-06-25
Branch: `time-dependent-no`

## Current Milestone

M0 is complete: branch context, safe documentation split, CPG Euler data utilities, CPU diagnostic smoke tests, and first real-dataset schema inspection are in place.

The next milestone is M1: build the first real-data adapter gate for the copied supersonic bump dataset, then decide whether to start with PCNO/MPCNO native point-cloud training or FNO remeshing sanity.

## Current State

- Branch created from PKU-CME `NeuralOperator` main.
- Reusable CPG Euler data/state/diagnostic utilities are under `utility/time_dependent_no/`.
- Thin inspection and fixture scripts are under `scripts/time_dependent_no/`.
- Tests are under `tests/time_dependent_no/`.
- Raw `cpggnspdes` scripts are intentionally not vendored.
- AutoDL has a local clone on this branch; exact machine paths are stored in ignored `LOCAL_CONTEXT.md`.
- The copied dataset folder is labeled `forward_300`, but extracted case files identify it as the supersonic bump dataset. See `docs/time_dependent_no/BUMP_300_DATASET_AUDIT.md`.

## Verified Dataset Facts

- HDF5 keys match the expected CPG schema: `Mach`, `edges`, `node_type`, `pos`, `pres`, `rho`, `v1`, `v2`.
- `train.h5`: 300 trajectories, 80 time steps, node counts roughly 19k to 23k.
- `test.h5`: 20 trajectories, 80 time steps, node counts roughly 20k to 23k.
- Extracted test cases include `Bump.jl`, `Bump.msh`, `Bump.inp`, `Mach.txt`, `params.txt`, and `outFO/sol_0.vtu` through `sol_80.vtu`.
- The HDF5 files expose graph nodes and edges but do not expose explicit mesh weights, cell areas, face normals, or face lengths as top-level keys.

## Validated Commands

Local Windows branch tests passed previously with Miniforge Python:

```bash
python -m pytest tests/time_dependent_no
python scripts/time_dependent_no/run_euler_fixture_diagnostics.py
```

On AutoDL, `h5py` was installed into the base Python for schema inspection, and the branch inspector generated small JSON summaries under ignored `artifacts/time_dependent_no/`.

## Next Small Step

1. Add a real-data smoke script that loads one HDF5 trajectory frame through `make_cpg_graph_frame` and prints tensor/array shapes without training.
2. Decide the first adapter path:
   - PCNO/MPCNO native graph/point-cloud adapter, or
   - FNO remeshing sanity pipeline.
3. Before training, define the first baseline gate and output artifact convention.

## Open Questions

- Should the first learned baseline be PCNO/MPCNO, since the HDF5 graph representation is already validated, or FNO, to quantify remeshing artifacts first?
- Should conservation diagnostics use approximate node weights initially, or should we first compute geometric weights from `Bump.msh` / `Bump.inp`?
- Is there a separate forward-facing-step dataset, or is `forward_300` only a mislabeled bump folder?

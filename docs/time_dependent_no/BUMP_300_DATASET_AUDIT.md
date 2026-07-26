# Supersonic Bump Dataset Audit

Date: 2026-06-25
Status: Frozen dataset-identity and raw-bundle provenance

This record preserves the first inspection that identified the copied CPG-style
Euler bundle. It is not the live schema definition or an experiment queue. Use
`CPG_EULER_DATASET_CONTRACT.md` for required keys, shapes, node types, and the
active reader convention. The exact AutoDL path remains in ignored
`LOCAL_CONTEXT.md`; committed documentation intentionally contains no SSH or
credential details.

## Dataset Identity

The local folder label was `forward_300`, but the recovered case files identify
the bundle as the supersonic bump dataset, not a forward-facing-step dataset:

- case scripts are named `Bump.jl`;
- meshes are `Bump.msh` and `Bump.inp`;
- case parameters are `chordLength` and `heightFrac`;
- every extracted test case includes `Mach.txt`; and
- saved VTU snapshots run from `outFO/sol_0.vtu` through `sol_80.vtu`.

## Reference-Solver Provenance

The representative recovered Trixi case records:

| Item | Observed value |
| --- | --- |
| equations | `CompressibleEulerEquations2D` |
| `gamma` | `1.4` |
| freestream state | `rho_inf = 1.4`, `p_inf = 1.0` |
| polynomial degree | `3` |
| physical interval | `(0.0, 2.0)` |
| saved-snapshot interval | `0.025` |
| volume flux | `flux_ranocha` with shock-capturing volume integral |
| surface flux | `flux_lax_friedrichs` |
| limiter | Zhang-Shu positivity limiter |
| boundaries | left inflow, right outflow, top/bottom/wall slip walls |

These facts identify the recovered bundle and its solver lineage. They do not
establish exact parity with every paper dataset, split, checkpoint, or evaluator.

## Raw-Bundle Provenance

| Item | Observed size or count |
| --- | ---: |
| complete copied folder | about `42 GB` |
| `train.h5` | about `35 GB` |
| `test.h5` | about `2.3 GB` |
| extracted `test/` tree | about `5.6 GB` |
| training trajectories | `300` |
| test trajectories | `20` |
| extracted VTU files per test case | `81`, indexed `0..80` |

Do not commit the raw HDF5 files, extracted VTU tree, checkpoints, or large logs.

## Generated Schema Summaries

The original inspection wrote two small summaries under the ignored artifact
root:

```text
artifacts/time_dependent_no/forward_300_train_schema_metadata.json
artifacts/time_dependent_no/forward_300_test_schema_metadata.json
```

These paths are historical provenance. The live schema contract is
`CPG_EULER_DATASET_CONTRACT.md`.

## Conservation Boundary

The HDF5 bundle exposes graph nodes and edges but does not itself provide a
validated control-volume map, cell volumes, physical face measures, unit
normals, oriented face connectivity, or reference boundary exchange. Equal-node
or approximate-weight sums on the bump graph are proxy diagnostics only. They
must not be reported as physical conservation, conservative flux closure, or
accurate boundary exchange. Such claims require independently recovered geometry
plus a validated mesh-to-graph and orientation contract.

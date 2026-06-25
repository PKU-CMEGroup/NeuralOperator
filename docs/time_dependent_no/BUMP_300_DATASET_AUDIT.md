# Supersonic Bump Dataset Audit

Date: 2026-06-25

This audit records the first schema inspection of the CPG-style Euler dataset copied to AutoDL under the local folder label `forward_300`. The committed documentation intentionally avoids SSH details and credentials. The exact machine path is recorded in the ignored `LOCAL_CONTEXT.md` on the AutoDL clone.

## High-Level Finding

The folder label is `forward_300`, but the extracted case files indicate this is the **supersonic bump** dataset, not the forward-facing-step dataset:

- extracted case scripts are named `Bump.jl`;
- meshes are `Bump.msh` and `Bump.inp`;
- case parameters are `chordLength` and `heightFrac`;
- each test case has `Mach.txt` and `outFO/sol_0.vtu` through `sol_80.vtu`.

The Trixi setup in a representative case uses:

| Item | Observed value |
| --- | --- |
| equations | `CompressibleEulerEquations2D` |
| `gamma` | `1.4` |
| freestream density | `rho_inf = 1.4` |
| freestream pressure | `p_inf = 1.0` |
| polynomial degree | `polydeg = 3` |
| time span | `(0.0, 2.0)` |
| saved solution interval | `dt = 0.025` |
| saved snapshots in extracted test case | `sol_0.vtu` through `sol_80.vtu` |
| volume flux | `flux_ranocha` with shock-capturing volume integral |
| surface flux | `flux_lax_friedrichs` |
| limiter | Zhang-Shu positivity limiter |
| boundaries | left inflow, right outflow, top/bottom/wall slip walls |

## Files Inspected

The copied dataset contains:

| Item | Observed size / count |
| --- | --- |
| full folder | about `42 GB` |
| `train.h5` | about `35 GB` |
| `test.h5` | about `2.3 GB` |
| extracted `test/` tree | about `5.6 GB` |
| extracted test cases | `20` directories |
| extracted VTU files per test case | `81` files, indexed `0..80` |

## HDF5 Keys

Both `train.h5` and `test.h5` expose the expected CPG keys in sampled groups:

```text
Mach, edges, node_type, pos, pres, rho, v1, v2
```

No missing required keys were found in the schema pass.

## Train HDF5 Summary

| Item | Observed value |
| --- | --- |
| trajectory groups | `300` |
| time steps | always `80` |
| node count range | `19345 .. 23359` |
| median node count | `21348` |
| edge count range | `38320 .. 46346` |
| median edge count | `42325` |

Representative first group, `0`:

| Key | Shape | Dtype | Chunks | Compression |
| --- | --- | --- | --- | --- |
| `pos` | `(80, 19531, 2)` | `float32` | `(5, 2442, 1)` | `gzip` |
| `edges` | `(80, 38692, 2)` | `int64` | none | none |
| `node_type` | `(80, 19531, 1)` | `int32` | none | none |
| `pres` | `(80, 19531, 1)` | `float64` | `(5, 1221, 1)` | `gzip` |
| `rho` | `(80, 19531, 1)` | `float64` | `(5, 1221, 1)` | `gzip` |
| `v1` | `(80, 19531, 1)` | `float64` | `(5, 1221, 1)` | `gzip` |
| `v2` | `(80, 19531, 1)` | `float64` | `(5, 1221, 1)` | `gzip` |
| `Mach` | `(80, 19531, 1)` | `float64` | none | none |

Representative ranges in first train group:

| Variable | Min | Max | Mean |
| --- | ---: | ---: | ---: |
| `rho` | `0.364292` | `8.01395` | `1.92995` |
| `pres` | `0.0910182` | `14.6251` | `2.2548` |
| `v1` | `-0.0318888` | `3.44238` | `2.37035` |
| `v2` | `-1.0641` | `1.70946` | `0.0332165` |
| `Mach` | `2.694` | `2.694` | `2.694` |

## Test HDF5 Summary

| Item | Observed value |
| --- | --- |
| trajectory groups | `20` |
| time steps | always `80` |
| node count range | `19529 .. 23417` |
| median node count | `21732` |
| edge count range | `38688 .. 46462` |
| median edge count | `43093` |

Representative first group, `00`:

| Key | Shape | Dtype | Chunks | Compression |
| --- | --- | --- | --- | --- |
| `pos` | `(80, 22465, 2)` | `float32` | `(5, 1405, 1)` | `gzip` |
| `edges` | `(80, 44560, 2)` | `int64` | none | none |
| `node_type` | `(80, 22465, 1)` | `int32` | none | none |
| `pres` | `(80, 22465, 1)` | `float64` | `(3, 1405, 1)` | `gzip` |
| `rho` | `(80, 22465, 1)` | `float64` | `(3, 1405, 1)` | `gzip` |
| `v1` | `(80, 22465, 1)` | `float64` | `(3, 1405, 1)` | `gzip` |
| `v2` | `(80, 22465, 1)` | `float64` | `(3, 1405, 1)` | `gzip` |
| `Mach` | `(80, 22465, 1)` | `float64` | none | none |

Representative ranges in first test group:

| Variable | Min | Max | Mean |
| --- | ---: | ---: | ---: |
| `rho` | `0.517745` | `7.55851` | `1.71307` |
| `pres` | `0.553485` | `16.3726` | `1.75278` |
| `v1` | `-0.0172994` | `3.1559` | `2.65928` |
| `v2` | `-0.848151` | `1.62526` | `0.0149664` |
| `Mach` | `2.888` | `2.888` | `2.888` |

## Node-Type Convention Check

The observed HDF5 node-type codes match the expected CPG convention:

| Code | Meaning |
| --- | --- |
| `0` | normal interior node |
| `1` | wall |
| `2` | outflow |
| `3` | inflow |

The first train group has per-frame counts equivalent to:

```text
normal: 18795
wall: 578
outflow: 65
inflow: 93
```

The first test group has per-frame counts equivalent to:

```text
normal: 21729
wall: 566
outflow: 77
inflow: 93
```

## Generated Local Artifacts

The AutoDL schema inspection generated small JSON summaries under the ignored artifact directory:

```text
artifacts/time_dependent_no/forward_300_train_schema_metadata.json
artifacts/time_dependent_no/forward_300_test_schema_metadata.json
```

Do not commit raw HDF5 files, extracted VTU files, checkpoints, or large logs.

## Immediate Implications

- The branch dataset reader contract is validated for this copied dataset.
- The first baseline should treat this as the supersonic bump task unless a separate forward-step dataset is provided.
- The dataset currently exposes graph nodes and edges but no explicit cell areas, face normals, or face lengths in the HDF5 keys. Conservation diagnostics should therefore start with equal-node or approximate weights and clearly label that limitation until physical geometric factors are recovered from mesh files or computed separately.
- FNO remeshing sanity remains necessary before any FNO failure claim.
- PCNO/MPCNO adapter work can begin from the HDF5 graph representation without waiting for raw Trixi regeneration.

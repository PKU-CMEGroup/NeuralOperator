# CPG Euler Dataset Contract

The first benchmark source is the CPG-style data used by the Structure-Preserving Graph Neural Solver for 2D Euler / hyperbolic conservation laws.

## Required HDF5 Keys

Each HDF5 file should contain one group per trajectory. The active contract expects each trajectory group to expose:

| Key | Meaning | Expected shape |
| --- | --- | --- |
| `pos` | node coordinates | `(T, N, dim)` or static `(N, dim)` |
| `edges` | graph edges | `(T, E, 2)` or static `(E, 2)` |
| `node_type` | node labels | `(T, N, 1)`, `(T, N)`, `(N, 1)`, or `(N,)` |
| `rho` | density | `(T, N, 1)` or `(T, N)` |
| `v1` | x velocity | `(T, N, 1)` or `(T, N)` |
| `v2` | y velocity | `(T, N, 1)` or `(T, N)` |
| `pres` | pressure | `(T, N, 1)` or `(T, N)` |
| `Mach` | freestream condition broadcast to nodes | temporal or static node column |

Primitive variable order is `[rho, v1, v2, pres]`.
Conservative variable order is `[rho, rho_v1, rho_v2, energy]`.
Use `gamma = 1.4` unless a dataset-specific config says otherwise.

## Observed Supersonic Bump Dataset

The first copied dataset was inspected on 2026-06-25. The local folder label is `forward_300`, but the extracted test cases are `Bump.*` files with `chordLength` and `heightFrac` parameters, so this should be treated as the supersonic bump dataset unless a separate forward-step dataset is provided.

Observed HDF5 facts:

| File | Groups | Time steps | Node-count range | Edge-count range |
| --- | ---: | ---: | ---: | ---: |
| `train.h5` | `300` | `80` | `19345 .. 23359` | `38320 .. 46346` |
| `test.h5` | `20` | `80` | `19529 .. 23417` | `38688 .. 46462` |

Observed keys are exactly:

```text
Mach, edges, node_type, pos, pres, rho, v1, v2
```

Representative group shapes:

```text
pos       (80, N, 2)
edges     (80, E, 2)
node_type (80, N, 1)
rho       (80, N, 1)
v1        (80, N, 1)
v2        (80, N, 1)
pres      (80, N, 1)
Mach      (80, N, 1)
```

The HDF5 schema does not expose explicit cell areas, face normals, edge lengths, or cell volumes as top-level keys. Conservation diagnostics should state whether they use equal-node weights, approximate geometric weights, or recovered mesh weights.

## Node Types

| Code | Meaning |
| --- | --- |
| `0` | normal interior node |
| `1` | wall |
| `2` | outflow |
| `3` | inflow |

Boundary diagnostics must distinguish clamped-boundary reproduction from free or prescribed-boundary rollout.

## Reference Graph Frame

For frame `t`, the reference reader convention is:

```text
x = [node_type, rho_t, v1_t, v2_t, pres_t, Mach]
y = [rho_{t+1}, v1_{t+1}, v2_{t+1}, pres_{t+1}]
pos = pos_t
edges = edges_t
future_primitives = primitive states from t+1 through t+num_steps
```

## Inspection Convention

Use the real dataset path from ignored local context. The active reusable API is `utility.time_dependent_no.euler2d.inspect_cpg_hdf5_file`; one-off inspection CLIs were removed from the tracked tree after the schema facts above were recorded.

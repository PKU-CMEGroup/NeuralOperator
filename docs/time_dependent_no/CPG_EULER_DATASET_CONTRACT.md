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

## Inspection Command

```bash
python scripts/time_dependent_no/inspect_cpg_euler_dataset.py /path/to/train.h5 --max-trajectories 2 --output artifacts/time_dependent_no/train_schema.json
```

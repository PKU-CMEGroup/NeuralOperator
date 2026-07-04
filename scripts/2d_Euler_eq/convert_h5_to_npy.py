#!/usr/bin/env python3
"""
Export an FPC-style train.h5 file to three .npy files:

  1. grid.npy     : node coordinates, shape [N, dim]
  2. elems.npy    : element connectivity, shape [M, max_nodes_per_elem + 1]
  3. features.npy : primitive variables in order [rho, u, v, p]

Each row in elems.npy is:
  [elem_dim, node_0, node_1, ...]

For example, a 2D triangle is:
  [2, i, j, k]

For dataset/fpcMulti.py files, features are read from:
  rho, v1, v2, pres

For dataset/fpc.py files, features are read from:
  density, velocity[..., 0], velocity[..., 1], pressure

The HDF5 files used by the dataset loaders are trajectory collections:
the top-level keys are trajectory groups, and each trajectory contains
time-indexed arrays such as pos, edges/cells, rho, v1, v2, pres.
"""

import argparse
import os
import re
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np


FEATURE_ORDER = ("rho", "u", "v", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FPC-style train.h5 trajectories to grid/cells/features .npy files."
    )
    parser.add_argument("--input", required=True, help="Path to train.h5")
    parser.add_argument(
        "--output-dir",
        default="exported_npy",
        help="Directory where .npy files will be written",
    )
    parser.add_argument(
        "--trajectory",
        default="0",
        help=(
            "Trajectory to export. Use an integer index among top-level groups "
            "or an exact group key. Default: 0"
        ),
    )
    parser.add_argument(
        "--all-trajectories",
        action="store_true",
        help="Export every top-level trajectory into output-dir/<trajectory_key>/",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Time frame used for grid/cell arrays when those arrays are time-indexed.",
    )
    parser.add_argument(
        "--feature-frame",
        type=int,
        default=None,
        help=(
            "If set, export features only at this frame with shape [N, 4]. "
            "By default, all feature frames are exported with shape [T, N, 4]."
        ),
    )
    parser.add_argument("--grid-name", default="grid.npy", help="Output filename for coordinates.")
    parser.add_argument("--elems-name", default="elems.npy", help="Output filename for elements.")
    parser.add_argument(
        "--cells-name",
        default=None,
        help="Legacy alias for --elems-name. If set, this output filename is used.",
    )
    parser.add_argument("--feature-name", default="features.npy", help="Output filename for features.")
    parser.add_argument(
        "--connectivity-key",
        default=None,
        help="Explicit HDF5 dataset name for connectivity, e.g. cells, elems, or edges.",
    )
    parser.add_argument(
        "--elem-dim",
        type=int,
        default=None,
        help=(
            "Element dimensionality to write in the first column. "
            "If omitted, it is inferred from node count and coordinate dimension."
        ),
    )
    parser.add_argument(
        "--padding-value",
        type=int,
        default=-1,
        help="Padding value for unused node slots in mixed/padded connectivity arrays.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the HDF5 tree before exporting.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print the HDF5 tree; do not export .npy files.",
    )
    return parser.parse_args()


def print_h5_tree(path: str) -> None:
    def visitor(name: str, obj: h5py.Dataset) -> None:
        indent = "  " * name.count("/")
        short_name = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- {short_name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}+ {short_name}/")

    print(f"HDF5 file: {path}")
    with h5py.File(path, "r") as f:
        if not f.keys():
            print("(empty file)")
            return
        f.visititems(visitor)


def has_keys(group: h5py.Group, keys: Iterable[str]) -> bool:
    return all(key in group for key in keys)


def resolve_trajectory_group(h5: h5py.File, trajectory: str) -> Tuple[str, h5py.Group]:
    root_field_names = {
        "pos",
        "edges",
        "cells",
        "node_type",
        "rho",
        "v1",
        "v2",
        "pres",
        "velocity",
        "pressure",
        "density",
    }

    if any(key in h5 for key in root_field_names):
        return ".", h5

    keys = list(h5.keys())
    if not keys:
        raise ValueError("The HDF5 file has no top-level groups or datasets.")

    if trajectory in h5:
        obj = h5[trajectory]
        if not isinstance(obj, h5py.Group):
            raise ValueError(f"Top-level key '{trajectory}' is not a trajectory group.")
        return trajectory, obj

    try:
        index = int(trajectory)
    except ValueError as exc:
        raise KeyError(
            f"Trajectory '{trajectory}' not found. Available top-level keys: {keys[:20]}"
        ) from exc

    if index < 0 or index >= len(keys):
        raise IndexError(
            f"Trajectory index {index} is out of range for {len(keys)} top-level groups."
        )

    key = keys[index]
    obj = h5[key]
    if not isinstance(obj, h5py.Group):
        raise ValueError(f"Top-level key '{key}' is not a trajectory group.")
    return key, obj


def safe_dir_name(name: str) -> str:
    if name == ".":
        return "root"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "trajectory"


def read_static_or_frame(dataset: h5py.Dataset, frame: int, field_name: str) -> np.ndarray:
    if dataset.ndim >= 3:
        if frame < 0 or frame >= dataset.shape[0]:
            raise IndexError(
                f"Frame {frame} is out of range for '{field_name}' with shape {dataset.shape}."
            )
        return np.asarray(dataset[frame])
    return np.asarray(dataset[()])


def read_feature_dataset(
    dataset: h5py.Dataset,
    *,
    n_nodes: int,
    frame: Optional[int],
    field_name: str,
) -> np.ndarray:
    if frame is None:
        return np.asarray(dataset[()])

    if dataset.ndim >= 3:
        if frame < 0 or frame >= dataset.shape[0]:
            raise IndexError(
                f"Feature frame {frame} is out of range for '{field_name}' with shape {dataset.shape}."
            )
        return np.asarray(dataset[frame])

    # Covers scalar fields stored as [T, N] rather than [T, N, 1].
    if dataset.ndim == 2 and dataset.shape[0] != n_nodes and dataset.shape[1] == n_nodes:
        if frame < 0 or frame >= dataset.shape[0]:
            raise IndexError(
                f"Feature frame {frame} is out of range for '{field_name}' with shape {dataset.shape}."
            )
        return np.asarray(dataset[frame])

    return np.asarray(dataset[()])


def scalar_feature_to_channel(arr: np.ndarray, *, n_nodes: int, field_name: str) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 1 and arr.shape[0] == n_nodes:
        return arr[:, None]

    if arr.ndim >= 2 and arr.shape[-1] == 1:
        return arr

    # Scalar time series sometimes appear as [T, N].
    if arr.ndim == 2 and arr.shape[1] == n_nodes:
        return arr[..., None]

    raise ValueError(
        f"Cannot interpret '{field_name}' as a scalar node feature. Got shape {arr.shape}."
    )


def velocity_to_uv(velocity: np.ndarray, *, field_name: str = "velocity") -> Tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(velocity)
    if velocity.ndim < 2 or velocity.shape[-1] < 2:
        raise ValueError(
            f"Cannot split '{field_name}' into u/v. Expected last dimension >= 2, got {velocity.shape}."
        )
    return velocity[..., 0:1], velocity[..., 1:2]


def load_grid(group: h5py.Group, frame: int) -> np.ndarray:
    if "pos" not in group:
        raise KeyError("Missing required coordinate dataset 'pos'.")

    grid = read_static_or_frame(group["pos"], frame, "pos").astype(np.float32, copy=False)
    if grid.ndim != 2:
        raise ValueError(f"Expected grid coordinates with shape [N, dim], got {grid.shape}.")
    return grid


def load_connectivity(
    group: h5py.Group,
    frame: int,
    connectivity_key: Optional[str],
) -> Tuple[str, np.ndarray]:
    if connectivity_key is not None:
        if connectivity_key not in group:
            raise KeyError(
                f"Connectivity dataset '{connectivity_key}' not found. "
                f"Available keys: {list(group.keys())}"
            )
        search_keys = (connectivity_key,)
    else:
        search_keys = ("elems", "elements", "cells", "edges", "faces", "face")

    for key in search_keys:
        if key in group:
            connectivity = read_static_or_frame(group[key], frame, key).astype(np.int32, copy=False)
            if connectivity.ndim != 2:
                raise ValueError(
                    f"Expected connectivity '{key}' with shape [M, nodes_per_cell], "
                    f"got {connectivity.shape}."
                )
            return key, connectivity

    raise KeyError(
        "Missing connectivity dataset. Expected one of: elems, elements, cells, edges, faces, face."
    )


def infer_elem_dim(node_count: int, grid_dim: int) -> int:
    if node_count <= 1:
        raise ValueError(f"Cannot infer element dimension from node count {node_count}.")
    if node_count == 2:
        return 1
    if node_count == 3:
        return 2
    if node_count == 4:
        return 2 if grid_dim <= 2 else 3
    return 3 if grid_dim >= 3 else 2


def connectivity_to_elems(
    connectivity: np.ndarray,
    *,
    grid_dim: int,
    elem_dim: Optional[int],
    padding_value: int,
) -> np.ndarray:
    connectivity = np.asarray(connectivity, dtype=np.int32)
    nelems, max_nodes_per_elem = connectivity.shape

    elems = np.full(
        (nelems, max_nodes_per_elem + 1),
        padding_value,
        dtype=np.int32,
    )

    if elem_dim is not None:
        elems[:, 0] = elem_dim
    else:
        for row_index, row in enumerate(connectivity):
            node_count = int(np.count_nonzero(row != padding_value))
            elems[row_index, 0] = infer_elem_dim(node_count, grid_dim)

    elems[:, 1:] = connectivity
    return elems


def load_features(
    group: h5py.Group,
    *,
    n_nodes: int,
    feature_frame: Optional[int],
) -> Tuple[str, np.ndarray]:
    if has_keys(group, ("rho", "v1", "v2", "pres")):
        rho = read_feature_dataset(group["rho"], n_nodes=n_nodes, frame=feature_frame, field_name="rho")
        u = read_feature_dataset(group["v1"], n_nodes=n_nodes, frame=feature_frame, field_name="v1")
        v = read_feature_dataset(group["v2"], n_nodes=n_nodes, frame=feature_frame, field_name="v2")
        p = read_feature_dataset(group["pres"], n_nodes=n_nodes, frame=feature_frame, field_name="pres")

        parts = [
            scalar_feature_to_channel(rho, n_nodes=n_nodes, field_name="rho"),
            scalar_feature_to_channel(u, n_nodes=n_nodes, field_name="v1"),
            scalar_feature_to_channel(v, n_nodes=n_nodes, field_name="v2"),
            scalar_feature_to_channel(p, n_nodes=n_nodes, field_name="pres"),
        ]
        return "rho/v1/v2/pres", np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    if has_keys(group, ("density", "velocity", "pressure")):
        rho = read_feature_dataset(
            group["density"], n_nodes=n_nodes, frame=feature_frame, field_name="density"
        )
        velocity = read_feature_dataset(
            group["velocity"], n_nodes=n_nodes, frame=feature_frame, field_name="velocity"
        )
        p = read_feature_dataset(
            group["pressure"], n_nodes=n_nodes, frame=feature_frame, field_name="pressure"
        )
        u, v = velocity_to_uv(velocity)

        parts = [
            scalar_feature_to_channel(rho, n_nodes=n_nodes, field_name="density"),
            scalar_feature_to_channel(u, n_nodes=n_nodes, field_name="velocity[..., 0]"),
            scalar_feature_to_channel(v, n_nodes=n_nodes, field_name="velocity[..., 1]"),
            scalar_feature_to_channel(p, n_nodes=n_nodes, field_name="pressure"),
        ]
        return "density/velocity/pressure", np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    raise KeyError(
        "Cannot find primitive feature datasets. Expected either "
        "rho/v1/v2/pres or density/velocity/pressure."
    )


def export_one_trajectory(
    group_key: str,
    group: h5py.Group,
    *,
    output_dir: str,
    frame: int,
    feature_frame: Optional[int],
    grid_name: str,
    elems_name: str,
    feature_name: str,
    connectivity_key: Optional[str],
    elem_dim: Optional[int],
    padding_value: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    grid = load_grid(group, frame)
    source_connectivity_key, connectivity = load_connectivity(group, frame, connectivity_key)
    elems = connectivity_to_elems(
        connectivity,
        grid_dim=grid.shape[1],
        elem_dim=elem_dim,
        padding_value=padding_value,
    )
    feature_source, features = load_features(group, n_nodes=grid.shape[0], feature_frame=feature_frame)

    grid_path = os.path.join(output_dir, grid_name)
    elems_path = os.path.join(output_dir, elems_name)
    feature_path = os.path.join(output_dir, feature_name)

    np.save(grid_path, grid)
    np.save(elems_path, elems)
    np.save(feature_path, features)

    feature_shape_note = "[T, N, 4]" if feature_frame is None and features.ndim == 3 else "[N, 4]"
    print(f"Exported trajectory: {group_key}")
    print(f"  grid       -> {grid_path} shape={grid.shape}")
    print(
        f"  elems      -> {elems_path} shape={elems.shape} "
        f"(source dataset: {source_connectivity_key}, first column: elem_dim)"
    )
    print(
        f"  features   -> {feature_path} shape={features.shape} {feature_shape_note} "
        f"order={FEATURE_ORDER} (source: {feature_source})"
    )


def main() -> None:
    args = parse_args()

    if args.inspect or args.inspect_only:
        print_h5_tree(args.input)
        if args.inspect_only:
            return
        print("")

    with h5py.File(args.input, "r") as h5:
        if args.all_trajectories:
            root_field_names = {"pos", "cells", "edges", "rho", "v1", "v2", "pres", "velocity"}
            if any(key in h5 for key in root_field_names):
                groups = [resolve_trajectory_group(h5, args.trajectory)]
            else:
                groups = [(key, h5[key]) for key in h5.keys() if isinstance(h5[key], h5py.Group)]

            if not groups:
                raise ValueError("No trajectory groups found to export.")

            for group_key, group in groups:
                traj_output_dir = os.path.join(args.output_dir, safe_dir_name(group_key))
                export_one_trajectory(
                    group_key,
                    group,
                    output_dir=traj_output_dir,
                    frame=args.frame,
                    feature_frame=args.feature_frame,
                    grid_name=args.grid_name,
                    elems_name=args.cells_name or args.elems_name,
                    feature_name=args.feature_name,
                    connectivity_key=args.connectivity_key,
                    elem_dim=args.elem_dim,
                    padding_value=args.padding_value,
                )
        else:
            group_key, group = resolve_trajectory_group(h5, args.trajectory)
            export_one_trajectory(
                group_key,
                group,
                output_dir=args.output_dir,
                frame=args.frame,
                feature_frame=args.feature_frame,
                grid_name=args.grid_name,
                elems_name=args.cells_name or args.elems_name,
                feature_name=args.feature_name,
                connectivity_key=args.connectivity_key,
                elem_dim=args.elem_dim,
                padding_value=args.padding_value,
            )


if __name__ == "__main__":
    main()

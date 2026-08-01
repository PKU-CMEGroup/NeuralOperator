from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.time_dependent_no.prepare_pcno_euler2d_shards import main as prepare_main


def _write_boundary_trajectory(group: h5py.Group, trajectory_index: int) -> None:
    num_steps = 4
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    perimeter = np.asarray(
        [[0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0]],
        dtype=np.int64,
    )
    spokes = np.asarray(
        [[4, node] for node in (0, 1, 2, 3, 5, 6, 7, 8)], dtype=np.int64
    )
    edges = np.concatenate((perimeter, spokes), axis=0)
    node_type = np.asarray([3, 1, 2, 3, 0, 2, 3, 1, 2], dtype=np.int64)[:, None]
    mach_value = 1.6 + 0.05 * trajectory_index
    mach = np.full((nodes.shape[0], 1), mach_value, dtype=np.float32)

    group.create_dataset("pos", data=np.repeat(nodes[None, ...], num_steps, axis=0))
    group.create_dataset("edges", data=np.repeat(edges[None, ...], num_steps, axis=0))
    group.create_dataset(
        "node_type", data=np.repeat(node_type[None, ...], num_steps, axis=0)
    )
    group.create_dataset("Mach", data=np.repeat(mach[None, ...], num_steps, axis=0))

    time = np.arange(num_steps, dtype=np.float32)[:, None, None]
    x = nodes[None, :, 0:1]
    y = nodes[None, :, 1:2]
    group.create_dataset(
        "rho", data=1.4 + 0.01 * trajectory_index + 0.002 * time + 0.001 * x
    )
    group.create_dataset("v1", data=mach_value + 0.01 * time + 0.005 * x)
    group.create_dataset("v2", data=0.02 * (y - 0.5) + 0.001 * time)
    group.create_dataset("pres", data=1.0 + 0.004 * time + 0.002 * x)


def prepare_boundary_synthetic_shards(tmp_path: Path) -> Path:
    source = tmp_path / "raw_boundary.h5"
    with h5py.File(source, "w") as handle:
        _write_boundary_trajectory(handle.create_group("0"), 0)
        _write_boundary_trajectory(handle.create_group("1"), 1)
    output = tmp_path / "boundary_shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(output),
            "--static-check",
            "all",
        ]
    )
    return output

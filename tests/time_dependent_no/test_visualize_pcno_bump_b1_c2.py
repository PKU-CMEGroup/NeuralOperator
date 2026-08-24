from __future__ import annotations

import math

import numpy as np

from scripts.time_dependent_no import visualize_pcno_bump_b1_c2 as visualization


def _curve_rows() -> list[dict[str, str]]:
    rows = []
    for architecture_index, architecture in enumerate(visualization.ARCHITECTURES):
        for seed_index, seed in enumerate(visualization.SEEDS):
            for count in visualization.TRAJECTORY_COUNTS:
                for step in (256, 512):
                    value = (
                        1.0
                        + architecture_index
                        + seed_index / 10
                        + count / 1000
                        + step / 10000
                    )
                    rows.append(
                        {
                            "architecture": architecture,
                            "seed": str(seed),
                            "trajectory_count": str(count),
                            "optimizer_step": str(step),
                            "metric": str(value),
                        }
                    )
    return rows


def test_surface_matrix_averages_three_seeds_without_interpolation() -> None:
    steps, matrix = visualization.surface_matrix(_curve_rows(), "pcno", "metric")

    assert steps == [256, 512]
    assert matrix.shape == (2, 6)
    assert matrix[0, 0] == np.mean(
        [1.0 + seed / 10 + 0.008 + 0.0256 for seed in range(3)]
    )


def test_ratio_surface_uses_geometric_seed_mean() -> None:
    rows = []
    for seed_index, seed in enumerate(visualization.SEEDS):
        for count in visualization.TRAJECTORY_COUNTS:
            rows.append(
                {
                    "seed": str(seed),
                    "trajectory_count": str(count),
                    "optimizer_step": "256",
                    "pcno_over_pcfno_metric": str(2.0 ** (seed_index - 1)),
                }
            )

    steps, matrix = visualization.ratio_surface_matrix(rows, "metric")

    assert steps == [256]
    assert matrix.shape == (1, 6)
    assert bool(np.allclose(matrix, 0.0))


def test_grid_edges_preserve_cell_centers() -> None:
    edges = visualization._grid_edges([1.0, 2.0, 4.0])

    assert edges.tolist() == [0.5, 1.5, 3.0, 5.0]
    assert math.isclose((edges[0] + edges[1]) / 2.0, 1.0)

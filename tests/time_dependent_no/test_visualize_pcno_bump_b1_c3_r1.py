from __future__ import annotations

import numpy as np

from scripts.time_dependent_no import visualize_pcno_bump_b1_c3_r1 as visualization


def _cell_rows() -> list[dict[str, str]]:
    rows = []
    for precision_index, precision in enumerate(visualization.PRECISIONS):
        for seed_index, seed in enumerate(visualization.SEEDS):
            for count in visualization.COUNTS:
                for architecture_index, architecture in enumerate(
                    visualization.ARCHITECTURES
                ):
                    value = 1.0 + precision_index + seed_index + architecture_index
                    rows.append(
                        {
                            "precision": precision,
                            "seed": str(seed),
                            "trajectory_count": str(count),
                            "architecture": architecture,
                            "metric": str(value),
                        }
                    )
    return rows


def test_grouped_values_preserve_three_seed_observations() -> None:
    means, stds, raw = visualization.grouped_values(
        _cell_rows(), precision="bf16", architecture="pcno", metric="metric"
    )

    assert means.tolist() == [2.0] * len(visualization.COUNTS)
    assert stds.tolist() == [1.0] * len(visualization.COUNTS)
    assert set(raw) == set(visualization.SEEDS)


def test_architecture_ratios_use_geometric_three_seed_mean() -> None:
    rows = []
    for seed_index, seed in enumerate(visualization.SEEDS):
        for count in visualization.COUNTS:
            rows.append(
                {
                    "precision": "bf16",
                    "seed": str(seed),
                    "trajectory_count": str(count),
                    "metric": "rollout",
                    "pcno_over_pcfno": str(2.0**seed_index),
                }
            )

    geometric, raw = visualization.architecture_ratio_values(
        rows, "rollout", "bf16"
    )

    assert bool(np.allclose(geometric, 2.0))
    assert raw[visualization.SEEDS[0]].tolist() == [1.0] * len(
        visualization.COUNTS
    )


def test_surface_matrix_preserves_step_count_grid_and_seed_mean() -> None:
    rows = []
    for seed_index, seed in enumerate(visualization.SEEDS):
        for step in (256, 512):
            for count in visualization.COUNTS:
                rows.append(
                    {
                        "seed": str(seed),
                        "architecture": "pcno",
                        "optimizer_step": str(step),
                        "trajectory_count": str(count),
                        "metric": str(step + count + seed_index),
                    }
                )

    steps, matrix = visualization.surface_matrix(rows, "pcno", "metric")

    assert steps.tolist() == [256, 512]
    assert matrix.shape == (2, len(visualization.COUNTS))
    assert matrix[0, 0] == 256 + visualization.COUNTS[0] + 1

from __future__ import annotations

import numpy as np

from scripts.time_dependent_no import visualize_pcno_bump_b1_c3 as visualization


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


def test_grouped_values_preserve_two_seed_observations() -> None:
    means, stds, raw = visualization.grouped_values(
        _cell_rows(), precision="bf16", architecture="pcno", metric="metric"
    )

    assert means.tolist() == [1.5] * len(visualization.COUNTS)
    assert bool(np.allclose(stds, np.sqrt(0.5)))
    assert raw[visualization.SEEDS[0]].tolist() == [1.0] * len(
        visualization.COUNTS
    )
    assert raw[visualization.SEEDS[1]].tolist() == [2.0] * len(
        visualization.COUNTS
    )


def test_architecture_ratio_values_use_geometric_seed_mean() -> None:
    rows = []
    for seed_index, seed in enumerate(visualization.SEEDS):
        for count in visualization.COUNTS:
            rows.append(
                {
                    "precision": "bf16",
                    "seed": str(seed),
                    "trajectory_count": str(count),
                    "metric": "outside_rollout_h79_relative_l2",
                    "pcno_over_pcfno": str(2.0 ** (2 * seed_index)),
                }
            )

    geometric, raw = visualization.architecture_ratio_values(
        rows, "outside_rollout_h79_relative_l2", "bf16"
    )

    assert bool(np.allclose(geometric, 2.0))
    assert raw[visualization.SEEDS[0]].tolist() == [1.0] * len(
        visualization.COUNTS
    )
    assert raw[visualization.SEEDS[1]].tolist() == [4.0] * len(
        visualization.COUNTS
    )

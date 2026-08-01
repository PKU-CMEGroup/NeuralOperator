from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.decompose_pcno_euler2d_rollout_error import (
    aggregate_rows,
    validate_args,
    weighted_decomposition_metrics,
)


def test_weighted_decomposition_metrics_closes_vector_and_energy_identities() -> None:
    propagated = np.full((4, 2), 2.0)
    fresh = np.ones((4, 2))
    metrics = weighted_decomposition_metrics(
        propagated + fresh,
        propagated,
        fresh,
        np.ones(4),
    )

    assert metrics["status"] == "available"
    assert metrics["propagated_magnitude_share"] == pytest.approx(2.0 / 3.0)
    assert metrics["propagated_fresh_cosine"] == pytest.approx(1.0)
    assert metrics["relative_reconstruction_residual"] < 1.0e-14
    assert metrics["relative_energy_identity_residual"] < 1.0e-14
    assert (
        metrics["propagated_energy_fraction_of_total"]
        + metrics["fresh_defect_energy_fraction_of_total"]
        + metrics["cross_energy_fraction_of_total"]
    ) == pytest.approx(1.0)


def test_aggregate_rows_keeps_endpoint_population_explicit() -> None:
    region = {
        "total_norm": 3.0,
        "propagated_norm": 2.0,
        "fresh_defect_norm": 1.0,
        "propagated_magnitude_share": 2.0 / 3.0,
        "propagated_fresh_cosine": 1.0,
        "propagated_energy_fraction_of_total": 4.0 / 9.0,
        "fresh_defect_energy_fraction_of_total": 1.0 / 9.0,
        "cross_energy_fraction_of_total": 4.0 / 9.0,
    }
    rows = [
        {
            "call_index": call,
            "regions": {
                name: dict(region)
                for name in (
                    "all_nodes_full",
                    "normal_nodes_full",
                    "boundary_nodes_full",
                    "smooth_full",
                    "front_support_full",
                    "smooth_highpass",
                    "front_support_highpass",
                )
            },
            "input_error": {
                "normal_full_norm": float(call),
                "smooth_highpass_norm": float(call),
                "normal_full_propagation_gain": 0.5,
                "smooth_highpass_propagation_gain": 0.25,
            },
        }
        for call in (1, 5)
        for _ in range(3)
    ]

    summary = aggregate_rows(rows, (1, 5))

    assert summary["1"]["row_count"] == 3
    assert summary["5"]["row_count"] == 3
    assert (
        summary["5"]["regions"]["normal_nodes_full"]["propagated_norm"]["median"] == 2.0
    )
    assert summary["5"]["input_error"]["normal_full_norm"]["mean"] == 5.0


def test_validate_args_rejects_existing_output_and_invalid_endpoints(
    tmp_path: Path,
) -> None:
    base = dict(
        output_dir=tmp_path / "new",
        expected_trajectory_count=30,
        start_frame=0,
        num_steps=79,
        endpoint_calls=[1, 20, 79],
        shock_quantile=0.9,
        shock_dilation_hops=2,
    )
    validate_args(argparse.Namespace(**base))

    duplicate = argparse.Namespace(**{**base, "endpoint_calls": [1, 1]})
    with pytest.raises(ValueError, match="unique calls"):
        validate_args(duplicate)

    base["output_dir"].mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_args(argparse.Namespace(**base))

from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_euler1d_resolution_transfer import (
    conservative_restrict,
    evaluate_one_step_by_input_frame,
    native_update_gaps,
    validate_resolution_family,
)
from utility.time_dependent_no.euler1d_data import Euler1DNPZ
from utility.time_dependent_no.euler1d_targets import make_target_adapter


def _matched_source(num_cells: int) -> Euler1DNPZ:
    domains = np.array([[0.0, 1.0], [0.1, 1.2]], dtype=np.float32)
    x = np.stack(
        [
            np.linspace(
                left + 0.5 * (right - left) / num_cells,
                right - 0.5 * (right - left) / num_cells,
                num_cells,
                dtype=np.float32,
            )
            for left, right in domains
        ]
    )
    data = np.ones((2, 3, num_cells, 3), dtype=np.float32)
    data[..., 1] = 0.25
    t = np.broadcast_to(
        np.array([0.0, 0.1, 0.2], dtype=np.float32),
        (2, 3),
    ).copy()
    return Euler1DNPZ(
        data=data,
        x=x,
        t=t,
        left_states=np.array([[1.0, 0.2, 1.0], [1.1, 0.3, 1.2]], dtype=np.float32),
        right_states=np.array([[0.2, 0.0, 0.1], [0.3, 0.0, 0.2]], dtype=np.float32),
        gamma=1.4,
        metadata={
            "domains": domains,
            "x_disc": np.array([0.25, 0.35], dtype=np.float32),
            "t_final": np.array([0.2, 0.2], dtype=np.float32),
            "nx": np.array(num_cells, dtype=np.int32),
        },
    )


def test_resolution_family_requires_exact_physical_case_identity():
    coarse = _matched_source(8)
    fine = _matched_source(16)

    summary = validate_resolution_family({"nx8": coarse, "nx16": fine})

    assert summary["physical_case_identity_exact"] is True
    assert summary["saved_time_identity_exact"] is True
    assert summary["datasets"]["nx8"]["num_cells"] == 8
    assert summary["datasets"]["nx16"]["num_cells"] == 16

    changed_left = fine.left_states.copy()
    changed_left[0, 0] += 0.01
    with pytest.raises(ValueError, match="left states differ"):
        validate_resolution_family(
            {"nx8": coarse, "nx16": replace(fine, left_states=changed_left)}
        )


def test_conservative_restriction_preserves_piecewise_constant_cells():
    coarse = np.array(
        [
            [1.0, 0.3, 1.2],
            [0.8, -0.1, 0.7],
            [0.4, 0.5, 0.2],
            [1.3, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    fine = np.repeat(coarse, 2, axis=0)

    restricted = conservative_restrict(fine, coarse_cells=4, gamma=1.4)

    np.testing.assert_allclose(restricted, coarse, rtol=1.0e-12, atol=1.0e-12)


class _ZeroResidual(torch.nn.Module):
    def forward(self, batch):
        return torch.zeros_like(batch.current_conservative)


def test_native_update_gap_is_zero_for_identical_constant_trajectories():
    rows = native_update_gaps(
        {"nx8": _matched_source(8), "nx16": _matched_source(16)},
        np.array([0, 1], dtype=np.int64),
        strides=[1, 2],
    )

    assert len(rows) == 3
    assert all(row["state_normalized_mean"] == pytest.approx(0.0) for row in rows)
    assert all(row["update_normalized_mean"] == pytest.approx(0.0) for row in rows)


def test_one_step_profile_reports_each_native_input_frame():
    source = _matched_source(8)

    rows = evaluate_one_step_by_input_frame(
        _ZeroResidual(),
        make_target_adapter("residual"),
        source,
        np.array([0, 1], dtype=np.int64),
        step_stride=1,
        device=torch.device("cpu"),
    )

    assert [row["input_frame"] for row in rows] == [0, 1]
    assert [row["target_frame"] for row in rows] == [1, 2]
    assert all(row["relative_l2_mean"] == pytest.approx(0.0) for row in rows)

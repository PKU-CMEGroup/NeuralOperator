from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.time_dependent_no.plot_corrective_ode_gaussian_xy_diagnostics import (
    DiagnosticConfig,
    PARENT_PACKET,
    PARENT_MANIFEST_SHA256,
    REPOSITORY_ROOT,
    SEED_COLORS,
    _load_json,
    load_training_index,
    planar_error,
    positive_limits,
    run,
    tubular_chart,
    visible_path,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file


def test_tubular_chart_maps_clean_states_to_unit_circle() -> None:
    theta = np.linspace(-math.pi, math.pi, 17)
    state = np.column_stack((theta, np.zeros_like(theta)))
    xy = tubular_chart(state)
    np.testing.assert_allclose(np.linalg.norm(xy, axis=1), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(xy[:, 0], np.cos(theta), atol=1.0e-14)
    np.testing.assert_allclose(xy[:, 1], np.sin(theta), atol=1.0e-14)


def test_tubular_chart_preserves_signed_normal_distance_in_display_strip() -> None:
    state = np.asarray(
        [[-2.0, -0.3], [-0.5, -0.1], [0.0, 0.0], [1.5, 0.2], [3.0, 0.3]]
    )
    xy = tubular_chart(state)
    radial_offset = np.linalg.norm(xy, axis=1) - 1.0
    np.testing.assert_allclose(radial_offset, state[:, 1], atol=1.0e-14)


def test_planar_error_is_direct_cartesian_distance() -> None:
    left = np.asarray([[0.0, 0.0], [math.pi / 2.0, 0.2]])
    right = np.asarray([[math.pi / 2.0, 0.0], [math.pi / 2.0, -0.1]])
    expected = np.linalg.norm(tubular_chart(left) - tubular_chart(right), axis=1)
    np.testing.assert_allclose(planar_error(left, right), expected)
    np.testing.assert_allclose(expected, [math.sqrt(2.0), 0.3], atol=1.0e-14)


def test_visible_path_stops_at_first_exit_and_marks_boundary() -> None:
    path = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.2, 0.31], [0.3, 0.0]])
    prefix, marker = visible_path(path, 0.3)
    np.testing.assert_allclose(prefix, tubular_chart(path[:2]))
    assert marker is not None
    np.testing.assert_allclose(np.linalg.norm(marker), 1.3, atol=1.0e-14)


def test_positive_limits_ignore_masked_values_and_are_ordered() -> None:
    lower, upper = positive_limits(
        [np.asarray([[np.nan, 0.0, 1.0e-9], [1.0e-4, 1.0e-2, 1.0]])]
    )
    assert lower >= 1.0e-7
    assert upper > lower


def test_display_contract_does_not_select_a_noise_scale() -> None:
    config = DiagnosticConfig().validated()
    assert config.method_display_sigma == 0.02
    assert config.central_gaussian_probability == 0.95
    assert SEED_COLORS == {17: "#FFFFFF", 29: "#56B4E9", 43: "#F0E442"}


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / PARENT_PACKET / "manifest.json").is_file(),
    reason="local ignored B2-GN integration packet is unavailable",
)
def test_parent_training_index_binds_all_checkpoint_hashes_when_available() -> None:
    parent = REPOSITORY_ROOT / PARENT_PACKET
    manifest_path = parent / "manifest.json"
    assert sha256_file(manifest_path) == PARENT_MANIFEST_SHA256
    index = load_training_index(parent, _load_json(manifest_path))
    assert len(index) == 51
    assert len({row["checkpoint"] for row in index.values()}) == 51


def test_run_refuses_an_existing_output_directory() -> None:
    target = Path("already_exists")
    with patch.object(Path, "mkdir", side_effect=FileExistsError) as mkdir:
        with pytest.raises(FileExistsError):
            run(target)
    mkdir.assert_called_once_with(parents=True, exist_ok=False)

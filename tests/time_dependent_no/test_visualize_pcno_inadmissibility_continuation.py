from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.visualize_pcno_inadmissibility_continuation import (
    INPUT_SCHEMA,
    animation_fields,
    build_continuous_field_map,
    invalid_nodes,
    load_bundle,
    rasterize_field,
    reference_scales,
)


def _arrays(node_count: int = 20) -> dict[str, np.ndarray]:
    frames = 80
    reference = np.zeros((frames, node_count, 4), dtype=np.float32)
    reference[..., 0] = 1.0
    reference[..., 3] = 2.5
    reference[..., 3] += np.linspace(0.0, 0.5, frames, dtype=np.float32)[:, None]
    n0 = reference.copy()
    correct = reference.copy()
    zero = reference.copy()
    correct[50, 17] = np.asarray([1.0, 2.0, 0.0, 1.0], dtype=np.float32)
    raw = correct[1:].copy()
    model_current = correct[:-1].copy()
    zero_raw = zero[1:].copy()
    zero_current = zero[:-1].copy()
    node_type = np.zeros((node_count, 1), dtype=np.int64)
    node_type[:4, 0] = np.asarray([1, 2, 3, 1])
    return {
        "schema": np.asarray(INPUT_SCHEMA),
        "trajectory": np.asarray("54"),
        "reference_states": reference,
        "positions": np.stack(
            np.meshgrid(
                np.linspace(0.0, 3.0, 5),
                np.linspace(0.0, 1.0, 4),
            ),
            axis=-1,
        ).reshape(-1, 2).astype(np.float32),
        "node_type": node_type,
        "node_weights_proxy": np.ones((node_count, 1), dtype=np.float32),
        "boundary_features": np.zeros((node_count, 3), dtype=np.float32),
        "physical_times": np.arange(frames, dtype=np.float64) * 0.025,
        "all_temporal_frames_retained": np.asarray(True),
        "N0_correct_deployed_states": n0,
        "D082_correct_deployed_states": correct,
        "D082_zero_inflow_deployed_states": zero,
        "D082_correct_raw_proposals": raw,
        "D082_correct_model_currents": model_current,
        "D082_zero_inflow_raw_proposals": zero_raw,
        "D082_zero_inflow_model_currents": zero_current,
    }


def test_load_bundle_requires_and_preserves_all_80_frames(tmp_path) -> None:
    path = tmp_path / "case_54.npz"
    np.savez_compressed(path, **_arrays())

    loaded = load_bundle(path)

    assert loaded["reference_states"].shape == (80, 20, 4)
    assert bool(loaded["all_temporal_frames_retained"].item()) is True


def test_reference_scales_ignore_extreme_model_outputs() -> None:
    arrays = _arrays()
    baseline = reference_scales(arrays["reference_states"])
    arrays["N0_correct_deployed_states"][79, :, 3] = 1.0e20

    observed = reference_scales(arrays["reference_states"])

    assert observed == baseline
    assert observed["pressure_max"] > observed["pressure_min"]


def test_continuous_field_map_uses_all_nodes_and_preserves_linear_fields() -> None:
    arrays = _arrays()
    positions = arrays["positions"]

    field_map = build_continuous_field_map(positions, raster_width=256)
    nodal_value = positions[:, 0] + 2.0 * positions[:, 1]
    raster = rasterize_field(nodal_value, field_map)
    grid_x, grid_y = np.meshgrid(
        field_map["x_coordinates"], field_map["y_coordinates"]
    )
    finite = np.isfinite(raster)

    assert field_map["source_node_count"] == positions.shape[0]
    assert field_map["shape"] == (85, 256)
    assert np.any(finite)
    assert np.allclose(raster[finite], (grid_x + 2.0 * grid_y)[finite])


def test_animation_fields_align_initial_and_79_model_residual_frames() -> None:
    arrays = _arrays()

    fields, _, _, sources = animation_fields(arrays, "field_intervention")

    assert all(field.shape == (80, 20) for field in fields)
    assert all(source.shape == (80, 20, 4) for source in sources)
    assert np.array_equal(fields[3][0], np.zeros(20))
    assert np.array_equal(fields[4][0], np.zeros(20))


def test_invalid_nodes_detects_finite_negative_internal_energy() -> None:
    arrays = _arrays()

    mask = invalid_nodes(arrays["D082_correct_deployed_states"][50])

    assert np.flatnonzero(mask).tolist() == [17]

from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.analyze_pcno_residual_structure import (
    _bundle_pathway_pair,
    _bundle_unified_pair,
)
from scripts.time_dependent_no.visualize_pcno_residual_structure import (
    _animation_arrays,
)
from utility.time_dependent_no.pcno_residual_structure import (
    algebra_identity_closure,
    characteristic_energy,
    decomposition_diagnostics,
    phase_projection,
    recurrence_diagnostics,
    region_energy_rows,
    sequence_diagnostics,
    shock_vortex_regions,
    spectral_energy_rows,
    weighted_rms,
)


def _grid(nx: int = 40, ny: int = 20) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx, dtype=np.float64) + 0.5) * 2.0 / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    return nodes, volumes


def _shock_state(nodes: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    density = np.where(nodes[:, 0] < 1.0, 1.0, 1.4)
    dx = nodes[:, 0] - 0.55
    dy = nodes[:, 1] - 0.5
    radius2 = np.square(dx) + np.square(dy)
    velocity_x = -0.03 * dy * np.exp(-radius2 / 0.02)
    velocity_y = 0.03 * dx * np.exp(-radius2 / 0.02)
    pressure = np.where(nodes[:, 0] < 1.0, 1.0, 1.7)
    energy = pressure / (gamma - 1.0) + 0.5 * density * (
        np.square(velocity_x) + np.square(velocity_y)
    )
    return np.stack(
        (density, density * velocity_x, density * velocity_y, energy), axis=-1
    )


def test_persistent_multiplicative_bias_has_high_coherence() -> None:
    _, volumes = _grid(4, 2)
    truth = np.ones((5, 8, 4), dtype=np.float64)
    defect = 0.3 * truth
    rows, summary = sequence_diagnostics(
        defect,
        truth,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert np.isclose(summary["aggregate_relative_residual_energy"], 0.3)
    assert np.isclose(summary["global_parallel_bias"], 0.3)
    assert np.isclose(summary["temporal_coherence"], 1.0)
    assert np.isclose(summary["net_relative_to_truth_change"], 0.3)
    assert np.isclose(rows[-1]["cumulative_parallel_bias"], 0.3)


def test_alternating_defect_has_energy_but_zero_net_accumulation() -> None:
    _, volumes = _grid(4, 2)
    truth = np.ones((4, 8, 4), dtype=np.float64)
    signs = np.asarray((1.0, -1.0, 1.0, -1.0)).reshape(-1, 1, 1)
    defect = 0.5 * signs * truth
    _, summary = sequence_diagnostics(
        defect,
        truth,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert np.isclose(summary["aggregate_relative_residual_energy"], 0.5)
    assert np.isclose(summary["net_defect_rms"], 0.0)
    assert np.isclose(summary["temporal_coherence"], 0.0)


def test_recurrence_and_signed_growth_close() -> None:
    _, volumes = _grid(4, 2)
    generator = np.random.default_rng(20260802)
    defects = generator.normal(size=(5, 8, 4))
    initial = generator.normal(size=(1, 8, 4))
    errors = np.concatenate((initial, initial + np.cumsum(defects, axis=0)), axis=0)
    rows, summary = recurrence_diagnostics(
        errors,
        defects,
        volumes=volumes,
        component_scale=np.asarray((1.0, 2.0, 3.0, 4.0)),
    )
    assert summary["maximum_recurrence_closure_rms"] < 1.0e-14
    assert summary["maximum_growth_closure_absolute"] < 1.0e-12
    assert max(abs(row["growth_closure"]) for row in rows) < 1.0e-12


def test_animation_arrays_reconstruct_unified_payload_redundancies() -> None:
    generator = np.random.default_rng(20260803)
    true = generator.normal(size=(3, 8, 4))
    coarse = generator.normal(size=(3, 8, 4))
    fine = generator.normal(size=(3, 8, 4))
    delta = coarse - fine
    cumulative = np.cumsum(delta, axis=0)
    key = "125x50_to_250x100"
    minimal = {
        f"true_increment__{key}": true,
        f"coarse_increment_free__{key}": coarse,
        f"fine_increment_free__{key}": fine,
        f"delta_free__{key}": delta,
        f"cumulative_delta_free__{key}": cumulative,
    }
    reconstructed = _animation_arrays(minimal, "125x50->250x100", "free")
    expected_growth = 2.0 * (cumulative - delta) * delta + np.square(delta)
    assert np.allclose(reconstructed[3], coarse - true)
    assert np.allclose(reconstructed[4], fine - true)
    assert np.allclose(reconstructed[7], expected_growth)

    explicit = dict(minimal)
    explicit[f"coarse_error_free__{key}"] = coarse - true
    explicit[f"fine_error_free__{key}"] = fine - true
    explicit[f"growth_free__{key}"] = expected_growth
    for reconstructed_value, explicit_value in zip(
        reconstructed,
        _animation_arrays(explicit, "125x50->250x100", "free"),
        strict=True,
    ):
        assert np.array_equal(reconstructed_value, explicit_value)


def test_generalized_commutator_identity_includes_input_gap() -> None:
    generator = np.random.default_rng(7)
    coarse_current = generator.normal(size=(8, 4))
    restricted_fine_current = coarse_current + 0.01
    coarse_prediction = coarse_current + generator.normal(size=(8, 4))
    restricted_fine_prediction = restricted_fine_current + generator.normal(size=(8, 4))
    closure = algebra_identity_closure(
        coarse_current,
        coarse_prediction,
        restricted_fine_current,
        restricted_fine_prediction,
    )
    assert np.max(np.abs(closure)) < 1.0e-15


def test_mesh_state_decomposition_preserves_cross_term() -> None:
    _, volumes = _grid(4, 2)
    generator = np.random.default_rng(9)
    mesh = generator.normal(size=(3, 8, 4))
    state = generator.normal(size=(3, 8, 4))
    total = mesh + state
    rows, summary = decomposition_diagnostics(
        total,
        mesh,
        state,
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert summary["maximum_decomposition_closure_rms"] < 1.0e-14
    for row in rows:
        reconstructed = (
            row["mesh_energy"] + row["state_energy"] + row["twice_cross_term"]
        )
        assert np.isclose(reconstructed, row["total_energy"])
        assert np.isclose(
            row["mesh_symmetric_attribution"] + row["state_symmetric_attribution"],
            1.0,
        )


def test_physical_regions_partition_and_phase_projection() -> None:
    nx, ny = 40, 20
    nodes, volumes = _grid(nx, ny)
    state = _shock_state(nodes)
    masks, phase_mode, normals = shock_vortex_regions(
        state, nodes, resolution=(nx, ny), gamma=1.4
    )
    partition = sum(
        masks[name].astype(np.int8)
        for name in (
            "partition_boundary",
            "partition_shock",
            "partition_vortex",
            "partition_smooth",
        )
    )
    assert np.all(partition == 1)
    displacement = 0.012
    defect = -displacement * phase_mode
    projection = phase_projection(
        defect,
        phase_mode,
        shock_mask=masks["shock_envelope_le_0.05"],
        volumes=volumes,
        component_scale=np.ones(4),
    )
    assert np.isclose(projection["estimated_displacement"], displacement)
    assert np.isclose(projection["phase_energy_fraction"], 1.0)
    rows = region_energy_rows(
        defect,
        volumes=volumes,
        component_scale=np.ones(4),
        masks=masks,
    )
    assert any(row["region"] == "partition_shock" for row in rows)
    assert normals.shape == (nx * ny, 2)


def test_windowed_spectrum_recovers_registered_wavelength_band() -> None:
    nx, ny = 80, 40
    nodes, _ = _grid(nx, ny)
    wavelength = 0.2
    wave = np.sin(2.0 * np.pi * nodes[:, 0] / wavelength)
    defect = np.zeros((nx * ny, 4), dtype=np.float64)
    defect[:, 0] = wave
    rows = spectral_energy_rows(
        defect,
        resolution=(nx, ny),
        domain_lengths=(2.0, 1.0),
        component_scale=np.ones(4),
    )
    target = next(
        row
        for row in rows
        if row["wavelength_min"] == 0.125 and row["wavelength_max"] == 0.25
    )
    assert target["spectral_energy_share"] > 0.9


def test_characteristic_coefficient_shares_close() -> None:
    nodes, _ = _grid(4, 2)
    reference = _shock_state(nodes)
    defect = np.zeros_like(reference)
    defect[:, 0] = 0.1
    normals = np.tile(np.asarray((1.0, 0.0)), (nodes.shape[0], 1))
    result = characteristic_energy(
        defect,
        reference,
        normals,
        shock_mask=np.ones(nodes.shape[0], dtype=bool),
        gamma=1.4,
        component_scale=np.ones(4),
    )
    shares = [
        result[f"{name}_coefficient_energy_share"]
        for name in ("acoustic_minus", "entropy", "shear", "acoustic_plus")
    ]
    assert np.isclose(sum(float(value) for value in shares), 1.0)
    assert result["maximum_eigenvector_condition_number"] < 1.0e4


def test_weighted_rms_uses_physical_volume_and_component_scale() -> None:
    value = np.asarray(((1.0, 0.0, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0)))
    actual = weighted_rms(
        value,
        volumes=np.asarray((1.0, 3.0)),
        component_scale=np.asarray((2.0, 1.0, 1.0, 1.0)),
    )
    expected = np.sqrt((1.0 * 0.5**2 + 3.0 * 1.5**2) / 4.0)
    assert np.isclose(actual, expected)


def test_pathway_bundle_saves_only_exact_extension_fields() -> None:
    nodes, volumes = _grid(4, 2)
    generator = np.random.default_rng(17)
    reference = generator.normal(size=(4, 8, 4))
    coarse = generator.normal(size=(4, 8, 4))
    fine = generator.normal(size=(4, 8, 4))
    mesh = generator.normal(size=(3, 8, 4))
    state = generator.normal(size=(3, 8, 4))
    arrays: dict[str, np.ndarray] = {}
    _bundle_pathway_pair(
        arrays,
        pair_name="4x2->8x4",
        nodes=nodes,
        volumes=volumes,
        reference_states=reference,
        free_coarse_states=coarse,
        free_fine_states=fine,
        mesh_delta=mesh,
        state_delta=state,
    )
    suffix = "4x2_to_8x4"
    assert set(arrays) == {
        f"nodes__{suffix}",
        f"volumes__{suffix}",
        f"reference_states__{suffix}",
        f"free_coarse_states__{suffix}",
        f"free_fine_states__{suffix}",
        f"mesh_delta__{suffix}",
        f"state_delta__{suffix}",
    }
    assert arrays[f"nodes__{suffix}"].dtype == np.float64
    assert arrays[f"reference_states__{suffix}"].dtype == np.float64
    assert np.allclose(arrays[f"mesh_delta__{suffix}"], mesh)


def test_unified_bundle_replays_free_fields_without_storage_rounding() -> None:
    nodes, volumes = _grid(4, 2)
    generator = np.random.default_rng(18)
    reference = generator.normal(size=(4, 8, 4))
    coarse = generator.normal(size=(4, 8, 4))
    fine = generator.normal(size=(4, 8, 4))
    mesh = generator.normal(size=(3, 8, 4))
    state = coarse[1:] - coarse[:-1] - (fine[1:] - fine[:-1]) - mesh
    arrays: dict[str, np.ndarray] = {}
    _bundle_unified_pair(
        arrays,
        pair_name="4x2->8x4",
        nodes=nodes,
        volumes=volumes,
        reference_states=reference,
        free_coarse_states=coarse,
        free_fine_states=fine,
        mesh_delta=mesh,
        state_delta=state,
    )
    suffix = "4x2_to_8x4"
    assert np.array_equal(
        arrays[f"true_increment__{suffix}"], np.diff(reference, axis=0)
    )
    assert np.array_equal(
        arrays[f"coarse_increment_free__{suffix}"], np.diff(coarse, axis=0)
    )
    assert np.array_equal(
        arrays[f"fine_increment_free__{suffix}"], np.diff(fine, axis=0)
    )
    assert np.max(np.abs(arrays[f"delta_free__{suffix}"] - (mesh + state))) < 2.0e-15
    assert all(value.dtype == np.float64 for value in arrays.values())

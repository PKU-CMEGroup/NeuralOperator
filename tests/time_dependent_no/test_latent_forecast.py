from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.run_euler1d_latent_forecast_pilot import (
    _chart_cohort_masks,
    _chart_sample_geometry,
    _conditioned_transition_code,
    _grouped_bootstrap_remap_cohort,
    _history_not_materially_better_gate,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    conservative_to_primitive_np,
    primitive_to_conservative_np,
)
from utility.time_dependent_no.latent_forecast import (
    Euler1DFrontSet,
    Euler1DLatentContext,
    IdentityRepresentation,
    LatentCode,
    LinearCodePCA,
    OracleFrontKinematicPOD,
    VolumeWeightedPOD,
    cell_edges_from_centers,
    conditional_future_diagnostics,
    decode_coarse_conservative,
    decode_registered_state,
    extract_euler1d_front_kinematics,
    extract_euler1d_fronts,
    fixed_scale_relative_l2,
    front_reconstruction_metrics,
    grouped_bootstrap_median_ratio,
    register_conservative_state,
    rollout_latent_raw,
)


def _context(*, batch_size: int = 1, num_cells: int = 8) -> Euler1DLatentContext:
    centers = (torch.arange(num_cells, dtype=torch.float64) + 0.5) / num_cells
    return Euler1DLatentContext(
        cell_centers=centers[None, :, None].repeat(batch_size, 1, 1),
        cell_volume=torch.full(
            (batch_size, num_cells), 1.0 / num_cells, dtype=torch.float64
        ),
        gamma=1.4,
        left_boundary_primitive=torch.tensor(
            [[1.0, 0.2, 1.0]], dtype=torch.float64
        ).repeat(batch_size, 1),
        right_initial_primitive=torch.tensor(
            [[0.8, -0.1, 0.9]], dtype=torch.float64
        ).repeat(batch_size, 1),
    )


def _positive_conservative(*, batch_size: int, num_cells: int) -> torch.Tensor:
    x = (torch.arange(num_cells, dtype=torch.float64) + 0.5) / num_cells
    rho = 1.0 + 0.1 * x[None].repeat(batch_size, 1)
    velocity = 0.2 - 0.05 * x[None].repeat(batch_size, 1)
    pressure = 1.0 + 0.2 * x[None].repeat(batch_size, 1)
    energy = pressure / 0.4 + 0.5 * rho * velocity.square()
    return torch.stack((rho, rho * velocity, energy), dim=-1)


def test_latent_context_contains_metadata_but_no_physical_recurrence_state() -> None:
    context = _context()

    assert {field.name for field in fields(context)} == {
        "cell_centers",
        "cell_volume",
        "gamma",
        "left_boundary_primitive",
        "right_initial_primitive",
    }
    assert not hasattr(context, "current_conservative")
    assert not hasattr(context, "target_conservative")
    assert not hasattr(context, "current_primitive")


def test_identity_representation_is_an_exact_conservative_roundtrip() -> None:
    context = _context(batch_size=2, num_cells=7)
    conservative = _positive_conservative(batch_size=2, num_cells=7)
    representation = IdentityRepresentation(num_cells=7)

    code = representation.encode(conservative, context)
    decoded = representation.decode(code, context)

    assert code.layout.spatial_tokens == 7
    assert code.layout.spatial_channels == 3
    torch.testing.assert_close(decoded.conservative, conservative, rtol=0.0, atol=0.0)
    assert torch.all(decoded.primitive[..., 0] > 0.0)
    assert torch.all(decoded.primitive[..., 2] > 0.0)


class _CountingIdentity(IdentityRepresentation):
    def __init__(self, num_cells: int) -> None:
        super().__init__(num_cells)
        self.encode_count = 0
        self.decode_count = 0

    def encode(
        self,
        conservative: torch.Tensor,
        context: Euler1DLatentContext,
    ) -> LatentCode:
        self.encode_count += 1
        return super().encode(conservative, context)

    def decode(
        self,
        code: LatentCode,
        context: Euler1DLatentContext,
        *,
        decoder_constraint: str = "none",
    ):
        self.decode_count += 1
        return super().decode(
            code,
            context,
            decoder_constraint=decoder_constraint,
        )


class _PressureFailureTransition:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        code: LatentCode,
        _context: Euler1DLatentContext,
        _dt: torch.Tensor,
    ) -> LatentCode:
        self.calls += 1
        values = code.values.clone()
        if self.calls == 2:
            values[:, 2::3] = -1.0
        return LatentCode(values=values, layout=code.layout)


class _NonfiniteCodeTransition:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        code: LatentCode,
        _context: Euler1DLatentContext,
        _dt: torch.Tensor,
    ) -> LatentCode:
        self.calls += 1
        values = code.values.clone()
        values[:, 0] = torch.nan
        return LatentCode(values=values, layout=code.layout)


class _MaskingDecoderIdentity(_CountingIdentity):
    def decode(
        self,
        code: LatentCode,
        context: Euler1DLatentContext,
        *,
        decoder_constraint: str = "none",
    ):
        self.decode_count += 1
        masked = LatentCode(
            values=torch.nan_to_num(
                code.values,
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            ),
            layout=code.layout,
        )
        return IdentityRepresentation.decode(
            self,
            masked,
            context,
            decoder_constraint=decoder_constraint,
        )


class _NonfiniteInitialIdentity(_CountingIdentity):
    def encode(
        self,
        conservative: torch.Tensor,
        context: Euler1DLatentContext,
    ) -> LatentCode:
        code = super().encode(conservative, context)
        values = code.values.clone()
        values[:, 0] = torch.nan
        return LatentCode(values=values, layout=code.layout)


def test_raw_rollout_rejects_nonfinite_dt_before_transition() -> None:
    context = _context(num_cells=6)
    initial = _positive_conservative(batch_size=1, num_cells=6)
    representation = _CountingIdentity(num_cells=6)
    transition = _PressureFailureTransition()

    with pytest.raises(ValueError, match="finite"):
        rollout_latent_raw(
            representation,
            transition,
            initial,
            context,
            torch.tensor([0.1, float("nan")], dtype=torch.float64),
        )

    assert representation.encode_count == 0
    assert representation.decode_count == 0
    assert transition.calls == 0


def test_raw_rollout_records_nonfinite_initial_code_before_transition() -> None:
    context = _context(num_cells=6)
    initial = _positive_conservative(batch_size=1, num_cells=6)
    representation = _NonfiniteInitialIdentity(num_cells=6)
    transition = _PressureFailureTransition()

    rollout = rollout_latent_raw(
        representation,
        transition,
        initial,
        context,
        torch.tensor([0.1], dtype=torch.float64),
    )

    assert representation.encode_count == 1
    assert representation.decode_count == 0
    assert transition.calls == 0
    assert rollout.valid_length == 0
    assert rollout.failure_cause == "nonfinite_initial_latent_code"
    assert rollout.records[0].step == 0


def test_raw_rollout_rejects_nonfinite_code_before_masking_decoder() -> None:
    context = _context(num_cells=6)
    initial = _positive_conservative(batch_size=1, num_cells=6)
    representation = _MaskingDecoderIdentity(num_cells=6)
    transition = _NonfiniteCodeTransition()

    rollout = rollout_latent_raw(
        representation,
        transition,
        initial,
        context,
        torch.tensor([0.1, 0.1], dtype=torch.float64),
    )

    assert transition.calls == 1
    assert representation.encode_count == 1
    assert representation.decode_count == 0
    assert rollout.valid_length == 0
    assert rollout.failure_cause == "nonfinite_latent_code"
    assert len(rollout.records) == 1
    assert not rollout.records[0].finite
    assert not rollout.records[0].admissible


def test_raw_rollout_encodes_once_never_reencodes_and_stops_on_invalid_state() -> None:
    context = _context(num_cells=6)
    initial = _positive_conservative(batch_size=1, num_cells=6)
    representation = _CountingIdentity(num_cells=6)
    transition = _PressureFailureTransition()

    rollout = rollout_latent_raw(
        representation,
        transition,
        initial,
        context,
        torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
    )

    assert representation.encode_count == 1
    assert representation.decode_count == 2
    assert transition.calls == 2
    assert len(rollout.codes) == 3
    assert len(rollout.decoded) == 2
    assert rollout.valid_length == 1
    assert rollout.failure_cause == "nonpositive_raw_pressure"
    assert [record.encode_calls for record in rollout.records] == [1, 1]
    assert [record.transition_calls for record in rollout.records] == [1, 2]
    assert [record.decode_calls for record in rollout.records] == [1, 2]
    assert [record.admissible for record in rollout.records] == [True, False]
    for record in rollout.records:
        assert not record.decode_reencode
        assert not record.clipping
        assert not record.density_floor
        assert not record.pressure_floor
        assert not record.limiter
        assert not record.projection
        assert not record.reset
        assert not record.truth_replacement


def test_volume_weighted_pod_roundtrips_at_full_rank_on_nonuniform_grid() -> None:
    rng = np.random.default_rng(20260720)
    states = rng.normal(size=(16, 5, 3))
    volume = np.array([0.03, 0.09, 0.18, 0.27, 0.43])
    component_scale = np.array([0.7, 1.3, 2.1])
    pod = VolumeWeightedPOD.fit(
        states,
        volume,
        component_scale,
        rank=states.shape[1] * states.shape[2],
    )

    reconstructed = pod.decode(pod.encode(states))

    assert pod.rank == 15
    assert pod.retained_energy == pytest.approx(1.0, abs=1.0e-14)
    np.testing.assert_allclose(reconstructed, states, rtol=1.0e-12, atol=1.0e-12)


def test_volume_weighted_pod_selects_the_smallest_energy_rank() -> None:
    coefficients = np.array(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ]
    )
    states = np.zeros((coefficients.shape[0], 4, 3))
    states[:, 0, 0] = coefficients[:, 0]
    states[:, 2, 1] = 1.0e-3 * coefficients[:, 1]

    pod = VolumeWeightedPOD.fit(
        states,
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.ones(3),
        energy_fraction=0.995,
    )

    assert pod.rank == 1
    assert pod.retained_energy >= 0.995


def test_front_extractor_finds_pressure_fronts_and_contact_with_invalid_slots() -> None:
    num_cells = 160
    x = (np.arange(num_cells) + 0.5) / num_cells
    complete = np.ones((num_cells, 3))
    complete[:, 1] = 0.2
    complete[40:120, 2] = 1.6
    complete[120:, 2] = 1.1
    complete[80:, 0] = 1.4

    one_pressure_front = np.ones((num_cells, 3))
    one_pressure_front[:, 1] = -0.1
    one_pressure_front[60:, 2] = 1.4
    constant = np.ones((num_cells, 3))
    constant[:, 1] = 0.0
    fronts = extract_euler1d_fronts(
        np.stack((complete, one_pressure_front, constant)),
        x,
    )

    np.testing.assert_array_equal(fronts.valid[0], [True, True, True])
    np.testing.assert_allclose(
        fronts.position_fraction[0], [0.25, 0.75, 0.5], atol=1.0 / num_cells
    )
    assert fronts.signed_strength[0, 0] > 0.0
    assert fronts.signed_strength[0, 1] < 0.0
    assert fronts.signed_strength[0, 2] > 0.0
    np.testing.assert_array_equal(fronts.valid[1], [True, False, False])
    np.testing.assert_array_equal(fronts.valid[2], [False, False, False])
    np.testing.assert_allclose(fronts.position_fraction[~fronts.valid], 0.0)
    np.testing.assert_allclose(fronts.thickness_fraction[~fronts.valid], 0.0)
    np.testing.assert_allclose(fronts.signed_strength[~fronts.valid], 0.0)

    roundtrip = Euler1DFrontSet.from_vector(fronts.vector())
    np.testing.assert_array_equal(roundtrip.valid, fronts.valid)
    np.testing.assert_allclose(roundtrip.position_fraction, fronts.position_fraction)


def test_front_vector_rejects_nonfinite_values_before_masking_invalid_slots() -> None:
    vector = np.zeros(12)
    vector[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        Euler1DFrontSet.from_vector(vector)


def test_chart_cohorts_isolate_conditioned_single_pressure_topology() -> None:
    fronts = Euler1DFrontSet(
        position_fraction=np.array(
            [
                [0.50, 0.00, 0.00],
                [0.20, 0.80, 0.00],
                [0.01, 0.00, 0.00],
                [0.40, 0.00, 0.60],
            ]
        ),
        thickness_fraction=np.full((4, 3), 0.02),
        signed_strength=np.ones((4, 3)),
        valid=np.array(
            [
                [True, False, False],
                [True, True, False],
                [True, False, False],
                [True, False, True],
            ]
        ),
        score=np.ones((4, 3)),
    )

    geometry = _chart_sample_geometry(fronts)
    cohorts = _chart_cohort_masks(geometry)

    np.testing.assert_array_equal(geometry["conditioned"], [True, True, False, True])
    np.testing.assert_array_equal(geometry["pressure_front_count"], [1, 2, 1, 1])
    np.testing.assert_array_equal(
        cohorts["conditioned_single_pressure"], [True, False, False, True]
    )
    np.testing.assert_array_equal(
        cohorts["conditioned_double_pressure"], [False, True, False, False]
    )
    np.testing.assert_array_equal(
        cohorts["unconditioned_single_pressure"], [False, False, True, False]
    )
    assert np.isnan(geometry["minimum_separation"][0])
    assert geometry["minimum_separation"][3] == pytest.approx(0.20)


def test_grouped_remap_cohort_bootstrap_is_deterministic_and_clustered() -> None:
    groups = np.repeat(np.arange(3), 2)
    mask = np.ones(6, dtype=bool)
    state_error = np.arange(1, 7, dtype=np.float64) * 1.0e-3
    admissible = np.ones(6, dtype=bool)
    truth_count = np.ones(6, dtype=np.int64)
    predicted_count = np.ones(6, dtype=np.int64)
    common_count = np.ones(6, dtype=np.int64)
    thickness_distortion = np.full((6, 3), np.nan)
    thickness_distortion[:, 0] = 1.0

    first = _grouped_bootstrap_remap_cohort(
        mask,
        groups,
        state_error,
        admissible,
        truth_count,
        predicted_count,
        common_count,
        thickness_distortion,
        repetitions=200,
        seed=123,
    )
    second = _grouped_bootstrap_remap_cohort(
        mask,
        groups,
        state_error,
        admissible,
        truth_count,
        predicted_count,
        common_count,
        thickness_distortion,
        repetitions=200,
        seed=123,
    )

    assert first == second
    assert first["front_recall"]["estimate"] == pytest.approx(1.0)
    assert first["front_precision"]["estimate"] == pytest.approx(1.0)
    assert first["thickness_symmetric_distortion_p95"]["ci95_upper"] == pytest.approx(
        1.0
    )
    assert first["admissible_fraction"]["ci95_lower"] == pytest.approx(1.0)
    assert first["state_relative_l2_p95"]["groups"] == 3
    assert first["state_relative_l2_p95"]["snapshots"] == 6


def test_front_kinematics_recovers_pressure_continuous_contact_speed() -> None:
    num_cells = 128
    x = (np.arange(num_cells) + 0.5) / num_cells
    primitive = np.ones((num_cells, 3))
    primitive[:, 1] = 0.37
    primitive[num_cells // 2 :, 0] = 1.8

    fronts = extract_euler1d_fronts(primitive, x)
    kinematics = extract_euler1d_front_kinematics(
        primitive,
        x,
        fronts=fronts,
    )

    np.testing.assert_array_equal(fronts.valid, [False, False, True])
    assert kinematics.speed[2] == pytest.approx(0.37, abs=1.0e-12)
    assert kinematics.consistency_residual[2] == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(kinematics.vector()[:2], 0.0)


def test_front_kinematics_recovers_galilean_shifted_normal_shock_speed() -> None:
    gamma = 1.4
    num_cells = 128
    x = (np.arange(num_cells) + 0.5) / num_cells
    upstream_mach = 2.0
    upstream_rho = 1.0
    upstream_pressure = 1.0
    upstream_velocity = upstream_mach * np.sqrt(
        gamma * upstream_pressure / upstream_rho
    )
    density_ratio = (
        (gamma + 1.0) * upstream_mach**2 / ((gamma - 1.0) * upstream_mach**2 + 2.0)
    )
    pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (upstream_mach**2 - 1.0)
    shock_speed = 0.4
    left = np.array([upstream_rho, upstream_velocity + shock_speed, upstream_pressure])
    right = np.array(
        [
            upstream_rho * density_ratio,
            upstream_velocity / density_ratio + shock_speed,
            upstream_pressure * pressure_ratio,
        ]
    )
    primitive = np.broadcast_to(left, (num_cells, 3)).copy()
    primitive[num_cells // 2 :] = right

    fronts = extract_euler1d_fronts(primitive, x, gamma=gamma)
    kinematics = extract_euler1d_front_kinematics(
        primitive,
        x,
        fronts=fronts,
        gamma=gamma,
    )

    assert fronts.valid[0]
    assert kinematics.speed[0] == pytest.approx(shock_speed, abs=1.0e-12)
    assert kinematics.consistency_residual[0] == pytest.approx(0.0, abs=1.0e-12)


def test_registration_preserves_integrals_and_reports_roundtrip_blur() -> None:
    num_cells = 64
    x = ((np.arange(num_cells) + 0.5) / num_cells) ** 1.3
    transition = 0.5 * (1.0 + np.tanh((x - 0.37) / 0.025))
    primitive = np.stack(
        (
            1.0 + 0.8 * transition,
            np.full(num_cells, 0.2),
            1.0 + 0.5 * transition,
        ),
        axis=-1,
    )
    conservative = primitive_to_conservative_np(primitive, gamma=1.4)
    fronts = extract_euler1d_fronts(primitive, x)
    registered, chart = register_conservative_state(conservative, x, fronts)
    decoded = decode_registered_state(registered, x, fronts)
    edges = cell_edges_from_centers(x)
    physical_volume = np.diff(edges) / (edges[-1] - edges[0])

    original_integral = np.sum(physical_volume[:, None] * conservative, axis=0)
    registered_integral = np.mean(registered, axis=0)
    decoded_integral = np.sum(physical_volume[:, None] * decoded, axis=0)
    np.testing.assert_allclose(registered_integral, original_integral, atol=1.0e-12)
    np.testing.assert_allclose(decoded_integral, original_integral, atol=1.0e-12)
    assert chart["valid_fronts"] == 1
    assert chart["minimum_chart_slope"] > 0.0

    relative_error = fixed_scale_relative_l2(
        decoded[None],
        conservative[None],
        physical_volume,
        np.ones(3),
    )[0]
    decoded_fronts = extract_euler1d_fronts(
        conservative_to_primitive_np(decoded, gamma=1.4),
        x,
    )
    blur = front_reconstruction_metrics(
        fronts,
        decoded_fronts,
        num_cells=num_cells,
    )
    assert 0.0 < relative_error < 0.01
    assert blur["common_fronts"] == 1
    assert blur["thickness_ratio_median"] > 1.0


def test_oracle_phase_registration_lowers_pod_rank_for_translated_steps() -> None:
    num_cells = 64
    x = (np.arange(num_cells) + 0.5) / num_cells
    volume = np.full(num_cells, 1.0 / num_cells)
    states = []
    registered_states = []
    for front_cell in range(14, 51, 3):
        primitive = np.empty((num_cells, 3))
        primitive[:front_cell] = [1.0, 0.2, 1.0]
        primitive[front_cell:] = [1.8, 0.2, 1.4]
        conservative = primitive_to_conservative_np(primitive, gamma=1.4)
        fronts = Euler1DFrontSet(
            position_fraction=np.array([front_cell / num_cells, 0.0, 0.0]),
            thickness_fraction=np.array([1.0 / num_cells, 0.0, 0.0]),
            signed_strength=np.array([0.4, 0.0, 0.0]),
            valid=np.array([True, False, False]),
            score=np.array([1.0, 0.0, 0.0]),
        )
        registered, _ = register_conservative_state(conservative, x, fronts)
        states.append(conservative)
        registered_states.append(registered)

    raw_pod = VolumeWeightedPOD.fit(
        np.stack(states),
        volume,
        np.ones(3),
        energy_fraction=0.995,
    )
    registered_pod = VolumeWeightedPOD.fit(
        np.stack(registered_states),
        volume,
        np.ones(3),
        energy_fraction=0.995,
    )

    assert raw_pod.rank > 4
    assert registered_pod.rank < raw_pod.rank


def test_kinematic_oracle_trades_three_residual_modes_at_fixed_total_size() -> None:
    num_cells = 64
    x = (np.arange(num_cells) + 0.5) / num_cells
    volume = np.full(num_cells, 1.0 / num_cells)
    primitive_states = []
    for front_cell in range(14, 51, 3):
        primitive = np.empty((num_cells, 3))
        primitive[:front_cell] = [1.0, 0.5, 1.0]
        primitive[front_cell:] = [1.8, 0.2, 1.4]
        primitive_states.append(primitive)
    primitive_batch = np.stack(primitive_states)

    representation = OracleFrontKinematicPOD.fit(
        primitive_batch,
        x,
        volume,
        np.ones(3),
        total_size=19,
    )
    code = representation.encode(primitive_batch, x)
    decoded = representation.decode_conservative(code, x)
    base_decoded = representation.base_representation.decode_conservative(
        code[:, :-3], x
    )

    assert representation.total_size == 19
    assert representation.base_representation.total_size == 16
    assert representation.residual_rank == 4
    assert code.shape == (primitive_batch.shape[0], 19)
    assert np.isfinite(code[:, -3:]).all()
    np.testing.assert_allclose(decoded, base_decoded, rtol=0.0, atol=0.0)


def test_conditional_ambiguity_detects_collisions_and_excludes_same_group() -> None:
    train_code = np.zeros((4, 1))
    query_code = np.zeros((1, 1))
    train_current = np.zeros((4, 1, 3))
    query_current = np.zeros((1, 1, 3))
    train_future = np.zeros_like(train_current)
    train_future[:, 0, 0] = [1.0, -1.0, 1.0, -1.0]
    query_future = np.zeros_like(query_current)
    query_future[0, 0, 0] = 1.0
    train_group = np.array([0, 1, 2, 3])

    diagnostics = conditional_future_diagnostics(
        train_code,
        train_current,
        train_future,
        query_code,
        query_current,
        query_future,
        np.array([1.0]),
        np.ones(3),
        neighbors=3,
        train_group=train_group,
        query_group=np.array([0]),
    )

    assert diagnostics.neighbor_radius[0] == pytest.approx(0.0)
    assert diagnostics.current_mismatch[0] == pytest.approx(0.0)
    assert diagnostics.future_ambiguity[0] > 1.0
    assert diagnostics.physical_knn_forecast_error[0] > 1.0
    assert np.all(train_group[diagnostics.neighbor_indices[0]] != 0)


def test_conditional_ambiguity_reports_divergent_encoded_futures() -> None:
    train_code = np.zeros((4, 1))
    query_code = np.zeros((1, 1))
    train_future_code = np.array([[-2.0], [2.0], [-2.0], [2.0]])
    query_future_code = np.zeros((1, 1))
    train_current = np.zeros((4, 1, 3))
    train_future = np.zeros_like(train_current)
    train_future[:, 0, 0] = 1.0
    query_current = np.zeros((1, 1, 3))
    query_future = np.zeros_like(query_current)
    query_future[:, 0, 0] = 1.0

    diagnostics = conditional_future_diagnostics(
        train_code,
        train_current,
        train_future,
        query_code,
        query_current,
        query_future,
        np.array([1.0]),
        np.ones(3),
        neighbors=3,
        train_future_code=train_future_code,
        query_future_code=query_future_code,
    )

    assert diagnostics.neighbor_radius[0] == pytest.approx(0.0)
    assert diagnostics.future_ambiguity[0] == pytest.approx(0.0)
    assert diagnostics.encoded_future_ambiguity is not None
    assert diagnostics.encoded_future_ambiguity[0] > 0.0
    assert diagnostics.latent_step_scale is not None
    assert diagnostics.latent_step_scale > 0.0
    assert diagnostics.summary()["encoded_future_ambiguity_median"] == pytest.approx(
        diagnostics.encoded_future_ambiguity[0]
    )


def test_history_nonimprovement_gate_uses_ci95_lower_bound() -> None:
    interval_crossing_threshold = {
        "ci95_lower": 0.79,
        "ci95_upper": 1.20,
    }
    interval_above_threshold = {
        "ci95_lower": 0.81,
        "ci95_upper": 1.20,
    }

    crossing_gate = _history_not_materially_better_gate(interval_crossing_threshold)
    above_gate = _history_not_materially_better_gate(interval_above_threshold)

    assert not crossing_gate["passed"]
    assert crossing_gate["value"] == pytest.approx(0.79)
    assert crossing_gate["comparison"] == ">"
    assert crossing_gate["threshold"] == pytest.approx(0.80)
    assert above_gate["passed"]


def test_conditioned_transition_code_includes_geometry_parameters_and_dt() -> None:
    x = np.array(
        [
            [0.125, 0.375, 0.625, 0.875],
            [1.125, 1.375, 1.625, 1.875],
        ],
        dtype=np.float32,
    )
    source = Euler1DNPZ(
        data=np.zeros((2, 3, 4, 3), dtype=np.float32),
        x=x,
        t=np.array([[0.0, 0.1, 0.3], [0.0, 0.1, 0.3]], dtype=np.float32),
        left_states=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        right_states=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )
    latent = np.array([[[20.0], [21.0]], [[30.0], [31.0]]])

    conditioned = _conditioned_transition_code(
        latent,
        source,
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([1, 2]),
    )

    assert conditioned.shape == (4, 10)
    np.testing.assert_allclose(conditioned[:, 0], [20.0, 21.0, 30.0, 31.0])
    np.testing.assert_allclose(conditioned[0, 1:7], [1, 2, 3, 7, 8, 9])
    np.testing.assert_allclose(conditioned[2, 1:7], [4, 5, 6, 10, 11, 12])
    np.testing.assert_allclose(conditioned[:, 7:9], [[0, 1], [0, 1], [1, 2], [1, 2]])
    np.testing.assert_allclose(conditioned[:, 9], [0.1, 0.2, 0.1, 0.2])


def test_history_pca_keeps_the_declared_current_code_size() -> None:
    time = np.linspace(-1.0, 1.0, 21)
    current = np.stack((time, np.square(time), np.sin(time)), axis=-1)
    previous = np.stack(
        (time - 0.1, np.square(time - 0.1), np.sin(time - 0.1)),
        axis=-1,
    )
    history = np.concatenate((current, previous), axis=-1)

    pca = LinearCodePCA.fit(history, output_size=current.shape[1])
    transformed = pca.transform(history)

    assert history.shape[1] == 2 * current.shape[1]
    assert pca.modes.shape == (current.shape[1], history.shape[1])
    assert transformed.shape == current.shape
    np.testing.assert_allclose(
        pca.modes @ pca.modes.T,
        np.eye(current.shape[1]),
        atol=1.0e-12,
    )


def test_conservative_decoder_enforces_coarse_means_without_positivity_repair() -> None:
    coarse = torch.tensor([[[1.0, 0.0, 2.5], [0.8, 0.1, 2.0]]], dtype=torch.float64)
    residual = torch.tensor(
        [
            [
                [2.0, 0.2, 0.0],
                [-4.0, -0.3, 0.0],
                [0.4, 0.1, 0.2],
                [-0.2, -0.1, -0.4],
            ]
        ],
        dtype=torch.float64,
    )
    volume = torch.tensor([[1.0, 3.0, 2.0, 1.0]], dtype=torch.float64)

    decoded = decode_coarse_conservative(
        coarse,
        residual,
        volume,
        constraint="coarse_conservative",
    )
    decoded_blocks = decoded.reshape(1, 2, 2, 3)
    volume_blocks = volume.reshape(1, 2, 2, 1)
    decoded_means = (decoded_blocks * volume_blocks).sum(dim=2) / volume_blocks.sum(
        dim=2
    )

    torch.testing.assert_close(decoded_means, coarse, rtol=0.0, atol=1.0e-14)
    assert decoded[0, 1, 0] < 0.0


def test_grouped_bootstrap_is_paired_deterministic_and_group_clustered() -> None:
    denominator = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    numerator = 0.5 * denominator
    groups = np.array([10, 10, 20, 20, 30, 30])

    first = grouped_bootstrap_median_ratio(
        numerator,
        denominator,
        groups,
        repetitions=500,
        seed=17,
    )
    second = grouped_bootstrap_median_ratio(
        numerator,
        denominator,
        groups,
        repetitions=500,
        seed=17,
    )

    assert first == second
    assert first["groups"] == 3
    assert first["repetitions"] == 500
    assert first["estimate"] == pytest.approx(0.5)
    assert first["ci95_lower"] == pytest.approx(0.5)
    assert first["ci95_upper"] == pytest.approx(0.5)

from pathlib import Path

import numpy as np
import torch

from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
    normal_flux_from_primitive,
    primitive_to_conservative,
    rusanov_flux_from_primitive,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DRolloutWindowDataset,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    collate_euler1d_rollout_windows,
    load_euler1d_npz,
)
from baselines.fno import SpectralConv1d
from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (
    CaseConfig,
    integrate_case,
)
from utility.time_dependent_no.euler1d_models import (
    CPGNetEuler1D,
    CPGStyleTargetEuler1DHead,
    CPGStylePilotEuler1DHead,
    FNOEuler1DHead,
    cell_features,
)
from utility.time_dependent_no.euler1d_targets import (
    CPGNetInterfaceTargetAdapter,
    ConservativeResidualTargetAdapter,
    ConservativeStateTargetAdapter,
    FluxTargetAdapter,
    InterfaceStateTargetAdapter,
    LimitedConservativeResidualTargetAdapter,
    LimitedFluxTargetAdapter,
    PhysicalFluxCorrectionTargetAdapter,
    PositiveLimitedInterfaceStateTargetAdapter,
    PrimitiveResidualTargetAdapter,
    ProjectedConservativeResidualTargetAdapter,
    RelativeInterfaceStateTargetAdapter,
    StateTargetAdapter,
    decode_relative_interface_traces,
    owner_neighbor_primitives,
)
from utility.time_dependent_no.fv import finite_volume_update


def _batch(num_cells=8):
    x = torch.linspace(0.0, 1.0, num_cells).unsqueeze(0)
    rho = 1.0 + 0.1 * x
    velocity = 0.2 + 0.05 * x
    pressure = 1.0 + 0.2 * x
    current = torch.stack((rho, velocity, pressure), dim=-1)
    target = current + torch.tensor([0.01, -0.02, 0.03])
    return make_euler1d_batch(
        current,
        x,
        torch.tensor([0.05]),
        target_primitive=target,
        left_boundary_primitive=current[:, 0],
        right_initial_primitive=current[:, -1],
    )


def test_euler1d_primitive_conservative_roundtrip_torch():
    primitive = torch.tensor([[[1.0, 0.2, 1.0], [0.8, -0.3, 0.5]]])

    conservative = primitive_to_conservative(primitive)
    recovered = conservative_to_primitive(conservative)

    torch.testing.assert_close(recovered, primitive)


def test_fno_cell_features_select_and_normalize_conservative_coordinates():
    batch = _batch(num_cells=6)
    mean = torch.tensor([[[0.5, -0.25, 1.0]]])
    std = torch.tensor([[[2.0, 4.0, 0.5]]])

    features = cell_features(
        batch,
        input_coordinates="conservative",
        input_mean=mean,
        input_std=std,
    )

    expected = (batch.current_conservative - mean) / std
    torch.testing.assert_close(features[..., :3], expected)
    torch.testing.assert_close(
        features[..., 3:],
        batch.geometry.cell_centers,
    )


def test_fno_conservative_features_use_retained_recurrent_state_exactly():
    batch = _batch(num_cells=6)
    retained = batch.current_conservative.clone()
    retained[..., 1] += 0.125
    retained_batch = make_euler1d_batch(
        conservative_to_primitive(retained),
        batch.geometry.cell_centers.squeeze(-1),
        batch.dt,
        current_conservative_state=retained,
    )

    features = cell_features(
        retained_batch,
        input_coordinates="conservative",
    )

    torch.testing.assert_close(features[..., :3], retained, rtol=0.0, atol=0.0)


def test_finite_volume_update_conserves_interior_flux_exchange():
    batch = _batch(num_cells=5)
    conservative = batch.current_conservative
    face_flux = torch.zeros(1, 6, 3)
    face_flux[:, 2] = torch.tensor([0.4, -0.1, 0.2])

    updated = finite_volume_update(conservative, face_flux, batch.geometry, batch.dt)
    before_total = (conservative * batch.geometry.cell_volume.unsqueeze(-1)).sum(dim=1)
    after_total = (updated * batch.geometry.cell_volume.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(after_total, before_total)


def test_state_target_adapter_decodes_direct_next_state():
    batch = _batch()
    adapter = StateTargetAdapter()

    prediction = adapter(batch.target_primitive, batch)

    torch.testing.assert_close(prediction.primitive, batch.target_primitive)
    torch.testing.assert_close(prediction.conservative, batch.target_conservative)


def test_conservative_state_target_adapter_decodes_direct_next_state():
    batch = _batch()
    adapter = ConservativeStateTargetAdapter()

    prediction = adapter(batch.target_conservative, batch)

    torch.testing.assert_close(prediction.conservative, batch.target_conservative)
    torch.testing.assert_close(prediction.primitive, batch.target_primitive)
    assert prediction.aux["raw_recurrence"] is True
    torch.testing.assert_close(
        prediction.aux["proposed_conservative"],
        batch.target_conservative,
    )


def test_conservative_state_target_adapter_exposes_nonpositive_raw_state():
    batch = _batch()
    raw_state = batch.target_conservative.clone()
    raw_state[..., 0] = -raw_state[..., 0]

    prediction = ConservativeStateTargetAdapter()(raw_state, batch)

    assert torch.any(prediction.primitive[..., 0] <= 0.0)
    assert prediction.aux["raw_recurrence"] is True


def test_conservative_residual_target_adapter_adds_current_state_delta():
    batch = _batch()
    adapter = ConservativeResidualTargetAdapter()
    raw_delta = batch.target_conservative - batch.current_conservative

    prediction = adapter(raw_delta, batch)

    torch.testing.assert_close(prediction.conservative, batch.target_conservative)
    torch.testing.assert_close(prediction.primitive, batch.target_primitive)


def test_conservative_residual_target_adapter_exposes_raw_nonpositive_state():
    batch = _batch()
    raw_delta = torch.zeros_like(batch.current_conservative)
    raw_delta[..., 0] = -2.0 * batch.current_conservative[..., 0]

    prediction = ConservativeResidualTargetAdapter()(raw_delta, batch)

    assert torch.any(prediction.primitive[..., 0] <= 0.0)
    assert prediction.aux["raw_recurrence"] is True
    torch.testing.assert_close(
        prediction.aux["proposed_conservative"],
        prediction.conservative,
    )


def test_projected_residual_matches_boundary_budget_and_shared_face_update():
    x = torch.tensor([[0.0, 0.07, 0.2, 0.45, 0.58, 0.83, 1.0]])
    current = torch.stack(
        (
            1.0 + 0.1 * x,
            0.2 + 0.05 * x,
            1.0 + 0.2 * x,
        ),
        dim=-1,
    )
    batch = make_euler1d_batch(current, x, torch.tensor([0.05]))
    torch.manual_seed(4)
    raw = torch.randn(1, 8, 3) * 0.02
    boundary_exchange = torch.tensor([[0.013, -0.021, 0.034]])
    raw[:, -1] = boundary_exchange

    prediction = ProjectedConservativeResidualTargetAdapter()(raw, batch)
    volume = batch.geometry.cell_volume.unsqueeze(-1)
    projected_delta = prediction.conservative - batch.current_conservative
    integrated_delta = (volume * projected_delta).sum(dim=1)

    torch.testing.assert_close(integrated_delta, boundary_exchange)
    reconstructed = finite_volume_update(
        batch.current_conservative,
        prediction.aux["face_flux"],
        batch.geometry,
        batch.dt,
    )
    torch.testing.assert_close(reconstructed, prediction.conservative)
    physical_flux = prediction.aux["face_flux"] * batch.geometry.face_normal
    torch.testing.assert_close(
        physical_flux.mean(dim=1),
        torch.zeros_like(boundary_exchange),
        atol=2.0e-7,
        rtol=0.0,
    )
    assert prediction.aux["raw_recurrence"] is True


def test_projected_residual_zero_output_is_identity_update():
    batch = _batch(num_cells=5)
    raw = torch.zeros(1, 6, 3)

    prediction = ProjectedConservativeResidualTargetAdapter()(raw, batch)

    torch.testing.assert_close(
        prediction.conservative,
        batch.current_conservative,
    )
    torch.testing.assert_close(
        prediction.aux["canonical_face_impulse"],
        torch.zeros(1, 6, 3),
    )


def test_primitive_residual_target_adapter_decodes_positive_flow_map():
    batch = _batch()
    rho_floor = 1.0e-8
    pressure_floor = 1.0e-8
    adapter = PrimitiveResidualTargetAdapter(
        rho_floor=rho_floor,
        pressure_floor=pressure_floor,
    )
    raw_delta = torch.empty_like(batch.current_primitive)
    raw_delta[..., 0] = torch.log(
        (batch.target_primitive[..., 0] - rho_floor)
        / (batch.current_primitive[..., 0] - rho_floor)
    )
    raw_delta[..., 1] = batch.target_primitive[..., 1] - batch.current_primitive[..., 1]
    raw_delta[..., 2] = torch.log(
        (batch.target_primitive[..., 2] - pressure_floor)
        / (batch.current_primitive[..., 2] - pressure_floor)
    )

    prediction = adapter(raw_delta, batch)

    torch.testing.assert_close(prediction.primitive, batch.target_primitive)
    torch.testing.assert_close(prediction.conservative, batch.target_conservative)

    unsafe_raw = torch.zeros_like(batch.current_primitive)
    unsafe_raw[..., 0] = -100.0
    unsafe_raw[..., 2] = -100.0
    floor_prediction = adapter(unsafe_raw, batch)
    assert torch.all(floor_prediction.primitive[..., 0] > 0.0)
    assert torch.all(floor_prediction.primitive[..., 2] > 0.0)


def test_limited_conservative_residual_target_adapter_preserves_admissibility():
    batch = _batch()
    adapter = LimitedConservativeResidualTargetAdapter(
        rho_floor=1.0e-6,
        pressure_floor=1.0e-6,
        safety=1.0,
    )
    safe_delta = batch.target_conservative - batch.current_conservative

    safe_prediction = adapter(safe_delta, batch)

    torch.testing.assert_close(safe_prediction.conservative, batch.target_conservative)
    torch.testing.assert_close(safe_prediction.primitive, batch.target_primitive)
    torch.testing.assert_close(
        safe_prediction.aux["limiter_theta"],
        torch.ones_like(safe_prediction.aux["limiter_theta"]),
    )

    unsafe_delta = torch.zeros_like(batch.current_conservative)
    unsafe_delta[..., 0] = -2.0 * batch.current_conservative[..., 0]
    unsafe_delta[..., 2] = -2.0 * batch.current_conservative[..., 2]

    limited_prediction = adapter(unsafe_delta, batch)
    proposed_primitive = conservative_to_primitive(
        limited_prediction.aux["proposed_conservative"]
    )
    raw_primitive = conservative_to_primitive(limited_prediction.conservative)

    assert torch.any(proposed_primitive[..., 0] <= 0.0)
    assert torch.any(proposed_primitive[..., 2] <= 0.0)
    assert not torch.allclose(
        limited_prediction.aux["proposed_conservative"],
        limited_prediction.conservative,
    )
    assert torch.all(raw_primitive[..., 0] > 0.0)
    assert torch.all(raw_primitive[..., 2] > 0.0)
    assert torch.any(limited_prediction.aux["limiter_theta"] < 1.0)


def test_flux_target_adapter_zero_flux_keeps_current_state():
    batch = _batch()
    adapter = FluxTargetAdapter()
    raw_flux = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)

    prediction = adapter(raw_flux, batch)

    torch.testing.assert_close(prediction.conservative, batch.current_conservative)
    torch.testing.assert_close(prediction.primitive, batch.current_primitive)
    torch.testing.assert_close(
        prediction.aux["proposed_conservative"],
        batch.current_conservative,
    )
    assert prediction.aux["raw_recurrence"] is True


def test_flux_target_adapter_exposes_inadmissible_raw_update_without_floor():
    batch = _batch(num_cells=5)
    adapter = FluxTargetAdapter()
    raw_flux = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)
    raw_flux[:, 1, 0] = 100.0

    prediction = adapter(raw_flux, batch)

    assert torch.any(prediction.primitive[..., 0] <= 0.0)
    torch.testing.assert_close(
        prediction.aux["proposed_conservative"],
        prediction.conservative,
    )
    assert prediction.aux["raw_recurrence"] is True


def test_limited_flux_target_adapter_zero_flux_keeps_current_state():
    batch = _batch()
    adapter = LimitedFluxTargetAdapter()
    raw_flux = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)

    prediction = adapter(raw_flux, batch)

    torch.testing.assert_close(prediction.conservative, batch.current_conservative)
    torch.testing.assert_close(prediction.primitive, batch.current_primitive)
    torch.testing.assert_close(
        prediction.aux["limiter_theta"],
        torch.ones_like(prediction.aux["limiter_theta"]),
    )


def test_limited_flux_target_adapter_limits_samplewise_and_preserves_total():
    batch = _batch(num_cells=5)
    adapter = LimitedFluxTargetAdapter(
        rho_floor=1.0e-6,
        pressure_floor=1.0e-6,
        safety=1.0,
    )
    raw_flux = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)
    raw_flux[:, 2] = torch.tensor([100.0, 0.0, 200.0])
    before_total = (
        batch.current_conservative * batch.geometry.cell_volume.unsqueeze(-1)
    ).sum(dim=1)

    prediction = adapter(raw_flux, batch)
    proposed_primitive = conservative_to_primitive(
        prediction.aux["proposed_conservative"]
    )
    primitive = conservative_to_primitive(prediction.conservative)
    after_total = (
        prediction.conservative * batch.geometry.cell_volume.unsqueeze(-1)
    ).sum(dim=1)

    assert prediction.aux["limiter_theta"].shape == (1,)
    assert torch.all(prediction.aux["limiter_theta"] < 1.0)
    assert torch.any(
        (proposed_primitive[..., 0] <= 0.0) | (proposed_primitive[..., 2] <= 0.0)
    )
    assert not torch.allclose(
        prediction.aux["proposed_conservative"],
        prediction.conservative,
    )
    assert torch.all(primitive[..., 0] > 0.0)
    assert torch.all(primitive[..., 2] > 0.0)
    torch.testing.assert_close(after_total, before_total)


def test_physical_flux_correction_zero_raw_matches_base_rusanov_update():
    batch = _batch()
    adapter = PhysicalFluxCorrectionTargetAdapter(safety=1.0)
    raw_correction = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)
    owner, neighbor = owner_neighbor_primitives(batch)
    base_flux = rusanov_flux_from_primitive(
        owner,
        neighbor,
        batch.geometry.face_normal,
        gamma=batch.gamma,
    )
    expected = finite_volume_update(
        batch.current_conservative,
        base_flux,
        batch.geometry,
        batch.dt,
    )

    prediction = adapter(raw_correction, batch)

    torch.testing.assert_close(prediction.aux["base_face_flux"], base_flux)
    torch.testing.assert_close(prediction.aux["face_flux"], base_flux)
    torch.testing.assert_close(
        prediction.aux["flux_correction"],
        torch.zeros_like(base_flux),
    )
    torch.testing.assert_close(prediction.conservative, expected)
    torch.testing.assert_close(
        prediction.aux["limiter_theta"],
        torch.ones_like(prediction.aux["limiter_theta"]),
    )


def test_physical_flux_correction_is_componentwise_bounded():
    batch = _batch()
    adapter = PhysicalFluxCorrectionTargetAdapter(correction_scale=0.25, safety=1.0)
    raw_correction = torch.full((1, batch.geometry.face_owner.shape[1], 3), 100.0)

    prediction = adapter(raw_correction, batch)
    correction = prediction.aux["flux_correction"]
    scale = prediction.aux["flux_correction_scale"]

    assert torch.all(correction.abs() <= 0.25 * scale + 1.0e-6)


def test_interface_state_adapter_matches_explicit_rusanov_update():
    batch = _batch()
    owner, neighbor = owner_neighbor_primitives(batch)
    raw_interface = torch.stack((owner, neighbor), dim=-2)
    adapter = InterfaceStateTargetAdapter()

    prediction = adapter(raw_interface, batch)
    flux = rusanov_flux_from_primitive(owner, neighbor, batch.geometry.face_normal)
    expected = finite_volume_update(
        batch.current_conservative, flux, batch.geometry, batch.dt
    )

    torch.testing.assert_close(prediction.conservative, expected)
    torch.testing.assert_close(prediction.aux["face_flux"], flux)


def test_relative_interface_zero_correction_matches_physical_rusanov_traces():
    batch = _batch()
    raw = torch.zeros(1, batch.geometry.face_owner.shape[1], 2, 3)
    owner, neighbor = owner_neighbor_primitives(batch)

    prediction = RelativeInterfaceStateTargetAdapter(flux_mode="rusanov")(
        raw,
        batch,
    )
    traces = prediction.aux["interface_primitive"]
    flux = rusanov_flux_from_primitive(
        owner,
        neighbor,
        batch.geometry.face_normal,
    )

    torch.testing.assert_close(traces[..., 0, :], owner)
    torch.testing.assert_close(traces[..., 1, :], neighbor)
    torch.testing.assert_close(prediction.aux["face_flux"], flux)
    assert prediction.aux["raw_recurrence"] is True


def test_relative_interface_uses_exponential_and_sound_speed_coordinates():
    batch = _batch()
    raw = torch.zeros(1, batch.geometry.face_owner.shape[1], 2, 3)
    raw[..., 0] = -2.0
    raw[..., 1] = 1.0
    raw[..., 2] = -3.0
    owner, _ = owner_neighbor_primitives(batch)
    traces = decode_relative_interface_traces(raw, batch)
    owner_trace = traces[..., 0, :]
    sound_speed = torch.sqrt(batch.gamma * owner[..., 2] / owner[..., 0])

    torch.testing.assert_close(
        owner_trace[..., 0],
        owner[..., 0] * torch.exp(torch.tensor(-2.0)),
    )
    torch.testing.assert_close(owner_trace[..., 1], owner[..., 1] + sound_speed)
    torch.testing.assert_close(
        owner_trace[..., 2],
        owner[..., 2] * torch.exp(torch.tensor(-3.0)),
    )
    assert torch.all(traces[..., 0] > 0.0)
    assert torch.all(traces[..., 2] > 0.0)
    torch.testing.assert_close(traces[:, 0, 1], batch.left_boundary_primitive)
    torch.testing.assert_close(
        traces[:, -1, 1, 1],
        -traces[:, -1, 0, 1],
    )


def test_relative_interface_central_decoder_matches_explicit_flux():
    batch = _batch()
    raw = torch.zeros(1, batch.geometry.face_owner.shape[1], 2, 3)
    prediction = RelativeInterfaceStateTargetAdapter(flux_mode="central")(
        raw,
        batch,
    )
    traces = prediction.aux["interface_primitive"]
    owner_flux = normal_flux_from_primitive(
        traces[..., 0, :],
        batch.geometry.face_normal,
    )
    neighbor_flux = normal_flux_from_primitive(
        traces[..., 1, :],
        batch.geometry.face_normal,
    )
    expected_flux = 0.5 * (owner_flux + neighbor_flux)

    torch.testing.assert_close(prediction.aux["face_flux"], expected_flux)


def test_positive_limited_interface_state_adapter_forces_positive_interface():
    batch = _batch()
    raw_interface = torch.full((1, batch.geometry.face_owner.shape[1], 2, 3), -50.0)
    adapter = PositiveLimitedInterfaceStateTargetAdapter(
        positive_transform="none",
        rho_floor=1.0e-6,
        pressure_floor=1.0e-6,
        safety=1.0,
    )

    prediction = adapter(raw_interface, batch)
    interface = prediction.aux["interface_primitive"]
    primitive = conservative_to_primitive(prediction.conservative)

    assert torch.all(interface[..., 0] > 0.0)
    assert torch.all(interface[..., 2] > 0.0)
    assert prediction.aux["limiter_theta"].shape == (1,)
    assert torch.all(prediction.aux["limiter_theta"] >= 0.0)
    assert torch.all(prediction.aux["limiter_theta"] <= 1.0)
    assert torch.all(primitive[..., 0] > 0.0)
    assert torch.all(primitive[..., 2] > 0.0)


def test_euler1d_npz_loader_and_collate():
    path = Path("artifacts/time_dependent_no/euler1d_test_fixture.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((2, 4, 4, 3), dtype=np.float32)
    data[..., 0] = 1.0
    data[..., 2] = 1.0
    x = np.tile(np.linspace(0.0, 1.0, 4, dtype=np.float32), (2, 1))
    t = np.tile(np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32), (2, 1))
    left = np.array([[1.0, 0.1, 1.0], [1.1, 0.2, 1.2]], dtype=np.float32)
    right = np.array([[0.5, 0.0, 0.5], [0.6, 0.0, 0.6]], dtype=np.float32)
    face_flux_integral = np.zeros((2, 3, 5, 3), dtype=np.float32)
    face_flux_integral[0, 0] = 0.1
    face_flux_integral[0, 1] = 0.2
    np.savez_compressed(
        path,
        data=data,
        x=x,
        t=t,
        left_states=left,
        right_states=right,
        gamma=np.array(1.4),
        variable_names=np.array(["rho", "u", "p"]),
        face_flux_integral=face_flux_integral,
    )

    loaded = load_euler1d_npz(path)
    dataset = Euler1DTimePairDataset(loaded)
    batch = collate_euler1d_pairs([dataset[0], dataset[1]])
    stride_dataset = Euler1DTimePairDataset(loaded, step_stride=2)
    stride_batch = collate_euler1d_pairs([stride_dataset[0]])
    rollout_dataset = Euler1DRolloutWindowDataset(loaded, rollout_steps=2)
    rollout_batch, rollout_targets, rollout_dt = collate_euler1d_rollout_windows(
        [rollout_dataset[0], rollout_dataset[1]]
    )
    burn_in_dataset = Euler1DRolloutWindowDataset(
        loaded,
        rollout_steps=2,
        burn_in_steps=1,
    )
    burn_in_batch, burn_in_targets, burn_in_dt = collate_euler1d_rollout_windows(
        [burn_in_dataset[0], burn_in_dataset[1]]
    )

    assert loaded.num_cases == 2
    assert len(dataset) == 6
    assert len(stride_dataset) == 4
    assert len(rollout_dataset) == 4
    assert len(burn_in_dataset) == 2
    assert batch.current_primitive.shape == (2, 4, 3)
    assert batch.geometry.face_owner.shape == (2, 5)
    torch.testing.assert_close(batch.target_face_flux[0], torch.ones(5, 3))
    torch.testing.assert_close(batch.target_face_flux[1], torch.full((5, 3), 2.0))
    torch.testing.assert_close(batch.dt, torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(
        batch.right_initial_primitive,
        torch.from_numpy(right[[0, 0]]),
    )
    torch.testing.assert_close(stride_batch.dt, torch.tensor([0.2]))
    torch.testing.assert_close(
        stride_batch.target_face_flux,
        torch.full((1, 5, 3), 1.5),
    )
    assert rollout_batch.current_primitive.shape == (2, 4, 3)
    assert rollout_targets.shape == (2, 2, 4, 3)
    torch.testing.assert_close(rollout_dt, torch.full((2, 2), 0.1))
    assert burn_in_batch.current_primitive.shape == (2, 4, 3)
    assert burn_in_targets.shape == (2, 3, 4, 3)
    torch.testing.assert_close(burn_in_dt, torch.full((2, 3), 0.1))


def test_ader_face_flux_integral_closes_saved_state_transition():
    case = CaseConfig(
        x_left=0.0,
        x_right=1.0,
        x_disc=0.35,
        left_state=np.array([1.0, 0.8, 1.0], dtype=np.float64),
        right_state=np.array([0.125, 0.0, 0.1], dtype=np.float64),
        t_final=0.02,
    )

    x, t, data, _, face_flux_integral, closure_max_abs = integrate_case(
        case,
        nx=32,
        n_steps=2,
        gamma=1.4,
        cfl=0.2,
        ng=3,
        rho_floor=1.0e-12,
        p_floor=1.0e-12,
        use_shock_flattening=True,
        use_hlle_on_troubled_faces=True,
        shock_sensor_threshold=0.05,
        shock_flatten_radius=4,
        return_face_flux_integral=True,
    )

    assert face_flux_integral.shape == (2, 33, 3)
    assert closure_max_abs < 1.0e-12
    dt = float(t[1] - t[0])
    batch = make_euler1d_batch(
        torch.from_numpy(data[0]).unsqueeze(0),
        torch.from_numpy(x).unsqueeze(0),
        dt,
        target_primitive=torch.from_numpy(data[1]).unsqueeze(0),
        target_face_flux=torch.from_numpy(face_flux_integral[0] / dt).unsqueeze(0),
        left_boundary_primitive=torch.from_numpy(
            case.left_state.astype(np.float32)
        ).unsqueeze(0),
        right_initial_primitive=torch.from_numpy(
            case.right_state.astype(np.float32)
        ).unsqueeze(0),
    )
    reconstructed = finite_volume_update(
        batch.current_conservative,
        batch.target_face_flux,
        batch.geometry,
        batch.dt,
    )

    torch.testing.assert_close(
        reconstructed,
        batch.target_conservative,
        rtol=3.0e-5,
        atol=3.0e-6,
    )

    macro_dt = float(t[2] - t[0])
    macro_batch = make_euler1d_batch(
        torch.from_numpy(data[0]).unsqueeze(0),
        torch.from_numpy(x).unsqueeze(0),
        macro_dt,
        target_primitive=torch.from_numpy(data[2]).unsqueeze(0),
        target_face_flux=torch.from_numpy(
            face_flux_integral.sum(axis=0) / macro_dt
        ).unsqueeze(0),
    )
    macro_reconstructed = finite_volume_update(
        macro_batch.current_conservative,
        macro_batch.target_face_flux,
        macro_batch.geometry,
        macro_batch.dt,
    )
    torch.testing.assert_close(
        macro_reconstructed,
        macro_batch.target_conservative,
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_cpg_style_pilot_head_output_shapes_are_resolution_flexible():
    for target, expected_channels in [
        ("state", 3),
        ("residual", 3),
        ("primitive_residual", 3),
        ("limited_residual", 3),
        ("flux", 3),
        ("limited_flux", 3),
        ("physical_flux_correction", 3),
    ]:
        head = CPGStylePilotEuler1DHead(
            target, hidden_dim=8, message_passing_steps=1, mlp_layers=2
        )
        for num_cells in [6, 9]:
            batch = _batch(num_cells=num_cells)
            raw = head(batch)
            count = (
                num_cells
                if target
                in ("state", "residual", "primitive_residual", "limited_residual")
                else num_cells + 1
            )
            assert raw.shape == (1, count, expected_channels)

    for target in ["interface", "relative_interface", "positive_limited_interface"]:
        head = CPGStylePilotEuler1DHead(
            target, hidden_dim=8, message_passing_steps=1, mlp_layers=2
        )
        raw = head(_batch(num_cells=7))
        assert raw.shape == (1, 8, 2, 3)


def test_cpg_style_target_head_output_shapes_are_resolution_flexible():
    for target, expected_channels in [
        ("limited_residual", 3),
        ("limited_flux", 3),
        ("physical_flux_correction", 3),
    ]:
        head = CPGStyleTargetEuler1DHead(
            target, hidden_dim=8, message_passing_steps=1, mlp_layers=2
        )
        for num_cells in [6, 9]:
            batch = _batch(num_cells=num_cells)
            raw = head(batch)
            count = num_cells if target == "limited_residual" else num_cells + 1
            assert raw.shape == (1, count, expected_channels)

    for target in ["interface", "relative_interface", "positive_limited_interface"]:
        head = CPGStyleTargetEuler1DHead(
            target, hidden_dim=8, message_passing_steps=1, mlp_layers=2
        )
        raw = head(_batch(num_cells=7))
        assert raw.shape == (1, 8, 2, 3)


def test_cpgnet_directed_message_passing_is_directional():
    torch.manual_seed(0)
    batch = _batch(num_cells=6)
    model = CPGNetEuler1D(
        hidden_dim=8,
        message_passing_steps=1,
        mlp_layers=2,
        edge_hidden_dim=4,
        edge_encoder_steps=1,
    )

    messages = model.directed_message_diagnostics(batch)
    assert not torch.allclose(
        messages["owner_side"][:, 1:-1],
        messages["neighbor_side"][:, 1:-1],
    )


def test_cpgnet_interior_face_cannot_see_beyond_message_passing_radius():
    torch.manual_seed(0)
    num_cells = 24
    message_steps = 2
    face = 12
    perturb_cell = 6
    x = torch.linspace(0.0, 1.0, num_cells).unsqueeze(0)
    current = torch.ones(1, num_cells, 3)
    current[..., 1] = 0.0
    perturbed = current.clone()
    perturbed[:, perturb_cell] = torch.tensor([1.8, 0.7, 2.2])
    boundary = torch.tensor([[1.0, 0.0, 1.0]])
    common = {
        "x": x,
        "dt": torch.tensor([0.05]),
        "left_boundary_primitive": boundary,
        "right_initial_primitive": boundary,
    }
    batch = make_euler1d_batch(current, **common)
    perturbed_batch = make_euler1d_batch(perturbed, **common)
    model = CPGNetEuler1D(
        hidden_dim=8,
        message_passing_steps=message_steps,
        mlp_layers=2,
        edge_hidden_dim=4,
        edge_encoder_steps=1,
    )

    baseline_interface = model(batch)
    perturbed_interface = model(perturbed_batch)

    torch.testing.assert_close(
        baseline_interface[:, face],
        perturbed_interface[:, face],
        rtol=0.0,
        atol=1.0e-7,
    )
    assert not torch.allclose(
        baseline_interface[:, perturb_cell + 1],
        perturbed_interface[:, perturb_cell + 1],
    )
    shift = face - perturb_cell
    baseline_shift_target = torch.roll(current[..., 0], shifts=shift, dims=1)
    perturbed_shift_target = torch.roll(perturbed[..., 0], shifts=shift, dims=1)
    assert (
        baseline_shift_target[:, face].item() != perturbed_shift_target[:, face].item()
    )


def test_cpgnet_decodes_positive_directed_states_with_physical_boundaries():
    batch = _batch(num_cells=5)
    model = CPGNetEuler1D(
        hidden_dim=8,
        message_passing_steps=1,
        mlp_layers=2,
        edge_hidden_dim=4,
        edge_encoder_steps=1,
    )

    interface = model(batch)

    assert interface.shape == (1, batch.geometry.face_owner.shape[1], 2, 3)
    assert torch.all(interface[..., 0] > 0.0)
    assert torch.all(interface[..., 2] > 0.0)
    torch.testing.assert_close(interface[:, 0, 1], batch.left_boundary_primitive)
    torch.testing.assert_close(interface[:, -1, 1, 0], interface[:, -1, 0, 0])
    torch.testing.assert_close(interface[:, -1, 1, 1], -interface[:, -1, 0, 1])
    torch.testing.assert_close(interface[:, -1, 1, 2], interface[:, -1, 0, 2])


def test_cpgnet_interface_adapter_matches_one_shared_rusanov_update():
    batch = _batch(num_cells=5)
    model = CPGNetEuler1D(
        hidden_dim=8,
        message_passing_steps=1,
        mlp_layers=2,
        edge_hidden_dim=4,
        edge_encoder_steps=1,
    )
    adapter = CPGNetInterfaceTargetAdapter()

    interface = model(batch)
    prediction = adapter(interface, batch)
    expected_flux = rusanov_flux_from_primitive(
        interface[..., 0, :],
        interface[..., 1, :],
        batch.geometry.face_normal,
        gamma=batch.gamma,
    )
    expected = finite_volume_update(
        batch.current_conservative,
        expected_flux,
        batch.geometry,
        batch.dt,
    )

    torch.testing.assert_close(prediction.aux["face_flux"], expected_flux)
    torch.testing.assert_close(prediction.conservative, expected)
    assert prediction.conservative.shape == batch.current_conservative.shape
    assert prediction.primitive.shape == batch.current_primitive.shape
    assert prediction.aux["raw_recurrence"] is True


def test_cpgnet_interface_adapter_does_not_hide_negative_cell_state():
    batch = _batch(num_cells=5)
    interface = torch.ones(1, 6, 2, 3)
    interface[..., 1] = 0.0
    interface[:, 2, :, 1] = 100.0

    prediction = CPGNetInterfaceTargetAdapter()(interface, batch)

    assert torch.any(prediction.primitive[..., 0] <= 0.0)


def test_spectral_conv1d_variable_width_forward_pass():
    conv = SpectralConv1d(in_channels=32, out_channels=64, modes1=5)
    x = torch.randn(2, 32, 16)

    y = conv(x)

    assert y.shape == (2, 64, 16)


def test_fno_head_variable_width_layers_forward_pass():
    head = FNOEuler1DHead(
        "limited_residual",
        modes=[4, 4, 4],
        width=64,
        layers=[32, 64, 64, 32],
        fc_dim=16,
    )

    raw = head(_batch(num_cells=12))

    assert raw.shape == (1, 12, 3)


def test_fno_head_output_shapes_are_resolution_flexible():
    for target in [
        "state",
        "conservative_state",
        "residual",
        "projected_residual",
        "primitive_residual",
        "limited_residual",
        "flux",
        "limited_flux",
        "physical_flux_correction",
        "interface",
        "relative_interface",
        "positive_limited_interface",
    ]:
        head = FNOEuler1DHead(
            target,
            modes=[3, 3, 3],
            width=8,
            layers=[8, 8, 8, 8],
            fc_dim=16,
        )
        for num_cells in [7, 10]:
            batch = _batch(num_cells=num_cells)
            raw = head(batch)
            if target == "projected_residual":
                assert raw.shape == (1, num_cells + 1, 3)
                volume = batch.geometry.cell_volume.unsqueeze(-1)
                torch.testing.assert_close(
                    (volume * raw[:, :-1]).sum(dim=1),
                    torch.zeros(1, 3),
                    atol=2.0e-6,
                    rtol=0.0,
                )
            elif target in (
                "state",
                "conservative_state",
                "residual",
                "primitive_residual",
                "limited_residual",
            ):
                assert raw.shape == (1, num_cells, 3)
            elif target in ("flux", "limited_flux", "physical_flux_correction"):
                assert raw.shape == (1, num_cells + 1, 3)
            else:
                assert raw.shape == (1, num_cells + 1, 2, 3)

from pathlib import Path

import numpy as np
import torch

from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
    rusanov_flux_from_primitive,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    load_euler1d_npz,
)
from utility.time_dependent_no.euler1d_models import CPGStyleEuler1DHead, FNOEuler1DHead
from utility.time_dependent_no.euler1d_targets import (
    ConservativeResidualTargetAdapter,
    FluxTargetAdapter,
    InterfaceStateTargetAdapter,
    StateTargetAdapter,
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
    )


def test_euler1d_primitive_conservative_roundtrip_torch():
    primitive = torch.tensor([[[1.0, 0.2, 1.0], [0.8, -0.3, 0.5]]])

    conservative = primitive_to_conservative(primitive)
    recovered = conservative_to_primitive(conservative)

    torch.testing.assert_close(recovered, primitive)


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


def test_conservative_residual_target_adapter_adds_current_state_delta():
    batch = _batch()
    adapter = ConservativeResidualTargetAdapter()
    raw_delta = batch.target_conservative - batch.current_conservative

    prediction = adapter(raw_delta, batch)

    torch.testing.assert_close(prediction.conservative, batch.target_conservative)
    torch.testing.assert_close(prediction.primitive, batch.target_primitive)


def test_flux_target_adapter_zero_flux_keeps_current_state():
    batch = _batch()
    adapter = FluxTargetAdapter()
    raw_flux = torch.zeros(1, batch.geometry.face_owner.shape[1], 3)

    prediction = adapter(raw_flux, batch)

    torch.testing.assert_close(prediction.conservative, batch.current_conservative)
    torch.testing.assert_close(prediction.primitive, batch.current_primitive)


def test_interface_state_adapter_matches_explicit_rusanov_update():
    batch = _batch()
    owner, neighbor = owner_neighbor_primitives(batch)
    raw_interface = torch.stack((owner, neighbor), dim=-2)
    adapter = InterfaceStateTargetAdapter()

    prediction = adapter(raw_interface, batch)
    flux = rusanov_flux_from_primitive(owner, neighbor, batch.geometry.face_normal)
    expected = finite_volume_update(batch.current_conservative, flux, batch.geometry, batch.dt)

    torch.testing.assert_close(prediction.conservative, expected)
    torch.testing.assert_close(prediction.aux["face_flux"], flux)


def test_euler1d_npz_loader_and_collate():
    path = Path("artifacts/time_dependent_no/euler1d_test_fixture.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((2, 3, 4, 3), dtype=np.float32)
    data[..., 0] = 1.0
    data[..., 2] = 1.0
    x = np.tile(np.linspace(0.0, 1.0, 4, dtype=np.float32), (2, 1))
    t = np.tile(np.array([0.0, 0.1, 0.2], dtype=np.float32), (2, 1))
    left = np.array([[1.0, 0.1, 1.0], [1.1, 0.2, 1.2]], dtype=np.float32)
    right = np.array([[0.5, 0.0, 0.5], [0.6, 0.0, 0.6]], dtype=np.float32)
    np.savez_compressed(
        path,
        data=data,
        x=x,
        t=t,
        left_states=left,
        right_states=right,
        gamma=np.array(1.4),
        variable_names=np.array(["rho", "u", "p"]),
    )

    loaded = load_euler1d_npz(path)
    dataset = Euler1DTimePairDataset(loaded)
    batch = collate_euler1d_pairs([dataset[0], dataset[1]])
    stride_dataset = Euler1DTimePairDataset(loaded, step_stride=2)
    stride_batch = collate_euler1d_pairs([stride_dataset[0]])

    assert loaded.num_cases == 2
    assert len(dataset) == 4
    assert len(stride_dataset) == 2
    assert batch.current_primitive.shape == (2, 4, 3)
    assert batch.geometry.face_owner.shape == (2, 5)
    torch.testing.assert_close(batch.dt, torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(stride_batch.dt, torch.tensor([0.2]))


def test_cpg_style_head_output_shapes_are_resolution_flexible():
    for target, expected_channels in [("state", 3), ("residual", 3), ("flux", 3)]:
        head = CPGStyleEuler1DHead(target, hidden_dim=8, message_passing_steps=1, mlp_layers=2)
        for num_cells in [6, 9]:
            batch = _batch(num_cells=num_cells)
            raw = head(batch)
            count = num_cells if target in ("state", "residual") else num_cells + 1
            assert raw.shape == (1, count, expected_channels)

    head = CPGStyleEuler1DHead("interface", hidden_dim=8, message_passing_steps=1, mlp_layers=2)
    raw = head(_batch(num_cells=7))
    assert raw.shape == (1, 8, 2, 3)


def test_fno_head_output_shapes_are_resolution_flexible():
    for target in ["state", "residual", "flux", "interface"]:
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
            if target in ("state", "residual"):
                assert raw.shape == (1, num_cells, 3)
            elif target == "flux":
                assert raw.shape == (1, num_cells + 1, 3)
            else:
                assert raw.shape == (1, num_cells + 1, 2, 3)

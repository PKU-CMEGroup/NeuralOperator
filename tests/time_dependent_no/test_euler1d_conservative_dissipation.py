from __future__ import annotations

import numpy as np
import torch

from scripts.time_dependent_no.probe_euler1d_conservative_dissipation import (
    bootstrap_mean_ci,
    diffusive_face_flux,
    pressure_front_effective_width_np,
)
from utility.time_dependent_no.euler1d import make_uniform_1d_geometry
from utility.time_dependent_no.fv import finite_volume_update


def test_diffusive_face_flux_is_conservative_laplacian() -> None:
    conservative = torch.tensor([[[0.0], [1.0], [0.0]]])
    geometry = make_uniform_1d_geometry(torch.tensor([[0.5, 1.5, 2.5]]))
    correction = diffusive_face_flux(
        conservative,
        geometry.cell_volume,
        torch.tensor([1.0]),
        0.1,
    )

    assert torch.equal(correction[:, 0], torch.zeros_like(correction[:, 0]))
    assert torch.equal(correction[:, -1], torch.zeros_like(correction[:, -1]))
    updated = finite_volume_update(
        conservative,
        correction,
        geometry,
        torch.tensor([1.0]),
    )
    expected = conservative + torch.tensor([[[0.1], [-0.2], [0.1]]])
    assert torch.allclose(updated, expected)
    assert torch.allclose(updated.sum(dim=1), conservative.sum(dim=1))


def test_pressure_front_effective_width_detects_smearing() -> None:
    x = np.arange(5, dtype=np.float64)
    sharp = np.zeros((1, 5, 3), dtype=np.float64)
    sharp[0, :, 2] = [0.0, 0.0, 1.0, 1.0, 1.0]
    smeared = sharp.copy()
    smeared[0, :, 2] = [0.0, 0.5, 1.0, 1.0, 1.0]

    sharp_width = pressure_front_effective_width_np(
        sharp,
        x,
        num_fronts=1,
    )
    smeared_width = pressure_front_effective_width_np(
        smeared,
        x,
        num_fronts=1,
    )

    assert np.allclose(sharp_width, 1.0)
    assert np.allclose(smeared_width, 2.0)


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    first = bootstrap_mean_ci(values, replicates=500, seed=7)
    second = bootstrap_mean_ci(values, replicates=500, seed=7)

    assert first == second
    assert first[0] < values.mean() < first[1]

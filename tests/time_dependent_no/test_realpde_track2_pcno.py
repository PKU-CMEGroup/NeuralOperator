from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

import scripts.time_dependent_no.train_realpde_track2_pcno as training
from pcno.pcno import compute_gradient
from utility.time_dependent_no.realpde_track2 import (
    RealPDETrack2PCNO,
    Track2Normalization,
    Track2WindowDataset,
    build_track2_geometry,
    build_track2_model_from_payload,
    discover_track2_files,
    estimate_track2_residual_scale,
    fit_track2_frame_normalization,
    load_track2_geometry_from_simulations,
    split_track2_files,
    track2_error_tensors,
)


def _track2_arrays(
    *,
    frames: int = 60,
    height: int = 8,
    width: int = 12,
    reynolds: int = 5000,
    angle: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-0.25, 0.45, width)
    y_axis = np.linspace(-0.15, 0.15, height)
    x = np.tile(x_axis[None, :], (height, 1))
    y = np.tile(y_axis[:, None], (1, width))
    time = np.arange(frames, dtype=np.float64)[:, None, None]
    phase = 0.013 * time + 0.00001 * reynolds + 0.02 * angle
    u = 0.8 + 0.03 * x[None] + 0.01 * y[None] + 0.02 * np.sin(phase + x[None])
    v = 0.01 * angle + 0.02 * y[None] + 0.01 * np.cos(phase - y[None])
    solid_column = 4 if angle == 0 else 6
    u[:, 4, solid_column] = 0.0
    v[:, 4, solid_column] = 0.0
    return x, y, u.astype(np.float32), v.astype(np.float32)


def _write_track2_file(
    path: Path,
    *,
    reynolds: int,
    angle: int,
    frames: int = 60,
) -> Path:
    x, y, u, v = _track2_arrays(
        frames=frames,
        reynolds=reynolds,
        angle=angle,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("re", data=reynolds)
        handle.create_dataset("aoa", data=angle)
    return path


def _prepare_files(directory: Path) -> list[Path]:
    directory.mkdir()
    return [
        _write_track2_file(
            directory / f"{reynolds}_{angle}.h5",
            reynolds=reynolds,
            angle=angle,
        )
        for angle in (0, 5)
        for reynolds in (5000, 6000)
    ]


def test_geometry_contract_and_gradient_are_physical() -> None:
    x, y, u, v = _track2_arrays(frames=2)
    solid = np.logical_and(np.all(u == 0.0, axis=0), np.all(v == 0.0, axis=0))
    geometry = build_track2_geometry(
        x[::2, ::2],
        y[::2, ::2],
        solid[::2, ::2],
        collar_width=0.03,
    )

    assert geometry.static_features.shape == (24, 9)
    assert geometry.feature_names[-5:] == (
        "inflow_collar",
        "outflow_collar",
        "bottom_farfield_collar",
        "top_farfield_collar",
        "airfoil_proximity",
    )
    assert np.isclose(geometry.node_weights.sum(), 1.0)
    assert np.all(geometry.node_weights > 0.0)
    assert geometry.contract["solid_policy"].startswith(
        "hard_zero_velocity_on_input_persistent_zero"
    )
    assert geometry.contract["quadrature_policy"].startswith("per_sample")

    field = 3.0 * geometry.nodes[:, 0] + 2.0 * geometry.nodes[:, 1]
    gradient = compute_gradient(
        torch.as_tensor(field).reshape(1, 1, -1),
        torch.as_tensor(geometry.directed_edges).unsqueeze(0),
        torch.as_tensor(geometry.edge_gradient_weights).unsqueeze(0),
    )
    expected = torch.tensor([3.0, 2.0]).reshape(1, 2, 1).expand_as(gradient)
    torch.testing.assert_close(gradient, expected, atol=2.0e-5, rtol=0.0)


def test_zero_initialized_model_is_exact_raw_persistence_with_solid_fill() -> None:
    x, y, u, v = _track2_arrays(frames=2)
    solid = np.logical_and(np.all(u == 0.0, axis=0), np.all(v == 0.0, axis=0))
    geometry = build_track2_geometry(
        x[::2, ::2],
        y[::2, ::2],
        solid[::2, ::2],
    )
    normalization = Track2Normalization(
        input_mean=np.array([0.7, 0.02]),
        target_mean=np.array([0.8, -0.01]),
        input_std=np.array([0.2, 0.05]),
        target_std=np.array([0.3, 0.08]),
    )
    model = RealPDETrack2PCNO(
        normalization=normalization,
        geometry=geometry,
        residual_scale=np.ones((20, 2), dtype=np.float32),
        n_modes=(1, 1),
        layers=(4, 4),
        fc_dim=4,
    )
    generator = torch.Generator().manual_seed(9)
    raw = torch.randn(
        2,
        20,
        geometry.height,
        geometry.width,
        2,
        generator=generator,
    )
    observed_solid = (
        geometry.static_features[:, geometry.feature_names.index("solid_mask")]
        .reshape(geometry.height, geometry.width)
        .astype(bool)
    )
    raw[:, :, observed_solid] = 0.0
    input_norm = model.raw_to_input(raw)
    prediction_raw = model.target_to_raw(model(input_norm))
    expected = raw[:, -1:, ...].expand_as(prediction_raw).clone()
    torch.testing.assert_close(prediction_raw, expected, atol=2.0e-6, rtol=0.0)
    inferred_solid = model.input_solid_mask(input_norm)
    torch.testing.assert_close(
        inferred_solid,
        torch.as_tensor(observed_solid).expand_as(inferred_solid),
    )
    with_pressure = model.forward_with_pressure(input_norm)
    assert with_pressure.shape[-1] == 3
    assert torch.count_nonzero(with_pressure[..., 2]) == 0

    dynamic_features = 42
    features = torch.randn(
        2,
        geometry.height * geometry.width,
        dynamic_features + len(geometry.feature_names),
        generator=generator,
    )
    ordinary = model.backbone(features, model._expanded_geometry(2))
    cached = model.backbone(
        features,
        model._expanded_geometry(2),
        fourier_tensors=model._expanded_fourier(2),
    )
    torch.testing.assert_close(cached, ordinary)


def test_data_split_normalization_windows_and_metrics(tmp_path: Path) -> None:
    files = _prepare_files(tmp_path / "data")
    metadata = discover_track2_files(files[0].parent)
    train_metadata, validation_metadata = split_track2_files(
        metadata,
        validation_fraction=0.25,
        seed=17,
    )
    assert len(train_metadata) == len(validation_metadata) == 2
    assert {item.angle_of_attack for item in train_metadata} == {0, 5}
    assert {item.angle_of_attack for item in validation_metadata} == {0, 5}

    normalization = fit_track2_frame_normalization(train_metadata)
    geometry = load_track2_geometry_from_simulations(files)
    assert geometry.contract["solid_support_source_trajectory_count"] == 4
    dataset = Track2WindowDataset(
        train_metadata,
        normalization,
        stride=7,
        expected_geometry=geometry,
        validate_solid=True,
    )
    input_norm, target_norm = dataset[0]
    assert input_norm.shape == target_norm.shape == (20, 4, 6, 2)
    scale = estimate_track2_residual_scale(dataset, maximum_windows=3)
    assert scale.shape == (20, 2)
    assert np.all(scale > 0.0)
    zero_errors = track2_error_tensors(
        target_norm.unsqueeze(0),
        target_norm.unsqueeze(0),
        normalization,
    )
    for value in zero_errors.values():
        torch.testing.assert_close(value, torch.zeros_like(value))


def _tiny_training_args(
    *,
    stage: str,
    data_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "--stage",
        stage,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--validation-fraction",
        "0.25",
        "--maximum-files",
        "4",
        "--window-stride",
        "8",
        "--modes",
        "1",
        "1",
        "--layers",
        "4",
        "4",
        "--fc-dim",
        "4",
        "--batch-size",
        "1",
        "--max-updates",
        "1",
        "--warmup-fraction",
        "0",
        "--residual-scale-windows",
        "3",
        "--evaluation-every",
        "1",
        "--evaluation-windows",
        "2",
        "--evaluation-batch-size",
        "1",
        "--rollout-trajectories",
        "1",
        "--rollout-blocks",
        "1",
        "--free-rollout-loss-every",
        "1",
        "--checkpoint-every",
        "1",
        "--device",
        "cpu",
        "--amp",
        "none",
    ]


def test_tiny_sim_to_real_training_and_checkpoint_reconstruction(
    tmp_path: Path,
) -> None:
    files = _prepare_files(tmp_path / "data")
    simulation_output = tmp_path / "simulation"
    training.main(
        _tiny_training_args(
            stage="sim_pretrain",
            data_dir=files[0].parent,
            output_dir=simulation_output,
        )
    )
    simulation_summary = json.loads(
        (simulation_output / "summary.json").read_text(encoding="utf-8")
    )
    assert simulation_summary["updates"] == 1
    assert (simulation_output / "last_training.pt").is_file()
    metric_lines = (
        (simulation_output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert "attached_second_call_loss" in json.loads(metric_lines[-1])["train"]

    statistics_path = tmp_path / "mean_std_real.pt"
    torch.save(
        (
            torch.tensor([0.80, 0.01, 0.0]),
            torch.tensor([0.81, 0.02, 0.0]),
            torch.tensor([0.20, 0.05, 0.0]),
            torch.tensor([0.21, 0.06, 0.0]),
        ),
        statistics_path,
    )
    real_output = tmp_path / "real"
    arguments = _tiny_training_args(
        stage="real_finetune",
        data_dir=files[0].parent,
        output_dir=real_output,
    )
    arguments.extend(
        [
            "--geometry-simulation",
            str(files[0]),
            "--normalization-path",
            str(statistics_path),
            "--init-checkpoint",
            str(simulation_output / "best_model.pt"),
        ]
    )
    training.main(arguments)

    payload = training.load_mapping(real_output / "best_model.pt")
    assert payload["target_contract"]["conservative_variables_available"] is False
    assert payload["parent"]["source_stage"] == "sim_pretrain"
    model = build_track2_model_from_payload(payload)
    assert not model.training
    assert next(model.parameters()).dtype == torch.float32
    assert (real_output / "summary.json").is_file()

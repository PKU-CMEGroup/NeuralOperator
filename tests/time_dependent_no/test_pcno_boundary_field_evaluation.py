from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_boundary_fields import (
    LoadedArm,
    Variant,
    _activation_rows_and_maps,
    _dynamic_resolution,
    _features_for_variant,
    _hook_equivalence,
    _model_call,
    _physical_wavelength_metrics,
    _variants,
)
from scripts.time_dependent_no.visualize_pcno_boundary_fields import (
    SENSITIVITY_FIELDS,
    animation_frame_indices,
    conservative_field,
    render_animation,
    render_sensitivity,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_FIELD_SEMANTIC_COLLAR,
    NODE_TYPE_FEATURE_OMITTED,
    Euler2DNormalization,
    PCNOEuler2DResidual,
)


def _semantic_model_and_sample() -> (
    tuple[PCNOEuler2DResidual, torch.Tensor, dict[str, torch.Tensor]]
):
    torch.manual_seed(20260806)
    model = PCNOEuler2DResidual(
        normalization=Euler2DNormalization(
            state_mean=np.asarray([1.0, 1.0, 0.0, 3.0]),
            state_scale=np.asarray([0.2, 0.3, 0.1, 0.5]),
            residual_scale=np.asarray([0.01, 0.02, 0.02, 0.04]),
            mach_mean=1.5,
            mach_scale=0.2,
        ),
        k_max=1,
        domain_lengths=(2.0, 1.0),
        layers=(8, 8, 8, 8, 8),
        fc_dim=7,
        zero_initialize=False,
        node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
        boundary_field_mode=BOUNDARY_FIELD_SEMANTIC_COLLAR,
        boundary_field_names=("y_symmetry", "x_extrapolation"),
    ).eval()
    nodes = torch.tensor(
        [
            [0.1, 0.1],
            [0.5, 0.1],
            [0.9, 0.1],
            [0.1, 0.7],
            [0.5, 0.7],
            [0.9, 0.7],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    directed_edges = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [0, 3],
            [3, 0],
            [1, 4],
            [4, 1],
            [2, 5],
            [5, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
        ],
        dtype=torch.int64,
    ).unsqueeze(0)
    primitive = torch.tensor(
        [
            [1.1, 1.2, 0.01, 1.0],
            [1.0, 1.1, 0.02, 1.0],
            [0.9, 1.0, 0.03, 0.9],
            [1.2, 1.3, -0.01, 1.1],
            [1.1, 1.2, -0.02, 1.0],
            [1.0, 1.1, -0.03, 0.9],
        ],
        dtype=torch.float32,
    )
    rho, velocity_x, velocity_y, pressure = primitive.unbind(dim=-1)
    energy = pressure / 0.4 + 0.5 * rho * (velocity_x.square() + velocity_y.square())
    current = torch.stack(
        (rho, rho * velocity_x, rho * velocity_y, energy), dim=-1
    ).unsqueeze(0)
    generator = torch.Generator().manual_seed(20260806)
    sample = {
        "node_mask": torch.ones((1, 6, 1)),
        "nodes": nodes,
        "node_weights": torch.full((1, 6, 1), 1.0 / 6.0),
        "node_rhos": torch.ones((1, 6, 1)),
        "directed_edges": directed_edges,
        "edge_gradient_weights": torch.randn(
            (1, directed_edges.shape[1], 2), generator=generator
        ),
        "node_type": torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.int64),
        "mach": torch.tensor([1.6]),
        "boundary_features": torch.tensor(
            [
                [
                    [0.7, 0.0],
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 0.5],
                    [0.0, 0.0],
                    [0.1, 1.0],
                ]
            ],
            dtype=torch.float32,
        ),
    }
    return model, current, sample


def test_variant_matrix_factorizes_semantic_interventions() -> None:
    variants = _variants(("y_symmetry", "x_extrapolation"))
    assert [variant.name for variant in variants] == [
        "N0_correct",
        "G1_correct",
        "G1_zero_all",
        "S1_correct",
        "S1_zero_all",
        "S1_zero_y_symmetry",
        "S1_zero_x_extrapolation",
    ]
    assert variants[-2].zero_field_indices == (0,)
    assert variants[-1].zero_field_indices == (1,)


def test_boundary_field_interventions_do_not_mutate_the_sample() -> None:
    _, _, sample = _semantic_model_and_sample()
    original = sample["boundary_features"].clone()
    zero_y = _features_for_variant(sample, Variant("S1", "zero_y", (0,)))
    zero_all = _features_for_variant(sample, Variant("G1", "zero_all", (0, 1)))
    assert zero_y is not None and zero_all is not None
    torch.testing.assert_close(zero_y[..., 0], torch.zeros_like(zero_y[..., 0]))
    torch.testing.assert_close(zero_y[..., 1], original[..., 1])
    torch.testing.assert_close(zero_all, torch.zeros_like(zero_all))
    torch.testing.assert_close(sample["boundary_features"], original)
    assert _features_for_variant(sample, Variant("N0", "correct")) is None


@torch.no_grad()
def test_hooks_are_equivalent_and_boundary_lift_identity_is_analytic() -> None:
    model, current, sample = _semantic_model_and_sample()
    arm = LoadedArm(
        arm="S1",
        run_dir=Path("synthetic"),
        summary={},
        split={},
        run_contract={},
        checkpoint={},
        model=model,
        checkpoint_sha256="0" * 64,
    )
    gate = _hook_equivalence(
        arm,
        sample,
        current[0].numpy(),
        boundary_policy=None,
        device=torch.device("cpu"),
        amp="none",
        absolute_limit=1.0e-7,
        relative_limit=1.0e-8,
    )
    assert gate["passed"] is True

    correct_fields = sample["boundary_features"]
    zero_fields = torch.zeros_like(correct_fields)
    correct_prediction, _, _, correct_trace = _model_call(
        model,
        sample,
        current,
        boundary_features=correct_fields,
        boundary_policy=None,
        device=torch.device("cpu"),
        amp="none",
        trace=True,
    )
    zero_prediction, _, _, zero_trace = _model_call(
        model,
        sample,
        current,
        boundary_features=zero_fields,
        boundary_policy=None,
        device=torch.device("cpu"),
        amp="none",
        trace=True,
    )
    assert not torch.equal(correct_prediction, zero_prediction)
    assert correct_trace is not None and zero_trace is not None
    rows, maps = _activation_rows_and_maps(
        model,
        correct_trace,
        zero_trace,
        correct_features=correct_fields,
        intervened_features=zero_fields,
        weights=np.full(6, 1.0 / 6.0),
        regions={"all": np.ones(6, dtype=bool)},
        common={"case_id": "synthetic"},
    )
    analytical = [
        row for row in rows if row["field"] == "analytical_full_post_lift_check"
    ][0]
    assert analytical["maximum_absolute_error"] <= 2.0e-7
    boundary_only = [
        row
        for row in rows
        if row["field"] == "analytical_boundary_only_post_lift_check"
    ][0]
    assert boundary_only["same_nonboundary_input"] is True
    assert boundary_only["maximum_absolute_error"] <= 2.0e-7
    np.testing.assert_allclose(
        maps["post_lift_difference_norm"],
        maps["analytical_boundary_lift_norm"],
        atol=2.0e-7,
        rtol=1.0e-6,
    )


def test_dynamic_grid_and_physical_frequency_contract() -> None:
    x = (np.arange(8) + 0.5) * 2.0 / 8
    y = (np.arange(4) + 0.5) / 4
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    assert _dynamic_resolution(nodes) == (8, 4)
    reference = np.stack(
        (
            1.0 + 0.1 * np.sin(4.0 * np.pi * nodes[:, 0]),
            np.ones(nodes.shape[0]),
            np.zeros(nodes.shape[0]),
            3.0 * np.ones(nodes.shape[0]),
        ),
        axis=-1,
    )
    metrics = _physical_wavelength_metrics(
        reference,
        reference,
        resolution=(8, 4),
        component_scale=np.ones(4),
        wavelength_min=0.125,
        wavelength_max=0.5,
    )
    assert metrics["metric_kind"] == "physical_wavelength_band"
    assert metrics["mode_count"] > 0
    assert metrics["relative_spectral_l2"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="tensor-product"):
        _dynamic_resolution(nodes[:-1])


def _write_animation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    nodes = np.asarray(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        dtype=np.float32,
    )
    reference = np.zeros((3, 4, 4), dtype=np.float32)
    reference[..., 0] = np.asarray([1.0, 1.1, 1.2])[:, None]
    reference[..., 1] = reference[..., 0]
    reference[..., 3] = 3.0
    arrays: dict[str, object] = {
        "schema": np.asarray("pcno_boundary_field_evaluation_v1"),
        "family": np.asarray("dynamic_fv"),
        "seed": np.asarray(19),
        "case_id": np.asarray("synthetic"),
        "resolution": np.asarray("2x2"),
        "physical_times": np.asarray([0.0, 0.1, 0.2]),
        "nodes": nodes,
        "physical_node_type": np.asarray([3, 1, 2, 3]),
        "boundary_distance": np.zeros(4, dtype=np.float32),
        "boundary_features": np.asarray(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        "boundary_field_names": np.asarray(["y_symmetry", "x_extrapolation"]),
        "reference_states": reference,
        "visualization_node_indices": np.arange(4),
        "visualization_only_subsampling": np.asarray(False),
        "requested_frame_count": np.asarray(3),
        "all_reference_frames_included": np.asarray(True),
        "post_failure_values": np.asarray("NaN"),
    }
    for mode in ("teacher_forced", "free_rollout"):
        for name, offset in (
            ("N0_correct", 0.01),
            ("G1_correct", 0.005),
            ("G1_zero_all", 0.008),
            ("S1_correct", 0.002),
            ("S1_zero_all", 0.007),
        ):
            value = reference.copy()
            value[1:, :, 0] += offset
            arrays[f"{mode}_{name}"] = value
    bundle = tmp_path / "bundle.npz"
    np.savez_compressed(bundle, **arrays)
    scales = tmp_path / "scales.json"
    scales.write_text(
        json.dumps(
            {
                "outcome_independent": True,
                "fields": {
                    "density": {
                        "minimum": 0.9,
                        "maximum": 1.3,
                        "error_abs_max": 0.05,
                        "difference_abs_max": 0.1,
                        "scale_source": "synthetic reference",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return bundle, scales


def test_animation_includes_every_frame_and_writes_hash_manifest(
    tmp_path: Path,
) -> None:
    bundle, scales = _write_animation_fixture(tmp_path)
    assert animation_frame_indices(3) == [0, 1, 2]
    assert conservative_field(
        np.load(bundle)["reference_states"], field="density"
    ).shape == (
        3,
        4,
    )
    output = tmp_path / "movie"
    manifest = render_animation(
        SimpleNamespace(
            bundle=bundle,
            reference_scales=scales,
            output_stem=output,
            kind="comparison",
            mode="free_rollout",
            field="density",
            gamma=1.4,
            units="s",
            fps=2,
            dpi=40,
            format="gif",
            overwrite=False,
        )
    )
    assert manifest["frame_indices"] == [0, 1, 2]
    assert manifest["all_frames_included"] is True
    assert output.with_suffix(".gif").is_file()
    assert (tmp_path / "movie_final.png").is_file()
    assert (tmp_path / "movie_final.pdf").is_file()


def test_sensitivity_renderer_uses_one_shared_scale(tmp_path: Path) -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    arrays: dict[str, object] = {
        "schema": np.asarray("pcno_boundary_field_evaluation_v1"),
        "family": np.asarray("dynamic_fv"),
        "case_id": np.asarray("synthetic"),
        "mode": np.asarray("free_rollout"),
        "call": np.asarray(2),
        "nodes": nodes,
    }
    for index, (name, _) in enumerate(SENSITIVITY_FIELDS, 1):
        arrays[name] = np.linspace(0.0, float(index), 4, dtype=np.float32)
    trace = tmp_path / "trace.npz"
    np.savez_compressed(trace, **arrays)
    output = tmp_path / "sensitivity"
    manifest = render_sensitivity(
        SimpleNamespace(
            trace=trace,
            output_stem=output,
            max_nodes=4,
            dpi=50,
            overwrite=False,
        )
    )
    assert manifest["scale"]["kind"] == "shared_log_norm_across_all_panels"
    assert output.with_suffix(".png").is_file()
    assert output.with_suffix(".pdf").is_file()

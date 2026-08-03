from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import (
    SOURCE_SNAPSHOT_SCHEMA,
    CaseData,
    _evaluate_mode,
    _hook_equivalence,
    _intervention_contract_rows,
    _intervention_specs,
    _load_bump_cases,
    _save_animation_payload,
    _snapshot_project_sources,
    _verify_source_snapshot,
)
from scripts.time_dependent_no.visualize_pcno_node_type_interventions import (
    SENSITIVITY_FIELDS,
    animation_frame_indices,
    conservative_field,
    render_animation,
    render_sensitivity,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)
from utility.time_dependent_no.pcno_node_type_interpretability import (
    PCNOActivationRecorder,
    activation_differences,
    node_type_intervention_name,
    replace_node_types_with_normal,
    tensor_equivalence_metrics,
    type_lift_delta,
)

ROOT = Path(__file__).resolve().parents[2]


def _model_and_inputs() -> tuple[
    PCNOEuler2DResidual,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    torch.manual_seed(20260802)
    normalization = Euler2DNormalization(
        state_mean=np.asarray([1.0, 1.0, 0.0, 3.0]),
        state_scale=np.asarray([0.2, 0.3, 0.1, 0.5]),
        residual_scale=np.asarray([0.01, 0.02, 0.02, 0.04]),
        mach_mean=1.5,
        mach_scale=0.2,
    )
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=2,
        domain_lengths=(2.0, 1.0),
        layers=(8, 8, 8, 8, 8),
        fc_dim=7,
        zero_initialize=False,
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
    generator = torch.Generator().manual_seed(20260802)
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
    }
    return model, current, sample


def _forward(
    model: PCNOEuler2DResidual,
    current: torch.Tensor,
    sample: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        current,
        node_mask=sample["node_mask"],
        nodes=sample["nodes"],
        node_weights=sample["node_weights"],
        node_rhos=sample["node_rhos"],
        directed_edges=sample["directed_edges"],
        edge_gradient_weights=sample["edge_gradient_weights"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )


def _case_data(
    model: PCNOEuler2DResidual,
    current: torch.Tensor,
    sample: dict[str, torch.Tensor],
) -> CaseData:
    initial = current[0].numpy().astype(np.float64)
    reference_states = np.stack((initial, initial, initial), axis=0)
    return CaseData(
        family="bump",
        case_id="synthetic",
        resolution=None,
        sample=sample,
        reference_states=reference_states,
        physical_times=np.asarray([0.0, 0.1, 0.2]),
        nodes=sample["nodes"][0].numpy().astype(np.float64),
        edges=sample["directed_edges"][0].numpy(),
        weights=sample["node_weights"][0].sum(dim=-1).numpy().astype(np.float64),
        physical_node_type=sample["node_type"][0].numpy(),
        boundary_distance=np.asarray([0.4, 0.0, 0.0, 0.0, 0.4, 0.0]),
        state_scale=model.state_scale[0, 0].numpy().astype(np.float64),
        residual_scale=model.residual_scale[0, 0].numpy().astype(np.float64),
        gamma=model.gamma,
        provenance={"fixture": "synthetic"},
    )


def test_bump_loader_owns_arrays_that_outlive_the_shard_cache(tmp_path) -> None:
    class MutableStore:
        def __init__(self) -> None:
            self.manifest = {"dt": 0.025}
            self.states_by_key: dict[str, np.ndarray] = {}
            self.arrays_by_key: dict[str, dict[str, np.ndarray]] = {}
            for index, key in enumerate(("0", "1", "2")):
                self.states_by_key[key] = np.full(
                    (3, 4, 4), float(index + 1), dtype=np.float32
                )
                self.arrays_by_key[key] = {
                    "nodes": np.asarray(
                        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                        dtype=np.float64,
                    ),
                    "directed_edges": np.asarray(
                        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]],
                        dtype=np.int64,
                    ),
                    "node_type": np.asarray([1, 0, 3, 2], dtype=np.int64),
                    "node_weights": np.full((4, 1), 0.25, dtype=np.float64),
                }

        def states(self, key: str) -> np.ndarray:
            return self.states_by_key[key]

        def array(self, key: str, name: str) -> np.ndarray:
            return self.arrays_by_key[key][name]

        def tensor_sample(
            self,
            key: str,
            time_index: int,
            *,
            step_stride: int,
            device: torch.device,
        ) -> dict[str, torch.Tensor]:
            del key, time_index, step_stride, device
            return {}

    split_path = tmp_path / "split.json"
    split_path.write_text('{"val_keys": ["0", "1", "2"]}', encoding="utf-8")
    args = SimpleNamespace(
        split_json=split_path,
        case_ids=None,
        rollout_calls=1,
    )
    checkpoint = {
        "step_stride": 1,
        "normalization": {
            "state_scale": [1.0, 1.0, 1.0, 1.0],
            "residual_scale": [1.0, 1.0, 1.0, 1.0],
            "gamma": 1.4,
        },
    }
    store = MutableStore()

    cases = _load_bump_cases(
        args,
        checkpoint,
        store,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    expected_nodes = cases[0].nodes.copy()
    expected_edges = cases[0].edges.copy()
    expected_types = cases[0].physical_node_type.copy()
    store.arrays_by_key["0"]["nodes"][:] = -1.0
    store.arrays_by_key["0"]["directed_edges"][:] = -1
    store.arrays_by_key["0"]["node_type"][:] = 0

    assert np.array_equal(cases[0].nodes, expected_nodes)
    assert np.array_equal(cases[0].edges, expected_edges)
    assert np.array_equal(cases[0].physical_node_type, expected_types)


@torch.no_grad()
def test_hooks_are_bitwise_equivalent_and_capture_every_pcno_stage() -> None:
    model, current, sample = _model_and_inputs()
    before = _forward(model, current, sample)
    recorder = PCNOActivationRecorder(model.backbone)
    with recorder:
        hooked = _forward(model, current, sample)
    trace = recorder.snapshot()
    after = _forward(model, current, sample)

    assert torch.equal(hooked, before)
    assert torch.equal(after, before)
    assert tensor_equivalence_metrics(before, hooked) == {
        "same_shape": True,
        "same_dtype": True,
        "exact_equal": True,
        "max_abs": 0.0,
        "relative_l2": 0.0,
    }
    assert trace["model_input"].shape == (1, 6, 12)
    assert trace["post_lift"].shape == (1, 6, 8)
    assert trace["normalized_residual"].shape == (1, 6, 4)
    for index in range(4):
        assert trace[f"block.{index}.input"].shape == (1, 6, 8)
        assert trace[f"block.{index}.output"].shape == (1, 6, 8)
        for branch in ("integral", "pointwise", "differential"):
            assert trace[f"block.{index}.{branch}"].shape == (1, 6, 8)


@torch.no_grad()
def test_analytical_type_lift_delta_matches_hooked_difference() -> None:
    model, current, sample = _model_and_inputs()
    correct_recorder = PCNOActivationRecorder(model.backbone)
    with correct_recorder:
        _forward(model, current, sample)
    correct = correct_recorder.snapshot()

    intervened_sample = dict(sample)
    intervened_sample["node_type"] = replace_node_types_with_normal(
        sample["node_type"], family="bump"
    )
    intervened_recorder = PCNOActivationRecorder(model.backbone)
    with intervened_recorder:
        _forward(model, current, intervened_sample)
    intervened = intervened_recorder.snapshot()
    differences = activation_differences(correct, intervened)
    expected = type_lift_delta(
        model.backbone,
        sample["node_type"],
        intervened_sample["node_type"],
    )

    torch.testing.assert_close(
        differences["post_lift"], expected, rtol=1.0e-6, atol=1.0e-7
    )
    torch.testing.assert_close(
        differences["post_lift"][:, [0, 4]],
        torch.zeros_like(differences["post_lift"][:, [0, 4]]),
        rtol=0.0,
        atol=0.0,
    )
    all_normal_channels = intervened["model_input"][..., 7:11]
    torch.testing.assert_close(
        all_normal_channels,
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(all_normal_channels),
        rtol=0.0,
        atol=0.0,
    )


def test_single_type_replacement_is_family_local() -> None:
    node_type = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    bump = replace_node_types_with_normal(node_type, family="bump", source_type=1)
    dynamic = replace_node_types_with_normal(
        node_type, family="dynamic_fv", source_type=1
    )

    assert bump.tolist() == [[0, 0, 2, 3]]
    assert dynamic.tolist() == [[0, 0, 2, 3]]
    assert node_type_intervention_name(family="bump", source_type=1) == (
        "wall_to_normal"
    )
    assert node_type_intervention_name(family="dynamic_fv", source_type=1) == (
        "y_symmetry_contact_to_interior"
    )
    with pytest.raises(ValueError, match="nonnormal"):
        replace_node_types_with_normal(node_type, family="bump", source_type=0)
    with pytest.raises(ValueError, match="unsupported node-type family"):
        replace_node_types_with_normal(node_type, family="mixed")


@torch.no_grad()
def test_evaluator_smoke_preserves_causal_and_free_rollout_contracts(
    tmp_path,
) -> None:
    model, current, sample = _model_and_inputs()
    case = _case_data(model, current, sample)
    output_dir = tmp_path / "node_type_smoke"
    (output_dir / "traces").mkdir(parents=True)
    (output_dir / "animations").mkdir()
    args = SimpleNamespace(
        rollout_calls=2,
        trace_cases=("synthetic",),
        trace_calls=(1, 2),
        trace_interventions=("all_normal",),
        amp="none",
        native_repeat_absolute_limit=2.0e-5,
        native_repeat_relative_limit=2.0e-8,
        shock_quantile=0.8,
        output_dir=output_dir,
        animation_max_nodes=6,
    )
    specs = _intervention_specs("bump", ("correct", "all_normal"))

    equivalence = _hook_equivalence(
        model,
        case,
        device=torch.device("cpu"),
        amp="none",
    )
    assert equivalence["passed"] is True

    contract_rows = _intervention_contract_rows(model, case, specs)
    all_normal_contract = next(
        row for row in contract_rows if row["intervention"] == "all_normal"
    )
    assert all_normal_contract["affected_node_count"] == 4
    assert all_normal_contract["affected_weight_fraction"] == pytest.approx(4.0 / 6.0)
    assert all_normal_contract["weighted_type_lift_rms"] > 0.0

    outcome_rows = []
    completion_rows = []
    frequency_rows = []
    activation_rows = []
    trace_files = []
    mode_payloads = {}
    for mode in ("teacher_forced", "free_rollout"):
        mode_payloads[mode] = _evaluate_mode(
            args,
            model,
            case,
            specs,
            mode=mode,
            device=torch.device("cpu"),
            outcome_rows=outcome_rows,
            completion_rows=completion_rows,
            frequency_rows=frequency_rows,
            activation_rows=activation_rows,
            trace_files=trace_files,
        )

    assert outcome_rows
    assert completion_rows
    assert frequency_rows
    assert len(trace_files) == 4
    same_state_checks = [
        row
        for row in activation_rows
        if row["field"] == "analytical_type_only_post_lift_check"
        and row["mode"] == "teacher_forced"
    ]
    assert len(same_state_checks) == 2
    assert all(row["same_nontype_input"] for row in same_state_checks)
    assert max(row["maximum_absolute_error"] for row in same_state_checks) < 1.0e-6
    free_second = next(
        row
        for row in activation_rows
        if row["field"] == "analytical_type_only_post_lift_check"
        and row["mode"] == "free_rollout"
        and row["call"] == 2
    )
    assert free_second["same_nontype_input"] is False
    assert free_second["maximum_absolute_error"] is None

    free_trace = next(
        path for path in trace_files if "free_rollout_call002_all_normal" in path.name
    )
    with np.load(free_trace, allow_pickle=False) as trace:
        np.testing.assert_allclose(
            trace["correct_residual"],
            trace["correct_prediction"] - trace["correct_current"],
        )
        np.testing.assert_allclose(
            trace["intervened_residual"],
            trace["intervened_prediction"] - trace["intervened_current"],
        )

    animation_path = _save_animation_payload(args, case, mode_payloads)
    with np.load(animation_path, allow_pickle=False) as animation:
        assert "teacher_forced_correct_prediction" in animation
        assert "teacher_forced_all_normal_prediction" in animation
        assert "teacher_forced_correct_minus_all_normal_prediction" in animation
        assert "free_rollout_correct_minus_all_normal_residual" in animation


def _primitive_state(
    density: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    pressure: np.ndarray,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    energy = pressure / (gamma - 1.0) + 0.5 * density * (velocity_x**2 + velocity_y**2)
    return np.stack(
        (
            density,
            density * velocity_x,
            density * velocity_y,
            energy,
        ),
        axis=-1,
    )


def test_visualizer_conservative_fields_and_frame_contract() -> None:
    state = np.asarray([[2.0, 6.0, 8.0, 27.5]])
    np.testing.assert_allclose(conservative_field(state, field="density"), [2.0])
    np.testing.assert_allclose(conservative_field(state, field="pressure"), [1.0])
    assert animation_frame_indices(6) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="positive"):
        animation_frame_indices(0)


def test_visualizers_write_fixed_scale_hash_bound_artifacts(tmp_path) -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    node_type = np.asarray([0, 1, 2, 3], dtype=np.int64)
    boundary_distance = np.asarray([1.0, 0.0, 0.0, 0.0])
    density = np.ones((2, 4))
    velocity_x = np.full((2, 4), 0.2)
    velocity_y = np.zeros((2, 4))
    correct_pressure = np.asarray([[0.95, 1.00, 1.05, 1.10], [1.00, 1.05, 1.10, 1.15]])
    all_normal_pressure = correct_pressure - 0.04
    current_pressure = correct_pressure - 0.02
    correct_prediction = _primitive_state(
        density, velocity_x, velocity_y, correct_pressure
    ).astype(np.float32)
    all_normal_prediction = _primitive_state(
        density, velocity_x, velocity_y, all_normal_pressure
    ).astype(np.float32)
    correct_current = _primitive_state(
        density, velocity_x, velocity_y, current_pressure
    ).astype(np.float32)
    all_normal_current = _primitive_state(
        density, velocity_x, velocity_y, current_pressure - 0.01
    ).astype(np.float32)
    bundle_path = tmp_path / "synthetic_animation.npz"
    np.savez_compressed(
        bundle_path,
        schema=np.asarray("pcno_node_type_interventions_v1"),
        family=np.asarray("dynamic_fv"),
        case_id=np.asarray("synthetic"),
        resolution=np.asarray("2x2"),
        visualization_only_subsampling=np.asarray(True),
        visualization_node_indices=np.arange(4, dtype=np.int64),
        physical_times=np.asarray([0.1, 0.2]),
        nodes=nodes,
        physical_node_type=node_type,
        boundary_distance=boundary_distance,
        teacher_forced_correct_prediction=correct_prediction,
        teacher_forced_correct_residual=(correct_prediction - correct_current),
        teacher_forced_all_normal_prediction=all_normal_prediction,
        teacher_forced_all_normal_residual=(all_normal_prediction - all_normal_current),
    )
    completion_path = tmp_path / "completion.csv"
    completion_path.write_text(
        "family,case_id,resolution,mode,intervention,call,finite,admissible\n"
        "dynamic_fv,synthetic,2x2,teacher_forced,all_normal,1,True,True\n"
        "dynamic_fv,synthetic,2x2,teacher_forced,all_normal,2,True,False\n",
        encoding="utf-8",
    )
    movie_stem = tmp_path / "movie"
    movie_manifest = render_animation(
        SimpleNamespace(
            bundles=[bundle_path],
            completion_csv=completion_path,
            output_stem=movie_stem,
            mode="teacher_forced",
            field="pressure",
            gamma=1.4,
            units="nondimensional",
            state_min=0.5,
            state_max=1.5,
            difference_abs_max=0.2,
            update_abs_max=0.2,
            linthresh=0.01,
            boundary_distance_max=1.0,
            scale_source="synthetic reference-only test contract",
            fps=2,
            dpi=45,
            format="gif",
            overwrite=False,
        )
    )
    assert movie_stem.with_suffix(".gif").is_file()
    assert (tmp_path / "movie_final.png").is_file()
    assert (tmp_path / "movie_final.pdf").is_file()
    assert movie_stem.with_suffix(".json").is_file()
    assert (
        movie_manifest["completion_and_admissibility"]["2x2"]["first_inadmissible_call"]
        == 2
    )
    assert movie_manifest["frame_contract"]["rendered_calls"] == [1, 2]
    assert movie_manifest["frame_contract"]["temporal_subsampling"] is False
    assert movie_manifest["frame_contract"]["all_comparable_frames_rendered"] is True
    assert movie_manifest["scale_contract"]["result_dependent_autoscaling"] is False

    trace_path = tmp_path / "synthetic_trace.npz"
    trace_arrays = {
        field: np.linspace(0.0, 0.3, 4, dtype=np.float32)
        for field, _ in SENSITIVITY_FIELDS
    }
    np.savez_compressed(
        trace_path,
        schema=np.asarray("pcno_node_type_interventions_v1"),
        family=np.asarray("dynamic_fv"),
        case_id=np.asarray("synthetic"),
        resolution=np.asarray("2x2"),
        mode=np.asarray("teacher_forced"),
        call=np.asarray(1, dtype=np.int64),
        physical_time=np.asarray(0.1),
        intervention=np.asarray("all_normal"),
        nodes=nodes,
        physical_node_type=node_type,
        boundary_distance=boundary_distance,
        **trace_arrays,
    )
    sensitivity_stem = tmp_path / "sensitivity"
    sensitivity_manifest = render_sensitivity(
        SimpleNamespace(
            traces=[trace_path],
            output_stem=sensitivity_stem,
            reference_norm=0.1,
            transformed_max=1.0,
            boundary_distance_max=1.0,
            max_nodes=4,
            scale_source="synthetic fixed sensitivity contract",
            dpi=55,
            overwrite=False,
        )
    )
    assert sensitivity_stem.with_suffix(".png").is_file()
    assert sensitivity_stem.with_suffix(".pdf").is_file()
    assert sensitivity_stem.with_suffix(".json").is_file()
    assert "not causal branch attribution" in (
        sensitivity_manifest["claim_boundary"]["not_claimed"]
    )


def test_isolated_source_manifest_requires_the_complete_project_inventory(
    tmp_path: Path,
) -> None:
    head = "a" * 40
    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in _snapshot_project_sources()
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "source_base_git_head": head,
                "source_sha256": source_hashes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        source_manifest=path,
        expected_source_manifest_sha256=sha256_file(path),
        expected_source_base_git_head=head,
        expected_source=[],
    )
    verified = _verify_source_snapshot(args)
    assert verified["source_sha256"] == source_hashes

    missing = dict(source_hashes)
    missing.pop(next(iter(missing)))
    path.write_text(
        json.dumps(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "source_base_git_head": head,
                "source_sha256": missing,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args.expected_source_manifest_sha256 = sha256_file(path)
    with pytest.raises(ValueError, match="source inventory mismatch"):
        _verify_source_snapshot(args)

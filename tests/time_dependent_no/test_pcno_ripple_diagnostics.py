from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from pcno.pcno import PCNO
from scripts.time_dependent_no.diagnose_pcno_euler2d_ripples import (
    BRANCH_GAIN_SENSITIVITY_SCHEMA,
    branch_cancellation_screen,
    branch_gain_sensitivity_selector,
    main as diagnose_main,
    mechanism_screen,
    paired_branch_response_selector,
    paired_branch_response_summary,
)
from scripts.time_dependent_no.prepare_pcno_euler2d_shards import (
    main as prepare_main,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    fit_normalization,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    fourier_gram_audit,
    fourier_reconstruction_audit,
    geometry_conditioning_features,
    graph_spectral_bands,
    node_highpass_amplitude,
    node_highpass_field,
    raw_admissibility_summary,
    spatial_correlation_summary,
    trace_pcno_branches,
)


def test_d013_entry_point_resolves_repo_imports_outside_worktree(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/time_dependent_no/diagnose_pcno_euler2d_ripples.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run frozen-checkpoint D013 ripple diagnostics" in result.stdout


def test_raw_euler_diagnostics_do_not_floor_negative_density() -> None:
    state = np.asarray(
        [[-1.0, 1.0, 0.0, 3.0], [1.0, 0.0, 0.0, 2.5]],
        dtype=np.float64,
    )
    primitive = conservative_to_primitive_raw(state)
    summary = raw_admissibility_summary(state)

    assert primitive[0, 1] == pytest.approx(-1.0)
    assert not summary["all_admissible"]
    assert summary["inadmissible_node_count"] == 1
    assert summary["min_density"] == pytest.approx(-1.0)


def test_fourier_gram_and_orthogonal_projection_separate_quadrature_leakage() -> None:
    axis = np.arange(8, dtype=np.float64) * (2.0 * np.pi / 8.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    regular_nodes = np.stack((x.ravel(), y.ravel()), axis=-1)
    modes = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    regular_summary, _ = fourier_gram_audit(
        regular_nodes,
        modes,
        np.ones(regular_nodes.shape[0]),
    )
    assert regular_summary["correlation_condition_number"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert regular_summary["max_absolute_off_diagonal"] < 1e-12

    irregular_nodes = regular_nodes[
        np.asarray([0, 1, 2, 3, 5, 8, 10, 13, 17, 22, 29, 37, 46, 55, 63])
    ]
    weights = np.linspace(0.5, 2.0, irregular_nodes.shape[0])
    field = 0.3 + np.cos(irregular_nodes[:, :1]) - 0.4 * np.sin(irregular_nodes[:, 1:2])
    summary, arrays = fourier_reconstruction_audit(
        irregular_nodes,
        modes,
        weights,
        field,
    )
    assert summary["mass_orthogonal_projection"]["rmse"] < 1e-10
    assert summary["pcno_uniform_formula"]["rmse"] > 1e-3
    assert summary["rmse_improvement_factor"] > 1e5
    np.testing.assert_allclose(
        arrays["mass_orthogonal_reconstruction"], field, atol=1e-10
    )


def test_graph_spectrum_places_alternating_path_signal_above_constant() -> None:
    edges = np.stack((np.arange(7), np.arange(1, 8)), axis=-1)
    weights = np.ones(8)
    constant = graph_spectral_bands(
        np.ones((8, 1)),
        edges,
        weights,
        lanczos_steps=8,
    )
    alternating = graph_spectral_bands(
        ((-1.0) ** np.arange(8))[:, None],
        edges,
        weights,
        lanczos_steps=8,
    )

    assert constant["high_band_fraction"] < 1e-10
    assert alternating["high_band_fraction"] > 0.9
    assert sum(alternating["band_fraction"]) == pytest.approx(1.0)


def test_geometry_features_and_correlations_are_graph_native() -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    node_type = np.asarray([1, 0, 0, 0])
    shock = np.asarray([False, False, True, False])
    features = geometry_conditioning_features(
        nodes,
        edges,
        directed,
        node_type,
        np.ones(4),
        shock_mask=shock,
    )
    assert features["boundary_graph_distance"][0] == pytest.approx(0.0)
    assert features["boundary_graph_distance"][2] == pytest.approx(2.0)
    assert features["shock_graph_distance"][2] == pytest.approx(0.0)
    correlations = spatial_correlation_summary(
        features["boundary_graph_distance"],
        {"distance": features["boundary_graph_distance"]},
    )
    assert correlations["distance"]["pearson"] == pytest.approx(1.0)
    assert correlations["distance"]["spearman"] == pytest.approx(1.0)


def test_branch_trace_replays_pcno_and_supports_counterfactual() -> None:
    torch.manual_seed(4)
    modes = torch.as_tensor(np.asarray([[[1.0]], [[2.0]]], dtype=np.float32))
    model = PCNO(
        ndims=1,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=3,
        out_dim=2,
    )
    nodes = torch.linspace(0.0, 1.0, 5).reshape(1, 5, 1)
    edges = torch.as_tensor(
        [[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]]
    )
    gradient_weights = torch.ones((1, 8, 1))
    node_weights = torch.full((1, 5, 1), 0.2)
    node_mask = torch.ones((1, 5, 1))
    model_input = torch.randn((1, 5, 3))
    aux = (node_mask, nodes, node_weights, edges, gradient_weights)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    expected = model(model_input, aux)
    actual, summaries = trace_pcno_branches(model, model_input, aux)
    paired_input = model_input + 0.01 * torch.randn_like(model_input)
    paired_actual, paired_summaries, paired_output = trace_pcno_branches(
        model,
        model_input,
        aux,
        paired_model_input=paired_input,
        return_paired_output=True,
        reference_smooth_region_mask=torch.ones((1, 5, 1), dtype=torch.bool),
        diagnostic_weight_maps={
            "equal_node_proxy": torch.ones((1, 5, 1)),
            "reconstructed_weight_proxy": torch.linspace(0.5, 1.5, 5).reshape(1, 5, 1),
        },
    )
    counterfactual, _ = trace_pcno_branches(
        model,
        model_input,
        aux,
        disabled_branch=(1, "spectral"),
        collect_summaries=False,
    )
    identity_gains = {
        (layer, branch): 1.0
        for layer in range(len(model.ws))
        for branch in ("spectral", "pointwise", "differential")
    }
    gain_identity, _ = trace_pcno_branches(
        model,
        model_input,
        aux,
        branch_gains=identity_gains,
        collect_summaries=False,
    )
    attenuated, _ = trace_pcno_branches(
        model,
        model_input,
        aux,
        branch_gains={(1, "pointwise"): 0.5},
        collect_summaries=False,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(paired_actual, expected)
    torch.testing.assert_close(paired_output, model(paired_input, aux))
    torch.testing.assert_close(gain_identity, expected, rtol=0.0, atol=0.0)
    assert len(summaries) == 2
    assert set(summaries[0]["branches"]) == {
        "spectral",
        "pointwise",
        "differential",
    }
    assert not torch.allclose(counterfactual, expected)
    assert not torch.allclose(attenuated, expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0.0, atol=0.0)
    assert len(paired_summaries) == 2
    for layer in paired_summaries:
        assert "input_hidden" in layer
        assert "post_activation_hidden" in layer
        bands = layer["graph_band_proxy"]
        assert set(bands) == {
            "input_hidden",
            "spectral",
            "pointwise",
            "differential",
            "combined_update",
            "post_activation_hidden",
        }
        smooth_band = bands["spectral"]["regions"]["smooth_reference"]
        assert smooth_band["equal_node_proxy"]["status"] == "available"
        assert (
            smooth_band["equal_node_proxy"]["relative_reconstruction_energy_residual"]
            < 1e-6
        )
        cancellation = layer["smooth_highpass_cancellation"]
        assert set(cancellation) == {
            "equal_node_proxy",
            "reconstructed_weight_proxy",
        }
        for metric in cancellation.values():
            assert metric["status"] == "available"
            assert metric["relative_identity_residual"] < 1e-5
            assert metric["sum_pair_cancellation_share"] == pytest.approx(
                metric["cancellation_fraction"], abs=1e-5
            )
        response = layer["paired_response"]
        assert set(response["branches"]) == {
            "spectral",
            "pointwise",
            "differential",
        }
        assert response["input_hidden_delta"]["weighted_rms"] > 0.0
        assert response["next_hidden_to_input_rms_gain"] > 0.0
        for metric in response["smooth_highpass_cancellation"].values():
            assert metric["status"] == "available"
            assert metric["relative_identity_residual"] < 1e-5
    response_summary = paired_branch_response_summary(
        [
            {
                "source": "perturbation_branch_response",
                "layers": paired_summaries,
            }
        ]
    )
    assert response_summary["status"] == "complete"
    assert response_summary["row_count"] == 1
    assert len(response_summary["layer_medians"]) == 2
    with pytest.raises(ValueError, match="separate diagnostics"):
        trace_pcno_branches(
            model,
            model_input,
            aux,
            disabled_branch=(0, "spectral"),
            paired_model_input=paired_input,
        )
    with pytest.raises(ValueError, match="must lie in"):
        trace_pcno_branches(
            model,
            model_input,
            aux,
            branch_gains={(0, "differential"): 1.1},
        )
    with pytest.raises(ValueError, match="requires paired_model_input"):
        trace_pcno_branches(
            model,
            model_input,
            aux,
            return_paired_output=True,
        )


def test_branch_cancellation_screen_uses_frozen_repeated_case_thresholds() -> None:
    def metric(
        cancellation: float,
        *,
        combined_energy: float = 0.7,
        identity_residual: float = 1e-8,
    ) -> dict:
        return {
            "status": "available",
            "cancellation_fraction": cancellation,
            "sum_branch_energy": combined_energy / (1.0 - cancellation),
            "combined_energy": combined_energy,
            "relative_identity_residual": identity_residual,
        }

    def paired_rows(
        *,
        cancellation: float,
        cases: tuple[str, ...] = ("a", "b"),
        combined_energy: float = 0.7,
        identity_residual: float = 1e-8,
    ) -> list[dict]:
        result = []
        for trajectory in cases:
            layers = []
            for layer_index in range(4):
                value = metric(
                    cancellation,
                    combined_energy=combined_energy,
                    identity_residual=identity_residual,
                )
                layers.append(
                    {
                        "layer": layer_index,
                        "paired_response": {
                            "input_hidden_delta": {"weighted_rms": 1.0},
                            "smooth_highpass_cancellation": {
                                "equal_node_proxy": dict(value),
                                "reconstructed_weight_proxy": dict(value),
                            },
                        },
                    }
                )
            result.append(
                {
                    "trajectory": trajectory,
                    "source": "perturbation_branch_response",
                    "call_index": 10,
                    "layers": layers,
                }
            )
        return result

    def absolute_rows() -> list[dict]:
        result = []
        for trajectory in ("a", "b"):
            for source, cancellation, combined_energy in (
                ("teacher_forced", 0.35, 1.0),
                ("rollout_state", 0.15, 1.5),
            ):
                layers = []
                for layer_index in range(4):
                    value = metric(
                        cancellation,
                        combined_energy=combined_energy,
                    )
                    layers.append(
                        {
                            "layer": layer_index,
                            "branches": {
                                name: {"weighted_rms": 1.0}
                                for name in (
                                    "spectral",
                                    "pointwise",
                                    "differential",
                                )
                            },
                            "smooth_highpass_cancellation": {
                                "equal_node_proxy": dict(value),
                                "reconstructed_weight_proxy": dict(value),
                            },
                        }
                    )
                result.append(
                    {
                        "trajectory": trajectory,
                        "source": source,
                        "call_index": 20,
                        "layers": layers,
                    }
                )
        return result

    supported = branch_cancellation_screen(
        paired_rows(cancellation=0.30) + absolute_rows()
    )
    assert supported["classification"] == "paired_error_response_cancellation_supported"
    assert (
        supported["absolute_output_context"]["classification"]
        == "late_teacher_rollout_cancellation_shift"
    )

    falsified = branch_cancellation_screen(paired_rows(cancellation=0.02))
    assert falsified["classification"] == "paired_error_response_cancellation_falsified"

    too_small = branch_cancellation_screen(
        paired_rows(cancellation=0.30, combined_energy=1e-20)
    )
    assert too_small["classification"] == "unresolved"
    reasons = too_small["case_decisions"][0]["weights"]["equal_node_proxy"][
        "invalid_layer_reasons"
    ]
    assert set(reasons.values()) == {"branch_highpass_energy_too_small"}

    bad_identity = branch_cancellation_screen(
        paired_rows(cancellation=0.30, identity_residual=1e-2)
    )
    assert bad_identity["classification"] == "unresolved"
    reasons = bad_identity["case_decisions"][0]["weights"]["equal_node_proxy"][
        "invalid_layer_reasons"
    ]
    assert set(reasons.values()) == {"identity_residual_too_large"}

    nonfinite = branch_cancellation_screen(
        paired_rows(cancellation=0.30, combined_energy=float("nan"))
    )
    assert nonfinite["classification"] == "unresolved"
    reasons = nonfinite["case_decisions"][0]["weights"]["equal_node_proxy"][
        "invalid_layer_reasons"
    ]
    assert set(reasons.values()) == {"nonfinite_sum_branch_energy"}

    assert (
        branch_cancellation_screen(paired_rows(cancellation=0.30, cases=("a",)))[
            "classification"
        ]
        == "unresolved"
    )


def test_paired_branch_selector_requires_repeated_layerwise_agreement() -> None:
    def row(trajectory: str, winner: str) -> dict:
        layers = []
        for layer_index in range(4):
            selected = winner if layer_index < 3 else "spectral"
            gains = {name: 1.0 for name in ("spectral", "pointwise", "differential")}
            roughness = dict(gains)
            gains[selected] = 3.0
            roughness[selected] = 4.0
            layers.append(
                {
                    "layer": layer_index,
                    "paired_response": {
                        "branch_to_input_rms_gain": gains,
                        "combined_update_to_input_rms_gain": 2.0,
                        "next_hidden_to_input_rms_gain": 2.5,
                        "branches": {
                            name: {"edge_to_node_energy_ratio": roughness[name]}
                            for name in gains
                        },
                    },
                }
            )
        return {
            "trajectory": trajectory,
            "source": "perturbation_branch_response",
            "call_index": 10,
            "layers": layers,
        }

    repeated = [row("a", "pointwise"), row("b", "pointwise")]
    selected = paired_branch_response_selector(repeated)
    assert selected["route"] == "local_pointwise_control"
    assert selected["selected_branch"] == "pointwise"
    assert all(item["required_layers"] == 3 for item in selected["row_decisions"])
    summary = paired_branch_response_summary(repeated)
    assert summary["selector"] == selected

    mixed = paired_branch_response_selector(
        [row("a", "pointwise"), row("b", "differential")]
    )
    assert mixed["route"] == "composite_or_unresolved"
    assert mixed["selected_branch"] is None
    assert (
        paired_branch_response_selector([row("a", "pointwise")])["route"]
        == "insufficient_repeated_rows"
    )


def test_branch_gain_selector_requires_six_cases_and_four_calls() -> None:
    def rows(
        *,
        teacher_branch: str | None,
        rollout_branch: str | None,
        calls: tuple[int, ...] = (1, 10, 30, 60),
    ) -> list[dict]:
        result = []
        for source, selected in (
            ("teacher_forced", teacher_branch),
            ("rollout_state", rollout_branch),
        ):
            for trajectory_index in range(6):
                for call_index in calls:
                    result.append(
                        {
                            "schema": BRANCH_GAIN_SENSITIVITY_SCHEMA,
                            "source": source,
                            "trajectory": str(trajectory_index),
                            "call_index": call_index,
                            "branch_decisions": {
                                name: {"passes_row_gate": name == selected}
                                for name in (
                                    "spectral",
                                    "pointwise",
                                    "differential",
                                )
                            },
                        }
                    )
        return result

    differential = branch_gain_sensitivity_selector(
        rows(
            teacher_branch="differential",
            rollout_branch="differential",
        )
    )
    assert differential["contract_complete"]
    assert differential["selected_branch"] == "differential"
    assert differential["route"] == "current_state_nonlinear_stencil_capacity"

    rollout_only = branch_gain_sensitivity_selector(
        rows(teacher_branch=None, rollout_branch="pointwise")
    )
    assert rollout_only["selected_branch"] == "pointwise"
    assert rollout_only["route"] == "generated_state_exposure_after_one_step_gate"

    stride_two_calls = (1, 5, 15, 30)
    stride_two = branch_gain_sensitivity_selector(
        rows(
            teacher_branch="pointwise",
            rollout_branch="pointwise",
            calls=stride_two_calls,
        ),
        expected_calls=stride_two_calls,
    )
    assert stride_two["contract_complete"]
    assert (
        stride_two["version"] == "branch_gain_sensitivity_selector_d052_v2_stride_aware"
    )
    assert stride_two["repetition_gate"]["exact_calls"] == list(stride_two_calls)
    assert stride_two["selected_branch"] == "pointwise"

    mismatched_calls = branch_gain_sensitivity_selector(
        rows(
            teacher_branch="pointwise",
            rollout_branch="pointwise",
            calls=stride_two_calls,
        )
    )
    assert not mismatched_calls["contract_complete"]
    assert mismatched_calls["route"] == "insufficient_repeated_rows"

    incomplete_rows = rows(
        teacher_branch="spectral",
        rollout_branch="spectral",
    )[:-1]
    incomplete = branch_gain_sensitivity_selector(incomplete_rows)
    assert not incomplete["contract_complete"]
    assert incomplete["route"] == "insufficient_repeated_rows"


def test_linear_highpass_and_additive_error_decomposition_close() -> None:
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    left = np.arange(8, dtype=np.float64).reshape(4, 2)
    right = np.flip(left, axis=0).copy()
    np.testing.assert_allclose(
        node_highpass_field(left + right, edges),
        node_highpass_field(left, edges) + node_highpass_field(right, edges),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        node_highpass_amplitude(left, edges),
        np.linalg.norm(node_highpass_field(left, edges), axis=-1),
    )


def test_mechanism_screen_routes_only_unique_supported_failure() -> None:
    def row(source: str, call: int, energy: float) -> dict:
        return {
            "trajectory": "a",
            "source": source,
            "call_index": call,
            "graph_spectrum": {
                "reconstructed_weight_proxy": {
                    "smooth_induced_subgraph": {
                        "high_band_energy": energy,
                        "total_weighted_energy": 2.0 * energy,
                    }
                }
            },
        }

    screen = mechanism_screen(
        [
            row("rollout_state", 1, 1.0),
            row("rollout_state", 2, 4.0),
            row("teacher_forced", 1, 1.0),
            row("teacher_forced", 2, 1.0),
        ],
        [],
        [],
        [{"local_gain": 1.1}],
    )

    assert screen["classification"] == "recurrent_amplification"
    assert screen["supported_screens"] == ["recurrent_amplification"]
    assert "generated-state exposure" in screen["selected_first_branch"]
    assert mechanism_screen([], [], [], [])["classification"] == "unresolved"

    tiny_energy_screen = mechanism_screen(
        [
            row("rollout_state", 1, 1e-20),
            row("rollout_state", 2, 4e-20),
            row("teacher_forced", 1, 1e-20),
            row("teacher_forced", 2, 1e-20),
        ],
        [],
        [],
        [{"local_gain": 1.1}],
    )
    assert tiny_energy_screen["classification"] == "unresolved"
    assert (
        tiny_energy_screen["evidence"][
            "late_rollout_to_teacher_smooth_high_band_energy_ratio"
        ]
        is None
    )
    assert (
        tiny_energy_screen["evidence"]["rollout_first_to_late_smooth_high_band_growth"]
        is None
    )


def _write_tiny_raw_h5(path) -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    node_type = np.asarray([3, 1, 2, 1], dtype=np.int64)[:, None]
    steps = 4
    with h5py.File(path, "w") as handle:
        group = handle.create_group("0")
        group.create_dataset("pos", data=np.repeat(nodes[None, ...], steps, axis=0))
        group.create_dataset("edges", data=np.repeat(edges[None, ...], steps, axis=0))
        group.create_dataset(
            "node_type",
            data=np.repeat(node_type[None, ...], steps, axis=0),
        )
        group.create_dataset(
            "Mach",
            data=np.full((steps, 4, 1), 1.5, dtype=np.float32),
        )
        time = np.arange(steps, dtype=np.float32)[:, None, None]
        x = nodes[None, :, 0:1]
        y = nodes[None, :, 1:2]
        group.create_dataset("rho", data=1.0 + 0.01 * time + 0.01 * x)
        group.create_dataset("v1", data=0.8 + 0.02 * time + 0.01 * x)
        group.create_dataset("v2", data=0.03 * y + 0.005 * time)
        group.create_dataset("pres", data=1.0 + 0.03 * time + 0.01 * y)


def test_d013_cli_writes_closed_raw_recurrence_bundle(tmp_path) -> None:
    source = tmp_path / "raw.h5"
    _write_tiny_raw_h5(source)
    shards = tmp_path / "shards"
    prepare_main(
        [
            "--source-h5",
            str(source),
            "--output-dir",
            str(shards),
            "--static-check",
            "all",
        ]
    )
    manifest_path = shards / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weight_provenance"] = "validated_physical_cell_volume_normalized"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    store = PCNOEuler2DShardStore(shards)
    normalization = fit_normalization(store, ["0"])
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        layers=[8, 8],
        fc_dim=8,
        zero_initialize=True,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.model_config(),
            "normalization": normalization.to_dict(),
            "data_manifest_digest": store.manifest_digest,
            "step_stride": 1,
            "val_keys": ["0"],
            "boundary_mode": "model_all_nodes",
            "raw_recurrence": True,
            "config_digest": "synthetic",
            "epoch": 0,
            "best_epoch": 0,
        },
        checkpoint,
    )
    store.close()
    output = tmp_path / "diagnostic"
    diagnose_main(
        [
            "--data-dir",
            str(shards),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--trajectory-keys",
            "0",
            "--trajectory-count",
            "1",
            "--deep-trajectory-count",
            "1",
            "--num-steps",
            "2",
            "--diagnostic-calls",
            "1",
            "2",
            "--trace-calls",
            "1",
            "2",
            "--basis-call",
            "1",
            "--perturbation-call",
            "1",
            "--branch-sensitivity-step",
            "0.01",
            "--device",
            "cpu",
        ]
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["completion_rate"] == pytest.approx(1.0)
    assert summary["diagnostic_contract"]["physical_conservation"].startswith(
        "not evaluated"
    )
    basis = summary["basis_audits"][0]
    span_counterfactual = basis["coordinate_span_basis_counterfactual"]
    assert span_counterfactual["coordinate_span"] == pytest.approx([1.0, 1.0])
    assert set(span_counterfactual["weights"]) == {
        "equal_node_proxy",
        "reconstructed_weight_proxy",
    }
    assert (
        "normalized_target_pressure"
        in span_counterfactual["weights"]["reconstructed_weight_proxy"]["fields"]
    )
    with np.load(output / basis["artifact"]) as artifact:
        assert "equal_node_proxy_coordinate_span_gram_gram" in artifact
        assert (
            "equal_node_proxy_normalized_target_pressure_coordinate_span_"
            "mass_orthogonal_reconstruction"
        ) in artifact
    with np.load(output / "trajectory_0.npz") as artifact:
        assert artifact["predictions"].shape == (2, 4, 4)
        assert artifact["boundary_mode"].item() == "model_all_nodes"
        assert artifact["failure_cause"].item() == "completed"
    branch_rows = [
        json.loads(line)
        for line in (output / "branches.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    paired = next(
        row for row in branch_rows if row["source"] == "perturbation_branch_response"
    )
    assert paired["smooth_highpass_mask"]["source"] == (
        "same-call_reference_current_only"
    )
    assert all("paired_response" in layer for layer in paired["layers"])
    assert all("smooth_highpass_cancellation" in layer for layer in paired["layers"])
    sensitivity_rows = [
        json.loads(line)
        for line in (output / "branch_sensitivities.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(sensitivity_rows) == 4
    assert summary["branch_gain_sensitivity"]["row_count"] == 4
    assert (
        summary["branch_gain_sensitivity"]["selector"]["route"]
        == "insufficient_repeated_rows"
    )
    for row in sensitivity_rows:
        assert row["schema"] == BRANCH_GAIN_SENSITIVITY_SCHEMA
        assert row["exact_replay_max_absolute_error"] < 1e-5
        assert len(row["interventions"]) == 3 * len(model.backbone.ws)
        assert set(row["branch_decisions"]) == {
            "spectral",
            "pointwise",
            "differential",
        }

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcno.pcno import PCNO
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

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcno.pcno import PCNO, compute_Fourier_modes

try:
    from scripts.time_dependent_no.visualize_pcno_resolution_pathways import (
        _sub_005_spectral_rows,
        animate_bundle,
    )
except ImportError:
    from visualize_pcno_resolution_pathways import (  # type: ignore[no-redef]
        _sub_005_spectral_rows,
        animate_bundle,
    )

try:
    from scripts.time_dependent_no.analyze_pcno_resolution_pathways import (
        _admissibility,
        _gain_row,
        _output_row,
    )
except ImportError:
    from analyze_pcno_resolution_pathways import (  # type: ignore[no-redef]
        _admissibility,
        _gain_row,
        _output_row,
    )

try:
    from utility.time_dependent_no.pcno_resolution_pathways import (
        BRANCH_NAMES,
        build_graph_ball_operator,
        decompose_differential_same_hidden_mesh_gap,
        decompose_spectral_same_hidden_mesh_gap,
        generic_spectral_energy_rows,
        graph_ball_average,
        graph_ball_device_invariants,
        graph_geometry_tolerance,
        graph_two_hop_physical_extents,
        graph_two_hop_physical_radius,
        prepare_graph_ball_operator,
        prolong_channel_first,
        restrict_channel_first,
        select_dominant_pathway,
        trace_backbone_output,
        trace_native_branch_replacement_output,
        trace_native_hidden_to_layer,
        trace_resolution_pair,
        trace_same_hidden_physical_radius_outputs,
        trace_same_hidden_subpath_replacements,
    )
except ImportError:
    from pcno_resolution_pathways import (  # type: ignore[no-redef]
        BRANCH_NAMES,
        build_graph_ball_operator,
        decompose_differential_same_hidden_mesh_gap,
        decompose_spectral_same_hidden_mesh_gap,
        generic_spectral_energy_rows,
        graph_ball_average,
        graph_ball_device_invariants,
        graph_geometry_tolerance,
        graph_two_hop_physical_extents,
        graph_two_hop_physical_radius,
        prepare_graph_ball_operator,
        prolong_channel_first,
        restrict_channel_first,
        select_dominant_pathway,
        trace_backbone_output,
        trace_native_branch_replacement_output,
        trace_native_hidden_to_layer,
        trace_resolution_pair,
        trace_same_hidden_physical_radius_outputs,
        trace_same_hidden_subpath_replacements,
    )


def _line_graph(node_count: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.stack(
        (
            np.arange(node_count, dtype=np.float64) * spacing,
            np.zeros(node_count, dtype=np.float64),
        ),
        axis=1,
    )
    edges = np.asarray(
        [
            edge
            for left in range(node_count - 1)
            for edge in ((left, left + 1), (left + 1, left))
        ],
        dtype=np.int64,
    )
    return nodes, edges


def test_graph_ball_operator_is_stochastic_and_reproduces_constants() -> None:
    nodes = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)))
    edges = np.asarray(((1, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3)))
    weights = np.asarray((1.0, 2.0, 3.0, 4.0))
    operator = build_graph_ball_operator(nodes, edges, weights, radius=1.1)
    prepared = prepare_graph_ball_operator(operator, device="cpu", dtype=torch.float32)

    assert operator.maximum_row_sum_error <= 2.0 * np.finfo(np.float64).eps
    assert operator.minimum_coefficient > 0.0
    assert operator.maximum_neighbors == 3
    invariants = graph_ball_device_invariants(prepared)
    assert invariants["maximum_row_sum_error"] <= 2.0e-7
    assert invariants["maximum_constant_error"] <= 2.0e-7
    constant = torch.full((2, 3, 4), 7.25)
    torch.testing.assert_close(
        graph_ball_average(constant, prepared), constant, atol=1.0e-6, rtol=0.0
    )

    values = torch.tensor([[[10.0, 20.0, 40.0, 80.0]]])
    expected = torch.tensor(
        [
            [
                [
                    (10.0 + 2.0 * 20.0) / 3.0,
                    (10.0 + 2.0 * 20.0 + 3.0 * 40.0) / 6.0,
                    (2.0 * 20.0 + 3.0 * 40.0) / 5.0,
                    80.0,
                ]
            ]
        ]
    )
    torch.testing.assert_close(graph_ball_average(values, prepared), expected)


def test_graph_ball_operator_is_invariant_to_edge_order_and_direction() -> None:
    nodes, edges = _line_graph(6, 0.2)
    weights = np.linspace(1.0, 2.0, nodes.shape[0])
    shuffled = np.concatenate((edges[::-1, ::-1], edges[[3, 0, 7]]), axis=0)
    left = build_graph_ball_operator(nodes, edges, weights, radius=0.41)
    right = build_graph_ball_operator(nodes, shuffled, weights, radius=0.41)

    np.testing.assert_array_equal(left.targets, right.targets)
    np.testing.assert_array_equal(left.sources, right.sources)
    np.testing.assert_allclose(
        left.coefficients, right.coefficients, rtol=0.0, atol=0.0
    )
    values = torch.arange(24, dtype=torch.float64).reshape(2, 2, 6)
    left_prepared = prepare_graph_ball_operator(
        left, device=values.device, dtype=values.dtype
    )
    right_prepared = prepare_graph_ball_operator(
        right, device=values.device, dtype=values.dtype
    )
    torch.testing.assert_close(
        graph_ball_average(values, left_prepared),
        graph_ball_average(values, right_prepared),
    )


def test_graph_ball_chunking_matches_dense_weighted_rows() -> None:
    nodes, edges = _line_graph(8, 0.1)
    weights = np.linspace(0.5, 1.5, nodes.shape[0])
    operator = build_graph_ball_operator(nodes, edges, weights, radius=0.21)
    prepared = prepare_graph_ball_operator(operator, device="cpu", dtype=torch.float64)
    values = torch.arange(48, dtype=torch.float64).reshape(2, 3, 8)
    dense = np.zeros((8, 8), dtype=np.float64)
    dense[operator.targets, operator.sources] = operator.coefficients
    expected = torch.einsum(
        "bcs,ts->bct", values, torch.as_tensor(dense, dtype=values.dtype)
    )

    torch.testing.assert_close(
        graph_ball_average(values, prepared, message_chunk_bytes=16), expected
    )


def test_graph_ball_geometry_tolerance_includes_float32_radius_boundary() -> None:
    nodes, edges = _line_graph(5, np.float32(0.1))
    nodes = nodes.astype(np.float32)
    operator = build_graph_ball_operator(nodes, edges, np.ones(5), radius=0.2)

    assert operator.geometry_tolerance == pytest.approx(graph_geometry_tolerance(nodes))
    center_sources = operator.sources[operator.targets == 2]
    np.testing.assert_array_equal(center_sources, np.arange(5))


def test_graph_ball_is_rigid_motion_permutation_and_mass_scale_invariant() -> None:
    nodes, edges = _line_graph(6, 0.2)
    weights = np.linspace(0.75, 1.75, nodes.shape[0])
    angle = 0.37
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    moved = nodes @ rotation.T + np.asarray((3.0, -4.0))
    baseline = build_graph_ball_operator(nodes, edges, weights, radius=0.41)
    transformed = build_graph_ball_operator(
        moved, edges[::-1], 9.0 * weights, radius=0.41
    )
    np.testing.assert_array_equal(baseline.targets, transformed.targets)
    np.testing.assert_array_equal(baseline.sources, transformed.sources)
    np.testing.assert_allclose(
        baseline.coefficients, transformed.coefficients, atol=2.0e-15, rtol=0.0
    )

    permutation = np.asarray((3, 0, 5, 2, 1, 4))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    permuted_edges = inverse[edges]
    permuted = build_graph_ball_operator(
        nodes[permutation], permuted_edges, weights[permutation], radius=0.41
    )
    values = torch.arange(12, dtype=torch.float64).reshape(1, 2, 6)
    baseline_prepared = prepare_graph_ball_operator(
        baseline, device="cpu", dtype=values.dtype
    )
    permuted_prepared = prepare_graph_ball_operator(
        permuted, device="cpu", dtype=values.dtype
    )
    permuted_values = values[..., permutation]
    restored = graph_ball_average(permuted_values, permuted_prepared)[..., inverse]
    torch.testing.assert_close(restored, graph_ball_average(values, baseline_prepared))


def test_same_hidden_physical_radius_trace_replays_a0_and_anchor_identity() -> None:
    model = _tiny_pcno()
    fine_aux = _grid_aux(4, 2)
    coarse_aux = _grid_aux(2, 1)
    fine_input = torch.randn((1, 8, 12))
    layer = len(model.ws) - 1

    def prepared(aux: tuple[torch.Tensor, ...], radius: float):
        operator = build_graph_ball_operator(
            aux[1][0].numpy(),
            aux[3][0].numpy(),
            aux[2][0, :, 0].numpy(),
            radius=radius,
        )
        return operator, prepare_graph_ball_operator(
            operator, device=fine_input.device, dtype=fine_input.dtype
        )

    coarse_radius = graph_two_hop_physical_radius(
        coarse_aux[1][0].numpy(),
        coarse_aux[3][0].numpy(),
        node_mask=np.ones(2, dtype=bool),
    )
    fine_radius = graph_two_hop_physical_radius(
        fine_aux[1][0].numpy(),
        fine_aux[3][0].numpy(),
        node_mask=np.ones(8, dtype=bool),
    )
    coarse_operator, coarse_prepared = prepared(coarse_aux, coarse_radius)
    fine_operator, fine_prepared = prepared(fine_aux, fine_radius)
    coarse_fixed_operator, _ = prepared(coarse_aux, coarse_radius)
    fine_fixed_operator, _ = prepared(fine_aux, fine_radius)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    outputs = trace_same_hidden_physical_radius_outputs(
        model,
        fine_input,
        coarse_aux,
        fine_aux,
        layer=layer,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
        coarse_local_operator=coarse_prepared,
        fine_local_operator=fine_prepared,
        coarse_fixed_operator=coarse_prepared,
        fine_fixed_operator=fine_prepared,
    )
    baseline = trace_same_hidden_subpath_replacements(
        model,
        fine_input,
        coarse_aux,
        fine_aux,
        layer=layer,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )

    assert set(outputs) == {"A0", "A1", "A2"}
    torch.testing.assert_close(outputs["A0"]["coarse_output"], baseline["baseline"])
    torch.testing.assert_close(
        outputs["A0"]["restricted_fine_output"], baseline["restricted_fine"]
    )
    assert outputs["A0"]["coarse_output"] is outputs["A0"]["native_coarse_output"]
    assert (
        outputs["A0"]["restricted_fine_output"]
        is outputs["A0"]["native_restricted_fine_output"]
    )
    assert (
        outputs["A1"]["coarse_smoothed_gradient"]
        is outputs["A2"]["coarse_smoothed_gradient"]
    )
    for key in outputs["A1"]:
        torch.testing.assert_close(outputs["A1"][key], outputs["A2"][key])
    for branch in (
        "coarse_spectral",
        "restricted_fine_spectral",
        "coarse_pointwise",
        "restricted_fine_pointwise",
    ):
        torch.testing.assert_close(outputs["A0"][branch], outputs["A1"][branch])
    for local, fixed in (
        (coarse_operator, coarse_fixed_operator),
        (fine_operator, fine_fixed_operator),
    ):
        np.testing.assert_array_equal(local.targets, fixed.targets)
        np.testing.assert_array_equal(local.sources, fixed.sources)
        np.testing.assert_array_equal(local.coefficients, fixed.coefficients)
        np.testing.assert_array_equal(local.distances, fixed.distances)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_physical_radius_trace_rejects_nonfinal_layer() -> None:
    model = _tiny_pcno()
    fine_aux = _grid_aux(4, 2)
    coarse_aux = _grid_aux(2, 1)
    fine_input = torch.randn((1, 8, 12))
    operator = build_graph_ball_operator(
        fine_aux[1][0].numpy(),
        fine_aux[3][0].numpy(),
        fine_aux[2][0, :, 0].numpy(),
        radius=1.0,
    )
    prepared = prepare_graph_ball_operator(
        operator, device="cpu", dtype=fine_input.dtype
    )
    with pytest.raises(ValueError, match="final layer"):
        trace_same_hidden_physical_radius_outputs(
            model,
            fine_input,
            coarse_aux,
            fine_aux,
            layer=0,
            coarse_resolution=(2, 1),
            fine_resolution=(4, 2),
            coarse_local_operator=prepared,
            fine_local_operator=prepared,
            coarse_fixed_operator=prepared,
            fine_fixed_operator=prepared,
        )


def test_two_hop_physical_radius_shrinks_under_graph_refinement() -> None:
    coarse_nodes, coarse_edges = _line_graph(5, 0.25)
    fine_nodes, fine_edges = _line_graph(9, 0.125)
    coarse_extents = graph_two_hop_physical_extents(coarse_nodes, coarse_edges)
    fine_extents = graph_two_hop_physical_extents(fine_nodes, fine_edges)

    np.testing.assert_allclose(coarse_extents, 0.5)
    np.testing.assert_allclose(fine_extents, 0.25)
    assert graph_two_hop_physical_radius(
        coarse_nodes, coarse_edges, node_mask=np.ones(5, dtype=bool)
    ) == pytest.approx(0.5)
    assert graph_two_hop_physical_radius(
        fine_nodes, fine_edges, node_mask=np.ones(9, dtype=bool)
    ) == pytest.approx(0.25)


def test_graph_ball_contract_rejects_invalid_geometry_weights_and_masks() -> None:
    nodes, edges = _line_graph(4, 0.25)
    with pytest.raises(ValueError, match="positive"):
        build_graph_ball_operator(nodes, edges, np.ones(4), radius=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        build_graph_ball_operator(
            nodes, edges, np.asarray((1.0, 0.0, 1.0, 1.0)), radius=0.5
        )
    with pytest.raises(ValueError, match="outside nodes"):
        build_graph_ball_operator(nodes, np.asarray(((0, 4),)), np.ones(4), radius=0.5)
    with pytest.raises(ValueError, match="select at least one"):
        graph_two_hop_physical_radius(nodes, edges, node_mask=np.zeros(4, dtype=bool))


def _grid_aux(nx: int, ny: int) -> tuple[torch.Tensor, ...]:
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) * 2.0 / nx
    y = (torch.arange(ny, dtype=torch.float32) + 0.5) / ny
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).unsqueeze(0)
    directed: list[tuple[int, int]] = []
    gradient: list[tuple[float, float]] = []
    for row in range(ny):
        for column in range(nx):
            center = row * nx + column
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other_x = column + dx
                other_y = row + dy
                if 0 <= other_x < nx and 0 <= other_y < ny:
                    directed.append((center, other_y * nx + other_x))
                    gradient.append((float(dx) * nx / 2.0, float(dy) * ny))
    node_count = nx * ny
    return (
        torch.ones((1, node_count, 1)),
        nodes,
        torch.full((1, node_count, 1), 1.0 / node_count),
        torch.as_tensor([directed], dtype=torch.int64),
        torch.as_tensor([gradient], dtype=torch.float32),
    )


def _tiny_pcno() -> PCNO:
    torch.manual_seed(20260802)
    modes = torch.as_tensor(
        compute_Fourier_modes(2, [1, 1], [2.0, 1.0]), dtype=torch.float32
    )
    return PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[8, 8, 8],
        fc_dim=6,
        in_dim=12,
        out_dim=4,
    )


def test_nested_torch_restriction_matches_explicit_block_average() -> None:
    value = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    actual = restrict_channel_first(
        value, fine_resolution=(4, 2), coarse_resolution=(2, 1)
    )
    expected = value.reshape(2, 3, 1, 2, 2, 2).mean(dim=(3, 5)).reshape(2, 3, 2)
    torch.testing.assert_close(actual, expected)
    prolonged = prolong_channel_first(
        actual, coarse_resolution=(2, 1), fine_resolution=(4, 2)
    )
    torch.testing.assert_close(
        restrict_channel_first(
            prolonged, fine_resolution=(4, 2), coarse_resolution=(2, 1)
        ),
        actual,
    )
    constant = torch.full((1, 5, 8), 7.25)
    torch.testing.assert_close(
        restrict_channel_first(
            constant, fine_resolution=(4, 2), coarse_resolution=(2, 1)
        ),
        torch.full((1, 5, 2), 7.25),
    )


def test_pair_trace_replays_outputs_and_closes_mesh_state_algebra() -> None:
    model = _tiny_pcno()
    fine_aux = _grid_aux(4, 2)
    coarse_aux = _grid_aux(2, 1)
    fine_input = torch.randn((1, 8, 12))
    coarse_input = restrict_channel_first(
        fine_input.permute(0, 2, 1),
        fine_resolution=(4, 2),
        coarse_resolution=(2, 1),
    ).permute(0, 2, 1)
    coarse_input[..., 7] += torch.tensor((0.0, 1.0))

    expected_coarse = model(coarse_input, coarse_aux)
    expected_fine = model(fine_input, fine_aux)
    coarse, fine, rows, fields = trace_resolution_pair(
        model,
        coarse_input,
        coarse_aux,
        fine_input,
        fine_aux,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
        collect_fields=True,
    )
    torch.testing.assert_close(coarse, expected_coarse, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(fine, expected_fine, rtol=1.0e-6, atol=1.0e-6)
    closures = [row for row in rows if row["record_kind"] == "closure"]
    branches = [row for row in rows if row["record_kind"] == "branch"]
    preactivations = [row for row in rows if row["record_kind"] == "preactivation"]
    assert max(float(row["relative_closure"] or 0.0) for row in closures) < 1.0e-5
    assert max(float(row["relative_closure"] or 0.0) for row in branches) < 1.0e-5
    assert max(float(row["relative_closure"] or 0.0) for row in preactivations) < 1.0e-5
    pointwise = [row for row in branches if row["branch"] == "pointwise"]
    assert (
        max(float(row["mesh_symmetric_relative"] or 0.0) for row in pointwise) < 1.0e-5
    )
    assert (
        max(
            float(row["pointwise_float64_mesh_symmetric_relative"] or 0.0)
            for row in pointwise
        )
        < 1.0e-14
    )
    assert all(row["mesh_to_combined_native_relative"] is not None for row in pointwise)
    assert any(row.get("input_group") == "node_type" for row in rows)
    assert "layer0_spectral_mesh" in fields


def test_fine_grained_same_hidden_pathways_close_exactly() -> None:
    model = _tiny_pcno()
    fine_aux = _grid_aux(4, 2)
    coarse_aux = _grid_aux(2, 1)
    fine_hidden = torch.randn((1, 8, 8))
    spectral = decompose_spectral_same_hidden_mesh_gap(
        model.sp_convs[0],
        fine_hidden,
        coarse_aux,
        fine_aux,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    torch.testing.assert_close(
        spectral["closure"],
        torch.zeros_like(spectral["closure"]),
        atol=2.0e-6,
        rtol=0.0,
    )
    coarse_hidden = restrict_channel_first(
        fine_hidden, fine_resolution=(4, 2), coarse_resolution=(2, 1)
    )
    coarse_bases = model(
        torch.zeros((1, 2, 12)), coarse_aux
    )  # exercise the same static geometry once
    del coarse_bases
    assert spectral["coarse_native"].shape == (1, 8, 2)
    torch.testing.assert_close(
        spectral["mesh_gap"],
        spectral["quadrature_response"]
        + spectral["subcell_hidden_response"]
        + spectral["synthesis_restriction_response"],
        atol=2.0e-6,
        rtol=1.0e-6,
    )

    differential = decompose_differential_same_hidden_mesh_gap(
        model.gws[0],
        fine_hidden,
        coarse_aux,
        fine_aux,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    torch.testing.assert_close(
        differential["pre_closure"],
        torch.zeros_like(differential["pre_closure"]),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        differential["output_closure"],
        torch.zeros_like(differential["output_closure"]),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert coarse_hidden.shape == (1, 8, 2)


def test_decoded_subpath_and_native_branch_replacements_are_nonmutating() -> None:
    model = _tiny_pcno()
    fine_aux = _grid_aux(4, 2)
    coarse_aux = _grid_aux(2, 1)
    fine_input = torch.randn((1, 8, 12))
    coarse_input = restrict_channel_first(
        fine_input.permute(0, 2, 1),
        fine_resolution=(4, 2),
        coarse_resolution=(2, 1),
    ).permute(0, 2, 1)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    outputs = trace_same_hidden_subpath_replacements(
        model,
        fine_input,
        coarse_aux,
        fine_aux,
        layer=0,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    assert set(outputs) == {
        "baseline",
        "fourier_quadrature",
        "fourier_subcell",
        "fourier_analysis",
        "fourier_synthesis",
        "fourier_full",
        "differential_gradient",
        "differential_fixed_hop",
        "differential_full",
        "pointwise_full",
        "restricted_fine",
    }
    assert all(value.shape == (1, 2, 4) for value in outputs.values())
    assert not torch.allclose(outputs["fourier_full"], outputs["baseline"])
    expected_fine = model(fine_input, fine_aux)
    expected_restricted_fine = restrict_channel_first(
        expected_fine.permute(0, 2, 1),
        fine_resolution=(4, 2),
        coarse_resolution=(2, 1),
    ).permute(0, 2, 1)
    torch.testing.assert_close(
        outputs["restricted_fine"],
        expected_restricted_fine,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    layer_one_outputs = trace_same_hidden_subpath_replacements(
        model,
        fine_input,
        coarse_aux,
        fine_aux,
        layer=1,
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    torch.testing.assert_close(
        layer_one_outputs["restricted_fine"],
        expected_restricted_fine,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    hidden = trace_native_hidden_to_layer(model, fine_input, fine_aux, layer=1)
    assert hidden.shape == (1, 8, 8)

    no_replacement = trace_native_branch_replacement_output(
        model,
        coarse_input,
        coarse_aux,
        fine_input,
        fine_aux,
        branch="spectral",
        replacement_layers=(),
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    expected_coarse = model(coarse_input, coarse_aux)
    torch.testing.assert_close(
        no_replacement["hybrid"], expected_coarse, atol=1.0e-6, rtol=1.0e-6
    )
    torch.testing.assert_close(
        no_replacement["restricted_fine"],
        expected_restricted_fine,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    replaced = trace_native_branch_replacement_output(
        model,
        coarse_input,
        coarse_aux,
        fine_input,
        fine_aux,
        branch="spectral",
        coarse_resolution=(2, 1),
        fine_resolution=(4, 2),
    )
    assert not torch.allclose(replaced["hybrid"], expected_coarse)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_gain_trace_is_identity_at_one_and_changes_selected_pathway() -> None:
    model = _tiny_pcno()
    aux = _grid_aux(4, 2)
    model_input = torch.randn((1, 8, 12))
    expected = model(model_input, aux)
    identity = trace_backbone_output(
        model,
        model_input,
        aux,
        branch_gains={
            (layer, branch): 1.0
            for layer in range(len(model.ws))
            for branch in BRANCH_NAMES
        },
    )
    attenuated = trace_backbone_output(
        model, model_input, aux, branch_gains={(0, "differential"): 0.99}
    )
    torch.testing.assert_close(identity, expected, rtol=1.0e-6, atol=1.0e-6)
    assert not torch.allclose(attenuated, expected)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        trace_backbone_output(
            model, model_input, aux, branch_gains={(0, "spectral"): 1.01}
        )


def test_generic_spectrum_recovers_a_physical_wavelength() -> None:
    nx, ny = 80, 40
    x = (np.arange(nx, dtype=np.float64) + 0.5) * 2.0 / nx
    wave = np.sin(2.0 * np.pi * x / 0.2)
    field = np.tile(wave, (ny, 1)).reshape(-1, 1)
    rows = generic_spectral_energy_rows(
        field, resolution=(nx, ny), domain_lengths=(2.0, 1.0)
    )
    target = next(
        row
        for row in rows
        if row["wavelength_min"] == 0.125 and row["wavelength_max"] == 0.25
    )
    assert target["spectral_energy_share"] > 0.9


def test_sub_005_spectral_share_is_exact_complement() -> None:
    common = {
        "case_id": "case",
        "pair": "coarse->fine",
        "call": "5",
        "branch": "differential",
        "layer": "all",
    }
    rows = [
        {**common, "spectral_energy_share": value}
        for value in ("0.20", "0.15", "0.10", "0.05")
    ]
    derived = _sub_005_spectral_rows(rows)
    assert len(derived) == 1
    assert derived[0]["sub_005_spectral_energy_share"] == pytest.approx(0.5)


def test_repeated_selector_requires_both_pairs_and_five_of_six_cases() -> None:
    cases = [f"case_{index}" for index in range(6)]
    pairs = ["coarse->native", "native->fine"]
    calls = [1, 5, 15, 30]
    rows = []
    for case_id in cases:
        for pair in pairs:
            for call in calls:
                for branch, elasticity in (
                    ("spectral", 1.2),
                    ("pointwise", 0.25),
                    ("differential", -0.1),
                ):
                    rows.append(
                        {
                            "case_id": case_id,
                            "pair": pair,
                            "call": call,
                            "branch": branch,
                            "defect_norm_elasticity": elasticity,
                        }
                    )
    selected = select_dominant_pathway(rows, case_ids=cases, pairs=pairs, calls=calls)
    assert selected["decision"] == "spectral"
    assert selected["families"]["spectral"]["qualifies_as_dominant"]
    with pytest.raises(ValueError, match="registered matrix"):
        select_dominant_pathway(rows[:-1], case_ids=cases, pairs=pairs, calls=calls)


def test_physical_gain_metric_preserves_residual_scale_and_signed_response() -> None:
    truth = np.ones((2, 4), dtype=np.float64)
    coarse = 1.3 * truth
    restricted_fine = truth.copy()
    volumes = np.asarray((1.0, 3.0))
    scale = np.ones(4)
    base, defect = _output_row(
        case_id="case",
        pair="coarse->fine",
        call=1,
        physical_time=0.02,
        truth=truth,
        coarse_increment=coarse,
        restricted_fine_increment=restricted_fine,
        volumes=volumes,
        residual_scale=scale,
        reference_floor=np.zeros_like(truth),
    )
    admissibility = {
        "finite": True,
        "admissible": True,
        "minimum_density": 1.0,
        "minimum_pressure": 1.0,
    }
    row, response = _gain_row(
        base=base,
        branch="spectral",
        layer=None,
        gain=0.99,
        truth=truth,
        base_defect=defect,
        coarse_increment=1.297 * truth,
        restricted_fine_increment=restricted_fine,
        volumes=volumes,
        residual_scale=scale,
        coarse_admissibility=admissibility,
        fine_admissibility=admissibility,
    )
    assert base["defect_relative_to_truth_increment"] == pytest.approx(0.3)
    assert row["defect_norm_ratio"] == pytest.approx(0.99)
    assert row["defect_norm_elasticity"] == pytest.approx(1.0)
    np.testing.assert_allclose(response, 0.3 * truth)


def test_exact_admissibility_api_is_bound() -> None:
    state = torch.tensor([[[1.0, 0.1, 0.0, 2.5]]])
    result = _admissibility(state, gamma=1.4)
    assert result["finite"]
    assert result["admissible"]
    assert result["minimum_density"] == pytest.approx(1.0)
    assert result["minimum_pressure"] > 0.0


def test_pathway_animation_uses_registered_bundle_fields(tmp_path) -> None:
    times = np.asarray((0.02, 0.04))
    shape = (times.size, 2, 4)
    arrays = {
        "schema": np.asarray("pcno_resolution_pathway_diagnostic_v1"),
        "case_id": np.asarray("case"),
        "physical_times": times,
        "residual_scale": np.ones(4),
    }
    pair_key = "2x1_to_4x2"
    for index, name in enumerate(
        (
            "true_increment",
            "baseline_defect",
            "spectral_response",
            "pointwise_response",
            "differential_response",
        ),
        start=1,
    ):
        arrays[f"{name}__{pair_key}"] = np.full(shape, 0.05 * index)
    bundle = tmp_path / "case.npz"
    np.savez_compressed(bundle, **arrays)
    output = tmp_path / "pathways.gif"
    animate_bundle(
        bundle,
        output,
        pair="2x1->4x2",
        limit=1.0,
        fps=1,
        dpi=30,
        frame_stride=1,
        overwrite=False,
    )
    assert output.is_file()
    assert output.with_suffix(".png").is_file()

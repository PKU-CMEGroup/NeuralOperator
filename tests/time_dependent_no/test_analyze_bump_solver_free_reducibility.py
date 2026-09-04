import json
from pathlib import Path

import numpy as np
import pytest

import scripts.time_dependent_no.analyze_bump_solver_free_reducibility as bump_pod
from scripts.time_dependent_no.analyze_bump_solver_free_reducibility import (
    DEFAULT_CONTRACT,
    _masked_centered_relative_error,
    aggregate_results,
    canonical_sha256,
    load_contract,
    normalized_node_weights,
    pod_from_gram,
    prefix_reconstructions,
    reconstruction_diagnostics,
    render_reducibility_figure,
    stage0_gate,
    temporal_snapshot_grams,
    write_reducibility_packet,
)
from utility.time_dependent_no.pcno_euler2d import Euler2DNormalization


def test_temporal_snapshot_grams_match_direct_uniform_and_proxy_metrics() -> None:
    rng = np.random.default_rng(9)
    states = rng.normal(size=(6, 4, 4))
    proxy = np.asarray([[0.1], [0.0], [0.3], [0.6]], dtype=np.float64)

    center, grams = temporal_snapshot_grams(
        states,
        start=0,
        stop=5,
        proxy_weights=proxy,
        block_nodes=2,
    )

    centered = states[:5] - np.mean(states[:5], axis=0, keepdims=True)
    np.testing.assert_allclose(center, np.mean(states[:5], axis=0))
    uniform = centered * np.sqrt(1.0 / (4 * 4))
    expected_uniform = uniform.reshape(5, -1) @ uniform.reshape(5, -1).T
    np.testing.assert_allclose(
        grams["uniform_normalized_state"], expected_uniform, rtol=1e-13, atol=1e-13
    )
    proxy_scale = np.sqrt(np.asarray([0.1, 0.0, 0.3, 0.6]) / 4.0)
    weighted = centered * proxy_scale[None, :, None]
    expected_proxy = weighted.reshape(5, -1) @ weighted.reshape(5, -1).T
    np.testing.assert_allclose(
        grams["proxy_mass_normalized_state"], expected_proxy, rtol=1e-13, atol=1e-13
    )


def test_proxy_weights_accept_zero_nodes_and_reject_invalid_values() -> None:
    observed = normalized_node_weights(
        3,
        metric="proxy_mass_normalized_state",
        proxy_weights=np.asarray([[1.0], [0.0], [3.0]]),
    )
    np.testing.assert_allclose(observed, [0.25, 0.0, 0.75])
    with pytest.raises(ValueError, match="nonnegative"):
        normalized_node_weights(
            2,
            metric="proxy_mass_normalized_state",
            proxy_weights=np.asarray([1.0, -1.0]),
        )
    with pytest.raises(ValueError, match="positive total"):
        normalized_node_weights(
            2,
            metric="proxy_mass_normalized_state",
            proxy_weights=np.zeros(2),
        )


def test_pod_recovers_rank_two_and_constant_sequence_is_explicit() -> None:
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=(9, 2))
    coefficients -= np.mean(coefficients, axis=0, keepdims=True)
    features = rng.normal(size=(2, 20))
    flat = coefficients @ features
    result = pod_from_gram(flat @ flat.T)

    assert result.numerical_rank == 2
    assert result.rank_at(0.999) == 2
    assert result.cumulative_explained_variance[1] == pytest.approx(1.0)
    assert result.zero_temporal_variance is False

    constant = pod_from_gram(np.zeros((5, 5), dtype=np.float64))
    assert constant.zero_temporal_variance is True
    assert constant.numerical_rank == 0
    assert constant.rank_at(0.999) == 0


def test_time_independent_field_cancels_under_trajectory_centering() -> None:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(7, 5, 4))
    offset = rng.normal(size=(1, 5, 4))
    proxy = np.arange(1, 6, dtype=np.float64)

    _, base = temporal_snapshot_grams(
        states, start=0, stop=7, proxy_weights=proxy, block_nodes=3
    )
    _, shifted = temporal_snapshot_grams(
        states + offset, start=0, stop=7, proxy_weights=proxy, block_nodes=2
    )
    for metric in base:
        np.testing.assert_allclose(base[metric], shifted[metric], atol=1e-13)


def test_distinct_native_mesh_sizes_are_analyzed_without_padding() -> None:
    rng = np.random.default_rng(17)
    first = rng.normal(size=(6, 3, 4))
    second = rng.normal(size=(6, 7, 4))
    _, first_grams = temporal_snapshot_grams(
        first, start=0, stop=6, proxy_weights=np.ones(3), block_nodes=2
    )
    _, second_grams = temporal_snapshot_grams(
        second, start=0, stop=6, proxy_weights=np.ones(7), block_nodes=3
    )
    assert first_grams["uniform_normalized_state"].shape == (6, 6)
    assert second_grams["uniform_normalized_state"].shape == (6, 6)
    assert not np.allclose(
        first_grams["uniform_normalized_state"],
        second_grams["uniform_normalized_state"],
    )


def test_prefix_reconstruction_recovers_future_states_in_prefix_span() -> None:
    rng = np.random.default_rng(12)
    spatial_modes = rng.normal(size=(2, 3, 4))
    baseline = rng.normal(size=(3, 4))
    prefix_coefficients = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    future_coefficients = np.asarray([[0.5, 0.25], [-0.2, 0.7], [0.0, -0.4]])
    states = np.concatenate(
        (
            baseline[None]
            + np.einsum("tk,knc->tnc", prefix_coefficients, spatial_modes),
            baseline[None]
            + np.einsum("tk,knc->tnc", future_coefficients, spatial_modes),
        ),
        axis=0,
    )
    center, grams = temporal_snapshot_grams(
        states,
        start=0,
        stop=4,
        proxy_weights=np.ones(3),
        block_nodes=2,
    )
    spectrum = pod_from_gram(grams["uniform_normalized_state"])
    outputs = prefix_reconstructions(
        states,
        prefix_start=0,
        prefix_stop=4,
        future_stop=7,
        center=center,
        spectrum=spectrum,
        node_weights=np.full(3, 1.0 / 3.0),
        ranks=(1, 2, 8),
        block_nodes=2,
    )

    assert outputs[1][0] == 1
    assert outputs[2][0] == 2
    assert outputs[8][0] == 2
    np.testing.assert_allclose(outputs[2][1], states[4:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outputs[8][1], states[4:], rtol=1e-12, atol=1e-12)


def test_masked_error_uses_prefix_centered_denominator() -> None:
    target = np.asarray([[[2.0], [4.0]], [[3.0], [5.0]]])
    reconstruction = target + 1.0
    center = np.asarray([[1.0], [2.0]])
    mask = np.asarray([[True, False], [True, False]])
    observed = _masked_centered_relative_error(reconstruction, target, center, mask)
    expected = np.sqrt(2.0 / (1.0**2 + 2.0**2))
    assert observed == pytest.approx(expected)


def test_reconstruction_diagnostics_preserve_identical_valid_state() -> None:
    pressure = np.asarray([1.0, 1.1, 2.0, 2.2, 2.3])
    conservative = np.zeros((3, 5, 4), dtype=np.float64)
    conservative[..., 0] = 1.0
    conservative[..., 3] = pressure[None] / 0.4
    normalization = Euler2DNormalization(
        state_mean=np.zeros(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        mach_mean=1.8,
        mach_scale=0.1,
    )
    geometry = {
        "nodes": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        ),
        "directed_edges": np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64),
        "node_type": np.asarray([0, 1, 0, 1, 0], dtype=np.int64),
    }
    observed = reconstruction_diagnostics(
        conservative,
        conservative,
        center=np.zeros((5, 4)),
        node_weights=np.full(5, 0.2),
        normalization=normalization,
        geometry=geometry,
    )

    assert observed["future_relative_error"] == 0.0
    assert observed["fraction_nonpositive_density"] == 0.0
    assert observed["fraction_nonpositive_pressure"] == 0.0
    assert observed["shock_centroid_distance_mean"] == 0.0
    assert observed["shock_thickness_ratio_mean"] == pytest.approx(1.0)
    assert observed["shock_strength_ratio_mean"] == pytest.approx(1.0)
    assert observed["boundary_relative_l2"] == 0.0


def test_stage0_gate_is_fixed_and_population_count_is_enforced() -> None:
    observed = stage0_gate(
        [8] * 24 + [7] * 8,
        reference_rank=7,
        minimum_count=24,
        total_count=32,
    )
    assert observed["passed"] is True
    assert observed["trajectories_above_reference_rank"] == 24
    with pytest.raises(ValueError, match="wrong number"):
        stage0_gate([8] * 31, reference_rank=7, minimum_count=24, total_count=32)


def test_contract_and_preregistration_hashes_close() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    canonical = dict(contract)
    claimed = canonical.pop("canonical_payload_sha256")
    assert claimed == canonical_sha256(canonical)
    assert contract["stage0_reducibility"]["execution_status"] == (
        "authorized_after_identity_checks"
    )
    assert contract["stage1"]["execution_status"].startswith("blocked_")
    assert contract["remote_execution_authorized"] is False
    assert contract["historical_test_population_opened"] is False


def test_contract_rejects_digest_change_and_protected_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    temporary = tmp_path / DEFAULT_CONTRACT.name
    monkeypatch.setattr(bump_pod, "DEFAULT_CONTRACT", temporary)

    changed = dict(original)
    changed["scientific_role"] = "changed"
    temporary.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical payload"):
        load_contract(temporary)

    protected = dict(original)
    protected["historical_test_population_opened"] = True
    canonical = dict(protected)
    canonical.pop("canonical_payload_sha256")
    protected["canonical_payload_sha256"] = canonical_sha256(canonical)
    temporary.write_text(json.dumps(protected), encoding="utf-8")
    with pytest.raises(ValueError, match="historical-test"):
        load_contract(temporary)


def test_render_reducibility_figure_writes_pdf_and_png(tmp_path: Path) -> None:
    spectrum_rows = []
    reconstruction_rows = []
    for metric, decay in (
        ("uniform_normalized_state", 0.82),
        ("proxy_mass_normalized_state", 0.87),
    ):
        for key in ("1", "2", "3"):
            for component in range(1, 9):
                spectrum_rows.append(
                    {
                        "trajectory_key": key,
                        "metric": metric,
                        "basis_scope": "oracle_model_inputs_0_78",
                        "component": component,
                        "cumulative_explained_variance": 1.0 - decay**component,
                    }
                )
            for rank in (2, 4, 7, 16, 32):
                reconstruction_rows.append(
                    {
                        "trajectory_key": key,
                        "metric": metric,
                        "requested_rank": rank,
                        "future_relative_error": 1.0 / rank,
                    }
                )
    pdf = tmp_path / "figure.pdf"
    png = tmp_path / "figure.png"
    render_reducibility_figure(
        pdf,
        png,
        spectrum_rows,
        reconstruction_rows,
        gate={"trajectories_above_reference_rank": 24, "total_trajectories": 32},
        max_components=8,
    )
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_aggregate_results_retains_non_tautological_gate() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    stage0 = contract["stage0_reducibility"]
    spectrum_rows = []
    reconstruction_rows = []
    for index, key in enumerate(stage0["audit_keys"]):
        for metric in stage0["metrics"]:
            spectrum_rows.append(
                {
                    "trajectory_key": key,
                    "metric": metric,
                    "basis_scope": "oracle_model_inputs_0_78",
                    "component": 1,
                    "rank_999_permille": 8 if index < 24 else 7,
                    "participation_rank": 4.0,
                    "entropy_rank": 5.0,
                }
            )
            for rank in stage0["reconstruction_ranks"]:
                reconstruction_rows.append(
                    {
                        "trajectory_key": key,
                        "metric": metric,
                        "requested_rank": rank,
                        "future_relative_error": 0.1,
                        "shock_centroid_distance_mean": 0.01,
                        "shock_thickness_ratio_mean": 1.0,
                        "shock_strength_ratio_mean": 1.0,
                        "fraction_nonpositive_pressure": 0.0,
                    }
                )
    aggregate = aggregate_results(
        spectrum_rows,
        reconstruction_rows,
        audit_keys=stage0["audit_keys"],
        stage0=stage0,
    )
    assert aggregate["linear_reducibility_contrast_gate"]["passed"] is True


def test_aggregate_results_refuse_incomplete_metric_coverage() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    stage0 = contract["stage0_reducibility"]
    key = stage0["audit_keys"][0]
    spectrum_rows = [
        {
            "trajectory_key": key,
            "metric": "uniform_normalized_state",
            "basis_scope": "oracle_model_inputs_0_78",
            "component": 1,
            "rank_999_permille": 8,
            "participation_rank": 4.0,
            "entropy_rank": 5.0,
        }
    ]
    reconstruction_rows = []

    with pytest.raises(ValueError, match="oracle spectrum rows"):
        aggregate_results(
            spectrum_rows,
            reconstruction_rows,
            audit_keys=stage0["audit_keys"],
            stage0=stage0,
        )


def test_aggregate_results_refuse_duplicate_key_as_complete_coverage() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    stage0 = contract["stage0_reducibility"]
    first_key = stage0["audit_keys"][0]
    spectrum_rows = [
        {
            "trajectory_key": first_key,
            "metric": "uniform_normalized_state",
            "basis_scope": "oracle_model_inputs_0_78",
            "component": 1,
            "rank_999_permille": 8,
            "participation_rank": 4.0,
            "entropy_rank": 5.0,
        }
        for _ in stage0["audit_keys"]
    ]
    with pytest.raises(ValueError, match="exactly once"):
        aggregate_results(
            spectrum_rows,
            [],
            audit_keys=stage0["audit_keys"],
            stage0=stage0,
        )


def test_packet_refuses_existing_output_before_opening_data(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_reducibility_packet(
            output,
            data_root=tmp_path / "missing-data",
        )

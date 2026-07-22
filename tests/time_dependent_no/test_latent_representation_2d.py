from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.diagnose_euler2d_latent_code_reachability import (
    classify_reachability,
)
from scripts.time_dependent_no.train_euler2d_latent_representation import (
    _load_state_batch,
    _manifest_splits,
    _select_spread,
)
from utility.time_dependent_no.latent_representation_2d import (
    LINE3_HANDOFF_SCHEMA,
    LINE4_LATENT_SIZE,
    SpatialTokenAutoencoder,
    WeightedSnapshotPOD,
    build_token_geometry,
    conditional_future_diagnostics,
    empirical_decoder_gains,
    fit_frozen_decoder_code,
    fit_channel_whitener,
    reconstruction_metrics,
    representation_artifact_ledger,
    token_weighted_mean,
    validate_line3_handoff,
)
from utility.time_dependent_no.pcno_euler2d import Euler2DNormalization


def _normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.asarray([1.0, 0.2, 0.0, 2.6]),
        state_scale=np.asarray([0.25, 0.3, 0.2, 0.8]),
        residual_scale=np.ones(4),
        mach_mean=1.1,
        mach_scale=0.1,
        weight_provenance="validated_physical_cell_volume_normalized",
    )


def _grid(nx: int, ny: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) * (2.0 / nx)
    y = (torch.arange(ny, dtype=torch.float32) + 0.5) * (1.0 / ny)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    volumes = torch.full((nx * ny,), 2.0 / (nx * ny), dtype=torch.float32)
    return nodes, volumes


def _positive_state(nodes: torch.Tensor, *, batch_size: int = 1) -> torch.Tensor:
    x = nodes[:, 0]
    y = nodes[:, 1]
    rho = 1.0 + 0.08 * x + 0.03 * y
    velocity_x = 0.2 + 0.02 * y
    velocity_y = 0.01 * torch.sin(torch.pi * x)
    pressure = 1.0 + 0.05 * x - 0.02 * y
    energy = pressure / 0.4 + 0.5 * rho * (velocity_x.square() + velocity_y.square())
    state = torch.stack(
        (rho, rho * velocity_x, rho * velocity_y, energy),
        dim=-1,
    )
    return state.unsqueeze(0).repeat(batch_size, 1, 1)


def _handoff_payload() -> dict[str, object]:
    return {
        "schema": LINE3_HANDOFF_SCHEMA,
        "status": "frozen",
        "line4_training_truth_authorized": True,
        "line4_transition_training_authorized": False,
        "line4_front_candidate_available": False,
        "dependencies": {
            "line4_reconstruction_and_closure_tests_may_reopen": True,
            "line4_transition_training_still_requires_representation_gates": True,
            "line4_assimilation_remains_blocked_by_raw_open_loop_gate": True,
        },
        "training_truth": {
            "data_manifest_digest": "manifest-digest",
            "normalization_digest": "normalization-digest",
            "grouped_split_digest": "split-digest",
            "state_convention": "conservative_[rho,rho_u,rho_v,E]",
            "weight_provenance": "validated_physical_cell_volume_normalized",
            "resolution_contract": {
                "stored_grid": [250, 100],
                "node_counts": [25000],
            },
        },
        "physical_baseline": {
            "split": "validation",
            "calls": 60,
            "raw_recurrence": True,
            "boundary_mode": "model_all_nodes",
            "inference_interventions": {
                "clipping": False,
                "decode_reencode_projection": False,
                "future_reference_boundary_values": False,
                "limiter": False,
                "primitive_floors": False,
            },
            "aggregates": {"pcno_baseline": {"completion_rate": 1.0}},
            "front_and_smooth_hierarchy": [{} for _ in range(24)],
            "grouped_parameter_ood": [{} for _ in range(8)],
            "cost": {"batch1_seconds_median": 0.01},
            "reference_contract": {"reference_checks": [{} for _ in range(24)]},
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_handoff_validation_releases_only_representation_work(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    digest = _write_json(path, _handoff_payload())

    audit = validate_line3_handoff(
        path,
        expected_sha256=digest,
        expected_data_manifest_digest="manifest-digest",
    )

    assert audit["training_truth_authorized"] is True
    assert audit["front_candidate_available"] is False
    assert audit["transition_training_authorized"] is False
    assert audit["assimilation_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("line4_transition_training_authorized",), True, "transition authorization"),
        (
            (
                "physical_baseline",
                "inference_interventions",
                "decode_reencode_projection",
            ),
            True,
            "intervention",
        ),
        (("physical_baseline", "cost"), {}, "cost contract"),
    ],
)
def test_handoff_validation_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = copy.deepcopy(_handoff_payload())
    target: dict[str, object] = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    handoff_path = tmp_path / "bad-handoff.json"
    digest = _write_json(handoff_path, payload)

    with pytest.raises(ValueError, match=message):
        validate_line3_handoff(handoff_path, expected_sha256=digest)


@pytest.mark.parametrize(("nx", "ny"), [(250, 100), (500, 200)])
def test_physical_token_lattice_is_resolution_independent(nx: int, ny: int) -> None:
    nodes, volumes = _grid(nx, ny)

    geometry = build_token_geometry(nodes, volumes)

    assert geometry.num_tokens == 250
    assert geometry.token_volumes.shape == (1, 250)
    torch.testing.assert_close(
        geometry.token_volumes,
        torch.full_like(geometry.token_volumes, 2.0 / 250),
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        geometry.log_relative_volume,
        torch.zeros_like(geometry.log_relative_volume),
        rtol=0.0,
        atol=5.0e-6,
    )


@pytest.mark.parametrize("trailing_dimensions", [(1,), (1, 1)])
def test_physical_token_lattice_accepts_shard_measure_shapes(
    trailing_dimensions: tuple[int, ...],
) -> None:
    nodes, volumes = _grid(8, 4)
    shaped_volumes = volumes.reshape(32, *trailing_dimensions)

    geometry = build_token_geometry(
        nodes,
        shaped_volumes,
        token_nx=2,
        token_ny=2,
    )

    assert geometry.volumes.shape == (1, 32)
    torch.testing.assert_close(geometry.volumes[0], volumes)


def test_matched_models_have_equal_capacity_and_declared_5000_code() -> None:
    generic = SpatialTokenAutoencoder(_normalization(), variant="generic")
    hybrid = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
    )

    assert generic.parameter_count == hybrid.parameter_count
    assert generic.latent_size == hybrid.latent_size == LINE4_LATENT_SIZE
    assert generic.contract()["fixed_conservative_channels"] == 0
    assert hybrid.contract()["fixed_conservative_channels"] == 4
    assert hybrid.contract()["learned_channels"] == 16
    assert hybrid.contract()["front_variables"] == 0
    assert hybrid.contract()["decode_reencode_projection"] is False


def test_hybrid_decoder_preserves_token_moments_and_backward() -> None:
    torch.manual_seed(4)
    nodes, volumes = _grid(8, 4)
    geometry = build_token_geometry(nodes, volumes, token_nx=2, token_ny=2)
    state = _positive_state(nodes)
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=12,
    )
    final_layer = model.point_decoder[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    torch.nn.init.normal_(final_layer.weight, std=0.05)
    torch.nn.init.normal_(final_layer.bias, std=0.02)

    decoded, code = model(state, geometry)
    loss = (decoded - state).square().mean()
    loss.backward()

    torch.testing.assert_close(
        token_weighted_mean(decoded.detach(), geometry),
        token_weighted_mean(state, geometry),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert code.shape == (1, 4, 6)
    assert model.point_decoder[-1].weight.grad is not None
    metrics = reconstruction_metrics(
        decoded.detach(),
        state,
        geometry,
        state_scale=model.state_scale,
    )
    assert metrics["token_moment_max_abs"] < 2.0e-6
    assert metrics["intervention_applied"] is False


def test_raw_hybrid_decode_exposes_inadmissibility_without_repair() -> None:
    nodes, volumes = _grid(4, 2)
    geometry = build_token_geometry(nodes, volumes, token_nx=2, token_ny=1)
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
        token_nx=2,
        token_ny=1,
        token_channels=6,
        hidden_channels=8,
    )
    code = torch.zeros((1, 2, 6), dtype=torch.float32)
    code[..., 0] = (-0.5 - model.state_mean[..., 0]) / model.state_scale[..., 0]

    decoded = model.decode(code, geometry)

    assert torch.all(decoded[..., 0] < 0.0)
    assert model.contract()["clipping"] is False
    assert model.contract()["floors"] is False
    assert model.contract()["limiter"] is False


def test_frozen_decoder_code_fit_changes_only_free_code() -> None:
    torch.manual_seed(13)
    nodes, volumes = _grid(8, 4)
    geometry = build_token_geometry(nodes, volumes, token_nx=2, token_ny=2)
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=12,
    )
    final_layer = model.point_decoder[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    torch.nn.init.normal_(final_layer.weight, std=0.5)
    torch.nn.init.normal_(final_layer.bias, std=0.02)
    state = _positive_state(nodes)
    with torch.no_grad():
        initial_code = model.encode(state, geometry)
        target_code = initial_code.clone()
        target_code[..., 4:] += 2.0 * torch.randn_like(target_code[..., 4:])
        target = model.decode(target_code, geometry)
    parameter_copies = [parameter.detach().clone() for parameter in model.parameters()]
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]

    fitted, diagnostics = fit_frozen_decoder_code(
        model,
        target,
        geometry,
        initial_code,
        max_iter=80,
        max_eval=110,
        history_size=10,
    )

    assert diagnostics["final_scaled_volume_mse"] < (
        0.2 * diagnostics["initial_scaled_volume_mse"]
    )
    assert diagnostics["per_state_fitting"] is True
    assert diagnostics["forecast_evidence"] is False
    torch.testing.assert_close(fitted[..., :4], initial_code[..., :4], rtol=0, atol=0)
    for parameter, expected, expected_requires_grad in zip(
        model.parameters(),
        parameter_copies,
        requires_grad,
        strict=True,
    ):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        assert parameter.requires_grad is expected_requires_grad


def test_frozen_decoder_code_fit_rejects_generic_model() -> None:
    nodes, volumes = _grid(8, 4)
    geometry = build_token_geometry(nodes, volumes, token_nx=2, token_ny=2)
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="generic",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=8,
    )
    state = _positive_state(nodes)
    code = model.encode(state, geometry)

    with pytest.raises(ValueError, match="conservative model"):
        fit_frozen_decoder_code(model, state, geometry, code, max_iter=2, max_eval=3)


def test_reachability_classification_requires_physics_and_nearby_code() -> None:
    control = {"maximum_density_pressure_overshoot": 0.03}

    def aggregates(*, l2: float, displacement: float) -> dict[str, dict[str, float]]:
        return {
            "amortized_encoder": control,
            "fitted_code": {
                "mean_relative_l2": l2,
                "minimum_admissible_fraction": 1.0,
                "mean_shock_strength_ratio": 1.01,
                "mean_shock_thickness_ratio": 0.99,
                "maximum_density_pressure_overshoot": 0.02,
                "maximum_normalized_free_code_displacement_rms": displacement,
            },
        }

    rejected, rejected_gates = classify_reachability(
        aggregates(l2=0.003, displacement=0.5)
    )
    distant, distant_gates = classify_reachability(
        aggregates(l2=0.001, displacement=1.5)
    )
    nearby, nearby_gates = classify_reachability(aggregates(l2=0.001, displacement=0.5))

    assert rejected == "decoder_manifold_rejected"
    assert not rejected_gates["mean_relative_l2_at_most_0p0021"]
    assert distant == "off_manifold_decoder_capacity_only"
    assert not distant_gates["all_fitted_codes_within_one_training_scale_rms"]
    assert nearby == "amortized_encoder_defect_supported"
    assert all(nearby_gates.values())


def test_same_code_decodes_on_doubled_query_resolution() -> None:
    coarse_nodes, coarse_volumes = _grid(8, 4)
    fine_nodes, fine_volumes = _grid(16, 8)
    coarse_geometry = build_token_geometry(
        coarse_nodes,
        coarse_volumes,
        token_nx=2,
        token_ny=2,
    )
    fine_geometry = build_token_geometry(
        fine_nodes,
        fine_volumes,
        token_nx=2,
        token_ny=2,
    )
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=8,
    )
    code = model.encode(_positive_state(coarse_nodes), coarse_geometry)

    decoded = model.decode(code, fine_geometry)
    expected_moments = model.state_mean + model.state_scale * code[..., :4]

    assert decoded.shape == (1, 128, 4)
    torch.testing.assert_close(
        token_weighted_mean(decoded, fine_geometry),
        expected_moments,
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_closure_diagnostic_detects_when_one_previous_code_matters() -> None:
    current_values: list[float] = []
    previous_values: list[float] = []
    for current in range(4):
        current_values.extend([float(current)] * 4)
        previous_values.extend([-1.0, -1.0, 1.0, 1.0])
    current_codes = torch.tensor(current_values, dtype=torch.float64).view(-1, 1, 1)
    previous_codes = torch.tensor(previous_values, dtype=torch.float64).view(-1, 1, 1)
    future_codes = previous_codes.clone()
    whitener = fit_channel_whitener(current_codes)

    diagnostics = conditional_future_diagnostics(
        current_codes,
        future_codes,
        whitener=whitener,
        previous_codes=previous_codes,
        neighbors=1,
        chunk_size=5,
    )

    assert diagnostics["current_only"]["future_neighbor_rms_mean"] > 0.0
    assert (
        diagnostics["one_previous_code"]["future_ambiguity_ratio_to_current_only"] < 0.8
    )
    assert diagnostics["one_previous_code"]["material_improvement_at_20_percent"]
    neighbors = diagnostics["current_only"]["neighbor_indices"]
    assert all(index != row for row, (index,) in enumerate(neighbors))


def test_decoder_gain_is_raw_and_finite() -> None:
    torch.manual_seed(8)
    nodes, volumes = _grid(8, 4)
    geometry = build_token_geometry(nodes, volumes, token_nx=2, token_ny=2)
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="generic",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=8,
    )
    final_layer = model.point_decoder[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    torch.nn.init.normal_(final_layer.weight, std=0.05)
    code = model.encode(_positive_state(nodes), geometry)
    directions = torch.randn((3, *code.shape))

    gains = empirical_decoder_gains(
        model,
        code,
        geometry,
        directions,
        epsilon=2.0e-3,
    )

    assert gains["directions"] == 3
    assert 0.0 < gains["mean_gain"] < float("inf")
    assert gains["raw_decode"] is True
    assert gains["intervention_applied"] is False


def test_weighted_pod_roundtrip_on_resolved_rank_two_data() -> None:
    torch.manual_seed(2)
    nodes, volumes = _grid(3, 2)
    base = _positive_state(nodes)[0]
    direction_a = 0.02 * torch.randn_like(base)
    direction_b = 0.02 * torch.randn_like(base)
    coefficients = torch.tensor(
        [[-2.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]
    )
    states = torch.stack(
        [base + a * direction_a + b * direction_b for a, b in coefficients],
        dim=0,
    )

    pod = WeightedSnapshotPOD.fit(
        states,
        volumes,
        _normalization(),
        rank=2,
    )
    reconstructed = pod.decode(pod.encode(states))

    torch.testing.assert_close(reconstructed, states, rtol=2.0e-5, atol=2.0e-5)
    assert pod.contract()["effective_rank"] == 2
    assert pod.contract()["resolution_contract"] == "fixed_mesh_only"
    assert pod.contract()["decode_reencode_projection"] is False


def test_artifact_ledger_names_states_and_forbids_analysis() -> None:
    model = SpatialTokenAutoencoder(
        _normalization(),
        variant="conservative_moment",
        token_nx=2,
        token_ny=2,
        token_channels=6,
        hidden_channels=8,
    )

    ledger = representation_artifact_ledger(
        model,
        split="validation",
        trajectory="synthetic-trajectory",
        physical_states_file="physical_states.npz",
        latent_states_file="latent_states.npz",
        valid_length=5,
        failure_cause="completed",
        seed=17,
    )

    assert ledger["physical_states_file"] == "physical_states.npz"
    assert ledger["latent_states_file"] == "latent_states.npz"
    assert len(ledger["encoder_sha256"]) == 64
    assert len(ledger["decoder_sha256"]) == 64
    assert ledger["analysis_applied"] is False
    assert ledger["ensemble_seeds"] == []
    assert ledger["transition_present"] is False
    assert not any(ledger["interventions"].values())

    with pytest.raises(ValueError, match="train or validation"):
        representation_artifact_ledger(
            model,
            split="test",
            trajectory="sealed-test",
            physical_states_file="physical_states.npz",
            latent_states_file="latent_states.npz",
            valid_length=1,
            failure_cause="completed",
            seed=17,
        )


class _FakeSplitStore:
    def __init__(self) -> None:
        self.manifest = {
            "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "declared_split_counts": {"train": 1, "validation": 1, "test": 1},
            "prepared_split_counts": {"train": 1, "validation": 1, "test": 1},
        }
        self._entries = {
            "a": {"split": "train"},
            "b": {"split": "validation"},
            "c": {"split": "test"},
        }
        self.states_calls = 0

    @property
    def keys(self) -> list[str]:
        return ["a", "b", "c"]

    def entry(self, key: str) -> dict[str, str]:
        return self._entries[key]

    def states(self, _key: str) -> np.ndarray:
        self.states_calls += 1
        return np.zeros((2, 3, 4), dtype=np.float32)


def test_smoke_manifest_contract_is_exact_and_test_arrays_stay_sealed() -> None:
    store = _FakeSplitStore()

    assert _manifest_splits(store) == (["a"], ["b"], ["c"])
    assert _select_spread(["a", "b", "c", "d"], 2) == ["a", "d"]
    with pytest.raises(ValueError, match="test arrays are sealed"):
        _load_state_batch(store, [("c", 0)], torch.device("cpu"))
    assert store.states_calls == 0


def test_smoke_manifest_rejects_overlap() -> None:
    store = _FakeSplitStore()
    store.manifest["splits"]["validation"] = ["a"]

    with pytest.raises(ValueError, match="partition"):
        _manifest_splits(store)

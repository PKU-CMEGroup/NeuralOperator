from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.evaluate_pcno_bump_geometric_consistency as d083_evaluator
from scripts.time_dependent_no.evaluate_pcno_bump_geometric_consistency import (
    GeometryArm,
    _assert_unique_metric_rows,
    _deterministic_identity_gate,
    _final_summary,
    _increment_identity_metrics,
    _matrix_coverage,
    _record_native_residual_regions,
)
from scripts.time_dependent_no.visualize_pcno_bump_geometric_consistency import (
    INPUT_CONTRACTS,
    INPUT_SCHEMA,
    _read_admissible_prefixes,
    _read_metrics,
    _verify_input,
    paired_admissible_prefix,
    reference_color_scales,
    residual_animation_fields,
)


def _artifact() -> dict[str, np.ndarray]:
    truth = np.zeros((4, 3, 4), dtype=np.float64)
    truth[1:, :, 0] = np.asarray([1.0, 3.0, 6.0])[:, None]
    native = np.array(truth, copy=True)
    native[1:, :, 0] += 0.25
    query = np.array(truth, copy=True)
    query[1:, :, 0] += 0.5
    rotated = np.array(truth, copy=True)
    rotated[1:, :, 0] -= 0.25
    mesh = np.full((3, 3, 4), 0.1)
    state = np.full((3, 3, 4), -0.025)
    return {
        "truth": truth,
        "native_correct": native,
        "native_all_normal": native,
        "query_correct_prolonged": query,
        "query_all_normal_prolonged": query,
        "rotated_correct_inverse": rotated,
        "rotated_all_normal_inverse": rotated,
        "query_mesh_correct": mesh,
        "query_state_correct": state,
        "query_mesh_all_normal": mesh,
        "query_state_all_normal": state,
        "rotation_mesh_correct": mesh,
        "rotation_state_correct": state,
        "rotation_mesh_all_normal": mesh,
        "rotation_state_all_normal": state,
    }


def test_reference_color_scales_ignore_predictions_and_use_full_rollout():
    artifact = _artifact()
    scales = reference_color_scales(artifact["truth"], 0)
    assert scales["residual_limit"] == 3.0
    assert scales["cumulative_limit"] == 6.0
    assert scales["signed_growth_limit"] == 45.0
    changed_prediction = np.full_like(artifact["native_correct"], 1.0e9)
    artifact["native_correct"] = changed_prediction
    assert reference_color_scales(artifact["truth"], 0) == scales


def test_residual_animation_fields_preserve_total_cumulative_and_growth_identity():
    artifact = _artifact()
    titles, fields, scales = residual_animation_fields(
        artifact, phase="g1a", type_arm="correct", component=0
    )
    assert len(titles) == len(fields) == 10
    np.testing.assert_allclose(fields[5], 0.075)
    np.testing.assert_allclose(fields[6][:, 0], np.asarray([0.075, 0.15, 0.225]))
    previous = fields[6] - fields[5]
    np.testing.assert_allclose(fields[9], 2.0 * previous * fields[5] + fields[5] ** 2)
    assert scales == reference_color_scales(artifact["truth"], 0)

    _, rotation_fields, _ = residual_animation_fields(
        artifact,
        phase="g1b",
        type_arm="all_normal",
        component=0,
        rotation_prediction_label="inverse-rotated prediction (fixed checkpoint modes)",
    )
    np.testing.assert_allclose(rotation_fields[5], fields[5])


def test_horizon_inadmissibility_is_not_counted_as_endpoint_completion():
    completion = []
    for geometry in ("native", "query"):
        for type_arm in ("correct", "all_normal"):
            failed = geometry == "query" and type_arm == "correct"
            completion.append(
                {
                    "case_id": "128",
                    "geometry": geometry,
                    "type_arm": type_arm,
                    "fully_admissible": not failed,
                    "first_inadmissible_call": 79 if failed else None,
                    "horizon_output_produced": True,
                }
            )
    metrics = []
    for geometry in ("native", "query"):
        for type_arm in ("correct", "all_normal"):
            metrics.append(
                {
                    "case_id": "128",
                    "phase": "g1a",
                    "arm": f"{geometry}_{type_arm}",
                    "mode": "free_rollout",
                    "call": 79,
                    "metric": "state_relative_l2",
                    "frame": "native",
                    "region": "all",
                    "component": "all",
                    "value": 0.25,
                }
            )

    coverage = _matrix_coverage(completion, requested_phase="g1a", cases=("128",))
    assert coverage["declared_matrix_attempt_complete"] is True
    assert coverage["declared_matrix_horizon_complete"] is False
    assert coverage["inadmissible_phase_arm_case_count"] == 1

    summary = _final_summary(
        metrics,
        completion,
        requested_phase="g1a",
        cases=("128",),
        horizon=79,
    )
    query_correct = next(row for row in summary if row["arm"] == "query_correct")
    assert query_correct["horizon_admissible_case_count"] == 0
    assert query_correct["endpoint_metric_case_count"] == 0
    assert query_correct["mean_horizon_state_relative_l2"] is None
    assert query_correct["endpoint_claim_allowed"] is False


def test_identity_metric_uses_increment_scale_and_restores_runtime_mode():
    current = np.zeros((3, 4), dtype=np.float64)
    reference = np.full((3, 4), 0.1, dtype=np.float64)
    value = reference + 0.001
    metrics = _increment_identity_metrics(
        value,
        reference,
        current,
        weights=np.ones(3),
        residual_scale=np.ones(4),
    )
    assert metrics["max_abs"] == pytest.approx(0.001)
    assert metrics["relative_l2"] == pytest.approx(0.01)
    assert metrics["pointwise_component_scaled_relative_max"] == pytest.approx(0.01)

    previous = torch.are_deterministic_algorithms_enabled()
    with _deterministic_identity_gate(torch.device("cpu")):
        assert torch.are_deterministic_algorithms_enabled()
    assert torch.are_deterministic_algorithms_enabled() is previous


def test_scientific_predict_serializes_multiple_type_arms(monkeypatch):
    calls = []

    def fake_batched(model, geometry, currents, type_arms, *, device):
        assert len(currents) == len(type_arms) == 1
        calls.append(type_arms[0])
        return {type_arms[0]: np.asarray(currents[0]) + len(calls)}

    monkeypatch.setattr(d083_evaluator, "_predict_batched", fake_batched)
    current_correct = np.zeros((2, 4))
    current_all_normal = np.ones((2, 4))
    predictions = d083_evaluator._predict(
        object(),
        object(),
        (current_correct, current_all_normal),
        ("correct", "all_normal"),
        device=torch.device("cpu"),
    )
    assert calls == ["correct", "all_normal"]
    np.testing.assert_array_equal(predictions["correct"], current_correct + 1)
    np.testing.assert_array_equal(predictions["all_normal"], current_all_normal + 2)


def test_fixed_rotation_predict_receives_direct_nodes_and_raw_state():
    captured = {}

    class Backbone:
        def __call__(self, model_input, aux, *, fourier_tensors):
            captured["aux_nodes"] = aux[1].detach().clone()
            captured["fourier"] = tuple(
                value.detach().clone() for value in fourier_tensors
            )
            return torch.zeros((*model_input.shape[:-1], 4), dtype=model_input.dtype)

    class Model:
        def __init__(self):
            self.backbone = Backbone()
            self.residual_scale = torch.ones((1, 1, 4), dtype=torch.float32)

        def normalized_input(self, current, *, nodes, **kwargs):
            captured["current"] = current.detach().clone()
            captured["input_nodes"] = nodes.detach().clone()
            return current

    nodes = torch.tensor([[[2.0, -1.0], [1.5, 0.25]]], dtype=torch.float32)
    sample = {
        "node_mask": torch.ones((1, 2, 1)),
        "nodes": nodes,
        "node_weights": torch.full((1, 2, 1), 0.5),
        "node_rhos": torch.full((1, 2, 1), 0.5),
        "directed_edges": torch.tensor([[[0, 1], [1, 0]]]),
        "edge_gradient_weights": torch.zeros((1, 2, 2)),
        "node_type": torch.tensor([[3, 2]]),
        "mach": torch.tensor([2.0]),
    }
    tensors = tuple(torch.zeros((1, 2, 1, 1)) for _ in range(6))
    geometry = GeometryArm(
        name="rotated",
        sample=sample,
        fourier_tensors=tensors,
        truth=np.zeros((2, 2, 4)),
        weights=np.ones(2),
    )
    raw_rotated_state = np.asarray([[1.0, -3.0, 2.0, 9.0], [0.8, -1.5, 0.5, 4.0]])
    prediction = d083_evaluator._predict_batched(
        Model(),
        geometry,
        (raw_rotated_state,),
        ("correct",),
        device=torch.device("cpu"),
    )["correct"]
    np.testing.assert_allclose(prediction, raw_rotated_state)
    torch.testing.assert_close(
        captured["current"][0], torch.tensor(raw_rotated_state, dtype=torch.float32)
    )
    torch.testing.assert_close(captured["input_nodes"], nodes)
    torch.testing.assert_close(captured["aux_nodes"], nodes)


def test_metric_identity_is_fail_closed_and_region_rows_are_disjoint():
    row = d083_evaluator._metric_row(
        case_id="128",
        phase="g1b",
        arm="native_correct",
        mode="teacher_forced",
        call=1,
        physical_time=0.025,
        metric="predicted_increment_relative_error",
        numerator=1.0,
        denominator=2.0,
        frame="native",
    )
    with pytest.raises(ValueError, match="duplicate D085 metric identity"):
        _assert_unique_metric_rows([row, {**row, "physical_time": 999.0}])

    case = type(
        "Case",
        (),
        {
            "case_id": "128",
            "node_weights": np.ones((2, 1)),
            "residual_scale": np.ones(4),
            "physical_times": np.asarray([0.0, 0.025]),
        },
    )()
    rows = []
    _record_native_residual_regions(
        rows,
        case,
        np.ones((2, 4)),
        np.zeros((2, 4)),
        {"all": np.ones(2, dtype=bool), "boundary": np.asarray([True, False])},
        phase="g1b",
        arm="native_correct",
        mode="teacher_forced",
        call=1,
        include_all_region=False,
    )
    assert [value["region"] for value in rows] == ["boundary"]


def test_visualizer_requires_terminal_receipt_binding(tmp_path):
    payload_path = tmp_path / "metrics.csv"
    payload_path.write_text("metric,value\nstate,0.1\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "pcno_declared_artifact_manifest_v1",
                "files": {"metrics.csv": digest(payload_path)},
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema": INPUT_SCHEMA,
        "status": "complete",
        "scientific_interpretation_allowed": True,
        "declared_matrix_attempt_complete": True,
        "declared_matrix_horizon_complete": True,
        "artifact_manifest_sha256": digest(manifest_path),
        "checkpoint_contract": {
            "fourier_policy": INPUT_CONTRACTS[INPUT_SCHEMA]["fourier_policy"]
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt_path = tmp_path / "terminal_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": INPUT_CONTRACTS[INPUT_SCHEMA]["receipt_schema"],
                "summary_sha256": digest(summary_path),
                "artifact_manifest_sha256": digest(manifest_path),
                "execution_status": "complete",
                "declared_matrix_attempt_complete": True,
                "declared_matrix_horizon_complete": True,
            }
        ),
        encoding="utf-8",
    )

    assert _verify_input(tmp_path) == summary
    summary_path.write_text(
        json.dumps({**summary, "case_count": 999}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not bind summary"):
        _verify_input(tmp_path)


def test_legacy_metric_reader_deduplicates_exact_rows_and_rejects_conflicts(tmp_path):
    fieldnames = [
        "case_id",
        "phase",
        "arm",
        "mode",
        "call",
        "physical_time",
        "metric",
        "frame",
        "region",
        "component",
        "numerator",
        "denominator",
        "value",
        "raw_physical_numerator",
        "raw_physical_denominator",
    ]
    row = {
        "case_id": "128",
        "phase": "g1b",
        "arm": "native_correct",
        "mode": "teacher_forced",
        "call": "1",
        "physical_time": "0.025",
        "metric": "predicted_increment_relative_error",
        "frame": "native",
        "region": "all",
        "component": "all",
        "numerator": "1.0",
        "denominator": "2.0",
        "value": "0.5",
        "raw_physical_numerator": "3.0",
        "raw_physical_denominator": "4.0",
    }
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row, row])
    rows, audit = _read_metrics(path)
    assert len(rows) == 1
    assert audit["exact_duplicate_rows_removed"] == 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row, {**row, "physical_time": "0.050"}])
    with pytest.raises(ValueError, match="conflicting duplicate metric identity"):
        _read_metrics(path)


def test_completion_prefix_excludes_first_inadmissible_proposal(tmp_path):
    path = tmp_path / "completion.csv"
    fieldnames = ("case_id", "geometry", "type_arm", "call", "admissible")
    rows = [
        {
            "case_id": "128",
            "geometry": "native",
            "type_arm": "correct",
            "call": 1,
            "admissible": True,
        },
        {
            "case_id": "128",
            "geometry": "native",
            "type_arm": "correct",
            "call": 2,
            "admissible": True,
        },
        {
            "case_id": "128",
            "geometry": "native",
            "type_arm": "correct",
            "call": 3,
            "admissible": False,
        },
        {
            "case_id": "128",
            "geometry": "rotated",
            "type_arm": "correct",
            "call": 1,
            "admissible": True,
        },
        {
            "case_id": "128",
            "geometry": "rotated",
            "type_arm": "correct",
            "call": 2,
            "admissible": False,
        },
        {
            "case_id": "172",
            "geometry": "native",
            "type_arm": "correct",
            "call": 1,
            "admissible": True,
        },
        {
            "case_id": "172",
            "geometry": "rotated",
            "type_arm": "correct",
            "call": 1,
            "admissible": False,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    prefixes = _read_admissible_prefixes(path)
    assert (
        paired_admissible_prefix(
            prefixes, case_id="128", phase="g1b", type_arm="correct"
        )["accepted_calls"]
        == 1
    )
    assert (
        paired_admissible_prefix(
            prefixes, case_id="172", phase="g1b", type_arm="correct"
        )["accepted_calls"]
        == 0
    )

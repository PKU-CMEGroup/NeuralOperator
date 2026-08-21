from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as runner,
)
from utility.time_dependent_no import (
    pcno_cross_resolution_teacher_forced as teacher,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    grouped_crossfit,
)
from utility.time_dependent_no.pcno_defect_corrections import (
    weighted_subspace_decomposition,
)

RESOLUTION = (8, 4)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = RESOLUTION
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    return nodes, volumes, node_type


def _masks(nodes: int) -> dict[str, np.ndarray]:
    boundary = np.zeros(nodes, dtype=bool)
    shock = np.zeros(nodes, dtype=bool)
    vortex = np.zeros(nodes, dtype=bool)
    smooth = np.zeros(nodes, dtype=bool)
    boundary[:8] = True
    shock[8:16] = True
    vortex[16:24] = True
    smooth[24:] = True
    return {
        "boundary_le_0.05": boundary,
        "partition_boundary": boundary,
        "partition_shock": shock,
        "partition_vortex": vortex,
        "partition_smooth": smooth,
    }


def _snapshot(
    case_id: str,
    input_call: int,
    *,
    seed: int,
    alpha: float = 0.3,
    beta: float = -0.2,
) -> DiagnosticSnapshot:
    rng = np.random.default_rng(seed)
    nodes = RESOLUTION[0] * RESOLUTION[1]
    coarse = rng.normal(size=(nodes, 4))
    fine = rng.normal(size=(nodes, 4))
    target = alpha * coarse + beta * fine
    zero = np.zeros_like(target)
    return DiagnosticSnapshot(
        case_id=case_id,
        group_id=case_id.split("_")[1],
        input_call=input_call,
        basis=NativeIncrementBasis(
            native_increment=zero,
            coarse_on_native=zero,
            fine_on_native=zero,
            native_minus_coarse=coarse,
            fine_minus_native=fine,
        ),
        target_correction=target,
        volumes=np.full(nodes, 2.0 / nodes),
        component_scale=np.asarray((0.5, 0.75, 1.0, 1.25)),
        masks=_masks(nodes),
    )


def _calibration_snapshots() -> list[DiagnosticSnapshot]:
    snapshots = []
    seed = 0
    for case_id in teacher.CALIBRATION_CASE_IDS:
        for input_call in teacher.FIT_INPUT_CALLS:
            snapshots.append(_snapshot(case_id, input_call, seed=seed))
            seed += 1
    return snapshots


def _closure() -> teacher.CalibrationClosureEvidence:
    return teacher.CalibrationClosureEvidence(
        checkpoint_contract=True,
        reference_contract=True,
        common_source_inventory=True,
        prediction_inventory=True,
        exact_source_identity=True,
        maximum_pre_model_nesting_floor=0.0,
        maximum_post_fp32_nesting_floor=1.0e-7,
        maximum_sign_closure=1.0e-14,
        maximum_increment_integral_closure=1.0e-14,
        maximum_transfer_floor_closure=1.0e-14,
        maximum_band_closure=1.0e-12,
        maximum_region_partition_error=0,
        maximum_repeat_abs_difference=0.0,
    )


def test_registered_population_and_time_cells_are_exact_and_disjoint() -> None:
    assert len(teacher.CALIBRATION_CASE_IDS) == 18
    assert len(teacher.EVALUATION_CASE_IDS) == 6
    assert not set(teacher.CALIBRATION_CASE_IDS) & set(teacher.EVALUATION_CASE_IDS)
    assert teacher.FIT_INPUT_CALLS == tuple(range(20))
    assert teacher.HELD_OUT_INPUT_CALLS == tuple(range(20, 30))
    assert not set(teacher.FIT_INPUT_CALLS) & set(teacher.HELD_OUT_INPUT_CALLS)
    assert teacher.CELL_CASES["joint_held_out"] == teacher.EVALUATION_CASE_IDS
    assert teacher.CELL_CALLS["joint_late_1"] == tuple(range(20, 25))
    assert teacher.CELL_CALLS["joint_late_2"] == tuple(range(25, 30))


def test_cached_rank8_projector_matches_inherited_weighted_qr() -> None:
    nodes, volumes, node_type = _grid()
    node_type[:4] = 1
    projector = teacher.build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
    )
    rng = np.random.default_rng(12)
    field = rng.normal(size=(nodes.shape[0], 4))
    split, closure = projector.split(field)
    inherited = weighted_subspace_decomposition(
        field,
        projector.basis,
        volumes,
        node_type == 0,
        component_scale=np.asarray((0.5, 0.75, 1.0, 1.25)),
    )
    assert projector.modes[:3] == ((0, 0), (1, 0), (0, 1))
    assert np.allclose(split["parallel"], inherited.parallel, atol=1.0e-12)
    assert np.allclose(split["orthogonal"], inherited.orthogonal, atol=1.0e-12)
    assert np.array_equal(split["excluded"], inherited.excluded_non_type0)
    assert closure["maximum_reconstruction_abs"] <= 1.0e-15


def test_calibration_qualification_recovers_stable_two_term_candidate() -> None:
    crossfit = grouped_crossfit(
        _calibration_snapshots(),
        resolution=RESOLUTION,
        expected_groups=teacher.CALIBRATION_GROUPS,
        expected_input_calls=teacher.FIT_INPUT_CALLS,
    )
    qualification = teacher.qualify_calibration(crossfit, _closure())
    assert crossfit["selected_model"] == "two_term"
    assert crossfit["selected_coefficients"] == pytest.approx((0.3, -0.2))
    assert qualification["status"] == "qualified"
    assert all(qualification["checks"].values())
    assert [row["same_sign_fold_count"] for row in qualification["coefficient_stability"]] == [
        9,
        9,
    ]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda crossfit, closure: (
                {**crossfit, "expected_input_calls": list(range(1, 21))},
                closure,
            ),
            "call_inventory_exact",
        ),
        (
            lambda crossfit, closure: (
                crossfit,
                replace(closure, maximum_transfer_floor_closure=2.0e-12),
            ),
            "transfer_floor_closure",
        ),
        (
            lambda crossfit, closure: (
                {**crossfit, "selected_model": "zero", "selected_coefficients": (0.0, 0.0)},
                closure,
            ),
            "selected_nonzero_model",
        ),
    ),
)
def test_calibration_qualification_fails_closed(
    mutation,
    failed_check: str,
) -> None:
    crossfit = grouped_crossfit(
        _calibration_snapshots(),
        resolution=RESOLUTION,
        expected_groups=teacher.CALIBRATION_GROUPS,
        expected_input_calls=teacher.FIT_INPUT_CALLS,
    )
    changed_crossfit, changed_closure = mutation(crossfit, _closure())
    qualification = teacher.qualify_calibration(changed_crossfit, changed_closure)
    assert qualification["status"] == "not_qualified"
    assert qualification["checks"][failed_check] is False


def test_payload_hash_and_qualified_loader_reject_tampering(tmp_path: Path) -> None:
    source_hash = "a" * 64
    payload = teacher.with_payload_sha256(
        {
            "schema": "pcno_cross_resolution_teacher_forced_calibration_v1",
            "source_manifest_sha256": source_hash,
            "qualification": {
                "status": "qualified",
                "checks": {"all": True},
                "selected_model": "two_term",
                "selected_coefficients": [0.3, -0.2],
            },
        }
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = teacher.require_qualified_calibration(
        path,
        expected_source_manifest_sha256=source_hash,
    )
    assert loaded["payload_sha256"] == payload["payload_sha256"]

    tampered = dict(payload)
    tampered["qualification"] = dict(payload["qualification"])
    tampered["qualification"]["selected_coefficients"] = [9.0, -0.2]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        teacher.require_qualified_calibration(
            path,
            expected_source_manifest_sha256=source_hash,
        )


def test_score_cell_reports_case_first_views_relations_and_oracle() -> None:
    nodes, volumes, node_type = _grid()
    projector = teacher.build_fixed_cosine_projector(nodes, volumes, node_type, rank=8)
    snapshots = []
    seed = 50
    for case_id in teacher.EVALUATION_CASE_IDS:
        for input_call in teacher.HELD_OUT_INPUT_CALLS:
            snapshots.append(_snapshot(case_id, input_call, seed=seed))
            seed += 1
    result = teacher.score_cell(
        snapshots,
        cell="joint_held_out",
        coefficients=(0.3, -0.2),
        resolution=RESOLUTION,
        projector=projector,
    )
    primary = [
        row
        for row in result["rows"]
        if row["scope"] == "population" and row["view"] == "band_large"
    ]
    assert len(primary) == 1
    assert primary[0]["skill_vs_zero"] == pytest.approx(1.0)
    assert primary[0]["rms_ratio_vs_zero"] == pytest.approx(0.0, abs=1.0e-7)
    assert primary[0]["median_case_cosine"] == pytest.approx(1.0)
    assert len(result["oracle_rows"]) == 2 * len(teacher.EVALUATION_CASE_IDS)
    assert result["maximum_closure"]["band"] <= 1.0e-12
    assert result["maximum_closure"]["subspace_reconstruction"] <= 1.0e-12

    relations = teacher.error_relation_rows(
        snapshots,
        cell="joint_held_out",
        resolution=RESOLUTION,
        projector=projector,
    )
    expected = (
        len(teacher.VIEW_SPECS)
        * len(teacher.RELATION_PAIRS)
        * (1 + len(teacher.EVALUATION_CASE_IDS))
    )
    assert len(relations) == expected
    assert all(row["offline_only"] is True for row in relations)


def _cell_payload(cell: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for view in {"band_large", *teacher.REQUIRED_FIELD_CONTROL_VIEWS}:
        rows.append(
            {
                "cell": cell,
                "view": view,
                "scope": "population",
                "case_id": None,
                "skill_vs_zero": 0.2,
                "rms_ratio_vs_zero": 0.9,
                "centered_r2": 0.1,
                "median_case_cosine": 0.5,
                "median_case_correlation": 0.4,
                "status": "ok",
                "all_case_scores_resolved": True,
                "target_rms": 1.0,
                "skill_status": "ok",
            }
        )
        for case_id in teacher.CELL_CASES[cell]:
            rows.append(
                {
                    "cell": cell,
                    "view": view,
                    "scope": "case",
                    "case_id": case_id,
                    "skill_vs_zero": 0.2,
                    "rms_ratio_vs_zero": 0.9,
                    "cosine": 0.5,
                    "correlation": 0.4,
                    "cosine_status": "ok",
                    "correlation_status": "ok",
                    "skill_status": "ok",
                    "target_rms": 1.0,
                }
            )
    return {
        "cell": cell,
        "case_ids": sorted(teacher.CELL_CASES[cell]),
        "input_calls": list(teacher.CELL_CALLS[cell]),
        "rows": rows,
    }


def _passing_gate_inputs():
    cells = {
        cell: _cell_payload(cell)
        for cell in (
            "time_only",
            "case_only",
            "joint_held_out",
            "joint_late_1",
            "joint_late_2",
        )
    }
    keys = {
        *(f"field::{view}" for view in teacher.REQUIRED_FIELD_CONTROL_VIEWS),
        *teacher.FRONT_CONTROL_KEYS,
        *(f"integral_rms::{name}" for name in teacher.INTEGRAL_COMPONENT_NAMES),
        *(f"integral_endpoint::{name}" for name in teacher.INTEGRAL_COMPONENT_NAMES),
    }
    controls = [
        teacher.ControlRatio(
            key=key,
            scope=scope,
            zero_rms=1.0,
            corrected_rms=1.0,
            ratio=1.0,
            status="ok",
        )
        for key in sorted(keys)
        for scope in ("population", *teacher.EVALUATION_CASE_IDS)
    ]
    proposals = [
        teacher.ProposalEvidence(case_id, input_call, True, True)
        for case_id in teacher.EVALUATION_CASE_IDS
        for input_call in teacher.HELD_OUT_INPUT_CALLS
    ]
    return cells, controls, proposals


def test_prospective_gate_passes_exact_inventory_and_fails_missing_control() -> None:
    cells, controls, proposals = _passing_gate_inputs()
    passed = teacher.prospective_gate(
        cell_payloads=cells,
        controls=controls,
        proposals=proposals,
    )
    assert passed["status"] == "passed"
    assert passed["p2_authorized"] is False
    assert all(passed["checks"].values())

    failed = teacher.prospective_gate(
        cell_payloads=cells,
        controls=controls[:-1],
        proposals=proposals,
    )
    assert failed["status"] == "failed"
    assert failed["checks"]["control_inventory_exact"] is False


def test_denominator_floor_and_ratio_threshold_fail_closed() -> None:
    cells, controls, proposals = _passing_gate_inputs()
    controls[0] = replace(controls[0], zero_rms=1.0e-8)
    result = teacher.prospective_gate(
        cell_payloads=cells,
        controls=controls,
        proposals=proposals,
    )
    assert result["status"] == "failed"
    assert result["checks"]["all_controls_no_harm"] is False

    cells, controls, proposals = _passing_gate_inputs()
    controls[0] = replace(controls[0], corrected_rms=1.0500001, ratio=1.0500001)
    result = teacher.prospective_gate(
        cell_payloads=cells,
        controls=controls,
        proposals=proposals,
    )
    assert result["status"] == "failed"


def test_evaluation_firewall_checks_qualification_before_runtime_or_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        runner,
        "_verify_frozen_source_manifest",
        lambda *_args, **_kwargs: {"schema": runner.SOURCE_MANIFEST_SCHEMA},
    )

    def reject(*_args, **_kwargs):
        calls.append("qualification")
        raise ValueError("calibration did not qualify; evaluation targets stay closed")

    def forbidden(*_args, **_kwargs):
        calls.append("runtime")
        raise AssertionError("runtime/reference loader must remain closed")

    monkeypatch.setattr(runner, "require_qualified_calibration", reject)
    monkeypatch.setattr(runner, "_build_runtime", forbidden)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a" * 64)
    args = SimpleNamespace(
        source_manifest=Path("source.json"),
        calibration=Path("calibration.json"),
    )
    with pytest.raises(ValueError, match="targets stay closed"):
        runner.run_evaluation(args)
    assert calls == ["qualification"]


def test_rms_control_uses_signed_errors_but_scores_magnitudes() -> None:
    rows = [
        {"zero_error": -2.0, "corrected_error": 1.0},
        {"zero_error": 2.0, "corrected_error": -1.0},
    ]
    control = runner._rms_control(key="integral", scope="case", rows=rows)
    assert control.zero_rms == pytest.approx(2.0)
    assert control.corrected_rms == pytest.approx(1.0)
    assert control.ratio == pytest.approx(0.5)
    assert control.status == "ok"

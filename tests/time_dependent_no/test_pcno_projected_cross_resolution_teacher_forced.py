from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base_runner,
)
from scripts.time_dependent_no import (
    evaluate_pcno_projected_cross_resolution_teacher_forced as runner,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    CELL_CALLS,
    EVALUATION_CASE_IDS,
    FIT_INPUT_CALLS,
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    REQUIRED_FIELD_CONTROL_VIEWS,
    ControlRatio,
    ProposalEvidence,
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_correction import (
    ProjectionClosure,
    project_parallel_field,
    projected_grouped_crossfit,
    score_projected_cell,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_teacher_forced import (
    ProjectedCalibrationClosureEvidence,
    adaptive_projected_gate,
    all_open_strength_groups,
    projected_grouped_crossfit_from_records,
    projected_statistic_record_from_dict,
    projected_statistic_record_to_dict,
    projected_statistic_records,
    qualify_projected_calibration,
)

RESOLUTION = (8, 4)
ALPHA = 0.35
BETA = -0.2


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = RESOLUTION
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    return nodes, volumes, node_type


def _projector():
    nodes, volumes, node_type = _grid()
    return build_fixed_cosine_projector(nodes, volumes, node_type, rank=8)


def _masks(node_count: int) -> dict[str, np.ndarray]:
    boundary = np.zeros(node_count, dtype=bool)
    shock = np.zeros(node_count, dtype=bool)
    vortex = np.zeros(node_count, dtype=bool)
    smooth = np.zeros(node_count, dtype=bool)
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
) -> DiagnosticSnapshot:
    rng = np.random.default_rng(seed)
    nodes, volumes, _ = _grid()
    projector = _projector()
    coarse = rng.normal(size=(nodes.shape[0], 4))
    fine = rng.normal(size=(nodes.shape[0], 4))
    projected_coarse, _ = project_parallel_field(coarse, projector)
    projected_fine, _ = project_parallel_field(fine, projector)
    noise, _ = projector.split(0.05 * rng.normal(size=(nodes.shape[0], 4)))
    target = (
        ALPHA * projected_coarse
        + BETA * projected_fine
        + noise["orthogonal"]
        + noise["excluded"]
    )
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
        volumes=volumes,
        component_scale=np.asarray((0.5, 0.75, 1.0, 1.25)),
        masks=_masks(nodes.shape[0]),
    )


def _snapshot_grid(
    case_ids: tuple[str, ...],
    input_calls: tuple[int, ...],
    *,
    first_seed: int = 0,
) -> list[DiagnosticSnapshot]:
    rows = []
    seed = first_seed
    for case_id in case_ids:
        for input_call in input_calls:
            rows.append(_snapshot(case_id, input_call, seed=seed))
            seed += 1
    return rows


def test_statistic_round_trip_matches_field_crossfit() -> None:
    groups = {
        f"e{group:02d}": (f"sv_e{group:02d}_y00", f"sv_e{group:02d}_y08")
        for group in range(4)
    }
    cases = tuple(case_id for members in groups.values() for case_id in members)
    calls = (0, 1, 2)
    snapshots = _snapshot_grid(cases, calls)
    projector = _projector()
    field_result = projected_grouped_crossfit(
        snapshots,
        resolution=RESOLUTION,
        projector=projector,
        expected_groups=groups,
        expected_input_calls=calls,
    )
    records = projected_statistic_records(
        snapshots,
        resolution=RESOLUTION,
        projector=projector,
    )
    serialized = json.loads(
        json.dumps([projected_statistic_record_to_dict(record) for record in records])
    )
    restored = tuple(projected_statistic_record_from_dict(row) for row in serialized)
    record_result = projected_grouped_crossfit_from_records(
        restored,
        projector=projector,
        expected_groups=groups,
        expected_input_calls=calls,
    )

    assert record_result["selected_model"] == field_result["selected_model"]
    assert record_result["selected_coefficients"] == pytest.approx(
        field_result["selected_coefficients"]
    )
    assert record_result["models"] == field_result["models"]
    assert record_result["projection_closure"] == field_result["projection_closure"]
    assert record_result["selected_coefficients"] == pytest.approx((ALPHA, BETA))


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "wrong_group", "bool_call", "mixed_metric", "extra_field"),
)
def test_statistic_records_fail_closed(mutation: str) -> None:
    groups = {
        "e00": ("sv_e00_y00", "sv_e00_y08"),
        "e01": ("sv_e01_y00", "sv_e01_y08"),
    }
    cases = tuple(case_id for members in groups.values() for case_id in members)
    snapshots = _snapshot_grid(cases, (0, 1))
    projector = _projector()
    records = list(
        projected_statistic_records(
            snapshots,
            resolution=RESOLUTION,
            projector=projector,
        )
    )
    if mutation == "duplicate":
        records[-1] = records[0]
    elif mutation == "wrong_group":
        records[0] = replace(records[0], group_id="e01")
    elif mutation == "bool_call":
        records[0] = replace(records[0], input_call=True)
    elif mutation == "mixed_metric":
        records[0] = replace(records[0], metric_contract_sha256="0" * 64)
    else:
        payload = projected_statistic_record_to_dict(records[0])
        payload["unexpected"] = True
        with pytest.raises(ValueError, match="schema"):
            projected_statistic_record_from_dict(payload)
        return

    with pytest.raises(ValueError):
        projected_grouped_crossfit_from_records(
            records,
            projector=projector,
            expected_groups=groups,
            expected_input_calls=(0, 1),
        )


def _calibration_closure() -> ProjectedCalibrationClosureEvidence:
    return ProjectedCalibrationClosureEvidence(
        checkpoint_contract=True,
        reference_contract=True,
        common_source_inventory=True,
        prediction_inventory=True,
        exact_source_identity=True,
        maximum_pre_model_nesting_floor=0.0,
        maximum_post_fp32_nesting_floor=1.0e-7,
        maximum_sign_closure=1.0e-15,
        maximum_increment_integral_closure=1.0e-15,
        maximum_transfer_floor_closure=1.0e-15,
        maximum_band_closure=1.0e-15,
        maximum_region_partition_error=0,
        maximum_repeat_abs_difference=0.0,
    )


def test_projected_calibration_qualification_is_exact_and_fail_closed() -> None:
    snapshots = _snapshot_grid(CALIBRATION_CASE_IDS, FIT_INPUT_CALLS)
    projector = _projector()
    records = projected_statistic_records(
        snapshots,
        resolution=RESOLUTION,
        projector=projector,
    )
    crossfit = projected_grouped_crossfit_from_records(
        records,
        projector=projector,
        expected_groups=CALIBRATION_GROUPS,
        expected_input_calls=FIT_INPUT_CALLS,
    )
    qualification = qualify_projected_calibration(crossfit, _calibration_closure())

    assert qualification["status"] == "qualified"
    assert all(qualification["checks"].values())
    assert qualification["selected_coefficients"] == pytest.approx((ALPHA, BETA))

    failed = qualify_projected_calibration(
        crossfit,
        replace(_calibration_closure(), maximum_sign_closure=1.0e-6),
    )
    assert failed["status"] == "not_qualified"
    assert failed["checks"]["mapped_sign_closure"] is False


def _passing_controls() -> list[ControlRatio]:
    keys = {
        *(f"field::{view}" for view in REQUIRED_FIELD_CONTROL_VIEWS),
        *FRONT_CONTROL_KEYS,
        *(f"integral_rms::{name}" for name in INTEGRAL_COMPONENT_NAMES),
        *(f"integral_endpoint::{name}" for name in INTEGRAL_COMPONENT_NAMES),
    }
    return [
        ControlRatio(
            key=key,
            scope=scope,
            zero_rms=1.0,
            corrected_rms=0.9,
            ratio=0.9,
            status="ok",
        )
        for key in sorted(keys)
        for scope in ("population", *EVALUATION_CASE_IDS)
    ]


def test_adaptive_gate_requires_full_field_no_harm_and_all_controls() -> None:
    all_cases = (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
    all_snapshots = _snapshot_grid(all_cases, ALL_INPUT_CALLS)
    projector = _projector()
    current_snapshots = [
        row
        for row in all_snapshots
        if (
            row.case_id in CALIBRATION_CASE_IDS
            and row.input_call in CELL_CALLS["time_only"]
        )
        or row.case_id in EVALUATION_CASE_IDS
    ]
    cell_payloads = {
        cell: score_projected_cell(
            base_runner._select_cell(current_snapshots, cell),
            cell=cell,
            coefficients=(ALPHA, BETA),
            resolution=RESOLUTION,
            projector=projector,
        )
        for cell in (
            "time_only",
            "case_only",
            "joint_held_out",
            "joint_late_1",
            "joint_late_2",
        )
    }
    records = projected_statistic_records(
        all_snapshots,
        resolution=RESOLUTION,
        projector=projector,
    )
    all_open = projected_grouped_crossfit_from_records(
        records,
        projector=projector,
        expected_groups=all_open_strength_groups(),
        expected_input_calls=ALL_INPUT_CALLS,
    )
    proposals = [
        ProposalEvidence(case_id, input_call, True, True)
        for case_id in EVALUATION_CASE_IDS
        for input_call in CELL_CALLS["joint_held_out"]
    ]
    controls = _passing_controls()
    gate = adaptive_projected_gate(
        cell_payloads=cell_payloads,
        controls=controls,
        proposals=proposals,
        all_open_crossfit=all_open,
        calibration_qualification={"status": "qualified", "checks": {"ok": True}},
        control_reconstruction_closure={
            "target_reconstruction": 0.0,
            "excluded_correction": 0.0,
            "model_calls": 0,
        },
    )

    assert gate["status"] == "continuation_request_supported"
    assert all(gate["checks"].values())
    assert gate["fresh_confirmation"] is False
    assert gate["recurrence_authorized"] is False

    harmed_cells = copy.deepcopy(cell_payloads)
    full_row = next(
        row
        for row in harmed_cells["joint_late_2"]["rows"]
        if row["scope"] == "population" and row["view"] == "full"
    )
    full_row["skill_vs_zero"] = -1.0e-12
    stopped = adaptive_projected_gate(
        cell_payloads=harmed_cells,
        controls=controls,
        proposals=proposals,
        all_open_crossfit=all_open,
        calibration_qualification={"status": "qualified", "checks": {"ok": True}},
        control_reconstruction_closure={
            "target_reconstruction": 0.0,
            "excluded_correction": 0.0,
            "model_calls": 0,
        },
    )
    assert stopped["status"] == "stopped"
    assert stopped["checks"]["all_aggregate_full_field_skills_nonnegative"] is False

    harmed_controls = [*controls]
    harmed_controls[0] = replace(harmed_controls[0], corrected_rms=1.06, ratio=1.06)
    stopped = adaptive_projected_gate(
        cell_payloads=cell_payloads,
        controls=harmed_controls,
        proposals=proposals,
        all_open_crossfit=all_open,
        calibration_qualification={"status": "qualified", "checks": {"ok": True}},
        control_reconstruction_closure={
            "target_reconstruction": 0.0,
            "excluded_correction": 0.0,
            "model_calls": 0,
        },
    )
    assert stopped["checks"]["all_controls_no_harm"] is False

    stopped = adaptive_projected_gate(
        cell_payloads=cell_payloads,
        controls=controls,
        proposals=proposals,
        all_open_crossfit=all_open,
        calibration_qualification={"status": "qualified", "checks": {"ok": True}},
        control_reconstruction_closure={
            "target_reconstruction": 1.0e-6,
            "excluded_correction": 0.0,
            "model_calls": 0,
        },
    )
    assert stopped["checks"]["control_reconstruction_closure"] is False


def test_evaluation_firewall_rejects_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_called = False

    def fail_calibration(*args, **kwargs):
        raise ValueError("calibration did not qualify")

    def mark_runtime(*args, **kwargs):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must remain closed")

    monkeypatch.setattr(runner, "_verify_frozen_source_manifest", lambda *args: {})
    monkeypatch.setattr(runner, "_require_qualified_calibration", fail_calibration)
    monkeypatch.setattr(runner, "_build_runtime", mark_runtime)
    args = argparse.Namespace(
        source_manifest=Path("source_manifest.json"),
        calibration=Path("calibration.json"),
        calibration_statistics=Path("statistics.json"),
    )
    with pytest.raises(ValueError, match="did not qualify"):
        runner.run_evaluation(args)
    assert runtime_called is False


def test_smoke_status_treats_closed_evaluation_targets_as_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot("sv_e01_y00", 0, seed=99)
    runtime = SimpleNamespace(native_projector=_projector())
    collected = base_runner.CollectionResult(
        snapshots=[snapshot],
        floor_rows=[],
        front_rows=[],
        integral_rows=[],
        proposal_rows=[],
        reference_rows=[],
        execution={"logical_model_calls": 3},
        maxima={},
    )
    monkeypatch.setattr(
        runner,
        "_build_runtime",
        lambda args: (runtime, {"payload_sha256": "source"}),
    )
    monkeypatch.setattr(base_runner, "_collect", lambda *args, **kwargs: collected)
    monkeypatch.setattr(base_runner, "_close_runtime", lambda active: None)
    monkeypatch.setattr(
        runner,
        "projected_statistic_records",
        lambda *args, **kwargs: (
            SimpleNamespace(projection_closure=ProjectionClosure(0.0, 0.0, 0.0, 0.0)),
        ),
    )
    output = tmp_path / "smoke.json"

    payload = runner.run_smoke(argparse.Namespace(output=output))

    assert payload["status"] == "passed"
    assert payload["checks"]["evaluation_targets_closed"] is True
    assert all(payload["checks"].values())


def test_all_open_groups_are_exact_strength_pairs() -> None:
    groups = all_open_strength_groups()
    assert len(groups) == 12
    assert {case_id for members in groups.values() for case_id in members} == {
        *CALIBRATION_CASE_IDS,
        *EVALUATION_CASE_IDS,
    }
    assert all(len(members) == 2 for members in groups.values())

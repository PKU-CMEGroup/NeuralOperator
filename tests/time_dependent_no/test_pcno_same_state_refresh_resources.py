from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import build_pcno_same_state_refresh_resources as cli
from scripts.time_dependent_no.analyze_pcno_binary_position_phase_rescore import (
    selector_active as a31_selector_active,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    parse_args as parse_training_args,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    transverse_velocity_position_descriptor,
)
from utility.time_dependent_no.pcno_same_state_refresh_resources import (
    BRANCH_STRENGTH_INDICES,
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_FINE_RESOLUTION,
    EXPECTED_NATIVE_RESOLUTION,
    MIDPOINT_STRENGTHS,
    RECURRENCE_STRENGTH_INDICES,
    RESOURCE_PLAN_SCHEMA,
    TRAINING_SEED,
    build_population_manifest,
    build_resource_plan,
    derive_nested_reference,
    frozen_selector_active,
    matched_training_arguments,
    resource_position_descriptor,
    validate_resource_plan,
)
from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    make_structured_fv_geometry,
    reference_contract_checks,
    run_shock_vortex_reference,
)

ROOT = Path(__file__).resolve().parents[2]


def test_resource_plan_freezes_new_grouped_population() -> None:
    plan = build_resource_plan(ROOT)
    assert plan["schema"] == RESOURCE_PLAN_SCHEMA
    assert validate_resource_plan(plan, ROOT) == plan
    assert len(plan["cases"]) == 18
    assert len(MIDPOINT_STRENGTHS) == 6
    assert BRANCH_STRENGTH_INDICES.isdisjoint(RECURRENCE_STRENGTH_INDICES)
    assert BRANCH_STRENGTH_INDICES | RECURRENCE_STRENGTH_INDICES == set(range(6))

    case_ids = [row["case_id"] for row in plan["cases"]]
    assert len(case_ids) == len(set(case_ids))
    assert all(case_id.startswith("sv_a46_") for case_id in case_ids)
    for group in {row["strength_group"] for row in plan["cases"]}:
        rows = [row for row in plan["cases"] if row["strength_group"] == group]
        assert len(rows) == 3
        assert sum(row["expected_selector_active"] for row in rows) == 2
        assert sum(row["inactive_control_retained"] for row in rows) == 1
    assert len(plan["allocation"]["branch_fit_case_ids"]) == 8
    assert len(plan["allocation"]["recurrence_case_ids"]) == 4
    assert plan["training_contract"]["fresh_seed"] == TRAINING_SEED
    assert (
        plan["training_contract"]["data_manifest_digest"]
        == EXPECTED_DATA_MANIFEST_DIGEST
    )


def test_resource_plan_rejects_tampering() -> None:
    plan = build_resource_plan(ROOT)
    tampered = json.loads(json.dumps(plan))
    tampered["cases"][0]["population"] = "recurrence"
    with pytest.raises(ValueError, match="payload digest"):
        validate_resource_plan(tampered, ROOT)


def test_matched_training_arguments_parse_to_exact_interface(tmp_path: Path) -> None:
    arguments = matched_training_arguments(tmp_path / "data", tmp_path / "run")
    parsed = parse_training_args(arguments)
    assert parsed.seed == TRAINING_SEED
    assert parsed.split_seed == 20260718
    assert parsed.split_mode == "manifest"
    assert parsed.step_stride == 2
    assert parsed.model_node_type_input == "physical"
    assert parsed.node_type_channel_control == "standard"
    assert parsed.boundary_field_mode == "none"
    assert parsed.boundary_residual_mode == "none"
    assert parsed.boundary_mode == "model_all_nodes"
    assert parsed.primary_objective == "normal_closed"
    assert parsed.init_checkpoint is None
    assert parsed.resume_checkpoint is None
    assert "--target-kind" not in arguments
    assert "--tiny-fit-face-rel-l2" not in arguments


@pytest.mark.parametrize(
    ("distance", "call"),
    (
        (0.0, 0),
        (0.8125, 7),
        (0.8125, 8),
        (0.7, 21),
        (0.7, 22),
        (np.nextafter(0.8125, 1.0), 29),
    ),
)
def test_resource_selector_is_exact_a31_selector(distance: float, call: int) -> None:
    assert frozen_selector_active(distance, call) is a31_selector_active(distance, call)


def test_resource_position_descriptor_is_exact_a31_descriptor() -> None:
    nodes = np.asarray(
        [[0.25, 0.1], [0.75, 0.4], [1.25, 0.6], [1.75, 0.9]],
        dtype=np.float64,
    )
    volumes = np.asarray([0.2, 0.3, 0.7, 0.8], dtype=np.float64)
    state = np.asarray(
        [
            [1.0, 0.0, -0.2, 3.0],
            [1.2, 0.0, 0.4, 3.0],
            [0.9, 0.0, -0.5, 3.0],
            [1.1, 0.0, 0.1, 3.0],
        ],
        dtype=np.float64,
    )
    expected = transverse_velocity_position_descriptor(
        state, nodes=nodes, volumes=volumes, y_min=0.0, y_max=1.0
    )
    actual = resource_position_descriptor(
        state, nodes=nodes, volumes=volumes, y_min=0.0, y_max=1.0
    )
    assert actual.status == expected.status
    assert actual.vertical_centroid == expected.vertical_centroid
    assert actual.normalized_wall_distance == expected.normalized_wall_distance
    assert actual.transverse_velocity_l1 == expected.transverse_velocity_l1


def test_small_cpu_dual_resolution_closes_conservative_restriction() -> None:
    fine_config = ShockVortexFVConfig(
        nx=12,
        ny=8,
        coarse_nx=6,
        coarse_ny=4,
        t_final=0.001,
        output_times=(0.0, 0.001),
        initial_quadrature_order=2,
    ).validated()
    result = run_shock_vortex_reference(
        fine_config, device=torch.device("cpu"), dtype=torch.float64
    )
    native_config = replace(fine_config, coarse_nx=3, coarse_ny=2).validated()
    native_geometry = make_structured_fv_geometry(native_config)
    derived = derive_nested_reference(
        result.states,
        fine_geometry=result.geometry,
        fine_resolution=(6, 4),
        native_geometry=native_geometry,
        native_resolution=(3, 2),
    )
    assert derived["native_states"].shape == (2, 6, 4)
    assert derived["nested_restriction_max_abs"] == 0.0
    assert derived["integrated_state_max_abs"] <= 1.0e-12
    assert derived["fine_volume_min"] > 0.0
    assert derived["native_volume_min"] > 0.0
    checks = reference_contract_checks(
        result,
        closure_relative_tolerance=1.0e-9,
        initial_quadrature_tolerance=1.0,
    )
    assert all(checks.values())


def _fake_summary(plan: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": row["case_id"],
        "status": "passed",
        "resource_plan_payload_sha256": plan["payload_sha256"],
        "source_sha256": plan["source_sha256"],
        "parameters": row["parameters"],
        "selector_active": row["expected_selector_active"],
        "physical_times_sha256": "d" * 64,
        "configuration_sha256": "b" * 64,
        "summary_sha256": "e" * 64,
        "reference_artifact_sha256": "c" * 64,
        "native_volume_min": 8.0e-5,
        "fine_volume_min": 2.0e-5,
    }


def test_population_manifest_binds_summaries_without_opening_truth() -> None:
    plan = build_resource_plan(ROOT)
    summaries = [_fake_summary(plan, row) for row in plan["cases"]]
    population, report = build_population_manifest(
        plan,
        plan_file_sha256="a" * 64,
        summary_rows=summaries,
        opened_case_ids=("sv_e00_y01", "sv_e14_y08"),
    )
    assert report["valid"] is True
    assert report["case_count"] == 18
    assert len(report["branch_active_case_ids"]) == 8
    assert len(report["recurrence_active_case_ids"]) == 4
    assert population["reference_arrays_opened"] is False
    assert population["truth_arrays_loaded"] is False
    assert population["activity"]["truth_array_loads_for_manifest"] == 0


def test_population_manifest_fails_on_missing_case() -> None:
    plan = build_resource_plan(ROOT)
    summaries = [_fake_summary(plan, row) for row in plan["cases"][:-1]]
    with pytest.raises(ValueError, match="exactly cover"):
        build_population_manifest(
            plan,
            plan_file_sha256="a" * 64,
            summary_rows=summaries,
            opened_case_ids=(),
        )


def test_cli_plan_and_training_command_are_write_once(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    assert cli.main(["plan", "--output-path", str(plan_path)]) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == RESOURCE_PLAN_SCHEMA
    with pytest.raises(FileExistsError):
        cli.main(["plan", "--output-path", str(plan_path)])

    command_path = tmp_path / "training_command.json"
    assert (
        cli.main(
            [
                "training-command",
                "--python",
                "/env/bin/python",
                "--data-dir",
                "/data",
                "--training-output-dir",
                "/run",
                "--output-path",
                str(command_path),
            ]
        )
        == 0
    )
    command = json.loads(command_path.read_text(encoding="utf-8"))
    assert command["argv"][0] == "/env/bin/python"
    assert command["independent_initialization"] is True
    assert command["a46_branch_predictions"] == 0


def test_frozen_production_resolutions_remain_nested() -> None:
    assert EXPECTED_FINE_RESOLUTION == (500, 200)
    assert EXPECTED_NATIVE_RESOLUTION == (250, 100)
    assert EXPECTED_FINE_RESOLUTION[0] % EXPECTED_NATIVE_RESOLUTION[0] == 0
    assert EXPECTED_FINE_RESOLUTION[1] % EXPECTED_NATIVE_RESOLUTION[1] == 0

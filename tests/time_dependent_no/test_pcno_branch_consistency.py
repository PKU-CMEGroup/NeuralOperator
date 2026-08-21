from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from scripts.time_dependent_no import evaluate_pcno_branch_consistency as evaluator
from utility.time_dependent_no.pcno_branch_consistency import (
    BranchConsistencyScore,
    branch_consistency_score,
    score_branch_population,
    select_branch,
    synthetic_summary,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

NATIVE = (8, 4)
SCALE = np.asarray((0.5, 0.75, 1.0, 1.25), dtype=np.float64)


def _grid():
    nx, ny = NATIVE
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.linspace(0.8, 1.2, nx * ny, dtype=np.float64)
    volumes *= 2.0 / volumes.sum()
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    projector = build_fixed_cosine_projector(nodes, volumes, node_type, rank=8)
    return volumes, projector


def _modal_field(projector, values: dict[tuple[int, int], float]) -> np.ndarray:
    coordinates = np.zeros((8, 4), dtype=np.float64)
    for cell, value in values.items():
        coordinates[cell] = value
    return reconstruct_modal_field(
        coordinates,
        projector,
        component_scale=SCALE,
    )


def _score(value: float | None, *, status: str = "ok") -> BranchConsistencyScore:
    return BranchConsistencyScore(
        status=status,
        score=value,
        commutator_rms=0.1,
        native_increment_rms=0.2,
        commutator_modal_energy=0.3,
        native_increment_modal_energy=0.4,
    )


def test_score_uses_exact_sp19_cells_and_is_scale_invariant() -> None:
    volumes, projector = _grid()
    active = FROZEN_ACTIVE_CELLS[0]
    inactive = next(
        cell
        for cell in ((mode, component) for mode in range(8) for component in range(4))
        if cell not in FROZEN_ACTIVE_CELLS
    )
    native = _modal_field(projector, {active: 2.0, inactive: 100.0})
    discrepancy = _modal_field(projector, {active: 0.5, inactive: 1000.0})

    score = branch_consistency_score(
        native,
        native + discrepancy,
        projector,
        volumes=volumes,
        component_scale=SCALE,
    )
    rescaled = branch_consistency_score(
        7.0 * native,
        7.0 * (native + discrepancy),
        projector,
        volumes=volumes,
        component_scale=SCALE,
    )

    assert score.status == "ok"
    assert score.score == pytest.approx(0.25, abs=1.0e-13)
    assert rescaled.score == pytest.approx(score.score, abs=1.0e-13)
    assert score.commutator_modal_energy == pytest.approx(0.25, abs=2.0e-13)
    assert score.native_increment_modal_energy == pytest.approx(4.0, abs=2.0e-13)


def test_score_fails_closed_on_small_denominator_and_bad_inputs() -> None:
    volumes, projector = _grid()
    zero = np.zeros((NATIVE[0] * NATIVE[1], 4), dtype=np.float64)
    active = _modal_field(projector, {FROZEN_ACTIVE_CELLS[0]: 1.0})

    score = branch_consistency_score(
        zero,
        active,
        projector,
        volumes=volumes,
        component_scale=SCALE,
    )
    assert score.status == "small_native_sp19_increment"
    assert score.score is None

    with pytest.raises(ValueError, match="equal shape"):
        branch_consistency_score(
            zero,
            active[:-1],
            projector,
            volumes=volumes,
            component_scale=SCALE,
        )
    with pytest.raises(ValueError, match="denominator_floor"):
        branch_consistency_score(
            zero,
            active,
            projector,
            volumes=volumes,
            component_scale=SCALE,
            denominator_floor=0.0,
        )


def test_selector_resolves_lower_score_and_abstains_to_raw() -> None:
    corrected = select_branch(_score(0.4), _score(0.3))
    raw = select_branch(_score(0.2), _score(0.3))
    tie = select_branch(_score(0.3), _score(0.3 + 5.0e-13))
    unresolved = select_branch(
        _score(None, status="small_native_sp19_increment"), _score(0.3)
    )

    assert corrected.selected_branch == "corrected" and corrected.resolved
    assert raw.selected_branch == "raw" and raw.resolved
    assert tie.selected_branch == "raw" and not tie.resolved
    assert tie.status == "tie_abstain_raw"
    assert unresolved.selected_branch == "raw" and not unresolved.resolved
    with pytest.raises(ValueError, match="tolerances"):
        select_branch(_score(0.2), _score(0.3), absolute_tolerance=-1.0)


def _row(
    case_id: str,
    input_call: int,
    *,
    selected: str,
    score_difference: float,
    raw: float,
    corrected: float,
) -> dict[str, object]:
    selected_error = raw if selected == "raw" else corrected
    return {
        "case_id": case_id,
        "input_call": input_call,
        "selector_resolved": True,
        "selected_branch": selected,
        "score_difference_corrected_minus_raw": score_difference,
        "lookahead_raw_rms": raw,
        "lookahead_corrected_rms": corrected,
        "lookahead_selected_rms": selected_error,
        "lookahead_oracle_rms": min(raw, corrected),
    }


def test_population_score_is_case_first_and_signed_relation_is_not_pooled() -> None:
    rows = [
        _row("a", 0, selected="corrected", score_difference=-1.0, raw=2.0, corrected=1.0),
        _row("a", 2, selected="corrected", score_difference=-2.0, raw=4.0, corrected=2.0),
        _row("b", 0, selected="raw", score_difference=1.0, raw=1.0, corrected=3.0),
    ]
    score = score_branch_population(rows, error_prefix="lookahead")

    raw_case_a_ms = (2.0**2 + 4.0**2) / 2.0
    corrected_case_a_ms = (1.0**2 + 2.0**2) / 2.0
    expected_raw = np.sqrt((raw_case_a_ms + 1.0**2) / 2.0)
    expected_corrected = np.sqrt((corrected_case_a_ms + 3.0**2) / 2.0)
    expected_selected = np.sqrt((corrected_case_a_ms + 1.0**2) / 2.0)
    assert score["population_rms"]["raw"] == pytest.approx(expected_raw)
    assert score["population_rms"]["corrected"] == pytest.approx(expected_corrected)
    assert score["population_rms"]["selected"] == pytest.approx(expected_selected)
    assert score["selected_to_raw_rms_ratio"] == pytest.approx(
        expected_selected / expected_raw
    )
    assert score["selector_accuracy"] == 1.0
    assert score["score_error_relation"]["sign_agreement"] == 1.0
    assert score["score_error_relation"]["cosine"] > 0.0


def test_population_score_rejects_duplicates_and_nonfinite_errors() -> None:
    row = _row(
        "a", 0, selected="raw", score_difference=1.0, raw=1.0, corrected=2.0
    )
    with pytest.raises(ValueError, match="unique"):
        score_branch_population([row, dict(row)], error_prefix="lookahead")
    with pytest.raises(ValueError, match="finite"):
        score_branch_population(
            [{**row, "lookahead_raw_rms": float("nan")}],
            error_prefix="lookahead",
        )


def test_truth_cannot_enter_selector_api_and_synthetic_cli_passes(capsys) -> None:
    parameters = set(inspect.signature(branch_consistency_score).parameters)
    assert "truth" not in parameters
    assert "reference" not in parameters
    assert synthetic_summary()["passed"] is True
    assert evaluator.main(["synthetic"]) == 0
    output = capsys.readouterr().out
    assert '"passed": true' in output


def test_registered_inventory_and_cost_are_exact() -> None:
    assert evaluator.CASE_IDS == (
        "sv_e12_y00",
        "sv_e12_y01",
        "sv_e12_y04",
        "sv_e12_y08",
    )
    assert evaluator.OUTPUT_CALLS == tuple(range(0, 31, 2))
    assert evaluator.TRANSITION_CALLS == tuple(range(0, 30, 2))
    assert evaluator.EXPECTED_NATIVE_CALLS == 240
    assert evaluator.EXPECTED_FINE_CALLS == 120
    assert evaluator.EXPECTED_LOGICAL_CALLS == 360
    assert len(FROZEN_ACTIVE_CELLS) == 19


def test_error_metric_front_keys_are_not_double_prefixed(monkeypatch) -> None:
    volumes = np.ones(3, dtype=np.float64)
    state = np.asarray(
        [[1.0, 0.0, 0.0, 2.5], [1.0, 0.0, 0.0, 2.5], [1.0, 0.0, 0.0, 2.5]]
    )
    runtime = type(
        "Runtime",
        (),
        {
            "native_projector": type(
                "Projector", (), {"interior_mask": np.asarray([False, True, False])}
            )(),
            "normalization": type("Normalization", (), {"gamma": 1.4})(),
        },
    )()
    monkeypatch.setattr(
        evaluator.base,
        "_front_errors",
        lambda *_: {
            "front_position": 0.0,
            "front_strength_log_ratio": 0.0,
            "front_thickness_log_ratio": 0.0,
        },
    )

    metrics = evaluator._error_metrics(
        state,
        state,
        runtime=runtime,
        state_scale=np.ones(4),
        volumes=volumes,
    )

    assert metrics["front_position"] == 0.0
    assert "front_front_position" not in metrics


def test_runtime_bundle_check_binds_one_float32_epsilon_floor() -> None:
    count = NATIVE[0] * NATIVE[1]
    nodes = np.linspace(0.01, 1.99, 2 * count, dtype=np.float64).reshape(count, 2)
    volumes = np.linspace(0.8, 1.2, count, dtype=np.float64)
    volumes *= 2.0 / volumes.sum()
    state_scale = np.asarray((1.01, 2.02, 3.03, 4.04), dtype=np.float64)
    residual_scale = np.asarray((0.101, 0.202, 0.303, 0.404), dtype=np.float64)
    runtime = type(
        "Runtime",
        (),
        {
            "geometry_by_resolution": {
                evaluator.NATIVE_RESOLUTION: type(
                    "Geometry",
                    (),
                    {"nodes": nodes, "node_measures": volumes[:, None]},
                )()
            },
            "normalization": type(
                "Normalization",
                (),
                {
                    "state_scale": state_scale,
                    "residual_scale": residual_scale,
                    "gamma": 1.4,
                },
            )(),
        },
    )()
    bundle = {
        "nodes": nodes.astype(np.float32),
        "volumes": volumes.astype(np.float32),
        "state_scale": state_scale.astype(np.float32),
        "residual_scale": residual_scale.astype(np.float32),
        "gamma": 1.4,
    }

    floors = evaluator._validate_runtime_bundle(runtime, {"sv_e12_y00": bundle})
    assert set(floors) == {"nodes", "volumes", "state_scale", "residual_scale"}
    assert all(
        row["maximum_absolute"] <= row["one_float32_epsilon_tolerance"]
        for row in floors.values()
    )
    assert any(row["maximum_absolute"] > 0.0 for row in floors.values())

    bad_bundle = {**bundle, "volumes": 1.001 * bundle["volumes"]}
    with pytest.raises(ValueError, match="float32 floor"):
        evaluator._validate_runtime_bundle(runtime, {"sv_e12_y00": bad_bundle})


def test_result_payload_boundary_converts_numpy_scalars_before_hashing() -> None:
    safe = evaluator.json_safe_with_paths(
        {
            "check": np.bool_(True),
            "count": np.int64(3),
            "score": np.float64(0.5),
        }
    )
    payload = evaluator.with_payload_sha256(safe)

    assert payload["check"] is True
    assert payload["count"] == 3
    assert payload["score"] == 0.5
    json.dumps(payload, allow_nan=False)

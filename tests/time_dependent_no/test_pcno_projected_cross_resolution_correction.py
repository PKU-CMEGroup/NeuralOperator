from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.time_dependent_no import (
    evaluate_pcno_projected_cross_resolution_correction as runner,
)
from utility.time_dependent_no import (
    pcno_cross_resolution_teacher_forced as teacher,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_correction import (
    PROJECTED_SELECTION_VIEW,
    project_parallel_field,
    projected_candidate_snapshot,
    projected_coefficient_stability,
    projected_correction_field,
    projected_grouped_crossfit,
    projected_native_prediction,
    projected_synchronized_fusion_step,
    score_projected_cell,
)
from utility.time_dependent_no.pcno_resolution_transfer import Resolution

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
    return teacher.build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
    )


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
    group_id: str,
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
        group_id=group_id,
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


def _crossfit_fixture():
    groups = {
        f"g{group}": tuple(f"g{group}_m{member}" for member in range(2))
        for group in range(4)
    }
    calls = (0, 1, 2)
    snapshots = []
    seed = 0
    for group_id, case_ids in groups.items():
        for case_id in case_ids:
            for input_call in calls:
                snapshots.append(
                    _snapshot(
                        case_id,
                        group_id,
                        input_call,
                        seed=seed,
                    )
                )
                seed += 1
    return snapshots, groups, calls


def test_parallel_projection_closes_is_idempotent_and_excludes_boundary() -> None:
    projector = _projector()
    rng = np.random.default_rng(10)
    field = rng.normal(size=(RESOLUTION[0] * RESOLUTION[1], 4))
    parallel, closure = project_parallel_field(field, projector)
    pieces, _ = projector.split(field)

    assert np.allclose(parallel, pieces["parallel"], atol=0.0, rtol=0.0)
    assert np.count_nonzero(parallel[~projector.interior_mask]) == 0
    assert closure.maximum_reconstruction_abs <= 1.0e-15
    assert closure.maximum_idempotence_abs <= 1.0e-14
    assert closure.relative_parallel_orthogonal_inner <= 1.0e-14
    assert closure.maximum_excluded_parallel_abs == 0.0

    nodes, volumes, node_type = _grid()
    wrong_rank = teacher.build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=7,
    )
    with pytest.raises(ValueError, match="rank-8"):
        project_parallel_field(field, wrong_rank)


def test_projected_crossfit_recovers_coefficients_and_is_permutation_invariant() -> (
    None
):
    snapshots, groups, calls = _crossfit_fixture()
    projector = _projector()
    result = projected_grouped_crossfit(
        snapshots,
        resolution=RESOLUTION,
        projector=projector,
        expected_groups=groups,
        expected_input_calls=calls,
    )
    reversed_result = projected_grouped_crossfit(
        list(reversed(snapshots)),
        resolution=RESOLUTION,
        projector=projector,
        expected_groups=groups,
        expected_input_calls=calls,
    )

    assert result == reversed_result
    assert result["selection_view"] == PROJECTED_SELECTION_VIEW
    assert result["selected_model"] == "two_term"
    assert result["selected_coefficients"] == pytest.approx((ALPHA, BETA))
    assert result["models"]["two_term"]["skill_vs_zero"] == pytest.approx(1.0)
    stability = projected_coefficient_stability(
        result,
        minimum_same_sign_folds=len(groups),
    )
    assert stability["status"] == "passed"
    assert all(row["same_sign_fold_count"] == len(groups) for row in stability["rows"])


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "bool_call", "volumes"))
def test_projected_crossfit_fails_closed_on_inventory_or_metric_drift(
    mutation: str,
) -> None:
    snapshots, groups, calls = _crossfit_fixture()
    if mutation == "missing":
        snapshots = snapshots[:-1]
    elif mutation == "duplicate":
        snapshots.append(snapshots[0])
    elif mutation == "bool_call":
        snapshots[0] = replace(snapshots[0], input_call=True)
    elif mutation == "volumes":
        snapshots[0] = replace(snapshots[0], volumes=snapshots[0].volumes * 2.0)

    with pytest.raises(ValueError):
        projected_grouped_crossfit(
            snapshots,
            resolution=RESOLUTION,
            projector=_projector(),
            expected_groups=groups,
            expected_input_calls=calls,
        )


def test_projected_prediction_changes_only_parallel_interior_component() -> None:
    snapshot = _snapshot("g0_m0", "g0", 0, seed=20)
    projector = _projector()
    raw = np.arange(RESOLUTION[0] * RESOLUTION[1] * 4, dtype=np.float32).reshape(-1, 4)

    zero = projected_native_prediction(
        raw,
        snapshot.basis,
        projector,
        alpha=0.0,
        beta=0.0,
    )
    corrected = projected_native_prediction(
        raw,
        snapshot.basis,
        projector,
        alpha=ALPHA,
        beta=BETA,
    )
    correction = projected_correction_field(
        snapshot.basis,
        projector,
        alpha=ALPHA,
        beta=BETA,
    )
    correction_parts, _ = projector.split(corrected - raw)

    assert np.array_equal(zero, raw)
    assert zero.dtype == raw.dtype
    assert np.allclose(corrected - raw, correction, atol=1.0e-12)
    assert np.max(np.abs(correction_parts["orthogonal"])) <= 1.0e-12
    assert np.count_nonzero(correction[~projector.interior_mask]) == 0


def test_projected_candidate_keeps_full_target_and_removes_only_parallel_error() -> (
    None
):
    snapshot = _snapshot("g0_m0", "g0", 0, seed=30)
    projector = _projector()
    transformed, closure = projected_candidate_snapshot(snapshot, projector)
    correction = (
        ALPHA * transformed.basis.native_minus_coarse
        + BETA * transformed.basis.fine_minus_native
    )
    before, _ = projector.split(snapshot.target_correction)
    after, _ = projector.split(snapshot.target_correction - correction)

    assert np.array_equal(transformed.target_correction, snapshot.target_correction)
    assert np.max(np.abs(after["parallel"])) <= 1.0e-10
    assert np.allclose(after["orthogonal"], before["orthogonal"], atol=1.0e-12)
    assert np.array_equal(after["excluded"], before["excluded"])
    assert closure.maximum_idempotence_abs <= 1.0e-14


def test_projected_score_cell_preserves_orthogonal_and_excluded_rms() -> None:
    projector = _projector()
    snapshots = []
    seed = 100
    for case_id in teacher.EVALUATION_CASE_IDS:
        for input_call in teacher.HELD_OUT_INPUT_CALLS:
            snapshots.append(
                _snapshot(
                    case_id,
                    case_id.split("_")[1],
                    input_call,
                    seed=seed,
                )
            )
            seed += 1
    payload = score_projected_cell(
        snapshots,
        cell="joint_held_out",
        coefficients=(ALPHA, BETA),
        resolution=RESOLUTION,
        projector=projector,
    )

    population = {
        row["view"]: row for row in payload["rows"] if row["scope"] == "population"
    }
    assert payload["correction_policy"] == PROJECTED_SELECTION_VIEW
    assert population["rank8_parallel"]["rms_ratio_vs_zero"] == pytest.approx(
        0.0,
        abs=1.0e-7,
    )
    assert population["rank8_orthogonal"]["rms_ratio_vs_zero"] == pytest.approx(1.0)
    assert population["rank8_excluded"]["rms_ratio_vs_zero"] == pytest.approx(1.0)
    assert population["full"]["skill_vs_zero"] > 0.0


def test_projected_synchronized_step_uses_three_common_source_predictions() -> None:
    contract = ResolutionContract(coarse=(4, 2), native=RESOLUTION, fine=(16, 8))
    projector = _projector()
    rng = np.random.default_rng(40)
    native_state = rng.normal(size=(RESOLUTION[0] * RESOLUTION[1], 4))
    expected = prepare_common_native_inputs(native_state, contract=contract)
    calls: list[tuple[Resolution, np.ndarray]] = []

    def predictor(resolution: Resolution, state: np.ndarray) -> np.ndarray:
        calls.append((resolution, np.array(state, copy=True)))
        return state + 0.01

    step = projected_synchronized_fusion_step(
        native_state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        alpha=0.0,
        beta=0.0,
    )

    assert [resolution for resolution, _ in calls] == [
        contract.coarse,
        contract.native,
        contract.fine,
    ]
    assert all(
        np.array_equal(state, expected.model_inputs[resolution])
        for resolution, state in calls
    )
    assert np.array_equal(step.next_native_state, step.predictions[contract.native])


def test_a1_cli_contract_and_synthetic_smoke_remain_scientific_input_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_sources = (
        "tests/time_dependent_no/test_pcno_projected_cross_resolution_correction.py",
    )
    monkeypatch.setattr(runner, "SOURCE_PATHS", synthetic_sources)
    monkeypatch.setattr(
        runner,
        "FROZEN_A2_SOURCE_SHA256",
        runner.sha256_files(synthetic_sources, root=runner.ROOT),
    )
    dry_run = runner.canonical_dry_run_contract()
    synthetic = runner.synthetic_smoke_summary(seed=0)
    teacher.verify_payload_sha256(dry_run)
    teacher.verify_payload_sha256(synthetic)

    assert dry_run["executes_checkpoint"] is False
    assert dry_run["executes_dataset_arrays"] is False
    assert dry_run["remote_execution_authorized"] is False
    assert dry_run["prior_evidence"]["open_population_reuse"] == (
        "adaptive_open_validation"
    )
    assert synthetic["status"] == "passed"
    assert all(synthetic["checks"].values())
    assert synthetic["checkpoint_loaded"] is False
    assert synthetic["dataset_loaded"] is False
    assert synthetic["reference_loaded"] is False

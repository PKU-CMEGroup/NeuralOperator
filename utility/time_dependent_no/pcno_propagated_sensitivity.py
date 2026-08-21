"""Target-free propagated-sensitivity diagnostics for W26-L5.

The correction is fixed before this module is called.  The only online signal
introduced here is the finite-difference response of one additional model step.
Reference errors are accepted by the offline scoring functions, never by the
lookahead function used to form an inference-time signal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    ModalCell,
    validate_active_cells,
)

DENOMINATOR_FLOOR = 1.0e-12
GAIN_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)
SELECTOR_ORDER: tuple[str, ...] = ("q25", "q50", "q75")


@dataclass(frozen=True)
class PropagatedLookahead:
    """Two native lookaheads from raw and corrected one-step proposals."""

    raw_prediction: np.ndarray
    corrected_prediction: np.ndarray
    response: np.ndarray


@dataclass(frozen=True)
class PolicyRecord:
    """Scalar counterfactual evidence for one case and input call."""

    case_id: str
    group_id: str
    input_call: int
    response_gain: float
    raw_state_energy: float
    corrected_state_energy: float
    raw_control_energy: Mapping[str, float]
    corrected_control_energy: Mapping[str, float]
    corrected_valid: bool


def _finite_field(value: np.ndarray, *, name: str) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError(f"{name} must be a finite node-by-component array")
    return field


def _metric_contract(
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    *,
    nodes: int,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(volumes, dtype=np.float64)
    scale = np.asarray(component_scale, dtype=np.float64)
    if mass.shape == (nodes, 1):
        mass = mass.reshape(nodes)
    if mass.shape != (nodes,) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("volumes must be finite, positive, and node aligned")
    if (
        scale.shape != (components,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "component_scale must be finite, positive, and component aligned"
        )
    return mass, scale


def propagated_lookahead(
    raw_proposal: np.ndarray,
    correction: np.ndarray,
    *,
    predictor: Callable[[np.ndarray], np.ndarray],
) -> PropagatedLookahead:
    """Evaluate the exact finite correction response using two model calls.

    This function deliberately has no reference/target argument.  A live SP19
    decision additionally needs the synchronized native and fine calls that
    construct ``correction``.
    """

    raw = _finite_field(raw_proposal, name="raw_proposal")
    delta = _finite_field(correction, name="correction")
    if delta.shape != raw.shape:
        raise ValueError("correction must align with raw_proposal")
    raw_prediction = _finite_field(
        predictor(np.array(raw, copy=True)), name="raw lookahead prediction"
    )
    corrected_prediction = _finite_field(
        predictor(np.array(raw + delta, copy=True)),
        name="corrected lookahead prediction",
    )
    if raw_prediction.shape != raw.shape or corrected_prediction.shape != raw.shape:
        raise ValueError("lookahead predictions must preserve the native state shape")
    return PropagatedLookahead(
        raw_prediction=raw_prediction,
        corrected_prediction=corrected_prediction,
        response=corrected_prediction - raw_prediction,
    )


def weighted_quadratic_statistics(
    baseline_error: np.ndarray,
    response: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | str | None]:
    """Return the exact quadratic identity for ``error + response``."""

    error = _finite_field(baseline_error, name="baseline_error")
    change = _finite_field(response, name="response")
    if change.shape != error.shape:
        raise ValueError("response must align with baseline_error")
    mass, scale = _metric_contract(
        volumes,
        component_scale,
        nodes=error.shape[0],
        components=error.shape[1],
    )
    if mask is not None:
        selected = np.asarray(mask)
        if selected.shape != (error.shape[0],) or selected.dtype != np.bool_:
            raise ValueError("mask must be a node-aligned boolean array")
        mass = mass * selected.astype(np.float64)
    normalization = float(np.sum(mass)) * error.shape[1]
    if normalization <= DENOMINATOR_FLOOR:
        return {
            "status": "small_measure",
            "error_energy": None,
            "response_energy": None,
            "cross": None,
            "corrected_energy": None,
            "quadratic_closure_abs": None,
            "directional_gamma": None,
            "cosine": None,
        }
    scaled_error = error / scale[None, :]
    scaled_change = change / scale[None, :]
    error_energy = float(
        np.sum(mass[:, None] * np.square(scaled_error)) / normalization
    )
    response_energy = float(
        np.sum(mass[:, None] * np.square(scaled_change)) / normalization
    )
    cross = float(np.sum(mass[:, None] * scaled_error * scaled_change) / normalization)
    corrected_direct = float(
        np.sum(mass[:, None] * np.square(scaled_error + scaled_change)) / normalization
    )
    corrected_quadratic = error_energy + 2.0 * cross + response_energy
    denominator = float(np.sqrt(error_energy * response_energy))
    return {
        "status": "ok",
        "error_energy": error_energy,
        "response_energy": response_energy,
        "cross": cross,
        "corrected_energy": corrected_direct,
        "quadratic_closure_abs": abs(corrected_direct - corrected_quadratic),
        "directional_gamma": (
            cross / response_energy if response_energy > DENOMINATOR_FLOOR**2 else None
        ),
        "cosine": (cross / denominator if denominator > DENOMINATOR_FLOOR**2 else None),
    }


def modal_partition_fields(
    field: np.ndarray,
    projector: Any,
    *,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> dict[str, np.ndarray]:
    """Split a field into an orthogonal rank-8/SP19 partition."""

    values = _finite_field(field, name="field")
    coordinates = modal_coordinates(
        values,
        projector,
        component_scale=component_scale,
    )
    cells = set(
        validate_active_cells(
            active_cells,
            rank=coordinates.shape[0],
            components=coordinates.shape[1],
        )
    )

    def reconstruct(selected: Callable[[int, int], bool]) -> np.ndarray:
        retained = np.zeros_like(coordinates)
        for mode in range(coordinates.shape[0]):
            for component in range(coordinates.shape[1]):
                if selected(mode, component):
                    retained[mode, component] = coordinates[mode, component]
        return reconstruct_modal_field(
            retained,
            projector,
            component_scale=component_scale,
        )

    constant = reconstruct(lambda mode, component: mode == 0)
    low_active = reconstruct(
        lambda mode, component: 1 <= mode <= 3 and (mode, component) in cells
    )
    upper_active = reconstruct(
        lambda mode, component: mode >= 4 and (mode, component) in cells
    )
    inactive = reconstruct(
        lambda mode, component: mode > 0 and (mode, component) not in cells
    )
    rank8 = constant + low_active + upper_active + inactive
    return {
        "full": np.array(values, copy=True),
        "constant": constant,
        "low_active_modes_1_3": low_active,
        "upper_active_modes_4_7": upper_active,
        "rank8_inactive": inactive,
        "sp19_active": low_active + upper_active,
        "high_rank_remainder": values - rank8,
    }


def _strict_call(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("input_call must be an integer")
    call = int(value)
    if call < 0:
        raise ValueError("input_call must be nonnegative")
    return call


def _validate_policy_inventory(
    records: Sequence[PolicyRecord],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> tuple[PolicyRecord, ...]:
    groups: dict[str, tuple[str, ...]] = {}
    seen_cases: set[str] = set()
    for raw_group, raw_cases in expected_groups.items():
        if not isinstance(raw_group, str) or not raw_group:
            raise ValueError("group names must be nonempty strings")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise TypeError("group cases must be a sequence")
        cases = tuple(raw_cases)
        if not cases or any(not isinstance(case, str) or not case for case in cases):
            raise ValueError("case IDs must be nonempty strings")
        if len(set(cases)) != len(cases) or seen_cases.intersection(cases):
            raise ValueError("expected groups must contain unique, disjoint cases")
        groups[raw_group] = cases
        seen_cases.update(cases)
    if not groups:
        raise ValueError("at least one expected group is required")
    calls = tuple(_strict_call(value) for value in expected_input_calls)
    if not calls or len(set(calls)) != len(calls):
        raise ValueError("expected_input_calls must be nonempty and unique")
    case_group = {case: group for group, cases in groups.items() for case in cases}
    canonical: list[PolicyRecord] = []
    pairs: set[tuple[str, int]] = set()
    expected_controls: set[str] | None = None
    for record in records:
        if not isinstance(record, PolicyRecord):
            raise TypeError("records must contain PolicyRecord values")
        call = _strict_call(record.input_call)
        if (
            record.case_id not in case_group
            or record.group_id != case_group[record.case_id]
        ):
            raise ValueError("policy record case/group inventory differs")
        pair = (record.case_id, call)
        if pair in pairs:
            raise ValueError("duplicate policy record")
        pairs.add(pair)
        control_keys = set(record.raw_control_energy)
        if control_keys != set(record.corrected_control_energy):
            raise ValueError("raw/corrected control inventories differ")
        if expected_controls is None:
            expected_controls = control_keys
        elif control_keys != expected_controls:
            raise ValueError("control inventory differs across records")
        numeric = (
            record.response_gain,
            record.raw_state_energy,
            record.corrected_state_energy,
            *record.raw_control_energy.values(),
            *record.corrected_control_energy.values(),
        )
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0 for value in numeric
        ):
            raise ValueError("policy energies and gains must be finite and nonnegative")
        if type(record.corrected_valid) is not bool:
            raise ValueError("corrected_valid must be a bool")
        canonical.append(record)
    expected_pairs = {(case, call) for case in seen_cases for call in calls}
    if pairs != expected_pairs:
        raise ValueError("policy record case/call inventory differs")
    return tuple(sorted(canonical, key=lambda row: (row.case_id, row.input_call)))


def _case_first_mean(
    records: Sequence[PolicyRecord],
    value: Callable[[PolicyRecord], float],
) -> float:
    cases = sorted({row.case_id for row in records})
    return float(
        np.mean(
            [
                np.mean([value(row) for row in records if row.case_id == case_id])
                for case_id in cases
            ]
        )
    )


def evaluate_gain_threshold(
    records: Sequence[PolicyRecord],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate an apply-if-low-gain policy with case-first weighting."""

    selected = {id(row): bool(row.response_gain <= threshold) for row in records}
    raw_state = _case_first_mean(records, lambda row: row.raw_state_energy)
    policy_state = _case_first_mean(
        records,
        lambda row: (
            row.corrected_state_energy if selected[id(row)] else row.raw_state_energy
        ),
    )
    state_status = "ok" if raw_state > DENOMINATOR_FLOOR**2 else "small_denominator"
    controls: dict[str, dict[str, float | str | None]] = {}
    control_keys = sorted(records[0].raw_control_energy) if records else []
    for key in control_keys:
        raw = _case_first_mean(records, lambda row, k=key: row.raw_control_energy[k])
        policy = _case_first_mean(
            records,
            lambda row, k=key: (
                row.corrected_control_energy[k]
                if selected[id(row)]
                else row.raw_control_energy[k]
            ),
        )
        status = "ok" if raw > DENOMINATOR_FLOOR**2 else "small_denominator"
        controls[key] = {
            "raw_energy": raw,
            "selected_energy": policy,
            "rms_ratio": float(np.sqrt(policy / raw)) if status == "ok" else None,
            "status": status,
        }
    selected_records = [row for row in records if selected[id(row)]]
    return {
        "threshold": float(threshold),
        "coverage": len(selected_records) / len(records) if records else 0.0,
        "selected_count": len(selected_records),
        "state_raw_energy": raw_state,
        "state_selected_energy": policy_state,
        "state_rms_ratio": (
            float(np.sqrt(policy_state / raw_state)) if state_status == "ok" else None
        ),
        "state_status": state_status,
        "controls": controls,
        "all_selected_valid": all(row.corrected_valid for row in selected_records),
    }


def _candidate_thresholds(records: Sequence[PolicyRecord]) -> dict[str, float]:
    gains = np.asarray([row.response_gain for row in records], dtype=np.float64)
    return {
        "never": -1.0,
        **{
            f"q{round(100 * quantile):02d}": float(
                np.quantile(gains, quantile, method="linear")
            )
            for quantile in GAIN_QUANTILES
        },
        "always": float(np.nextafter(np.max(gains), np.inf)),
    }


def select_gain_threshold(records: Sequence[PolicyRecord]) -> dict[str, Any]:
    """Select a preregistered nontrivial threshold or fail closed to zero."""

    evaluations = {
        name: evaluate_gain_threshold(records, threshold=threshold)
        for name, threshold in _candidate_thresholds(records).items()
    }
    qualified: list[tuple[float, int, str]] = []
    for order, name in enumerate(SELECTOR_ORDER):
        evidence = evaluations[name]
        controls_resolved = all(
            row["status"] == "ok" for row in evidence["controls"].values()
        )
        controls_safe = controls_resolved and all(
            float(row["rms_ratio"]) <= 1.0 + 1.0e-8
            for row in evidence["controls"].values()
        )
        ratio = evidence["state_rms_ratio"]
        if (
            evidence["state_status"] == "ok"
            and ratio is not None
            and float(ratio) < 1.0
            and controls_safe
            and evidence["all_selected_valid"]
        ):
            qualified.append((float(ratio), order, name))
    selected_name = min(qualified)[2] if qualified else "never"
    selected = evaluations[selected_name]
    return {
        "selected_name": selected_name,
        "selected_threshold": selected["threshold"],
        "selected_evidence": selected,
        "candidate_evidence": evaluations,
        "nontrivial_selected": selected_name in SELECTOR_ORDER,
    }


def grouped_gain_threshold_crossfit(
    records: Sequence[PolicyRecord],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Leave-one-physical-group-out cross-fit of the low-gain selector."""

    canonical = _validate_policy_inventory(
        records,
        expected_groups=expected_groups,
        expected_input_calls=expected_input_calls,
    )
    folds: list[dict[str, Any]] = []
    selected_by_pair: dict[tuple[str, int], bool] = {}
    for held_group in sorted(expected_groups):
        train = [row for row in canonical if row.group_id != held_group]
        held = [row for row in canonical if row.group_id == held_group]
        selection = select_gain_threshold(train)
        threshold = float(selection["selected_threshold"])
        held_evidence = evaluate_gain_threshold(held, threshold=threshold)
        for row in held:
            selected_by_pair[(row.case_id, row.input_call)] = bool(
                row.response_gain <= threshold
            )
        folds.append(
            {
                "held_group": held_group,
                "selected_name": selection["selected_name"],
                "selected_threshold": threshold,
                "training": selection["selected_evidence"],
                "held_out": held_evidence,
            }
        )

    def selected(row: PolicyRecord) -> bool:
        return selected_by_pair[(row.case_id, row.input_call)]

    oof_records = []
    for row in canonical:
        use = selected(row)
        oof_records.append(
            PolicyRecord(
                case_id=row.case_id,
                group_id=row.group_id,
                input_call=row.input_call,
                response_gain=0.0 if use else 1.0,
                raw_state_energy=row.raw_state_energy,
                corrected_state_energy=row.corrected_state_energy,
                raw_control_energy=row.raw_control_energy,
                corrected_control_energy=row.corrected_control_energy,
                corrected_valid=row.corrected_valid,
            )
        )
    oof = evaluate_gain_threshold(oof_records, threshold=0.0)
    case_wins = 0
    group_wins = 0
    for case_id in sorted({row.case_id for row in canonical}):
        case_rows = [row for row in canonical if row.case_id == case_id]
        raw = float(np.mean([row.raw_state_energy for row in case_rows]))
        corrected = float(
            np.mean(
                [
                    row.corrected_state_energy
                    if selected(row)
                    else row.raw_state_energy
                    for row in case_rows
                ]
            )
        )
        case_wins += corrected < raw
    for group_id in sorted(expected_groups):
        group_rows = [row for row in canonical if row.group_id == group_id]
        raw = _case_first_mean(group_rows, lambda row: row.raw_state_energy)
        corrected = _case_first_mean(
            group_rows,
            lambda row: (
                row.corrected_state_energy if selected(row) else row.raw_state_energy
            ),
        )
        group_wins += corrected < raw
    names = [row["selected_name"] for row in folds]
    nontrivial_names = [name for name in names if name in SELECTOR_ORDER]
    modal_count = (
        Counter(nontrivial_names).most_common(1)[0][1] if nontrivial_names else 0
    )
    controls_resolved = all(row["status"] == "ok" for row in oof["controls"].values())
    controls_safe = controls_resolved and all(
        float(row["rms_ratio"]) <= 1.0 + 1.0e-8 for row in oof["controls"].values()
    )
    checks = {
        "all_folds_select_nontrivial": all(name in SELECTOR_ORDER for name in names),
        "selector_stable_at_least_seven_of_nine": len(folds) == 9 and modal_count >= 7,
        "oof_state_rms_ratio_at_most_0p995": oof["state_status"] == "ok"
        and float(oof["state_rms_ratio"]) <= 0.995,
        "minimum_fourteen_case_wins": case_wins >= 14,
        "minimum_eight_group_wins": group_wins >= 8,
        "all_oof_controls_resolved_and_no_harm": controls_safe,
        "all_selected_lookaheads_valid": oof["all_selected_valid"],
        "nondegenerate_oof_coverage": 0.10 <= float(oof["coverage"]) <= 0.90,
    }
    return {
        "folds": folds,
        "oof": oof,
        "case_win_count": case_wins,
        "group_win_count": group_wins,
        "selector_counts": dict(Counter(names)),
        "full_calibration_selection": select_gain_threshold(canonical),
        "prospective_gate": {
            "status": "evaluation_authorized" if all(checks.values()) else "stopped",
            "evaluation_authorized": all(checks.values()),
            "checks": checks,
        },
    }


def grouped_directional_crossfit(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Cross-fit ``baseline_error ~= gamma * response`` for each view."""

    calls = {_strict_call(value) for value in expected_input_calls}
    case_group = {
        case_id: group_id
        for group_id, case_ids in expected_groups.items()
        for case_id in case_ids
    }
    expected_pairs = {(case, call) for case in case_group for call in calls}
    views = sorted({str(row.get("view")) for row in rows})
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for view in views:
        view_rows = [row for row in rows if row.get("view") == view]
        lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in view_rows:
            case_id = row.get("case_id")
            group_id = row.get("group_id")
            call = _strict_call(row.get("input_call"))
            if case_id not in case_group or group_id != case_group[case_id]:
                raise ValueError("directional row case/group inventory differs")
            pair = (case_id, call)
            if pair in lookup:
                raise ValueError("duplicate directional row")
            if row.get("status") != "ok":
                raise ValueError("directional row contains an unresolved metric")
            for key in ("error_energy", "response_energy", "cross", "corrected_energy"):
                if not np.isfinite(float(row[key])):
                    raise ValueError("directional statistics must be finite")
            lookup[pair] = row
        if set(lookup) != expected_pairs:
            raise ValueError(f"directional inventory differs for view {view!r}")

        def case_first_sum(
            selected_rows: Sequence[Mapping[str, Any]], key: str
        ) -> float:
            cases = sorted({str(row["case_id"]) for row in selected_rows})
            return float(
                np.sum(
                    [
                        np.mean(
                            [
                                float(row[key])
                                for row in selected_rows
                                if row["case_id"] == case
                            ]
                        )
                        for case in cases
                    ]
                )
            )

        oof_error = 0.0
        oof_fit_error = 0.0
        oof_dose_error = 0.0
        fold_gammas: list[float] = []
        for held_group in sorted(expected_groups):
            train = [row for row in view_rows if row["group_id"] != held_group]
            held = [row for row in view_rows if row["group_id"] == held_group]
            pp = case_first_sum(train, "response_energy")
            pe = case_first_sum(train, "cross")
            status = "ok" if pp > DENOMINATOR_FLOOR**2 else "small_denominator"
            gamma = pe / pp if status == "ok" else None
            dose = float(np.clip(-gamma, 0.0, 1.0)) if gamma is not None else 0.0
            held_ee = case_first_sum(held, "error_energy")
            held_pp = case_first_sum(held, "response_energy")
            held_pe = case_first_sum(held, "cross")
            fit_error = (
                held_ee - 2.0 * gamma * held_pe + gamma * gamma * held_pp
                if gamma is not None
                else held_ee
            )
            dose_error = held_ee + 2.0 * dose * held_pe + dose * dose * held_pp
            oof_error += held_ee
            oof_fit_error += fit_error
            oof_dose_error += dose_error
            if gamma is not None:
                fold_gammas.append(float(gamma))
            output_rows.append(
                {
                    "view": view,
                    "held_group": held_group,
                    "status": status,
                    "gamma": gamma,
                    "clipped_dose": dose,
                    "held_error_energy": held_ee,
                    "held_fit_error": fit_error,
                    "held_dose_error": dose_error,
                }
            )
        full_ee = case_first_sum(view_rows, "error_energy")
        full_corrected = case_first_sum(view_rows, "corrected_energy")
        case_cosines = []
        for case_id in sorted(case_group):
            case_rows = [row for row in view_rows if row["case_id"] == case_id]
            ee = float(np.sum([row["error_energy"] for row in case_rows]))
            pp = float(np.sum([row["response_energy"] for row in case_rows]))
            pe = float(np.sum([row["cross"] for row in case_rows]))
            denominator = float(np.sqrt(ee * pp))
            if denominator > DENOMINATOR_FLOOR**2:
                case_cosines.append(pe / denominator)
        sign_stable = bool(fold_gammas) and (
            all(value < 0.0 for value in fold_gammas)
            or all(value > 0.0 for value in fold_gammas)
        )
        summaries[view] = {
            "status": "ok"
            if len(fold_gammas) == len(expected_groups)
            else "small_denominator",
            "fold_gamma_min": min(fold_gammas) if fold_gammas else None,
            "fold_gamma_max": max(fold_gammas) if fold_gammas else None,
            "fold_gamma_sign_stable": sign_stable,
            "oof_prediction_r2": (
                1.0 - oof_fit_error / oof_error
                if oof_error > DENOMINATOR_FLOOR**2
                else None
            ),
            "oof_clipped_dose_rms_ratio": (
                float(np.sqrt(oof_dose_error / oof_error))
                if oof_error > DENOMINATOR_FLOOR**2
                else None
            ),
            "always_full_correction_rms_ratio": (
                float(np.sqrt(full_corrected / full_ee))
                if full_ee > DENOMINATOR_FLOOR**2
                else None
            ),
            "median_case_cosine": (
                float(np.median(case_cosines)) if case_cosines else None
            ),
            "case_cosine_status": "ok" if case_cosines else "small_denominator",
        }
    return {"fold_rows": output_rows, "views": summaries}

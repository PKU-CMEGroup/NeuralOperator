from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import utility.time_dependent_no.su2_naca0012_relabel as relabel
from utility.time_dependent_no.su2_naca_trajectory import NACA_HISTORY_FIELDS
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    SU2_BINARY_MAGIC,
    SU2_FIELD_NAME_BYTES,
    read_su2_binary_restart,
    sha256_file,
)


def _write_restart(
    path: Path,
    values: np.ndarray,
    fields: tuple[str, ...] = NACA_NATIVE_RESTART_FIELDS,
) -> None:
    assert values.shape[1] == len(fields)
    header = (SU2_BINARY_MAGIC, len(fields), values.shape[0], 0, 0)
    with path.open("wb") as handle:
        handle.write(struct.pack("<5i", *header))
        for field in fields:
            encoded = field.encode("ascii")
            handle.write(encoded + b"\0" * (SU2_FIELD_NAME_BYTES - len(encoded)))
        handle.write(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))


def _native_values(point_count: int = 4) -> np.ndarray:
    values = np.empty((point_count, len(NACA_NATIVE_RESTART_FIELDS)), dtype=np.float64)
    for column in range(values.shape[1]):
        values[:, column] = 10.0 * column + np.arange(point_count, dtype=np.float64)
    values[:, NACA_NATIVE_RESTART_FIELDS.index("x")] = np.linspace(
        0.0, 1.0, point_count
    )
    values[:, NACA_NATIVE_RESTART_FIELDS.index("y")] = np.linspace(
        -0.5, 0.5, point_count
    )
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Density")] = 1.0
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Momentum_x")] = 0.1
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Momentum_y")] = 0.2
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Energy")] = 3.0
    values[:, NACA_NATIVE_RESTART_FIELDS.index("Nu_Tilde")] = 0.5
    values[0, NACA_NATIVE_RESTART_FIELDS.index("Pressure")] = -0.0
    return values


def _authoritative_state(point_count: int = 4) -> np.ndarray:
    rows = [
        [1.10, 0.15, 0.22, 3.10, -0.01],
        [1.20, 0.16, 0.23, 3.20, 0.02],
        [1.30, 0.17, 0.24, 3.30, 0.03],
        [1.40, 0.18, 0.25, 3.40, 0.04],
    ]
    return np.asarray(rows[:point_count], dtype=np.float32)


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype="<f8").view("<u8")


def _write_history(
    path: Path,
    *,
    time_iter: int,
    inner_iter: int = 9,
    relrms_density: float = -3.0,
    row_count: int = 1,
) -> None:
    row: list[float | int] = []
    for field in NACA_HISTORY_FIELDS:
        if field == "Time_Iter":
            row.append(time_iter)
        elif field == "Inner_Iter":
            row.append(inner_iter)
        elif field == "relrms[Rho]":
            row.append(relrms_density)
        else:
            row.append(-4.0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NACA_HISTORY_FIELDS)
        for _ in range(row_count):
            writer.writerow(row)


def _all_receipt_checks(value: bool = True) -> dict[str, bool]:
    return {name: value for name in relabel.REQUIRED_RECEIPT_CHECKS}


def test_train_center_maps_to_absolute_bdf2_indices_and_config() -> None:
    step = relabel.relabel_step_index(956)

    assert step.previous_index == 955
    assert step.current_index == 956
    assert step.restart_iter == 957
    assert step.time_iter == 958
    assert step.expected_output_index == 957
    assert step.input_filenames == (
        "relabel_input_00955.dat",
        "relabel_input_00956.dat",
    )
    assert step.output_filename == "relabel_output_00957.dat"
    assert step.history_filename == "history_00957.csv"
    overrides = relabel.relabel_config_overrides(956)
    assert overrides["RESTART_ITER"] == "957"
    assert overrides["TIME_ITER"] == "958"
    assert overrides["INNER_ITER"] == "10"
    assert overrides["CONV_FIELD"] == "REL_RMS_DENSITY"
    assert overrides["CONV_RESIDUAL_MINVAL"] == "-3.0"
    assert overrides["SOLUTION_FILENAME"] == "relabel_input"
    assert overrides["RESTART_FILENAME"] == "relabel_output"
    assert overrides["WINDOW_START_ITER"] == "958"
    assert overrides["HISTORY_WRT_FREQ_INNER"] == "0"
    assert overrides["OUTPUT_FILES"] == "( RESTART )"

    for center in (955, 1194):
        with pytest.raises(PermissionError, match="open train population"):
            relabel.relabel_step_index(center)
    with pytest.raises(TypeError, match="integer"):
        relabel.relabel_step_index(True)


def test_restart_writer_replaces_only_float32_evolved_fields(tmp_path: Path) -> None:
    template = tmp_path / "clean_00956.dat"
    destination = tmp_path / "relabel_input_00956.dat"
    original = _native_values()
    _write_restart(template, original)
    source_bytes = template.read_bytes()
    source_sha256 = sha256_file(template)
    state = _authoritative_state()

    record = relabel.write_restart_from_verified_template(
        template_path=template.resolve(),
        expected_template_sha256=source_sha256,
        destination_path=destination.resolve(),
        authoritative_state=state,
    )

    written = read_su2_binary_restart(destination)
    dynamic_indices = [written.fields.index(name) for name in NACA_DYNAMIC_FIELDS]
    auxiliary_indices = [
        index for index in range(written.num_fields) if index not in dynamic_indices
    ]
    assert written.fields == NACA_NATIVE_RESTART_FIELDS
    assert written.header == (
        SU2_BINARY_MAGIC,
        len(NACA_NATIVE_RESTART_FIELDS),
        original.shape[0],
        0,
        0,
    )
    assert np.array_equal(written.values[:, dynamic_indices], state.astype(np.float64))
    assert np.array_equal(
        _bits(written.values[:, auxiliary_indices]),
        _bits(original[:, auxiliary_indices]),
    )
    assert template.read_bytes() == source_bytes
    assert sha256_file(template) == source_sha256
    assert record["replaced_fields"] == list(NACA_DYNAMIC_FIELDS)
    assert record["bitwise_changed_fields"] == list(NACA_DYNAMIC_FIELDS)
    assert record["non_evolved_fields_bitwise_preserved"] is True
    assert record["source_immutable"] is True
    assert record["atomic_destination"] is True


def test_restart_writer_is_strict_and_rolls_back_failed_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "clean_00956.dat"
    destination = tmp_path / "relabel_input_00956.dat"
    _write_restart(template, _native_values())
    source_bytes = template.read_bytes()

    with pytest.raises(ValueError, match="verified train metadata"):
        relabel.write_restart_from_verified_template(
            template_path=template.resolve(),
            expected_template_sha256="0" * 64,
            destination_path=destination.resolve(),
            authoritative_state=_authoritative_state(),
        )
    assert not destination.exists()

    with pytest.raises(TypeError, match="float32"):
        relabel.write_restart_from_verified_template(
            template_path=template.resolve(),
            expected_template_sha256=sha256_file(template),
            destination_path=destination.resolve(),
            authoritative_state=_authoritative_state().astype(np.float64),
        )
    assert not destination.exists()

    def reject_verification(**_: object) -> None:
        raise ValueError("synthetic verification rejection")

    monkeypatch.setattr(relabel, "_verify_written_restart", reject_verification)
    with pytest.raises(ValueError, match="synthetic verification rejection"):
        relabel.write_restart_from_verified_template(
            template_path=template.resolve(),
            expected_template_sha256=sha256_file(template),
            destination_path=destination.resolve(),
            authoritative_state=_authoritative_state(),
        )
    assert not destination.exists()
    assert template.read_bytes() == source_bytes
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_restart_writer_never_overwrites_destination(tmp_path: Path) -> None:
    template = tmp_path / "clean_00956.dat"
    destination = tmp_path / "relabel_input_00956.dat"
    _write_restart(template, _native_values())
    destination.write_bytes(b"owned-by-caller")

    with pytest.raises(FileExistsError):
        relabel.write_restart_from_verified_template(
            template_path=template.resolve(),
            expected_template_sha256=sha256_file(template),
            destination_path=destination.resolve(),
            authoritative_state=_authoritative_state(),
        )
    assert destination.read_bytes() == b"owned-by-caller"


def test_restart_writer_rejects_non_native_template(tmp_path: Path) -> None:
    template = tmp_path / "reduced.dat"
    destination = tmp_path / "relabel_input_00956.dat"
    fields = NACA_NATIVE_RESTART_FIELDS[:-1]
    _write_restart(template, _native_values()[:, :-1], fields)

    with pytest.raises(ValueError, match="native 19-field schema"):
        relabel.write_restart_from_verified_template(
            template_path=template.resolve(),
            expected_template_sha256=sha256_file(template),
            destination_path=destination.resolve(),
            authoritative_state=_authoritative_state(),
        )
    assert not destination.exists()


def test_admissibility_reports_negative_nu_without_rejecting() -> None:
    state = _authoritative_state()

    report = relabel.naca_state_admissibility(state)

    assert report["finite"]["passed"] is True
    assert report["density"]["passed"] is True
    assert report["internal_energy_density"]["passed"] is True
    assert report["ideal_gas_pressure"]["passed"] is True
    assert report["thermodynamic_gate_passed"] is True
    assert report["nu_tilde"]["acceptance_gate"] is False
    assert report["nu_tilde"]["negative_count"] == 1
    assert report["nu_tilde"]["policy"] == "report_without_clip_mask_or_rejection"


@pytest.mark.parametrize(
    ("state", "failing_section"),
    [
        (np.asarray([[0.0, 0.1, 0.2, 3.0, 0.0]]), "density"),
        (np.asarray([[1.0, 3.0, 0.0, 1.0, 0.0]]), "internal_energy_density"),
        (np.asarray([[1.0, 0.1, 0.2, np.nan, 0.0]]), "finite"),
    ],
)
def test_admissibility_fails_closed_on_flow_state(
    state: np.ndarray, failing_section: str
) -> None:
    report = relabel.naca_state_admissibility(state)

    assert report[failing_section]["passed"] is False
    assert report["thermodynamic_gate_passed"] is False


def test_history_validator_accepts_exact_threshold_and_last_inner_index(
    tmp_path: Path,
) -> None:
    step = relabel.relabel_step_index(956)
    history = tmp_path / step.history_filename
    _write_history(
        history,
        time_iter=step.expected_output_index,
        inner_iter=9,
        relrms_density=-3.0,
    )

    record = relabel.validate_relabel_history(history.resolve(), step=step)

    assert record["time_iter"] == 957
    assert record["inner_iter"] == 9
    assert record["inner_iter_limit"] == 10
    assert record["convergence"] == {
        "field": "relrms[Rho]",
        "threshold_lte": -3.0,
        "observed": -3.0,
        "passed": True,
    }


def test_validators_reject_forged_nontrain_step(tmp_path: Path) -> None:
    protected = relabel.RelabelStepIndex(
        center=1755,
        previous_index=1754,
        current_index=1755,
        restart_iter=1756,
        time_iter=1757,
        expected_output_index=1756,
    )

    with pytest.raises(PermissionError, match="open train population"):
        relabel.validate_relabel_history(
            (tmp_path / protected.history_filename).resolve(), step=protected
        )


@pytest.mark.parametrize(
    ("time_offset", "inner_iter", "residual", "row_count", "message"),
    [
        (0, 9, -2.999, 1, "did not meet"),
        (1, 9, -3.1, 1, "time index"),
        (0, 10, -3.1, 1, "inner iteration"),
        (0, 9, -3.1, 2, "exactly one final row"),
    ],
)
def test_history_validator_fails_closed(
    tmp_path: Path,
    time_offset: int,
    inner_iter: int,
    residual: float,
    row_count: int,
    message: str,
) -> None:
    step = relabel.relabel_step_index(956)
    history = tmp_path / step.history_filename
    _write_history(
        history,
        time_iter=step.expected_output_index + time_offset,
        inner_iter=inner_iter,
        relrms_density=residual,
        row_count=row_count,
    )

    with pytest.raises(ValueError, match=message):
        relabel.validate_relabel_history(history.resolve(), step=step)


def test_output_validator_enforces_mesh_and_flow_but_not_nu_sign(
    tmp_path: Path,
) -> None:
    step = relabel.relabel_step_index(956)
    output = tmp_path / step.output_filename
    values = _native_values()
    dynamic_indices = [
        NACA_NATIVE_RESTART_FIELDS.index(name) for name in NACA_DYNAMIC_FIELDS
    ]
    values[:, dynamic_indices] = _authoritative_state().astype(np.float64)
    _write_restart(output, values)
    coordinate_indices = [NACA_NATIVE_RESTART_FIELDS.index(name) for name in ("x", "y")]
    coordinates = values[:, coordinate_indices].copy()

    dynamic, record = relabel.validate_relabel_output(
        output.resolve(), step=step, expected_coordinates=coordinates
    )

    assert np.array_equal(dynamic, values[:, dynamic_indices])
    assert record["index"] == step.expected_output_index
    assert record["passed"] is True
    assert record["admissibility"]["nu_tilde"]["negative_count"] == 1
    assert record["admissibility"]["thermodynamic_gate_passed"] is True

    shifted_coordinates = coordinates.copy()
    shifted_coordinates[0, 0] += 1.0
    with pytest.raises(ValueError, match="fixed mesh"):
        relabel.validate_relabel_output(
            output.resolve(), step=step, expected_coordinates=shifted_coordinates
        )


def test_output_validator_rejects_nonpositive_pressure(tmp_path: Path) -> None:
    step = relabel.relabel_step_index(956)
    output = tmp_path / step.output_filename
    values = _native_values()
    energy_index = NACA_NATIVE_RESTART_FIELDS.index("Energy")
    values[:, energy_index] = 0.001
    _write_restart(output, values)
    coordinate_indices = [NACA_NATIVE_RESTART_FIELDS.index(name) for name in ("x", "y")]

    with pytest.raises(ValueError, match="thermodynamic admissibility"):
        relabel.validate_relabel_output(
            output.resolve(),
            step=step,
            expected_coordinates=values[:, coordinate_indices].copy(),
        )


def test_receipts_are_derived_self_hashed_and_fail_closed(tmp_path: Path) -> None:
    success_path = tmp_path / "success.json"
    success = relabel.write_fail_closed_receipt(
        success_path.resolve(), center=1075, checks=_all_receipt_checks()
    )

    assert success["status"] == "validation_succeeded"
    assert success["scientifically_usable"] is True
    assert success["index_contract"]["history_indices"] == [1074, 1075]
    assert success["index_contract"]["restart_iter"] == 1076
    assert success["index_contract"]["time_iter"] == 1077
    assert success["index_contract"]["expected_output_index"] == 1076
    assert success["nu_tilde_acceptance_gate"] is False
    assert relabel.load_verified_relabel_receipt(success_path.resolve()) == success

    failed_checks = _all_receipt_checks()
    failed_checks["convergence_relrms_density"] = False
    failure_path = tmp_path / "failure.json"
    failure = relabel.write_fail_closed_receipt(
        failure_path.resolve(), center=1075, checks=failed_checks
    )
    assert failure["status"] == "validation_failed"
    assert failure["scientifically_usable"] is False
    assert failure["error"] == "one or more frozen validation checks failed"
    assert relabel.load_verified_relabel_receipt(failure_path.resolve()) == failure


def test_receipt_rejects_missing_check_and_tampering(tmp_path: Path) -> None:
    incomplete = _all_receipt_checks()
    incomplete.pop("source_immutable")
    with pytest.raises(ValueError, match="frozen required set"):
        relabel.write_fail_closed_receipt(
            (tmp_path / "incomplete.json").resolve(), center=1075, checks=incomplete
        )

    receipt = tmp_path / "receipt.json"
    relabel.write_fail_closed_receipt(
        receipt.resolve(), center=1075, checks=_all_receipt_checks()
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["status"] = "validation_failed"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical payload SHA256"):
        relabel.load_verified_relabel_receipt(receipt.resolve())

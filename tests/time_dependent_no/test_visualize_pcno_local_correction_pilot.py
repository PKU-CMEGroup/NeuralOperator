from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.time_dependent_no.visualize_pcno_local_correction_pilot as visual
from utility.time_dependent_no.pcno_artifacts import sha256_file


def _payload(path: Path, *, cumulative_offset: float = 0.0) -> None:
    calls = 2
    nodes = 4
    shape = (calls, nodes, len(visual.COMPONENTS))
    state_shape = (calls + 1, nodes, len(visual.COMPONENTS))
    truth = np.full(shape, 0.1, dtype=np.float32)
    reference = np.zeros(state_shape, dtype=np.float32)
    reference[1:] = np.cumsum(truth, axis=0)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(visual.RESULT_IDENTITY[0]),
        "case_id": np.asarray("case"),
        "resolution": np.asarray("2x2"),
        "nodes": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "weights": np.ones(nodes),
        "node_type": np.arange(nodes),
        "physical_times": np.asarray([0.0, 0.1, 0.2]),
        "state_scale": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "residual_scale": np.asarray([0.5, 1.0, 1.5, 2.0]),
        "reference_states": reference,
        "true_increment": truth,
    }
    for index, arm in enumerate(visual.VISUAL_ARMS):
        defect = np.full(shape, index * 0.001, dtype=np.float32)
        cumulative = np.cumsum(defect, axis=0)
        if arm == "combined_shock_isotropic":
            cumulative[-1, 0, 0] += cumulative_offset
        prediction = truth + defect
        states = np.zeros(state_shape, dtype=np.float32)
        states[1:] = np.cumsum(prediction, axis=0)
        arrays[f"states__{arm}"] = states
        arrays[f"base_increment__{arm}"] = prediction
        arrays[f"persistent_correction__{arm}"] = np.zeros(shape, dtype=np.float32)
        arrays[f"local_correction__{arm}"] = np.zeros(shape, dtype=np.float32)
        arrays[f"defect__{arm}"] = defect
        arrays[f"cumulative_defect__{arm}"] = cumulative
    np.savez_compressed(path, **arrays)


def test_payload_replays_increment_recurrence_and_cumulative_defect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.npz"
    _payload(path)
    payload = visual._load_payload(path, {"schema": visual.RESULT_IDENTITY[0]})
    fields = visual._animation_fields(payload, "combined_shock_isotropic")
    assert fields["true_increment"].shape == (2, 4, 4)
    assert np.allclose(
        fields["combined_prediction"] - fields["true_increment"],
        fields["combined_defect"],
    )
    assert np.allclose(
        fields["combined_cumulative"],
        np.cumsum(fields["combined_defect"], axis=0),
    )
    assert np.all(np.isfinite(fields["combined_growth"]))

    _payload(path, cumulative_offset=1e-3)
    with pytest.raises(ValueError, match="cumulative-defect replay"):
        visual._load_payload(path, {"schema": visual.RESULT_IDENTITY[0]})


def test_dynamic_node_type_meanings_are_family_local(tmp_path: Path) -> None:
    path = tmp_path / "payload.npz"
    _payload(path)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["node_type"] = np.asarray([0, 1, 2, 4])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="geometry, clock, or scales"):
        visual._load_payload(path, {"schema": visual.RESULT_IDENTITY[0]})


def _call_row(
    case_id: str,
    arm: str,
    *,
    state: float,
    instant: float,
    cumulative: float,
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "arm": arm,
        "call": "1",
        "physical_time": "0.1",
        "state_error_rms": str(state),
        "instant_defect_rms": str(instant),
        "cumulative_defect_rms": str(cumulative),
    }


def test_temporal_aggregation_forms_ratios_before_case_median() -> None:
    rows = [
        _call_row("case_a", "persistent", state=1.0, instant=2.0, cumulative=4.0),
        _call_row("case_b", "persistent", state=100.0, instant=20.0, cumulative=40.0),
    ]
    for arm in visual.COMBINED_ARMS:
        rows.extend(
            [
                _call_row("case_a", arm, state=2.0, instant=4.0, cumulative=8.0),
                _call_row("case_b", arm, state=50.0, instant=10.0, cumulative=20.0),
            ]
        )
    case_rows, aggregate = visual._temporal_rows(rows)
    assert len(case_rows) == 6
    assert len(aggregate) == 3
    assert all(row["median__state_error_ratio"] == 1.25 for row in aggregate)
    assert all(row["median__instant_defect_ratio"] == 1.25 for row in aggregate)
    assert all(row["median__cumulative_defect_ratio"] == 1.25 for row in aggregate)


def _write_csv(path: Path, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def test_result_verification_rejects_mutated_declared_output(tmp_path: Path) -> None:
    results = tmp_path / "results"
    payload_dir = results / "visual_payloads"
    payload_dir.mkdir(parents=True)
    payload_path = payload_dir / "payload.npz"
    _payload(payload_path)
    payload_relative = payload_path.relative_to(results).as_posix()
    payload_hash = sha256_file(payload_path)

    for name in sorted(visual.REQUIRED_TABLES - {"visual_payload_inventory.csv"}):
        _write_csv(results / name, ("case_id",))
    inventory = results / "visual_payload_inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "sha256", "arm_count"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": payload_relative,
                "sha256": payload_hash,
                "arm_count": len(visual.VISUAL_ARMS),
            }
        )
    output_hashes = {
        path.relative_to(results).as_posix(): sha256_file(path)
        for path in results.glob("*.csv")
    }
    output_hashes[payload_relative] = payload_hash
    summary = {
        "schema": visual.RESULT_IDENTITY[0],
        "experiment_contract": visual.RESULT_IDENTITY[1],
        "status": "complete",
        "contract_checks_passed": True,
        "scientific_interpretation_allowed": True,
        "family": "dynamic_fv",
        "boundary_policy": "model_all_nodes raw recurrence; base policy unchanged",
        "output_hashes": output_hashes,
    }
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    observed, payloads = visual._verify_results(results)
    assert observed["status"] == "complete"
    assert payloads == [payload_path.resolve()]

    with (results / "case_contracts.csv").open("a", encoding="utf-8") as handle:
        handle.write("mutated\n")
    with pytest.raises(ValueError, match="output digest mismatch"):
        visual._verify_results(results)

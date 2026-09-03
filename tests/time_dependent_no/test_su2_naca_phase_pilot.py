from __future__ import annotations

import json
import math
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from utility.time_dependent_no import su2_naca_phase_pilot as pilot
from utility.time_dependent_no.su2_naca_trajectory import NACA_TRAJECTORY_RECEIPT_SCHEMA

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY / "docs/time_dependent_no/R0_NACA_PHASE_PILOT_CONTRACT.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _write_self_hashed(path: Path, payload: dict[str, object]) -> dict[str, object]:
    rendered = dict(payload)
    rendered["canonical_payload_sha256"] = sha256(
        pilot._canonical_bytes(rendered)
    ).hexdigest()
    path.write_text(
        json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rendered


def _periodic_arrays(
    *,
    period: int = 64,
    count: int = 1501,
    modulation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time = np.arange(count, dtype=np.float64)
    phase = 2.0 * np.pi * time / period
    envelope = np.ones(count) if modulation is None else modulation
    node_pattern = np.asarray([-1.0, -0.25, 0.4, 1.1])
    states = np.empty((count, len(node_pattern), 5), dtype=np.float64)
    for field in range(5):
        states[:, :, field] = (
            2.0
            + field
            + envelope[:, None]
            * (0.05 + 0.01 * field)
            * np.sin(phase[:, None] + 0.17 * field + 0.2 * node_pattern[None, :])
        )
    history = {
        "CL": envelope * np.sin(phase),
        "CD": 0.3 + 0.04 * envelope * np.sin(phase + 0.3),
        "CFx": np.zeros(count),
        "CFy": np.zeros(count),
    }
    weights = np.vstack(
        (
            np.ones(len(node_pattern)),
            np.ones(len(node_pattern)),
            np.asarray([0.1, 0.2, 0.3, 0.4]),
        )
    )
    return states, weights, history


def test_exact_cycle_supports_phase_only_population() -> None:
    states, weights, history = _periodic_arrays()

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "PHASE_ONLY_SUPPORTED"
    assert extension is None
    assert diagnostics["period"]["period_steps"] == pytest.approx(64.0, rel=1e-3)
    assert diagnostics["five_field_recurrence"]["passed"] is True
    assert set(diagnostics["five_field_recurrence"]["views"]) == set(pilot._ALL_VIEWS)


def test_cd_half_period_is_recorded_without_overriding_cl_phase_clock() -> None:
    states, weights, history = _periodic_arrays()
    time = np.arange(len(states), dtype=np.float64)
    history["CD"] = 0.3 + 0.04 * np.sin(4.0 * np.pi * time / 64.0)

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "PHASE_ONLY_SUPPORTED"
    assert extension is None
    assert (
        diagnostics["CD_harmonic_diagnostic"]["half_period_harmonic_detected"] is True
    )
    assert diagnostics["CD_harmonic_diagnostic"]["can_change_CL_period"] is False


def test_slow_modulation_rejects_phase_only_population() -> None:
    time = np.arange(1501, dtype=np.float64)
    envelope = 1.0 + 0.25 * np.sin(2.0 * np.pi * time / (11.0 * 64.0))
    states, weights, history = _periodic_arrays(modulation=envelope)

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "MODULATED_LOW_DIMENSIONAL"
    assert extension is None
    assert diagnostics["period"]["coherent"] is True
    assert diagnostics["stationarity"]["passed"] is False


def test_insufficient_cycles_requests_frozen_known_period_extension() -> None:
    states, weights, history = _periodic_arrays(period=250)

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "INSUFFICIENT_CYCLES"
    assert diagnostics["period"]["period_steps"] == pytest.approx(250.0, rel=2e-2)
    assert extension == {
        "reason": "fewer_than_eight_complete_postburn_cycles",
        "rule": "known_period_append_max_1000_or_four_periods_rounded_to_500",
        "current_final_output_index": 1999,
        "append_steps": 1000,
        "requested_final_output_index": 2999,
        "same_accumulated_prefix_required": True,
        "model_blind": True,
    }


def test_unresolved_boundary_requests_deterministic_1500_step_extension() -> None:
    states, weights, history = _periodic_arrays(period=400)

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "INSUFFICIENT_CYCLES"
    assert diagnostics["period"]["boundary_limited"] is True
    assert extension is not None
    assert extension["append_steps"] == 1500
    assert extension["requested_final_output_index"] == 3499


def test_unresolved_nonboundary_period_still_requests_extension() -> None:
    states, weights, history = _periodic_arrays()
    history["CL"] = np.random.default_rng(932).standard_normal(len(states))

    outcome, diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert diagnostics["period"]["coherent"] is False
    assert diagnostics["period"]["boundary_limited"] is False
    assert diagnostics["period"]["unresolved_evidence"] is True
    assert outcome == "INSUFFICIENT_CYCLES"
    assert extension is not None
    assert extension["append_steps"] == 1500


def test_genuinely_incoherent_period_does_not_request_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states, weights, history = _periodic_arrays()
    monkeypatch.setattr(
        pilot,
        "_estimate_period",
        lambda _signal, _contract: {
            "coherent": False,
            "boundary_limited": False,
            "unresolved_evidence": False,
            "failure": "ACF_and_spectral_periods_disagree",
        },
    )

    outcome, _diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "NO_COHERENT_PERIOD"
    assert extension is None


def test_improving_transient_uses_known_period_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states, weights, history = _periodic_arrays(period=250)
    monkeypatch.setattr(
        pilot,
        "_estimate_period",
        lambda _signal, _contract: {
            "coherent": True,
            "period_steps": 250.0,
            "crossing_offsets": [0.0, 250.0],
        },
    )
    monkeypatch.setattr(pilot, "_alias_scan", lambda _states, _period: {"passed": True})
    monkeypatch.setattr(
        pilot,
        "_stationarity",
        lambda _states, _history, _crossings, _period: {
            "passed": False,
            "available_complete_cycles": 8,
            "late_metrics_improve_monotonically": True,
        },
    )

    outcome, _diagnostics, extension = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )

    assert outcome == "TRANSIENT_NOT_SETTLED"
    assert extension is not None
    assert extension["rule"] == (
        "known_period_append_max_1000_or_four_periods_rounded_to_500"
    )
    assert extension["requested_final_output_index"] == 2999


def test_one_field_and_gated_view_failure_cannot_be_rescued_by_area_view() -> None:
    period = 32
    cycle_count = 9
    time = np.arange(cycle_count * period + 1, dtype=np.float64)
    phase = 2.0 * np.pi * time / period
    states = np.zeros((len(time), 4, 5), dtype=np.float64)
    states[:] = np.arange(5, dtype=np.float64)
    states += 0.1 * np.sin(phase)[:, None, None]
    for cycle in range(cycle_count):
        start = cycle * period
        stop = (cycle + 1) * period
        states[start:stop, 0, 4] += 0.15 * (-1.0) ** cycle
    cycles = [(float(k * period), float((k + 1) * period)) for k in range(cycle_count)]
    weights = np.vstack(
        (
            np.ones(4),
            np.ones(4),
            np.asarray([1.0e-10, 1.0, 1.0, 1.0]),
        )
    )

    recurrence = pilot._cycle_recurrence(states, cycles, weights)

    assert recurrence["views"]["uniform_node"]["passed"] is False
    assert recurrence["views"]["uniform_node"]["q90_each_field"]["Nu_Tilde"] > 0.15
    assert recurrence["views"]["physical_vertex_area"]["passed"] is True
    assert recurrence["passed"] is False
    assert recurrence["physical_vertex_area_can_qualify_alone"] is False


def test_weighted_recurrence_matches_hand_calculation() -> None:
    values = np.asarray(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )
    weights = np.asarray([1.0, 3.0])
    expected = math.sqrt((1.0 + 3.0 * 4.0 + 9.0 + 3.0 * 16.0) / (2.0 * 4.0))

    observed = pilot._weighted_rms(values, weights)

    assert observed.tolist() == pytest.approx([expected])


def test_alias_scan_component_normalization_and_zero_variation_field() -> None:
    time = np.arange(1501, dtype=np.float64)
    phase = 2.0 * np.pi * time / 64.0
    amplitudes = np.asarray([1.0e-3, 1.0, 10.0, 1.0e3])
    states = np.empty((len(time), 3, 5), dtype=np.float64)
    for field, amplitude in enumerate(amplitudes):
        states[:, :, field] = amplitude * np.sin(phase)[:, None]
    states[:, :, 4] = 7.0

    result = pilot._alias_scan(states, 64.0)

    assert result["passed"] is True
    assert result["period_minimizer_steps"] == 64
    assert all(math.isfinite(value) for value in result["scan"].values())
    normalized_scales = [
        result["normalization"][name]["uniform_node_RMS_temporal_variation"] / amplitude
        for name, amplitude in zip(
            pilot.NACA_DYNAMIC_FIELDS[:4], amplitudes, strict=True
        )
    ]
    assert normalized_scales == pytest.approx([normalized_scales[0]] * 4)
    assert result["normalization"]["Nu_Tilde"] == {
        "uniform_node_RMS_temporal_variation": 0.0,
        "zero_variation_contributes_zero": True,
    }
    assert set(result["component_scan"]["Nu_Tilde"].values()) == {0.0}


def test_malformed_packet_fails_closed_with_atomic_self_hashed_receipt(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    receipt_path = packet / "trajectory_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": NACA_TRAJECTORY_RECEIPT_SCHEMA,
                "canonical_payload_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    mesh = packet / "unsteady_naca0012_mesh.su2"
    mesh.write_text("malformed\n", encoding="utf-8")
    history = packet / "history_00499.csv"
    history.write_text("malformed\n", encoding="utf-8")
    output = tmp_path / pilot.NACA_PHASE_PILOT_RECEIPT_FILENAME

    receipt = pilot.analyze_su2_naca0012_phase_pilot(
        contract_path=CONTRACT_PATH.resolve(),
        trajectory_receipt_path=receipt_path.resolve(),
        mesh_path=mesh.resolve(),
        history_path=history.resolve(),
        output_path=output.resolve(),
    )

    assert receipt["outcome"] == "INVALID_ARTIFACT"
    assert receipt["native_solver_executed"] is False
    assert receipt["PCNO_inputs_consumed"] is False
    assert receipt["extension_request"] is None
    assert output.is_file()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    verified = pilot.load_verified_phase_pilot_receipt(output)
    assert verified == receipt


def test_source_end_inventory_failure_still_emits_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    receipt_path = packet / "trajectory_receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    mesh = packet / "mesh.su2"
    mesh.write_text("invalid\n", encoding="utf-8")
    history = packet / "history_00499.csv"
    history.write_text("invalid\n", encoding="utf-8")
    output = tmp_path / pilot.NACA_PHASE_PILOT_RECEIPT_FILENAME
    start_inventory = pilot._source_inventory()
    calls = 0

    def inventory() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return start_inventory
        raise OSError("synthetic end-inventory failure")

    monkeypatch.setattr(pilot, "_source_inventory", inventory)

    receipt = pilot.analyze_su2_naca0012_phase_pilot(
        contract_path=CONTRACT_PATH.resolve(),
        trajectory_receipt_path=receipt_path.resolve(),
        mesh_path=mesh.resolve(),
        history_path=history.resolve(),
        output_path=output.resolve(),
    )

    assert calls == 2
    assert receipt["outcome"] == "INVALID_ARTIFACT"
    assert "synthetic end-inventory failure" in receipt["error"]
    assert "inventory_error" in receipt["provenance_bracket"]["source_at_end"]
    assert pilot.load_verified_phase_pilot_receipt(output) == receipt


def test_accumulated_prefix_analysis_is_incrementally_equivalent() -> None:
    states, weights, history = _periodic_arrays()
    split = 733
    accumulated_states = np.concatenate((states[:split], states[split:]), axis=0)
    accumulated_history = {
        key: np.concatenate((value[:split], value[split:]))
        for key, value in history.items()
    }

    direct = pilot._analyze_arrays(
        states=states,
        weights=weights,
        history=history,
        contract=_contract(),
        last_index=1999,
    )
    incremental = pilot._analyze_arrays(
        states=accumulated_states,
        weights=weights,
        history=accumulated_history,
        contract=_contract(),
        last_index=1999,
    )

    assert incremental == direct


def test_packet_end_bracket_detects_restart_mutation(tmp_path: Path) -> None:
    receipt = tmp_path / "trajectory_receipt.json"
    mesh = tmp_path / "mesh.su2"
    history = tmp_path / "history_00499.csv"
    restart = tmp_path / "trajectory_flow_00499.dat"
    for path, content in (
        (receipt, b"receipt"),
        (mesh, b"mesh"),
        (history, b"history"),
        (restart, b"restart-before"),
    ):
        path.write_bytes(content)
    restart_record = {"index": 499, **_record(restart)}
    ordered = sha256(pilot._canonical_bytes({"files": [restart_record]})).hexdigest()
    storage_path = tmp_path / "trajectory_storage_manifest.json"
    storage = _write_self_hashed(
        storage_path,
        {
            "schema": pilot.NACA_TRAJECTORY_STORAGE_SCHEMA,
            "files": [restart_record],
        },
    )
    packet = {
        "trajectory_receipt": _record(receipt),
        "storage_manifest": _record(storage_path),
        "storage_manifest_payload_sha256": storage["canonical_payload_sha256"],
        "mesh": _record(mesh),
        "history": _record(history),
        "output_count": 1,
        "ordered_restart_records_sha256": ordered,
    }
    pilot._verify_packet_end(tmp_path, packet)

    restart.write_bytes(b"restart-after")

    with pytest.raises(ValueError, match="end-bracket restart 499"):
        pilot._verify_packet_end(tmp_path, packet)


def test_extension_is_capped_and_never_wraps_the_resource_limit() -> None:
    assert (
        pilot._extension_request(last_index=6499, period=250.0, reason="test") is None
    )
    request = pilot._extension_request(last_index=5999, period=None, reason="test")
    assert request is not None
    assert request["requested_final_output_index"] == 6499
    assert request["append_steps"] == 500


def test_cli_help_runs_from_outside_repository(tmp_path: Path) -> None:
    script = (
        REPOSITORY / "scripts/time_dependent_no/analyze_su2_naca0012_phase_pilot.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--trajectory-receipt" in completed.stdout

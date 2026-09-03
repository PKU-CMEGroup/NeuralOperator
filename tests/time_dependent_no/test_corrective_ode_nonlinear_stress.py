from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    PROVENANCE_SNAPSHOT_NAME,
    NonlinearStressConfig,
    _common_prefix_errors,
    build_training_contract,
    classify_scientific_outcome,
    evaluate_forced_rollouts,
    evaluate_response_bank,
    flows,
    forcing_bank,
    heldout_phases,
    qualify_trusted_solver,
    rollout_with_forcing,
    run_study,
    train_models,
    verify_packet,
)


@contextmanager
def _runtime_directory(name: str):
    root = Path("artifacts/time_dependent_no").resolve()
    path = root / f"pytest_corrective_ode_nonlinear_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != root:
            raise AssertionError(
                "refusing to clean a test path outside the artifact root"
            )
        shutil.rmtree(resolved)


def _tiny_config(**overrides) -> NonlinearStressConfig:
    values = {
        "train_phases": 16,
        "hidden_width": 8,
        "hidden_layers": 1,
        "training_steps": 2,
        "batch_size": 8,
        "seeds": (5,),
        "noise_steps": 3,
        "noise_phases": 8,
        "noise_sequences": 2,
        "min_common_prefix_steps": 1,
    }
    values.update(overrides)
    return NonlinearStressConfig(**values).validated()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest_and_closeout(output: Path, manifest: dict[str, object]) -> None:
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    closeout_path = output / "closeout.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    closeout["manifest_sha256"] = _sha256(manifest_path)
    _write_json(closeout_path, closeout)


def test_trusted_solver_closes_invariance_response_and_regime_signs() -> None:
    config = NonlinearStressConfig()
    qualification = qualify_trusted_solver(config)
    assert qualification["status"] == "pass", qualification
    assert qualification["structural"]["common_clean_successor_bytes"]
    assert qualification["structural"]["sampled_discrete_map_safety_pass"]
    assert qualification["structural"]["trusted_local_secant_order_pass"]
    cycle_logs = {
        key: row["exact_full_cycle_log_gain"]
        for key, row in qualification["regimes"].items()
    }
    assert cycle_logs["C"] < 0.0
    assert cycle_logs["M"] == 0.0
    assert cycle_logs["E"] > 0.0
    local_secant_means = qualification["structural"][
        "trusted_clean_reference_local_secant_log_gain_means"
    ]
    assert local_secant_means["C"] < -0.5
    assert abs(local_secant_means["M"]) <= 0.5
    assert local_secant_means["E"] > 0.5
    assert local_secant_means["C"] < local_secant_means["M"] < local_secant_means["E"]
    assert all(
        row["discrete_map_32_safety_pass"]
        and row["discrete_map_64_safety_pass"]
        and row["trusted_local_secant_threshold_pass"]
        for row in qualification["regimes"].values()
    )
    assert (
        max(
            row["max_32_vs_64_lifted_difference"]
            for row in qualification["regimes"].values()
        )
        <= 1.0e-8
    )

    phase = heldout_phases(config)
    clean = np.column_stack((phase, np.zeros_like(phase)))
    outputs = [flow.advance(clean) for flow in flows(config).values()]
    assert all(np.array_equal(outputs[0], value) for value in outputs[1:])
    assert np.max(np.abs(outputs[0][:, 1])) == 0.0


def test_matched_training_contract_uses_shared_inputs_and_intended_targets() -> None:
    config = _tiny_config()
    datasets, metadata = build_training_contract(config)
    assert set(datasets) == {"CLEAN", "RECOVERY", "DYN_C", "DYN_M", "DYN_E"}
    assert len({values[0].shape[0] for values in datasets.values()}) == 1
    recovery_inputs, recovery_targets = datasets["RECOVERY"]
    for regime in ("C", "M", "E"):
        dynamic_inputs, dynamic_targets = datasets[f"DYN_{regime}"]
        assert np.array_equal(recovery_inputs, dynamic_inputs)
        displaced = dynamic_inputs[metadata["clean_rows_per_arm"] :]
        assert dynamic_targets[metadata["clean_rows_per_arm"] :] == pytest.approx(
            flows(config)[regime].advance(displaced)
        )
    clean_targets = datasets["CLEAN"][1][: metadata["clean_rows_per_arm"]]
    assert recovery_targets[: metadata["clean_rows_per_arm"]] == pytest.approx(
        clean_targets
    )
    assert recovery_targets[metadata["clean_rows_per_arm"] :] == pytest.approx(
        clean_targets
    )
    assert metadata["input_digests"]["RECOVERY"] == metadata["input_digests"]["DYN_E"]


def test_forcing_is_antithetic_shared_and_applied_after_complete_map() -> None:
    config = _tiny_config()
    initial, forcing, digest = forcing_bank(config, config.primary_noise_sigma)
    assert digest
    shaped = forcing[..., 1].reshape(
        config.noise_steps, config.noise_phases, config.noise_sequences
    )
    assert np.sum(shaped, axis=2) == pytest.approx(0.0)

    def step(state: np.ndarray) -> np.ndarray:
        result = state.copy()
        result[:, 0] += 0.2
        result[:, 1] = 2.0 * state[:, 1] + 0.01
        return result

    values = rollout_with_forcing(step, initial, forcing)
    assert values[1, :, 0] == pytest.approx(initial[:, 0] + 0.2)
    assert values[1, :, 1] == pytest.approx(0.01 + forcing[0, :, 1])
    assert values[2, :, 1] == pytest.approx(
        2.0 * values[1, :, 1] + 0.01 + forcing[1, :, 1]
    )


def test_rollout_retains_nonfinite_states_and_common_prefix_uses_truth_exit() -> None:
    initial = np.zeros((2, 2), dtype=np.float64)
    forcing = np.zeros((3, 2, 2), dtype=np.float64)
    calls = {"count": 0}

    def step(state: np.ndarray) -> np.ndarray:
        result = state.copy()
        result[:, 0] += 0.1
        result[:, 1] += 0.01
        if calls["count"] == 0:
            result[1, 1] = np.inf
        calls["count"] += 1
        return result

    values = rollout_with_forcing(step, initial, forcing)
    assert np.isfinite(values[:, 0, :]).all()
    assert np.isinf(values[1, 1, 1])
    assert np.isnan(values[2:, 1, :]).all()

    clean_reference = np.zeros((4, 2, 2), dtype=np.float64)
    forced_reference = clean_reference.copy()
    forced_reference[1:, 0, 1] = 0.2
    first = clean_reference.copy()
    second = clean_reference.copy()
    first[1:, 0, 0] = 5.0
    second[1:, 0, 0] = 7.0
    first[1:, 1, 0] = 0.1
    second[1:, 1, 0] = 0.2

    prefix = _common_prefix_errors(
        first,
        second,
        clean_reference,
        forced_reference,
        tube_radius=0.15,
        minimum_steps=2,
    )
    assert prefix["first_clean_error"] == pytest.approx(0.1)
    assert prefix["first_forced_error"] == pytest.approx(0.1)
    assert prefix["second_clean_error"] == pytest.approx(0.2)
    assert prefix["second_forced_error"] == pytest.approx(0.2)
    assert prefix["common_prefix_trajectory_coverage"] == pytest.approx(0.5)
    assert 0.0 < prefix["common_prefix_fraction"] < 1.0
    assert prefix["common_prefix_median_steps"] == pytest.approx(1.5)
    assert len(prefix["unit_rows"]) == 2
    assert prefix["unit_rows"][0]["first_forced_reference_exit_step"] == 1
    assert prefix["common_prefix_required_coverage_pass"] is False


def test_tiny_training_preserves_pairing_and_emits_effective_response_fields() -> None:
    config = _tiny_config()
    with _runtime_directory("training") as runtime:
        models, rows, paired = train_models(
            config, checkpoint_dir=runtime / "checkpoints"
        )
        assert set(models[5]) == {"CLEAN", "RECOVERY", "DYN_C", "DYN_M", "DYN_E"}
        assert len(rows) == 5
        assert len({row["initial_parameter_digest"] for row in rows}) == 1
        assert len({row["batch_plan_digest"] for row in rows}) == 1
        assert all(math.isfinite(row["final_loss"]) for row in rows)
        assert paired["seeds"]["5"]["arms"].keys() == models[5].keys()
        metrics, queries, profiles, checks = evaluate_response_bank(models, config)
        assert len(metrics) == 9
        assert queries and profiles
        assert all("trusted_response_defect_rms" in row for row in metrics)
        assert all("normalized_inner_flow_defect" in row for row in metrics)
        assert all("normal_response_magnitude_rms" in row for row in metrics)
        assert all(
            "learned_clean_reference_local_secant_log_gain_mean" in row
            for row in metrics
        )
        assert "clean_trace_checks" in checks
        assert "local_secant_checks" in checks
        assert "response_qualification_pass" in checks
        (
            curve_rows,
            endpoint_rows,
            unit_rows,
            prefix_unit_rows,
            rollout_checks,
        ) = evaluate_forced_rollouts(models, config, training_rows=rows)
        assert curve_rows and endpoint_rows and unit_rows and prefix_unit_rows
        learned_endpoint = next(
            row for row in endpoint_rows if row["seed"] == 5 and row["arm"] == "DYN"
        )
        assert learned_endpoint["source_arm"].startswith("DYN_")
        assert learned_endpoint["checkpoint"].endswith(".pt")
        for key in (
            "checkpoint_sha256",
            "parameter_digest",
            "trajectory_digest",
            "clean_reference_digest",
            "forced_reference_digest",
            "clean_phase_unwrapped_error_auc",
            "pre_exit_clean_phase_unwrapped_error_rms",
            "final_nonfinite_fraction",
            "median_first_nonfinite_step",
            "mean_finite_fraction",
            "mean_tube_residence_fraction",
        ):
            assert key in learned_endpoint
        assert 0.0 <= learned_endpoint["final_nonfinite_fraction"] <= 1.0
        learned_unit = next(
            row for row in unit_rows if row["seed"] == 5 and row["arm"] == "DYN"
        )
        for key in (
            "trajectory_index",
            "phase_index",
            "forcing_sequence_index",
            "clean_phase_unwrapped_error_auc",
            "pre_exit_clean_phase_unwrapped_error_rms",
            "first_tube_crossing_step",
            "first_exit_step",
            "first_safety_exit_step",
            "first_nonfinite_step",
            "tube_residence_steps",
            "tube_residence_fraction",
        ):
            assert key in learned_unit
        expected_sigmas = {str(value) for value in config.noise_sigmas if value > 0.0}
        assert set(rollout_checks["P5_by_sigma"]) == expected_sigmas
        assert rollout_checks["shared_arm_checks"]
        assert all(
            row["trajectory_digest_reused"]
            and row["checkpoint_sha256_reused"]
            and row["parameter_digest_reused"]
            for row in rollout_checks["shared_arm_checks"].values()
        )
        assert all(
            "common_prefix_required_coverage_pass" in row
            for row in rollout_checks["common_prefix_rows"]
        )
        assert rollout_checks["P3_metric"] == "mean_tube_residence_fraction"
        assert all("common_prefix_steps" in row for row in prefix_unit_rows)


def test_response_magnitude_uses_only_the_normal_component() -> None:
    class PhaseOnlyResponse(torch.nn.Module):
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            result = state.clone()
            result[:, 0] = state[:, 0] + 0.2 + 3.0 * state[:, 1]
            result[:, 1] = 0.0
            return result

    config = _tiny_config()
    model = PhaseOnlyResponse()
    models = {
        5: {
            "CLEAN": model,
            "RECOVERY": model,
            "DYN_C": model,
            "DYN_M": model,
            "DYN_E": model,
        }
    }
    metrics, _, profiles, _ = evaluate_response_bank(models, config)
    assert all(
        row["normal_response_magnitude_rms"] == pytest.approx(0.0) for row in metrics
    )
    assert any(abs(row["learned_phase_response"]) > 1.0 for row in profiles)


def test_scientific_classification_is_hierarchical() -> None:
    checks = {
        "P1_target_realization": True,
        "P2_response_regime_ordering": True,
        "P3_dynamic_regime_interaction": True,
        "P3_recovery_beats_dynamic_expansive": True,
        "P4_dual_target_tradeoff": True,
        "P5_contractive_control": True,
    }
    assert (
        classify_scientific_outcome(checks, response_qualification_pass=True)
        == "all_registered_predictions_supported"
    )
    partial = {**checks, "P5_contractive_control": False}
    assert (
        classify_scientific_outcome(partial, response_qualification_pass=True)
        == "registered_predictions_partially_supported"
    )
    rollout_falsified = {
        **checks,
        "P3_dynamic_regime_interaction": False,
        "P3_recovery_beats_dynamic_expansive": False,
        "P4_dual_target_tradeoff": False,
        "P5_contractive_control": False,
    }
    assert (
        classify_scientific_outcome(rollout_falsified, response_qualification_pass=True)
        == "registered_predictions_falsified"
    )
    rollout_positive_but_unqualified = {
        **checks,
        "P1_target_realization": False,
    }
    assert (
        classify_scientific_outcome(
            rollout_positive_but_unqualified,
            response_qualification_pass=False,
        )
        == "response_qualification_failed"
    )
    with pytest.raises(ValueError, match="conflicts"):
        classify_scientific_outcome(
            rollout_positive_but_unqualified,
            response_qualification_pass=True,
        )


def test_tiny_packet_verifies_and_rejects_tampering() -> None:
    config = _tiny_config(training_steps=1)
    with _runtime_directory("packet") as runtime:
        output = runtime / "result"
        result = run_study(output_dir=output, config=config)
        json.dumps(result, allow_nan=False)
        assert result["verification"]["status"] == "verified"
        assert result["verification"]["canonical_contract"] is False
        assert (output / "nonlinear_ode_regime_stress.pdf").is_file()
        assert (output / "rollout_units.csv").is_file()
        assert (output / "common_prefix_units.csv").is_file()
        learned_summary = json.loads(
            (output / "learned_summary.json").read_text(encoding="utf-8")
        )
        assert "shared_arm_checks" in learned_summary["rollout_checks"]
        assert "P5_by_sigma" in learned_summary["rollout_checks"]
        packet_only = verify_packet(output, verify_current_sources=False)
        assert packet_only["status"] == "verified"
        extra_path = output / "unbound.txt"
        extra_path.write_text("unbound\n", encoding="utf-8")
        with pytest.raises(ValueError, match="output inventory mismatch"):
            verify_packet(output)
        extra_path.unlink()
        summary_path = output / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="output hash mismatch"):
            verify_packet(output)


def test_packet_rejects_source_inventory_hash_and_anchor_mismatches() -> None:
    config = _tiny_config(training_steps=1)
    with _runtime_directory("provenance") as runtime:
        baseline = runtime / "baseline"
        result = run_study(output_dir=baseline, config=config)
        assert result["verification"]["sources"] == 5

        missing_source = runtime / "missing_source"
        shutil.copytree(baseline, missing_source)
        manifest = json.loads(
            (missing_source / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["sources"].pop("utility/time_dependent_no/pcno_artifacts.py")
        _rewrite_manifest_and_closeout(missing_source, manifest)
        with pytest.raises(ValueError, match="source inventory mismatch"):
            verify_packet(missing_source)

        source_hash = runtime / "source_hash"
        shutil.copytree(baseline, source_hash)
        manifest = json.loads(
            (source_hash / "manifest.json").read_text(encoding="utf-8")
        )
        snapshot_path = source_hash / PROVENANCE_SNAPSHOT_NAME
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        source_name = "utility/time_dependent_no/pcno_artifacts.py"
        manifest["sources"][source_name]["sha256"] = "0" * 64
        snapshot["sources"][source_name]["sha256"] = "0" * 64
        _write_json(snapshot_path, snapshot)
        manifest["outputs"][PROVENANCE_SNAPSHOT_NAME] = {
            "sha256": _sha256(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
        }
        _rewrite_manifest_and_closeout(source_hash, manifest)
        with pytest.raises(ValueError, match="current source mismatch"):
            verify_packet(source_hash)

        anchor = runtime / "anchor"
        shutil.copytree(baseline, anchor)
        manifest = json.loads((anchor / "manifest.json").read_text(encoding="utf-8"))
        manifest["compatibility_anchors"]["affine_packet"]["actual_manifest_sha256"] = (
            "0" * 64
        )
        _rewrite_manifest_and_closeout(anchor, manifest)
        with pytest.raises(ValueError, match="recorded compatibility anchor mismatch"):
            verify_packet(anchor)

from __future__ import annotations

import csv
import json
import math
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no.run_corrective_ode_gaussian_normal_noise import (
    AUDIT_SCHEMA,
    GaussianNormalNoiseConfig,
    _forcing_preflight,
    _gaussian_log_density,
    _profile_signature,
    _reviewed_source_hashes,
    _source_records,
    build_training_arrays,
    build_training_schedule,
    prepare_schedules,
    qualify_scale_response,
    run_study,
    schedule_record,
    spearman_correlation,
    summarize_gaussian_tail,
    train_models,
    validate_audit_record,
    verify_packet,
)
from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    NonlinearStressConfig,
    flows,
)


@contextmanager
def _runtime_path(name: str):
    artifact_root = Path("artifacts/time_dependent_no").resolve()
    path = artifact_root / f"pytest_corrective_ode_gaussian_{name}_{uuid.uuid4().hex}"
    if path.exists():
        raise AssertionError("test runtime path unexpectedly exists")
    try:
        yield path
    finally:
        if path.exists():
            resolved = path.resolve()
            if resolved.parent != artifact_root:
                raise AssertionError("refusing cleanup outside the artifact root")
            shutil.rmtree(resolved)


def _tiny_config(
    *,
    train_noise_stds: tuple[float, ...] = (0.005, 0.01),
    training_steps: int = 2,
) -> GaussianNormalNoiseConfig:
    base = NonlinearStressConfig(
        train_phases=16,
        hidden_width=8,
        hidden_layers=1,
        batch_size=128,
        training_steps=training_steps,
        seeds=(5,),
        noise_steps=3,
        noise_phases=8,
        noise_sequences=2,
        min_common_prefix_steps=1,
    )
    return GaussianNormalNoiseConfig(
        base=base, train_noise_stds=train_noise_stds
    ).validated()


def test_canonical_forcing_bank_matches_frozen_parent() -> None:
    records = _forcing_preflight(GaussianNormalNoiseConfig(), require_parent_match=True)
    assert set(records) == {"0", "0.001", "0.005"}
    assert all(row["parent_match_required"] for row in records.values())
    assert all(row["parent_forcing_sign_digest_match"] for row in records.values())


def test_normal_only_schedule_mixture_targets_and_zero_scale_collapse() -> None:
    config = _tiny_config()
    assert GaussianNormalNoiseConfig().expected_checkpoint_count == 51
    schedule = build_training_schedule(config, 5)
    assert schedule["clean_phase_indices"].shape == (2, 64)
    assert schedule["noisy_phase_indices"].shape == (2, 64)
    assert schedule["standard_normal"].shape == (2, 64)
    assert not np.array_equal(
        schedule["standard_normal"][0], schedule["standard_normal"][1]
    )
    assert schedule_record(schedule) == schedule_record(
        build_training_schedule(config, 5)
    )

    clean_inputs, clean_targets = build_training_arrays(
        config, schedule, sigma_train=0.0, arm="CLEAN"
    )
    recovery_inputs, recovery_targets = build_training_arrays(
        config, schedule, sigma_train=0.005, arm="RECOVERY"
    )
    dynamic_inputs, dynamic_targets = build_training_arrays(
        config, schedule, sigma_train=0.005, arm="DYN_E"
    )
    assert clean_inputs.shape == (2, 128, 2)
    assert np.count_nonzero(clean_inputs[..., 1]) == 0
    assert np.count_nonzero(recovery_inputs[:, :64, 1]) == 0
    assert recovery_inputs[:, 64:, 1] == pytest.approx(
        0.005 * schedule["standard_normal"]
    )
    assert np.array_equal(recovery_inputs, dynamic_inputs)
    assert np.array_equal(recovery_inputs[..., 0], clean_inputs[..., 0])

    clean_projection = recovery_inputs.copy()
    clean_projection[..., 1] = 0.0
    expected_recovery = (
        flows(config.base)["C"]
        .advance(clean_projection.reshape(-1, 2))
        .reshape(recovery_targets.shape)
    )
    assert recovery_targets == pytest.approx(expected_recovery)
    expected_dynamic_noisy = (
        flows(config.base)["E"]
        .advance(dynamic_inputs[:, 64:, :].reshape(-1, 2))
        .reshape(dynamic_targets[:, 64:, :].shape)
    )
    assert dynamic_targets[:, 64:, :] == pytest.approx(expected_dynamic_noisy)
    assert dynamic_targets[:, :64, :] == pytest.approx(recovery_targets[:, :64, :])

    zero_recovery = build_training_arrays(
        config, schedule, sigma_train=0.0, arm="RECOVERY"
    )
    zero_dynamic = build_training_arrays(config, schedule, sigma_train=0.0, arm="DYN_E")
    assert np.array_equal(zero_recovery[0], clean_inputs)
    assert np.array_equal(zero_recovery[1], clean_targets)
    assert np.array_equal(zero_dynamic[0], clean_inputs)
    assert np.array_equal(zero_dynamic[1], clean_targets)

    larger_inputs, _ = build_training_arrays(
        config, schedule, sigma_train=0.01, arm="RECOVERY"
    )
    assert larger_inputs[:, 64:, 1] == pytest.approx(2.0 * recovery_inputs[:, 64:, 1])


def test_unclipped_tail_summary_and_safety_failure() -> None:
    config = _tiny_config()
    schedule = build_training_schedule(config, 5)
    tail = summarize_gaussian_tail(config, schedule, seed=5, sigma_train=0.01)
    realized = np.abs(0.01 * schedule["standard_normal"])
    assert tail["max_abs_radius"] == pytest.approx(float(np.max(realized)))
    assert set(tail["abs_radius_quantiles"]) == {
        "0.5",
        "0.9",
        "0.95",
        "0.99",
        "0.999",
    }
    assert tail["unique_draw_count"] == realized.size
    assert tail["repeated_model_presentations"] == 4 * realized.size
    assert tail["clipped"] is False
    assert tail["rejected_or_resampled"] is False
    assert tail["theoretical_fraction_beyond_tube"] == pytest.approx(
        math.erfc(config.base.tube_radius / (math.sqrt(2.0) * 0.01))
    )

    unsafe = _tiny_config(train_noise_stds=(100.0,))
    with pytest.raises(ValueError, match="refusing clipping"):
        prepare_schedules(unsafe)


def test_profile_spearman_and_response_qualification_rules() -> None:
    config = _tiny_config(train_noise_stds=(0.005, 0.01, 0.02, 0.04))
    profile_rows = []
    for radius in config.profile_abs_radii:
        densities = [
            _gaussian_log_density(radius, sigma) for sigma in config.train_noise_stds
        ]
        maximum = max(densities)
        for sigma, density in zip(config.train_noise_stds, densities, strict=True):
            positive_error = maximum + 1.0 - density
            profile_rows.append(
                {
                    "seed": 5,
                    "regime": "",
                    "arm": "RECOVERY",
                    "sigma_train": sigma,
                    "radius_abs": radius,
                    "gaussian_log_density": density,
                    "clean_return_error_rms": positive_error,
                    "trusted_flow_defect_rms": None,
                }
            )
            for regime in ("C", "M", "E"):
                profile_rows.append(
                    {
                        "seed": 5,
                        "regime": regime,
                        "arm": "DYN",
                        "sigma_train": sigma,
                        "radius_abs": radius,
                        "gaussian_log_density": density,
                        "clean_return_error_rms": None,
                        "trusted_flow_defect_rms": positive_error,
                    }
                )
    assert _profile_signature(profile_rows, config, arm="RECOVERY")["signature_pass"]
    assert _profile_signature(profile_rows, config, arm="DYN")["signature_pass"]
    assert spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 2, 3, 4], [7, 7, 7, 7]) == 0.0

    response_rows = []
    secants = {"C": -0.8, "M": 0.0, "E": 0.8}
    for regime in ("C", "M", "E"):
        response_rows.extend(
            [
                {
                    "seed": 5,
                    "regime": regime,
                    "arm": "RECOVERY",
                    "trusted_response_defect_rms": 0.2,
                    "inner_return_error_rms": 0.01,
                    "normal_response_magnitude_rms": 0.1,
                    "clean_lifted_error_rms": 0.001,
                    "normalized_inner_flow_defect": 0.2,
                    "learned_clean_reference_local_secant_log_gain_mean": 0.0,
                },
                {
                    "seed": 5,
                    "regime": regime,
                    "arm": "DYN",
                    "trusted_response_defect_rms": 0.05,
                    "inner_return_error_rms": 0.1,
                    "normal_response_magnitude_rms": 0.5,
                    "clean_lifted_error_rms": 0.001,
                    "normalized_inner_flow_defect": 0.1,
                    "learned_clean_reference_local_secant_log_gain_mean": secants[
                        regime
                    ],
                },
            ]
        )
    qualified = qualify_scale_response(response_rows, config, sigma_train=0.005)
    assert qualified["rollout_qualified"]
    response_rows[-1]["normalized_inner_flow_defect"] = 0.151
    unqualified = qualify_scale_response(response_rows, config, sigma_train=0.005)
    assert not unqualified["rollout_qualified"]
    assert unqualified["rollout_interpretation"].startswith("descriptive_only")


def test_tiny_training_pairs_inputs_initialization_and_checkpoints() -> None:
    config = _tiny_config(training_steps=1)
    schedules, _ = prepare_schedules(config)
    with _runtime_path("training") as runtime:
        models, rows, paired = train_models(
            config, schedules, checkpoint_dir=runtime / "checkpoints"
        )
        assert len(rows) == config.expected_checkpoint_count == 9
        assert set(models) == {0.005, 0.01}
        assert set(models[0.005][5]) == {
            "CLEAN",
            "RECOVERY",
            "DYN_C",
            "DYN_M",
            "DYN_E",
        }
        assert len({row["initial_parameter_digest"] for row in rows}) == 1
        assert len({row["standard_normal_schedule_digest"] for row in rows}) == 1
        for sigma in config.train_noise_stds:
            inputs = {
                row["input_schedule_digest"]
                for row in rows
                if row["arm"] != "CLEAN" and row["sigma_train"] == sigma
            }
            assert len(inputs) == 1
        assert all(math.isfinite(row["final_schedule_mse"]) for row in rows)
        clean_row = next(row for row in rows if row["arm"] == "CLEAN")
        response_row = next(row for row in rows if row["arm"] == "RECOVERY")
        assert clean_row["clean_rows_per_update"] == 128
        assert clean_row["corrupted_rows_per_update"] == 0
        assert response_row["clean_rows_per_update"] == 64
        assert response_row["corrupted_rows_per_update"] == 64
        assert len(list((runtime / "checkpoints").glob("*.pt"))) == 9
        seed_record = paired["seeds"]["5"]
        assert (
            seed_record["initial_parameter_digest"]
            == rows[0]["initial_parameter_digest"]
        )


def test_noncanonical_smoke_packet_verifies_scale_names_and_refuses_tamper() -> None:
    config = _tiny_config(train_noise_stds=(0.005,), training_steps=1)
    source_records = _source_records()
    audit = {
        "schema": AUDIT_SCHEMA,
        "verdict": "AUDIT_PASS",
        "reviewed_sources": _reviewed_source_hashes(source_records),
    }
    validate_audit_record(audit, source_records)
    tampered_audit = json.loads(json.dumps(audit))
    tampered_audit["reviewed_sources"].pop(next(iter(source_records)))
    with pytest.raises(ValueError, match="exact current source set"):
        validate_audit_record(tampered_audit, source_records)

    with _runtime_path("packet") as output:
        result = run_study(
            output_dir=output,
            config=config,
            allow_noncanonical_smoke_without_audit=True,
        )
        assert result["verification"]["status"] == "verified"
        assert result["verification"]["canonical_contract"] is False
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["canonical_contract"] is False
        assert manifest["expected_checkpoint_count"] == 5
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["response_qualification_contract"] == "B2_GN_per_seed_scale"
        assert summary["inherited_b2_nl_scientific_classification_reused"] is False
        assert "response_qualification_by_scale" in summary
        assert "P1_recovery_density_fidelity" in summary["prediction_diagnostics"]

        with (output / "training.csv").open(encoding="utf-8", newline="") as handle:
            training_header = next(csv.reader(handle))
        with (output / "rollout_endpoints.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rollout_header = next(csv.reader(handle))
        assert "sigma_train" in training_header
        assert "sigma_force" not in training_header
        assert "sigma_force" in rollout_header
        assert "sigma" not in rollout_header
        assert "rollout_interpretation" in rollout_header
        assert (output / "radial_profile_metrics.csv").is_file()
        assert (output / "gaussian_scale_response.pdf").is_file()
        assert verify_packet(output)["checkpoint_count"] == 5

        summary["scientific_classification"] = "tampered"
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="output inventory or hash mismatch"):
            verify_packet(output)

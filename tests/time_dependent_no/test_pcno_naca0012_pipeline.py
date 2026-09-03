from __future__ import annotations

import inspect
import json
import math
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from scripts.time_dependent_no import evaluate_pcno_naca0012 as evaluator
from scripts.time_dependent_no import prepare_pcno_naca0012_dataset as preparer
from scripts.time_dependent_no import train_pcno_naca0012 as trainer
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    recurrent_step,
)


def _record(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _write_self_hashed(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = trainer._self_hashed(value)
    trainer._write_json(path, payload)
    return payload


def test_epoch_one_status_is_json_safe_before_first_selection() -> None:
    status = trainer._running_status_payload(
        seed=17,
        epoch=1,
        best_epoch=None,
        best_score=math.inf,
    )

    assert status["last_completed_epoch"] == 1
    assert status["best_epoch"] is None
    assert status["best_development_score"] is None
    json.dumps(status, allow_nan=False)

    selected = trainer._running_status_payload(
        seed=17,
        epoch=5,
        best_epoch=5,
        best_score=math.inf,
    )
    assert selected["best_development_score"] == "Infinity"
    json.dumps(selected, allow_nan=False)


def test_source_manifest_closure_includes_all_runtime_and_entrypoint_sources() -> None:
    assert preparer.SOURCE_FILES == (
        "pcno/__init__.py",
        "pcno/geo_utility.py",
        "pcno/pcno.py",
        "utility/__init__.py",
        "utility/adam.py",
        "utility/losses.py",
        "utility/normalizer.py",
        "utility/time_dependent_no/__init__.py",
        "utility/time_dependent_no/pcno_naca0012.py",
        "utility/time_dependent_no/su2_naca_phase_pilot.py",
        "utility/time_dependent_no/su2_naca_trajectory.py",
        "utility/time_dependent_no/su2_restart_contract.py",
        "utility/time_dependent_no/su2_native_replay.py",
        "scripts/time_dependent_no/prepare_pcno_naca0012_dataset.py",
        "scripts/time_dependent_no/train_pcno_naca0012.py",
        "scripts/time_dependent_no/evaluate_pcno_naca0012.py",
    )


def test_source_inventory_precedes_local_imports_and_binds_exact_bytes() -> None:
    module_source = Path(preparer.__file__).read_text(encoding="utf-8")
    capture_position = module_source.index(
        "_SOURCE_RECORDS_AT_IMPORT = _read_source_records()"
    )
    assert capture_position < module_source.index("import numpy as np")
    assert capture_position < module_source.index(
        "from utility.time_dependent_no.pcno_naca0012 import"
    )

    reader_source = inspect.getsource(preparer._read_source_records)
    assert ".read_bytes()" in reader_source
    assert ".stat(" not in reader_source
    assert "sha256_file" not in reader_source

    captured = preparer.captured_source_records()
    assert captured == trainer.SOURCE_RECORDS_AT_IMPORT
    assert captured is not trainer.SOURCE_RECORDS_AT_IMPORT
    assert tuple(captured) == preparer.SOURCE_FILES
    for relative, record in captured.items():
        payload = (preparer.REPO_ROOT / relative).read_bytes()
        assert record == {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }


def test_train_and_evaluate_compare_import_time_source_inventory() -> None:
    loading_source = inspect.getsource(trainer._load_dataset)
    evaluation_source = inspect.getsource(evaluator._reverify_live_source)

    assert 'source.get("files") != SOURCE_RECORDS_AT_IMPORT' in loading_source
    assert 'source.get("files") != SOURCE_RECORDS_AT_IMPORT' in evaluation_source
    assert "_canonical_sha256(SOURCE_RECORDS_AT_IMPORT)" in loading_source


def test_stage0_manifest_helper_binds_exact_verified_bytes(tmp_path: Path) -> None:
    path = tmp_path / "R0_SU2_NACA_RESOURCE_MANIFEST.json"
    payload = json.dumps(
        {"schema": "time_dependent_no.su2_naca0012_resources.v1"},
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(payload)
    expected_sha256 = sha256(payload).hexdigest()

    assert preparer._verified_resource_manifest_bytes(path, expected_sha256) == payload
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="verified parent evidence"):
        preparer._verified_resource_manifest_bytes(path, expected_sha256)

    materialize_source = inspect.getsource(preparer.materialize)
    assert "context.resource_manifest_sha256" in materialize_source
    assert materialize_source.count("_verified_resource_manifest_bytes(") >= 3
    assert '"resource_manifest_sha256": resource_manifest_sha256' in materialize_source


def test_entrypoints_use_verified_context_and_strict_dataset_factories() -> None:
    preparation_source = inspect.getsource(preparer.materialize)
    assert "validate_parent_evidence(" in preparation_source
    assert "build_naca_geometry(context)" in preparation_source
    assert "extract_verified_trajectory_state(" in preparation_source
    assert "extract_verified_diagnostic_state(" in preparation_source

    loading_source = inspect.getsource(trainer._load_dataset)
    assert "load_verified_naca_geometry(" in loading_source
    assert "build_naca_bdf2_dataset(" in loading_source
    assert trainer.OPEN_ROLES == ("train", "development")
    assert "for role in OPEN_ROLES:" in loading_source


def test_checkpoint_and_summary_numeric_encodings_compare_canonically() -> None:
    assert evaluator._same_json_number(math.inf, "Infinity")
    assert evaluator._same_json_number(-math.inf, "-Infinity")
    assert evaluator._same_json_number(math.nan, "NaN")
    assert evaluator._same_json_number(np.float64(1.25), 1.25)
    assert not evaluator._same_json_number("Infinity", "-Infinity")
    with pytest.raises(TypeError, match="boolean"):
        evaluator._same_json_number(True, 1.0)


def test_recursive_json_safety_encodes_all_nonfinite_scalars() -> None:
    value = {
        "finite": np.float64(1.5),
        "nested": [math.inf, (np.float32(-math.inf), {"nan": np.float64(math.nan)})],
    }

    safe = evaluator._json_safe(value)

    assert safe == {
        "finite": 1.5,
        "nested": ["Infinity", ["-Infinity", {"nan": "NaN"}]],
    }
    json.dumps(safe, allow_nan=False)


class _ConstantResidual(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: dict[str, torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        del geometry_batch, fourier_tensors
        assert features.dtype == torch.float32
        return torch.full(
            features.shape[:2] + (5,),
            0.5,
            dtype=features.dtype,
            device=features.device,
        )


def test_one_step_recurrence_stays_float32() -> None:
    normalization = NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.arange(1.0, 6.0),
        state_rms=np.ones(5),
    )
    previous = torch.zeros((1, 2, 5), dtype=torch.float32)
    current = torch.full((1, 2, 5), 2.0, dtype=torch.float32)
    geometry_batch = {"static_features": torch.zeros((1, 2, 6), dtype=torch.float32)}

    returned_current, next_state = recurrent_step(
        _ConstantResidual(),
        previous,
        current,
        geometry_batch,
        normalization,
    )

    expected = current + 0.5 * torch.arange(1.0, 6.0, dtype=torch.float32)
    assert returned_current is current
    assert next_state.dtype == torch.float32
    torch.testing.assert_close(next_state, expected.expand_as(current))


def test_dataset_final_manifest_binds_every_file_and_detects_mutation(
    tmp_path: Path,
) -> None:
    for index, name in enumerate(sorted(evaluator.EXPECTED_DATASET_FILES)):
        (tmp_path / name).write_bytes(f"artifact-{index}".encode())
    files = {
        name: _record(tmp_path / name, tmp_path)
        for name in sorted(evaluator.EXPECTED_DATASET_FILES)
    }
    final = {
        "schema": evaluator.FINAL_HASH_MANIFEST_SCHEMA,
        "contract_sha256": evaluator.BASELINE_CONTRACT_SHA256,
        "files": files,
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    final_path = tmp_path / "final_hash_manifest.json"
    final_path.write_text(json.dumps(final, sort_keys=True), encoding="utf-8")
    packet_sha256 = sha256(final_path.read_bytes()).hexdigest()

    evaluator._reverify_dataset_packet(tmp_path, packet_sha256)

    changed = tmp_path / min(evaluator.EXPECTED_DATASET_FILES)
    changed.write_bytes(b"mutated after verification")
    with pytest.raises(ValueError, match="changed"):
        evaluator._reverify_dataset_packet(tmp_path, packet_sha256)


def test_npy_snapshot_is_read_once_immutable_and_hash_checked(tmp_path: Path) -> None:
    path = tmp_path / "states.npy"
    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.save(path, original)
    record = _record(path, tmp_path)

    snapshot = trainer._load_npy_snapshot(tmp_path, record, path.name)
    np.save(path, np.full((3, 4), -1.0, dtype=np.float64))

    np.testing.assert_array_equal(snapshot, original)
    assert snapshot.flags.writeable is False
    with pytest.raises(ValueError, match="differs"):
        trainer._load_npy_snapshot(tmp_path, record, path.name)


def test_terminal_summaries_report_both_error_families_for_every_view() -> None:
    anchors = (10, 20)
    errors: dict[tuple[int, int, str, str], float] = {}
    relative_errors: dict[tuple[int, int, str, str], float] = {}
    for method_offset, method in enumerate(("pcno", "persistence"), start=1):
        for view_offset, view in enumerate(evaluator.VIEWS, start=1):
            for horizon in evaluator.HORIZONS:
                for anchor_offset, anchor in enumerate(anchors, start=1):
                    key = (anchor, horizon, view, method)
                    errors[key] = float(method_offset + view_offset + anchor_offset)
                    relative_errors[key] = errors[key] / 10.0

            summary = evaluator._terminal_summary(
                errors,
                relative_errors,
                anchors=anchors,
                method=method,
                view=view,
            )
            assert set(summary) == {str(value) for value in evaluator.HORIZONS}
            for record in summary.values():
                assert set(record) == {
                    "train_state_scale_error",
                    "component_balanced_relative_l2",
                }
                assert set(record["train_state_scale_error"]) == {
                    "median",
                    "q90_nearest_rank",
                    "maximum",
                }


def test_rollout_windows_are_aggregated_for_every_method_and_view() -> None:
    anchors = (10, 20)
    values: dict[tuple[int, int, str, str], float] = {}
    for method in ("pcno", "persistence"):
        for view in evaluator.VIEWS:
            for anchor in anchors:
                for horizon in range(1, 209):
                    values[(anchor, horizon, view, method)] = float(anchor + horizon)

            early = evaluator._window_summary(
                values,
                anchors=anchors,
                method=method,
                view=view,
                first=1,
                last=35,
            )
            late = evaluator._window_summary(
                values,
                anchors=anchors,
                method=method,
                view=view,
                first=174,
                last=208,
            )
            assert early["finite_anchor_count"] == len(anchors)
            assert late["finite_anchor_count"] == len(anchors)
            assert set(early["per_anchor"]) == {"10", "20"}
            assert set(late["all_anchor_summary_failed_windows_are_infinity"]) == {
                "median",
                "q90_nearest_rank",
                "maximum",
            }

    values[(10, 180, "uniform_node", "pcno")] = math.inf
    failed = evaluator._window_summary(
        values,
        anchors=anchors,
        method="pcno",
        view="uniform_node",
        first=174,
        last=208,
    )
    assert failed["per_anchor"]["10"] == "Infinity"
    assert failed["finite_anchor_count"] == 1


def test_one_step_summary_covers_both_methods_and_all_views() -> None:
    rows: list[dict[str, Any]] = []
    for method in ("pcno", "persistence"):
        for view in evaluator.VIEWS:
            for index in range(238):
                row: dict[str, Any] = {
                    "method": method,
                    "view": view,
                    "normalized_residual_rmse": float(index + 1),
                    "next_state_relative_l2": float(index + 1) / 100.0,
                }
                for field_offset, field in enumerate(evaluator.FIELDS):
                    row[f"normalized_residual_rmse_{field}"] = float(
                        index + field_offset + 1
                    )
                    row[f"next_state_relative_l2_{field}"] = (
                        float(index + field_offset + 1) / 100.0
                    )
                rows.append(row)

    summary = evaluator._one_step_method_view_summaries(rows)

    assert set(summary) == {"pcno", "persistence"}
    for method in summary.values():
        assert set(method["views"]) == set(evaluator.VIEWS)
        for view in method["views"].values():
            assert view["transition_count"] == 238
            assert set(view["normalized_residual_rmse_per_field"]) == set(
                evaluator.FIELDS
            )
            assert set(view["next_state_relative_l2_per_field"]) == set(
                evaluator.FIELDS
            )


def test_admissibility_reports_first_finite_positive_step_per_failure_mode() -> None:
    rows: list[dict[str, Any]] = []
    for method in ("pcno", "persistence"):
        for horizon in range(1, 7):
            rows.append(
                {
                    "method": method,
                    "anchor": 40,
                    "horizon": horizon,
                    "finite": horizon != 4,
                    "density_nonpositive_fraction": 0.1
                    if method == "pcno" and horizon in (3, 4)
                    else 0.0,
                    "internal_energy_nonpositive_fraction_among_valid_density": 0.1
                    if method == "pcno" and horizon == 5
                    else 0.0,
                    "nu_tilde_negative_fraction": 0.1
                    if method == "pcno" and horizon == 2
                    else 0.0,
                }
            )

    summary = evaluator._admissibility_first_occurrence(rows, (40,))

    assert summary["pcno"]["40"] == {
        "density_nonpositive_first_step": 3,
        "internal_energy_nonpositive_among_valid_density_first_step": 5,
        "nu_tilde_negative_first_step": 2,
        "invalid_density_fraction_reported_separately": True,
    }
    assert summary["persistence"]["40"] == {
        "density_nonpositive_first_step": None,
        "internal_energy_nonpositive_among_valid_density_first_step": None,
        "nu_tilde_negative_first_step": None,
        "invalid_density_fraction_reported_separately": True,
    }


@pytest.mark.parametrize(
    ("adequate_count", "phenotype_pass", "margin_pass", "expected"),
    (
        (2, False, False, "R0_INCONCLUSIVE_INADEQUATE_TRAINING"),
        (3, False, False, "R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE"),
        (3, False, True, "R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE"),
        (3, True, False, "R0_INCONCLUSIVE_NATIVE_REPLAY_MARGIN_FAILED"),
        (3, True, True, "R0_BASELINE_PHENOTYPE_SUPPORTED"),
    ),
)
def test_r0_decision_precedence(
    adequate_count: int,
    phenotype_pass: bool,
    margin_pass: bool,
    expected: str,
) -> None:
    assert (
        evaluator._r0_decision(
            adequate_count=adequate_count,
            phenotype_pass=phenotype_pass,
            margin_pass=margin_pass,
        )
        == expected
    )


def test_cpu_smoke_and_authorization_are_bound_to_cpu_execution(
    tmp_path: Path,
) -> None:
    dataset_manifest = {
        "canonical_payload_sha256": "dataset-payload",
        "source_manifest": {"source_set_sha256": "source-set"},
        "geometry": {"num_nodes": 7},
    }
    dataset_packet_sha256 = "dataset-final"
    binding = trainer._device_binding(torch.device("cpu"))
    smoke_path = tmp_path / "smoke.json"
    smoke = _write_self_hashed(
        smoke_path,
        {
            "schema": trainer.SMOKE_SCHEMA,
            "status": "succeeded",
            "contract_sha256": trainer.BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": "dataset-payload",
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": "source-set",
            "seed": 17,
            "effective_batch_size": 4,
            "microbatch_size": 4,
            "gradient_accumulation_steps": 1,
            "full_resolution_num_nodes": 7,
            "forward_backward_and_adamw_step_completed": True,
            "loss": 1.0,
            "gradient_norm_before_clip": 2.0,
            "scientific_hyperparameters_changed": False,
            "checkpoint_written": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "production_device_binding": binding,
        },
    )
    smoke_file_sha256 = sha256(smoke_path.read_bytes()).hexdigest()
    authorization_path = tmp_path / "authorization.json"
    _write_self_hashed(
        authorization_path,
        {
            "schema": trainer.AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "contract_sha256": trainer.BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": "dataset-payload",
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": "source-set",
            "resource_smoke_payload_sha256": smoke["canonical_payload_sha256"],
            "resource_smoke_file_sha256": smoke_file_sha256,
            "production_device_binding": binding,
            "allowed_actions": ["train_pcno_naca0012"],
            "authorized_seeds": list(trainer.SEEDS),
            "pcno_code_audited": True,
            "pcno_training_authorized": True,
            "prospective_opened": False,
            "sealed_opened": False,
            "authorized_by": "synthetic-test",
        },
    )
    arguments = SimpleNamespace(
        authorization=authorization_path,
        resource_smoke_receipt=smoke_path,
        microbatch_size=4,
        gradient_accumulation_steps=1,
    )

    loaded_smoke, loaded_authorization = trainer._validate_production_authorization(
        arguments,
        dataset_manifest,
        dataset_packet_sha256,
        torch.device("cpu"),
    )
    assert loaded_smoke["production_device_binding"] == binding
    assert loaded_authorization["production_device_binding"] == binding

    incompatible_path = tmp_path / "cuda-smoke.json"
    incompatible = dict(smoke)
    incompatible.pop("canonical_payload_sha256")
    incompatible["production_device_binding"] = {
        **binding,
        "device_type": "cuda",
    }
    _write_self_hashed(incompatible_path, incompatible)
    arguments.resource_smoke_receipt = incompatible_path
    with pytest.raises(PermissionError, match="does not authorize"):
        trainer._validate_production_authorization(
            arguments,
            dataset_manifest,
            dataset_packet_sha256,
            torch.device("cpu"),
        )


def test_staged_evaluation_writer_promotes_complete_packet(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"

    def writer(staging: Path) -> None:
        (staging / "marker.txt").write_text("complete", encoding="utf-8")

    evaluator._write_staged_evaluation_packet(output, writer)

    assert (output / "marker.txt").read_text(encoding="utf-8") == "complete"
    assert not (tmp_path / ".evaluation.staging").exists()


def test_staged_evaluation_writer_cleans_partial_failure(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"

    def writer(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("synthetic disk failure")

    with pytest.raises(
        evaluator.EvaluationOutputError,
        match="output packet write failed",
    ):
        evaluator._write_staged_evaluation_packet(output, writer)

    assert not output.exists()
    assert not (tmp_path / ".evaluation.staging").exists()


def test_evaluation_output_failure_is_classified_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    def fail(_arguments: Any) -> dict[str, Any]:
        raise evaluator.EvaluationOutputError("synthetic output failure")

    def capture(output: Path, classification: str, error: BaseException) -> None:
        observed.update(
            output=output,
            classification=classification,
            error_type=type(error).__name__,
        )

    monkeypatch.setattr(evaluator, "evaluate", fail)
    monkeypatch.setattr(evaluator, "_failure_receipt", capture)
    status = evaluator.main(
        [
            "--contract",
            str(tmp_path / "contract.json"),
            "--replay-qualification",
            str(tmp_path / "qualification.json"),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--training-dir",
            str(tmp_path / "seed17"),
            "--training-dir",
            str(tmp_path / "seed29"),
            "--training-dir",
            str(tmp_path / "seed43"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--device",
            "cpu",
        ]
    )

    assert status == 3
    assert observed["classification"] == "INFRASTRUCTURE_FAILURE"
    assert observed["error_type"] == "EvaluationOutputError"


@pytest.mark.parametrize("module", (preparer, trainer, evaluator))
def test_entry_parsers_expose_no_role_horizon_or_resume_knobs(module: Any) -> None:
    options = {
        option
        for action in module._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        option.startswith(("--role", "--horizon", "--resume")) for option in options
    )


@pytest.mark.parametrize("module", (preparer, trainer, evaluator))
def test_artifact_helpers_reject_symbolic_links(
    module: Any,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"bound bytes")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symbolic links unavailable: {error}")

    if module is trainer:
        with pytest.raises(ValueError, match="aliased"):
            trainer._verify_record(
                tmp_path,
                {
                    "relative_path": link.name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target.read_bytes()).hexdigest(),
                },
                link.name,
            )
    else:
        with pytest.raises(ValueError, match="aliased"):
            module._file_record(link, tmp_path)

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_pcno_naca0012_successor as evaluator
from utility.time_dependent_no.pcno_naca0012 import NACANormalization


def _write_self_hashed(path: Path, value: dict) -> tuple[dict, str]:
    payload = evaluator._self_hashed(value)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload, evaluator._file_sha256(path)


def _synthetic_prediction_inputs():
    anchors = list(range(8))
    method_result = {}
    for seed in evaluator.SEEDS:
        for rank, deployment in enumerate(evaluator.PRIMARY_DEPLOYMENTS, start=1):
            method_result[f"{deployment[0]}:{seed}:{deployment[1]}:{deployment[2]}"] = {
                "per_anchor": {
                    str(anchor): {
                        metric: float(rank) for metric in evaluator.PRIMARY_METRICS
                    }
                    for anchor in anchors
                }
            }
    arm_medians = {
        "CLEAN": 2.0,
        "IID_RECOVERY": 1.0,
        "ERROR_SUBSPACE_RECOVERY": 2.0,
        "DETACHED_PUSHFORWARD": 4.0,
    }
    offline_result = {
        f"{arm}:{seed}": {
            metric: {"median": median} for metric in evaluator.MEDIATOR_METRICS
        }
        for seed in evaluator.SEEDS
        for arm, median in arm_medians.items()
    }
    return method_result, offline_result, anchors


def _synthetic_predictions():
    method_result, offline_result, anchors = _synthetic_prediction_inputs()
    return evaluator._prediction_matrix(method_result, offline_result, anchors)


def _synthetic_prospective_control(tmp_path: Path):
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    digest = "a" * 64
    source = evaluator._source_snapshot()
    parent_bindings = {
        key: digest
        for key in (
            "inherited_r0_contract_sha256",
            "preregistration_sha256",
            "dataset_final_hash_manifest_sha256",
            "dataset_manifest_payload_sha256",
            "calibration_final_hash_manifest_sha256",
            "calibration_payload_sha256",
            "development_evaluation_final_hash_manifest_sha256",
            "development_scientific_files_sha256",
            "development_input_manifest_payload_sha256",
            "development_result_payload_sha256",
            "development_evaluator_source_set_sha256",
            "production_authority_set_sha256",
            "training_source_set_sha256",
            "trajectory_receipt_file_sha256",
            "trajectory_receipt_payload_sha256",
            "trajectory_storage_manifest_file_sha256",
            "trajectory_storage_manifest_payload_sha256",
            "ordered_restart_records_sha256",
        )
    }
    parent_bindings["training_packets"] = [
        {
            "arm": arm,
            "seed": seed,
            "final_hash_manifest_sha256": digest,
            "checkpoint_sha256": digest,
            "source_set_sha256": digest,
        }
        for arm in evaluator.LEARNED_ARMS
        for seed in evaluator.SEEDS
    ]
    freeze = evaluator._self_hashed(
        {
            "schema": evaluator.PROSPECTIVE_FREEZE_SCHEMA,
            "status": "complete",
            "experiment_id": evaluator.EXPERIMENT_ID,
            "successor_contract_sha256": digest,
            "population_role": "prospective",
            "prospective_frames_inclusive": list(evaluator.PROSPECTIVE_FRAMES),
            "anchors": list(evaluator.PROSPECTIVE_ANCHORS),
            "seeds": list(evaluator.SEEDS),
            "primary_deployments": [
                list(item) for item in evaluator.PRIMARY_DEPLOYMENTS
            ],
            "identity_parity_deployment": ["CLEAN", "identity", "corrected"],
            "intervention_magnitude_deployment": [
                "PATH_PROJECTION",
                "path_projection",
                "raw_pre_correction",
            ],
            "primary_metrics": list(evaluator.PRIMARY_METRICS),
            "mediator_metrics": list(evaluator.MEDIATOR_METRICS),
            "horizons": list(evaluator.HORIZONS),
            "full_trace_inclusive": list(evaluator.FULL_TRACE),
            "late_window_inclusive": list(evaluator.LATE_WINDOW),
            "tie_ratio_inclusive": [0.95, 1.05],
            "aggregation": "paired_per_anchor_ratio_then_seedwise_median",
            "sampling": "deterministic_finite_population_common_trajectory",
            "expected_result_cardinalities": evaluator.EXPECTED_RESULT_CARDINALITIES,
            "parent_bindings": parent_bindings,
            "evaluator_source_set_sha256": source["source_set_sha256"],
            "predictions": _synthetic_predictions(),
            "claim_boundary": evaluator.FREEZE_CLAIM_BOUNDARY,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    evaluator._write_json(freeze_root / "prospective_freeze.json", freeze)
    evaluator._write_json(freeze_root / "source_manifest.json", source)
    files = {
        name: evaluator._file_record(freeze_root / name, freeze_root)
        for name in evaluator.FREEZE_OUTPUT_FILES
    }
    evaluator._write_json(
        freeze_root / "final_hash_manifest.json",
        {
            "schema": evaluator.FINAL_HASH_MANIFEST_SCHEMA,
            "experiment_id": evaluator.EXPERIMENT_ID,
            "packet_role": "prospective_freeze",
            "successor_contract_sha256": digest,
            "population_role": "prospective",
            "files": files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
        },
    )
    freeze_final_sha256 = evaluator._file_sha256(
        freeze_root / "final_hash_manifest.json"
    )
    authorization_path = tmp_path / "authorization.json"
    authorization, _ = _write_self_hashed(
        authorization_path,
        {
            "schema": evaluator.PROSPECTIVE_AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "experiment_id": evaluator.EXPERIMENT_ID,
            "successor_contract_sha256": digest,
            "prospective_freeze_final_hash_manifest_sha256": freeze_final_sha256,
            "prospective_freeze_payload_sha256": freeze["canonical_payload_sha256"],
            "prospective_freeze_source_set_sha256": source["source_set_sha256"],
            "prospective_freeze_root_sha256": evaluator._canonical_path_sha256(
                freeze_root
            ),
            "allowed_actions": [evaluator.PROSPECTIVE_ACTION],
            "authorized_population_role": "prospective",
            "prospective_reveal_authorized": True,
            "sealed_reveal_authorized": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "authorized_by": "synthetic-owner",
        },
    )
    return (
        evaluator._load_prospective_control(freeze_root, authorization_path),
        authorization,
    )


def _rewrite_synthetic_freeze(control, value: dict) -> None:
    payload = {
        key: item for key, item in value.items() if key != "canonical_payload_sha256"
    }
    payload = evaluator._self_hashed(payload)
    evaluator._write_json(
        control.freeze_packet.root / "prospective_freeze.json", payload
    )
    final = json.loads(json.dumps(control.freeze_packet.final))
    final["files"]["prospective_freeze.json"] = evaluator._file_record(
        control.freeze_packet.root / "prospective_freeze.json",
        control.freeze_packet.root,
    )
    evaluator._write_json(
        control.freeze_packet.root / "final_hash_manifest.json", final
    )


def test_prospective_refuses_before_any_other_argument_is_inspected() -> None:
    with pytest.raises(
        evaluator.ProtectedPopulationError,
        match="no scientific input path was inspected",
    ):
        evaluator.evaluate(argparse.Namespace(mode="prospective"))


def test_prospective_cli_requires_authorization_before_full_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Path]] = []

    def refuse(freeze_path: Path, authorization_path: Path):
        calls.append((freeze_path, authorization_path))
        raise evaluator.ProtectedPopulationError("synthetic refusal")

    monkeypatch.setattr(evaluator, "_load_prospective_control", refuse)
    monkeypatch.setattr(
        evaluator,
        "_parser",
        lambda mode: pytest.fail("full parser ran before authorization"),
    )
    freeze_path = tmp_path / "freeze"
    authorization_path = tmp_path / "authorization.json"
    assert (
        evaluator.main(
            [
                "--mode",
                "prospective",
                "--prospective-freeze-dir",
                str(freeze_path),
                "--prospective-authorization",
                str(authorization_path),
                "--dataset-dir",
                str(tmp_path / "must-not-be-inspected"),
            ]
        )
        == 4
    )
    assert calls == [(freeze_path, authorization_path)]


@pytest.mark.parametrize(
    "mutation",
    (
        "dummy_records",
        "duplicate_pair_key",
        "invalid_classification",
        "nonfinite_ratio",
        "duplicate_mediator_key",
        "invalid_relation",
        "semantic_field",
    ),
)
def test_malformed_freeze_is_refused_before_scientific_parser(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control, _ = _synthetic_prospective_control(tmp_path)
    freeze = json.loads(json.dumps(control.freeze))
    pairwise = freeze["predictions"]["pairwise"]
    mediators = freeze["predictions"]["mediators"]
    if mutation == "dummy_records":
        freeze["predictions"] = {
            "pairwise": [{"fixture": index} for index in range(60)],
            "mediators": [{"fixture": index} for index in range(54)],
        }
    elif mutation == "duplicate_pair_key":
        pairwise[1] = dict(pairwise[0])
    elif mutation == "invalid_classification":
        pairwise[0]["classification"] = "unavailable"
    elif mutation == "nonfinite_ratio":
        pairwise[0]["paired_ratio_median"] = "NaN"
    elif mutation == "duplicate_mediator_key":
        mediators[1] = dict(mediators[0])
    elif mutation == "invalid_relation":
        mediators[0]["relation_to_clean"] = "unavailable"
    elif mutation == "semantic_field":
        freeze["aggregation"] = "unregistered"
    else:  # pragma: no cover - the parameterization is closed above
        raise AssertionError("unknown synthetic mutation")
    _rewrite_synthetic_freeze(control, freeze)
    monkeypatch.setattr(
        evaluator,
        "_parser",
        lambda mode: pytest.fail("scientific parser ran after malformed freeze"),
    )
    assert (
        evaluator.main(
            [
                "--mode",
                "prospective",
                "--prospective-freeze-dir",
                str(control.freeze_packet.root),
                "--prospective-authorization",
                str(control.authorization_path),
                "--dataset-dir",
                str(tmp_path / "must-not-be-inspected"),
            ]
        )
        == 4
    )


def test_prospective_freeze_cli_has_no_scientific_population_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[argparse.Namespace] = []

    def fake_freeze(arguments: argparse.Namespace) -> dict:
        captured.append(arguments)
        return {"status": "synthetic-freeze"}

    monkeypatch.setattr(evaluator, "freeze_prospective", fake_freeze)
    assert (
        evaluator.main(
            [
                "--mode",
                "prospective-freeze",
                "--contract",
                str(tmp_path / "contract.json"),
                "--successor-contract",
                str(tmp_path / "successor.json"),
                "--preregistration",
                str(tmp_path / "preregistration.md"),
                "--development-evaluation-dir",
                str(tmp_path / "development"),
                "--output-dir",
                str(tmp_path / "freeze"),
            ]
        )
        == 0
    )
    assert len(captured) == 1
    assert not hasattr(captured[0], "dataset_dir")
    assert not hasattr(captured[0], "trajectory_dir")
    assert not hasattr(captured[0], "prospective_authorization")


def test_prospective_freeze_output_overlap_is_refused_before_source_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    development = tmp_path / "development"
    arguments = argparse.Namespace(
        contract=tmp_path / "r0.json",
        successor_contract=tmp_path / "successor.json",
        preregistration=tmp_path / "preregistration.md",
        development_evaluation_dir=development,
        output_dir=development / "freeze",
    )
    monkeypatch.setattr(
        evaluator,
        "_source_snapshot",
        lambda: pytest.fail("source read occurred after freeze output overlap"),
    )
    with pytest.raises(
        ValueError, match="freeze output overlaps development evaluation"
    ):
        evaluator.freeze_prospective(arguments)


def test_development_output_overlap_is_refused_before_source_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    arguments = argparse.Namespace(
        mode="development",
        contract=tmp_path / "r0.json",
        successor_contract=tmp_path / "successor.json",
        preregistration=tmp_path / "preregistration.md",
        dataset_dir=dataset,
        calibration_dir=tmp_path / "calibration",
        training_dir=[tmp_path / "training"],
        resource_smoke_receipt=[tmp_path / "smoke.json"],
        execution_authorization=[tmp_path / "authorization.json"],
        output_dir=dataset / "evaluation",
    )
    monkeypatch.setattr(
        evaluator,
        "_source_snapshot",
        lambda: pytest.fail("source read occurred after development output overlap"),
    )
    with pytest.raises(
        ValueError, match="development evaluation output overlaps dataset packet"
    ):
        evaluator.evaluate(arguments)


def test_prediction_matrix_freezes_exact_primary_and_mediator_choices() -> None:
    method_result, offline_result, anchors = _synthetic_prediction_inputs()
    predictions = evaluator._prediction_matrix(method_result, offline_result, anchors)
    assert len(predictions["pairwise"]) == 60
    assert {record["classification"] for record in predictions["pairwise"]} == {
        "left_better"
    }
    assert len(predictions["mediators"]) == 54
    assert {
        relation: sum(
            record["relation_to_clean"] == relation
            for record in predictions["mediators"]
        )
        for relation in ("lower", "equal", "higher")
    } == {"lower": 18, "equal": 18, "higher": 18}


def test_prediction_matrix_rejects_unavailable_mediator_ratio() -> None:
    method_result, offline_result, anchors = _synthetic_prediction_inputs()
    offline_result["IID_RECOVERY:17"][evaluator.MEDIATOR_METRICS[0]]["median"] = (
        math.nan
    )
    with pytest.raises(ValueError, match="must be finite"):
        evaluator._prediction_matrix(method_result, offline_result, anchors)


def test_primary_infinity_predictions_build_and_assess_canonically() -> None:
    method_result, offline_result, anchors = _synthetic_prediction_inputs()
    values_by_metric = {
        evaluator.PRIMARY_METRICS[0]: (1.0, 0.0, math.inf, 1.0, math.inf),
        evaluator.PRIMARY_METRICS[1]: (0.0, 1.0, math.inf, 0.0, math.inf),
    }
    for seed in evaluator.SEEDS:
        for rank, deployment in enumerate(evaluator.PRIMARY_DEPLOYMENTS):
            summary = method_result[
                f"{deployment[0]}:{seed}:{deployment[1]}:{deployment[2]}"
            ]
            for anchor in anchors:
                for metric, values in values_by_metric.items():
                    summary["per_anchor"][str(anchor)][metric] = values[rank]

    predictions = evaluator._prediction_matrix(method_result, offline_result, anchors)
    evaluator._validate_prediction_matrix(predictions)
    lookup = {
        (
            record["seed"],
            record["metric"],
            tuple(record["left"]),
            tuple(record["right"]),
        ): record
        for record in predictions["pairwise"]
    }
    clean = evaluator.PRIMARY_DEPLOYMENTS[0]
    iid = evaluator.PRIMARY_DEPLOYMENTS[1]
    error_subspace = evaluator.PRIMARY_DEPLOYMENTS[2]
    path = evaluator.PRIMARY_DEPLOYMENTS[4]
    finite_over_zero = lookup[(17, evaluator.PRIMARY_METRICS[0], clean, iid)]
    zero_over_finite = lookup[(17, evaluator.PRIMARY_METRICS[1], clean, iid)]
    infinity_over_infinity = lookup[
        (17, evaluator.PRIMARY_METRICS[0], error_subspace, path)
    ]
    assert finite_over_zero["paired_ratio_median"] == "Infinity"
    assert finite_over_zero["classification"] == "right_better"
    assert zero_over_finite["paired_ratio_median"] == 0.0
    assert zero_over_finite["classification"] == "left_better"
    assert infinity_over_infinity["paired_ratio_median"] == 1.0
    assert infinity_over_infinity["classification"] == "tie"

    assessment = evaluator._prediction_assessment(
        predictions, method_result, offline_result, anchors
    )
    assert assessment["pairwise_exact_match_count"] == 60
    closed = evaluator._self_hashed(evaluator._json_safe({"assessment": assessment}))
    json.dumps(closed, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    "invalid_ratio",
    (math.inf, math.nan, "NaN", "-Infinity", "arbitrary"),
)
def test_prediction_validation_rejects_noncanonical_primary_values(
    invalid_ratio,
) -> None:
    predictions = json.loads(json.dumps(_synthetic_predictions()))
    predictions["pairwise"][0]["paired_ratio_median"] = invalid_ratio
    with pytest.raises((TypeError, ValueError)):
        evaluator._validate_prediction_matrix(predictions)


def test_observed_unavailable_mediator_does_not_abort_assessment() -> None:
    method_result, offline_result, anchors = _synthetic_prediction_inputs()
    frozen = evaluator._prediction_matrix(method_result, offline_result, anchors)
    unavailable_metric = "one_prefix_response_transverse_rms"
    offline_result["IID_RECOVERY:17"][unavailable_metric]["median"] = math.nan
    observed = evaluator._prediction_matrix(
        method_result,
        offline_result,
        anchors,
        observed_mediator_availability=True,
    )
    with pytest.raises(ValueError, match="prediction fields differ"):
        evaluator._validate_prediction_matrix(observed)
    assessment = evaluator._prediction_assessment(
        frozen, method_result, offline_result, anchors
    )
    record = next(
        item
        for item in assessment["mediators"]
        if item["arm"] == "IID_RECOVERY"
        and item["seed"] == 17
        and item["metric"] == unavailable_metric
    )
    assert record["available"] is False
    assert record["ratio_to_clean"] is None
    assert record["relation_to_clean"] == "unavailable"
    assert record["predicted_relation_to_clean"] == "lower"
    assert record["relation_match"] is False
    assert assessment["mediator_exact_match_count"] == 53
    assert assessment["mediator_total"] == 54
    closed = evaluator._self_hashed(evaluator._json_safe({"assessment": assessment}))
    json.dumps(closed, sort_keys=True, allow_nan=False)


def test_prospective_control_and_access_receipt_are_independently_bound(
    tmp_path: Path,
) -> None:
    control, authorization = _synthetic_prospective_control(tmp_path)
    receipt_path = evaluator._prospective_access_receipt_path(control)
    receipt = evaluator._write_prospective_access_receipt(receipt_path, control)
    assert receipt.value["access_recorded_before_first_scientific_read"] is True
    assert receipt.value["prospective_opened"] is True
    assert receipt.value["sealed_opened"] is False
    assert (
        receipt.value["prospective_authorization_payload_sha256"]
        == (authorization["canonical_payload_sha256"])
    )
    evaluator._reverify_prospective_access_receipt(receipt)
    with pytest.raises(
        evaluator.ProtectedPopulationError, match="authorization was consumed"
    ):
        evaluator._write_prospective_access_receipt(receipt_path, control)
    control.authorization_path.write_text(
        json.dumps(authorization, indent=2), encoding="utf-8"
    )
    reformatted_control = evaluator._load_prospective_control(
        control.freeze_packet.root, control.authorization_path
    )
    assert (
        evaluator._prospective_access_receipt_path(reformatted_control) == receipt_path
    )
    with pytest.raises(
        evaluator.ProtectedPopulationError, match="authorization was consumed"
    ):
        evaluator._write_prospective_access_receipt(receipt_path, reformatted_control)
    alternate = tmp_path / "alternate-receipt.json"
    with pytest.raises(evaluator.ProtectedPopulationError, match="deterministic path"):
        evaluator._write_prospective_access_receipt(alternate, reformatted_control)
    assert not alternate.exists()
    tampered = {
        key: value
        for key, value in authorization.items()
        if key != "canonical_payload_sha256"
    }
    tampered["allowed_actions"] = []
    _write_self_hashed(control.authorization_path, tampered)
    with pytest.raises(
        evaluator.ProtectedPopulationError,
        match="prospective authorization differs",
    ):
        evaluator._load_prospective_control(
            control.freeze_packet.root, control.authorization_path
        )


def test_extra_authorization_field_is_refused_before_scientific_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control, authorization = _synthetic_prospective_control(tmp_path)
    rewritten = {
        key: value
        for key, value in authorization.items()
        if key != "canonical_payload_sha256"
    }
    rewritten["unexpected_extension"] = True
    _write_self_hashed(control.authorization_path, rewritten)
    monkeypatch.setattr(
        evaluator,
        "_parser",
        lambda mode: pytest.fail("scientific parser ran after authorization tamper"),
    )
    assert (
        evaluator.main(
            [
                "--mode",
                "prospective",
                "--prospective-freeze-dir",
                str(control.freeze_packet.root),
                "--prospective-authorization",
                str(control.authorization_path),
            ]
        )
        == 4
    )


def test_relocated_freeze_is_refused_before_scientific_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control, _ = _synthetic_prospective_control(tmp_path)
    relocated = tmp_path / "relocated-freeze"
    shutil.copytree(control.freeze_packet.root, relocated)
    monkeypatch.setattr(
        evaluator,
        "_parser",
        lambda mode: pytest.fail("scientific parser ran after freeze relocation"),
    )
    assert (
        evaluator.main(
            [
                "--mode",
                "prospective",
                "--prospective-freeze-dir",
                str(relocated),
                "--prospective-authorization",
                str(control.authorization_path),
            ]
        )
        == 4
    )


def test_prospective_receipt_claim_allows_exactly_one_concurrent_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    control, _ = _synthetic_prospective_control(tmp_path)
    receipt_path = evaluator._prospective_access_receipt_path(control)
    real_open = evaluator.os.open
    barrier = Barrier(2)

    def synchronized_open(path, flags, mode=0o777, *, dir_fd=None):
        if (
            evaluator._absolute_path(Path(path))
            == evaluator._absolute_path(receipt_path)
            and flags & evaluator.os.O_EXCL
        ):
            barrier.wait(timeout=10.0)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evaluator.os, "open", synchronized_open)

    def consume():
        try:
            return evaluator._write_prospective_access_receipt(receipt_path, control)
        except evaluator.ProtectedPopulationError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(consume) for _ in range(2)]
        results = [future.result() for future in futures]
    receipts = [
        result
        for result in results
        if isinstance(result, evaluator.ProspectiveAccessReceipt)
    ]
    refusals = [
        result
        for result in results
        if isinstance(result, evaluator.ProtectedPopulationError)
    ]
    assert len(receipts) == 1
    assert len(refusals) == 1
    assert "authorization was consumed" in str(refusals[0])
    winner = receipts[0]
    expected_bytes = (
        json.dumps(winner.value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    assert receipt_path.read_bytes() == expected_bytes
    assert evaluator._file_sha256(receipt_path) == winner.file_sha256
    evaluator._reverify_prospective_access_receipt(winner)


def test_no_replace_directory_publish_refuses_empty_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "owner.txt").write_bytes(b"staged owner")
    with pytest.raises(OSError):
        evaluator._rename_directory_no_replace(source, destination)
    assert (source / "owner.txt").read_bytes() == b"staged owner"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_staging_directory_is_unique_under_requested_parent(tmp_path: Path) -> None:
    first = evaluator._make_staging_directory(tmp_path, prefix=".packet.staging-")
    second = evaluator._make_staging_directory(tmp_path, prefix=".packet.staging-")
    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.name.startswith(".packet.staging-")
    assert second.name.startswith(".packet.staging-")


def test_staged_packet_concurrent_loser_cleans_only_own_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    barrier = Barrier(2)

    def publish(owner: str) -> bool:
        def writer(staging: Path) -> None:
            (staging / "owner.txt").write_text(owner, encoding="utf-8")
            barrier.wait(timeout=10.0)

        try:
            evaluator._write_staged_packet(output, writer)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, owner) for owner in ("first", "second")]
        outcomes = [future.result() for future in futures]
    assert sorted(outcomes) == [False, True]
    assert (output / "owner.txt").read_text(encoding="utf-8") in {"first", "second"}
    assert not list(tmp_path.glob(".published.staging-*"))


@pytest.mark.parametrize(
    "case, message",
    (
        ("wrong_receipt", "deterministic path"),
        ("output_inside_freeze", "output overlaps prospective freeze"),
        ("output_contains_dataset", "output overlaps dataset dir"),
        ("output_equals_dataset", "output overlaps dataset dir"),
        ("receipt_inside_dataset", "receipt overlaps dataset dir"),
        ("receipt_equals_dataset", "receipt overlaps dataset dir"),
        ("receipt_output_overlap", "receipt and evaluation output overlap"),
    ),
)
def test_prospective_path_overlap_is_refused_before_scientific_read(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control, _ = _synthetic_prospective_control(tmp_path)
    receipt = evaluator._prospective_access_receipt_path(control)
    output = tmp_path / "evaluation"
    dataset = tmp_path / "dataset"
    if case == "wrong_receipt":
        receipt = tmp_path / "alternate-receipt.json"
    elif case == "output_inside_freeze":
        output = control.freeze_packet.root / "evaluation"
    elif case == "output_contains_dataset":
        dataset = output / "dataset"
    elif case == "output_equals_dataset":
        dataset = output
    elif case == "receipt_inside_dataset":
        dataset = receipt.parent
    elif case == "receipt_equals_dataset":
        dataset = receipt
    elif case == "receipt_output_overlap":
        output = receipt
    else:  # pragma: no cover - the parameterization is closed above
        raise AssertionError("unknown overlap case")
    arguments = argparse.Namespace(
        mode="prospective",
        prospective_freeze_dir=control.freeze_packet.root,
        prospective_authorization=control.authorization_path,
        prospective_access_receipt=receipt,
        output_dir=output,
        dataset_dir=dataset,
    )
    monkeypatch.setattr(evaluator, "_load_prospective_control", lambda *args: control)
    monkeypatch.setattr(
        evaluator,
        "_source_snapshot",
        lambda: pytest.fail("scientific source/path read occurred after overlap"),
    )
    with pytest.raises(evaluator.ProtectedPopulationError, match=message):
        evaluator.evaluate(arguments)


def test_authorized_loader_materializes_only_exact_prospective_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trajectory_root = tmp_path / "trajectory"
    trajectory_root.mkdir()
    manifest_path = trajectory_root / "storage_manifest.json"
    manifest_path.write_bytes(b"synthetic manifest")
    prospective_digest = "b" * 64
    for index in range(
        evaluator.PROSPECTIVE_FRAMES[0], evaluator.PROSPECTIVE_FRAMES[1] + 1
    ):
        (trajectory_root / f"trajectory_flow_{index:05d}.dat").write_bytes(b"x")
    records = [
        {
            "index": index,
            "file": f"trajectory_flow_{index:05d}.dat",
            "bytes": 1,
            "sha256": (
                evaluator._file_sha256(
                    trajectory_root / f"trajectory_flow_{index:05d}.dat"
                )
                if evaluator.PROSPECTIVE_FRAMES[0]
                <= index
                <= evaluator.PROSPECTIVE_FRAMES[1]
                else prospective_digest
            ),
        }
        for index in range(499, 2000)
    ]
    payload_digest = "c" * 64
    ordered_digest = "d" * 64
    storage = {
        "status": "validated",
        "canonical_payload_sha256": payload_digest,
        "output_contract": {"first_index": 499, "last_index": 1999, "count": 1501},
        "aggregate": {"ordered_file_records_sha256": ordered_digest},
        "files": records,
    }
    baseline = {
        "parent_evidence": {
            "trajectory": {
                "storage_manifest_file_sha256": evaluator._file_sha256(manifest_path),
                "storage_manifest_payload_sha256": payload_digest,
                "ordered_restart_records_sha256": ordered_digest,
            }
        }
    }
    opened: list[int] = []

    def fake_extract(path: Path, **kwargs):
        del kwargs
        opened.append(int(path.stem.rsplit("_", 1)[1]))
        return np.zeros((2, len(evaluator.FIELDS)), dtype=np.float64)

    monkeypatch.setattr(
        evaluator, "_load_verified_storage_manifest", lambda path: storage
    )
    monkeypatch.setattr(evaluator, "_extract_native_state", fake_extract)
    monkeypatch.setattr(
        evaluator, "_reverify_prospective_control", lambda control: None
    )
    monkeypatch.setattr(
        evaluator, "_reverify_prospective_access_receipt", lambda receipt: None
    )
    population = evaluator._load_authorized_prospective_population(
        trajectory_dir=trajectory_root,
        storage_manifest_path=manifest_path,
        baseline=baseline,
        geometry=SimpleNamespace(
            num_nodes=2, native_coordinates=np.zeros((2, 2), dtype=np.float64)
        ),
        control=SimpleNamespace(),  # type: ignore[arg-type]
        receipt=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert opened == list(
        range(evaluator.PROSPECTIVE_FRAMES[0], evaluator.PROSPECTIVE_FRAMES[1] + 1)
    )
    assert population.indices.tolist() == opened
    assert population.states.shape == (240, 2, len(evaluator.FIELDS))
    assert population.provenance["sealed_opened"] is False


def test_selection_score_accepts_only_canonical_positive_infinity() -> None:
    assert (
        evaluator._selection_score("Infinity", label="score", json_artifact=True)
        == math.inf
    )
    assert (
        evaluator._selection_score(math.inf, label="score", json_artifact=False)
        == math.inf
    )
    with pytest.raises(ValueError, match="noncanonical"):
        evaluator._selection_score(math.inf, label="score", json_artifact=True)
    for invalid in ("-Infinity", "NaN"):
        with pytest.raises(ValueError, match="finite or canonical"):
            evaluator._selection_score(invalid, label="score", json_artifact=True)


def test_all_learned_arms_must_share_pairing_hashes_within_seed() -> None:
    packets = [
        SimpleNamespace(
            arm=arm,
            seed=seed,
            summary={
                "initial_model_state_sha256": f"{seed:064x}",
                "presentation_schedule_sha256": f"{seed + 1:064x}",
            },
        )
        for arm in evaluator.LEARNED_ARMS
        for seed in evaluator.SEEDS
    ]
    evaluator._verify_paired_training_hashes(packets)  # type: ignore[arg-type]
    packets[-1].summary["presentation_schedule_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="not paired"):
        evaluator._verify_paired_training_hashes(packets)  # type: ignore[arg-type]


def test_training_receipts_are_strictly_bound_and_reverified(
    request: pytest.FixtureRequest,
) -> None:
    successor_sha256 = "a" * 64
    preregistration_sha256 = "b" * 64
    inherited_sha256 = "c" * 64
    dataset_payload_sha256 = "d" * 64
    dataset_sha256 = "e" * 64
    calibration_payload_sha256 = "f" * 64
    calibration_sha256 = "1" * 64
    source_sha256 = "2" * 64
    initial_model_sha256 = "3" * 64
    presentation_sha256 = "4" * 64
    intervention_rng_sha256 = "5" * 64
    device_binding = {"device": "cuda:0", "name": "fixture"}
    runtime = {"torch": "fixture", "device": "cuda:0"}
    common = {
        "experiment_id": evaluator.EXPERIMENT_ID,
        "arm": "CLEAN",
        "seed": 17,
        "successor_contract_sha256": successor_sha256,
        "preregistration_sha256": preregistration_sha256,
        "inherited_r0_contract_sha256": inherited_sha256,
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "calibration_payload_sha256": calibration_payload_sha256,
        "calibration_final_hash_manifest_sha256": calibration_sha256,
        "source_set_sha256": source_sha256,
        "prospective_opened": False,
        "sealed_opened": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }
    test_root = Path(__file__).resolve().parent
    smoke_path = test_root / ".evaluator_authority_smoke_test.json"
    authorization_path = test_root / ".evaluator_authority_auth_test.json"
    if smoke_path.exists() or authorization_path.exists():
        raise RuntimeError("stale evaluator authority test artifact exists")
    request.addfinalizer(lambda: smoke_path.unlink(missing_ok=True))
    request.addfinalizer(lambda: authorization_path.unlink(missing_ok=True))
    smoke, smoke_file_sha256 = _write_self_hashed(
        smoke_path,
        {
            "schema": "smoke.v1",
            "status": "succeeded",
            **common,
            "effective_batch_size": 4,
            "microbatch_size": 4,
            "gradient_accumulation_steps": 1,
            "full_resolution_num_nodes": 3,
            "paired_initialization_and_clean_order": True,
            "initial_model_state_sha256": initial_model_sha256,
            "presentation_schedule_sha256": presentation_sha256,
            "intervention_rng": {
                "initial_generator_state_sha256": intervention_rng_sha256
            },
            "training_model_calls": 1,
            "loss": 0.25,
            "gradient_norm_before_clip": 0.5,
            "production_device_binding": device_binding,
            "runtime": runtime,
            "forward_backward_and_adamw_step_completed": True,
            "scientific_hyperparameters_changed": False,
            "checkpoint_written": False,
        },
    )
    authorization, authorization_file_sha256 = _write_self_hashed(
        authorization_path,
        {
            "schema": "authorization.v1",
            "status": "authorized",
            "experiment_id": evaluator.EXPERIMENT_ID,
            "successor_contract_sha256": successor_sha256,
            "preregistration_sha256": preregistration_sha256,
            "inherited_r0_contract_sha256": inherited_sha256,
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "calibration_final_hash_manifest_sha256": calibration_sha256,
            "source_set_sha256": source_sha256,
            "resource_smoke_payload_sha256": smoke["canonical_payload_sha256"],
            "resource_smoke_file_sha256": smoke_file_sha256,
            "production_device_binding": device_binding,
            "allowed_actions": ["train_pcno_naca0012_successor"],
            "authorized_arms": list(evaluator.LEARNED_ARMS),
            "authorized_seeds": list(evaluator.SEEDS),
            "pcno_code_audited": True,
            "successor_training_authorized": True,
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
            "online_defect_trigger": False,
            "authorized_by": "fixture-owner",
        },
    )
    inputs = {
        **common,
        "resource_smoke_payload_sha256": smoke["canonical_payload_sha256"],
        "resource_smoke_file_sha256": smoke_file_sha256,
        "authorization_payload_sha256": authorization["canonical_payload_sha256"],
        "authorization_file_sha256": authorization_file_sha256,
        "production_device_binding": device_binding,
    }
    packet = evaluator.TrainingPacket(
        packet=None,  # type: ignore[arg-type]
        arm="CLEAN",
        seed=17,
        config={
            "microbatch_size": 4,
            "gradient_accumulation_steps": 1,
            "presentation": {"presentation_schedule_sha256": presentation_sha256},
            "intervention_rng": {
                "initial_generator_state_sha256": intervention_rng_sha256
            },
        },
        inputs=inputs,
        runtime={"runtime": runtime},
        summary={
            "source_set_sha256": source_sha256,
            "initial_model_state_sha256": initial_model_sha256,
            "presentation_schedule_sha256": presentation_sha256,
        },
        checkpoint_record={},
    )
    successor = {
        "packet_schemas": {
            "resource_smoke": "smoke.v1",
            "execution_authorization": "authorization.v1",
        },
        "preregistration_sha256": preregistration_sha256,
        "inherited_r0_contract_sha256": inherited_sha256,
    }
    authorities = evaluator._verify_production_authorities(
        smoke_paths=[smoke_path],
        authorization_paths=[authorization_path],
        training_packets=[packet],
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        calibration_sha256=calibration_sha256,
        geometry=SimpleNamespace(num_nodes=3),  # type: ignore[arg-type]
    )
    assert authorities.records[0]["authorization_file_sha256"] == (
        authorization_file_sha256
    )
    smoke_path.write_text(smoke_path.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluator._reverify_authorities(authorities)


def test_offline_diagnostics_use_same_frame_state_metrics_symmetrically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indices = np.asarray([9, 10, 11, 12], dtype=np.int64)
    states = np.arange(4 * 2 * 5, dtype=np.float64).reshape(4, 2, 5)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_recurrent_step(
        model, previous, current, geometry, normalization, **kwargs
    ):
        del model, previous, geometry, normalization, kwargs
        return current, current + 1.0

    def fake_state_metrics(prediction, target, **kwargs):
        del kwargs
        calls.append((np.asarray(prediction), np.asarray(target)))
        return {"marker": len(calls)}

    monkeypatch.setattr(evaluator, "recurrent_step", fake_recurrent_step)
    monkeypatch.setattr(evaluator, "_state_metrics", fake_state_metrics)
    monkeypatch.setattr(
        evaluator,
        "_vector_diagnostics",
        lambda *args, **kwargs: {"finite": True, "normalized_rms": 1.0},
    )
    monkeypatch.setattr(
        evaluator,
        "_project_state",
        lambda projector, state: evaluator.Projection(
            np.asarray(state), 0.0, 0, 0.0, None
        ),
    )

    class Model:
        @staticmethod
        def prepare_fourier_tensors(geometry):
            del geometry

    class Geometry:
        @staticmethod
        def expand(batch_size, device):
            return {"batch_size": batch_size, "device": str(device)}

    normalization = NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.ones(5),
        state_rms=np.ones(5),
    )
    reference = {
        index: evaluator.Projection(states[offset], 0.0, 0, 0.0, None)
        for offset, index in enumerate(indices)
    }
    rows, cost = evaluator._offline_diagnostics(
        packet=SimpleNamespace(arm="CLEAN", seed=17),  # type: ignore[arg-type]
        model=Model(),  # type: ignore[arg-type]
        states=states,
        indices=indices,
        anchors=[10],
        geometry=Geometry(),  # type: ignore[arg-type]
        normalization=normalization,
        projector=object(),  # type: ignore[arg-type]
        reference_cache=reference,
        graph_edges=np.asarray([[0, 1]], dtype=np.int64),
        weights={},
        structure_context=(np.asarray([]),) * 6,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0][1], states[2])
    np.testing.assert_array_equal(calls[1][1], states[3])
    np.testing.assert_array_equal(calls[2][1], states[3])
    assert rows[0]["clean_generated_state_marker"] == 1
    assert rows[0]["clean_forced_state_marker"] == 2
    assert rows[0]["one_prefix_state_marker"] == 3
    assert cost["state_transition_evaluations"] == 3

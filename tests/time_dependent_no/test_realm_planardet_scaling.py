from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import train_realm_planardet_scaling as trainer
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_TRAIN_GROUPS,
    PlanarDetMetadata,
    sha256_file,
)
from utility.time_dependent_no.realm_planardet_artifacts import build_source_manifest
from utility.time_dependent_no.realm_planardet_runtime import scheduled_step_window
from utility.time_dependent_no.realm_planardet_scaling import (
    EXPECTED_PARAMETER_COUNTS,
    PREFLIGHT_SCHEMA,
    SCALING_SOURCE_PATHS,
    PlanarDetScalingContract,
    active_train_groups,
    balanced_case_indices,
    maximin_train_indices,
    scheduled_scaling_window,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _contract(**changes: object) -> PlanarDetScalingContract:
    values: dict[str, object] = {
        "run_id": "d093_planardet_scaling_unit",
        "architecture": "pcno",
        "train_trajectory_count": 3,
        "seed": 11,
        "total_steps": 200,
        "one_call_steps": 49,
        "validation_interval": 50,
        "checkpoint_eligible_from_step": 100,
        "competence_npe_ceiling": 0.5,
        "max_wall_seconds": 3600.0,
        "preflight_result_sha256": "1" * 64,
        "normalizer_arrays_sha256": "2" * 64,
        "data_audit_final_manifest_sha256": "3" * 64,
        "open_manifest_payload_sha256": "4" * 64,
        "persistence_npe": 1.0,
        "linear_extrapolation_npe": 2.0,
    }
    values.update(changes)
    return PlanarDetScalingContract(**values)  # type: ignore[arg-type]


def _metadata() -> PlanarDetMetadata:
    y = np.linspace(0.009, 0.001, 4, dtype=np.float32)
    x = np.linspace(0.020, 0.001, 5, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")
    coordinates = np.stack((y_grid, x_grid)).astype(np.float32)
    return PlanarDetMetadata(
        released_coords_xy=np.stack((x_grid.T, y_grid.T)).astype(np.float32),
        canonical_coords_yx=coordinates,
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        dx=float(abs(x[1] - x[0])),
        dy=float(abs(y[1] - y[0])),
        times=tuple(float(index) for index in range(50)),
        train_groups=PLANARDET_TRAIN_GROUPS,
        val_groups=("sampling_phi1-290",),
        test_groups=("sampling_phi12e-1-330",),
    )


def test_three_case_subset_is_outcome_independent_physical_maximin() -> None:
    assert maximin_train_indices(3) == (4, 5, 6)
    assert active_train_groups(3) == (
        "sampling_phi8e-1-330",
        "sampling_phi8e-1-290",
        "sampling_phi12e-1-290",
    )
    assert maximin_train_indices(7) == tuple(range(7))
    with pytest.raises(ValueError, match="3 or 7"):
        maximin_train_indices(4)


def test_balanced_presentations_match_d092_at_n7_and_close_over_three_steps() -> None:
    for step in (1, 49, 50, 491, 5000):
        expected = scheduled_step_window(step, one_call_steps=490, seed=0, case_count=7)
        actual = scheduled_scaling_window(
            step, one_call_steps=490, seed=0, case_count=7
        )
        assert actual == expected

    counts: Counter[int] = Counter()
    for step in range(1, 4):
        indices = balanced_case_indices(step, case_count=3, seed=0)
        assert len(indices) == 7
        counts.update(indices)
    assert counts == {0: 7, 1: 7, 2: 7}


def test_contract_roundtrip_freezes_transductive_control_and_presentations() -> None:
    contract = _contract()
    assert PlanarDetScalingContract.from_payload(contract.payload()) == contract
    config = contract.frozen_training_config()
    assert config["data_exposure"]["active_train_indices"] == [4, 5, 6]
    assert config["data_exposure"]["total_presentations"] == 1400
    assert config["data_exposure"]["normalizer_control_is_transductive"] is True
    assert config["optimizer"]["weight_decay"] == 0.0
    assert config["scope"]["architecture_regularization_is_matched"] is True
    pcfno_config = _contract(architecture="pcfno").frozen_training_config()
    assert pcfno_config["model"]["family"] == "PCFNO"
    assert pcfno_config["model"]["gradient_branch"] is False
    assert pcfno_config["model"]["expected_trainable_parameters"] == 10_706_669

    changed = contract.payload()
    changed["architecture"] = "ffno"
    with pytest.raises(ValueError, match="digest"):
        PlanarDetScalingContract.from_payload(changed)


def test_exact_production_model_counts_and_zero_heads() -> None:
    metadata = _metadata()
    for architecture in ("pcno", "pcfno", "ffno"):
        model = trainer._build_model(architecture, metadata)
        count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert count == EXPECTED_PARAMETER_COUNTS[architecture]
        if architecture == "ffno":
            final = model.output_projection[-1]
            assert torch.count_nonzero(final.weight) == 0
            assert torch.count_nonzero(final.bias) == 0
        elif architecture == "pcfno":
            assert model.model_contract()["gradient_branch"] is False
            assert not any("backbone.gws" in key for key in model.state_dict())


def test_source_manifest_includes_both_architectures_and_scaling_contract() -> None:
    manifest = build_source_manifest(REPO_ROOT, entrypoints=SCALING_SOURCE_PATHS)
    paths = {row["path"] for row in manifest["files"]}
    assert set(SCALING_SOURCE_PATHS).issubset(paths)
    assert {
        "pcno/pcno.py",
        "utility/time_dependent_no/realm_pcno.py",
        "utility/time_dependent_no/realm_ffno.py",
    }.issubset(paths)


def test_preflight_binding_rejects_source_or_input_drift(tmp_path: Path) -> None:
    source = {"schema": "source", "canonical_payload_sha256": "5" * 64}
    payload: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "architecture": "pcno",
        "parameter_count": EXPECTED_PARAMETER_COUNTS["pcno"],
        "expected_parameter_count": EXPECTED_PARAMETER_COUNTS["pcno"],
        "finite_two_call_optimizer_step": True,
        "passes_memory_gate": True,
        "source_manifest": source,
        "inputs": {
            "open_manifest_payload_sha256": "4" * 64,
            "normalizer_arrays_sha256": "2" * 64,
            "data_audit_final_manifest_sha256": "3" * 64,
        },
        "trajectory_arrays_opened": False,
        "test_object_opened": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    contract = _contract(preflight_result_sha256=sha256_file(path))
    assert (
        trainer._validate_preflight(path, contract=contract, source_manifest=source)
        == payload
    )
    with pytest.raises(ValueError, match="authorize"):
        trainer._validate_preflight(
            path,
            contract=contract,
            source_manifest={"schema": "drift"},
        )


def test_scaling_entrypoint_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        trainer.main(["--help"])
    assert exc_info.value.code == 0
    assert "preflight" in capsys.readouterr().out

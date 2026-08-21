from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.time_dependent_no import evaluate_realm_planardet_pcno as evaluator
from scripts.time_dependent_no import smoke_realm_planardet_pcno as smoke
from scripts.time_dependent_no import train_realm_planardet_pcno as trainer
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_OPEN_MANIFEST_SHA256,
    sha256_file,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
    structured_state_sha256,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    EXPECTED_PARAMETER_COUNTS,
    PlanarDetTrainingContract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _contract() -> PlanarDetTrainingContract:
    digest = "1" * 64
    return PlanarDetTrainingContract(
        run_id="d000_planardet_trainer_unit",
        seed=11,
        width=96,
        total_steps=200,
        one_call_steps=49,
        validation_interval=50,
        checkpoint_eligible_from_step=100,
        competence_npe_ceiling=0.5,
        max_wall_seconds=3600.0,
        smoke_result_sha256=digest,
        normalizer_arrays_sha256=digest,
        data_audit_final_manifest_sha256=digest,
        open_manifest_payload_sha256=digest,
        persistence_npe=1.0,
        linear_extrapolation_npe=2.0,
    )


def _validation(score: float = 0.25) -> dict[str, object]:
    return {
        "realm_npe_mean": score,
        "all_normalized_finite": True,
        "all_decoded_finite": True,
        "all_released_state_admissible": True,
        "all_bounded_10x_train_max": True,
        "all_pMax_nondecreasing_from_input": True,
    }


def test_structured_state_hash_supports_scalars_without_changing_vector_bytes() -> None:
    assert structured_state_sha256(
        {"tensor": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    ) == "22f39ef83a289ee45e53448042e4fa585d5c9d43358ef7c21924299d02f52b43"
    scalar_hash = structured_state_sha256(torch.tensor(1.0, dtype=torch.float32))
    assert scalar_hash == structured_state_sha256(
        torch.tensor(1.0, dtype=torch.float32)
    )
    assert scalar_hash != structured_state_sha256(
        torch.tensor(2.0, dtype=torch.float32)
    )


def test_checkpoint_roundtrip_restores_cpu_model_optimizer_scheduler_and_rng() -> None:
    contract = _contract()
    torch.manual_seed(5)
    model = nn.Linear(3, 2)
    optimizer, scheduler = trainer._build_optimizer_and_scheduler(model, contract)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_model_hash = structured_state_sha256(model.state_dict())
    provenance = {
        "config_digest": "a" * 64,
        "input_digest": "b" * 64,
        "source_digest": "c" * 64,
        "runtime_digest": "d" * 64,
    }
    signature = canonical_json_sha256(provenance)
    checkpoint = trainer.build_last_checkpoint(
        model,
        optimizer,
        scheduler,
        contract=contract,
        run_signature=signature,
        completed_step=1,
        best_step=0,
        best_score=math.inf,
        best_model_state_sha256=None,
        history=({"completed_step": 1},),
        elapsed_seconds_total=4.5,
        provenance=provenance,
    )

    restored = nn.Linear(3, 2)
    restored_optimizer, restored_scheduler = trainer._build_optimizer_and_scheduler(
        restored, contract
    )
    state = trainer.restore_last_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        restored_scheduler,
        contract=contract,
        expected_run_signature=signature,
    )
    assert state[:4] == (1, 0, math.inf, None)
    assert state[5] == 4.5
    assert structured_state_sha256(restored.state_dict()) == expected_model_hash
    assert (
        restored_optimizer.state_dict()["param_groups"]
        == optimizer.state_dict()["param_groups"]
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_best_checkpoint_binds_model_normalizer_and_competence() -> None:
    contract = _contract()
    model = nn.Linear(2, 2)
    provenance = {
        "config_digest": "a" * 64,
        "input_digest": "b" * 64,
        "source_digest": "c" * 64,
        "runtime_digest": "d" * 64,
    }
    signature = canonical_json_sha256(provenance)
    normalizer = {
        "mean": torch.zeros(13),
        "scale": torch.ones(13),
        "source_sha256": "1" * 64,
    }
    checkpoint = trainer.build_best_checkpoint(
        model,
        contract=contract,
        run_signature=signature,
        completed_step=100,
        validation=_validation(),
        normalizer_state=normalizer,
        provenance=provenance,
    )
    assert checkpoint["selection_value"] == 0.25
    assert checkpoint["competence_gate"]["all_gates_pass"]
    assert checkpoint["model_state_sha256"] == structured_state_sha256(
        checkpoint["model_state"]
    )
    assert checkpoint["normalizer_state_sha256"] == structured_state_sha256(
        checkpoint["normalizer_state"]
    )


def test_a2_binding_requires_exact_closed_audit_smoke_and_source(
    tmp_path: Path,
) -> None:
    normalizer = tmp_path / "normalizer.npz"
    normalizer.write_bytes(b"normalizer")
    normalizer_sha = sha256_file(normalizer)
    audit = {
        "schema": trainer.AUDIT_MANIFEST_SCHEMA,
        "files": {"normalizer_arrays.npz": normalizer_sha},
        "open_manifest_sha256": PLANARDET_OPEN_MANIFEST_SHA256,
        "self_hash_excluded": True,
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    source = {"schema": "source", "canonical_payload_sha256": "2" * 64}
    smoke_payload: dict[str, object] = {
        "schema": trainer.SMOKE_SCHEMA,
        "status": "pass",
        "passes_full_grid_step": True,
        "passes_memory_gate": True,
        "test_object_opened": False,
        "dataset_array_opened": False,
        "configuration": {
            "layers": [96] * 5,
            "parameter_count": EXPECTED_PARAMETER_COUNTS[96],
        },
        "prior_width128_result_sha256": "3" * 64,
        "source_manifest": source,
    }
    smoke_payload["canonical_payload_sha256"] = canonical_json_sha256(smoke_payload)
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(smoke_payload, sort_keys=True), encoding="utf-8")
    contract = replace(
        _contract(),
        normalizer_arrays_sha256=normalizer_sha,
        data_audit_final_manifest_sha256=sha256_file(audit_path),
        smoke_result_sha256=sha256_file(smoke_path),
    )

    validated_audit, validated_smoke = trainer._validate_a2_bindings(
        contract,
        audit_manifest_path=audit_path,
        normalizer_arrays_path=normalizer,
        smoke_result_path=smoke_path,
        source_manifest=source,
    )
    assert validated_audit == audit
    assert validated_smoke == smoke_payload
    with pytest.raises(ValueError, match="authorize"):
        trainer._validate_a2_bindings(
            contract,
            audit_manifest_path=audit_path,
            normalizer_arrays_path=normalizer,
            smoke_result_path=smoke_path,
            source_manifest={"schema": "drift"},
        )


def test_smoke_width_ladder_requires_a_closed_memory_failure(tmp_path: Path) -> None:
    source = {"schema": "source", "canonical_payload_sha256": "2" * 64}
    prior: dict[str, object] = {
        "schema": smoke.SCHEMA,
        "status": "insufficient_headroom",
        "passes_full_grid_step": True,
        "passes_memory_gate": False,
        "configuration": {"layers": [128] * 5},
        "test_object_opened": False,
        "dataset_array_opened": False,
        "source_manifest": source,
        "prior_width128_result_sha256": None,
    }
    prior["canonical_payload_sha256"] = canonical_json_sha256(prior)
    path = tmp_path / "width128.json"
    path.write_text(json.dumps(prior, sort_keys=True), encoding="utf-8")
    assert smoke.validate_width_ladder(128, None, source_manifest=source) is None
    assert smoke.validate_width_ladder(96, path, source_manifest=source) == sha256_file(
        path
    )

    changed = dict(prior)
    changed["status"] = "failed"
    changed["canonical_payload_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "canonical_payload_sha256"
        }
    )
    path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        smoke.validate_width_ladder(96, path, source_manifest=source)


def test_source_manifest_covers_all_transitive_planardet_entrypoints() -> None:
    manifest = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    paths = {row["path"] for row in manifest["files"]}
    assert set(EXECUTABLE_ENTRYPOINTS).issubset(paths)
    assert {
        "pcno/pcno.py",
        "utility/time_dependent_no/realm_pcno.py",
        "utility/time_dependent_no/realm_planardet.py",
        "utility/time_dependent_no/realm_planardet_metrics.py",
        "utility/time_dependent_no/realm_planardet_runtime.py",
    }.issubset(paths)
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "canonical_payload_sha256"
    }
    assert manifest["canonical_payload_sha256"] == canonical_json_sha256(unsigned)


def test_output_isolation_and_registered_stop_guards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "output"
    assert not trainer._prepare_output_directory(
        output,
        data_root=data,
        resume_checkpoint=None,
    )
    last = output / "last.pt"
    last.write_bytes(b"checkpoint")
    assert trainer._prepare_output_directory(
        output,
        data_root=data,
        resume_checkpoint=last,
    )
    with pytest.raises(ValueError, match="registered pre-final"):
        trainer._validate_stop_after_step(51, _contract())
    assert trainer._validate_stop_after_step(50, _contract()) == 50


def test_evaluator_prefix_and_gap_helpers_preserve_censoring() -> None:
    assert evaluator._accepted_prefix([True, True, False, True]) == 2
    assert evaluator._accepted_prefix([True, True]) == 2
    gap = evaluator._free_minus_teacher(
        {"npe_total_by_call": [0.2, None, 0.7]},
        {"npe_total_by_call": [0.1, 0.3, 0.4]},
    )
    assert gap == pytest.approx([0.1, None, 0.3], nan_ok=True)


@pytest.mark.parametrize("entrypoint", (trainer.main, evaluator.main, smoke.main))
def test_planardet_gpu_entrypoint_help_is_safe(
    entrypoint: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        entrypoint(["--help"])  # type: ignore[operator]
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out

"""Evaluate the single retained PlanarDet PCNO checkpoint on open validation.

The evaluator executes fresh truth-input pairs, their ordered teacher-forced
view, and one free recurrence with the same checkpoint and decoder.  It accepts
only a complete closed training directory and the exact open train/validation
tree; no test object or checkpoint-selection option is exposed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    parse_manifest_payload,
    predict_one_call,
)
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
)
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_COORDINATE_ORDER,
    DOMAIN_LENGTHS_XY,
    PLANARDET_OPEN_MANIFEST_SHA256,
    PLANARDET_TRAIN_GROUPS,
    PLANARDET_VAL_GROUPS,
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
    structured_state_sha256,
)
from utility.time_dependent_no.realm_planardet_metrics import (
    compare_planardet_structure,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    CHANNELS,
    EXPECTED_PARAMETER_COUNTS,
    FC_DIM,
    MODE_COUNTS_XY,
    VALIDATION_HORIZON,
    PlanarDetNormalizerBundle,
    PlanarDetTrainingContract,
    competence_gate,
    load_normalized_trajectories,
    load_normalizer_bundle,
    planardet_boundary_mask,
    summarize_planardet_predictions,
    validate_frozen_controls,
    validation_control_summary,
)

DEVICE_NAME = "cuda:0"
DTYPE = torch.float32
BEST_CHECKPOINT_SCHEMA = "w26_l4_planardet_pd0_a3_best_v1"
TRAINING_STATUS_SCHEMA = "w26_l4_planardet_pd0_a3_status_v1"
TRAINING_FINAL_MANIFEST_SCHEMA = (
    "w26_l4_planardet_pd0_a3_training_final_hash_manifest_v1"
)
TRAINING_FILES = frozenset(
    {
        "preregistration.json",
        "config.json",
        "input_manifest.json",
        "source_manifest.json",
        "runtime_manifest.json",
        "history.json",
        "best.pt",
        "last.pt",
        "status.json",
        "final_hash_manifest.json",
        "summary.json",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON input: {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"evaluation output already exists: {path.name}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_signed_payload(payload: Mapping[str, Any], *, name: str) -> None:
    digest = payload.get("canonical_payload_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    if digest != canonical_json_sha256(unsigned):
        raise ValueError(f"{name} canonical payload digest differs")


def _write_npy(path: Path, values: torch.Tensor) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"evaluation array already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"evaluation array temporary exists: {path.name}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, values.detach().cpu().numpy(), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _prepare_output_directory(
    output_dir: Path,
    *,
    training_dir: Path,
    data_root: Path,
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("evaluation output directory must be absent")
    if any(
        _is_within(output_dir, root) or _is_within(root, output_dir)
        for root in (training_dir, data_root)
    ):
        raise ValueError(
            "evaluation output, training output, and data root must be disjoint"
        )
    output_dir.mkdir(parents=True)


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _runtime_manifest(device: torch.device, *, seed: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a3_evaluation_cuda_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_closed_training_tree(
    training_dir: Path,
    *,
    current_source_manifest: Mapping[str, Any],
) -> tuple[
    PlanarDetTrainingContract,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    if training_dir.is_symlink() or not training_dir.is_dir():
        raise ValueError("training directory must be an existing non-symlink directory")
    actual: set[str] = set()
    for path in training_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("training directory contains a non-regular object")
        if any(
            part.casefold() == "test" for part in path.relative_to(training_dir).parts
        ):
            raise ValueError("training directory contains a sealed test path")
        actual.add(path.name)
    if actual != TRAINING_FILES:
        raise ValueError("training directory inventory differs from the closed schema")

    final_manifest = _load_json(training_dir / "final_hash_manifest.json")
    expected_hashed = TRAINING_FILES - {"final_hash_manifest.json", "summary.json"}
    files = final_manifest.get("files")
    if (
        final_manifest.get("schema") != TRAINING_FINAL_MANIFEST_SCHEMA
        or final_manifest.get("self_hash_excluded") is not True
        or final_manifest.get("test_object_opened") is not False
        or not isinstance(files, Mapping)
        or set(files) != expected_hashed
    ):
        raise ValueError("training final hash manifest differs")
    for name, expected_sha256 in files.items():
        if sha256_file(training_dir / name) != expected_sha256:
            raise ValueError(f"closed training artifact hash differs: {name}")

    status = _load_json(training_dir / "status.json")
    summary = _load_json(training_dir / "summary.json")
    if (
        status.get("schema") != TRAINING_STATUS_SCHEMA
        or status.get("complete") is not True
        or status.get("test_object_opened") is not False
        or summary.get("final_hash_manifest_sha256")
        != sha256_file(training_dir / "final_hash_manifest.json")
        or {
            key: value
            for key, value in summary.items()
            if key != "final_hash_manifest_sha256"
        }
        != status
    ):
        raise ValueError("training status is incomplete or inconsistent")
    preregistration = _load_json(training_dir / "preregistration.json")
    contract = PlanarDetTrainingContract.from_payload(preregistration)
    config = _load_json(training_dir / "config.json")
    if config != contract.frozen_training_config():
        raise ValueError("training config differs from preregistration")
    source_manifest = _load_json(training_dir / "source_manifest.json")
    if source_manifest != current_source_manifest:
        raise ValueError("current executable sources differ from closed training")
    input_manifest = _load_json(training_dir / "input_manifest.json")
    runtime_manifest = _load_json(training_dir / "runtime_manifest.json")
    for name, payload in (
        ("training config", config),
        ("training source manifest", source_manifest),
        ("training input manifest", input_manifest),
        ("training runtime manifest", runtime_manifest),
    ):
        _validate_signed_payload(payload, name=name)
    return contract, status, input_manifest, config


def _validate_best_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    contract: PlanarDetTrainingContract,
    status: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    training_dir: Path,
    bundle: PlanarDetNormalizerBundle,
) -> Mapping[str, Any]:
    source = _load_json(training_dir / "source_manifest.json")
    runtime = _load_json(training_dir / "runtime_manifest.json")
    provenance = {
        "config_digest": config["canonical_payload_sha256"],
        "input_digest": input_manifest["canonical_payload_sha256"],
        "source_digest": source["canonical_payload_sha256"],
        "runtime_digest": runtime["canonical_payload_sha256"],
    }
    run_signature = canonical_json_sha256(provenance)
    model_state = checkpoint.get("model_state")
    normalizer_state = checkpoint.get("normalizer_state")
    expected_model_config = {
        "channels": CHANNELS,
        "mode_counts_xy": MODE_COUNTS_XY,
        "layers": (contract.width,) * 5,
        "fc_dim": FC_DIM,
        "activation": "gelu",
        "zero_initialize_head": True,
    }
    if (
        checkpoint.get("schema") != BEST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != contract.run_id
        or checkpoint.get("run_signature") != run_signature
        or checkpoint.get("completed_step") != status.get("best_step")
        or checkpoint.get("selection_value")
        != status.get("best_teacher_forced_realm_npe_mean")
        or checkpoint.get("provenance") != provenance
        or checkpoint.get("resume_supported") is not False
        or checkpoint.get("test_object_opened") is not False
        or checkpoint.get("model_config") != expected_model_config
        or not isinstance(model_state, Mapping)
        or structured_state_sha256(model_state) != checkpoint.get("model_state_sha256")
        or not isinstance(normalizer_state, Mapping)
        or structured_state_sha256(normalizer_state)
        != checkpoint.get("normalizer_state_sha256")
        or structured_state_sha256(bundle.checkpoint_state())
        != checkpoint.get("normalizer_state_sha256")
    ):
        raise ValueError("retained best checkpoint identity or provenance differs")
    return model_state


def _predict_truth_inputs(
    model: RealmRegularGridPCNO,
    truth_normalized: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[bool], list[float]]:
    predictions = torch.full_like(truth_normalized[1:], torch.nan)
    finite: list[bool] = []
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            current = truth_normalized[frame].unsqueeze(0).to(device)
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposal = predict_one_call(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            is_finite = bool(torch.isfinite(proposal).all())
            finite.append(is_finite)
            predictions[frame].copy_(proposal[0].to(device="cpu", dtype=DTYPE))
    return predictions, finite, call_seconds


def _predict_free_recurrence(
    model: RealmRegularGridPCNO,
    initial: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int | None, list[float]]:
    predictions = torch.empty(
        (VALIDATION_HORIZON, *initial.shape), dtype=DTYPE, device="cpu"
    )
    current = initial.unsqueeze(0).to(device)
    nonfinite_call: int | None = None
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposal = predict_one_call(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            if not bool(torch.isfinite(proposal).all()):
                nonfinite_call = frame + 1
                predictions = predictions[:frame]
                break
            predictions[frame].copy_(proposal[0].to(device="cpu", dtype=DTYPE))
            current = proposal
    return predictions, nonfinite_call, call_seconds


def _accepted_prefix(finite: Sequence[bool]) -> int:
    for index, value in enumerate(finite):
        if not value:
            return index
    return len(finite)


def _view_summary(
    predictions: torch.Tensor,
    truth_normalized: torch.Tensor,
    truth_native: torch.Tensor,
    current_native: torch.Tensor,
    *,
    bundle: PlanarDetNormalizerBundle,
    boundary_mask: torch.Tensor,
) -> tuple[dict[str, Any] | None, torch.Tensor]:
    if predictions.shape[0] == 0:
        return None, torch.empty_like(predictions)
    normalizer = bundle.normalizer(1, dtype=DTYPE)
    decoded = normalizer.decode(predictions, inverse_domain_policy="nan")
    summary = summarize_planardet_predictions(
        predictions,
        truth_normalized[: predictions.shape[0]],
        decoded,
        truth_native[: predictions.shape[0]],
        current_native[: predictions.shape[0]],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    return summary, decoded


def _structure_summary(
    decoded_predictions: torch.Tensor,
    validation_native: torch.Tensor,
    *,
    metadata_times: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any] | None:
    calls = decoded_predictions.shape[0]
    if calls == 0:
        return None
    predicted_sequence = torch.cat(
        (validation_native[:1], decoded_predictions), dim=0
    ).numpy()
    truth_sequence = validation_native[: calls + 1].numpy()
    return compare_planardet_structure(
        predicted_sequence,
        truth_sequence,
        x=np.asarray(x),
        y=np.asarray(y),
        times=np.asarray(metadata_times[: calls + 1]),
    )


def _free_minus_teacher(
    free: Mapping[str, Any] | None,
    teacher: Mapping[str, Any] | None,
) -> list[float | None]:
    if free is None or teacher is None:
        return []
    free_values = free.get("npe_total_by_call")
    teacher_values = teacher.get("npe_total_by_call")
    if not isinstance(free_values, list) or not isinstance(teacher_values, list):
        raise TypeError("view summaries lack per-call normalized errors")
    result: list[float | None] = []
    for free_value, teacher_value in zip(free_values, teacher_values, strict=False):
        if isinstance(free_value, (int, float)) and isinstance(
            teacher_value, (int, float)
        ):
            difference = float(free_value) - float(teacher_value)
            result.append(difference if math.isfinite(difference) else None)
        else:
            result.append(None)
    return result


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    contract, training_status, training_inputs, training_config = (
        _validate_closed_training_tree(
            args.training_dir,
            current_source_manifest=source_manifest,
        )
    )
    manifest_payload = _load_json(args.manifest)
    if canonical_json_sha256(manifest_payload) != contract.open_manifest_payload_sha256:
        raise ValueError("open manifest payload differs from preregistration")
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    if (
        manifest_summary["manifest_sha256"] != PLANARDET_OPEN_MANIFEST_SHA256
        or metadata.train_groups != PLANARDET_TRAIN_GROUPS
        or metadata.val_groups != PLANARDET_VAL_GROUPS
        or training_inputs.get("open_manifest_payload_sha256")
        != canonical_json_sha256(manifest_payload)
        or training_inputs.get("normalizer_arrays_sha256")
        != contract.normalizer_arrays_sha256
        or training_inputs.get("test_object_opened") is not False
    ):
        raise ValueError("evaluation inputs differ from closed training inputs")
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=contract.normalizer_arrays_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    normalizer = bundle.normalizer(1, dtype=DTYPE)
    validation_normalized, validation_native_batched = load_normalized_trajectories(
        args.data_root,
        split="val",
        groups=PLANARDET_VAL_GROUPS,
        normalizer=normalizer,
        retain_native=True,
    )
    if validation_native_batched is None:
        raise RuntimeError("validation native truth was not retained")
    controls = validation_control_summary(validation_normalized[0])
    validate_frozen_controls(controls, contract)

    checkpoint = torch.load(
        args.training_dir / "best.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("retained checkpoint root must be a mapping")
    model_state = _validate_best_checkpoint(
        checkpoint,
        contract=contract,
        status=training_status,
        input_manifest=training_inputs,
        config=training_config,
        training_dir=args.training_dir,
        bundle=bundle,
    )

    _configure_determinism(contract.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("evaluation requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime_manifest = _runtime_manifest(device, seed=contract.seed)

    geometry = build_realm_regular_grid_geometry(
        metadata.canonical_coords_yx,
        domain_lengths_xy=DOMAIN_LENGTHS_XY,
        released_coordinate_order=CANONICAL_COORDINATE_ORDER,
    )
    model = RealmRegularGridPCNO(
        config=RealmPCNOConfig(
            channels=CHANNELS,
            mode_counts_xy=MODE_COUNTS_XY,
            layers=(contract.width,) * 5,
            fc_dim=FC_DIM,
            zero_initialize_head=True,
        ),
        geometry=geometry,
    ).to(device=device, dtype=DTYPE)
    model.load_state_dict(model_state, strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNTS[contract.width]:
        raise RuntimeError("evaluation PCNO parameter count differs")
    coordinates = bundle.canonical_coordinates_yx.unsqueeze(0).to(
        device=device, dtype=DTYPE
    )
    validation_truth = validation_normalized[0]
    validation_native = validation_native_batched[0]
    boundary_mask = planardet_boundary_mask(
        torch.from_numpy(metadata.x.copy()), torch.from_numpy(metadata.y.copy())
    )

    started_at = time.monotonic()
    teacher_predictions, teacher_finite, teacher_call_seconds = _predict_truth_inputs(
        model,
        validation_truth,
        coordinates,
        device=device,
    )
    teacher_prefix = _accepted_prefix(teacher_finite)
    fresh_summary, _teacher_decoded_all = _view_summary(
        teacher_predictions,
        validation_truth[1:],
        validation_native[1:],
        validation_native[:-1],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )
    ordered_summary, teacher_decoded_prefix = _view_summary(
        teacher_predictions[:teacher_prefix],
        validation_truth[1 : teacher_prefix + 1],
        validation_native[1 : teacher_prefix + 1],
        validation_native[:teacher_prefix],
        bundle=bundle,
        boundary_mask=boundary_mask,
    )

    free_predictions, free_nonfinite_call, free_call_seconds = _predict_free_recurrence(
        model,
        validation_truth[0],
        coordinates,
        device=device,
    )
    free_normalizer = bundle.normalizer(1, dtype=DTYPE)
    free_decoded_for_current = free_normalizer.decode(
        free_predictions, inverse_domain_policy="nan"
    )
    free_current_native = torch.cat(
        (validation_native[:1], free_decoded_for_current[:-1]), dim=0
    )
    free_summary, free_decoded = _view_summary(
        free_predictions,
        validation_truth[1 : free_predictions.shape[0] + 1],
        validation_native[1 : free_predictions.shape[0] + 1],
        free_current_native,
        bundle=bundle,
        boundary_mask=boundary_mask,
    )

    if fresh_summary is None:
        raise RuntimeError("fresh-pair evaluation unexpectedly has no attempted calls")
    truth_input_gate = competence_gate(fresh_summary, contract)
    structure = {
        "ordered_teacher_forced": _structure_summary(
            teacher_decoded_prefix,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
        "free_recurrence": _structure_summary(
            free_decoded,
            validation_native,
            metadata_times=metadata.times,
            x=metadata.x,
            y=metadata.y,
        ),
    }

    _prepare_output_directory(
        args.output_dir,
        training_dir=args.training_dir,
        data_root=args.data_root,
    )
    teacher_array_path = args.output_dir / "teacher_prediction_normalized.npy"
    free_array_path = args.output_dir / "free_prediction_normalized.npy"
    _write_npy(teacher_array_path, teacher_predictions)
    _write_npy(free_array_path, free_predictions)
    arrays = {
        "teacher_prediction_normalized.npy": {
            "sha256": sha256_file(teacher_array_path),
            "shape": list(teacher_predictions.shape),
            "dtype": str(teacher_predictions.numpy().dtype),
            "view_ownership": ["fresh_pairs", "ordered_teacher_forced"],
        },
        "free_prediction_normalized.npy": {
            "sha256": sha256_file(free_array_path),
            "shape": list(free_predictions.shape),
            "dtype": str(free_predictions.numpy().dtype),
            "view_ownership": ["free_recurrence"],
        },
    }
    closure_gates = {
        "closed_training_verified": True,
        "source_manifest_matches_training": True,
        "open_tree_verified": len(inventory) == len(entries),
        "fresh_pairs_all_attempted": len(teacher_finite) == VALIDATION_HORIZON,
        "ordered_teacher_censoring_recorded": teacher_prefix
        == _accepted_prefix(teacher_finite),
        "free_censoring_recorded": free_predictions.shape[0]
        == (
            VALIDATION_HORIZON
            if free_nonfinite_call is None
            else free_nonfinite_call - 1
        ),
        "prediction_arrays_persisted": True,
        "test_object_opened": False,
    }
    closure_gates["all_gates_pass"] = (
        all(
            value for key, value in closure_gates.items() if key != "test_object_opened"
        )
        and closure_gates["test_object_opened"] is False
    )
    result: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a3_open_validation_evaluation_v1",
        "run_id": contract.run_id,
        "training_run_signature": training_status["run_signature"],
        "checkpoint": {
            "relative_path": "best.pt",
            "sha256": sha256_file(args.training_dir / "best.pt"),
            "completed_step": checkpoint["completed_step"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "selection_value": checkpoint["selection_value"],
        },
        "validation_case": PLANARDET_VAL_GROUPS[0],
        "controls": controls,
        "views": {
            "fresh_pairs": {
                "aggregation": "49 independent truth-input adjacent pairs",
                "attempted_calls": len(teacher_finite),
                "finite_call_mask": teacher_finite,
                "call_seconds": teacher_call_seconds,
                "summary": fresh_summary,
            },
            "ordered_teacher_forced": {
                "aggregation": "one ordered trajectory with truth restored before every call",
                "valid_length": teacher_prefix,
                "censored_from_call": (
                    teacher_prefix + 1 if teacher_prefix < VALIDATION_HORIZON else None
                ),
                "summary": ordered_summary,
            },
            "free_recurrence": {
                "aggregation": "one trajectory initialized only from released frame zero",
                "valid_length": free_predictions.shape[0],
                "nonfinite_proposal_call": free_nonfinite_call,
                "call_seconds": free_call_seconds,
                "summary": free_summary,
            },
        },
        "free_minus_teacher_npe_by_call": _free_minus_teacher(
            free_summary, ordered_summary
        ),
        "structure": structure,
        "truth_input_competence_gate": truth_input_gate,
        "evaluator_closure": closure_gates,
        "mechanism_interpretation_allowed": bool(
            truth_input_gate["all_gates_pass"] and closure_gates["all_gates_pass"]
        ),
        "prediction_arrays": arrays,
        "timing_seconds": {
            "evaluation": time.monotonic() - started_at,
            "teacher_total": sum(teacher_call_seconds),
            "free_total": sum(free_call_seconds),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "anti_claims": [
            "this is open validation and not sealed test evidence",
            "fresh-versus-free differences establish propagated-input necessity, not a training-mechanism cause",
            "pMax structure metrics use an independent declared algorithm, not an exact paper implementation",
            "released fields do not support a physical conservation claim",
            "one checkpoint, seed, and validation condition do not establish a generic architecture failure",
        ],
    }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    _write_json(args.output_dir / "result.json", result)
    _write_json(args.output_dir / "source_manifest.json", source_manifest)
    _write_json(args.output_dir / "runtime_manifest.json", runtime_manifest)
    final_manifest = {
        "schema": "w26_l4_planardet_pd0_a3_evaluation_final_hash_manifest_v1",
        "files": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "teacher_prediction_normalized.npy",
                "free_prediction_normalized.npy",
                "result.json",
                "source_manifest.json",
                "runtime_manifest.json",
            )
        },
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    summary = {
        "schema": "w26_l4_planardet_pd0_a3_evaluation_summary_v1",
        "run_id": contract.run_id,
        "truth_input_competence_pass": truth_input_gate["all_gates_pass"],
        "evaluator_closure_pass": closure_gates["all_gates_pass"],
        "mechanism_interpretation_allowed": result["mechanism_interpretation_allowed"],
        "teacher_valid_length": teacher_prefix,
        "free_valid_length": free_predictions.shape[0],
        "result_sha256": sha256_file(args.output_dir / "result.json"),
        "final_hash_manifest_sha256": sha256_file(
            args.output_dir / "final_hash_manifest.json"
        ),
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_evaluation(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

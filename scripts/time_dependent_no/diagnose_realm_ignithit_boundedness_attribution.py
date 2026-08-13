"""Run the exact D091 matched IgnitHIT boundedness attribution diagnostic.

This is inference-only. It accepts the exact terminal D089 and D090 artifact
inventories, the frozen open IgnitHIT inputs, and a new isolated output
directory. Real arrays, checkpoints, CUDA, or remote execution require a
separate authorization beyond D091 A1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import (
    capture_realm_ignithit_domain_link_failure as d090,
)
from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_FIELDS,
    MagnitudeEnvelope,
    canonical_json_sha256,
)
from utility.time_dependent_no.realm_boundedness_attribution import (
    attribute_free_first_events,
    channel_envelope_records,
    evaluate_normalized_modes,
    summarize_attribution_verdict,
    summarize_envelope_events,
    validate_recurrence_trace,
)
from utility.time_dependent_no.realm_ffno import (
    normalize_realm_coordinates,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    BOUNDEDNESS_EXPANSION_FACTOR,
    BOUNDEDNESS_QUANTILE,
    VAL_GROUPS,
    sha256_file,
)

d089 = d090.d089
parent = d090.parent
replay_support = d090.replay_support

STABLE_ID = "D091"
RUN_ID = "d091_realm_ignithit_p1d_boundedness_attribution_20260813a"
HORIZON = 29
STEP50_ID = "d089_step50_last"
STEP100_ID = "d090_step100_diagnostic"

D090_RUN_SIGNATURE = (
    "f9012ab69410f418c9405bce2f69a8f68597241cc44f5374d3021e1d2e5aa04b"
)
D090_MODEL_FILE_SHA256 = (
    "ee6bd1e6dc21897fb602ab0cb024f36190fe2b83dc20854b30727070d5e801b0"
)
D090_MODEL_STATE_SHA256 = (
    "1af738ece2b2f4167e51905cd67622a052c0fc7b0e96714251130af0745c2f93"
)
D090_FINAL_MANIFEST_SHA256 = (
    "521e9312a8fcb3951beaa5ba6ccb138fd0b7ef2953c0b0cbae0afdfb091e1247"
)
D090_FILE_SHA256 = {
    "contract.json": (
        "a8344eea3ee25197275d9a9037e8e56af59ca7d9935502130845e8aa4e3ac82a"
    ),
    "eligibility.json": (
        "4280ff1f511f896348ab805e9f35abaa5d12385f6bda2b27158ed274d06233f7"
    ),
    "final_hash_manifest.json": D090_FINAL_MANIFEST_SHA256,
    "input_manifest.json": (
        "87e212a322790d78de480b4b338fbf5f3c34ae3247b2062d273899e1179758f4"
    ),
    "parent_identity.json": (
        "fe7045e1c3096a6d285da0571d21c67c84e11166e7ceb7209f1bea84014ea9fb"
    ),
    "replay_trace.json": (
        "3f17d9f0f8d14abc49e97472e937b52b3429afbeea4e44f00e3bc78c3283042b"
    ),
    "runtime_manifest.json": (
        "3067ee9f3d954460f3cab96d7820466ed23723591269c1237d247f822cf33265"
    ),
    "source_manifest.json": (
        "7237393c5329b924862d84a6248790e7dc1703dada3bf44be1c97538a6adb647"
    ),
    "stage_manifest.json": (
        "b2051e125b7bd0b190cf44112b18c5c58df8f24ebbe484b19809c2d8c300d9b2"
    ),
    "step100_model.pt": D090_MODEL_FILE_SHA256,
    "summary.json": (
        "e17432d874c79093da2a5aa06ffc900f6927392374014aef6bcfe2e03c6fab6d"
    ),
    "validation_row.json": (
        "7c3ceefffa82d4a64df313ba7b0bce4bf366307e13f218004d8e1ff0960d8304"
    ),
}

D090_STAGE_EXPECTED = {
    "frame0_normalized_sha256": (
        "c8bca37b03a880c69576b32a16b1d6b69fff80994c89ca466ccb5f30d0afabc9"
    ),
    "truth_normalized_sha256": (
        "05569939cc5ae72e55784d2d5b43736077bf426330a6535946a7b3ea5c94ad15"
    ),
    "truth_native_sha256": (
        "c5313ea185329a0ab5ea15009d83c3b8bde1e79058825feb487efacd01b9ebc1"
    ),
    "coordinates_sha256": (
        "a6681c1ec5e82e0be30c75b468738d084f2fe713016f83229ec18c843d36802d"
    ),
    "free_normalized_sha256": (
        "c10f2e2db30d3e417f16aebe4b58b702c2dab3bfcfe46c92d48d785ffda5e13e"
    ),
    "free_decoded_sha256": (
        "77db67759bfd1fee18e9eebef87375487fc99e6593d5134fc6716daf04b4672b"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--d089-output-dir", type=Path, required=True)
    parser.add_argument("--d090-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-digest", required=True)
    return parser


def diagnostic_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "d091_realm_boundedness_attribution_contract_v1",
        "stable_id": STABLE_ID,
        "run_id": RUN_ID,
        "checkpoint_ids": [STEP50_ID, STEP100_ID],
        "checkpoint_model_state_sha256": {
            STEP50_ID: d090.PARENT_LAST_MODEL_STATE_SHA256,
            STEP100_ID: D090_MODEL_STATE_SHA256,
        },
        "ordered_validation_groups": list(VAL_GROUPS),
        "ordered_channels": list(IGNITHIT_FIELDS),
        "start_frame": 0,
        "calls": HORIZON,
        "modes": ["free_recurrence", "teacher_forced"],
        "teacher_forced_input": "exact_normalized_truth_frame_h_minus_1",
        "free_recurrence": "current=deployed_domain_linked_proposal",
        "envelope_quantile": BOUNDEDNESS_QUANTILE,
        "envelope_expansion_factor": BOUNDEDNESS_EXPANSION_FACTOR,
        "boundedness_threshold_inclusive": True,
        "first_event_order": ["case", "call", "channel", "row", "column"],
        "spatial_support": (
            "strict_abs_gt_inclusive_limit_count_and_minimum_row_column_bbox"
        ),
        "accepted_prefix": "first_failure_call_minus_one_else_H29_censored",
        "attribution_verdict": (
            "event_channel_counts_with_mixed_and_censored_outcomes_preserved"
        ),
        "finite_unbounded_feedback": "unchanged_deployed_proposal",
        "normalized_nonfinite_policy": "domain_link_raises_before_deployment",
        "decoded_nonfinite_feedback": "finite_normalized_deployed_state_is_unchanged",
        "selection": "none",
        "training": "none",
        "retry": "forbidden",
        "output_repair": "none",
        "test_object_available": False,
        "residual_arm_authorized": False,
        "planardet_authorized": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_exact_inventory(
    root: Path, expected: Mapping[str, str], *, name: str
) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"{name} directory is missing")
    actual_paths = tuple(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in actual_paths):
        raise ValueError(f"{name} must contain only regular files")
    actual_names = {path.name for path in actual_paths}
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        unexpected = sorted(actual_names - set(expected))
        raise ValueError(
            f"{name} inventory differs; missing={missing}, unexpected={unexpected}"
        )
    return replay_support.validate_file_hashes(root, expected)


def _is_within(child: Path, parent_dir: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent_dir.resolve(strict=False)
    return child_resolved == parent_resolved or parent_resolved in child_resolved.parents


def prepare_new_output_directory(
    output_dir: Path,
    *,
    protected_dirs: Sequence[Path],
    protected_files: Sequence[Path],
) -> None:
    resolved_output = output_dir.resolve(strict=False)
    for protected in protected_dirs:
        if _is_within(output_dir, protected) or _is_within(protected, output_dir):
            raise ValueError("D091 output and protected directory must not overlap")
    for protected in protected_files:
        if resolved_output == protected.resolve(strict=False):
            raise ValueError("D091 output and protected file must differ")
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("D091 output directory must be absent or empty")
    else:
        output_dir.mkdir(parents=True)


def source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_domain_link.py",
        REPO_ROOT
        / "utility"
        / "time_dependent_no"
        / "realm_boundedness_attribution.py",
        REPO_ROOT / "scripts" / "time_dependent_no" / "train_realm_ignithit_ffno.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "train_realm_ignithit_domain_linked_ffno.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "diagnose_realm_ignithit_decode_failure.py",
        REPO_ROOT
        / "scripts"
        / "time_dependent_no"
        / "capture_realm_ignithit_domain_link_failure.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d091_realm_boundedness_attribution_source_v1",
        "files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the exact registered identity")


def validate_step50_checkpoint(
    checkpoint: Mapping[str, Any], history_file: Mapping[str, Any]
) -> None:
    _require_equal(
        checkpoint.get("schema"), d089.LAST_CHECKPOINT_SCHEMA, name="D089 schema"
    )
    _require_equal(checkpoint.get("run_id"), d089.RUN_ID, name="D089 run ID")
    _require_equal(
        checkpoint.get("run_signature"),
        d090.PARENT_RUN_SIGNATURE,
        name="D089 run signature",
    )
    _require_equal(checkpoint.get("completed_step"), 50, name="D089 completed step")
    _require_equal(
        checkpoint.get("model_state_sha256"),
        d090.PARENT_LAST_MODEL_STATE_SHA256,
        name="D089 model-state declaration",
    )
    _require_equal(
        parent.structured_state_sha256(checkpoint.get("model_state")),
        d090.PARENT_LAST_MODEL_STATE_SHA256,
        name="D089 model state",
    )
    _require_equal(
        checkpoint.get("provenance"), d090.PARENT_PROVENANCE, name="D089 provenance"
    )
    _require_equal(checkpoint.get("resume_supported"), True, name="D089 resume role")
    _require_equal(checkpoint.get("best_step"), 1, name="D089 best step")
    _require_equal(
        checkpoint.get("best_score"), d090.PARENT_BEST_SCORE, name="D089 best score"
    )
    _require_equal(
        checkpoint.get("best_model_state_sha256"),
        d090.PARENT_BEST_MODEL_STATE_SHA256,
        name="D089 best model state",
    )
    _require_equal(
        checkpoint.get("history"), history_file.get("rows"), name="D089 history"
    )


def validate_step100_checkpoint(
    checkpoint: Mapping[str, Any], normalizer_state: Mapping[str, Any]
) -> None:
    _require_equal(
        checkpoint.get("schema"), d090.CAPTURE_CHECKPOINT_SCHEMA, name="D090 schema"
    )
    _require_equal(checkpoint.get("run_id"), d090.RUN_ID, name="D090 run ID")
    _require_equal(
        checkpoint.get("diagnostic_run_signature"),
        D090_RUN_SIGNATURE,
        name="D090 run signature",
    )
    _require_equal(
        checkpoint.get("model_state_sha256"),
        D090_MODEL_STATE_SHA256,
        name="D090 model-state declaration",
    )
    _require_equal(
        parent.structured_state_sha256(checkpoint.get("model_state")),
        D090_MODEL_STATE_SHA256,
        name="D090 model state",
    )
    checkpoint_normalizer = checkpoint.get("normalizer_state")
    if not isinstance(checkpoint_normalizer, Mapping):
        raise TypeError("D090 checkpoint normalizer_state must be a mapping")
    normalizer_digest = parent.structured_state_sha256(normalizer_state)
    _require_equal(
        parent.structured_state_sha256(checkpoint_normalizer),
        normalizer_digest,
        name="D090 embedded normalizer state",
    )
    _require_equal(
        checkpoint.get("normalizer_state_sha256"),
        normalizer_digest,
        name="D090 normalizer-state declaration",
    )
    _require_equal(checkpoint.get("diagnostic_only"), True, name="D090 role")
    _require_equal(checkpoint.get("selection_eligible"), False, name="D090 selection")
    _require_equal(checkpoint.get("resume_supported"), False, name="D090 resume")
    _require_equal(checkpoint.get("test_object_opened"), False, name="D090 test flag")


def _load_model(
    checkpoint: Mapping[str, Any], normalizer_state: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    model = d089.build_model(normalizer_state).to(device=device, dtype=parent.DTYPE)
    if trainable_parameter_count(model) != 8_936_460:
        raise RuntimeError("D091 model parameter count differs")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    model.load_state_dict(state, strict=True)
    return model


def _mode_payload(
    *,
    checkpoint_id: str,
    mode: str,
    decoded: torch.Tensor,
    envelope: MagnitudeEnvelope,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = channel_envelope_records(
        decoded,
        envelope,
        expansion_factor=BOUNDEDNESS_EXPANSION_FACTOR,
        case_keys=VAL_GROUPS,
        channel_names=IGNITHIT_FIELDS,
        checkpoint_id=checkpoint_id,
        mode=mode,
    )
    events = summarize_envelope_events(
        records,
        case_keys=VAL_GROUPS,
        channel_names=IGNITHIT_FIELDS,
        checkpoint_id=checkpoint_id,
        mode=mode,
        horizon=decoded.shape[1],
    )
    return records, events


def run_attribution(args: argparse.Namespace) -> dict[str, Any]:
    validate_exact_inventory(
        args.d089_output_dir, d090.PARENT_FILE_SHA256, name="D089 parent"
    )
    validate_exact_inventory(args.d090_output_dir, D090_FILE_SHA256, name="D090 parent")
    prepare_new_output_directory(
        args.output_dir,
        protected_dirs=(args.data_root, args.d089_output_dir, args.d090_output_dir),
        protected_files=(args.manifest, args.normalizer_arrays),
    )
    if (
        not isinstance(args.expected_source_digest, str)
        or len(args.expected_source_digest) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_source_digest)
    ):
        raise ValueError("expected source digest must be 64 lowercase hex characters")
    contract = diagnostic_contract()
    source = source_manifest()
    _require_equal(
        source["canonical_payload_sha256"],
        args.expected_source_digest,
        name="reviewed D091 executed source",
    )

    d089_input = parent._load_json(args.d089_output_dir / "input_manifest.json")
    d089_config = parent._load_json(args.d089_output_dir / "config.json")
    d089_source = parent._load_json(args.d089_output_dir / "source_manifest.json")
    d089_runtime = parent._load_json(args.d089_output_dir / "runtime_manifest.json")
    d089_history = parent._load_json(args.d089_output_dir / "history.json")
    d090_validation_row = parent._load_json(
        args.d090_output_dir / "validation_row.json"
    )
    d090_summary = parent._load_json(args.d090_output_dir / "summary.json")
    d090_stage = parent._load_json(args.d090_output_dir / "stage_manifest.json")

    manifest_payload = parent._load_json(args.manifest)
    metadata, current_input = replay_support._current_input_manifest(
        manifest_payload, data_root=args.data_root
    )
    parent._configure_determinism()
    device = torch.device(parent.DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D091 requires exactly one visible CUDA device")
    torch.cuda.set_device(device)

    with d089.activated_contract():
        (
            state_normalizer,
            trajectory_normalizer,
            train_max_abs,
            normalizer_state,
        ) = parent._normalizers_from_arrays(args.normalizer_arrays)
        _require_equal(
            parent.frozen_training_contract(), d089_config, name="D089 config"
        )
        _require_equal(current_input, d089_input, name="D089 input manifest")
        _require_equal(parent._source_manifest(), d089_source, name="D089 source")
        current_runtime = parent._runtime_manifest(device)
        _require_equal(current_runtime, d089_runtime, name="D089 runtime")

        validation_normalized, validation_native = parent._load_normalized_trajectories(
            args.data_root,
            "val",
            VAL_GROUPS,
            state_normalizer,
            retain_native=True,
        )
        if validation_native is None:
            raise RuntimeError("D091 matching validation truth was not retained")
        validation_normalized = validation_normalized.to(device)
        validation_native = validation_native.to(device)
        coordinates = normalize_realm_coordinates(
            torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=parent.DTYPE)
        ).to(device)

        step50_checkpoint = torch.load(
            args.d089_output_dir / "last.pt", map_location="cpu", weights_only=False
        )
        step100_checkpoint = torch.load(
            args.d090_output_dir / "step100_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(step50_checkpoint, Mapping) or not isinstance(
            step100_checkpoint, Mapping
        ):
            raise TypeError("registered checkpoint roots must be mappings")
        validate_step50_checkpoint(step50_checkpoint, d089_history)
        validate_step100_checkpoint(step100_checkpoint, normalizer_state)

        truth_normalized = validation_normalized[:, 1 : HORIZON + 1]
        truth_decoded = validation_native[:, 1 : HORIZON + 1]
        envelope = MagnitudeEnvelope(
            max_abs=train_max_abs.to(device=device, dtype=parent.DTYPE),
            quantile=BOUNDEDNESS_QUANTILE,
            channel_axis=0,
        )
        truth_records, truth_events = _mode_payload(
            checkpoint_id="matching_truth",
            mode="matching_truth",
            decoded=truth_decoded,
            envelope=envelope,
        )

        ratio_records: dict[str, Any] = {
            "schema": "d091_realm_boundedness_ratio_records_v1",
            "truth": truth_records,
            "checkpoints": {},
        }
        event_summary: dict[str, Any] = {
            "schema": "d091_realm_boundedness_event_summary_v1",
            "truth": truth_events,
            "checkpoints": {},
        }
        stage_manifest: dict[str, Any] = {
            "schema": "d091_realm_boundedness_stage_manifest_v1",
            "frame0_normalized_sha256": parent.structured_state_sha256(
                validation_normalized[:, 0]
            ),
            "truth_normalized_sha256": parent.structured_state_sha256(
                truth_normalized
            ),
            "truth_native_sha256": parent.structured_state_sha256(truth_decoded),
            "coordinates_sha256": parent.structured_state_sha256(coordinates),
            "checkpoints": {},
        }
        replay_equivalence: dict[str, Any] = {}
        checkpoints = (
            (STEP50_ID, step50_checkpoint),
            (STEP100_ID, step100_checkpoint),
        )
        for checkpoint_id, checkpoint in checkpoints:
            model = _load_model(checkpoint, normalizer_state, device)
            predictions = evaluate_normalized_modes(
                model,
                validation_normalized,
                coordinates,
                calls=HORIZON,
            )
            if predictions.free_recurrence.shape[1] != HORIZON:
                raise RuntimeError(
                    f"{checkpoint_id} returned native nonfiniteness before H29"
                )
            validate_recurrence_trace(
                predictions.free_inputs,
                predictions.free_recurrence,
            )
            if not torch.equal(
                predictions.teacher_inputs,
                validation_normalized[:, :HORIZON],
            ):
                raise RuntimeError("teacher-forced inputs differ from exact truth")
            free_decoded = trajectory_normalizer.decode(
                predictions.free_recurrence, inverse_domain_policy="nan"
            )
            teacher_decoded = trajectory_normalizer.decode(
                predictions.teacher_forced, inverse_domain_policy="nan"
            )
            free_records, free_events = _mode_payload(
                checkpoint_id=checkpoint_id,
                mode="free_recurrence",
                decoded=free_decoded,
                envelope=envelope,
            )
            teacher_records, teacher_events = _mode_payload(
                checkpoint_id=checkpoint_id,
                mode="teacher_forced",
                decoded=teacher_decoded,
                envelope=envelope,
            )
            attribution = attribute_free_first_events(
                free_records,
                teacher_records,
                case_keys=VAL_GROUPS,
                channel_names=IGNITHIT_FIELDS,
                checkpoint_id=checkpoint_id,
                horizon=HORIZON,
            )
            ratio_records["checkpoints"][checkpoint_id] = {
                "free_recurrence": free_records,
                "teacher_forced": teacher_records,
            }
            event_summary["checkpoints"][checkpoint_id] = {
                "free_recurrence": free_events,
                "teacher_forced": teacher_events,
                "attribution": attribution,
            }
            existing_summary = parent.summarize_validation_predictions(
                predictions.free_recurrence,
                truth_normalized,
                free_decoded,
                truth_decoded,
                case_keys=VAL_GROUPS,
                train_max_abs=train_max_abs,
            )
            existing_summary.update(d089.domain_link_diagnostics(model))
            if checkpoint_id == STEP50_ID:
                rows = d089_history.get("rows")
                if not isinstance(rows, list) or not rows:
                    raise ValueError("D089 history rows are missing")
                expected_summary = rows[-1].get("validation")
            else:
                expected_summary = d090_validation_row.get("validation")
            _require_equal(
                existing_summary,
                expected_summary,
                name=f"{checkpoint_id} free replay summary",
            )
            expected_bounded_calls = sum(
                int(row["bounded_calls"]) for row in existing_summary["per_case"]
            )
            _require_equal(
                free_events["bounded_case_calls"],
                expected_bounded_calls,
                name=f"{checkpoint_id} bounded call count",
            )
            _require_equal(
                free_events["maximum_ratio"],
                existing_summary["max_boundedness_ratio"],
                name=f"{checkpoint_id} maximum envelope ratio",
            )
            replay_equivalence[checkpoint_id] = {
                "passed": True,
                "paper_compatible_free_recurrence": {
                    "realm_npe_mean": existing_summary["realm_npe_mean"],
                    "npe_total_case_first_by_call": existing_summary[
                        "npe_total_case_first_by_call"
                    ],
                    "npe_group_case_first_by_call": existing_summary[
                        "npe_group_case_first_by_call"
                    ],
                },
                "bounded_case_calls": free_events["bounded_case_calls"],
                "maximum_ratio": free_events["maximum_ratio"],
            }
            stage_manifest["checkpoints"][checkpoint_id] = {
                "model_state_sha256": checkpoint["model_state_sha256"],
                "free_input_normalized_sha256": parent.structured_state_sha256(
                    predictions.free_inputs
                ),
                "free_normalized_sha256": parent.structured_state_sha256(
                    predictions.free_recurrence
                ),
                "free_decoded_sha256": parent.structured_state_sha256(free_decoded),
                "teacher_forced_normalized_sha256": parent.structured_state_sha256(
                    predictions.teacher_forced
                ),
                "teacher_input_normalized_sha256": parent.structured_state_sha256(
                    predictions.teacher_inputs
                ),
                "teacher_forced_decoded_sha256": parent.structured_state_sha256(
                    teacher_decoded
                ),
                "shape": list(predictions.free_recurrence.shape),
                "dtype": str(predictions.free_recurrence.dtype),
                "device": str(predictions.free_recurrence.device),
                "recurrence_exact": True,
            }
            del model

    for key in (
        "frame0_normalized_sha256",
        "truth_normalized_sha256",
        "truth_native_sha256",
        "coordinates_sha256",
    ):
        _require_equal(stage_manifest[key], D090_STAGE_EXPECTED[key], name=key)
    step100_stage = stage_manifest["checkpoints"][STEP100_ID]
    _require_equal(
        step100_stage["free_normalized_sha256"],
        D090_STAGE_EXPECTED["free_normalized_sha256"],
        name="D090 free normalized prediction",
    )
    _require_equal(
        step100_stage["free_decoded_sha256"],
        D090_STAGE_EXPECTED["free_decoded_sha256"],
        name="D090 free decoded prediction",
    )
    _require_equal(
        d090_stage.get("canonical_payload_sha256"),
        "6ccdf8e2f0e99dd2b742718388e1dd7f4340df950ed57388f3a5db81c8d30911",
        name="D090 stage manifest",
    )
    _require_equal(
        d090_summary.get("run_signature"), D090_RUN_SIGNATURE, name="D090 summary"
    )

    checkpoint_identity = {
        "schema": "d091_realm_boundedness_checkpoint_identity_v1",
        STEP50_ID: {
            "file_sha256": d090.PARENT_LAST_SHA256,
            "model_state_sha256": d090.PARENT_LAST_MODEL_STATE_SHA256,
            "completed_step": 50,
            "selection_role": "retained_last_nonbest",
        },
        STEP100_ID: {
            "file_sha256": D090_MODEL_FILE_SHA256,
            "model_state_sha256": D090_MODEL_STATE_SHA256,
            "completed_step": 100,
            "selection_role": "diagnostic_only_nonselectable",
        },
        "d089_run_signature": d090.PARENT_RUN_SIGNATURE,
        "d090_run_signature": D090_RUN_SIGNATURE,
    }
    stage_manifest["canonical_payload_sha256"] = canonical_json_sha256(stage_manifest)
    attribution_verdict = summarize_attribution_verdict(
        {
            checkpoint_id: payload["attribution"]
            for checkpoint_id, payload in event_summary["checkpoints"].items()
        }
    )
    summary: dict[str, Any] = {
        "schema": "d091_realm_boundedness_attribution_summary_v1",
        "stable_id": STABLE_ID,
        "run_id": RUN_ID,
        "contract_digest": contract["canonical_payload_sha256"],
        "source_digest": source["canonical_payload_sha256"],
        "runtime_digest": current_runtime["canonical_payload_sha256"],
        "stage_manifest_digest": stage_manifest["canonical_payload_sha256"],
        "replay_equivalence": replay_equivalence,
        "attribution_verdict": attribution_verdict,
        "checkpoint_count": 2,
        "mode_count": 2,
        "case_count": len(VAL_GROUPS),
        "call_count": HORIZON,
        "channel_count": len(IGNITHIT_FIELDS),
        "training_executed": False,
        "selection_executed": False,
        "retry_executed": False,
        "output_repair_used": False,
        "test_object_opened": False,
        "residual_arm_executed": False,
        "planardet_executed": False,
        "claim_boundary": (
            "exact inference-input attribution on the frozen open validation "
            "population; no training-factor or benchmark-wide causal claim"
        ),
    }
    payloads = (
        ("contract.json", contract),
        ("checkpoint_identity.json", checkpoint_identity),
        ("input_manifest.json", current_input),
        ("source_manifest.json", source),
        ("runtime_manifest.json", current_runtime),
        ("stage_manifest.json", stage_manifest),
        ("ratio_records.json", ratio_records),
        ("event_summary.json", event_summary),
        ("summary.json", summary),
    )
    for name, payload in payloads:
        parent._write_json_atomic(args.output_dir / name, payload)
    retained_names = tuple(name for name, _ in payloads)
    final_manifest = {
        "schema": "d091_realm_boundedness_final_hash_manifest_v1",
        "files": {
            name: sha256_file(args.output_dir / name) for name in retained_names
        },
        "self_hash_excluded": True,
    }
    parent._write_json_atomic(
        args.output_dir / "final_hash_manifest.json", final_manifest
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_attribution(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

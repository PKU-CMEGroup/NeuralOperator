#!/usr/bin/env python3
"""Continue trajectory 05 from the exact retained D041 failed proposal.

Calls 1--33 are loaded verbatim from the retained D041 trajectory artifact.
Only calls 34--79 are executed by the current digest-bound checkpoint/source.
The recurrence ignores inadmissibility but never modifies the state.

This is a descriptive numerical-map diagnostic, not an admissible Euler
rollout or a literal replay of the unrecovered dirty D041 runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_pcno_euler2d_unchecked_rollout import (  # noqa: E402
    AMPLITUDE_EXPLOSION_RATIO,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_D041_TRAJECTORY05_SHA256,
    EXPECTED_FIRST_EXCLUDED_CALL,
    EXPECTED_RECURRENT_CALLS,
    EXPECTED_SOURCE_FRAMES,
    EXPECTED_TRAJECTORY,
    GLOBAL_ERROR_EXPLOSION_THRESHOLD,
    build_call_rows,
    corner_geometry_summary,
    first_call,
    json_safe,
    make_visualizations,
    runtime_identity,
    sha256_file,
    unchecked_rollout,
    write_csv,
    write_json,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    build_model,
    load_checkpoint,
    preprocessing_contract_audit,
    select_device,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
)

SCHEMA = "pcno_euler2d_spliced_unchecked_post_admissibility_v1"
CONTINUATION_FIRST_CALL = EXPECTED_FIRST_EXCLUDED_CALL + 1
CONTINUATION_CALLS = EXPECTED_RECURRENT_CALLS - EXPECTED_FIRST_EXCLUDED_CALL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d041-trajectory-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-key", default=EXPECTED_TRAJECTORY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--animation-dpi", type=int, default=105)
    parser.add_argument("--animation-format", choices=("mp4", "gif"), default="gif")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if str(args.trajectory_key) != EXPECTED_TRAJECTORY:
        raise ValueError("this registered diagnostic is restricted to trajectory 05")
    if args.fps <= 0.0 or args.animation_dpi < 50:
        raise ValueError("fps must be positive and animation dpi at least 50")
    for path in (
        args.checkpoint,
        args.d041_trajectory_artifact,
        args.data_dir,
        args.training_data_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_digest = sha256_file(args.checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint digest {checkpoint_digest} does not match the D041 checkpoint"
        )
    artifact_digest = sha256_file(args.d041_trajectory_artifact)
    if artifact_digest != EXPECTED_D041_TRAJECTORY05_SHA256:
        raise ValueError(
            f"D041 artifact digest {artifact_digest} does not match the retained bundle"
        )


def load_d041_splice(path: Path) -> dict[str, Any]:
    required = {
        "schema",
        "trajectory_key",
        "checkpoint_sha256",
        "checkpoint_config_digest",
        "test_manifest_digest",
        "training_manifest_digest",
        "boundary_mode",
        "baseline_boundary_mode",
        "baseline_valid_length",
        "baseline_failure_call",
        "baseline_failure_cause",
        "initial_conservative",
        "reference_targets_conservative",
        "pcno_baseline_predictions_conservative",
        "baseline_failed_proposal",
        "positions",
        "edges",
        "node_type",
        "reconstructed_node_weights_proxy",
        "physical_target_times",
        "physical_delta_t",
    }
    with np.load(path, allow_pickle=False) as bundle:
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"D041 splice artifact is missing fields: {missing}")
        scalar = {
            name: bundle[name].item() for name in required if bundle[name].ndim == 0
        }
        arrays = {
            name: np.array(bundle[name], copy=True)
            for name in required
            if bundle[name].ndim != 0
        }
    if str(scalar["schema"]) != "pcno_euler2d_official_rollout_v1":
        raise ValueError("unsupported D041 trajectory artifact schema")
    if str(scalar["trajectory_key"]) != EXPECTED_TRAJECTORY:
        raise ValueError("D041 splice artifact is not trajectory 05")
    if str(scalar["checkpoint_sha256"]) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("D041 splice artifact binds a different checkpoint")
    if (
        str(scalar["boundary_mode"]) != "model_all_nodes"
        or str(scalar["baseline_boundary_mode"]) != "model_all_nodes"
    ):
        raise ValueError("D041 splice prefix is not the raw model_all_nodes baseline")
    if (
        int(scalar["baseline_valid_length"]) != 32
        or int(scalar["baseline_failure_call"]) != EXPECTED_FIRST_EXCLUDED_CALL
        or str(scalar["baseline_failure_cause"]) != "inadmissible_state"
    ):
        raise ValueError("D041 trajectory 05 failure contract has changed")

    initial = np.asarray(arrays["initial_conservative"], dtype=np.float32)
    targets = np.asarray(arrays["reference_targets_conservative"], dtype=np.float32)
    accepted = np.asarray(
        arrays["pcno_baseline_predictions_conservative"], dtype=np.float32
    )
    failed = np.asarray(arrays["baseline_failed_proposal"], dtype=np.float32)
    if initial.ndim != 2 or initial.shape[-1] != 4:
        raise ValueError("D041 initial state must have shape [N,4]")
    if targets.shape != (EXPECTED_RECURRENT_CALLS, *initial.shape):
        raise ValueError("D041 reference targets do not contain calls 1--79")
    if accepted.shape != (32, *initial.shape) or failed.shape != initial.shape:
        raise ValueError("D041 accepted prefix and failed proposal do not align")
    prefix = np.concatenate((accepted, failed[None]), axis=0)
    references = np.concatenate((initial[None], targets), axis=0)
    return {
        "initial": initial,
        "references": references,
        "prefix": prefix,
        "failed": failed,
        "positions": np.asarray(arrays["positions"], dtype=np.float32),
        "edges": np.asarray(arrays["edges"], dtype=np.int64),
        "node_type": np.asarray(arrays["node_type"], dtype=np.int64),
        "node_weights": np.asarray(
            arrays["reconstructed_node_weights_proxy"], dtype=np.float32
        ),
        "physical_target_times": np.asarray(
            arrays["physical_target_times"], dtype=np.float64
        ),
        "dt": float(scalar["physical_delta_t"]),
        "checkpoint_config_digest": str(scalar["checkpoint_config_digest"]),
        "test_manifest_digest": str(scalar["test_manifest_digest"]),
        "training_manifest_digest": str(scalar["training_manifest_digest"]),
    }


def assemble_spliced_predictions(
    prefix: np.ndarray, continuation: np.ndarray
) -> np.ndarray:
    frozen = np.asarray(prefix, dtype=np.float32)
    suffix = np.asarray(continuation, dtype=np.float32)
    if frozen.ndim != 3 or frozen.shape[0] != EXPECTED_FIRST_EXCLUDED_CALL:
        raise ValueError("the frozen prefix must contain calls 1--33")
    if suffix.shape != (CONTINUATION_CALLS, *frozen.shape[1:]):
        raise ValueError("the continuation must contain calls 34--79")
    result = np.concatenate((frozen, suffix), axis=0)
    if result.shape[0] != EXPECTED_RECURRENT_CALLS:
        raise ValueError("the spliced sequence must contain calls 1--79")
    if not np.array_equal(result[:EXPECTED_FIRST_EXCLUDED_CALL], frozen):
        raise ValueError("concatenation changed the frozen D041 prefix")
    return result


def exact_device_state(
    failed: np.ndarray, *, template: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = np.asarray(failed, dtype=np.float32)
    if template.dtype != torch.float32:
        raise ValueError("the D041 recurrence state contract requires float32")
    current = torch.as_tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    observed = current[0].detach().cpu().numpy()
    delta = np.asarray(observed, dtype=np.float64) - np.asarray(
        source, dtype=np.float64
    )
    max_abs = float(np.max(np.abs(delta)))
    exact = bool(np.array_equal(observed, source))
    if not exact or max_abs != 0.0:
        raise ValueError(
            f"D041 call-33 CPU-device-CPU round trip is not exact: {max_abs:.3e}"
        )
    return current, {
        "dtype": str(current.dtype),
        "shape": list(current.shape),
        "cpu_device_cpu_exact": exact,
        "cpu_device_cpu_max_abs": max_abs,
    }


def _assert_exact_array(name: str, observed: np.ndarray, expected: np.ndarray) -> None:
    left = np.asarray(observed)
    right = np.asarray(expected)
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"current shard {name} does not exactly match D041 artifact")


def _first_threshold_call(
    rows: Sequence[Mapping[str, Any]], field: str, threshold: float
) -> int | None:
    return next(
        (
            int(row["call"])
            for row in rows
            if row[field] is not None and float(row[field]) >= threshold
        ),
        None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    splice = load_d041_splice(args.d041_trajectory_artifact)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    if int(checkpoint["step_stride"]) != 1:
        raise ValueError("the D041 checkpoint must use model step stride 1")
    if str(checkpoint["config_digest"]) != splice["checkpoint_config_digest"]:
        raise ValueError("checkpoint config digest does not match D041 artifact")

    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    preprocessing = preprocessing_contract_audit(checkpoint, training_store, test_store)
    if training_store.manifest_digest != splice["training_manifest_digest"]:
        raise ValueError("training manifest digest does not match D041 artifact")
    if test_store.manifest_digest != splice["test_manifest_digest"]:
        raise ValueError("test manifest digest does not match D041 artifact")

    key = str(args.trajectory_key)
    if key not in test_store.keys:
        raise KeyError(f"test shard is missing trajectory {key}")
    states = np.asarray(test_store.states(key), dtype=np.float32)
    if states.shape[0] != EXPECTED_SOURCE_FRAMES:
        raise ValueError("trajectory 05 must contain exactly 80 source states")
    _assert_exact_array("states", states, splice["references"])
    positions = np.array(test_store.array(key, "nodes"), copy=True)
    edges = np.array(test_store.array(key, "edges"), copy=True)
    node_type = np.array(test_store.array(key, "node_type"), copy=True).reshape(-1)
    node_weights = np.array(test_store.array(key, "node_weights"), copy=True)
    directed_edges = np.array(test_store.array(key, "directed_edges"), copy=True)
    gradient_weights = np.array(
        test_store.array(key, "edge_gradient_weights"), copy=True
    )
    _assert_exact_array("positions", positions, splice["positions"])
    _assert_exact_array("edges", edges, splice["edges"])
    _assert_exact_array("node_type", node_type, splice["node_type"])
    _assert_exact_array(
        "reconstructed node weights",
        node_weights.reshape(-1),
        splice["node_weights"].reshape(-1),
    )
    if float(test_store.manifest["dt"]) != splice["dt"]:
        raise ValueError("current shard time step does not match D041 artifact")

    geometry = corner_geometry_summary(
        positions, node_type, node_weights, directed_edges, gradient_weights
    )
    if geometry["index"] != 6 or geometry["node_type"] != 3:
        raise ValueError("trajectory 05 upper-left corner contract has changed")

    model = build_model(checkpoint, device)
    sample = dict(test_store.tensor_sample(key, 0, step_stride=1, device=device))
    current, roundtrip = exact_device_state(
        splice["failed"], template=sample["current"], device=device
    )
    sample["current"] = current
    rollout = unchecked_rollout(
        model,
        sample,
        num_steps=CONTINUATION_CALLS,
        device=device,
    )
    predictions = assemble_spliced_predictions(splice["prefix"], rollout["predictions"])
    call_seconds = [float("nan")] * EXPECTED_FIRST_EXCLUDED_CALL + list(
        rollout["call_seconds"]
    )
    rows = build_call_rows(
        predictions,
        splice["references"],
        splice["initial"],
        node_weights,
        np.asarray(checkpoint["normalization"]["state_scale"], dtype=np.float64),
        gamma=float(checkpoint["normalization"]["gamma"]),
        dt=splice["dt"],
        node_index=int(geometry["index"]),
        call_seconds=call_seconds,
    )
    first_inadmissible = first_call(rows, "inadmissible_node_count")
    if first_inadmissible != EXPECTED_FIRST_EXCLUDED_CALL:
        raise ValueError(
            f"spliced sequence first becomes inadmissible at {first_inadmissible}, not 33"
        )

    events = {
        "first_inadmissible_call": first_inadmissible,
        "first_nonpositive_internal_energy_call": first_call(
            rows, "nonpositive_internal_energy_node_count"
        ),
        "first_nonpositive_pressure_call": first_call(
            rows, "nonpositive_pressure_node_count"
        ),
        "first_nonpositive_density_call": first_call(
            rows, "nonpositive_density_node_count"
        ),
        "first_nonfinite_conservative_call": first_call(
            rows, "nonfinite_conservative_node_count"
        ),
        "first_100x_reference_amplitude_call": _first_threshold_call(
            rows,
            "max_abs_conservative_to_reference_max_ratio",
            AMPLITUDE_EXPLOSION_RATIO,
        ),
        "first_global_proxy_l2_at_least_10_call": _first_threshold_call(
            rows,
            "scaled_relative_l2_reconstructed_weight_proxy",
            GLOBAL_ERROR_EXPLOSION_THRESHOLD,
        ),
    }

    args.output_dir.mkdir(parents=True)
    trajectory_path = args.output_dir / "trajectory05_spliced_unchecked_raw.npz"
    np.savez_compressed(
        trajectory_path,
        schema=np.asarray(SCHEMA),
        trajectory_key=np.asarray(key),
        initial_conservative=splice["initial"],
        reference_states_conservative=splice["references"],
        spliced_predictions_conservative=predictions,
        frozen_d041_prefix_conservative=splice["prefix"],
        current_source_continuation_conservative=rollout["predictions"],
        positions=positions,
        edges=edges,
        node_type=node_type,
        reconstructed_node_weights_proxy=node_weights,
        physical_times=np.arange(EXPECTED_SOURCE_FRAMES, dtype=np.float64)
        * splice["dt"],
        checkpoint_sha256=np.asarray(EXPECTED_CHECKPOINT_SHA256),
        source_state_frames=np.asarray(EXPECTED_SOURCE_FRAMES, dtype=np.int64),
        recurrent_calls=np.asarray(EXPECTED_RECURRENT_CALLS, dtype=np.int64),
        frozen_prefix_calls=np.asarray(EXPECTED_FIRST_EXCLUDED_CALL, dtype=np.int64),
        newly_executed_calls=np.asarray(CONTINUATION_CALLS, dtype=np.int64),
        encoded_animation_frames=np.asarray(EXPECTED_SOURCE_FRAMES, dtype=np.int64),
    )
    call_metrics_path = args.output_dir / "call_metrics.csv"
    write_csv(call_metrics_path, rows)
    splice_path = args.output_dir / "splice_contract.csv"
    write_csv(
        splice_path,
        [
            {
                "frozen_prefix_first_call": 1,
                "frozen_prefix_last_call": EXPECTED_FIRST_EXCLUDED_CALL,
                "continuation_first_call": CONTINUATION_FIRST_CALL,
                "continuation_last_call": EXPECTED_RECURRENT_CALLS,
                "frozen_prefix_calls": EXPECTED_FIRST_EXCLUDED_CALL,
                "newly_executed_calls": CONTINUATION_CALLS,
                "cpu_device_cpu_exact": roundtrip["cpu_device_cpu_exact"],
                "cpu_device_cpu_max_abs": roundtrip["cpu_device_cpu_max_abs"],
            }
        ],
    )
    visual_paths = make_visualizations(
        args.output_dir,
        predictions,
        splice["references"],
        splice["initial"],
        positions,
        node_type,
        directed_edges,
        rows,
        gamma=float(checkpoint["normalization"]["gamma"]),
        fps=args.fps,
        animation_dpi=args.animation_dpi,
        animation_format=args.animation_format,
        upper_left_index=int(geometry["index"]),
    )

    runtime = runtime_identity(device)
    runtime["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        sha256_file(Path(__file__))
    )
    summary = {
        "schema": SCHEMA,
        "scientific_role": (
            "current bound checkpoint/source continuation from the exact retained "
            "D041 failed state; not an admissible Euler rollout, literal dirty-runtime "
            "replay, method comparison, or conservation claim"
        ),
        "trajectory": key,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_config_digest": checkpoint["config_digest"],
        "boundary_mode": "model_all_nodes",
        "raw_conservative_recurrence": True,
        "state_modifications": {
            "positivity_stop": False,
            "floor": False,
            "clipping": False,
            "smoothing": False,
            "boundary_replacement": False,
            "future_reference_boundary": False,
        },
        "frame_contract": {
            "source_state_frames": EXPECTED_SOURCE_FRAMES,
            "requested_physical_horizon_transitions": EXPECTED_RECURRENT_CALLS,
            "model_step_stride": 1,
            "frozen_d041_prefix_calls": EXPECTED_FIRST_EXCLUDED_CALL,
            "newly_executed_recurrent_calls": CONTINUATION_CALLS,
            "new_execution_call_numbers": [
                CONTINUATION_FIRST_CALL,
                EXPECTED_RECURRENT_CALLS,
            ],
            "encoded_animation_frames": int(predictions.shape[0] + 1),
            "animation_format": args.animation_format,
            "visualization_subsampling": False,
            "includes_initial_state": True,
        },
        "splice_contract": {
            "artifact_sha256": EXPECTED_D041_TRAJECTORY05_SHA256,
            "prefix_origin": "retained D041 calls 1--32 plus failed proposal call 33",
            "prefix_recomputed": False,
            "call_33_device_roundtrip": roundtrip,
            "u0_replay_failure": {
                "call": 7,
                "max_abs": 1.431e-5,
                "relative_l2": 5.373e-8,
                "max_abs_tolerance": 1.0e-5,
                "relative_l2_tolerance": 1.0e-6,
            },
            "historical_dirty_runtime_identity_recovered": False,
        },
        "events": events,
        "blowup_contract": {
            "definitive_numerical_blowup": "any nonfinite conservative component",
            "severe_amplitude_explosion": (
                "finite max absolute conservative component reaches 100 times the "
                "maximum absolute reference component"
            ),
            "severe_global_error_explosion": (
                "reconstructed-weight scaled relative L2 reaches 10"
            ),
            "thresholds_are_diagnostic_not_physical": True,
        },
        "final_frame": rows[-1],
        "upper_left_corner": geometry,
        "preprocessing_contract": preprocessing,
        "runtime": runtime,
        "artifacts": {},
    }
    output_paths = [
        trajectory_path,
        call_metrics_path,
        splice_path,
        *visual_paths,
    ]
    summary["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_paths
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "events": events,
                "final": json_safe(rows[-1]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

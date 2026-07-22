#!/usr/bin/env python3
"""Test whether the frozen Line-4A decoder can reach sharp validation states.

This command performs declared per-state latent-code fitting with the encoder,
decoder weights, conservative moment channels, data cohort, and physical metric
contract frozen. It is representation-capacity evidence only: it contains no
latent transition, rollout, test access, filtering, or deployable analysis step.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    endpoint_metrics,
)
from scripts.time_dependent_no.train_euler2d_latent_representation import (  # noqa: E402
    SCHEMA as SOURCE_SCHEMA,
    SMOKE_FRAMES,
    SMOKE_TRAIN_TRAJECTORIES,
    SMOKE_VALIDATION_TRAJECTORIES,
    _append_jsonl,
    _load_geometry,
    _load_state_batch,
    _prepare_output,
    _primitive_overshoot,
    _select_device,
    _select_spread,
    _set_seed,
    _smoke_data_preflight,
    _synchronize,
    _write_json,
    _write_rows,
    audit_inputs,
)
from utility.time_dependent_no.latent_representation_2d import (  # noqa: E402
    LINE4_REPRESENTATION_SCHEMA,
    SpatialTokenAutoencoder,
    fit_frozen_decoder_code,
    reconstruction_metrics,
    repeat_token_geometry,
    representation_artifact_ledger,
    sha256_file,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DShardStore,
    digest_mapping,
)

SCHEMA = "line4a_frozen_decoder_code_reachability_v1"
FIXED_CHANNELS = 4
MAX_ITER = 250
MAX_EVAL = 320
HISTORY_SIZE = 20
CODE_SCALE_FLOOR = 1.0e-6
RECONSTRUCTION_L2_LIMIT = 0.0021
FRONT_RELATIVE_TOLERANCE = 0.05
CODE_NEIGHBORHOOD_RMS_LIMIT = 1.0
GPU_HOUR_LIMIT = 0.1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "oracle"), default="preflight")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args(argv)


def _load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    audit: Mapping[str, Any],
    normalization: Euler2DNormalization,
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("expected checkpoint SHA-256 must contain 64 hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("frozen representation checkpoint digest mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("representation checkpoint must contain a mapping")
    if checkpoint.get("schema") != SOURCE_SCHEMA or checkpoint.get("stage") != "smoke":
        raise ValueError("wrong frozen representation checkpoint schema or stage")
    if checkpoint.get("seed") is None:
        raise ValueError("frozen representation checkpoint is missing its seed")
    if checkpoint.get("data_manifest_digest") != audit["data_manifest_digest"]:
        raise ValueError("checkpoint and staged data manifest digests differ")
    if digest_mapping(checkpoint.get("normalization", {})) != digest_mapping(
        normalization.to_dict()
    ):
        raise ValueError("checkpoint and supplied normalization differ")
    if (
        checkpoint.get("test_keys") != []
        or checkpoint.get("test_arrays_read") is not False
    ):
        raise ValueError("checkpoint does not preserve the sealed test contract")
    if checkpoint.get("transition_present") is not False:
        raise ValueError("checkpoint unexpectedly contains a latent transition")
    if checkpoint.get("analysis_applied") is not False:
        raise ValueError("checkpoint unexpectedly contains an analysis intervention")
    if int(checkpoint.get("optimizer_steps_per_model", -1)) != 800:
        raise ValueError("checkpoint does not match the completed 800-step smoke")

    selected_train = _select_spread(train_keys, SMOKE_TRAIN_TRAJECTORIES)
    selected_validation = _select_spread(
        validation_keys,
        SMOKE_VALIDATION_TRAJECTORIES,
    )
    if checkpoint.get("train_keys") != selected_train:
        raise ValueError("checkpoint training cohort differs from the frozen smoke")
    if checkpoint.get("validation_keys") != selected_validation:
        raise ValueError("checkpoint validation cohort differs from the frozen smoke")
    contracts = checkpoint.get("model_contracts")
    states = checkpoint.get("model_states")
    if not isinstance(contracts, Mapping) or not isinstance(states, Mapping):
        raise ValueError("checkpoint lacks model contracts or states")
    contract = contracts.get("conservative_moment")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint lacks the conservative-moment model")
    if (
        contract.get("variant") != "conservative_moment"
        or int(contract.get("fixed_conservative_channels", -1)) != FIXED_CHANNELS
        or int(contract.get("latent_size", -1)) != 5000
        or contract.get("front_variables") != 0
        or contract.get("decode_reencode_projection") is not False
    ):
        raise ValueError("checkpoint conservative-moment contract changed")
    if "conservative_moment" not in states:
        raise ValueError("checkpoint lacks conservative-moment weights")
    return checkpoint, selected_train, selected_validation


def _build_frozen_model(
    checkpoint: Mapping[str, Any],
    normalization: Euler2DNormalization,
    device: torch.device,
) -> SpatialTokenAutoencoder:
    model = SpatialTokenAutoencoder(
        normalization,
        variant="conservative_moment",
    ).to(device)
    model.load_state_dict(
        checkpoint["model_states"]["conservative_moment"], strict=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def _training_free_code_scale(
    model: SpatialTokenAutoencoder,
    store: PCNOEuler2DShardStore,
    train_keys: Sequence[str],
    geometry,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    items = [(key, int(frame)) for key in train_keys for frame in SMOKE_FRAMES]
    free_codes: list[torch.Tensor] = []
    for start in range(0, len(items), 4):
        batch_items = items[start : start + 4]
        states = _load_state_batch(store, batch_items, device)
        batch_geometry = repeat_token_geometry(geometry, len(batch_items))
        code = model.encode(states, batch_geometry)
        free_codes.append(code[..., FIXED_CHANNELS:].reshape(-1, 16))
    stacked = torch.cat(free_codes, dim=0)
    raw_scale = stacked.std(dim=0, unbiased=False)
    floored = raw_scale < CODE_SCALE_FLOOR
    scale = raw_scale.clamp_min(CODE_SCALE_FLOOR)
    digest = hashlib.sha256(
        scale.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    return scale, {
        "training_trajectories": list(train_keys),
        "frames": list(SMOKE_FRAMES),
        "token_samples": int(stacked.shape[0]),
        "free_channels": int(stacked.shape[1]),
        "scale_floor": CODE_SCALE_FLOOR,
        "floored_channels": int(floored.sum().cpu()),
        "minimum_raw_scale": float(raw_scale.min().cpu()),
        "maximum_raw_scale": float(raw_scale.max().cpu()),
        "scale_sha256": digest,
    }


def _reconstruction_row(
    *,
    variant: str,
    trajectory: str,
    frame: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry,
    normalization: Euler2DNormalization,
    edges: np.ndarray,
    displacement_rms: float,
    per_state_fitting: bool,
) -> dict[str, Any]:
    prediction_numpy = prediction[0].detach().cpu().numpy()
    target_numpy = target[0].detach().cpu().numpy()
    return {
        "variant": variant,
        "trajectory": trajectory,
        "frame": frame,
        **reconstruction_metrics(
            prediction,
            target,
            geometry,
            state_scale=normalization.state_scale,
            gamma=normalization.gamma,
        ),
        **endpoint_metrics(
            prediction_numpy,
            target_numpy,
            positions=geometry.nodes[0].detach().cpu().numpy(),
            edges=edges,
            volumes=geometry.volumes[0].detach().cpu().numpy(),
            component_scale=normalization.state_scale,
            gamma=normalization.gamma,
            shock_quantile=0.9,
        ),
        "density_pressure_overshoot": _primitive_overshoot(
            prediction,
            target,
            gamma=normalization.gamma,
        ),
        "normalized_free_code_displacement_rms": displacement_rms,
        "per_state_fitting": per_state_fitting,
        "forecast_evidence": False,
        "contact_behavior": "not_present_in_frozen_shock_vortex_family",
        "intervention_applied": per_state_fitting,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    ignored = {
        "variant",
        "trajectory",
        "frame",
        "per_state_fitting",
        "forecast_evidence",
        "intervention_applied",
        "smooth_region_contract",
        "contact_behavior",
    }
    output: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate: dict[str, Any] = {
            "rows": len(selected),
            "completion_rate": 1.0,
            "intervention_applied": variant == "fitted_code",
        }
        for name in sorted(set().union(*(row.keys() for row in selected)) - ignored):
            values = [
                float(row[name])
                for row in selected
                if not isinstance(row.get(name), bool)
                and isinstance(row.get(name), (int, float, np.integer, np.floating))
                and math.isfinite(float(row[name]))
            ]
            if values:
                aggregate[f"mean_{name}"] = float(np.mean(values))
                aggregate[f"minimum_{name}"] = float(np.min(values))
                aggregate[f"maximum_{name}"] = float(np.max(values))
        output[variant] = aggregate
    return output


def classify_reachability(
    aggregates: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, bool]]:
    control = aggregates["amortized_encoder"]
    fitted = aggregates["fitted_code"]
    gates = {
        "mean_relative_l2_at_most_0p0021": (
            float(fitted["mean_relative_l2"]) <= RECONSTRUCTION_L2_LIMIT
        ),
        "all_states_raw_admissible": (
            float(fitted["minimum_admissible_fraction"]) >= 1.0
        ),
        "mean_shock_strength_within_5_percent": (
            abs(float(fitted["mean_shock_strength_ratio"]) - 1.0)
            <= FRONT_RELATIVE_TOLERANCE
        ),
        "mean_shock_thickness_within_5_percent": (
            abs(float(fitted["mean_shock_thickness_ratio"]) - 1.0)
            <= FRONT_RELATIVE_TOLERANCE
        ),
        "maximum_overshoot_not_increased": (
            float(fitted["maximum_density_pressure_overshoot"])
            <= float(control["maximum_density_pressure_overshoot"]) + 1.0e-8
        ),
        "all_fitted_codes_within_one_training_scale_rms": (
            float(fitted["maximum_normalized_free_code_displacement_rms"])
            <= CODE_NEIGHBORHOOD_RMS_LIMIT
        ),
    }
    physical_names = [name for name in gates if "training_scale" not in name]
    physical_pass = all(gates[name] for name in physical_names)
    if not physical_pass:
        return "decoder_manifold_rejected", gates
    if not gates["all_fitted_codes_within_one_training_scale_rms"]:
        return "off_manifold_decoder_capacity_only", gates
    return "amortized_encoder_defect_supported", gates


def _write_artifacts(
    output_dir: Path,
    *,
    trajectories: Sequence[str],
    frames: Sequence[int],
    targets: Sequence[np.ndarray],
    control_decoded: Sequence[np.ndarray],
    fitted_decoded: Sequence[np.ndarray],
    initial_codes: Sequence[np.ndarray],
    fitted_codes: Sequence[np.ndarray],
    displacements: Sequence[float],
) -> None:
    np.savez_compressed(
        output_dir / "physical_states.npz",
        schema=np.asarray(SCHEMA),
        split=np.asarray("validation"),
        trajectories=np.asarray(trajectories),
        frames=np.asarray(frames, dtype=np.int64),
        target=np.stack(targets),
        amortized_decoded=np.stack(control_decoded),
        fitted_decoded=np.stack(fitted_decoded),
    )
    np.savez_compressed(
        output_dir / "latent_states.npz",
        schema=np.asarray(SCHEMA),
        split=np.asarray("validation"),
        trajectories=np.asarray(trajectories),
        frames=np.asarray(frames, dtype=np.int64),
        initial_code=np.stack(initial_codes),
        fitted_code=np.stack(fitted_codes),
        normalized_free_code_displacement_rms=np.asarray(
            displacements,
            dtype=np.float64,
        ),
    )


def _run_oracle(
    args: argparse.Namespace,
    store: PCNOEuler2DShardStore,
    audit: Mapping[str, Any],
    normalization: Euler2DNormalization,
    checkpoint: Mapping[str, Any],
    selected_train: Sequence[str],
    selected_validation: Sequence[str],
) -> None:
    device = _select_device(args.device)
    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = _build_frozen_model(checkpoint, normalization, device)
    geometry = _load_geometry(store, selected_train[0], device)
    edges = np.array(
        store.array(selected_validation[0], "edges"),
        dtype=np.int64,
        copy=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = perf_counter()
    code_scale, scale_contract = _training_free_code_scale(
        model,
        store,
        selected_train,
        geometry,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    state_results: list[dict[str, Any]] = []
    trajectories: list[str] = []
    frames: list[int] = []
    targets: list[np.ndarray] = []
    control_decoded: list[np.ndarray] = []
    fitted_decoded: list[np.ndarray] = []
    initial_codes: list[np.ndarray] = []
    fitted_codes: list[np.ndarray] = []
    displacements: list[float] = []
    metrics_path = args.output_dir / "metrics.jsonl"

    for trajectory in selected_validation:
        for frame in SMOKE_FRAMES:
            target = _load_state_batch(store, [(trajectory, int(frame))], device)
            with torch.no_grad():
                initial_code = model.encode(target, geometry)
                control = model.decode(initial_code, geometry)
            fitted_code, optimization = fit_frozen_decoder_code(
                model,
                target,
                geometry,
                initial_code,
                fixed_channels=FIXED_CHANNELS,
                max_iter=MAX_ITER,
                max_eval=MAX_EVAL,
                history_size=HISTORY_SIZE,
            )
            with torch.no_grad():
                fitted = model.decode(fitted_code, geometry)
                standardized_delta = (
                    fitted_code[..., FIXED_CHANNELS:]
                    - initial_code[..., FIXED_CHANNELS:]
                ) / code_scale.view(1, 1, -1)
                displacement = float(
                    standardized_delta.square().mean().sqrt().detach().cpu()
                )

            for evaluation, loss in enumerate(optimization.pop("loss_trace"), start=1):
                _append_jsonl(
                    metrics_path,
                    {
                        "trajectory": trajectory,
                        "frame": int(frame),
                        "closure_evaluation": evaluation,
                        "scaled_volume_mse": loss,
                        "per_state_fitting": True,
                        "test_arrays_read": False,
                        "transition_present": False,
                        "analysis_applied": False,
                    },
                )
            state_results.append(
                {
                    "trajectory": trajectory,
                    "frame": int(frame),
                    "normalized_free_code_displacement_rms": displacement,
                    **optimization,
                }
            )
            rows.append(
                _reconstruction_row(
                    variant="amortized_encoder",
                    trajectory=trajectory,
                    frame=int(frame),
                    prediction=control,
                    target=target,
                    geometry=geometry,
                    normalization=normalization,
                    edges=edges,
                    displacement_rms=0.0,
                    per_state_fitting=False,
                )
            )
            rows.append(
                _reconstruction_row(
                    variant="fitted_code",
                    trajectory=trajectory,
                    frame=int(frame),
                    prediction=fitted,
                    target=target,
                    geometry=geometry,
                    normalization=normalization,
                    edges=edges,
                    displacement_rms=displacement,
                    per_state_fitting=True,
                )
            )
            trajectories.append(trajectory)
            frames.append(int(frame))
            targets.append(target[0].detach().cpu().numpy())
            control_decoded.append(control[0].detach().cpu().numpy())
            fitted_decoded.append(fitted[0].detach().cpu().numpy())
            initial_codes.append(initial_code[0].detach().cpu().numpy())
            fitted_codes.append(fitted_code[0].detach().cpu().numpy())
            displacements.append(displacement)

    _synchronize(device)
    wall_seconds = perf_counter() - start_time
    aggregates = _aggregate_rows(rows)
    classification, gates = classify_reachability(aggregates)
    _write_rows(args.output_dir / "reconstruction_rows.csv", rows)
    _write_rows(
        args.output_dir / "summary.csv",
        [{"variant": name, **values} for name, values in aggregates.items()],
    )
    _write_artifacts(
        args.output_dir,
        trajectories=trajectories,
        frames=frames,
        targets=targets,
        control_decoded=control_decoded,
        fitted_decoded=fitted_decoded,
        initial_codes=initial_codes,
        fitted_codes=fitted_codes,
        displacements=displacements,
    )

    gpu_hours = wall_seconds / 3600.0
    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "classification": classification,
        "hypothesis": (
            "the frozen conservative decoder contains a nearby sharp admissible "
            "code and the amortized encoder is the failed causal component"
        ),
        "smallest_matched_control": (
            "the frozen amortized encoder code for the same checkpoint, states, "
            "geometry, and raw decoder"
        ),
        "audit": dict(audit),
        "configuration": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "seed": args.seed,
            "split": "validation",
            "training_trajectories_for_code_scale": list(selected_train),
            "validation_trajectories": list(selected_validation),
            "frames": list(SMOKE_FRAMES),
            "states_fitted": len(state_results),
            "device": str(device),
            "fixed_channels": FIXED_CHANNELS,
            "free_channels": 16,
            "optimizer": {
                "name": "lbfgs_strong_wolfe",
                "learning_rate": 1.0,
                "max_iter": MAX_ITER,
                "max_eval": MAX_EVAL,
                "history_size": HISTORY_SIZE,
                "tolerance_grad": 1.0e-7,
                "tolerance_change": 1.0e-9,
            },
        },
        "stage_contract": {
            "checkpoint_frozen": True,
            "encoder_frozen": True,
            "decoder_frozen": True,
            "exact_moment_channels_fixed": True,
            "per_state_code_fitting": True,
            "front_loss": False,
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
            "transition_present": False,
            "rollout_present": False,
            "test_arrays_read": False,
            "analysis_applied": False,
            "forecast_evidence": False,
        },
        "code_neighborhood": scale_contract,
        "per_state_optimization": state_results,
        "reconstruction": aggregates,
        "promotion_gates": gates,
        "physical_reachability_pass": all(
            value for name, value in gates.items() if "training_scale" not in name
        ),
        "nearby_code_pass": gates["all_fitted_codes_within_one_training_scale_rms"],
        "encoder_remediation_eligible": (
            classification == "amortized_encoder_defect_supported"
        ),
        "promotion_eligible": False,
        "closure_evidence": False,
        "autonomous_forecast_evidence": False,
        "assimilation_evidence": False,
        "cost": {
            "wall_seconds": wall_seconds,
            "gpu_hours": gpu_hours,
            "gpu_hour_limit": GPU_HOUR_LIMIT,
            "within_budget": gpu_hours < GPU_HOUR_LIMIT,
            "maximum_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "failure_interpretation": {
            "decoder_manifold_rejected": (
                "per-state fitting cannot satisfy the frozen physical gates; stop "
                "the smooth decoder family without serious training"
            ),
            "off_manifold_decoder_capacity_only": (
                "sharp states are reachable only through distant codes; this does "
                "not isolate an amortized encoder or authorize a transition"
            ),
            "amortized_encoder_defect_supported": (
                "nearby sharp states are reachable; one separately authorized "
                "frozen-decoder encoder-remediation smoke may proceed"
            ),
        }[classification],
        "next_decision": (
            "review this mechanism artifact; serious representation training, "
            "latent transition training, test access, and filtering remain blocked"
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)

    base_ledger = representation_artifact_ledger(
        model,
        split="validation",
        trajectory="matched_validation_bank",
        physical_states_file="physical_states.npz",
        latent_states_file="latent_states.npz",
        valid_length=len(state_results),
        failure_cause=classification,
        seed=args.seed,
    )
    artifact_names = (
        "preflight.json",
        "metrics.jsonl",
        "reconstruction_rows.csv",
        "summary.csv",
        "summary.json",
        "physical_states.npz",
        "latent_states.npz",
    )
    ledger = {
        "schema": SCHEMA,
        "model": base_ledger,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "valid_length": len(state_results),
        "failure_cause": classification,
        "observation_contract": None,
        "ensemble_seeds": [],
        "analysis_applied": False,
        "diagnostic_interventions": {"per_state_code_fitting": True},
        "deployment_interventions": {
            "clipping": False,
            "floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
        },
        "forecast_evidence": False,
        "files": {
            name: {"sha256": sha256_file(args.output_dir / name)}
            for name in artifact_names
        },
    }
    _write_json(args.output_dir / "artifact_ledger.json", ledger)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _prepare_output(args.output_dir)
    device = _select_device(args.device)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=4)
    try:
        audit, normalization, train_keys, validation_keys = audit_inputs(
            store,
            handoff_path=args.handoff,
            normalization_path=args.normalization,
        )
        checkpoint, selected_train, selected_validation = _load_checkpoint(
            args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
            audit=audit,
            normalization=normalization,
            train_keys=train_keys,
            validation_keys=validation_keys,
        )
        audit = {
            **audit,
            "schema": SCHEMA,
            "source_representation_schema": LINE4_REPRESENTATION_SCHEMA,
            "source_checkpoint_schema": SOURCE_SCHEMA,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "smoke_data_readiness": _smoke_data_preflight(
                store,
                train_keys,
                validation_keys,
                device=device,
            ),
            "authorized_work": "frozen_decoder_code_reachability_only",
            "per_state_fitting": args.stage == "oracle",
            "forecast_evidence": False,
            "transition_present": False,
            "rollout_present": False,
            "test_arrays_read": False,
            "analysis_applied": False,
        }
        _write_json(args.output_dir / "preflight.json", audit)
        if args.stage == "preflight":
            return
        _run_oracle(
            args,
            store,
            audit,
            normalization,
            checkpoint,
            selected_train,
            selected_validation,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the bounded Line-4A 2D representation preflight or matched smoke test.

The command trains only two matched encoders/decoders. It has no latent
transition, rollout, test-split evaluation, joint fine-tuning, or filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
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
from utility.time_dependent_no.latent_representation_2d import (  # noqa: E402
    LINE4_REPRESENTATION_SCHEMA,
    SpatialTokenAutoencoder,
    build_token_geometry,
    conditional_future_diagnostics,
    empirical_decoder_gains,
    fit_channel_whitener,
    reconstruction_metrics,
    repeat_token_geometry,
    representation_artifact_ledger,
    scaled_volume_mse,
    sha256_file,
    validate_line3_handoff,
    volume_relative_l2,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DShardStore,
    conservative_to_primitive_torch,
    digest_mapping,
)

SCHEMA = "line4a_euler2d_matched_representation_smoke_v1"
SMOKE_STEPS = 800
SMOKE_BATCH_SIZE = 4
SMOKE_TRAIN_TRAJECTORIES = 4
SMOKE_VALIDATION_TRAJECTORIES = 2
SMOKE_FRAMES = (0, 15, 30, 45, 60)
SMOKE_EVALUATION_INTERVAL = 100
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "smoke"), default="preflight")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _manifest_splits(
    store: PCNOEuler2DShardStore,
) -> tuple[list[str], list[str], list[str]]:
    names = ("train", "validation", "test")
    raw = store.manifest.get("splits")
    if not isinstance(raw, Mapping) or set(raw) != set(names):
        raise ValueError(
            "manifest must declare exact train, validation, and test splits"
        )
    splits: dict[str, list[str]] = {}
    for name in names:
        values = raw[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(f"manifest split {name!r} is invalid")
        if len(values) != len(set(values)):
            raise ValueError(f"manifest split {name!r} contains duplicate keys")
        splits[name] = list(values)
    partition = [key for name in names for key in splits[name]]
    if len(partition) != len(set(partition)) or set(partition) != set(store.keys):
        raise ValueError("manifest splits do not exactly partition the shard store")
    counts = {name: len(splits[name]) for name in names}
    for field in ("declared_split_counts", "prepared_split_counts"):
        recorded = store.manifest.get(field)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"manifest is missing {field}")
        if {name: int(recorded.get(name, -1)) for name in names} != counts:
            raise ValueError(f"{field} disagrees with the manifest partition")
    for name in names:
        for key in splits[name]:
            if store.entry(key).get("split") != name:
                raise ValueError(f"trajectory {key!r} disagrees with split {name!r}")
    return splits["train"], splits["validation"], splits["test"]


def _load_normalization(path: Path) -> tuple[Euler2DNormalization, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("normalization artifact must contain a mapping")
    normalization = Euler2DNormalization.from_mapping(payload)
    canonical = normalization.to_dict()
    return normalization, canonical


def audit_inputs(
    store: PCNOEuler2DShardStore,
    *,
    handoff_path: Path,
    normalization_path: Path,
) -> tuple[dict[str, Any], Euler2DNormalization, list[str], list[str]]:
    """Bind the smoke test to the exact frozen handoff and keep test sealed."""

    handoff = validate_line3_handoff(
        handoff_path,
        expected_data_manifest_digest=store.manifest_digest,
    )
    normalization, normalization_payload = _load_normalization(normalization_path)
    normalization_digest = digest_mapping(normalization_payload)
    if normalization_digest != handoff["normalization_digest"]:
        raise ValueError("normalization and frozen handoff digests differ")
    train_keys, validation_keys, test_keys = _manifest_splits(store)
    split_contract = {
        "train": train_keys,
        "validation": validation_keys,
        "test": test_keys,
        "split_groups": {
            key: store.entry(key).get("split_group_id") for key in store.keys
        },
    }
    if digest_mapping(split_contract) != handoff["grouped_split_digest"]:
        raise ValueError("grouped split and frozen handoff digests differ")
    if len(train_keys) != 84 or len(validation_keys) != 24:
        raise ValueError(
            "frozen Line-4A split must contain 84 train and 24 validation cases"
        )
    allowed_keys = train_keys + validation_keys
    geometry_digests = {store.entry(key).get("geometry_digest") for key in allowed_keys}
    node_counts = {int(store.entry(key)["num_nodes"]) for key in allowed_keys}
    if len(geometry_digests) != 1 or node_counts != {25000}:
        raise ValueError("Line-4A smoke requires one shared 25000-cell geometry")
    if normalization.weight_provenance != "validated_physical_cell_volume_normalized":
        raise ValueError("normalization does not carry the physical-volume contract")
    return (
        {
            "schema": SCHEMA,
            "representation_schema": LINE4_REPRESENTATION_SCHEMA,
            "handoff": handoff,
            "data_manifest_digest": store.manifest_digest,
            "normalization_digest": normalization_digest,
            "grouped_split_digest": handoff["grouped_split_digest"],
            "train_count": len(train_keys),
            "validation_count": len(validation_keys),
            "sealed_test_count": len(test_keys),
            "test_arrays_read": False,
            "shared_geometry_digest": next(iter(geometry_digests)),
            "node_count": 25000,
            "authorized_work": "representation_reconstruction_and_closure_only",
            "transition_present": False,
            "rollout_present": False,
            "assimilation_present": False,
        },
        normalization,
        train_keys,
        validation_keys,
    )


def _select_spread(keys: Sequence[str], count: int) -> list[str]:
    if count < 1 or count > len(keys):
        raise ValueError("spread selection count is invalid")
    indices = np.linspace(0, len(keys) - 1, count, dtype=np.int64)
    selected = [str(keys[int(index)]) for index in indices]
    if len(set(selected)) != count:
        raise ValueError("spread selection produced duplicate keys")
    return selected


def _select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _smoke_data_preflight(
    store: PCNOEuler2DShardStore,
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
    *,
    device: torch.device,
) -> dict[str, Any]:
    selected_train = _select_spread(train_keys, SMOKE_TRAIN_TRAJECTORIES)
    selected_validation = _select_spread(
        validation_keys,
        SMOKE_VALIDATION_TRAJECTORIES,
    )
    selected = selected_train + selected_validation
    checked_files = 0
    for key in selected:
        entry = store.entry(key)
        if entry.get("split") == "test":
            raise ValueError("test trajectory entered the staged smoke cohort")
        folder = store.root / str(entry["folder"])
        array_digests = entry.get("array_sha256")
        if not isinstance(array_digests, Mapping) or not array_digests:
            raise ValueError(f"trajectory {key!r} lacks array digests")
        for name, expected_digest in array_digests.items():
            path = folder / f"{name}.npy"
            if sha256_file(path) != str(expected_digest):
                raise ValueError(f"staged array digest mismatch for {key}:{name}")
            checked_files += 1
        if not (folder / "metadata.json").is_file():
            raise FileNotFoundError(f"staged trajectory {key!r} lacks metadata")
        states = store.states(key)
        if states.shape != (61, 25000, 4):
            raise ValueError(f"trajectory {key!r} has unexpected state shape")
        _load_state_batch(store, [(key, 0), (key, 60)], device)
    geometry = _load_geometry(store, selected_train[0], device)
    if geometry.nodes.shape != (1, 25000, 2) or geometry.num_tokens != 250:
        raise ValueError("staged smoke geometry does not satisfy the token contract")
    return {
        "train_trajectories": selected_train,
        "validation_trajectories": selected_validation,
        "test_trajectories": [],
        "array_files_hashed": checked_files,
        "loaded_endpoint_frames_per_trajectory": [0, 60],
        "token_count": geometry.num_tokens,
        "node_count": geometry.nodes.shape[1],
        "test_arrays_read": False,
    }


def _load_geometry(
    store: PCNOEuler2DShardStore,
    key: str,
    device: torch.device,
):
    nodes = torch.as_tensor(
        np.array(store.array(key, "nodes"), copy=True),
        dtype=torch.float32,
        device=device,
    )
    node_measures = torch.as_tensor(
        np.array(store.array(key, "node_measures"), copy=True),
        dtype=torch.float32,
        device=device,
    )
    geometry = build_token_geometry(nodes, node_measures)
    return geometry


def _load_state_batch(
    store: PCNOEuler2DShardStore,
    items: Sequence[tuple[str, int]],
    device: torch.device,
) -> torch.Tensor:
    if not items:
        raise ValueError("state batch must not be empty")
    arrays: list[np.ndarray] = []
    for key, frame in items:
        if store.entry(key).get("split") == "test":
            raise ValueError("test arrays are sealed during the representation gate")
        states = store.states(key)
        if frame < 0 or frame >= states.shape[0]:
            raise IndexError(f"invalid frame {frame} for trajectory {key}")
        arrays.append(np.array(states[frame], dtype=np.float32, copy=True))
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)


def _build_models(
    normalization: Euler2DNormalization,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, SpatialTokenAutoencoder]:
    torch.manual_seed(seed)
    generic = SpatialTokenAutoencoder(normalization, variant="generic").to(device)
    hybrid = SpatialTokenAutoencoder(
        normalization,
        variant="conservative_moment",
    ).to(device)
    hybrid.load_state_dict(generic.state_dict(), strict=True)
    if generic.parameter_count != hybrid.parameter_count:
        raise AssertionError("matched representation parameter counts differ")
    return {"generic": generic, "conservative_moment": hybrid}


def _primitive_overshoot(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float,
) -> float:
    prediction_primitive = conservative_to_primitive_torch(prediction, gamma=gamma)
    target_primitive = conservative_to_primitive_torch(target, gamma=gamma)
    overshoots: list[torch.Tensor] = []
    for component in (0, 3):
        target_values = target_primitive[..., component]
        prediction_values = prediction_primitive[..., component]
        lower = target_values.amin()
        upper = target_values.amax()
        scale = (upper - lower).clamp_min(torch.finfo(target.dtype).eps)
        overshoots.append(
            torch.maximum(
                (prediction_values.amax() - upper).clamp_min(0.0),
                (lower - prediction_values.amin()).clamp_min(0.0),
            )
            / scale
        )
    return float(torch.stack(overshoots).amax().detach().cpu())


@torch.no_grad()
def _evaluate_models(
    models: Mapping[str, SpatialTokenAutoencoder],
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    frames: Sequence[int],
    geometry,
    normalization: Euler2DNormalization,
    *,
    device: torch.device,
    capture: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, np.ndarray]]]]:
    positions = geometry.nodes[0].detach().cpu().numpy()
    volumes = geometry.volumes[0].detach().cpu().numpy()
    edges = np.asarray(store.array(keys[0], "edges"), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    captured: dict[str, dict[str, dict[str, np.ndarray]]] = {
        name: {} for name in models
    }
    for key in keys:
        items = [(key, int(frame)) for frame in frames]
        target = _load_state_batch(store, items, device)
        batched_geometry = repeat_token_geometry(geometry, len(items))
        target_numpy = target.detach().cpu().numpy()
        for name, model in models.items():
            model.eval()
            prediction, code = model(target, batched_geometry)
            prediction_numpy = prediction.detach().cpu().numpy()
            code_numpy = code.detach().cpu().numpy()
            if capture:
                captured[name][key] = {
                    "frames": np.asarray(frames, dtype=np.int64),
                    "target": target_numpy,
                    "decoded": prediction_numpy,
                    "code": code_numpy,
                }
            for offset, frame in enumerate(frames):
                sample_geometry = geometry
                target_sample = target[offset : offset + 1]
                prediction_sample = prediction[offset : offset + 1]
                row = {
                    "variant": name,
                    "trajectory": key,
                    "frame": int(frame),
                    **reconstruction_metrics(
                        prediction_sample,
                        target_sample,
                        sample_geometry,
                        state_scale=model.state_scale,
                        gamma=model.gamma,
                    ),
                    **endpoint_metrics(
                        prediction_numpy[offset],
                        target_numpy[offset],
                        positions=positions,
                        edges=edges,
                        volumes=volumes,
                        component_scale=normalization.state_scale,
                        gamma=model.gamma,
                        shock_quantile=0.9,
                    ),
                    "density_pressure_overshoot": _primitive_overshoot(
                        prediction_sample,
                        target_sample,
                        gamma=model.gamma,
                    ),
                    "contact_behavior": "not_present_in_frozen_shock_vortex_family",
                }
                rows.append(row)
    return rows, captured


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    variants = sorted({str(row["variant"]) for row in rows})
    output: dict[str, dict[str, Any]] = {}
    ignored = {
        "variant",
        "trajectory",
        "frame",
        "intervention_applied",
        "smooth_region_contract",
        "contact_behavior",
    }
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        names = sorted(set().union(*(row.keys() for row in selected)) - ignored)
        aggregate: dict[str, Any] = {"rows": len(selected)}
        for name in names:
            values = [
                float(row[name])
                for row in selected
                if row.get(name) is not None
                and isinstance(row.get(name), (int, float, np.integer, np.floating))
                and math.isfinite(float(row[name]))
            ]
            if values:
                aggregate[f"mean_{name}"] = float(np.mean(values))
                aggregate[f"maximum_{name}"] = float(np.max(values))
        aggregate["completion_rate"] = 1.0
        aggregate["intervention_applied"] = False
        output[variant] = aggregate
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


@torch.no_grad()
def _closure_diagnostics(
    models: Mapping[str, SpatialTokenAutoencoder],
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    geometry,
    *,
    device: torch.device,
) -> dict[str, Any]:
    encoded: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    groups: list[str] = []
    for key in keys:
        frame_count = int(store.states(key).shape[0])
        if frame_count != 61:
            raise ValueError("smoke closure contract expects 61 saved states")
        per_model: dict[str, list[torch.Tensor]] = {name: [] for name in models}
        for start in range(0, frame_count, 8):
            frames = list(range(start, min(start + 8, frame_count)))
            state = _load_state_batch(store, [(key, frame) for frame in frames], device)
            batch_geometry = repeat_token_geometry(geometry, len(frames))
            for name, model in models.items():
                model.eval()
                per_model[name].append(model.encode(state, batch_geometry).cpu())
        for name in models:
            sequence = torch.cat(per_model[name], dim=0)
            encoded[name].append(
                torch.stack((sequence[:-2], sequence[1:-1], sequence[2:]), dim=1)
            )
        groups.extend([key] * (frame_count - 2))

    output: dict[str, Any] = {}
    for name in models:
        triples = torch.cat(encoded[name], dim=0)
        previous = triples[:, 0]
        current = triples[:, 1]
        future = triples[:, 2]
        whitener = fit_channel_whitener(current)
        output[name] = conditional_future_diagnostics(
            current,
            future,
            whitener=whitener,
            exclusion_groups=groups,
            previous_codes=previous,
            neighbors=min(4, len(keys) - 1),
            chunk_size=32,
        )
        output[name]["cohort"] = {
            "split": "validation",
            "trajectories": list(keys),
            "frames": "1_through_59_with_previous_and_next",
            "same_trajectory_neighbors_excluded": True,
        }
    output["pod_comparison"] = {
        "status": "not_run_in_engineering_smoke",
        "reason": "rank-5000 POD requires the preregistered full training snapshot set",
    }
    return output


def _fine_query_geometry(device: torch.device):
    nx, ny = 500, 200
    x = (torch.arange(nx, dtype=torch.float32, device=device) + 0.5) * (2.0 / nx)
    y = (torch.arange(ny, dtype=torch.float32, device=device) + 0.5) * (1.0 / ny)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    volumes = torch.full(
        (nx * ny,),
        2.0 / (nx * ny),
        dtype=torch.float32,
        device=device,
    )
    return build_token_geometry(nodes, volumes)


@torch.no_grad()
def _conditioning_diagnostics(
    models: Mapping[str, SpatialTokenAutoencoder],
    store: PCNOEuler2DShardStore,
    key: str,
    geometry,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    state = _load_state_batch(store, [(key, 30)], device)
    fine_geometry = _fine_query_geometry(device)
    output: dict[str, Any] = {}
    for offset, (name, model) in enumerate(models.items()):
        model.eval()
        code = model.encode(state, geometry)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 101 + offset)
        directions = torch.randn(
            (4, *code.shape),
            dtype=code.dtype,
            device=device,
            generator=generator,
        )
        coarse = empirical_decoder_gains(
            model,
            code,
            geometry,
            directions,
            epsilon=1.0e-3,
        )
        fine = empirical_decoder_gains(
            model,
            code,
            fine_geometry,
            directions,
            epsilon=1.0e-3,
        )
        coarse_prediction = model.decode(code, geometry)
        fine_prediction = model.decode(code, fine_geometry)
        downsampled = fine_prediction.view(1, 100, 2, 250, 2, 4).mean(dim=(2, 4))
        downsampled = downsampled.reshape(1, 25000, 4)
        denominator = coarse["mean_gain"]
        output[name] = {
            "stored_grid": [250, 100],
            "query_grid": [500, 200],
            "coarse": coarse,
            "fine": fine,
            "mean_gain_resolution_ratio": (
                None if denominator == 0.0 else fine["mean_gain"] / denominator
            ),
            "query_consistency_relative_l2": float(
                volume_relative_l2(downsampled, coarse_prediction, geometry).cpu()
            ),
            "raw_decode": True,
            "intervention_applied": False,
        }
    return output


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _cost_diagnostics(
    model: SpatialTokenAutoencoder,
    state: torch.Tensor,
    geometry,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    for _ in range(3):
        code = model.encode(state, geometry)
        model.decode(code, geometry)
    _synchronize(device)

    def measured(operation, repeats: int = 20) -> list[float]:
        values: list[float] = []
        for _ in range(repeats):
            _synchronize(device)
            start = perf_counter()
            operation()
            _synchronize(device)
            values.append(perf_counter() - start)
        return values

    code = model.encode(state, geometry)
    encode_seconds = measured(lambda: model.encode(state, geometry))
    decode_seconds = measured(lambda: model.decode(code, geometry))
    total_seconds = measured(lambda: model(state, geometry))

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, 0.95)),
        }

    return {
        "batch_size": 1,
        "state_size": int(state.numel()),
        "latent_size": model.latent_size,
        "encode": summary(encode_seconds),
        "decode": summary(decode_seconds),
        "encode_decode": summary(total_seconds),
        "decodes_per_reconstruction": 1,
        "device": str(device),
    }


def _write_captured_artifacts(
    output_dir: Path,
    models: Mapping[str, SpatialTokenAutoencoder],
    captured: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    ledgers: list[dict[str, Any]] = []
    for variant, trajectories in captured.items():
        for key, arrays in trajectories.items():
            safe_key = "".join(
                character if character.isalnum() else "_" for character in key
            )
            physical_name = f"physical_{variant}_{safe_key}.npz"
            latent_name = f"latent_{variant}_{safe_key}.npz"
            np.savez_compressed(
                artifact_dir / physical_name,
                schema=np.asarray(LINE4_REPRESENTATION_SCHEMA),
                trajectory=np.asarray(key),
                split=np.asarray("validation"),
                frames=arrays["frames"],
                target=arrays["target"],
                decoded=arrays["decoded"],
            )
            np.savez_compressed(
                artifact_dir / latent_name,
                schema=np.asarray(LINE4_REPRESENTATION_SCHEMA),
                trajectory=np.asarray(key),
                split=np.asarray("validation"),
                frames=arrays["frames"],
                code=arrays["code"],
            )
            ledgers.append(
                representation_artifact_ledger(
                    models[variant],
                    split="validation",
                    trajectory=key,
                    physical_states_file=f"artifacts/{physical_name}",
                    latent_states_file=f"artifacts/{latent_name}",
                    valid_length=int(len(arrays["frames"])),
                    failure_cause="completed",
                    seed=seed,
                )
            )
    return ledgers


def _run_smoke(
    args: argparse.Namespace,
    store: PCNOEuler2DShardStore,
    audit: Mapping[str, Any],
    normalization: Euler2DNormalization,
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
) -> None:
    device = _select_device(args.device)
    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    selected_train = _select_spread(train_keys, SMOKE_TRAIN_TRAJECTORIES)
    selected_validation = _select_spread(
        validation_keys,
        SMOKE_VALIDATION_TRAJECTORIES,
    )
    geometry = _load_geometry(store, selected_train[0], device)
    models = _build_models(normalization, device=device, seed=args.seed)
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for name, model in models.items()
    }
    training_bank = [(key, frame) for key in selected_train for frame in SMOKE_FRAMES]
    rng = np.random.default_rng(args.seed + 17)
    metrics_path = args.output_dir / "metrics.jsonl"
    start_time = perf_counter()
    final_losses: dict[str, float] = {}
    for step in range(1, SMOKE_STEPS + 1):
        indices = rng.integers(0, len(training_bank), size=SMOKE_BATCH_SIZE)
        items = [training_bank[int(index)] for index in indices]
        state = _load_state_batch(store, items, device)
        batch_geometry = repeat_token_geometry(geometry, SMOKE_BATCH_SIZE)
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(state, batch_geometry)
            loss = scaled_volume_mse(
                prediction,
                state,
                batch_geometry,
                model.state_scale,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {name} reconstruction loss")
            loss.backward()
            optimizer.step()
            final_losses[name] = float(loss.detach().cpu())
        if step == 1 or step % SMOKE_EVALUATION_INTERVAL == 0:
            evaluation_rows, _ = _evaluate_models(
                models,
                store,
                selected_validation,
                SMOKE_FRAMES,
                geometry,
                normalization,
                device=device,
                capture=False,
            )
            _append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "training_loss": dict(final_losses),
                    "validation": _aggregate_rows(evaluation_rows),
                    "test_arrays_read": False,
                    "transition_present": False,
                    "analysis_applied": False,
                },
            )
    _synchronize(device)
    training_seconds = perf_counter() - start_time

    final_rows, captured = _evaluate_models(
        models,
        store,
        selected_validation,
        SMOKE_FRAMES,
        geometry,
        normalization,
        device=device,
        capture=True,
    )
    aggregates = _aggregate_rows(final_rows)
    _write_rows(args.output_dir / "reconstruction_rows.csv", final_rows)
    _write_rows(
        args.output_dir / "summary.csv",
        [{"variant": variant, **values} for variant, values in aggregates.items()],
    )
    closure = _closure_diagnostics(
        models,
        store,
        selected_validation,
        geometry,
        device=device,
    )
    conditioning = _conditioning_diagnostics(
        models,
        store,
        selected_validation[0],
        geometry,
        device=device,
        seed=args.seed,
    )
    state_for_cost = _load_state_batch(
        store,
        [(selected_validation[0], 30)],
        device,
    )
    cost = {
        name: _cost_diagnostics(
            model,
            state_for_cost,
            geometry,
            device=device,
        )
        for name, model in models.items()
    }
    artifact_ledgers = _write_captured_artifacts(
        args.output_dir,
        models,
        captured,
        seed=args.seed,
    )
    _write_json(args.output_dir / "closure.json", closure)
    _write_json(args.output_dir / "conditioning.json", conditioning)
    _write_json(args.output_dir / "artifact_ledger.json", {"items": artifact_ledgers})

    checkpoint = {
        "schema": SCHEMA,
        "stage": "smoke",
        "seed": args.seed,
        "data_manifest_digest": audit["data_manifest_digest"],
        "normalization": normalization.to_dict(),
        "train_keys": selected_train,
        "validation_keys": selected_validation,
        "test_keys": [],
        "test_arrays_read": False,
        "model_contracts": {name: model.contract() for name, model in models.items()},
        "model_states": {name: model.state_dict() for name, model in models.items()},
        "optimizer_steps_per_model": SMOKE_STEPS,
        "transition_present": False,
        "analysis_applied": False,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    generic_error = aggregates["generic"]["mean_relative_l2"]
    hybrid_error = aggregates["conservative_moment"]["mean_relative_l2"]
    summary = {
        "schema": SCHEMA,
        "status": "completed_engineering_smoke",
        "hypothesis": (
            "exact conservative token moments improve closure or decoder conditioning "
            "without broadening fronts or degrading matched reconstruction"
        ),
        "smallest_matched_control": (
            "generic and conservative-moment 25x10x20 spatial-token autoencoders "
            "with identical initialization, parameter count, batches, and optimizer"
        ),
        "audit": dict(audit),
        "stage_contract": {
            "representation_pretraining": "state_reconstruction_only",
            "recurrent_training": False,
            "joint_fine_tuning": False,
            "front_loss": False,
            "noise_augmentation": False,
            "transition_present": False,
            "rollout_present": False,
            "test_arrays_read": False,
            "analysis_applied": False,
        },
        "configuration": {
            "seed": args.seed,
            "steps_per_model": SMOKE_STEPS,
            "batch_size": SMOKE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_trajectories": selected_train,
            "validation_trajectories": selected_validation,
            "frames": list(SMOKE_FRAMES),
            "device": str(device),
        },
        "representation_ladder": {
            "identity": {
                "status": "analytic_control",
                "reconstruction_relative_l2": 0.0,
                "latent_size": 100000,
            },
            "pod_rank_5000": {
                "status": "deferred_to_full_matched_run",
                "reason": "smoke bank has fewer than 5001 independent snapshots",
            },
            "oracle_front": {
                "status": "consumed_prior_bounded_diagnostic_no_2d_candidate",
            },
            "generic": models["generic"].contract(),
            "conservative_moment": models["conservative_moment"].contract(),
        },
        "reconstruction": aggregates,
        "closure": closure,
        "decoder_conditioning": conditioning,
        "cost": {
            "training_wall_seconds": training_seconds,
            "models": cost,
        },
        "matched_relative_l2_ratio_hybrid_to_generic": hybrid_error / generic_error,
        "promotion_thresholds_not_evaluated_by_smoke": {
            "mean_reconstruction_relative_l2": 0.0021,
            "admissible_fraction": 1.0,
            "front_strength_and_thickness_relative_tolerance": 0.05,
            "closure_ratio_to_rank_5000_pod": 0.8,
            "doubled_resolution_gain_ratio": 1.25,
        },
        "failure_interpretation": (
            "a smoke failure is an implementation or fixed-candidate failure; it "
            "does not authorize a dimension, loss, or architecture sweep"
        ),
        "estimated_and_actual_cost": {
            "preregistered_max_total_gpu_hours": 24.0,
            "actual_smoke_wall_seconds": training_seconds,
            "smoke_runs_consumed": 1,
            "maximum_smoke_runs": 2,
            "maximum_serious_runs": 2,
        },
        "next_decision": (
            "review smoke gates; only then authorize one full rank-5000-POD-matched "
            "representation run, never a latent transition or filter from this artifact"
        ),
        "promotion_eligible": False,
        "artifact_count": len(artifact_ledgers),
    }
    _write_json(args.output_dir / "summary.json", summary)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _prepare_output(args.output_dir)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=4)
    try:
        audit, normalization, train_keys, validation_keys = audit_inputs(
            store,
            handoff_path=args.handoff,
            normalization_path=args.normalization,
        )
        audit = {
            **audit,
            "smoke_data_readiness": _smoke_data_preflight(
                store,
                train_keys,
                validation_keys,
                device=_select_device(args.device),
            ),
        }
        _write_json(args.output_dir / "preflight.json", audit)
        if args.stage == "preflight":
            return
        _run_smoke(
            args,
            store,
            audit,
            normalization,
            train_keys,
            validation_keys,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()

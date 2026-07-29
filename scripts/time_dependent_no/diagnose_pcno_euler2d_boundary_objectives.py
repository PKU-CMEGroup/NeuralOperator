#!/usr/bin/env python3
"""Audit whether causal boundary auxiliaries supply useful interior gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    build_model,
    load_checkpoint,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (  # noqa: E402
    NEAR_BOUNDARY_AUXILIARY,
    PROJECTED_BOUNDARY_AUXILIARY,
    autocast_context,
    boundary_auxiliary_loss,
    close_boundary,
    contract_forward_sample,
    select_device,
)
from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    INFLOW_NODE,
    OUTFLOW_NODE,
    WALL_NODE,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    boundary_band_normal_node_mask,
    build_graph_causal_boundary_policy,
    conservative_to_primitive_torch,
    normal_node_mask,
    proxy_mass_weights,
    weighted_scaled_mse,
)

OBJECTIVES = (
    "interior",
    "near_1",
    "near_2",
    "near_3",
    "projected_target",
    "raw_boundary",
    "projected_inflow",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-stream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--pairs-per-time-stratum", type=int, default=4)
    parser.add_argument("--gradient-contribution-fraction", type=float, default=0.1)
    parser.add_argument("--max-auxiliary-weight", type=float, default=10.0)
    parser.add_argument("--boundary-max-source-hops", type=int, default=3)
    parser.add_argument("--boundary-rho-inf", type=float, default=1.4)
    parser.add_argument("--boundary-p-inf", type=float, default=1.0)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_stratified_pairs(
    path: Path,
    *,
    epoch_index: int,
    pairs_per_stratum: int,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, Any]]:
    if epoch_index < 0 or pairs_per_stratum < 1:
        raise ValueError("epoch index must be nonnegative and stratum size positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or epoch_index >= len(epochs):
        raise ValueError("sample stream does not contain the requested epoch")
    rows = epochs[epoch_index].get("pairs")
    if not isinstance(rows, list):
        raise ValueError("sample stream epoch lacks its literal pair list")
    strata: dict[str, list[tuple[str, int]]] = {
        "early": [],
        "middle": [],
        "late": [],
    }
    for row in rows:
        pair = (str(row["trajectory"]), int(row["time_index"]))
        time_index = pair[1]
        name = "early" if time_index <= 25 else "middle" if time_index <= 52 else "late"
        if len(strata[name]) < pairs_per_stratum:
            strata[name].append(pair)
        if all(len(values) == pairs_per_stratum for values in strata.values()):
            break
    if any(len(values) != pairs_per_stratum for values in strata.values()):
        raise ValueError("sample stream cannot fill all three time strata")
    strata["all"] = [
        pair for name in ("early", "middle", "late") for pair in strata[name]
    ]
    return strata, payload


def node_type_mask(sample: Mapping[str, torch.Tensor], node_kind: int) -> torch.Tensor:
    node_type = sample["node_type"]
    if node_type.ndim == 3:
        node_type = node_type[..., 0]
    return sample["node_mask"] * (node_type == node_kind).to(
        dtype=sample["node_mask"].dtype
    ).unsqueeze(-1)


def objective_loss(
    name: str,
    *,
    prediction: torch.Tensor,
    raw_prediction: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    model: PCNOEuler2DResidual,
    policy: Mapping[str, Any],
) -> torch.Tensor:
    if name == "interior":
        mask = normal_node_mask(sample["node_type"], sample["node_mask"])
        return weighted_scaled_mse(
            prediction,
            sample["target"],
            sample["node_weights"],
            mask,
            model.state_scale,
        )
    if name.startswith("near_"):
        hops = int(name.rsplit("_", 1)[1])
        return boundary_auxiliary_loss(
            prediction,
            sample["target"],
            sample,
            model,
            boundary_policy=policy,
            kind=NEAR_BOUNDARY_AUXILIARY,
            near_boundary_hops=hops,
        )
    if name == "projected_target":
        return boundary_auxiliary_loss(
            prediction,
            sample["target"],
            sample,
            model,
            boundary_policy=policy,
            kind=PROJECTED_BOUNDARY_AUXILIARY,
            near_boundary_hops=3,
        )
    if name == "raw_boundary":
        mask = sample["node_mask"] - normal_node_mask(
            sample["node_type"], sample["node_mask"]
        )
        return weighted_scaled_mse(
            raw_prediction,
            sample["target"],
            sample["node_weights"],
            mask,
            model.state_scale,
        )
    if name == "projected_inflow":
        projected_target = close_boundary(sample["target"], policy, gamma=model.gamma)
        return weighted_scaled_mse(
            prediction,
            projected_target,
            sample["node_weights"],
            node_type_mask(sample, INFLOW_NODE),
            model.state_scale,
        )
    raise ValueError(f"unsupported objective: {name}")


def cohort_gradient(
    objective: str,
    pairs: Sequence[tuple[str, int]],
    *,
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    policies: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    amp: str,
) -> tuple[list[torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for key, time_index in pairs:
        sample = store.tensor_batch(
            key,
            [time_index],
            step_stride=1,
            device=device,
        )
        with autocast_context(device, amp):
            prediction, raw_prediction, _ = contract_forward_sample(
                model,
                sample,
                sample["current"],
                boundary_policy=policies[key],
            )
            loss = objective_loss(
                objective,
                prediction=prediction,
                raw_prediction=raw_prediction,
                sample=sample,
                model=model,
                policy=policies[key],
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"nonfinite {objective} loss on {key}:{time_index}"
            )
        (loss / len(pairs)).backward()
        loss_sum += float(loss.detach().cpu())
    gradients = [
        (
            torch.zeros_like(parameter, device="cpu")
            if parameter.grad is None
            else parameter.grad.detach().float().cpu().clone()
        )
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    model.zero_grad(set_to_none=True)
    return gradients, loss_sum / len(pairs)


def gradient_comparison(
    reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]
) -> dict[str, float]:
    reference_square = 0.0
    candidate_square = 0.0
    dot = 0.0
    for reference_part, candidate_part in zip(reference, candidate, strict=True):
        reference_flat = reference_part.reshape(-1)
        candidate_flat = candidate_part.reshape(-1)
        reference_square += float(torch.dot(reference_flat, reference_flat))
        candidate_square += float(torch.dot(candidate_flat, candidate_flat))
        dot += float(torch.dot(reference_flat, candidate_flat))
    reference_norm = math.sqrt(reference_square)
    candidate_norm = math.sqrt(candidate_square)
    denominator = reference_norm * candidate_norm
    return {
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "norm_ratio": candidate_norm / reference_norm if reference_norm > 0.0 else 0.0,
        "cosine": dot / denominator if denominator > 0.0 else 0.0,
    }


@torch.no_grad()
def closure_and_support_audit(
    pairs: Sequence[tuple[str, int]],
    *,
    store: PCNOEuler2DShardStore,
    policies: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    gamma: float,
) -> dict[str, Any]:
    type_names = {
        "inflow": INFLOW_NODE,
        "wall": WALL_NODE,
        "outflow": OUTFLOW_NODE,
    }
    squared = {name: np.zeros(4, dtype=np.float64) for name in type_names}
    maximum = {name: np.zeros(4, dtype=np.float64) for name in type_names}
    counts = {name: 0 for name in type_names}
    band_mass = {str(hops): [] for hops in (1, 2, 3)}
    for key, time_index in pairs:
        sample = store.tensor_batch(key, [time_index], step_stride=1, device=device)
        target = sample["target"]
        closed = close_boundary(target, policies[key], gamma=gamma)
        target_primitive = conservative_to_primitive_torch(target, gamma=gamma)
        closed_primitive = conservative_to_primitive_torch(closed, gamma=gamma)
        difference = closed_primitive - target_primitive
        for name, node_kind in type_names.items():
            selected = node_type_mask(sample, node_kind)[..., 0].to(dtype=torch.bool)
            values = difference[selected].double()
            if values.numel() == 0:
                continue
            squared[name] += values.square().sum(dim=0).cpu().numpy()
            maximum[name] = np.maximum(
                maximum[name], values.abs().amax(dim=0).cpu().numpy()
            )
            counts[name] += int(values.shape[0])
        normal = normal_node_mask(sample["node_type"], sample["node_mask"])
        normal_weights = proxy_mass_weights(sample["node_weights"], normal).sum()
        for hops in (1, 2, 3):
            band = boundary_band_normal_node_mask(
                sample["node_type"],
                sample["node_mask"],
                sample["directed_edges"],
                max_hops=hops,
            )
            band_mass[str(hops)].append(
                float(
                    (
                        proxy_mass_weights(sample["node_weights"], band).sum()
                        / normal_weights
                    ).cpu()
                )
            )
    return {
        "reference_closure_primitive_rms": {
            name: (
                np.sqrt(squared[name] / counts[name]).tolist() if counts[name] else None
            )
            for name in type_names
        },
        "reference_closure_primitive_max_abs": {
            name: (maximum[name].tolist() if counts[name] else None)
            for name in type_names
        },
        "node_counts": counts,
        "near_boundary_proxy_mass_fraction_mean": {
            hops: float(np.mean(values)) for hops, values in band_mass.items()
        },
    }


def candidate_decision(
    objective: str,
    comparisons: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    contribution_fraction: float,
    max_weight: float,
) -> dict[str, Any]:
    rows = {name: comparisons[name][objective] for name in comparisons}
    full = rows["all"]
    stratum_cosines = [rows[name]["cosine"] for name in ("early", "middle", "late")]
    ratio = float(full["norm_ratio"])
    weight = contribution_fraction / ratio if ratio > 0.0 else math.inf
    passed = (
        all(
            math.isfinite(value)
            for value in [ratio, weight, full["cosine"], *stratum_cosines]
        )
        and ratio > 0.0
        and float(full["cosine"]) >= 0.0
        and min(stratum_cosines) >= -0.1
        and 0.0 < weight <= max_weight
    )
    return {
        "objective": objective,
        "passed_gradient_gate": passed,
        "full_cosine": float(full["cosine"]),
        "minimum_time_stratum_cosine": float(min(stratum_cosines)),
        "full_norm_ratio": ratio,
        "target_auxiliary_gradient_fraction": contribution_fraction,
        "recommended_weight": float(f"{weight:.8g}") if passed else None,
        "weight_cap": max_weight,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")
    if not 0.0 < args.gradient_contribution_fraction <= 1.0:
        raise ValueError("gradient contribution fraction must lie in (0,1]")
    if args.max_auxiliary_weight <= 0.0:
        raise ValueError("maximum auxiliary weight must be positive")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["boundary_mode"] != "model_all_nodes":
        raise ValueError("this audit requires the exact raw D041 parent")
    store = PCNOEuler2DShardStore(args.data_dir)
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and data manifest differ")
    if (
        args.expected_manifest_sha256
        and store.manifest_digest != args.expected_manifest_sha256
    ):
        raise ValueError("data manifest SHA-256 mismatch")

    strata, stream = load_stratified_pairs(
        args.sample_stream,
        epoch_index=args.epoch_index,
        pairs_per_stratum=args.pairs_per_time_stratum,
    )
    train_keys = {str(key) for key in checkpoint["train_keys"]}
    if any(key not in train_keys for key, _ in strata["all"]):
        raise ValueError("gradient audit stream contains a non-training trajectory")

    torch.set_float32_matmul_precision("high")
    model = build_model(checkpoint, device)
    unique_keys = sorted({key for key, _ in strata["all"]})
    policies = {}
    policy_digests = {}
    for key in unique_keys:
        policy, metadata = build_graph_causal_boundary_policy(
            store,
            key,
            device=device,
            max_source_hops=args.boundary_max_source_hops,
            rho_inf=args.boundary_rho_inf,
            p_inf=args.boundary_p_inf,
        )
        if int(metadata["fallback_target_count"]) != 0:
            raise ValueError(f"trajectory {key} uses a fallback boundary target")
        policies[key] = policy
        policy_digests[key] = metadata["policy_digest"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    losses: dict[str, dict[str, float]] = {}
    for cohort_name in ("early", "middle", "late", "all"):
        pairs = strata[cohort_name]
        reference_gradient, reference_loss = cohort_gradient(
            "interior",
            pairs,
            model=model,
            store=store,
            policies=policies,
            device=device,
            amp=args.amp,
        )
        comparisons[cohort_name] = {}
        losses[cohort_name] = {"interior": reference_loss}
        for objective in OBJECTIVES[1:]:
            candidate_gradient, candidate_loss = cohort_gradient(
                objective,
                pairs,
                model=model,
                store=store,
                policies=policies,
                device=device,
                amp=args.amp,
            )
            comparisons[cohort_name][objective] = gradient_comparison(
                reference_gradient, candidate_gradient
            )
            losses[cohort_name][objective] = candidate_loss
            del candidate_gradient
        del reference_gradient

    closure = closure_and_support_audit(
        strata["all"],
        store=store,
        policies=policies,
        device=device,
        gamma=float(checkpoint["normalization"].get("gamma", 1.4)),
    )
    decisions = {
        "projected_target": candidate_decision(
            "projected_target",
            comparisons,
            contribution_fraction=args.gradient_contribution_fraction,
            max_weight=args.max_auxiliary_weight,
        ),
        "near_boundary_band_3": candidate_decision(
            "near_3",
            comparisons,
            contribution_fraction=args.gradient_contribution_fraction,
            max_weight=args.max_auxiliary_weight,
        ),
        "raw_boundary_reference": candidate_decision(
            "raw_boundary",
            comparisons,
            contribution_fraction=args.gradient_contribution_fraction,
            max_weight=args.max_auxiliary_weight,
        ),
    }
    summary = {
        "schema": "pcno_euler2d_boundary_objective_gradient_audit_v1",
        "status": "complete",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "boundary_mode": checkpoint["boundary_mode"],
        },
        "data_manifest_digest": store.manifest_digest,
        "sample_stream": {
            "path": str(args.sample_stream),
            "sha256": sha256_file(args.sample_stream),
            "declared_digest": stream.get("sample_stream_digest"),
            "epoch_index": args.epoch_index,
            "pairs": {
                name: [
                    {"trajectory": key, "time_index": time_index}
                    for key, time_index in pairs
                ]
                for name, pairs in strata.items()
            },
        },
        "policy_digests": policy_digests,
        "gradient_losses": losses,
        "gradient_comparisons_to_interior": comparisons,
        "closure_and_support": closure,
        "candidate_decisions": decisions,
        "raw_boundary_is_diagnostic_only": False,
        "raw_boundary_role": "l3r_ra0p_gradient_normalized_candidate",
        "projected_inflow_expected_zero_gradient": True,
        "future_reference_boundary_values_at_inference": False,
        "clipping_smoothing_limiter": False,
        "device": str(device),
        "amp": args.amp,
        "gpu_max_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gpu_max_reserved_memory_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "code_sha256": {
            "diagnostic": sha256_file(Path(__file__)),
            "trainer": sha256_file(
                ROOT / "scripts/time_dependent_no/train_pcno_euler2d_residual.py"
            ),
            "pcno_euler2d": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_euler2d.py"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {"summary": str(args.output_dir / "summary.json"), "decisions": decisions},
            indent=2,
        )
    )
    store.close()


if __name__ == "__main__":
    main()

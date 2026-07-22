"""Probe frozen CPGNet sensitivity to oracle boundary inputs.

This diagnostic replays individual saved model calls from an enriched
evaluate_cpg_release.py artifact.  It changes only non-normal input states and
measures the frozen checkpoint response at predeclared target nodes.  The
lagged-current and positive-scale variants are sensitivity counterfactuals, not
physically legal inflow, wall, or outflow implementations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from utility.time_dependent_no.cpg_reach import (  # noqa: E402
    build_incoming_travel_adjacency,
    cpg_dependency_mask,
    euler_characteristic_travel_times,
    minimum_predecessor_times,
)
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_REFERENCE_COMMIT,
    sha256_file,
    validate_cpg_reference_source,
)
from utility.time_dependent_no.euler2d import PRIMITIVE_NAMES  # noqa: E402


PROBE_SCHEMA = "cpg_frozen_boundary_sensitivity_v1"
MESSAGE_PASSING_LAYERS = 12
EPS = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reach-csv", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 20, 40, 58, 78])
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["shock", "smooth"],
        choices=["shock", "smooth", "upstream", "downstream"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive-scale-amplitude", type=float, default=0.01)
    parser.add_argument("--baseline-atol", type=float, default=5.0e-5)
    return parser.parse_args(argv)


def counterfactual_masks(
    injection_mask: np.ndarray,
    dependency_mask: np.ndarray,
    endpoint_cone_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return predeclared boundary groups for one target output."""

    injection = np.asarray(injection_mask, dtype=bool)
    dependency = np.asarray(dependency_mask, dtype=bool)
    cone = np.asarray(endpoint_cone_mask, dtype=bool)
    if injection.shape != dependency.shape or injection.shape != cone.shape:
        raise ValueError("counterfactual masks must share one node shape")
    return {
        "all_injected": injection,
        "support_injected": injection & dependency,
        "endpoint_cone_injected": injection & cone,
        "support_outside_endpoint_cone": injection & dependency & ~cone,
    }


def lagged_boundary_state(
    official_state: np.ndarray,
    reference_current: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Replace selected official next-reference boundary inputs by current truth."""

    official = np.asarray(official_state, dtype=np.float64)
    current = np.asarray(reference_current, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if official.shape != current.shape or official.ndim != 2 or official.shape[1] != 4:
        raise ValueError("boundary states must share shape (num_nodes, 4)")
    if selected.shape != (official.shape[0],):
        raise ValueError("boundary mask must match node count")
    result = official.copy()
    result[selected] = current[selected]
    return result


def positive_scale_boundary_state(
    official_state: np.ndarray,
    mask: np.ndarray,
    *,
    amplitude: float,
    gamma: float = 1.4,
) -> np.ndarray:
    """Apply a small positive rho/p scale probe to selected boundary nodes."""

    official = np.asarray(official_state, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if official.ndim != 2 or official.shape[1] != 4:
        raise ValueError("official_state must have shape (num_nodes, 4)")
    if selected.shape != (official.shape[0],):
        raise ValueError("boundary mask must match node count")
    if not 0.0 < amplitude < 0.25:
        raise ValueError("amplitude must lie in (0, 0.25)")
    result = official.copy()
    result[selected, 0] *= 1.0 + amplitude
    result[selected, 3] *= 1.0 + gamma * amplitude
    if np.any(result[:, 0] <= 0.0) or np.any(result[:, 3] <= 0.0):
        raise ValueError("positive-scale probe received an inadmissible state")
    return result


def sensitivity_metrics(
    *,
    baseline_output: np.ndarray,
    candidate_output: np.ndarray,
    target_truth: np.ndarray,
    target_node: int,
    normal_mask: np.ndarray,
    official_input: np.ndarray,
    candidate_input: np.ndarray,
    changed_mask: np.ndarray,
) -> dict[str, Any]:
    """Summarize target-local and normal-node output response."""

    baseline = np.asarray(baseline_output, dtype=np.float64)
    candidate = np.asarray(candidate_output, dtype=np.float64)
    truth = np.asarray(target_truth, dtype=np.float64)
    official = np.asarray(official_input, dtype=np.float64)
    changed = np.asarray(candidate_input, dtype=np.float64)
    normal = np.asarray(normal_mask, dtype=bool)
    selected = np.asarray(changed_mask, dtype=bool)
    if (
        baseline.shape != candidate.shape
        or baseline.shape != truth.shape
        or baseline.shape != official.shape
        or baseline.shape != changed.shape
        or baseline.ndim != 2
        or baseline.shape[1] != 4
    ):
        raise ValueError("primitive arrays must share shape (num_nodes, 4)")
    if normal.shape != (baseline.shape[0],) or selected.shape != normal.shape:
        raise ValueError("node masks must match primitive arrays")
    if target_node < 0 or target_node >= baseline.shape[0]:
        raise ValueError("target node lies outside primitive arrays")

    target_delta = candidate[target_node] - baseline[target_node]
    baseline_error = baseline[target_node] - truth[target_node]
    candidate_error = candidate[target_node] - truth[target_node]
    normal_delta = candidate[normal] - baseline[normal]
    input_delta = changed[selected] - official[selected]
    result: dict[str, Any] = {
        "changed_node_count": int(np.count_nonzero(selected)),
        "target_output_delta_l2": float(np.linalg.norm(target_delta)),
        "target_output_relative_delta": float(
            np.linalg.norm(target_delta) / (np.linalg.norm(baseline[target_node]) + EPS)
        ),
        "baseline_target_error_l2": float(np.linalg.norm(baseline_error)),
        "candidate_target_error_l2": float(np.linalg.norm(candidate_error)),
        "target_error_l2_change": float(
            np.linalg.norm(candidate_error) - np.linalg.norm(baseline_error)
        ),
        "normal_output_delta_rmse": float(
            np.sqrt(np.mean(normal_delta * normal_delta)) if normal_delta.size else 0.0
        ),
        "input_delta_l2": float(np.linalg.norm(input_delta)),
        "input_delta_max_abs": float(
            np.max(np.abs(input_delta)) if input_delta.size else 0.0
        ),
    }
    for index, name in enumerate(PRIMITIVE_NAMES):
        result[f"target_output_delta_{name}"] = float(target_delta[index])
        result[f"input_delta_rmse_{name}"] = float(
            np.sqrt(np.mean(input_delta[:, index] ** 2)) if input_delta.size else 0.0
        )
    return result


def _read_artifact(path: Path) -> dict[str, Any]:
    required = (
        "model_inputs",
        "model_mach",
        "reference_current",
        "raw_predicteds",
        "targets",
        "pos",
        "edges",
        "node_type",
        "injection_mask",
        "model_pos",
        "directed_edges",
        "edge_attr_before_model",
    )
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing enriched datasets {missing}")
        data = {name: np.asarray(handle[name]) for name in required}
        data["attributes"] = {
            str(name): _json_scalar(value) for name, value in handle.attrs.items()
        }
    shape = data["model_inputs"].shape
    for name in ("reference_current", "raw_predicteds", "targets"):
        if data[name].shape != shape:
            raise ValueError(f"{name} does not match model_inputs shape {shape}")
    if len(shape) != 3 or shape[-1] != 4:
        raise ValueError(f"model_inputs must have shape (T,N,4), got {shape}")
    data["node_type"] = np.asarray(data["node_type"]).reshape(-1).astype(np.int64)
    data["injection_mask"] = np.asarray(data["injection_mask"]).reshape(-1).astype(bool)
    if not np.array_equal(data["injection_mask"], data["node_type"] != 0):
        raise ValueError("injection mask does not match every non-normal node")
    if str(data["attributes"].get("reference_commit", "")) != CPG_REFERENCE_COMMIT:
        raise ValueError("artifact is not bound to the pinned CPG reference commit")
    if str(data["attributes"].get("boundary_mode", "")) != "oracle_next_reference":
        raise ValueError("artifact does not use the audited oracle boundary contract")
    return data


def _read_target_rows(
    path: Path,
    *,
    artifact_name: str,
    frames: Sequence[int],
    regions: Sequence[str],
) -> list[dict[str, Any]]:
    requested_frames = set(frames)
    requested_regions = set(regions)
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["artifact_name"] != artifact_name:
                continue
            frame = int(row["frame"])
            region = row["region"]
            key = (frame, region)
            if (
                frame in requested_frames
                and region in requested_regions
                and key not in selected
            ):
                selected[key] = {
                    "frame": frame,
                    "region": region,
                    "target_node": int(row["target_node"]),
                }
    expected = {(frame, region) for frame in frames for region in regions}
    missing = sorted(expected - set(selected))
    if missing:
        raise ValueError(
            f"reach CSV is missing requested frame/region targets: {missing}"
        )
    return [selected[key] for key in sorted(selected)]


def _import_reference_runtime(reference_repo: Path) -> dict[str, Any]:
    reference = str(reference_repo.resolve())
    if reference not in sys.path:
        sys.path.insert(0, reference)
    try:
        import torch
        from torch_geometric.data import Data

        from modelEdgeUpd.simulator import Simulator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the frozen probe requires torch, torch-geometric, and the pinned "
            "external CPGNet checkout"
        ) from exc
    return {"torch": torch, "Data": Data, "Simulator": Simulator}


def _select_device(torch: Any, name: str, gpu: int) -> Any:
    use_cuda = name == "cuda" or (name == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(gpu)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


def _make_model(api: dict[str, Any], checkpoint: Path, device: Any) -> Any:
    model = api["Simulator"](
        message_passing_num=MESSAGE_PASSING_LAYERS,
        node_input_size=6,
        edge_input_size=5,
        device=device,
    )
    model.load_checkpoint(str(checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _run_model(
    *,
    api: dict[str, Any],
    model: Any,
    data: dict[str, Any],
    state: np.ndarray,
    frame: int,
    device: Any,
) -> np.ndarray:
    torch = api["torch"]
    node_type = torch.as_tensor(data["node_type"], dtype=torch.float32)
    mach = torch.as_tensor(
        np.asarray(data["model_mach"][frame]).reshape(-1), dtype=torch.float32
    )
    primitive = torch.as_tensor(state, dtype=torch.float32)
    x = torch.cat((node_type[:, None], primitive, mach[:, None]), dim=1)
    graph = api["Data"](
        x=x,
        y=torch.as_tensor(data["targets"][frame], dtype=torch.float32),
        pos=torch.as_tensor(data["model_pos"], dtype=torch.float32),
        edge_index=torch.as_tensor(data["directed_edges"].T, dtype=torch.long),
        edge_attr=torch.as_tensor(data["edge_attr_before_model"], dtype=torch.float32),
    )
    graph.edge_ind_unique = torch.as_tensor(data["edges"].T, dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        output = model(graph, sequence_noise=None)
    return output.detach().cpu().numpy()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty sensitivity table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["region"]), str(row["input_variant"]), str(row["node_group"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (region, variant, group), values in sorted(groups.items()):
        target_delta = np.asarray(
            [row["target_output_delta_l2"] for row in values], dtype=np.float64
        )
        error_change = np.asarray(
            [row["target_error_l2_change"] for row in values], dtype=np.float64
        )
        output.append(
            {
                "region": region,
                "input_variant": variant,
                "node_group": group,
                "count": len(values),
                "target_output_delta_l2_mean": float(np.mean(target_delta)),
                "target_output_delta_l2_max": float(np.max(target_delta)),
                "target_error_l2_change_mean": float(np.mean(error_change)),
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.positive_scale_amplitude < 0.25:
        raise ValueError("positive-scale-amplitude must lie in (0, 0.25)")
    if args.baseline_atol <= 0.0:
        raise ValueError("baseline-atol must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    reference = validate_cpg_reference_source(args.reference_repo)
    data = _read_artifact(args.artifact)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    expected_checkpoint = str(data["attributes"].get("checkpoint_sha256", ""))
    if checkpoint_sha256 != expected_checkpoint:
        raise ValueError("checkpoint digest does not match the enriched artifact")
    target_rows = _read_target_rows(
        args.reach_csv,
        artifact_name=args.artifact.name,
        frames=args.frames,
        regions=args.regions,
    )
    if any(
        frame < 0 or frame >= data["model_inputs"].shape[0] for frame in args.frames
    ):
        raise ValueError("requested frame lies outside the artifact")

    api = _import_reference_runtime(args.reference_repo)
    torch = api["torch"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(torch, args.device, args.gpu)
    model = _make_model(api, args.checkpoint, device)

    rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    normal_mask = data["node_type"] == 0
    macro_dt = float(data["attributes"]["dt"])
    baseline_by_frame: dict[int, np.ndarray] = {}
    cache: dict[tuple[int, str, bytes], np.ndarray] = {}

    for target_record in target_rows:
        frame = target_record["frame"]
        target_node = target_record["target_node"]
        official = np.asarray(data["model_inputs"][frame], dtype=np.float64)
        current = np.asarray(data["reference_current"][frame], dtype=np.float64)
        if frame not in baseline_by_frame:
            baseline = _run_model(
                api=api,
                model=model,
                data=data,
                state=official,
                frame=frame,
                device=device,
            )
            expected = np.asarray(data["raw_predicteds"][frame], dtype=np.float64)
            max_abs = float(np.max(np.abs(baseline - expected)))
            if max_abs > args.baseline_atol:
                raise RuntimeError(
                    f"frame {frame} baseline replay differs by {max_abs}, "
                    f"above tolerance {args.baseline_atol}"
                )
            baseline_by_frame[frame] = baseline
            replay_checks.append({"frame": frame, "max_abs_error": max_abs})
        baseline = baseline_by_frame[frame]

        states = np.stack((current, data["targets"][frame]), axis=0)
        travel = euler_characteristic_travel_times(data["pos"], data["edges"], states)
        incoming = build_incoming_travel_adjacency(
            travel["directed_edges"],
            travel["plus_travel_time"],
            official.shape[0],
        )
        predecessor_time, _ = minimum_predecessor_times(
            incoming, target_node, cutoff=macro_dt
        )
        endpoint_cone = predecessor_time <= macro_dt * (1.0 + 1.0e-12)
        dependency, _ = cpg_dependency_mask(
            data["edges"],
            official.shape[0],
            target_node,
            message_passing_layers=MESSAGE_PASSING_LAYERS,
        )
        groups = counterfactual_masks(data["injection_mask"], dependency, endpoint_cone)

        variants: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for group_name, mask in groups.items():
            variants.append(
                (
                    "lagged_reference_current",
                    group_name,
                    mask,
                    lagged_boundary_state(official, current, mask),
                )
            )
        outside = groups["support_outside_endpoint_cone"]
        variants.append(
            (
                "positive_rho_pressure_scale",
                "support_outside_endpoint_cone",
                outside,
                positive_scale_boundary_state(
                    official,
                    outside,
                    amplitude=args.positive_scale_amplitude,
                ),
            )
        )

        for variant_name, group_name, mask, candidate_input in variants:
            cache_key = (frame, variant_name, mask.tobytes())
            if not np.any(mask):
                candidate_output = baseline
            elif cache_key in cache:
                candidate_output = cache[cache_key]
            else:
                candidate_output = _run_model(
                    api=api,
                    model=model,
                    data=data,
                    state=candidate_input,
                    frame=frame,
                    device=device,
                )
                cache[cache_key] = candidate_output
            metrics = sensitivity_metrics(
                baseline_output=baseline,
                candidate_output=candidate_output,
                target_truth=data["targets"][frame],
                target_node=target_node,
                normal_mask=normal_mask,
                official_input=official,
                candidate_input=candidate_input,
                changed_mask=mask,
            )
            rows.append(
                {
                    "artifact_name": args.artifact.name,
                    "trajectory_key": str(data["attributes"].get("trajectory_key", "")),
                    "frame": frame,
                    "region": target_record["region"],
                    "target_node": target_node,
                    "input_variant": variant_name,
                    "node_group": group_name,
                    "wall_node_count": int(
                        np.count_nonzero(mask & (data["node_type"] == 1))
                    ),
                    "outflow_node_count": int(
                        np.count_nonzero(mask & (data["node_type"] == 2))
                    ),
                    "inflow_node_count": int(
                        np.count_nonzero(mask & (data["node_type"] == 3))
                    ),
                    **metrics,
                }
            )

    _write_csv(args.output_dir / "sensitivity_rows.csv", rows)
    summary = {
        "schema": PROBE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "frozen release-checkpoint boundary-input sensitivity; no rollout, "
            "legal-boundary baseline, paper identity, or autonomous-solver claim"
        ),
        "source_sha256": {
            "script": sha256_file(Path(__file__)),
            "artifact": sha256_file(args.artifact),
            "reach_csv": sha256_file(args.reach_csv),
        },
        "reference": reference,
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "seed": args.seed,
        "frames": args.frames,
        "regions": args.regions,
        "baseline_replay": replay_checks,
        "counterfactual_contract": {
            "lagged_reference_current": (
                "replace selected oracle next-reference boundary inputs by the "
                "same trajectory's current reference boundary state"
            ),
            "positive_rho_pressure_scale": (
                f"pointwise-positive {100.0 * args.positive_scale_amplitude:g} "
                "percent diagnostic on selected oracle boundary inputs; not a "
                "physical boundary condition"
            ),
            "support_outside_endpoint_cone": (
                "inside exact 13-hop model support but outside the directed "
                "endpoint-sampled characteristic cone"
            ),
            "endpoint_speed_caveat": (
                "saved endpoints do not bound unresolved accepted DG substeps"
            ),
        },
        "legal_boundary_status": (
            "not implemented: wall normals, solver-degree mapping, and validated "
            "inflow/outflow operators remain missing"
        ),
        "positive_scale_amplitude": args.positive_scale_amplitude,
        "row_count": len(rows),
        "aggregate": _aggregate(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())

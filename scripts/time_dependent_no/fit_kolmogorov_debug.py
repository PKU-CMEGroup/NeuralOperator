"""Fixed solver-labelled debug fit on 32 already-open development pairs.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_debug --parent <packet>
--parent-source <frozen-source-root> --output <fresh-directory> --device cuda``.
No solver is called. Neither this fit nor its in-sample rollout is held-out
accuracy evidence. The real parent manifest must be audited and pinned first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO

RUN_ID = "CM_NEXT_KF_DEBUG_FIT_20260906A"
PARENT_RUN_ID = "CM_NEXT_KF_LONG_20260906A"
# Completed development parent: independently audited before this debug contract.
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "ebbfa9bf70eb4df892ce9d14e785d1c2f4e54e6c87e06da7d661424f39e8a939"
)
EXPECTED_REFERENCE_SHA256 = (
    "6c3c1938318deb52c8873243692bd06daed4f9958e010556fee3c8666fe3dbda"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_SOURCES = (
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
    "tests/time_dependent_no/test_screen_kolmogorov_trajectory_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
)
SOURCE_PATHS = (
    "scripts/time_dependent_no/fit_kolmogorov_debug.py",
    "tests/time_dependent_no/test_fit_kolmogorov_debug.py",
    "utility/time_dependent_no/pcno_kolmogorov.py",
    "tests/time_dependent_no/test_pcno_kolmogorov.py",
    "pcno/__init__.py",
    "pcno/pcno.py",
    "pcno/geo_utility.py",
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
GATE_LIMITS = {
    "clean_spatial_relative_l2": 1e-3,
    "spatial_response_over_displacement_rms": 1e-2,
    "temporal_response_over_displacement_rms": 1e-3,
    "fine_discarded_state_relative_l2": 1e-3,
    "late_endpoint_relative_l2": 2e-3,
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _bound_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("manifest path must identify a file inside its declared root")
    return path


def validate_parent(
    parent: Path, source_root: Path, *, unit_fixture: bool = False
) -> dict:
    """Verify complete packet and frozen source bytes before any array is opened."""
    manifest_path = parent / "artifact_manifest.json"
    manifest_hash = _hash(manifest_path)
    if not unit_fixture and manifest_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ValueError(
            "real parent manifest has not passed the pinned completed-packet audit"
        )
    manifest = _read(manifest_path)
    expected_id = PARENT_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if manifest["run_id"] != expected_id or manifest["source_stable"] is not True:
        raise ValueError("parent identity or source stability mismatch")
    if set(manifest["sources"]) != set(PARENT_SOURCES):
        raise ValueError("parent must bind the exact three frozen sources")
    for name, digest in manifest["sources"].items():
        if _hash(_bound_file(source_root, name)) != digest:
            raise ValueError(f"parent frozen source hash mismatch: {name}")
    if manifest["sources"][PARENT_SOURCES[-1]] != EXPECTED_REFERENCE_SHA256:
        raise ValueError(
            "parent reference solver source is not the qualified fixed source"
        )
    for name, digest in manifest["artifacts"].items():
        if _hash(_bound_file(parent, name)) != digest:
            raise ValueError(f"parent artifact hash mismatch: {name}")
    if (
        "result.json" not in manifest["artifacts"]
        or "launch.json" not in manifest["artifacts"]
    ):
        raise ValueError("parent result and launch must be artifact-bound")
    result, launch = _read(parent / "result.json"), _read(parent / "launch.json")
    if (
        result["run_id"] != expected_id
        or launch["run_id"] != expected_id
        or result["status"] != "completed"
        or result["source_stable"] is not True
        or result["engineering_gates_pass"] is not True
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or launch["sources"] != manifest["sources"]
        or result["protocol"] != launch["protocol"]
        or result["scope"]
        != ("unit_test_only" if unit_fixture else "development_trajectory_readiness")
    ):
        raise ValueError("parent completion, role, gates or source bindings mismatch")
    protocol = result["protocol"]
    expected_n, expected_steps = (16, 32) if unit_fixture else (128, 512)
    if protocol["resolution"] != expected_n or protocol["steps"] != expected_steps:
        raise ValueError(
            "parent grid or trajectory length is outside this fixed debug contract"
        )
    if protocol["seeds"] != [2026090603, 2026090604]:
        raise ValueError("parent must contain only the two declared development seeds")
    cases = result["cases"]
    if [case["seed"] for case in cases] != protocol["seeds"]:
        raise ValueError("parent case population mismatch")
    expected_queries = {
        (a, probe, sign)
        for a in protocol["anchor_steps"]
        for probe, sign in (
            ("clean", 0),
            ("low_retained", -1),
            ("low_retained", 1),
            ("high_retained", -1),
            ("high_retained", 1),
        )
    }
    for case in cases:
        if (
            case["status"] != "completed"
            or case["last_retained_step"] != expected_steps
        ):
            raise ValueError("parent trajectory is incomplete")
        identities = [
            (r["anchor_step"], r["probe"], r["sign"]) for r in case["queries"]
        ]
        if (
            len(identities) != len(expected_queries)
            or set(identities) != expected_queries
        ):
            raise ValueError("parent query identities are missing or duplicated")
        if [r["step"] for r in case["late_rows"]] != list(
            range(1, protocol["late_horizon"] + 1)
        ):
            raise ValueError("parent late comparison is incomplete")
    queries = [r for case in cases for r in case["queries"]]
    samples = {
        "clean_spatial_relative_l2": [
            r["spatial_relative_l2"] for r in queries if not r["sign"]
        ],
        "spatial_response_over_displacement_rms": [
            r["spatial_response_over_displacement_rms"] for r in queries if r["sign"]
        ],
        "temporal_response_over_displacement_rms": [
            r["temporal_response_over_displacement_rms"] for r in queries if r["sign"]
        ],
        "fine_discarded_state_relative_l2": [
            r["fine_discarded_state_relative_l2"] for r in queries
        ],
        "late_endpoint_relative_l2": [
            case["late_rows"][-1]["restricted_state_relative_l2"] for case in cases
        ],
    }
    if (
        set(result["numeric_gates"]) != set(GATE_LIMITS)
        or launch["gate_limits"] != GATE_LIMITS
    ):
        raise ValueError("parent numerical gate contract mismatch")
    for name, values in samples.items():
        gate = result["numeric_gates"][name]
        if (
            not values
            or not all(math.isfinite(v) and v >= 0 for v in values)
            or max(values) > GATE_LIMITS[name]
            or gate["pass"] is not True
            or gate["complete"] is not True
            or gate["limit"] != GATE_LIMITS[name]
            or gate["maximum"] != max(values)
            or gate["sample_count"] != len(values)
            or gate["expected_sample_count"] != len(values)
        ):
            raise ValueError(f"parent numerical gate is not verified: {name}")
    return {"manifest_sha256": manifest_hash, "manifest": manifest, "result": result}


def _load_first_pairs(parent: Path, verified: dict) -> np.ndarray:
    """Open only seed0603 blocks containing states0..32; never another seed's arrays."""
    case = verified["result"]["cases"][0]
    manifest = verified["manifest"]["artifacts"]
    chosen = {}
    for block in case["blocks"]:
        if block["last_step"] < 0 or block["first_step"] > 32:
            continue
        if manifest.get(block["file"]) != block["sha256"]:
            raise ValueError(
                "selected block is not bound to the parent artifact manifest"
            )
        with np.load(_bound_file(parent, block["file"]), allow_pickle=False) as data:
            indices, states = data["steps"], data["states"]
            if states.dtype != np.float64 or states.shape != (
                len(indices),
                case["config"]["resolution"],
                case["config"]["resolution"],
            ):
                raise ValueError(
                    "selected states must be full float64 scalar-vorticity arrays"
                )
            if indices.tolist() != list(
                range(block["first_step"], block["last_step"] + 1)
            ):
                raise ValueError(
                    "selected block indices are not the recorded contiguous range"
                )
            for index, state in zip(indices, states, strict=True):
                if not 0 <= index <= 32:
                    continue
                if int(index) in chosen or not np.isfinite(state).all():
                    raise ValueError("selected states are duplicated or nonfinite")
                chosen[int(index)] = state.copy()
    if set(chosen) != set(range(33)):
        raise ValueError("exact states0..32 are not available")
    states = np.stack([chosen[i] for i in range(33)])
    n = states.shape[-1]
    modes = np.fft.fftfreq(n) * n
    mask = (abs(modes[:, None]) <= n // 3) & (abs(modes[None, :]) <= n // 3)
    mask[0, 0] = False
    restricted = np.fft.ifft2(np.fft.fft2(states) * mask).real
    rms = np.sqrt(np.mean(states**2, axis=(1, 2)))
    residual = np.sqrt(np.mean((states - restricted) ** 2, axis=(1, 2)))
    if np.any(residual / np.maximum(rms, np.sqrt(n * n) * np.finfo(float).eps) > 1e-11):
        raise ValueError(
            "selected float64 states are not canonical; no hidden input projection"
        )
    return states


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    energy = np.mean(target.astype(np.float64) ** 2, axis=(1, 2))
    error = np.mean(difference**2, axis=(1, 2))
    return {
        "mse": float(error.mean()),
        "global_relative_l2": float(np.sqrt(error.mean() / energy.mean()))
        if energy.mean()
        else None,
        "per_step_relative_l2": [
            float(np.sqrt(e / t)) if t else None
            for e, t in zip(error, energy, strict=True)
        ],
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_debug_fit(
    parent_packet: Path,
    parent_source_root: Path,
    output: Path,
    device_name: str,
    *,
    unit_fixture: bool = False,
):
    output, parent_packet = Path(output), Path(parent_packet)
    if output.exists():
        raise FileExistsError(output)
    if device_name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    verified = validate_parent(
        parent_packet, Path(parent_source_root), unit_fixture=unit_fixture
    )
    selected = _load_first_pairs(parent_packet, verified)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    scale = float(np.sqrt(np.mean(selected[:32] ** 2)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("input-only training RMS must be finite and positive")
    input_array = selected[:32].astype(np.float32)
    target_array = selected[1:33].astype(np.float32)
    config = {
        "resolution": 16 if unit_fixture else 128,
        "modes": 2 if unit_fixture else 12,
        "width": 4 if unit_fixture else 64,
        "depth": 1 if unit_fixture else 4,
        "fc_dim": 8 if unit_fixture else 128,
        "batch_size": 8,
        "updates": 4 if unit_fixture else 256,
    }
    sources_before = {name: _hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    run_id = RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    identity = {
        "run_id": run_id,
        "scope": "synthetic_unit_fixture"
        if unit_fixture
        else "solver_labelled_development_debug_fit",
        "parent_run_id": verified["result"]["run_id"],
        "parent_manifest_sha256": verified["manifest_sha256"],
        "parent_sources": verified["manifest"]["sources"],
        "selected_seed": 2026090603,
        "input_steps": [0, 31],
        "target_steps": [1, 32],
        "selected_array_hashes": {
            "source_states_float64": _array_hash(selected),
            "inputs": _array_hash(input_array),
            "targets": _array_hash(target_array),
        },
        "config": config,
        "dtype": "float32",
        "tf32": False,
        "fitted_input_rms_float64": scale,
        "optimizer": {
            "name": "Adam",
            "lr": 1e-3,
            "seed": 17,
            "batch_order": "consecutive0..31, repeated without shuffle",
        },
        "zero_initialized_fc2": True,
        "persistence_comparator": "repeat the same unmodified float32 initial state for all 32 times",
        "shared_output_restriction": "mean-zero rectangular 2/3 Fourier restriction, with raw output retained",
        "online_solver_calls": 0,
        "heldout_access": False,
        "generalization_claim": False,
        "pde_accuracy_claim": False,
        "prospective_evidence": False,
        "sources_before": sources_before,
    }
    _write(output / "launch.json", identity)
    started = perf_counter()
    result = dict(identity)
    try:
        model = PeriodicVorticityPCNO(
            **{
                k: config[k]
                for k in ("resolution", "modes", "width", "depth", "fc_dim")
            },
            train_scale=scale,
        ).to(device)
        with torch.no_grad():
            model.pcno.fc2.weight.zero_()
            model.pcno.fc2.bias.zero_()
        inputs = torch.from_numpy(input_array).to(device)
        targets = torch.from_numpy(target_array).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def teacher():
            raw, restricted = [], []
            with torch.no_grad():
                for begin in range(0, 32, config["batch_size"]):
                    prediction = model(inputs[begin : begin + config["batch_size"]])
                    raw.append(prediction["raw_next"].cpu().numpy())
                    restricted.append(prediction["next_state"].cpu().numpy())
            return np.concatenate(raw), np.concatenate(restricted)

        initial_raw, initial_next = teacher()
        times = []
        with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
            for update in range(config["updates"]):
                begin = (update * config["batch_size"]) % 32
                _sync(device)
                step_started = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs[begin : begin + config["batch_size"]])
                loss = (
                    (
                        (
                            prediction["next_state"]
                            - targets[begin : begin + config["batch_size"]]
                        )
                        / model.train_scale
                    )
                    .square()
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite debug training loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
                optimizer.step()
                _sync(device)
                times.append(perf_counter() - step_started)
                log.write(
                    json.dumps(
                        {
                            "update": update,
                            "first_pair": begin,
                            "batch_size": config["batch_size"],
                            "normalized_mse_before_update": float(loss.detach()),
                            "gradient_norm": float(gradient),
                            "seconds": times[-1],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                log.flush()
        model.eval()
        teacher_raw, teacher_next = teacher()
        current = inputs[:1]
        rollout_raw, rollout_next = [], []
        with torch.no_grad():
            for _ in range(32):
                prediction = model(current)
                rollout_raw.append(prediction["raw_next"].cpu().numpy()[0])
                current = prediction["next_state"]
                rollout_next.append(current.cpu().numpy()[0])
        arrays = {
            "source_states_float64": selected,
            "inputs": inputs.cpu().numpy(),
            "targets": targets.cpu().numpy(),
            "normalized_inputs": (inputs / model.train_scale).cpu().numpy(),
            "normalized_targets": (targets / model.train_scale).cpu().numpy(),
            "initial_teacher_raw": initial_raw,
            "initial_teacher_next": initial_next,
            "teacher_raw": teacher_raw,
            "teacher_next": teacher_next,
            "rollout_raw": np.stack(rollout_raw),
            "rollout_next": np.stack(rollout_next),
            "persistence_next": np.repeat(inputs[:1].cpu().numpy(), 32, axis=0),
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("nonfinite final debug prediction array")
        np.savez_compressed(output / "predictions.npz", **arrays)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_updates": config["updates"],
                "config": config,
                "identity": identity,
            },
            output / "terminal_checkpoint.pt",
        )
        completion = {
            "status": "completed",
            "actual_model_scale_float32": float(model.train_scale),
            "array_hashes": {
                name: _array_hash(value) for name, value in arrays.items()
            },
            "initial_teacher_metrics": _metrics(initial_next, arrays["targets"]),
            "teacher_metrics": _metrics(teacher_next, arrays["targets"]),
            "rollout_metrics": _metrics(arrays["rollout_next"], arrays["targets"]),
            "persistence_rollout_metrics": _metrics(
                arrays["persistence_next"], arrays["targets"]
            ),
            "teacher_projection_rms": float(
                np.sqrt(np.mean((teacher_raw.astype(np.float64) - teacher_next) ** 2))
            ),
            "rollout_projection_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            arrays["rollout_raw"].astype(np.float64)
                            - arrays["rollout_next"]
                        )
                        ** 2
                    )
                )
            ),
            "update_seconds": times,
            "timing_warmup_updates_excluded": 3,
            "median_update_seconds": float(np.median(times[3:])),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
        }
        # Keep serialization checks inside the protected region. A last-step
        # overflow must produce a failed packet, not lose the result receipt.
        json.dumps(completion, allow_nan=False)
        result.update(completion)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
    result["sources_after"] = {name: _hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    if sources_before != result["sources_after"]:
        result["status"] = "invalid_source"
    result.update(
        seconds=perf_counter() - started,
        device=device_name,
        runtime={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        interpretation="Optimization/resource and in-sample self-composition debug evidence only; no held-out ranking or PDE generalization claim.",
    )
    _write(output / "result.json", result)
    _write(
        output / "artifact_manifest.json",
        {
            "run_id": run_id,
            "sources": sources_before,
            "parent_manifest_sha256": verified["manifest_sha256"],
            "artifacts": {
                p.name: _hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    result = run_debug_fit(args.parent, args.parent_source, args.output, args.device)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "seconds": result["seconds"],
            }
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

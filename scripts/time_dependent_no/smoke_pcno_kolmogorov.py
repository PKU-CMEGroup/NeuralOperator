"""Bounded synthetic adapter fit (CPU) or full-grid resource smoke (CUDA).

Run ``python -m scripts.time_dependent_no.smoke_pcno_kolmogorov --device
cpu|cuda --output <fresh-directory>``. This is not PDE training or qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO,
    canonicalize_vorticity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "scripts/time_dependent_no/smoke_pcno_kolmogorov.py",
    "tests/time_dependent_no/test_smoke_pcno_kolmogorov.py",
    "utility/time_dependent_no/pcno_kolmogorov.py",
    "tests/time_dependent_no/test_pcno_kolmogorov.py",
    "pcno/pcno.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)


@dataclass(frozen=True)
class SmokeCase:
    resolution: int
    width: int
    modes: int
    depth: int
    fc_dim: int
    batch_size: int
    updates: int
    inference_repeats: int = 10


CPU_CASES = (SmokeCase(16, 8, 2, 2, 16, 2, 100),)
CUDA_CASES = tuple(SmokeCase(n, 64, 12, 4, 128, 1, 11, 20) for n in (128, 256))


def _hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def synthetic_pair(n: int, batch_size: int, device: torch.device):
    """Two-mode states and a deliberately non-PDE, learnable residual target."""
    coordinate = torch.arange(n, device=device, dtype=torch.float32) * (
        2 * torch.pi / n
    )
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    phase = torch.arange(batch_size, device=device, dtype=torch.float32)[:, None, None]
    state = canonicalize_vorticity(
        torch.cos(x + y + phase) + 0.3 * torch.sin(2 * x - y)
    )
    target = canonicalize_vorticity(0.97 * state + 0.03 * torch.cos(4 * y))
    return state, target


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(case: SmokeCase, device: torch.device, log_path: Path) -> dict:
    torch.manual_seed(17)
    started = perf_counter()
    model = PeriodicVorticityPCNO(
        case.resolution,
        train_scale=1.0,
        modes=case.modes,
        width=case.width,
        depth=case.depth,
        fc_dim=case.fc_dim,
    ).to(device)
    state, target = synthetic_pair(case.resolution, case.batch_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    setup_seconds = perf_counter() - started
    with torch.no_grad():
        initial_loss = float(torch.mean((model(state)["next_state"] - target) ** 2))
    if not np.isfinite(initial_loss) or initial_loss <= 0.0:
        raise RuntimeError("synthetic initial loss must be finite and positive")
    losses, update_seconds, gradient_norms = [], [], []
    model.train()
    with log_path.open("a", encoding="utf-8") as log:
        for update in range(case.updates):
            _sync(device)
            step_started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(state)
            loss = torch.mean((output["next_state"] - target) ** 2)
            loss.backward()
            # Measure without clipping; a finite loss alone does not check gradients.
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float("inf"),
                error_if_nonfinite=True,
            )
            optimizer.step()
            _sync(device)
            seconds = perf_counter() - step_started
            loss_value = float(loss.detach())
            if not np.isfinite(loss_value):
                raise RuntimeError("nonfinite synthetic training loss")
            losses.append(loss_value)
            update_seconds.append(seconds)
            gradient_norms.append(float(gradient_norm))
            log.write(
                json.dumps(
                    {
                        "resolution": case.resolution,
                        "update": update,
                        "loss": loss_value,
                        "seconds": seconds,
                        "gradient_norm": gradient_norms[-1],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            log.flush()

    model.eval()
    inference_seconds = []
    with torch.no_grad():
        # Excluded inference warmup; timed calls use the same cached geometry.
        output = model(state)
        for _ in range(case.inference_repeats):
            _sync(device)
            step_started = perf_counter()
            output = model(state)
            _sync(device)
            inference_seconds.append(perf_counter() - step_started)
        final_loss = float(torch.mean((output["next_state"] - target) ** 2))
    loss_ratio = final_loss / initial_loss
    projection_rms = float(output["projection_residual"].square().mean().sqrt())
    if not all(
        np.isfinite(value) for value in (final_loss, loss_ratio, projection_rms)
    ):
        raise RuntimeError("nonfinite synthetic final metrics")
    warmup_count = min(3, case.updates - 1)
    return {
        "status": "completed",
        "config": asdict(case),
        "input_sha256": _tensor_hash(state),
        "target_sha256": _tensor_hash(target),
        "output_sha256": _tensor_hash(output["next_state"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "setup_seconds": setup_seconds,
        "seconds": perf_counter() - started,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_over_initial_loss": loss_ratio,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "update_seconds": update_seconds,
        "timing_warmup_updates_excluded": warmup_count,
        "median_update_seconds": float(np.median(update_seconds[warmup_count:])),
        "inference_seconds": inference_seconds,
        "median_inference_seconds": float(np.median(inference_seconds)),
        "projection_residual_rms": projection_rms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }


def run_smoke(output: Path, device_name: str, *, cases=None) -> dict:
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    device = torch.device(device_name)
    default_cases = CPU_CASES if device_name == "cpu" else CUDA_CASES
    selected = default_cases if cases is None else cases
    if not selected or any(c.updates < 2 or c.inference_repeats < 1 for c in selected):
        raise ValueError(
            "smoke requires nonempty cases, >=2 updates and positive repeats"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources_before = _hashes()
    torch.set_num_threads(2)
    if device_name == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    run_id = "CM_NEXT_KF_PCNO_SMOKE_20260906A_" + device_name.upper()
    if cases is not None and cases != default_cases:
        run_id += "__UNIT_FIXTURE"
    started = perf_counter()
    _write_json(
        output / "launch.json",
        {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources_before,
            "cases": [asdict(case) for case in selected],
            "device": device_name,
        },
    )
    records = []
    for case in selected:
        try:
            record = run_case(case, device, output / "updates.jsonl")
        except (RuntimeError, ValueError, FloatingPointError) as error:
            record = {
                "status": "failed",
                "config": asdict(case),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        records.append(record)
        _write_json(output / "case_records.json", records)
        print(
            json.dumps({"resolution": case.resolution, "status": record["status"]}),
            flush=True,
        )
        if record["status"] != "completed":
            break
    sources_after = _hashes()
    status = (
        "completed" if all(r["status"] == "completed" for r in records) else "failed"
    )
    if sources_before != sources_after:
        status = "invalid_source"
    result = {
        "run_id": run_id,
        "status": status,
        "device": device_name,
        "synthetic_only": True,
        "solver_calls": 0,
        "scientific_training": False,
        "existing_data_or_checkpoint_access": False,
        "recipe": "omega=cos(x+y+phase)+.3sin(2x-y); target=P(.97omega+.03cos(4y)); phase=0,...,B-1",
        "optimizer": {"name": "Adam", "lr": 0.002, "seed": 17, "train_scale": 1.0},
        "precision": "float32, no AMP; CUDA TF32 disabled",
        "cpu_threads": 2,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device_name == "cuda"
            else None,
        },
        "sources_before": sources_before,
        "sources_after": sources_after,
        "seconds": perf_counter() - started,
        "cases": records,
        "tiny_fit_90_percent_reduction": (
            records[0].get("final_over_initial_loss", float("inf")) < 0.1
            if device_name == "cpu" and status == "completed"
            else None
        ),
        "interpretation": "Adapter optimization/resource evidence only; no PDE accuracy, regime or population qualification.",
    }
    _write_json(output / "result.json", result)
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    _write_json(
        output / "artifact_manifest.json",
        {
            "run_id": run_id,
            "sources": sources_before,
            "artifacts": artifacts,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.output, args.device)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Bounded development-only trajectory readiness, not population qualification.

Run ``python -m scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness
--output <fresh-directory>``. The parent directory must already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
    resize_dealiased_vorticity,
)

RUN_ID = "CM_NEXT_KF_LONG_20260906A"
PARENT_MANIFEST_SHA256 = (
    "03aef5dd3055a99b8ac5eceed6c32ad3d4091bac0d06bd7a929636128d19e216"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
    "tests/time_dependent_no/test_screen_kolmogorov_trajectory_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
)
INITIAL_MODES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3))
GATE_LIMITS = {
    "clean_spatial_relative_l2": 1.0e-3,
    "spatial_response_over_displacement_rms": 1.0e-2,
    "temporal_response_over_displacement_rms": 1.0e-3,
    "fine_discarded_state_relative_l2": 1.0e-3,
    "late_endpoint_relative_l2": 2.0e-3,
}


@dataclass(frozen=True)
class TrajectoryProtocol:
    resolution: int = 128
    seeds: tuple[int, ...] = (2026090603, 2026090604)
    steps: int = 512
    anchor_steps: tuple[int, ...] = (0, 64, 256, 512)
    late_anchor: int = 256
    late_horizon: int = 8
    block_steps: int = 32
    macro_dt: float = 0.05
    dt_max: float = 0.002
    high_mode: tuple[int, int] = (18, 17)
    wall_seconds: float = 90 * 60
    fixture_label: str = "development_trajectory_readiness"


PROTOCOL = TrajectoryProtocol()


class BudgetExceeded(RuntimeError):
    """The compute budget ended; this is not a scientific failure."""


class BudgetedReferenceStepper(KolmogorovReferenceStepper):
    """Keep the deadline visible inside adaptive integration, without core edits."""

    def __init__(self, config, deadline):
        super().__init__(config)
        self.deadline = deadline

    def _stable_substep(self, omega_hat, remaining):
        if perf_counter() >= self.deadline:
            raise BudgetExceeded("wall budget reached between RK4 substeps")
        return super()._stable_substep(omega_hat, remaining)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: np.ndarray) -> str:
    values = np.ascontiguousarray(state, dtype="<f8")
    digest = hashlib.sha256(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(value * value)))


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return _rms(value - reference) / max(_rms(reference), np.finfo(float).eps)


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.arange(n, dtype=np.float64) * (2 * np.pi / n)
    return np.meshgrid(coordinate, coordinate, indexing="ij")


def _initial(stepper: KolmogorovReferenceStepper, seed: int) -> np.ndarray:
    x, y = _grid(stepper.config.resolution)
    phases = np.random.default_rng(seed).uniform(0.0, 2 * np.pi, len(INITIAL_MODES))
    smooth = sum(
        np.cos(kx * x + ky * y + phase)
        for (kx, ky), phase in zip(INITIAL_MODES, phases, strict=True)
    )
    laminar = stepper.laminar_vorticity()
    return stepper.canonicalize(
        laminar + smooth * (0.25 * _rms(laminar) / _rms(smooth))
    )


def _advance(stepper, state, deadline):
    if perf_counter() >= deadline:
        raise BudgetExceeded("wall budget reached between solver calls")
    started = perf_counter()
    result = stepper.advance_canonical(state)
    return result.state, {
        **asdict(result.diagnostics),
        "seconds": perf_counter() - started,
    }


def _query_rows(stepper, half, fine, anchor, protocol, deadline):
    """Yield complete same-input query rows; no intermediate fine restriction."""
    n = protocol.resolution
    x, y = _grid(n)
    probes = [("clean", 0, anchor)]
    for name, wave in (("low_retained", (1, 1)), ("high_retained", protocol.high_mode)):
        direction = stepper.canonicalize(np.cos(wave[0] * x + wave[1] * y))
        direction *= 0.01 * _rms(anchor) / _rms(direction)
        for sign in (-1, 1):
            probes.append((name, sign, stepper.canonicalize(anchor + sign * direction)))
    clean = None
    for name, sign, query in probes:
        coarse_next, coarse_call = _advance(stepper, query, deadline)
        half_next, half_call = _advance(half, query, deadline)
        lifted = resize_dealiased_vorticity(query, 2 * n)
        fine_next, fine_call = _advance(fine, lifted, deadline)
        restricted = resize_dealiased_vorticity(fine_next, n)
        lift_restricted = resize_dealiased_vorticity(restricted, 2 * n)
        if clean is None:
            clean = (coarse_next, half_next, fine_next, restricted)
        responses = (
            coarse_next - clean[0],
            half_next - clean[1],
            fine_next - clean[2],
            restricted - clean[3],
        )
        displacement_rms = _rms(query - anchor)
        discarded_response = responses[2] - resize_dealiased_vorticity(
            responses[3], 2 * n
        )
        row = {
            "probe": name,
            "sign": sign,
            "query_sha256": _state_hash(query),
            "coarse_next_sha256": _state_hash(coarse_next),
            "fine_next_sha256": _state_hash(fine_next),
            "displacement_over_anchor_rms": displacement_rms / _rms(anchor),
            "base_vs_half_dt_relative_l2": _relative(coarse_next, half_next),
            "spatial_relative_l2": _relative(half_next, restricted),
            "fine_discarded_state_relative_l2": _relative(lift_restricted, fine_next),
            "input_lift_roundtrip_relative_l2": _relative(
                resize_dealiased_vorticity(lifted, n), query
            ),
            "temporal_response_over_displacement_rms": _rms(responses[0] - responses[1])
            / displacement_rms
            if sign
            else None,
            "spatial_response_over_displacement_rms": _rms(responses[1] - responses[3])
            / displacement_rms
            if sign
            else None,
            "fine_discarded_response_over_displacement_rms": _rms(discarded_response)
            / displacement_rms
            if sign
            else None,
            "full_fine_response_mismatch_over_displacement_rms": _rms(
                resize_dealiased_vorticity(responses[1], 2 * n) - responses[2]
            )
            / displacement_rms
            if sign
            else None,
            "trusted_response_gain": _rms(responses[0]) / displacement_rms
            if sign
            else None,
            "fine_restricted_response_gain": _rms(responses[3]) / displacement_rms
            if sign
            else None,
            "full_fine_response_gain": _rms(responses[2]) / displacement_rms
            if sign
            else None,
            "calls": {
                "coarse": coarse_call,
                "half_dt": half_call,
                "fine_half_dt": fine_call,
            },
        }
        yield row


def _late_rows(half, fine, anchor, protocol, deadline):
    coarse_state = anchor.copy()
    fine_state = resize_dealiased_vorticity(anchor, 2 * protocol.resolution)
    for step in range(1, protocol.late_horizon + 1):
        coarse_state, coarse_call = _advance(half, coarse_state, deadline)
        fine_state, fine_call = _advance(fine, fine_state, deadline)
        restricted = resize_dealiased_vorticity(fine_state, protocol.resolution)
        yield {
            "step": step,
            "restricted_state_relative_l2": _relative(coarse_state, restricted),
            "fine_discarded_state_relative_l2": _relative(
                resize_dealiased_vorticity(restricted, 2 * protocol.resolution),
                fine_state,
            ),
            "coarse_sha256": _state_hash(coarse_state),
            "fine_sha256": _state_hash(fine_state),
            "calls": {"coarse_half_dt": coarse_call, "fine_half_dt": fine_call},
        }


def _flush_block(output, seed, indices, states, record):
    if not states:
        return
    name = f"trajectory_{seed}_{indices[0]:05d}_{indices[-1]:05d}.npz"
    path = output / name
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream, steps=np.asarray(indices, dtype=np.int64), states=np.stack(states)
        )
    temporary.replace(path)
    block = {
        "file": name,
        "sha256": _sha256(path),
        "first_step": indices[0],
        "last_step": indices[-1],
        "states": len(states),
    }
    record["blocks"].append(block)
    record["last_retained_step"] = indices[-1]
    record["last_retained_state_sha256"] = _state_hash(states[-1])
    _json(
        output / f"checkpoint_{seed}.json",
        {
            "seed": seed,
            "last_step": indices[-1],
            "state_sha256": record["last_retained_state_sha256"],
            "block": block,
            "meaning": "last complete float64 block; no hidden solver state",
        },
    )
    indices.clear()
    states.clear()


def _run_case(output, protocol, seed, deadline):
    config = KolmogorovReferenceConfig(
        resolution=protocol.resolution,
        viscosity=0.01,
        linear_drag=0.1,
        forcing_amplitude=1.0,
        forcing_wavenumber=4,
        macro_dt=protocol.macro_dt,
        dt_max=protocol.dt_max,
    )
    stepper = BudgetedReferenceStepper(config, deadline)
    half = BudgetedReferenceStepper(
        replace(config, dt_max=config.dt_max / 2, cfl=config.cfl / 2), deadline
    )
    fine = BudgetedReferenceStepper(
        replace(
            config,
            resolution=2 * config.resolution,
            dt_max=config.dt_max / 2,
            cfl=config.cfl / 2,
        ),
        deadline,
    )
    record = {
        "seed": seed,
        "status": "running",
        "config": asdict(config),
        "comparison_configs": {
            "half_dt": asdict(half.config),
            "fine_half_dt": asdict(fine.config),
        },
        "blocks": [],
        "last_retained_step": None,
        "trajectory_diagnostics": [],
        "queries": [],
        "late_rows": [],
    }
    case_path = output / f"case_{seed}.json"
    indices, states = [], []
    started = perf_counter()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            current = _initial(stepper, seed)
            indices.append(0)
            states.append(current)
            _flush_block(output, seed, indices, states, record)
            with (output / f"progress_{seed}.jsonl").open(
                "a", encoding="utf-8"
            ) as progress:
                for index in range(protocol.steps + 1):
                    if index:
                        current, timing = _advance(stepper, current, deadline)
                        indices.append(index)
                        states.append(current)
                    else:
                        timing = None
                    diagnostic = {
                        "step": index,
                        "structure": asdict(stepper.diagnostics_canonical(current)),
                        "state_sha256": _state_hash(current),
                        "call": timing,
                    }
                    record["trajectory_diagnostics"].append(diagnostic)
                    progress.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                    progress.flush()
                    if (
                        len(states) >= protocol.block_steps
                        or index in protocol.anchor_steps
                        or index == protocol.steps
                    ):
                        _flush_block(output, seed, indices, states, record)
                        _json(case_path, record)
                    if index in protocol.anchor_steps:
                        for row in _query_rows(
                            stepper, half, fine, current, protocol, deadline
                        ):
                            record["queries"].append({"anchor_step": index, **row})
                            _json(case_path, record)
                    if index == protocol.late_anchor:
                        for row in _late_rows(half, fine, current, protocol, deadline):
                            record["late_rows"].append(row)
                            _json(case_path, record)
                record["status"] = "completed"
    except BudgetExceeded as error:
        record.update(status="incomplete_budget", error=str(error))
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        record.update(
            status="solver_failure", error_type=type(error).__name__, error=str(error)
        )
    finally:
        _flush_block(output, seed, indices, states, record)
        record["seconds"] = perf_counter() - started
        _json(case_path, record)
    return record


def numeric_gates(records, protocol):
    queries = [row for case in records for row in case["queries"]]
    complete = len(records) == len(protocol.seeds) and all(
        case["status"] == "completed" for case in records
    )
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
            r["restricted_state_relative_l2"]
            for c in records
            for r in c["late_rows"]
            if r["step"] == protocol.late_horizon
        ],
    }
    anchor_count = len(protocol.seeds) * len(protocol.anchor_steps)
    expected_counts = {
        "clean_spatial_relative_l2": anchor_count,
        "spatial_response_over_displacement_rms": 4 * anchor_count,
        "temporal_response_over_displacement_rms": 4 * anchor_count,
        "fine_discarded_state_relative_l2": 5 * anchor_count,
        "late_endpoint_relative_l2": len(protocol.seeds),
    }
    return {
        name: {
            "maximum": max(values) if values else None,
            "limit": GATE_LIMITS[name],
            "sample_count": len(values),
            "expected_sample_count": expected_counts[name],
            "complete": complete and len(values) == expected_counts[name],
            "pass": max(values) <= GATE_LIMITS[name]
            if complete and values and len(values) == expected_counts[name]
            else None,
        }
        for name, values in samples.items()
    }


def run_screen(output: Path, *, protocol: TrajectoryProtocol = PROTOCOL):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < 512 * 1024**2:
        raise RuntimeError("at least 512 MiB free disk space is required")
    output.mkdir()
    sources_before = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    started = perf_counter()
    deadline = started + protocol.wall_seconds
    run_id = RUN_ID if protocol == PROTOCOL else RUN_ID + "__UNIT_FIXTURE"
    _json(
        output / "launch.json",
        {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": asdict(protocol),
            "gate_limits": GATE_LIMITS,
            "sources": sources_before,
            "free_disk_bytes_at_launch": free_bytes,
        },
    )
    records = []
    for seed in protocol.seeds:
        record = _run_case(output, protocol, seed, deadline)
        records.append(record)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": record["status"],
                    "last_retained_step": record["last_retained_step"],
                }
            ),
            flush=True,
        )
        if record["status"] != "completed":
            break
    sources_after = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    status = (
        "completed"
        if all(case["status"] == "completed" for case in records)
        else records[-1]["status"]
    )
    if sources_before != sources_after:
        status = "invalid_source"
    gates = numeric_gates(records, protocol)
    result = {
        "run_id": run_id,
        "status": status,
        "scope": protocol.fixture_label,
        "parent_screen_manifest_sha256": PARENT_MANIFEST_SHA256,
        "protocol": asdict(protocol),
        "initial_recipe": {
            "modes": INITIAL_MODES,
            "phases": "NumPy default_rng(seed), one uniform [0,2pi) phase per listed cosine",
            "perturbation_rms_over_laminar_rms": 0.25,
            "zero_mean_velocity": True,
        },
        "probe_recipe": {
            "low_mode": [1, 1],
            "high_mode": protocol.high_mode,
            "signed_rms_over_anchor": [-0.01, 0.01],
            "temporal_pair": "same input; halve both dt_max and cfl so CFL-limited steps refine",
            "spatial_pair": "same input; both grids use half dt_max and half cfl",
            "fine_response": "full fine clean-displaced response; restriction used only for diagnostics",
            "late_comparison": "independent full coarse/fine composition from lifted late_anchor; both use half dt_max and half cfl; endpoint is gated",
        },
        "budget_accounting": "checked before every solver call and RK4 substep; flushing and final hashes occur after compute stops",
        "numeric_gates": gates,
        "engineering_gates_pass": all(g["pass"] for g in gates.values())
        if status == "completed"
        else None,
        "qualification_claim": False,
        "stationarity_claim": False,
        "manifold_claim": False,
        "population_qualification_claim": False,
        "primary_pde_selection": False,
        "existing_scientific_data_access": False,
        "generated_solver_states": True,
        "model_access": False,
        "remote_execution": False,
        "sources_before": sources_before,
        "sources_after": sources_after,
        "source_stable": sources_before == sources_after,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "seconds": perf_counter() - started,
        "cases": records,
        "interpretation": "Numerical gates concern this development-only finite bank; no stationarity, manifold, population or primary-PDE claim. Budget expiration is incomplete execution, not scientific failure. No threshold is imposed on discarded fine response.",
    }
    _json(output / "result.json", result)
    artifacts = {p.name: _sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    _json(
        output / "artifact_manifest.json",
        {
            "run_id": run_id,
            "sources": sources_before,
            "source_stable": result["source_stable"],
            "artifacts": artifacts,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_screen(args.output)
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

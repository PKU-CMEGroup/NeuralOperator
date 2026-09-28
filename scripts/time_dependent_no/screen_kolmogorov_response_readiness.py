"""Fixed N128/N256 successor to the fresh local response screen, not qualification.

Run with ``python -m scripts.time_dependent_no.screen_kolmogorov_response_readiness
--output <new-directory>``. No existing trajectories or learned models are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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

RUN_ID = "CM_NEXT_KF_R0_20260906B"
PARENT_RUN_ID = "CM_NEXT_KF_R0_20260906A"
PARENT_MANIFEST_SHA256 = (
    "d2101f544b9e2d55e52165a9a1d98d24fbc02ba481ea7dff71f23429e286a464"
)
PARENT_SOURCE_ARCHIVE_SHA256 = (
    "44b2b94ab922bb743ca44539a4007bd77fcfb37c8f86a4129d90f19cbcec9e33"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "scripts/time_dependent_no/screen_kolmogorov_response_readiness.py",
    "tests/time_dependent_no/test_screen_kolmogorov_response_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
)
INITIAL_MODES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3))


@dataclass(frozen=True)
class ScreenProtocol:
    """Fixed scientific CLI protocol; replacements are synthetic test fixtures."""

    resolution: int = 128
    viscosities: tuple[float, ...] = (0.01, 0.005)
    seeds: tuple[int, ...] = (2026090601, 2026090602)
    macro_dt: float = 0.05
    dt_max: float = 0.002
    anchor_steps: int = 4
    high_mode: tuple[int, int] = (18, 17)
    fixture_label: str = "preliminary_grid_refinement"


PROTOCOL = ScreenProtocol()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return _rms(value - reference) / max(_rms(reference), np.finfo(float).eps)


def _mode(stepper: KolmogorovReferenceStepper, mode: tuple[int, int]) -> np.ndarray:
    coordinate = np.arange(stepper.config.resolution) * (
        2.0 * np.pi / stepper.config.resolution
    )
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.cos(mode[0] * x + mode[1] * y)


def _initial_state(stepper: KolmogorovReferenceStepper, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coordinate = np.arange(stepper.config.resolution) * (
        2.0 * np.pi / stepper.config.resolution
    )
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(INITIAL_MODES))
    smooth = sum(
        np.cos(kx * x + ky * y + phase)
        for (kx, ky), phase in zip(INITIAL_MODES, phases, strict=True)
    )
    laminar = stepper.laminar_vorticity()
    smooth *= 0.25 * _rms(laminar) / _rms(smooth)
    return stepper.canonicalize(laminar + smooth)


def _advance(
    stepper: KolmogorovReferenceStepper, state: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    started = perf_counter()
    result = stepper.advance_canonical(state)
    return result.state, {
        **asdict(result.diagnostics),
        "seconds": perf_counter() - started,
    }


def _case(protocol: ScreenProtocol, viscosity: float, seed: int) -> dict[str, object]:
    config = KolmogorovReferenceConfig(
        resolution=protocol.resolution,
        viscosity=viscosity,
        linear_drag=0.1,
        forcing_amplitude=1.0,
        forcing_wavenumber=4,
        macro_dt=protocol.macro_dt,
        dt_max=protocol.dt_max,
    ).validated()
    stepper = KolmogorovReferenceStepper(config)
    half = KolmogorovReferenceStepper(replace(config, dt_max=config.dt_max / 2.0))
    fine = KolmogorovReferenceStepper(
        replace(config, resolution=2 * config.resolution, dt_max=config.dt_max / 2.0)
    )
    initial = _initial_state(stepper, seed)
    states = [initial]
    trajectory_calls = []
    for _ in range(protocol.anchor_steps):
        state, record = _advance(stepper, states[-1])
        states.append(state)
        trajectory_calls.append(record)

    repeated, repeat_call = _advance(stepper, initial.copy())
    restart_index = protocol.anchor_steps // 2
    restarted = KolmogorovReferenceStepper(config)
    restarted_state = states[restart_index].copy()
    restart_calls = []
    for _ in range(restart_index, protocol.anchor_steps):
        restarted_state, record = _advance(restarted, restarted_state)
        restart_calls.append(record)

    anchor = states[-1]
    anchor_rms = _rms(anchor)
    raw = anchor + 0.01 * anchor_rms * (
        1.0 + _mode(stepper, (config.resolution // 3 + 2, 0))
    )
    raw_rejection = None
    try:
        stepper.advance_canonical(raw)
    except ValueError as error:
        raw_rejection = str(error)

    probes: list[tuple[str, int, np.ndarray]] = [("clean", 0, anchor)]
    for name, wave in (("low_retained", (1, 1)), ("high_retained", protocol.high_mode)):
        direction = stepper.canonicalize(_mode(stepper, wave))
        direction *= 0.01 * anchor_rms / _rms(direction)
        for sign in (-1, 1):
            probes.append((name, sign, stepper.canonicalize(anchor + sign * direction)))

    clean_next = None
    clean_half_next = None
    clean_restricted_next = None
    rows = []
    for name, sign, query in probes:
        coarse_next, coarse_call = _advance(stepper, query)
        half_next, half_call = _advance(half, query)
        lifted = resize_dealiased_vorticity(query, fine.config.resolution)
        fine_next, fine_call = _advance(fine, lifted)
        restricted = resize_dealiased_vorticity(fine_next, config.resolution)
        lifted_restriction = resize_dealiased_vorticity(
            restricted, fine.config.resolution
        )
        if clean_next is None:
            clean_next = coarse_next
            clean_half_next = half_next
            clean_restricted_next = restricted
        displacement_rms = _rms(query - anchor)
        response = coarse_next - clean_next
        half_response = half_next - clean_half_next
        restricted_response = restricted - clean_restricted_next
        response_rms = _rms(response)
        rows.append(
            {
                "probe": name,
                "sign": sign,
                "query_sha256": _array_sha256(query),
                "coarse_next_sha256": _array_sha256(coarse_next),
                "half_dt_next_sha256": _array_sha256(half_next),
                "fine_next_sha256": _array_sha256(fine_next),
                "displacement_over_anchor_rms": displacement_rms / anchor_rms,
                "trusted_response_over_anchor_rms": response_rms / anchor_rms,
                "trusted_response_gain": (
                    response_rms / displacement_rms if sign else None
                ),
                "half_dt_response_gain": (
                    _rms(half_response) / displacement_rms if sign else None
                ),
                "fine_restricted_response_gain": (
                    _rms(restricted_response) / displacement_rms if sign else None
                ),
                "base_vs_half_response_over_displacement_rms": (
                    _rms(response - half_response) / displacement_rms if sign else None
                ),
                "coarse_half_vs_fine_restricted_response_over_displacement_rms": (
                    _rms(half_response - restricted_response) / displacement_rms
                    if sign
                    else None
                ),
                "base_vs_half_dt_relative_l2": _relative_error(coarse_next, half_next),
                "coarse_half_vs_restricted_fine_half_relative_l2": _relative_error(
                    half_next, restricted
                ),
                "fine_discarded_relative_l2": _relative_error(
                    lifted_restriction, fine_next
                ),
                "input_lift_roundtrip_relative_l2": _relative_error(
                    resize_dealiased_vorticity(lifted, config.resolution), query
                ),
                "query_projection_relative_l2": stepper.projection_relative_l2(query),
                "output_projection_relative_l2": stepper.projection_relative_l2(
                    coarse_next
                ),
                "query_structure": asdict(stepper.diagnostics_canonical(query)),
                "next_structure": asdict(stepper.diagnostics_canonical(coarse_next)),
                "calls": {
                    "coarse": coarse_call,
                    "half_dt": half_call,
                    "fine_half_dt": fine_call,
                },
            }
        )

    return {
        "status": "completed",
        "viscosity": viscosity,
        "seed": seed,
        "config": asdict(config),
        "initial_sha256": _array_sha256(initial),
        "anchor_sha256": _array_sha256(anchor),
        "initial_structure": asdict(stepper.diagnostics_canonical(initial)),
        "anchor_structure": asdict(stepper.diagnostics_canonical(anchor)),
        "anchor_rms": anchor_rms,
        "restart_at_macro_step": restart_index,
        "restart_exact": bool(np.array_equal(restarted_state, anchor)),
        "repeat_exact": bool(np.array_equal(repeated, states[1])),
        "raw_input_rejected": raw_rejection is not None,
        "raw_rejection": raw_rejection,
        "raw_input_projection_relative_l2": stepper.projection_relative_l2(raw),
        "canonicalized_raw_projection_relative_l2": stepper.projection_relative_l2(
            stepper.canonicalize(raw)
        ),
        "trajectory_calls": trajectory_calls,
        "repeat_call": repeat_call,
        "restart_calls": restart_calls,
        "probe_rows": rows,
    }


def run_screen(
    output: Path, *, protocol: ScreenProtocol = PROTOCOL
) -> dict[str, object]:
    """Write a new bounded packet; failures are evidence, never a promotion gate."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources_before = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    started = perf_counter()
    rows = []
    for viscosity in protocol.viscosities:
        for seed in protocol.seeds:
            case_started = perf_counter()
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    row = _case(protocol, viscosity, seed)
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                OverflowError,
            ) as error:
                row = {
                    "status": "failed",
                    "viscosity": viscosity,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            row["seconds"] = perf_counter() - case_started
            rows.append(row)
            # Preserve each finished case if a subsequent case is interrupted.
            (output / "case_records.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    sources_after = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    source_stable = sources_before == sources_after
    status = (
        "completed"
        if all(row["status"] == "completed" for row in rows)
        else "incomplete"
    )
    if not source_stable:
        status = "invalid_source"
    result = {
        "run_id": RUN_ID if protocol == PROTOCOL else RUN_ID + "__UNIT_FIXTURE",
        "parent_screen": {
            "run_id": PARENT_RUN_ID,
            "artifact_manifest_sha256": PARENT_MANIFEST_SHA256,
            "source_archive_sha256": PARENT_SOURCE_ARCHIVE_SHA256,
            "relation": "same physical initial recipe; anchor regenerated on successor grid",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": protocol.fixture_label,
        "qualification_claim": False,
        "stationarity_claim": False,
        "population_qualification_claim": False,
        "coarse_state_closure_claim": False,
        "primary_pde_selection": False,
        "existing_scientific_data_access": False,
        "generated_solver_states": True,
        "checkpoint_access": False,
        "model_access": False,
        "remote_execution": False,
        "protocol": asdict(protocol),
        "initial_recipe": {
            "modes": INITIAL_MODES,
            "randomness": "NumPy default_rng(seed), one uniform [0,2pi) phase per listed cosine",
            "perturbation_rms_over_laminar_rms": 0.25,
            "mean_velocity": [0.0, 0.0],
        },
        "probe_recipe": {
            "low_retained_mode": [1, 1],
            "high_retained_mode": protocol.high_mode,
            "signed_displacement_over_anchor_rms": [-0.01, 0.01],
            "geometry_interpretation": "Fourier directions only, not tangent or normal labels",
            "spatial_comparison": "same lifted input; both grids use half dt_max",
            "response_convergence": "difference between clean-displaced responses, divided by input displacement RMS",
            "discarded_content": "full fine successor minus lift of its coarse restriction",
            "raw_rejection_input": "anchor + .01 anchor_RMS * (1 + cos((N//3+2)*x))",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "sources_before": sources_before,
        "sources_after": sources_after,
        "source_stable": source_stable,
        "seconds": perf_counter() - started,
        "cases": rows,
        "interpretation": (
            "Preliminary finite-grid one-step response, resolution, restart and cost evidence only. "
            "No learned predictions, stationarity assessment, chaotic-regime inference, "
            "population qualification or primary-case decision. Numerical state projection "
            "is not a training-manifold projection."
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": result["run_id"],
        "sources": sources_before,
        "source_stable": result["source_stable"],
        "artifacts": {
            name: _sha256(output / name)
            for name in ("case_records.json", "result.json")
        },
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
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
        )
    )
    if result["status"] != "completed" or not result["source_stable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

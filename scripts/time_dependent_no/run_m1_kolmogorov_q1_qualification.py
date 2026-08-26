"""Run the registered M1-Q1 Kolmogorov solver-only qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
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

RUN_ID = "M1-KF-Q1-20260826A"
REPO_ROOT = Path(__file__).resolve().parents[2]
INITIAL_SEEDS = (2026082601, 2026082602, 2026082603, 2026082604)
PERTURBATION_SEEDS = (2026082701, 2026082702, 2026082703)
PERTURBATION_BANDS = ((1, 4), (5, 10), (11, 20))
SOURCE_PATHS = (
    "utility/time_dependent_no/kolmogorov_reference.py",
    "tests/time_dependent_no/test_kolmogorov_reference.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_a1_preflight.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_qualification.py",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_TRACKER.md",
)


@dataclass(frozen=True)
class Q1Settings:
    """Execution sizes separated from the physical reference-map contract."""

    burnin_calls: int
    extended_burnin_calls: int
    observation_calls: int
    path_horizon: int
    structure_horizon: int
    spatial_resolution: int
    initial_rms: float
    time_steps: tuple[float, ...]
    perturbation_bands: tuple[tuple[int, int], ...]


def _full_contract() -> tuple[KolmogorovReferenceConfig, Q1Settings]:
    return KolmogorovReferenceConfig(), Q1Settings(
        burnin_calls=256,
        extended_burnin_calls=512,
        observation_calls=128,
        path_horizon=16,
        structure_horizon=64,
        spatial_resolution=128,
        initial_rms=4.0,
        time_steps=(0.002, 0.001, 0.0005),
        perturbation_bands=PERTURBATION_BANDS,
    )


def _quick_contract() -> tuple[KolmogorovReferenceConfig, Q1Settings]:
    return (
        KolmogorovReferenceConfig(
            resolution=18,
            viscosity=0.02,
            linear_drag=0.1,
            forcing_amplitude=0.2,
            forcing_wavenumber=2,
            macro_dt=0.01,
            dt_max=0.002,
        ),
        Q1Settings(
            burnin_calls=4,
            extended_burnin_calls=8,
            observation_calls=4,
            path_horizon=2,
            structure_horizon=4,
            spatial_resolution=36,
            initial_rms=1.0,
            time_steps=(0.002, 0.001, 0.0005),
            perturbation_bands=((1, 2), (3, 4), (5, 8)),
        ),
    )


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(reference)),
        math.sqrt(reference.size) * np.finfo(np.float64).eps,
    )
    return float(np.linalg.norm(value - reference) / denominator)


def _relative_mean_change(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(
        float(np.mean(np.abs(np.concatenate((first, second))))),
        np.finfo(np.float64).eps,
    )
    return float(abs(float(np.mean(second)) - float(np.mean(first))) / denominator)


def _spearman_time(values: np.ndarray) -> float:
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size)
    time = np.arange(values.size, dtype=np.float64)
    correlation = np.corrcoef(time, ranks)[0, 1]
    return float(correlation)


def _normalized_wasserstein(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.mean(np.abs(reference))), np.finfo(float).eps)
    return float(np.mean(np.abs(np.sort(value) - np.sort(reference))) / denominator)


def _normalized_spectrum(
    stepper: KolmogorovReferenceStepper, states: np.ndarray
) -> np.ndarray:
    spectra = []
    for state in states:
        _, energy = stepper.kinetic_energy_spectrum_canonical(state)
        spectra.append(energy)
    maximum_size = max(spectrum.size for spectrum in spectra)
    padded = np.zeros((len(spectra), maximum_size), dtype=np.float64)
    for index, spectrum in enumerate(spectra):
        padded[index, : spectrum.size] = spectrum
    mean_spectrum = np.mean(padded, axis=0)
    total = max(float(np.sum(mean_spectrum)), np.finfo(np.float64).eps)
    return mean_spectrum / total


def _spectrum_total_variation(value: np.ndarray, reference: np.ndarray) -> float:
    size = max(value.size, reference.size)
    value_padded = np.zeros(size, dtype=np.float64)
    reference_padded = np.zeros(size, dtype=np.float64)
    value_padded[: value.size] = value
    reference_padded[: reference.size] = reference
    return float(0.5 * np.sum(np.abs(value_padded - reference_padded)))


def _initial_state(
    stepper: KolmogorovReferenceStepper, seed: int, target_rms: float
) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=stepper.state_shape)
    canonical = stepper.canonicalize(raw)
    rms = float(np.sqrt(np.mean(canonical**2)))
    if rms <= np.finfo(np.float64).tiny:
        raise RuntimeError("initial-state canonicalization produced zero RMS")
    return stepper.canonicalize(canonical * (target_rms / rms))


def _array_sha256(value: np.ndarray) -> str:
    values = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _state_metadata(
    stepper: KolmogorovReferenceStepper, state: np.ndarray
) -> dict[str, float | str]:
    return {
        "sha256": _array_sha256(state),
        "rms": float(np.sqrt(np.mean(state**2))),
        "projection_relative_l2": stepper.projection_relative_l2(state),
    }


def _advance_final(
    stepper: KolmogorovReferenceStepper, state: np.ndarray, calls: int
) -> np.ndarray:
    current = state
    for _ in range(calls):
        current = stepper.advance_canonical(current).state
    return current


def _observation_window(
    stepper: KolmogorovReferenceStepper, state: np.ndarray, calls: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rollout, _ = stepper.rollout_canonical(state, calls)
    energy = np.empty(calls, dtype=np.float64)
    enstrophy = np.empty(calls, dtype=np.float64)
    for index, observed in enumerate(rollout[1:]):
        diagnostics = stepper.diagnostics_canonical(observed)
        energy[index] = diagnostics.kinetic_energy
        enstrophy[index] = diagnostics.enstrophy
    return rollout[-1], energy, enstrophy


def _stationarity_rows(
    energy_series: list[np.ndarray], enstrophy_series: list[np.ndarray]
) -> tuple[list[dict[str, float | int | bool]], dict[str, object]]:
    rows: list[dict[str, float | int | bool]] = []
    energy_rhos = []
    enstrophy_rhos = []
    for seed, energy, enstrophy in zip(
        INITIAL_SEEDS, energy_series, enstrophy_series, strict=True
    ):
        midpoint = energy.size // 2
        energy_change = _relative_mean_change(energy[:midpoint], energy[midpoint:])
        enstrophy_change = _relative_mean_change(
            enstrophy[:midpoint], enstrophy[midpoint:]
        )
        energy_rho = _spearman_time(energy)
        enstrophy_rho = _spearman_time(enstrophy)
        energy_rhos.append(energy_rho)
        enstrophy_rhos.append(enstrophy_rho)
        rows.append(
            {
                "seed": seed,
                "energy_half_change": energy_change,
                "enstrophy_half_change": enstrophy_change,
                "energy_spearman": energy_rho,
                "enstrophy_spearman": enstrophy_rho,
                "half_change_pass": energy_change <= 0.1 and enstrophy_change <= 0.1,
            }
        )

    def drift_summary(values: list[float]) -> dict[str, int | bool]:
        positive = sum(value >= 0.5 for value in values)
        negative = sum(value <= -0.5 for value in values)
        return {
            "positive_count": positive,
            "negative_count": negative,
            "shared_drift_pass": max(positive, negative) <= 0.75 * len(values),
        }

    summary = {
        "all_half_changes_pass": all(bool(row["half_change_pass"]) for row in rows),
        "energy_drift": drift_summary(energy_rhos),
        "enstrophy_drift": drift_summary(enstrophy_rhos),
    }
    summary["pass"] = (
        bool(summary["all_half_changes_pass"])
        and bool(
            summary["energy_drift"]["shared_drift_pass"]  # type: ignore[index]
        )
        and bool(summary["enstrophy_drift"]["shared_drift_pass"])
    )  # type: ignore[index]
    return rows, summary


def _burnin_and_stationarity(
    stepper: KolmogorovReferenceStepper, settings: Q1Settings
) -> tuple[list[np.ndarray], dict[str, object], float]:
    started = perf_counter()
    initial_states = [
        _initial_state(stepper, seed, settings.initial_rms) for seed in INITIAL_SEEDS
    ]
    first_burnin = [
        _advance_final(stepper, state, settings.burnin_calls)
        for state in initial_states
    ]
    first_outputs = [
        _observation_window(stepper, state, settings.observation_calls)
        for state in first_burnin
    ]
    first_rows, first_summary = _stationarity_rows(
        [output[1] for output in first_outputs],
        [output[2] for output in first_outputs],
    )
    attempts = [
        {
            "burnin_calls": settings.burnin_calls,
            "rows": first_rows,
            "summary": first_summary,
        }
    ]
    if bool(first_summary["pass"]):
        chosen_states = first_burnin
        chosen_burnin = settings.burnin_calls
    else:
        extension = (
            settings.extended_burnin_calls
            - settings.burnin_calls
            - settings.observation_calls
        )
        if extension < 0:
            raise ValueError(
                "extended burn-in must include the first observation window"
            )
        extended_burnin = [
            _advance_final(stepper, output[0], extension) for output in first_outputs
        ]
        second_outputs = [
            _observation_window(stepper, state, settings.observation_calls)
            for state in extended_burnin
        ]
        second_rows, second_summary = _stationarity_rows(
            [output[1] for output in second_outputs],
            [output[2] for output in second_outputs],
        )
        attempts.append(
            {
                "burnin_calls": settings.extended_burnin_calls,
                "rows": second_rows,
                "summary": second_summary,
            }
        )
        chosen_states = extended_burnin
        chosen_burnin = settings.extended_burnin_calls
    chosen_summary = attempts[-1]["summary"]
    result = {
        "attempts": attempts,
        "chosen_burnin_calls": chosen_burnin,
        "extension_used": len(attempts) == 2,
        "pass": bool(chosen_summary["pass"]),  # type: ignore[index]
        "initial_states": [
            {
                "seed": seed,
                **_state_metadata(stepper, state),
            }
            for seed, state in zip(INITIAL_SEEDS, initial_states, strict=True)
        ],
        "chosen_post_burnin_states": [
            {
                "seed": seed,
                **_state_metadata(stepper, state),
            }
            for seed, state in zip(INITIAL_SEEDS, chosen_states, strict=True)
        ],
    }
    return chosen_states, result, perf_counter() - started


def _band_perturbation(
    stepper: KolmogorovReferenceStepper,
    clean: np.ndarray,
    seed: int,
    band: tuple[int, int],
) -> np.ndarray:
    n = stepper.config.resolution
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    mode_x, mode_y = np.meshgrid(modes, modes, indexing="ij")
    radius = np.sqrt(mode_x**2 + mode_y**2)
    mask = (radius >= band[0]) & (radius <= band[1])
    raw = np.random.default_rng(seed).normal(size=(n, n))
    perturbation_hat = np.fft.fft2(raw) * mask
    perturbation_hat[0, 0] = 0.0
    perturbation = stepper.canonicalize(np.fft.ifft2(perturbation_hat).real)
    target_rms = 0.03 * float(np.sqrt(np.mean(clean**2)))
    perturbation_rms = float(np.sqrt(np.mean(perturbation**2)))
    if perturbation_rms <= np.finfo(np.float64).tiny:
        raise RuntimeError("band perturbation has zero RMS")
    return stepper.canonicalize(clean + perturbation * (target_rms / perturbation_rms))


def _calibration_inputs(
    stepper: KolmogorovReferenceStepper,
    clean_states: list[np.ndarray],
    perturbation_bands: tuple[tuple[int, int], ...],
) -> list[dict[str, object]]:
    if len(perturbation_bands) != 3:
        raise ValueError("exactly three perturbation bands are required")
    inputs: list[dict[str, object]] = []
    for index, clean in enumerate(clean_states[:3]):
        inputs.append(
            {
                "case": f"clean_{index}",
                "family": "clean",
                "state": clean,
                "seed": INITIAL_SEEDS[index],
            }
        )
        displaced = _band_perturbation(
            stepper,
            clean,
            PERTURBATION_SEEDS[index],
            perturbation_bands[index],
        )
        clean_rms = float(np.sqrt(np.mean(clean**2)))
        inputs.append(
            {
                "case": f"displaced_{index}",
                "family": "displaced",
                "state": displaced,
                "seed": PERTURBATION_SEEDS[index],
                "band": list(perturbation_bands[index]),
                "relative_perturbation_rms": float(
                    np.sqrt(np.mean((displaced - clean) ** 2)) / clean_rms
                ),
            }
        )
    return inputs


def _calibration_metadata(
    stepper: KolmogorovReferenceStepper, inputs: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for item in inputs:
        state = item["state"]
        if not isinstance(state, np.ndarray):
            raise TypeError("calibration input state is not an array")
        row = {key: value for key, value in item.items() if key != "state"}
        row.update(_state_metadata(stepper, state))
        rows.append(row)
    return rows


def _trajectory_structure(
    stepper: KolmogorovReferenceStepper, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.empty(states.shape[0], dtype=np.float64)
    enstrophy = np.empty(states.shape[0], dtype=np.float64)
    for index, state in enumerate(states):
        diagnostics = stepper.diagnostics_canonical(state)
        energy[index] = diagnostics.kinetic_energy
        enstrophy[index] = diagnostics.enstrophy
    spectrum = _normalized_spectrum(stepper, states)
    return energy, enstrophy, spectrum


def _time_refinement(
    base_config: KolmogorovReferenceConfig,
    settings: Q1Settings,
    inputs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, np.ndarray], float]:
    started = perf_counter()
    steppers = {
        step: KolmogorovReferenceStepper(replace(base_config, dt_max=step))
        for step in settings.time_steps
    }
    fine_step = min(settings.time_steps)
    candidate_step = max(settings.time_steps)
    rows: list[dict[str, object]] = []
    fine_rollouts: dict[str, np.ndarray] = {}
    maximum_projection = 0.0
    all_finite = True
    for item in inputs:
        state = item["state"]
        if not isinstance(state, np.ndarray):
            raise TypeError("calibration input state is not an array")
        rollouts = {
            step: steppers[step].rollout_canonical(state, settings.structure_horizon)[0]
            for step in settings.time_steps
        }
        projections = {
            step: max(steppers[step].projection_relative_l2(value) for value in rollout)
            for step, rollout in rollouts.items()
        }
        maximum_projection = max(maximum_projection, *projections.values())
        all_finite = all_finite and all(
            bool(np.all(np.isfinite(rollout))) for rollout in rollouts.values()
        )
        fine = rollouts[fine_step]
        fine_rollouts[str(item["case"])] = fine
        fine_energy, fine_enstrophy, fine_spectrum = _trajectory_structure(
            steppers[fine_step], fine[1:]
        )
        for step in settings.time_steps:
            if step == fine_step:
                continue
            rollout = rollouts[step]
            energy, enstrophy, spectrum = _trajectory_structure(
                steppers[step], rollout[1:]
            )
            projection = projections[step]
            rows.append(
                {
                    "case": item["case"],
                    "family": item["family"],
                    "dt_max": step,
                    "reference_dt_max": fine_step,
                    "one_step_relative_l2": _relative_l2(rollout[1], fine[1]),
                    "path_horizon": settings.path_horizon,
                    "path_relative_l2": _relative_l2(
                        rollout[settings.path_horizon], fine[settings.path_horizon]
                    ),
                    "energy_wasserstein": _normalized_wasserstein(energy, fine_energy),
                    "enstrophy_wasserstein": _normalized_wasserstein(
                        enstrophy, fine_enstrophy
                    ),
                    "spectrum_total_variation": _spectrum_total_variation(
                        spectrum, fine_spectrum
                    ),
                    "maximum_projection_relative_l2": projection,
                    "finite": bool(np.all(np.isfinite(rollout))),
                }
            )

    candidate_rows = [row for row in rows if row["dt_max"] == candidate_step]
    family_summaries = {}
    for family in ("clean", "displaced"):
        family_rows = [row for row in candidate_rows if row["family"] == family]
        one_step = np.asarray(
            [row["one_step_relative_l2"] for row in family_rows], dtype=np.float64
        )
        path = np.asarray(
            [row["path_relative_l2"] for row in family_rows], dtype=np.float64
        )
        family_summaries[family] = {
            "one_step_median": float(np.median(one_step)),
            "one_step_maximum": float(np.max(one_step)),
            "path_median": float(np.median(path)),
            "path_maximum": float(np.max(path)),
        }
    one_step_pass = all(
        summary["one_step_median"] < 1.0e-5 and summary["one_step_maximum"] < 1.0e-4
        for summary in family_summaries.values()
    )
    path_pass = all(
        summary["path_median"] < 1.0e-3 and summary["path_maximum"] < 5.0e-3
        for summary in family_summaries.values()
    )
    structure_pass = all(
        float(row[metric]) < 0.02
        for row in candidate_rows
        for metric in (
            "energy_wasserstein",
            "enstrophy_wasserstein",
            "spectrum_total_variation",
        )
    )
    summary = {
        "candidate_dt_max": candidate_step,
        "reference_dt_max": fine_step,
        "families": family_summaries,
        "one_step_pass": one_step_pass,
        "path_pass": path_pass,
        "structure_pass": structure_pass,
        "all_finite": all_finite,
        "maximum_projection_relative_l2": maximum_projection,
    }
    summary["pass"] = (
        one_step_pass
        and path_pass
        and structure_pass
        and all_finite
        and maximum_projection <= base_config.canonical_tolerance
    )
    return {"rows": rows, "summary": summary}, fine_rollouts, perf_counter() - started


def _spatial_context(
    base_config: KolmogorovReferenceConfig,
    settings: Q1Settings,
    inputs: list[dict[str, object]],
    fine_rollouts: dict[str, np.ndarray],
) -> tuple[dict[str, object], float]:
    started = perf_counter()
    fine_step = min(settings.time_steps)
    high_config = replace(
        base_config,
        resolution=settings.spatial_resolution,
        dt_max=fine_step,
    )
    high_stepper = KolmogorovReferenceStepper(high_config)
    rows = []
    for item in inputs:
        state = item["state"]
        if not isinstance(state, np.ndarray):
            raise TypeError("calibration input state is not an array")
        lifted = resize_dealiased_vorticity(state, settings.spatial_resolution)
        high_rollout, _ = high_stepper.rollout_canonical(lifted, settings.path_horizon)
        restricted_one = resize_dealiased_vorticity(
            high_rollout[1], base_config.resolution
        )
        restricted_path = resize_dealiased_vorticity(
            high_rollout[settings.path_horizon], base_config.resolution
        )
        low_rollout = fine_rollouts[str(item["case"])]
        rows.append(
            {
                "case": item["case"],
                "family": item["family"],
                "one_step_relative_l2": _relative_l2(restricted_one, low_rollout[1]),
                "path_horizon": settings.path_horizon,
                "path_relative_l2": _relative_l2(
                    restricted_path, low_rollout[settings.path_horizon]
                ),
                "high_projection_relative_l2": max(
                    high_stepper.projection_relative_l2(value) for value in high_rollout
                ),
                "finite": bool(np.all(np.isfinite(high_rollout))),
            }
        )
    return {
        "rows": rows,
        "status": "pending_model_relative_factor_of_four_gate",
        "all_finite": all(bool(row["finite"]) for row in rows),
        "one_step_median": float(
            np.median([row["one_step_relative_l2"] for row in rows])
        ),
        "one_step_maximum": float(
            np.max([row["one_step_relative_l2"] for row in rows])
        ),
        "path_median": float(np.median([row["path_relative_l2"] for row in rows])),
        "path_maximum": float(np.max([row["path_relative_l2"] for row in rows])),
    }, perf_counter() - started


def _repeat_worker(
    config_values: dict[str, object], state: np.ndarray
) -> dict[str, object]:
    config = KolmogorovReferenceConfig(**config_values).validated()
    result = KolmogorovReferenceStepper(config).advance_canonical(state)
    return {"state": result.state, "diagnostics": asdict(result.diagnostics)}


def _process_repeatability(
    config: KolmogorovReferenceConfig, state: np.ndarray
) -> tuple[dict[str, object], float]:
    started = perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        futures = [
            executor.submit(_repeat_worker, asdict(config), state) for _ in range(2)
        ]
        outputs = [future.result() for future in futures]
    first = outputs[0]["state"]
    second = outputs[1]["state"]
    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
        raise TypeError("repeatability worker did not return arrays")
    denominator = max(float(np.sqrt(np.mean(first**2))), np.finfo(np.float64).eps)
    scaled_rms = float(np.sqrt(np.mean((first - second) ** 2)) / denominator)
    accounting_equal = outputs[0]["diagnostics"] == outputs[1]["diagnostics"]
    result = {
        "bitwise_equal": bool(np.array_equal(first, second)),
        "scaled_rms": scaled_rms,
        "accounting_equal": accounting_equal,
        "first_diagnostics": outputs[0]["diagnostics"],
        "second_diagnostics": outputs[1]["diagnostics"],
    }
    result["pass"] = scaled_rms <= 1.0e-13 and accounting_equal
    return result, perf_counter() - started


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    sources = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered source is missing: {relative}")
        sources[relative] = _sha256(path)
    return sources


def _verify_source_binding(source_commit: str, *, quick: bool) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", *SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if quick:
        return {
            "mode": "quick_allows_uncommitted_sources",
            "head": head,
            "requested_source_commit": source_commit,
            "source_status": status.splitlines(),
        }
    if source_commit != head:
        raise ValueError(
            f"--source-commit must equal current HEAD {head}, got {source_commit}"
        )
    if status:
        raise RuntimeError("registered source paths differ from HEAD:\n" + status)
    return {
        "mode": "exact_clean_head",
        "head": head,
        "requested_source_commit": source_commit,
        "source_status": [],
    }


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _write_packet(
    output_dir: Path,
    result: dict[str, object],
    source_commit: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": result["run_id"],
        "source_commit": source_commit,
        "artifacts": {"result.json": _sha256(result_path)},
        "sources": source_hashes,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small synthetic plumbing check, not the registered Q1 result",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _ensure_empty_output_dir(args.output_dir)
    source_binding_start = _verify_source_binding(args.source_commit, quick=args.quick)
    source_hashes_start = _source_hashes()
    base_config, settings = _quick_contract() if args.quick else _full_contract()
    base_config.validated()
    total_started = perf_counter()

    base_stepper = KolmogorovReferenceStepper(base_config)
    clean_states, stationarity, stationarity_seconds = _burnin_and_stationarity(
        base_stepper, settings
    )
    inputs = _calibration_inputs(
        base_stepper, clean_states, settings.perturbation_bands
    )
    time_refinement, fine_rollouts, time_seconds = _time_refinement(
        base_config, settings, inputs
    )
    spatial_context, spatial_seconds = _spatial_context(
        base_config, settings, inputs, fine_rollouts
    )
    repeatability, repeatability_seconds = _process_repeatability(
        base_config, clean_states[0]
    )
    fixed_grid_pass = (
        bool(stationarity["pass"])
        and bool(time_refinement["summary"]["pass"])  # type: ignore[index]
        and bool(repeatability["pass"])
    )
    quick_plumbing_pass = (
        bool(time_refinement["summary"]["all_finite"])  # type: ignore[index]
        and bool(spatial_context["all_finite"])
        and bool(repeatability["pass"])
    )
    if args.quick:
        classification = (
            "quick_plumbing_pass" if quick_plumbing_pass else "quick_plumbing_fail"
        )
    elif fixed_grid_pass:
        classification = (
            "qualified_fixed_grid_after_burnin_extension"
            if bool(stationarity["extension_used"])
            else "qualified_fixed_grid"
        )
    else:
        classification = "failed_fixed_grid_qualification"

    result: dict[str, object] = {
        "run_id": f"{RUN_ID}-QUICK" if args.quick else RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_quick" if args.quick else "registered_solver_only_q1",
        "classification": classification,
        "fixed_grid_pass": fixed_grid_pass,
        "spatial_status": spatial_context["status"],
        "source_commit": args.source_commit,
        "source_binding": {},
        "reference_config": asdict(base_config),
        "settings": asdict(settings),
        "population": {
            "initial_seeds": list(INITIAL_SEEDS),
            "perturbation_seeds": list(PERTURBATION_SEEDS),
            "perturbation_bands": [list(band) for band in settings.perturbation_bands],
            "calibration_inputs": _calibration_metadata(base_stepper, inputs),
        },
        "stationarity": stationarity,
        "time_refinement": time_refinement,
        "spatial_context": spatial_context,
        "process_repeatability": repeatability,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "multiprocessing_start_method": "spawn",
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "timing_seconds": {
            "stationarity": stationarity_seconds,
            "time_refinement": time_seconds,
            "spatial_context": spatial_seconds,
            "process_repeatability": repeatability_seconds,
            "total": perf_counter() - total_started,
        },
        "data_access": False,
        "checkpoint_access": False,
        "training": False,
        "remote_execution": False,
        "test_access": False,
    }
    source_binding_end = _verify_source_binding(args.source_commit, quick=args.quick)
    source_hashes_end = _source_hashes()
    if source_hashes_start != source_hashes_end:
        raise RuntimeError("registered source hashes changed during execution")
    result["source_binding"] = {
        "start": source_binding_start,
        "end": source_binding_end,
        "hashes_stable_during_execution": True,
    }
    manifest = _write_packet(
        args.output_dir,
        result,
        args.source_commit,
        source_hashes_end,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "classification": classification,
                "fixed_grid_pass": fixed_grid_pass,
                "spatial_status": spatial_context["status"],
                "total_seconds": result["timing_seconds"]["total"],  # type: ignore[index]
                "result_sha256": manifest["artifacts"]["result.json"],  # type: ignore[index]
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

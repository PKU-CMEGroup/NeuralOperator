"""Run the registered M1-Q1-R1 solver-only diagnostic stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no.run_m1_kolmogorov_q1_qualification import (
    INITIAL_SEEDS,
    PERTURBATION_BANDS,
    PERTURBATION_SEEDS,
    _band_perturbation,
    _initial_state,
    _process_repeatability,
    _relative_l2,
    _relative_mean_change,
    _sha256,
    _spearman_time,
    _spectrum_total_variation,
    _state_metadata,
)
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
    resize_dealiased_vorticity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIX_RUN_ID = "M1-KF-Q1-R1-MIX-20260826A"
SPATIAL_RUN_ID = "M1-KF-Q1-R1-SPAT-20260826A"
PARENT_RUN_ID = "M1-KF-Q1-20260826A"
PARENT_SOURCE_COMMIT = "d396c45f2acd0bbea971b3285b98f406bcea4e74"
PARENT_RESULT_SHA256 = (
    "9ffc7a23c62704dad04af03996336ebff793d1cb3d8fb25da6f6f01176018002"
)
PARENT_MANIFEST_SHA256 = (
    "d1ef9b4e77cfe9866ab4beee82129972b2f1f6cd6ae3ee6ca8ba93a9653317ae"
)
PARENT_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts/time_dependent_no/m1_kolmogorov_q1_20260826a"
)
SOURCE_PATHS = (
    "utility/time_dependent_no/kolmogorov_reference.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_qualification.py",
    "scripts/time_dependent_no/run_m1_kolmogorov_q1_r1_diagnostic.py",
    "tests/time_dependent_no/test_kolmogorov_reference.py",
    "tests/time_dependent_no/test_kolmogorov_q1_r1_diagnostic.py",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_PREREGISTRATION.md",
    "docs/time_dependent_no/M1_KOLMOGOROV_INFORMATION_COMPARISON_TRACKER.md",
)


@dataclass(frozen=True)
class R1Settings:
    """Frozen execution sizes and diagnostic gates for one R1 stage."""

    total_calls: int
    capture_calls: tuple[int, ...]
    burnin_candidates: tuple[int, ...]
    observation_calls: int
    block_calls: int
    half_change_maximum: float
    drift_spearman_minimum: float
    split_rhat_maximum: float
    pooled_ess_minimum: float
    reproduction_burnin: int
    perturbation_bands: tuple[tuple[int, int], ...]
    spatial_resolutions: tuple[int, int, int]
    spatial_horizon: int
    spatial_dt_max: float
    spatial_h1_median_maximum: float
    spatial_h1_maximum_maximum: float
    spatial_horizon_median_maximum: float
    spatial_horizon_maximum_maximum: float
    spatial_contraction_maximum: float
    initial_rms: float
    workers: int
    progress_every: int

    def validated(self) -> R1Settings:
        if self.total_calls < 4:
            raise ValueError("total_calls must be at least four")
        if not self.capture_calls or self.capture_calls[0] != 0:
            raise ValueError("capture_calls must start at zero")
        if tuple(sorted(set(self.capture_calls))) != self.capture_calls:
            raise ValueError("capture_calls must be sorted and unique")
        if self.capture_calls[-1] != self.total_calls:
            raise ValueError("capture_calls must include total_calls")
        if self.reproduction_burnin not in self.capture_calls:
            raise ValueError("reproduction_burnin must be captured")
        if self.observation_calls < 4 or self.observation_calls % 2:
            raise ValueError("observation_calls must be an even integer at least four")
        if self.block_calls < 2 or self.observation_calls % self.block_calls:
            raise ValueError("block_calls must divide observation_calls")
        if any(
            burnin < 0 or burnin + self.observation_calls > self.total_calls
            for burnin in self.burnin_candidates
        ):
            raise ValueError("every burn-in window must fit inside total_calls")
        if len(self.perturbation_bands) != 3:
            raise ValueError("exactly three perturbation bands are required")
        if tuple(sorted(self.spatial_resolutions)) != self.spatial_resolutions:
            raise ValueError("spatial resolutions must be strictly increasing")
        if len(set(self.spatial_resolutions)) != 3:
            raise ValueError("spatial resolutions must be unique")
        if self.spatial_horizon < 1:
            raise ValueError("spatial_horizon must be positive")
        if self.workers < 1 or self.workers > 4:
            raise ValueError("workers must lie in [1, 4]")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        return self


def _full_contract() -> tuple[KolmogorovReferenceConfig, R1Settings]:
    return KolmogorovReferenceConfig(), R1Settings(
        total_calls=2048,
        capture_calls=(0, 256, 512, 768, 1024, 1280, 1536, 2048),
        burnin_candidates=(512, 768, 1024, 1280, 1536),
        observation_calls=512,
        block_calls=128,
        half_change_maximum=0.10,
        drift_spearman_minimum=0.5,
        split_rhat_maximum=1.05,
        pooled_ess_minimum=100.0,
        reproduction_burnin=512,
        perturbation_bands=PERTURBATION_BANDS,
        spatial_resolutions=(64, 128, 256),
        spatial_horizon=16,
        spatial_dt_max=0.0005,
        spatial_h1_median_maximum=0.0125,
        spatial_h1_maximum_maximum=0.025,
        spatial_horizon_median_maximum=0.05,
        spatial_horizon_maximum_maximum=0.10,
        spatial_contraction_maximum=0.5,
        initial_rms=4.0,
        workers=4,
        progress_every=256,
    ).validated()


def _quick_contract() -> tuple[KolmogorovReferenceConfig, R1Settings]:
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
        R1Settings(
            total_calls=32,
            capture_calls=(0, 8, 16, 24, 32),
            burnin_candidates=(8, 16),
            observation_calls=16,
            block_calls=4,
            half_change_maximum=0.10,
            drift_spearman_minimum=0.5,
            split_rhat_maximum=1.05,
            pooled_ess_minimum=4.0,
            reproduction_burnin=8,
            perturbation_bands=((1, 2), (3, 4), (5, 8)),
            spatial_resolutions=(18, 36, 72),
            spatial_horizon=2,
            spatial_dt_max=0.0005,
            spatial_h1_median_maximum=0.0125,
            spatial_h1_maximum_maximum=0.025,
            spatial_horizon_median_maximum=0.05,
            spatial_horizon_maximum_maximum=0.10,
            spatial_contraction_maximum=0.5,
            initial_rms=1.0,
            workers=2,
            progress_every=8,
        ).validated(),
    )


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Return the biased normalized autocorrelation of one finite series."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 4:
        raise ValueError("values must be a one-dimensional series of length >= 4")
    if not np.all(np.isfinite(series)):
        raise ValueError("values must be finite")
    centered = series - np.mean(series)
    variance_sum = float(np.dot(centered, centered))
    if variance_sum <= np.finfo(np.float64).tiny:
        result = np.zeros(series.size, dtype=np.float64)
        result[0] = 1.0
        return result
    transform_size = 1 << (2 * series.size - 1).bit_length()
    transformed = np.fft.rfft(centered, n=transform_size)
    covariance = np.fft.irfft(transformed * np.conjugate(transformed))[: series.size]
    return np.asarray(covariance / covariance[0], dtype=np.float64)


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate IAT with Geyer's initial-positive monotone pair sequence."""

    correlation = autocorrelation_fft(values)
    pair_sums: list[float] = []
    for lag in range(1, correlation.size - 1, 2):
        pair_sum = float(correlation[lag] + correlation[lag + 1])
        if pair_sum <= 0.0:
            break
        if pair_sums:
            pair_sum = min(pair_sum, pair_sums[-1])
        pair_sums.append(pair_sum)
    return max(1.0, float(1.0 + 2.0 * sum(pair_sums)))


def split_rhat(chains: np.ndarray) -> float:
    """Return classical split-R-hat for equal-length finite chains."""

    values = np.asarray(chains, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("chains must have shape [at least 2, at least 4]")
    if values.shape[1] % 2:
        raise ValueError("chain length must be even")
    if not np.all(np.isfinite(values)):
        raise ValueError("chains must be finite")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(half * np.var(np.mean(split, axis=1), ddof=1))
    if within <= np.finfo(np.float64).tiny:
        return 1.0 if between <= np.finfo(np.float64).tiny else float("inf")
    variance = ((half - 1.0) / half) * within + between / half
    return max(1.0, float(np.sqrt(max(variance, 0.0) / within)))


def pooled_effective_sample_size(chains: np.ndarray) -> tuple[float, list[float]]:
    """Return summed per-chain ESS and the corresponding IAT values."""

    values = np.asarray(chains, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("chains must be a two-dimensional nonempty array")
    taus = [integrated_autocorrelation_time(chain) for chain in values]
    effective = min(
        float(values.size),
        float(sum(values.shape[1] / tau for tau in taus)),
    )
    return effective, taus


def _progress(event: str, **payload: object) -> None:
    print(
        json.dumps({"event": event, **payload}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"registered source is missing: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


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


def _verify_parent_packet() -> tuple[dict[str, object], dict[str, object]]:
    result_path = PARENT_ARTIFACT_DIR / "result.json"
    manifest_path = PARENT_ARTIFACT_DIR / "artifact_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("the immutable parent Q1 packet is missing")
    if _sha256(result_path) != PARENT_RESULT_SHA256:
        raise RuntimeError("parent Q1 result hash mismatch")
    if _sha256(manifest_path) != PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent Q1 artifact-manifest hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != PARENT_RUN_ID:
        raise RuntimeError("parent Q1 run identity mismatch")
    if result.get("classification") != "failed_fixed_grid_qualification":
        raise RuntimeError("parent Q1 classification mismatch")
    if manifest.get("source_commit") != PARENT_SOURCE_COMMIT:
        raise RuntimeError("parent Q1 source commit mismatch")
    if manifest.get("artifacts", {}).get("result.json") != PARENT_RESULT_SHA256:
        raise RuntimeError("parent Q1 manifest does not bind the expected result")
    source_rows = []
    for relative, expected in manifest.get("sources", {}).items():
        payload = subprocess.run(
            ["git", "show", f"{PARENT_SOURCE_COMMIT}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"parent source mismatch at {relative}")
        source_rows.append({"path": relative, "sha256": actual})
    binding = {
        "run_id": PARENT_RUN_ID,
        "source_commit": PARENT_SOURCE_COMMIT,
        "result_sha256": PARENT_RESULT_SHA256,
        "artifact_manifest_sha256": PARENT_MANIFEST_SHA256,
        "source_count": len(source_rows),
        "all_commit_sources_match": True,
    }
    return result, binding


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _mixing_worker(
    config_values: dict[str, object],
    settings: R1Settings,
    seed: int,
) -> dict[str, object]:
    stepper = KolmogorovReferenceStepper(
        KolmogorovReferenceConfig(**config_values).validated()
    )
    state = _initial_state(stepper, seed, settings.initial_rms)
    calls = settings.total_calls
    energy = np.empty(calls + 1, dtype=np.float64)
    enstrophy = np.empty(calls + 1, dtype=np.float64)
    palinstrophy = np.empty(calls + 1, dtype=np.float64)
    _, first_spectrum = stepper.kinetic_energy_spectrum_canonical(state)
    spectra = np.empty((calls + 1, first_spectrum.size), dtype=np.float64)
    substeps = np.empty(calls, dtype=np.int32)
    minimum_substep = np.empty(calls, dtype=np.float64)
    maximum_substep = np.empty(calls, dtype=np.float64)
    captures: dict[int, dict[str, float | str]] = {}

    def record(index: int, value: np.ndarray) -> None:
        diagnostics = stepper.diagnostics_canonical(value)
        _, spectrum = stepper.kinetic_energy_spectrum_canonical(value)
        energy[index] = diagnostics.kinetic_energy
        enstrophy[index] = diagnostics.enstrophy
        palinstrophy[index] = diagnostics.palinstrophy
        spectra[index] = spectrum
        if index in settings.capture_calls:
            captures[index] = _state_metadata(stepper, value)

    _progress("mixing_chain_start", seed=seed, calls=calls)
    record(0, state)
    for call in range(1, calls + 1):
        advanced = stepper.advance_canonical(state)
        state = advanced.state
        substeps[call - 1] = advanced.diagnostics.substeps
        minimum_substep[call - 1] = advanced.diagnostics.minimum_substep
        maximum_substep[call - 1] = advanced.diagnostics.maximum_substep
        record(call, state)
        if call % settings.progress_every == 0 or call == calls:
            _progress("mixing_chain_progress", seed=seed, call=call, total=calls)
    _progress("mixing_chain_complete", seed=seed)
    return {
        "seed": seed,
        "energy": energy,
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "spectra": spectra,
        "substeps": substeps,
        "minimum_substep": minimum_substep,
        "maximum_substep": maximum_substep,
        "captures": captures,
    }


def _shared_drift_summary(values: list[float], threshold: float) -> dict[str, object]:
    positive = sum(value >= threshold for value in values)
    negative = sum(value <= -threshold for value in values)
    return {
        "positive_count": positive,
        "negative_count": negative,
        "pass": max(positive, negative) < len(values),
    }


def _mixing_metric_summary(
    values: np.ndarray,
    seeds: list[int],
    settings: R1Settings,
) -> dict[str, object]:
    rows = []
    correlations = []
    half = values.shape[1] // 2
    for seed, chain in zip(seeds, values, strict=True):
        correlation = _spearman_time(chain)
        correlations.append(correlation)
        blocks = chain.reshape(-1, settings.block_calls).mean(axis=1)
        denominator = max(float(np.mean(np.abs(chain))), np.finfo(np.float64).eps)
        rows.append(
            {
                "seed": seed,
                "half_change": _relative_mean_change(chain[:half], chain[half:]),
                "spearman": correlation,
                "block_means": [float(value) for value in blocks],
                "block_relative_range": float(
                    (np.max(blocks) - np.min(blocks)) / denominator
                ),
            }
        )
    effective, taus = pooled_effective_sample_size(values)
    for row, tau in zip(rows, taus, strict=True):
        row["integrated_autocorrelation_time"] = tau
        row["effective_sample_size"] = values.shape[1] / tau
    half_change_pass = all(
        float(row["half_change"]) <= settings.half_change_maximum for row in rows
    )
    drift = _shared_drift_summary(correlations, settings.drift_spearman_minimum)
    rhat = split_rhat(values)
    return {
        "rows": rows,
        "split_rhat": rhat,
        "pooled_effective_sample_size": effective,
        "half_change_pass": half_change_pass,
        "shared_drift": drift,
        "rhat_pass": rhat <= settings.split_rhat_maximum,
        "ess_pass": effective >= settings.pooled_ess_minimum,
        "pass": half_change_pass
        and bool(drift["pass"])
        and rhat <= settings.split_rhat_maximum
        and effective >= settings.pooled_ess_minimum,
    }


def _mixing_summary(
    outputs: list[dict[str, object]],
    settings: R1Settings,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    ordered = sorted(outputs, key=lambda output: int(output["seed"]))
    seeds = [int(output["seed"]) for output in ordered]
    series = {
        name: np.stack([np.asarray(output[name]) for output in ordered], axis=0)
        for name in (
            "energy",
            "enstrophy",
            "palinstrophy",
            "spectra",
            "substeps",
            "minimum_substep",
            "maximum_substep",
        )
    }
    candidates = []
    selected_burnin = None
    for burnin in settings.burnin_candidates:
        start = burnin + 1
        stop = start + settings.observation_calls
        energy_summary = _mixing_metric_summary(
            series["energy"][:, start:stop], seeds, settings
        )
        enstrophy_summary = _mixing_metric_summary(
            series["enstrophy"][:, start:stop], seeds, settings
        )
        passed = bool(energy_summary["pass"]) and bool(enstrophy_summary["pass"])
        candidates.append(
            {
                "burnin_calls": burnin,
                "observation_start_call": start,
                "observation_end_call": stop - 1,
                "energy": energy_summary,
                "enstrophy": enstrophy_summary,
                "pass": passed,
            }
        )
        if selected_burnin is None and passed:
            selected_burnin = burnin
    arrays = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "calls": np.arange(settings.total_calls + 1, dtype=np.int64),
        **series,
    }
    summary = {
        "candidates": candidates,
        "selected_burnin_calls": selected_burnin,
        "candidate_found": selected_burnin is not None,
        "capture_rows": [
            {
                "seed": int(output["seed"]),
                "states": [
                    {"call": int(call), **metadata}
                    for call, metadata in sorted(
                        output["captures"].items()  # type: ignore[union-attr]
                    )
                ],
            }
            for output in ordered
        ],
        "all_finite": all(np.all(np.isfinite(value)) for value in series.values()),
        "substep_count_minimum": int(np.min(series["substeps"])),
        "substep_count_maximum": int(np.max(series["substeps"])),
    }
    return summary, arrays


def _verify_mixing_parent_replay(
    parent: dict[str, object], summary: dict[str, object]
) -> dict[str, object]:
    stationarity = parent["stationarity"]
    if not isinstance(stationarity, dict):
        raise TypeError("parent stationarity payload is malformed")
    expected_initial = {
        int(row["seed"]): str(row["sha256"]) for row in stationarity["initial_states"]
    }
    expected_burnin = {
        int(row["seed"]): str(row["sha256"])
        for row in stationarity["chosen_post_burnin_states"]
    }
    if int(stationarity["chosen_burnin_calls"]) != 512:
        raise RuntimeError("parent did not select the registered 512-call state")
    rows = []
    for chain in summary["capture_rows"]:
        seed = int(chain["seed"])
        by_call = {int(row["call"]): row for row in chain["states"]}
        initial_match = by_call[0]["sha256"] == expected_initial[seed]
        burnin_match = by_call[512]["sha256"] == expected_burnin[seed]
        rows.append(
            {
                "seed": seed,
                "initial_hash_match": initial_match,
                "call_512_hash_match": burnin_match,
            }
        )
    if not all(
        row["initial_hash_match"] and row["call_512_hash_match"] for row in rows
    ):
        raise RuntimeError("R1 mixing state replay does not match parent Q1")
    return {"rows": rows, "pass": True}


def _relative_scalar(value: float, reference: float) -> float:
    denominator = max(abs(reference), np.finfo(np.float64).eps)
    return float(abs(value - reference) / denominator)


def _normalized_state_spectrum(
    stepper: KolmogorovReferenceStepper, state: np.ndarray
) -> np.ndarray:
    _, spectrum = stepper.kinetic_energy_spectrum_canonical(state)
    total = max(float(np.sum(spectrum)), np.finfo(np.float64).eps)
    return spectrum / total


def _spatial_case(
    base_config: KolmogorovReferenceConfig,
    settings: R1Settings,
    case: str,
    family: str,
    base_state: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    steppers = {}
    rollouts = {}
    status_rows = []
    for resolution in settings.spatial_resolutions:
        config = replace(
            base_config,
            resolution=resolution,
            dt_max=settings.spatial_dt_max,
        )
        stepper = KolmogorovReferenceStepper(config)
        state = (
            base_state
            if resolution == base_config.resolution
            else resize_dealiased_vorticity(base_state, resolution)
        )
        _progress(
            "spatial_case_resolution_start",
            case=case,
            resolution=resolution,
        )
        rollout, records = stepper.rollout_canonical(state, settings.spatial_horizon)
        projection = max(stepper.projection_relative_l2(value) for value in rollout)
        steppers[resolution] = stepper
        rollouts[resolution] = rollout
        status_rows.append(
            {
                "resolution": resolution,
                "finite": bool(np.all(np.isfinite(rollout))),
                "maximum_projection_relative_l2": projection,
                "substep_count_minimum": min(record.substeps for record in records),
                "substep_count_maximum": max(record.substeps for record in records),
            }
        )
        _progress(
            "spatial_case_resolution_complete",
            case=case,
            resolution=resolution,
        )

    rows = []
    for low_resolution, high_resolution in zip(
        settings.spatial_resolutions[:-1],
        settings.spatial_resolutions[1:],
        strict=True,
    ):
        low_stepper = steppers[low_resolution]
        high_stepper = steppers[high_resolution]
        low_rollout = rollouts[low_resolution]
        high_rollout = rollouts[high_resolution]
        pair = f"{low_resolution}_to_{high_resolution}"
        for horizon in range(1, settings.spatial_horizon + 1):
            low_state = low_rollout[horizon]
            high_state = high_rollout[horizon]
            restricted_high = resize_dealiased_vorticity(high_state, low_resolution)
            low_diagnostics = low_stepper.diagnostics_canonical(low_state)
            high_diagnostics = high_stepper.diagnostics_canonical(high_state)
            rows.append(
                {
                    "case": case,
                    "family": family,
                    "pair": pair,
                    "low_resolution": low_resolution,
                    "high_resolution": high_resolution,
                    "horizon": horizon,
                    "state_relative_l2": _relative_l2(low_state, restricted_high),
                    "energy_relative_difference": _relative_scalar(
                        low_diagnostics.kinetic_energy,
                        high_diagnostics.kinetic_energy,
                    ),
                    "enstrophy_relative_difference": _relative_scalar(
                        low_diagnostics.enstrophy,
                        high_diagnostics.enstrophy,
                    ),
                    "palinstrophy_relative_difference": _relative_scalar(
                        low_diagnostics.palinstrophy,
                        high_diagnostics.palinstrophy,
                    ),
                    "spectrum_total_variation": _spectrum_total_variation(
                        _normalized_state_spectrum(low_stepper, low_state),
                        _normalized_state_spectrum(high_stepper, high_state),
                    ),
                    "finite": bool(
                        np.all(np.isfinite(low_state))
                        and np.all(np.isfinite(high_state))
                    ),
                }
            )
    return rows, status_rows


def _advance_to_burnin(
    stepper: KolmogorovReferenceStepper,
    seed: int,
    settings: R1Settings,
) -> tuple[np.ndarray, np.ndarray]:
    initial = _initial_state(stepper, seed, settings.initial_rms)
    state = initial
    for call in range(1, settings.reproduction_burnin + 1):
        state = stepper.advance_canonical(state).state
        if call % settings.progress_every == 0 or call == settings.reproduction_burnin:
            _progress(
                "spatial_burnin_progress",
                seed=seed,
                call=call,
                total=settings.reproduction_burnin,
            )
    return initial, state


def _spatial_seed_worker(
    config_values: dict[str, object],
    settings: R1Settings,
    index: int,
) -> dict[str, object]:
    base_config = KolmogorovReferenceConfig(**config_values).validated()
    stepper = KolmogorovReferenceStepper(base_config)
    seed = INITIAL_SEEDS[index]
    _progress("spatial_seed_start", seed=seed)
    initial, clean = _advance_to_burnin(stepper, seed, settings)
    displaced = _band_perturbation(
        stepper,
        clean,
        PERTURBATION_SEEDS[index],
        settings.perturbation_bands[index],
    )
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    inputs = (
        (f"clean_{index}", "clean", clean),
        (f"displaced_{index}", "displaced", displaced),
    )
    rows = []
    status_rows = []
    input_rows = []
    for case, family, state in inputs:
        case_rows, case_status = _spatial_case(
            base_config, settings, case, family, state
        )
        rows.extend(case_rows)
        status_rows.extend(
            {"case": case, "family": family, **row} for row in case_status
        )
        metadata: dict[str, object] = {
            "case": case,
            "family": family,
            **_state_metadata(stepper, state),
        }
        if family == "displaced":
            metadata.update(
                {
                    "seed": PERTURBATION_SEEDS[index],
                    "band": list(settings.perturbation_bands[index]),
                    "relative_perturbation_rms": float(
                        np.sqrt(np.mean((displaced - clean) ** 2)) / clean_rms
                    ),
                }
            )
        else:
            metadata["seed"] = seed
        input_rows.append(metadata)
    _progress("spatial_seed_complete", seed=seed)
    return {
        "seed": seed,
        "initial": _state_metadata(stepper, initial),
        "post_burnin": _state_metadata(stepper, clean),
        "inputs": input_rows,
        "rows": rows,
        "status_rows": status_rows,
        "repeat_state": clean if index == 0 else None,
    }


def _endpoint_summary(
    rows: list[dict[str, object]], pair: str, horizon: int
) -> dict[str, object]:
    selected = [
        row for row in rows if row["pair"] == pair and int(row["horizon"]) == horizon
    ]
    values = np.asarray(
        [row["state_relative_l2"] for row in selected], dtype=np.float64
    )
    families = {}
    for family in ("clean", "displaced"):
        family_values = np.asarray(
            [row["state_relative_l2"] for row in selected if row["family"] == family],
            dtype=np.float64,
        )
        families[family] = {
            "median": float(np.median(family_values)),
            "maximum": float(np.max(family_values)),
        }
    return {
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "families": families,
    }


def summarize_spatial_rows(
    rows: list[dict[str, object]], settings: R1Settings
) -> dict[str, object]:
    """Aggregate adjacent-resolution state errors and apply the R1 screen."""

    low, candidate, high = settings.spatial_resolutions
    lower_pair = f"{low}_to_{candidate}"
    upper_pair = f"{candidate}_to_{high}"
    endpoints = {
        lower_pair: {
            "h1": _endpoint_summary(rows, lower_pair, 1),
            "horizon": _endpoint_summary(rows, lower_pair, settings.spatial_horizon),
        },
        upper_pair: {
            "h1": _endpoint_summary(rows, upper_pair, 1),
            "horizon": _endpoint_summary(rows, upper_pair, settings.spatial_horizon),
        },
    }
    upper_h1 = endpoints[upper_pair]["h1"]
    upper_horizon = endpoints[upper_pair]["horizon"]
    lower_h1 = endpoints[lower_pair]["h1"]
    lower_horizon = endpoints[lower_pair]["horizon"]
    denominator_floor = np.finfo(np.float64).eps
    ratios = {
        "h1_median": float(
            upper_h1["median"] / max(lower_h1["median"], denominator_floor)
        ),
        "h1_maximum": float(
            upper_h1["maximum"] / max(lower_h1["maximum"], denominator_floor)
        ),
        "horizon_median": float(
            upper_horizon["median"] / max(lower_horizon["median"], denominator_floor)
        ),
        "horizon_maximum": float(
            upper_horizon["maximum"] / max(lower_horizon["maximum"], denominator_floor)
        ),
    }
    absolute_pass = (
        float(upper_h1["median"]) <= settings.spatial_h1_median_maximum
        and float(upper_h1["maximum"]) <= settings.spatial_h1_maximum_maximum
        and float(upper_horizon["median"]) <= settings.spatial_horizon_median_maximum
        and float(upper_horizon["maximum"]) <= settings.spatial_horizon_maximum_maximum
    )
    contraction_pass = all(
        value <= settings.spatial_contraction_maximum for value in ratios.values()
    )
    structure_maxima = {
        metric: float(max(float(row[metric]) for row in rows))
        for metric in (
            "energy_relative_difference",
            "enstrophy_relative_difference",
            "palinstrophy_relative_difference",
            "spectrum_total_variation",
        )
    }
    return {
        "endpoints": endpoints,
        "upper_to_lower_error_ratios": ratios,
        "absolute_pass": absolute_pass,
        "contraction_pass": contraction_pass,
        "structure_maxima_all_calls": structure_maxima,
        "all_finite": all(bool(row["finite"]) for row in rows),
        "screen_without_repeatability": absolute_pass and contraction_pass,
    }


def _verify_spatial_parent_replay(
    parent: dict[str, object], outputs: list[dict[str, object]]
) -> dict[str, object]:
    stationarity = parent["stationarity"]
    population = parent["population"]
    if not isinstance(stationarity, dict) or not isinstance(population, dict):
        raise TypeError("parent Q1 replay payload is malformed")
    expected_initial = {
        int(row["seed"]): str(row["sha256"]) for row in stationarity["initial_states"]
    }
    expected_burnin = {
        int(row["seed"]): str(row["sha256"])
        for row in stationarity["chosen_post_burnin_states"]
    }
    expected_inputs = {
        str(row["case"]): str(row["sha256"]) for row in population["calibration_inputs"]
    }
    rows = []
    for output in outputs:
        seed = int(output["seed"])
        initial_match = output["initial"]["sha256"] == expected_initial[seed]
        burnin_match = output["post_burnin"]["sha256"] == expected_burnin[seed]
        input_matches = {
            str(row["case"]): row["sha256"] == expected_inputs[str(row["case"])]
            for row in output["inputs"]
        }
        rows.append(
            {
                "seed": seed,
                "initial_hash_match": initial_match,
                "call_512_hash_match": burnin_match,
                "input_hash_matches": input_matches,
            }
        )
    if not all(
        row["initial_hash_match"]
        and row["call_512_hash_match"]
        and all(row["input_hash_matches"].values())
        for row in rows
    ):
        raise RuntimeError("R1 spatial inputs do not match parent Q1")
    return {"rows": rows, "pass": True}


def _run_mixing(
    base_config: KolmogorovReferenceConfig,
    settings: R1Settings,
) -> tuple[dict[str, object], dict[str, np.ndarray], float]:
    started = perf_counter()
    context = mp.get_context("spawn")
    outputs = []
    with ProcessPoolExecutor(
        max_workers=settings.workers, mp_context=context
    ) as executor:
        futures = [
            executor.submit(_mixing_worker, asdict(base_config), settings, seed)
            for seed in INITIAL_SEEDS
        ]
        for future in as_completed(futures):
            outputs.append(future.result())
    summary, arrays = _mixing_summary(outputs, settings)
    return summary, arrays, perf_counter() - started


def _run_spatial(
    base_config: KolmogorovReferenceConfig,
    settings: R1Settings,
) -> tuple[dict[str, object], list[dict[str, object]], float]:
    started = perf_counter()
    context = mp.get_context("spawn")
    outputs = []
    with ProcessPoolExecutor(
        max_workers=min(settings.workers, 3), mp_context=context
    ) as executor:
        futures = [
            executor.submit(_spatial_seed_worker, asdict(base_config), settings, index)
            for index in range(3)
        ]
        for future in as_completed(futures):
            outputs.append(future.result())
    outputs.sort(key=lambda output: int(output["seed"]))
    rows = [row for output in outputs for row in output["rows"]]
    status_rows = [row for output in outputs for row in output["status_rows"]]
    repeat_state = outputs[0]["repeat_state"]
    if not isinstance(repeat_state, np.ndarray):
        raise TypeError("first spatial worker did not return its clean state")
    repeat_resolution = settings.spatial_resolutions[-1]
    repeat_config = replace(
        base_config,
        resolution=repeat_resolution,
        dt_max=settings.spatial_dt_max,
    )
    repeatability, _ = _process_repeatability(
        repeat_config,
        resize_dealiased_vorticity(repeat_state, repeat_resolution),
    )
    summary = summarize_spatial_rows(rows, settings)
    maximum_projection = max(
        float(row["maximum_projection_relative_l2"]) for row in status_rows
    )
    closure_pass = (
        all(bool(row["finite"]) for row in status_rows)
        and maximum_projection <= base_config.canonical_tolerance
    )
    summary.update(
        {
            "status_rows": status_rows,
            "input_rows": [row for output in outputs for row in output["inputs"]],
            "initial_rows": [
                {"seed": output["seed"], **output["initial"]} for output in outputs
            ],
            "post_burnin_rows": [
                {"seed": output["seed"], **output["post_burnin"]} for output in outputs
            ],
            "maximum_projection_relative_l2": maximum_projection,
            "closure_pass": closure_pass,
            "repeatability": repeatability,
        }
    )
    summary["screen_pass"] = (
        bool(summary["screen_without_repeatability"])
        and bool(summary["all_finite"])
        and closure_pass
        and bool(repeatability["pass"])
    )
    return summary, rows, perf_counter() - started


def _write_packet(
    output_dir: Path,
    result: dict[str, object],
    arrays: dict[str, np.ndarray] | None,
    source_hashes: dict[str, str],
    source_commit: str,
    parent_binding: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if arrays is not None:
        series_path = output_dir / "series.npz"
        np.savez_compressed(series_path, **arrays)
        artifacts[series_path.name] = _sha256(series_path)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts[result_path.name] = _sha256(result_path)
    manifest = {
        "run_id": result["run_id"],
        "source_commit": source_commit,
        "parent": parent_binding,
        "artifacts": artifacts,
        "sources": source_hashes,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mixing", "spatial"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run reduced synthetic plumbing, not a registered R1 result",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _ensure_empty_output_dir(args.output_dir)
    parent, parent_binding = _verify_parent_packet()
    binding_start = _verify_source_binding(args.source_commit, quick=args.quick)
    hashes_start = _source_hashes()
    base_config, settings = _quick_contract() if args.quick else _full_contract()
    base_config.validated()
    if not args.quick and asdict(base_config) != parent["reference_config"]:
        raise RuntimeError("R1 base configuration differs from parent Q1")
    total_started = perf_counter()

    arrays = None
    if args.stage == "mixing":
        diagnostic, arrays, stage_seconds = _run_mixing(base_config, settings)
        parent_replay = (
            {"status": "not_applicable_in_quick_mode"}
            if args.quick
            else _verify_mixing_parent_replay(parent, diagnostic)
        )
        if args.quick:
            classification = (
                "quick_plumbing_pass"
                if bool(diagnostic["all_finite"])
                else "quick_plumbing_fail"
            )
        else:
            classification = (
                "burnin_candidate_found"
                if bool(diagnostic["candidate_found"])
                else "no_burnin_candidate"
            )
        run_id = f"{MIX_RUN_ID}-QUICK" if args.quick else MIX_RUN_ID
        payload_key = "mixing_diagnostic"
    else:
        diagnostic, rows, stage_seconds = _run_spatial(base_config, settings)
        diagnostic["rows"] = rows
        parent_replay = (
            {"status": "not_applicable_in_quick_mode"}
            if args.quick
            else _verify_spatial_parent_replay(
                parent,
                [
                    {
                        "seed": row["seed"],
                        "initial": row,
                        "post_burnin": next(
                            item
                            for item in diagnostic["post_burnin_rows"]
                            if item["seed"] == row["seed"]
                        ),
                        "inputs": [
                            item
                            for item in diagnostic["input_rows"]
                            if item["case"].endswith(
                                str(INITIAL_SEEDS.index(int(row["seed"])))
                            )
                        ],
                    }
                    for row in diagnostic["initial_rows"]
                ],
            )
        )
        if args.quick:
            quick_plumbing_pass = (
                bool(diagnostic["all_finite"])
                and bool(diagnostic["closure_pass"])
                and bool(diagnostic["repeatability"]["pass"])
            )
            classification = (
                "quick_plumbing_pass" if quick_plumbing_pass else "quick_plumbing_fail"
            )
        else:
            classification = (
                "provisional_spatial_candidate"
                if bool(diagnostic["screen_pass"])
                else "spatial_screen_failed"
            )
        run_id = f"{SPATIAL_RUN_ID}-QUICK" if args.quick else SPATIAL_RUN_ID
        payload_key = "spatial_diagnostic"

    result: dict[str, object] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"synthetic_quick_{args.stage}"
        if args.quick
        else f"registered_solver_only_r1_{args.stage}",
        "stage": args.stage,
        "classification": classification,
        "source_commit": args.source_commit,
        "source_binding": {},
        "parent_binding": parent_binding,
        "parent_state_replay": parent_replay,
        "reference_config": asdict(base_config),
        "settings": asdict(settings),
        payload_key: diagnostic,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "multiprocessing_start_method": "spawn",
            "workers": settings.workers,
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
            "stage": stage_seconds,
            "total": perf_counter() - total_started,
        },
        "data_access": False,
        "checkpoint_access": False,
        "training": False,
        "remote_execution": False,
        "test_access": False,
        "full_state_trajectory_retained": False,
    }
    binding_end = _verify_source_binding(args.source_commit, quick=args.quick)
    hashes_end = _source_hashes()
    if hashes_start != hashes_end:
        raise RuntimeError("registered R1 source hashes changed during execution")
    result["source_binding"] = {
        "start": binding_start,
        "end": binding_end,
        "hashes_stable_during_execution": True,
    }
    manifest = _write_packet(
        args.output_dir,
        result,
        arrays,
        hashes_end,
        args.source_commit,
        parent_binding,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "classification": classification,
                "stage": args.stage,
                "total_seconds": result["timing_seconds"]["total"],  # type: ignore[index]
                "result_sha256": manifest["artifacts"]["result.json"],  # type: ignore[index]
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

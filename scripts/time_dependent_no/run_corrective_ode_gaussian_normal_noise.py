#!/usr/bin/env python3
"""Run and verify the source-bound B2-GN Gaussian normal-noise ODE study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    NonlinearStressConfig,
    ResidualMLP,
    _state_dict_digest,
    evaluate_forced_rollouts,
    evaluate_response_bank,
    flows,
    forcing_bank,
    heldout_phases,
    model_step,
    qualify_trusted_solver,
)
from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    verify_packet as verify_parent_packet,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    digest_array,
    git_state,
    runtime_environment,
    sha256_file,
    write_csv,
)

STUDY_SCHEMA = "corrective_ode_gaussian_normal_noise_v1"
CONFIG_SCHEMA = "corrective_ode_gaussian_normal_noise_config_v1"
PREDICTION_SCHEMA = "corrective_ode_gaussian_normal_noise_predictions_v1"
MANIFEST_SCHEMA = "corrective_ode_gaussian_normal_noise_manifest_v1"
CLOSEOUT_SCHEMA = "corrective_ode_gaussian_normal_noise_closeout_v1"
PROVENANCE_SCHEMA = "corrective_ode_gaussian_normal_noise_provenance_v1"
AUDIT_SCHEMA = "corrective_ode_gaussian_normal_audit_v1"

PARENT_PACKET_DIR = Path(
    "artifacts/time_dependent_no/corrective_ode_nonlinear_stress_20260830a"
)
PARENT_MANIFEST_SHA256 = (
    "8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40"
)

SOURCE_FILES = (
    "docs/time_dependent_no/B2_GN_GAUSSIAN_NORMAL_NOISE_PREREGISTRATION.md",
    "scripts/time_dependent_no/run_corrective_ode_gaussian_normal_noise.py",
    "tests/time_dependent_no/test_corrective_ode_gaussian_normal_noise.py",
    "scripts/time_dependent_no/run_corrective_ode_nonlinear_stress.py",
    "scripts/time_dependent_no/run_corrective_ode_study.py",
    "utility/time_dependent_no/pcno_artifacts.py",
)

TRAINING_NOISE_LAW = "input_normal_only_iid_gaussian"
FORCING_LAW = "post_map_paired_rademacher"
PAPER_ARM_COLORS = {
    "CLEAN": "#4C78A8",
    "RECOVERY": "#F58518",
    "DYN": "#E45756",
}
REGIME_LINESTYLES = {"C": "-", "M": "--", "E": ":"}
REGIME_MARKERS = {"C": "o", "M": "s", "E": "^"}


@dataclass(frozen=True)
class GaussianNormalNoiseConfig:
    """Frozen Gaussian extension around the qualified B2-NL system."""

    base: NonlinearStressConfig = field(default_factory=NonlinearStressConfig)
    train_noise_stds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04)
    clean_rows_per_update: int = 64
    noisy_rows_per_update: int = 64
    schedule_seed_offset: int = 30_000_019
    tail_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.999)
    profile_abs_radii: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)

    def validated(self) -> GaussianNormalNoiseConfig:
        self.base.validated()
        if self.clean_rows_per_update != 64 or self.noisy_rows_per_update != 64:
            raise ValueError("B2-GN requires exactly 64 clean and 64 noisy rows")
        if self.base.batch_size != 128:
            raise ValueError("B2-GN requires batch size 128")
        if not self.train_noise_stds:
            raise ValueError("train_noise_stds must be nonempty")
        if tuple(sorted(set(self.train_noise_stds))) != self.train_noise_stds:
            raise ValueError("train_noise_stds must be positive, unique, and sorted")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in self.train_noise_stds
        ):
            raise ValueError("training standard deviations must be finite and positive")
        if self.schedule_seed_offset < 0:
            raise ValueError("schedule_seed_offset must be nonnegative")
        if not self.tail_quantiles or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.tail_quantiles
        ):
            raise ValueError("tail quantiles must lie strictly between zero and one")
        if tuple(sorted(set(self.tail_quantiles))) != self.tail_quantiles:
            raise ValueError("tail quantiles must be unique and sorted")
        if self.profile_abs_radii != (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
            raise ValueError("B2-GN radial profile bank differs from preregistration")
        return self

    @property
    def expected_checkpoint_count(self) -> int:
        return len(self.base.seeds) * (1 + 4 * len(self.train_noise_stds))


def _config_payload(config: GaussianNormalNoiseConfig) -> dict[str, Any]:
    return {"schema": CONFIG_SCHEMA, "config": asdict(config)}


def _config_from_payload(value: Mapping[str, Any]) -> GaussianNormalNoiseConfig:
    if value.get("schema") != CONFIG_SCHEMA or not isinstance(
        value.get("config"), Mapping
    ):
        raise ValueError("invalid B2-GN configuration payload")
    record = dict(value["config"])
    base_record = record.pop("base", None)
    if not isinstance(base_record, Mapping):
        raise TypeError("B2-GN configuration lacks its base contract")
    tuple_fields = {
        "training_radii",
        "inner_query_radii",
        "outer_query_radii",
        "seeds",
        "noise_sigmas",
    }
    normalized_base = {
        key: tuple(item) if key in tuple_fields else item
        for key, item in base_record.items()
    }
    for key in ("train_noise_stds", "tail_quantiles", "profile_abs_radii"):
        if key in record:
            record[key] = tuple(record[key])
    return GaussianNormalNoiseConfig(
        base=NonlinearStressConfig(**normalized_base), **record
    ).validated()


def registered_predictions() -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "outcome_blind": True,
        "predictions": {
            "recovery_signature": (
                "At a fixed nonzero radius, larger Gaussian log density should "
                "correlate with lower RECOVERY clean-return error."
            ),
            "dynamics_signature": (
                "At a fixed nonzero radius, larger Gaussian log density should "
                "correlate with lower DYN trusted-flow defect while qualified "
                "models retain C/M/E response ordering."
            ),
            "fidelity_retention_tradeoff": (
                "Training-noise scale can trade clean or phase fidelity against normal retention."
            ),
            "mediator_before_outcome": (
                "A rollout interpretation requires its fixed-bank mediator to move first."
            ),
        },
        "all_scales_reported": True,
        "long_rollout_scale_selection": False,
        "ode_to_pde_transfer_claim": False,
    }


def build_training_schedule(
    config: GaussianNormalNoiseConfig, seed: int
) -> dict[str, np.ndarray]:
    """Build the paired, fresh-per-update phase and Gaussian schedule."""

    config.validated()
    if seed not in config.base.seeds:
        raise ValueError("schedule seed is not registered")
    rng = np.random.default_rng(seed + config.schedule_seed_offset)
    shape = (config.base.training_steps, config.clean_rows_per_update)
    clean_indices = rng.integers(
        0, config.base.train_phases, size=shape, dtype=np.int64
    )
    noisy_indices = rng.integers(
        0, config.base.train_phases, size=shape, dtype=np.int64
    )
    standard_normal = rng.standard_normal(shape).astype(np.float64, copy=False)
    if not np.isfinite(standard_normal).all():
        raise FloatingPointError("Gaussian schedule contains non-finite draws")
    return {
        "clean_phase_indices": clean_indices,
        "noisy_phase_indices": noisy_indices,
        "standard_normal": standard_normal,
    }


def schedule_record(schedule: Mapping[str, np.ndarray]) -> dict[str, Any]:
    expected = {"clean_phase_indices", "noisy_phase_indices", "standard_normal"}
    if set(schedule) != expected:
        raise ValueError("training schedule has an unexpected field set")
    clean = np.asarray(schedule["clean_phase_indices"], dtype=np.int64)
    noisy = np.asarray(schedule["noisy_phase_indices"], dtype=np.int64)
    normal = np.asarray(schedule["standard_normal"], dtype=np.float64)
    if clean.shape != noisy.shape or clean.shape != normal.shape:
        raise ValueError("training schedule arrays are not aligned")
    return {
        "shape": list(clean.shape),
        "clean_phase_indices_digest": digest_array(clean),
        "noisy_phase_indices_digest": digest_array(noisy),
        "standard_normal_digest": digest_array(normal),
    }


def _tail_probability(threshold: float, sigma_train: float) -> float:
    return math.erfc(threshold / (math.sqrt(2.0) * sigma_train))


def summarize_gaussian_tail(
    config: GaussianNormalNoiseConfig,
    schedule: Mapping[str, np.ndarray],
    *,
    seed: int,
    sigma_train: float,
) -> dict[str, Any]:
    config.validated()
    if sigma_train not in config.train_noise_stds:
        raise ValueError("tail summary scale is not registered")
    radius = sigma_train * np.asarray(schedule["standard_normal"], dtype=np.float64)
    absolute = np.abs(radius).reshape(-1)
    tube = config.base.tube_radius
    safety = config.base.safety_radius
    quantiles = {
        format(value, ".6g"): float(np.quantile(absolute, value, method="linear"))
        for value in config.tail_quantiles
    }
    tube_count = int(np.count_nonzero(absolute > tube))
    safety_count = int(np.count_nonzero(absolute > safety))
    return {
        "seed": seed,
        "sigma_train": sigma_train,
        "variance_train": sigma_train**2,
        "training_noise_law": TRAINING_NOISE_LAW,
        "unique_draw_count": int(absolute.size),
        "repeated_model_presentations": int(4 * absolute.size),
        "max_abs_radius": float(np.max(absolute)),
        "abs_radius_quantiles": quantiles,
        "quantile_method": "numpy_linear",
        "tube_radius": tube,
        "realized_count_beyond_tube": tube_count,
        "realized_fraction_beyond_tube": float(tube_count / absolute.size),
        "theoretical_fraction_beyond_tube": _tail_probability(tube, sigma_train),
        "safety_radius": safety,
        "realized_count_beyond_safety": safety_count,
        "realized_fraction_beyond_safety": float(safety_count / absolute.size),
        "theoretical_fraction_beyond_safety": _tail_probability(safety, sigma_train),
        "clipped": False,
        "rejected_or_resampled": False,
    }


def prepare_schedules(
    config: GaussianNormalNoiseConfig,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Freeze all schedules and fail before any trusted solver labeling."""

    config.validated()
    schedules: dict[int, dict[str, np.ndarray]] = {}
    seed_records: dict[str, Any] = {}
    all_tail_rows: list[dict[str, Any]] = []
    for seed in config.base.seeds:
        schedule = build_training_schedule(config, seed)
        schedules[seed] = schedule
        tails = [
            summarize_gaussian_tail(config, schedule, seed=seed, sigma_train=sigma)
            for sigma in config.train_noise_stds
        ]
        all_tail_rows.extend(tails)
        seed_records[str(seed)] = {
            "schedule": schedule_record(schedule),
            "tails": tails,
        }
    unsafe = [row for row in all_tail_rows if row["realized_count_beyond_safety"]]
    if unsafe:
        raise ValueError(
            "frozen Gaussian schedule exceeds the qualified safety radius; "
            "refusing clipping, rejection, resampling, or solver labeling"
        )
    return schedules, {
        "schema": "corrective_ode_gaussian_schedule_v1",
        "fresh_per_update": True,
        "paired_across_arms_and_scales_within_seed": True,
        "clean_rows_per_update": config.clean_rows_per_update,
        "noisy_rows_per_update": config.noisy_rows_per_update,
        "seeds": seed_records,
    }


def _phase_grid(config: GaussianNormalNoiseConfig) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, config.base.train_phases, endpoint=False)


def build_training_arrays(
    config: GaussianNormalNoiseConfig,
    schedule: Mapping[str, np.ndarray],
    *,
    sigma_train: float,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one arm's exact per-update inputs and trusted targets."""

    config.validated()
    if sigma_train < 0.0 or not math.isfinite(sigma_train):
        raise ValueError("sigma_train must be finite and nonnegative")
    valid_arms = {"CLEAN", "RECOVERY", "DYN_C", "DYN_M", "DYN_E"}
    if arm not in valid_arms:
        raise ValueError(f"unknown training arm: {arm}")
    phase = _phase_grid(config)
    clean_indices = np.asarray(schedule["clean_phase_indices"], dtype=np.int64)
    noisy_indices = np.asarray(schedule["noisy_phase_indices"], dtype=np.int64)
    normal = np.asarray(schedule["standard_normal"], dtype=np.float64)
    expected_shape = (config.base.training_steps, config.clean_rows_per_update)
    if clean_indices.shape != expected_shape or noisy_indices.shape != expected_shape:
        raise ValueError("phase schedule shape does not match the configuration")
    if normal.shape != expected_shape:
        raise ValueError("normal schedule shape does not match the configuration")
    clean_inputs = np.stack(
        (phase[clean_indices], np.zeros_like(clean_indices, dtype=np.float64)), axis=-1
    )
    second_radius = np.zeros_like(normal) if arm == "CLEAN" else sigma_train * normal
    second_inputs = np.stack((phase[noisy_indices], second_radius), axis=-1)
    inputs = np.concatenate((clean_inputs, second_inputs), axis=1)

    clean_projection = inputs.copy()
    clean_projection[..., 1] = 0.0
    clean_targets = flows(config.base)["C"].advance(clean_projection.reshape(-1, 2))
    targets = clean_targets.reshape(inputs.shape)
    if arm.startswith("DYN_"):
        key = arm.removeprefix("DYN_")
        noisy_targets = flows(config.base)[key].advance(second_inputs.reshape(-1, 2))
        targets[:, config.clean_rows_per_update :, :] = noisy_targets.reshape(
            second_inputs.shape
        )
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise FloatingPointError("training arrays contain non-finite values")
    return inputs, targets


def _schedule_mse(
    model: ResidualMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> float:
    flat_inputs = inputs.reshape(-1, 2)
    flat_targets = targets.reshape(-1, 2)
    squared_sum = 0.0
    with torch.no_grad():
        for start in range(0, flat_inputs.shape[0], chunk_size):
            stop = start + chunk_size
            difference = model(flat_inputs[start:stop]) - flat_targets[start:stop]
            squared_sum += float(torch.sum(difference * difference).item())
    return squared_sum / float(flat_targets.numel())


def _sigma_tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _train_one_model(
    config: GaussianNormalNoiseConfig,
    *,
    initial_state: Mapping[str, torch.Tensor],
    initial_digest: str,
    inputs_np: np.ndarray,
    targets_np: np.ndarray,
    seed: int,
    arm: str,
    sigma_train: float,
    schedule_digest: str,
    checkpoint_dir: Path,
) -> tuple[ResidualMLP, dict[str, Any]]:
    model = ResidualMLP(
        width=config.base.hidden_width, hidden_layers=config.base.hidden_layers
    ).to(dtype=torch.float32)
    model.load_state_dict(initial_state)
    if _state_dict_digest(model.state_dict()) != initial_digest:
        raise AssertionError("paired initialization did not close")
    inputs = torch.as_tensor(inputs_np, dtype=torch.float32)
    targets = torch.as_tensor(targets_np, dtype=torch.float32)
    initial_loss = _schedule_mse(model, inputs, targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base.learning_rate)
    last_loss = math.nan
    for update in range(config.base.training_steps):
        prediction = model(inputs[update])
        loss = torch.mean((prediction - targets[update]) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite {arm} loss at seed {seed}, sigma_train={sigma_train}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    final_loss = _schedule_mse(model, inputs, targets)
    scale_fragment = "zero" if sigma_train == 0.0 else _sigma_tag(sigma_train)
    checkpoint = checkpoint_dir / (
        f"seed_{seed}_sigma_train_{scale_fragment}_{arm.lower()}.pt"
    )
    atomic_torch_save(model.state_dict(), checkpoint)
    reloaded = ResidualMLP(
        width=config.base.hidden_width, hidden_layers=config.base.hidden_layers
    ).to(dtype=torch.float32)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    reloaded.eval()
    parameter_digest = _state_dict_digest(reloaded.state_dict())
    if parameter_digest != _state_dict_digest(model.state_dict()):
        raise AssertionError("checkpoint round trip changed parameters")
    record = {
        "seed": seed,
        "arm": arm,
        "sigma_train": sigma_train,
        "variance_train": sigma_train**2,
        "training_noise_law": (
            "clean_zero_noise_control" if arm == "CLEAN" else TRAINING_NOISE_LAW
        ),
        "clean_rows_per_update": (
            config.base.batch_size if arm == "CLEAN" else config.clean_rows_per_update
        ),
        "corrupted_rows_per_update": (
            0 if arm == "CLEAN" else config.noisy_rows_per_update
        ),
        "paired_second_phase_schedule_rows": config.noisy_rows_per_update,
        "batch_size": config.base.batch_size,
        "updates": config.base.training_steps,
        "initial_schedule_mse": initial_loss,
        "last_update_mse": last_loss,
        "final_schedule_mse": final_loss,
        "input_schedule_digest": digest_array(inputs_np),
        "target_schedule_digest": digest_array(targets_np),
        "standard_normal_schedule_digest": schedule_digest,
        "initial_parameter_digest": initial_digest,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_digest": parameter_digest,
    }
    return reloaded, record


def train_models(
    config: GaussianNormalNoiseConfig,
    schedules: Mapping[int, Mapping[str, np.ndarray]],
    *,
    checkpoint_dir: Path,
) -> tuple[
    dict[float, dict[int, dict[str, ResidualMLP]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Train 3 CLEAN plus 48 scale-specific models under the canonical contract."""

    config.validated()
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    models_by_scale = {sigma: {} for sigma in config.train_noise_stds}
    training_rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {"seeds": {}}
    for seed in config.base.seeds:
        schedule = schedules[seed]
        normal_digest = digest_array(
            np.asarray(schedule["standard_normal"], dtype=np.float64)
        )
        torch.manual_seed(seed)
        template = ResidualMLP(
            width=config.base.hidden_width, hidden_layers=config.base.hidden_layers
        ).to(dtype=torch.float32)
        initial_state = {
            name: value.detach().clone()
            for name, value in template.state_dict().items()
        }
        initial_digest = _state_dict_digest(initial_state)
        clean_inputs, clean_targets = build_training_arrays(
            config, schedule, sigma_train=0.0, arm="CLEAN"
        )
        clean_model, clean_record = _train_one_model(
            config,
            initial_state=initial_state,
            initial_digest=initial_digest,
            inputs_np=clean_inputs,
            targets_np=clean_targets,
            seed=seed,
            arm="CLEAN",
            sigma_train=0.0,
            schedule_digest=normal_digest,
            checkpoint_dir=checkpoint_dir,
        )
        training_rows.append(clean_record)
        seed_record: dict[str, Any] = {
            "initial_parameter_digest": initial_digest,
            "schedule": schedule_record(schedule),
            "checkpoints": {"CLEAN": clean_record},
        }
        for sigma_train in config.train_noise_stds:
            scale_models: dict[str, ResidualMLP] = {"CLEAN": clean_model}
            scale_records: dict[str, Any] = {}
            shared_input_digest: str | None = None
            for arm in ("RECOVERY", "DYN_C", "DYN_M", "DYN_E"):
                inputs, targets = build_training_arrays(
                    config, schedule, sigma_train=sigma_train, arm=arm
                )
                current_input_digest = digest_array(inputs)
                if shared_input_digest is None:
                    shared_input_digest = current_input_digest
                elif current_input_digest != shared_input_digest:
                    raise AssertionError("paired response-arm inputs differ")
                model, record = _train_one_model(
                    config,
                    initial_state=initial_state,
                    initial_digest=initial_digest,
                    inputs_np=inputs,
                    targets_np=targets,
                    seed=seed,
                    arm=arm,
                    sigma_train=sigma_train,
                    schedule_digest=normal_digest,
                    checkpoint_dir=checkpoint_dir,
                )
                scale_models[arm] = model
                scale_records[arm] = record
                training_rows.append(record)
            models_by_scale[sigma_train][seed] = scale_models
            seed_record["checkpoints"][_sigma_tag(sigma_train)] = scale_records
        paired["seeds"][str(seed)] = seed_record
    if len(training_rows) != config.expected_checkpoint_count:
        raise AssertionError("trained checkpoint count does not match the contract")
    return models_by_scale, training_rows, paired


def _tag_response_row(row: Mapping[str, Any], sigma_train: float) -> dict[str, Any]:
    arm = str(row["arm"])
    actual_sigma = 0.0 if arm == "CLEAN" else sigma_train
    return {
        **row,
        "sigma_train": actual_sigma,
        "variance_train": actual_sigma**2,
        "comparison_sigma_train": sigma_train,
        "training_noise_law": (
            "clean_zero_noise_control" if arm == "CLEAN" else TRAINING_NOISE_LAW
        ),
    }


def _augment_query_output_response(
    rows: list[dict[str, Any]],
    models: Mapping[int, Mapping[str, ResidualMLP]],
    config: GaussianNormalNoiseConfig,
) -> None:
    lookup = {
        (
            int(row["seed"]),
            str(row["regime"]),
            str(row["arm"]),
            str(row["bank"]),
            float(row["radius"]),
            float(row["theta"]),
        ): row
        for row in rows
    }
    phase = heldout_phases(config.base)
    clean = np.column_stack((phase, np.zeros_like(phase)))
    flow_by_key = flows(config.base)
    for seed in config.base.seeds:
        for regime, trusted_flow in flow_by_key.items():
            for arm, model in (
                ("CLEAN", models[seed]["CLEAN"]),
                ("RECOVERY", models[seed]["RECOVERY"]),
                ("DYN", models[seed][f"DYN_{regime}"]),
            ):
                clean_output = model_step(model, clean)
                trusted_clean = trusted_flow.advance(clean)
                for bank, radii in (
                    ("inner", config.base.inner_query_radii),
                    ("outer", config.base.outer_query_radii),
                ):
                    for radius in radii:
                        query = clean.copy()
                        query[:, 1] = radius
                        learned = model_step(model, query)
                        trusted = trusted_flow.advance(query)
                        for index, theta in enumerate(phase):
                            row = lookup[
                                (seed, regime, arm, bank, radius, float(theta))
                            ]
                            row.update(
                                {
                                    "learned_phase_response": float(
                                        learned[index, 0] - clean_output[index, 0]
                                    ),
                                    "learned_normal_response": float(
                                        learned[index, 1] - clean_output[index, 1]
                                    ),
                                    "trusted_phase_response": float(
                                        trusted[index, 0] - trusted_clean[index, 0]
                                    ),
                                    "trusted_normal_response": float(
                                        trusted[index, 1] - trusted_clean[index, 1]
                                    ),
                                    "output_clean_set_distance": float(
                                        abs(learned[index, 1])
                                    ),
                                }
                            )


def evaluate_radial_profile(
    models: Mapping[int, Mapping[str, ResidualMLP]],
    config: GaussianNormalNoiseConfig,
    *,
    sigma_train: float,
) -> list[dict[str, Any]]:
    """Evaluate signed fixed-radius profiles without defining a hard width."""

    phase = heldout_phases(config.base)
    clean = np.column_stack((phase, np.zeros_like(phase)))
    clean_target = flows(config.base)["C"].advance(clean)
    rows: list[dict[str, Any]] = []
    for seed in config.base.seeds:
        recovery = models[seed]["RECOVERY"]
        for radius_abs in config.profile_abs_radii:
            signed_queries = np.concatenate(
                [
                    np.column_stack((phase, np.full_like(phase, sign * radius_abs)))
                    for sign in (-1.0, 1.0)
                ],
                axis=0,
            )
            repeated_clean_target = np.tile(clean_target, (2, 1))
            recovery_output = model_step(recovery, signed_queries)
            recovery_error = np.sqrt(
                np.sum((recovery_output - repeated_clean_target) ** 2, axis=1)
            )
            rows.append(
                {
                    "seed": seed,
                    "regime": "",
                    "arm": "RECOVERY",
                    "sigma_train": sigma_train,
                    "variance_train": sigma_train**2,
                    "radius_abs": radius_abs,
                    "signed_radius_count": 2,
                    "phase_count_per_sign": int(phase.size),
                    "gaussian_log_density": _gaussian_log_density(
                        radius_abs, sigma_train
                    ),
                    "clean_return_error_rms": float(
                        np.sqrt(np.mean(recovery_error**2))
                    ),
                    "trusted_flow_defect_rms": None,
                }
            )
            for regime, trusted_flow in flows(config.base).items():
                dynamic = models[seed][f"DYN_{regime}"]
                dynamic_output = model_step(dynamic, signed_queries)
                trusted_output = trusted_flow.advance(signed_queries)
                dynamic_error = np.sqrt(
                    np.sum((dynamic_output - trusted_output) ** 2, axis=1)
                )
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "arm": "DYN",
                        "sigma_train": sigma_train,
                        "variance_train": sigma_train**2,
                        "radius_abs": radius_abs,
                        "signed_radius_count": 2,
                        "phase_count_per_sign": int(phase.size),
                        "gaussian_log_density": _gaussian_log_density(
                            radius_abs, sigma_train
                        ),
                        "clean_return_error_rms": None,
                        "trusted_flow_defect_rms": float(
                            np.sqrt(np.mean(dynamic_error**2))
                        ),
                    }
                )
    return rows


def _gaussian_log_density(radius_abs: float, sigma_train: float) -> float:
    if radius_abs <= 0.0 or sigma_train <= 0.0:
        raise ValueError("Gaussian profile density requires positive radius and scale")
    return float(
        -math.log(sigma_train)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * (radius_abs / sigma_train) ** 2
    )


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("rank data must be a finite nonempty vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return zero for a flat vector, matching the frozen flat-trend falsifier."""

    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - np.mean(first_rank)
    second_centered = second_rank - np.mean(second_rank)
    denominator = math.sqrt(
        float(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def qualify_scale_response(
    response_rows: Sequence[Mapping[str, Any]],
    config: GaussianNormalNoiseConfig,
    *,
    sigma_train: float,
) -> dict[str, Any]:
    """Apply the frozen per-seed/scale response gate, never the old classification."""

    lookup = {
        (int(row["seed"]), str(row["regime"]), str(row["arm"])): row
        for row in response_rows
    }
    seed_checks: dict[str, Any] = {}
    for seed in config.base.seeds:
        regime_checks: dict[str, Any] = {}
        dynamic_clean_errors: dict[str, float] = {}
        secants: dict[str, float] = {}
        for regime in ("C", "M", "E"):
            recovery = lookup[(seed, regime, "RECOVERY")]
            dynamic = lookup[(seed, regime, "DYN")]
            dynamic_clean_errors[regime] = float(dynamic["clean_lifted_error_rms"])
            secants[regime] = float(
                dynamic["learned_clean_reference_local_secant_log_gain_mean"]
            )
            regime_checks[regime] = {
                "dyn_normalized_inner_defect_at_most_0p15": bool(
                    float(dynamic["normalized_inner_flow_defect"]) <= 0.15
                ),
                "dyn_clean_error_at_most_0p002": bool(
                    dynamic_clean_errors[regime] <= 0.002
                ),
                "dyn_lower_trusted_response_defect": bool(
                    float(dynamic["trusted_response_defect_rms"])
                    < float(recovery["trusted_response_defect_rms"])
                ),
                "recovery_lower_clean_return_error": bool(
                    float(recovery["inner_return_error_rms"])
                    < float(dynamic["inner_return_error_rms"])
                ),
                "recovery_smaller_normal_response": bool(
                    float(recovery["normal_response_magnitude_rms"])
                    < float(dynamic["normal_response_magnitude_rms"])
                ),
            }
        minimum_clean = min(dynamic_clean_errors.values())
        maximum_clean = max(dynamic_clean_errors.values())
        clean_ratio = maximum_clean / max(minimum_clean, 1.0e-15)
        shared_checks = {
            "dyn_clean_cross_regime_ratio_at_most_2": bool(clean_ratio <= 2.0),
            "local_secant_C_below_minus_0p5": bool(secants["C"] < -0.5),
            "local_secant_M_abs_at_most_0p5": bool(abs(secants["M"]) <= 0.5),
            "local_secant_E_above_0p5": bool(secants["E"] > 0.5),
            "local_secant_ordered_C_lt_M_lt_E": bool(
                secants["C"] < secants["M"] < secants["E"]
            ),
        }
        passed = all(
            all(regime_record.values()) for regime_record in regime_checks.values()
        ) and all(shared_checks.values())
        seed_checks[str(seed)] = {
            "sigma_train": sigma_train,
            "regimes": regime_checks,
            "dynamic_clean_errors": dynamic_clean_errors,
            "dynamic_clean_error_max_min_ratio": clean_ratio,
            "local_secant_means": secants,
            "shared_checks": shared_checks,
            "response_qualified": bool(passed),
        }
    scale_pass = all(row["response_qualified"] for row in seed_checks.values())
    return {
        "sigma_train": sigma_train,
        "variance_train": sigma_train**2,
        "seeds": seed_checks,
        "rollout_qualified": bool(scale_pass),
        "rollout_interpretation": (
            "mechanism_qualified"
            if scale_pass
            else "descriptive_only_response_qualification_failed"
        ),
        "response_qualification_contract": "B2_GN_per_seed_scale",
        "inherited_b2_nl_scientific_classification_reused": False,
    }


def _tag_rollout_row(
    row: Mapping[str, Any], sigma_train: float, qualification: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(row)
    if "sigma" not in result:
        raise ValueError("inherited rollout row lacks its forcing scale")
    sigma_force = float(result.pop("sigma"))
    arm = str(result.get("arm", ""))
    if arm == "TRUSTED_ORACLE":
        actual_sigma: float | None = None
    elif arm == "CLEAN":
        actual_sigma = 0.0
    else:
        actual_sigma = sigma_train
    result.update(
        {
            "sigma_train": actual_sigma,
            "variance_train": None if actual_sigma is None else actual_sigma**2,
            "comparison_sigma_train": sigma_train,
            "sigma_force": sigma_force,
            "forcing_law": FORCING_LAW,
            "rollout_qualified": bool(qualification["rollout_qualified"]),
            "rollout_interpretation": str(qualification["rollout_interpretation"]),
        }
    )
    return result


def evaluate_models(
    models_by_scale: Mapping[float, Mapping[int, Mapping[str, ResidualMLP]]],
    training_rows: Sequence[Mapping[str, Any]],
    config: GaussianNormalNoiseConfig,
) -> dict[str, Any]:
    response_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    radial_profile_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    qualification_by_scale: dict[str, Any] = {}
    clean_training = [row for row in training_rows if row["arm"] == "CLEAN"]
    for sigma_train in config.train_noise_stds:
        scale_models = models_by_scale[sigma_train]
        metrics, queries, profiles, _legacy_response_checks = evaluate_response_bank(
            scale_models, config.base
        )
        _augment_query_output_response(queries, scale_models, config)
        tagged_metrics = [_tag_response_row(row, sigma_train) for row in metrics]
        response_rows.extend(tagged_metrics)
        query_rows.extend(_tag_response_row(row, sigma_train) for row in queries)
        profile_rows.extend(_tag_response_row(row, sigma_train) for row in profiles)
        radial_profile_rows.extend(
            evaluate_radial_profile(scale_models, config, sigma_train=sigma_train)
        )
        qualification = qualify_scale_response(
            tagged_metrics, config, sigma_train=sigma_train
        )
        qualification_by_scale[_sigma_tag(sigma_train)] = qualification

        scale_training = clean_training + [
            row
            for row in training_rows
            if row["arm"] != "CLEAN" and float(row["sigma_train"]) == sigma_train
        ]
        curves, endpoints, units, prefixes, inherited_checks = evaluate_forced_rollouts(
            scale_models, config.base, training_rows=scale_training
        )
        curve_rows.extend(
            _tag_rollout_row(row, sigma_train, qualification) for row in curves
        )
        endpoint_rows.extend(
            _tag_rollout_row(row, sigma_train, qualification) for row in endpoints
        )
        unit_rows.extend(
            _tag_rollout_row(row, sigma_train, qualification) for row in units
        )
        prefix_rows.extend(
            _tag_rollout_row(row, sigma_train, qualification) for row in prefixes
        )
        audit[_sigma_tag(sigma_train)] = {
            "forcing_sign_digests": inherited_checks["forcing_sign_digests"],
            "shared_arm_checks": inherited_checks["shared_arm_checks"],
            "old_b2_nl_response_or_rollout_classification_reused": False,
        }
    return {
        "response_rows": response_rows,
        "query_rows": query_rows,
        "profile_rows": profile_rows,
        "radial_profile_rows": radial_profile_rows,
        "curve_rows": curve_rows,
        "endpoint_rows": endpoint_rows,
        "unit_rows": unit_rows,
        "prefix_rows": prefix_rows,
        "audit": audit,
        "qualification_by_scale": qualification_by_scale,
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"cannot aggregate {key}")
    return float(np.mean(values))


def _profile_signature(
    rows: Sequence[Mapping[str, Any]],
    config: GaussianNormalNoiseConfig,
    *,
    arm: str,
) -> dict[str, Any]:
    if arm not in {"RECOVERY", "DYN"}:
        raise ValueError("profile signature arm must be RECOVERY or DYN")
    correlations: list[dict[str, Any]] = []
    regimes = ("",) if arm == "RECOVERY" else ("C", "M", "E")
    error_key = (
        "clean_return_error_rms" if arm == "RECOVERY" else "trusted_flow_defect_rms"
    )
    for seed in config.base.seeds:
        for regime in regimes:
            for radius_abs in config.profile_abs_radii:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["seed"]) == seed
                        and row["arm"] == arm
                        and row["regime"] == regime
                        and float(row["radius_abs"]) == radius_abs
                    ),
                    key=lambda row: float(row["sigma_train"]),
                )
                if len(selected) != len(config.train_noise_stds):
                    raise AssertionError("radial profile does not cover every scale")
                if tuple(float(row["sigma_train"]) for row in selected) != tuple(
                    config.train_noise_stds
                ):
                    raise AssertionError("radial profile scale ordering mismatch")
                correlation = spearman_correlation(
                    [float(row["gaussian_log_density"]) for row in selected],
                    [-float(row[error_key]) for row in selected],
                )
                correlations.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "radius_abs": radius_abs,
                        "spearman_log_density_vs_negative_error": correlation,
                    }
                )
    values = np.asarray(
        [row["spearman_log_density_vs_negative_error"] for row in correlations],
        dtype=np.float64,
    )
    median = float(np.median(values))
    positive_fraction = float(np.mean(values > 0.0))
    return {
        "arm": arm,
        "sampling_unit": ("seed_radius" if arm == "RECOVERY" else "seed_regime_radius"),
        "correlations": correlations,
        "median_spearman": median,
        "positive_fraction": positive_fraction,
        "median_positive": bool(median > 0.0),
        "majority_positive": bool(positive_fraction > 0.5),
        "signature_pass": bool(median > 0.0 and positive_fraction > 0.5),
    }


def build_scale_summary(
    evaluation: Mapping[str, Any], config: GaussianNormalNoiseConfig
) -> dict[str, Any]:
    response = evaluation["response_rows"]
    queries = evaluation["query_rows"]
    endpoints = evaluation["endpoint_rows"]
    radial_profiles = evaluation["radial_profile_rows"]
    scale_rows: list[dict[str, Any]] = []
    for sigma_train in config.train_noise_stds:
        recovery_metrics = [
            row
            for row in response
            if row["arm"] == "RECOVERY"
            and row["regime"] == "C"
            and float(row["comparison_sigma_train"]) == sigma_train
        ]
        recovery_queries = [
            row
            for row in queries
            if row["arm"] == "RECOVERY"
            and row["regime"] == "C"
            and row["bank"] == "inner"
            and float(row["comparison_sigma_train"]) == sigma_train
        ]
        row: dict[str, Any] = {
            "sigma_train": sigma_train,
            "variance_train": sigma_train**2,
            "recovery_inner_return_error_rms": _mean(
                recovery_metrics, "inner_return_error_rms"
            ),
            "recovery_inner_output_clean_set_distance_mean": _mean(
                recovery_queries, "output_clean_set_distance"
            ),
            "recovery_normal_response_magnitude_rms": _mean(
                recovery_metrics, "normal_response_magnitude_rms"
            ),
            "recovery_clean_lifted_error_rms": _mean(
                recovery_metrics, "clean_lifted_error_rms"
            ),
        }
        for regime in ("C", "M", "E"):
            dynamic = [
                item
                for item in response
                if item["arm"] == "DYN"
                and item["regime"] == regime
                and float(item["comparison_sigma_train"]) == sigma_train
            ]
            forced = [
                item
                for item in endpoints
                if int(item["seed"]) >= 0
                and item["arm"] == "DYN"
                and item["regime"] == regime
                and float(item["comparison_sigma_train"]) == sigma_train
                and float(item["sigma_force"]) == config.base.primary_noise_sigma
            ]
            row[f"dyn_{regime}_inner_flow_defect_rms"] = _mean(
                dynamic, "inner_flow_defect_rms"
            )
            row[f"dyn_{regime}_clean_lifted_error_rms"] = _mean(
                dynamic, "clean_lifted_error_rms"
            )
            row[f"dyn_{regime}_mean_tube_residence_fraction"] = _mean(
                forced, "mean_tube_residence_fraction"
            )
        scale_rows.append(row)

    recovery_signature = _profile_signature(radial_profiles, config, arm="RECOVERY")
    dynamics_signature = _profile_signature(radial_profiles, config, arm="DYN")
    return {
        "schema": "corrective_ode_gaussian_scale_response_v1",
        "status": "complete",
        "scientific_classification": "scale_response_report_complete",
        "all_registered_scales_reported": len(scale_rows)
        == len(config.train_noise_stds),
        "long_rollout_scale_selection_performed": False,
        "response_qualification_contract": "B2_GN_per_seed_scale",
        "inherited_b2_nl_scientific_classification_reused": False,
        "scale_rows": scale_rows,
        "response_qualification_by_scale": evaluation["qualification_by_scale"],
        "prediction_diagnostics": {
            "P1_recovery_density_fidelity": recovery_signature,
            "P2_dynamics_density_fidelity": dynamics_signature,
        },
        "ode_to_pde_transfer_claim": False,
    }


def _plot_scale_response(
    summary: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = summary["scale_rows"]
    x = np.asarray([float(row["sigma_train"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)

    axes[0, 0].plot(
        x,
        [row["recovery_inner_return_error_rms"] for row in rows],
        color=PAPER_ARM_COLORS["RECOVERY"],
        marker="o",
        label="clean-return error",
    )
    axes[0, 0].plot(
        x,
        [row["recovery_inner_output_clean_set_distance_mean"] for row in rows],
        color=PAPER_ARM_COLORS["RECOVERY"],
        linestyle="--",
        marker="s",
        label="output distance",
    )
    axes[0, 0].set_title("(a) Recovery response")
    axes[0, 0].set_ylabel("fixed inner-bank error")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for regime in ("C", "M", "E"):
        axes[0, 1].plot(
            x,
            [row[f"dyn_{regime}_inner_flow_defect_rms"] for row in rows],
            color=PAPER_ARM_COLORS["DYN"],
            linestyle=REGIME_LINESTYLES[regime],
            marker=REGIME_MARKERS[regime],
            label=f"DYN-{regime}",
        )
    axes[0, 1].set_title("(b) Dynamics fidelity")
    axes[0, 1].set_ylabel("trusted-flow defect")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        x,
        [row["recovery_clean_lifted_error_rms"] for row in rows],
        color=PAPER_ARM_COLORS["RECOVERY"],
        marker="o",
        label="RECOVERY",
    )
    for regime in ("C", "M", "E"):
        axes[1, 0].plot(
            x,
            [row[f"dyn_{regime}_clean_lifted_error_rms"] for row in rows],
            color=PAPER_ARM_COLORS["DYN"],
            linestyle=REGIME_LINESTYLES[regime],
            marker=REGIME_MARKERS[regime],
            label=f"DYN-{regime}",
        )
    axes[1, 0].set_title("(c) Clean one-step fidelity")
    axes[1, 0].set_ylabel("clean lifted RMS error")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for regime in ("C", "M", "E"):
        axes[1, 1].plot(
            x,
            [row[f"dyn_{regime}_mean_tube_residence_fraction"] for row in rows],
            color=PAPER_ARM_COLORS["DYN"],
            linestyle=REGIME_LINESTYLES[regime],
            marker=REGIME_MARKERS[regime],
            label=f"DYN-{regime}",
        )
    axes[1, 1].set_title(rf"(d) Forced retention, $\sigma_{{\rm force}}={0.005:g}$")
    axes[1, 1].set_ylabel("mean tube residence fraction")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [format(value, ".3g") for value in x])
        axis.set_xlabel(r"training standard deviation $\sigma_{\rm train}$")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        path = REPOSITORY_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return records


def _reviewed_source_hashes(
    source_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    return {name: str(record["sha256"]) for name, record in source_records.items()}


def validate_audit_record(
    value: Mapping[str, Any], source_records: Mapping[str, Mapping[str, Any]]
) -> None:
    if value.get("schema") != AUDIT_SCHEMA:
        raise ValueError("independent audit record has the wrong schema")
    if value.get("verdict") != "AUDIT_PASS":
        raise ValueError("independent audit record did not issue AUDIT_PASS")
    reviewed = value.get("reviewed_sources")
    if reviewed != _reviewed_source_hashes(source_records):
        raise ValueError(
            "independent audit did not review the exact current source set"
        )


def _forcing_preflight(
    config: GaussianNormalNoiseConfig, *, require_parent_match: bool
) -> dict[str, Any]:
    parent_summary = _load_json(
        REPOSITORY_ROOT / PARENT_PACKET_DIR / "learned_summary.json"
    )
    parent_rollout = parent_summary.get("rollout_checks")
    if not isinstance(parent_rollout, Mapping):
        raise TypeError("frozen B2-NL parent lacks rollout checks")
    parent_digests = parent_rollout.get("forcing_sign_digests")
    if not isinstance(parent_digests, Mapping):
        raise TypeError("frozen B2-NL parent lacks forcing-sign digests")
    expected_parent_keys = {str(float(value)) for value in config.base.noise_sigmas}
    if require_parent_match and set(parent_digests) != expected_parent_keys:
        raise ValueError("frozen B2-NL forcing-scale inventory mismatch")

    records: dict[str, Any] = {}
    for sigma_force in config.base.noise_sigmas:
        initial, forcing, sign_digest = forcing_bank(config.base, sigma_force)
        parent_value = parent_digests.get(str(float(sigma_force)))
        parent_digest = None if parent_value is None else str(parent_value)
        parent_match = sign_digest == parent_digest
        if require_parent_match and not parent_match:
            raise ValueError("regenerated forcing signs differ from the frozen parent")
        records[format(sigma_force, ".12g")] = {
            "sigma_force": sigma_force,
            "forcing_law": FORCING_LAW,
            "initial_digest": digest_array(initial),
            "forcing_digest": digest_array(forcing),
            "forcing_sign_digest": sign_digest,
            "parent_forcing_sign_digest": parent_digest,
            "parent_forcing_sign_digest_match": parent_match,
            "parent_match_required": require_parent_match,
        }
    return records


def _parent_anchor(*, verify_current_sources: bool) -> dict[str, Any]:
    manifest_path = REPOSITORY_ROOT / PARENT_PACKET_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual = sha256_file(manifest_path)
    if actual != PARENT_MANIFEST_SHA256:
        raise ValueError("frozen B2-NL parent manifest hash mismatch")
    verification = verify_parent_packet(
        REPOSITORY_ROOT / PARENT_PACKET_DIR,
        verify_current_sources=verify_current_sources,
    )
    return {
        "packet": PARENT_PACKET_DIR.as_posix(),
        "manifest_sha256": actual,
        "verified": True,
        "verification": verification,
    }


def _output_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "closeout.json"}:
            continue
        records[path.relative_to(output_dir).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return records


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def verify_packet(
    output_dir: str | Path, *, verify_current_sources: bool = True
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("B2-GN packet lacks manifest.json or closeout.json")
    manifest = _load_json(manifest_path)
    closeout = _load_json(closeout_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported B2-GN manifest schema")
    if (
        closeout.get("schema") != CLOSEOUT_SCHEMA
        or closeout.get("status") != "complete"
    ):
        raise ValueError("invalid B2-GN closeout record")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("B2-GN closeout manifest hash mismatch")
    config = _config_from_payload(_load_json(root / "config.json"))
    expected_count = config.expected_checkpoint_count
    if manifest.get("expected_checkpoint_count") != expected_count:
        raise ValueError("B2-GN expected checkpoint count mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_FILES):
        raise ValueError("B2-GN source inventory mismatch")
    if verify_current_sources and sources != _source_records():
        raise ValueError("current B2-GN sources differ from the packet")
    audit_record = _load_json(root / "audit_record.json")
    canonical_contract = bool(manifest.get("canonical_contract"))
    audit_verified = (
        audit_record.get("schema") == AUDIT_SCHEMA
        and audit_record.get("verdict") == "AUDIT_PASS"
    )
    if audit_verified:
        validate_audit_record(audit_record, sources)
    elif canonical_contract:
        raise ValueError("canonical packet lacks its independent AUDIT_PASS")
    elif audit_record.get("explicit_smoke_bypass") is not True:
        raise ValueError("noncanonical packet lacks its explicit smoke bypass")
    if manifest.get("audit_record_sha256") != sha256_file(root / "audit_record.json"):
        raise ValueError("B2-GN audit-record hash mismatch")
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise TypeError("B2-GN manifest lacks its frozen parent")
    current_parent = _parent_anchor(verify_current_sources=verify_current_sources)
    if (
        parent.get("packet") != current_parent["packet"]
        or parent.get("manifest_sha256") != current_parent["manifest_sha256"]
    ):
        raise ValueError("B2-GN parent binding mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or outputs != _output_records(root):
        raise ValueError("B2-GN output inventory or hash mismatch")
    checkpoints = [name for name in outputs if name.startswith("checkpoints/")]
    if len(checkpoints) != expected_count or any(
        not name.endswith(".pt") for name in checkpoints
    ):
        raise ValueError("B2-GN checkpoint inventory mismatch")
    schedule_summary = _load_json(root / "schedule_summary.json")
    _, regenerated = prepare_schedules(config)
    if schedule_summary != regenerated:
        raise ValueError("B2-GN schedule regeneration mismatch")
    preflight = _load_json(root / "preflight.json")
    if (
        preflight.get("schema") != "corrective_ode_gaussian_preflight_v1"
        or preflight.get("status") != "pass"
        or preflight.get("source_hashes") != _reviewed_source_hashes(sources)
        or preflight.get("forcing_banks")
        != _forcing_preflight(config, require_parent_match=canonical_contract)
    ):
        raise ValueError("B2-GN preflight record mismatch")
    if bool(preflight.get("independent_audit_verified")) != audit_verified:
        raise ValueError("B2-GN audit/preflight canonicality mismatch")
    summary = _load_json(root / "summary.json")
    if (
        summary.get("scientific_classification") != "scale_response_report_complete"
        or summary.get("response_qualification_contract") != "B2_GN_per_seed_scale"
        or summary.get("inherited_b2_nl_scientific_classification_reused") is not False
    ):
        raise ValueError("B2-GN summary has invalid classification semantics")
    with (root / "training.csv").open(encoding="utf-8", newline="") as handle:
        training_fields = next(csv.reader(handle))
    with (root / "rollout_endpoints.csv").open(encoding="utf-8", newline="") as handle:
        rollout_fields = next(csv.reader(handle))
    if "sigma_train" not in training_fields or "sigma_force" in training_fields:
        raise ValueError("B2-GN training scale schema is ambiguous")
    if "sigma_force" not in rollout_fields or "sigma" in rollout_fields:
        raise ValueError("B2-GN rollout forcing scale schema is ambiguous")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "output_count": len(outputs),
        "checkpoint_count": len(checkpoints),
        "current_sources_verified": verify_current_sources,
        "parent_manifest_sha256": current_parent["manifest_sha256"],
        "canonical_contract": canonical_contract,
    }


def run_study(
    *,
    output_dir: str | Path,
    config: GaussianNormalNoiseConfig | None = None,
    audit_record_path: str | Path | None = None,
    allow_noncanonical_smoke_without_audit: bool = False,
) -> dict[str, Any]:
    study_config = (config or GaussianNormalNoiseConfig()).validated()
    canonical_config = asdict(study_config) == asdict(GaussianNormalNoiseConfig())
    if allow_noncanonical_smoke_without_audit and canonical_config:
        raise ValueError("the independent audit cannot be bypassed canonically")
    source_snapshot = _source_records()
    audit_record: dict[str, Any] | None = None
    audit_path: Path | None = None
    if audit_record_path is not None:
        audit_path = Path(audit_record_path)
        audit_record = _load_json(audit_path)
        validate_audit_record(audit_record, source_snapshot)
    elif not allow_noncanonical_smoke_without_audit:
        raise ValueError("B2-GN run requires an independent --audit-record")

    parent_snapshot = _parent_anchor(verify_current_sources=True)
    qualification = qualify_trusted_solver(study_config.base)
    if qualification["status"] != "pass":
        raise AssertionError("trusted nonlinear ODE solver qualification failed")
    forcing_preflight = _forcing_preflight(
        study_config, require_parent_match=canonical_config
    )
    schedules, schedule_summary = prepare_schedules(study_config)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "config.json", _config_payload(study_config))
    atomic_write_json(root / "predictions.json", registered_predictions())
    if audit_path is not None:
        shutil.copy2(audit_path, root / "audit_record.json")
    else:
        atomic_write_json(
            root / "audit_record.json",
            {
                "schema": "corrective_ode_gaussian_noncanonical_smoke_bypass_v1",
                "verdict": "NOT_A_CANONICAL_AUDIT",
                "explicit_smoke_bypass": True,
            },
        )
    atomic_write_json(
        root / "source_compatibility_snapshot.json",
        {
            "schema": PROVENANCE_SCHEMA,
            "sources": source_snapshot,
            "parent": parent_snapshot,
        },
    )
    atomic_write_json(root / "solver_qualification.json", qualification)
    atomic_write_json(
        root / "preflight.json",
        {
            "schema": "corrective_ode_gaussian_preflight_v1",
            "status": "pass",
            "canonical_config": canonical_config,
            "independent_audit_verified": audit_record is not None,
            "parent": parent_snapshot,
            "source_hashes": _reviewed_source_hashes(source_snapshot),
            "solver_qualification_status": qualification["status"],
            "forcing_banks": forcing_preflight,
        },
    )
    atomic_write_json(root / "schedule_summary.json", schedule_summary)
    torch.set_num_threads(1)
    models, training_rows, paired = train_models(
        study_config, schedules, checkpoint_dir=root / "checkpoints"
    )
    evaluation = evaluate_models(models, training_rows, study_config)
    summary = build_scale_summary(evaluation, study_config)

    write_csv(root / "training.csv", training_rows)
    write_csv(root / "response_metrics.csv", evaluation["response_rows"])
    write_csv(root / "query_bank_metrics.csv", evaluation["query_rows"])
    write_csv(root / "response_profiles.csv", evaluation["profile_rows"])
    write_csv(root / "radial_profile_metrics.csv", evaluation["radial_profile_rows"])
    write_csv(root / "rollout_curves.csv", evaluation["curve_rows"])
    write_csv(root / "rollout_endpoints.csv", evaluation["endpoint_rows"])
    write_csv(root / "rollout_units.csv", evaluation["unit_rows"])
    write_csv(root / "common_prefix_units.csv", evaluation["prefix_rows"])
    atomic_write_json(
        root / "learned_summary.json",
        {
            "schema": "corrective_ode_gaussian_learned_v1",
            "paired_contract": paired,
            "evaluation_audit": evaluation["audit"],
            "response_qualification_contract": "B2_GN_per_seed_scale",
            "inherited_b2_nl_scientific_classification_reused": False,
        },
    )
    atomic_write_json(root / "summary.json", summary)
    atomic_write_json(
        root / "runtime_environment.json", runtime_environment(torch.device("cpu"))
    )
    _plot_scale_response(
        summary,
        evaluation["response_rows"],
        output_pdf=root / "gaussian_scale_response.pdf",
        output_png=root / "gaussian_scale_response.png",
    )

    if _source_records() != source_snapshot:
        raise ValueError("B2-GN sources changed during execution")
    if (
        _parent_anchor(verify_current_sources=True)["manifest_sha256"]
        != parent_snapshot["manifest_sha256"]
    ):
        raise ValueError("B2-GN parent changed during execution")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study_schema": STUDY_SCHEMA,
        "canonical_contract": bool(canonical_config and audit_record is not None),
        "expected_checkpoint_count": study_config.expected_checkpoint_count,
        "scientific_classification": summary["scientific_classification"],
        "git": git_state(),
        "parent": parent_snapshot,
        "sources": source_snapshot,
        "audit_record_sha256": sha256_file(root / "audit_record.json"),
        "schedule_digests": {
            seed: record["schedule"]
            for seed, record in schedule_summary["seeds"].items()
        },
        "outputs": _output_records(root),
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(
        root / "closeout.json",
        {
            "schema": CLOSEOUT_SCHEMA,
            "status": "complete",
            "manifest_sha256": sha256_file(root / "manifest.json"),
        },
    )
    verification = verify_packet(root)
    return {"output_dir": str(root), "summary": summary, "verification": verification}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the bounded local-CPU B2-GN Gaussian normal-noise ODE study."
        )
    )
    parser.add_argument("--mode", choices=("run", "verify"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-record",
        type=Path,
        help="required with --mode run; independent exact-source AUDIT_PASS JSON",
    )
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help="with --mode verify, skip current-source hash checks",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "verify":
        result = verify_packet(
            args.output_dir, verify_current_sources=not args.packet_only
        )
    else:
        if args.packet_only:
            raise ValueError("--packet-only is valid only with --mode verify")
        if args.audit_record is None:
            raise ValueError("--mode run requires --audit-record PATH")
        result = run_study(
            output_dir=args.output_dir, audit_record_path=args.audit_record
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

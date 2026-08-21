"""Frozen contracts for the REALM PlanarDet architecture/data scaling study.

The study changes only the model family and the number of supervised released
training trajectories.  Preprocessing, sample presentations per optimizer
step, temporal windows, optimizer, scheduler, loss, and validation population
remain fixed.  The released test trajectory is metadata-only and is never an
accepted input to this module or its trainer.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet import PLANARDET_TRAIN_GROUPS
from utility.time_dependent_no.realm_planardet_runtime import (
    ADAM_BETAS,
    ADAM_EPS,
    CHANNELS,
    FC_DIM,
    MAX_LR,
    MODE_COUNTS_XY,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    PARAMETERIZATION,
    VALIDATION_HORIZON,
    WEIGHT_DECAY,
    StepWindow,
    scheduled_step_window,
)

PREREGISTRATION_SCHEMA = "w26_l4_planardet_arch_data_scaling_preregistration_v1"
PREFLIGHT_SCHEMA = "w26_l4_planardet_arch_data_scaling_full_grid_preflight_v1"
TRAINING_CONFIG_SCHEMA = "w26_l4_planardet_arch_data_scaling_config_v1"

ARCHITECTURES = ("pcno", "pcfno", "ffno")
TRAIN_TRAJECTORY_COUNTS = (3, 7)
PRESENTATIONS_PER_STEP = 7
PCNO_WIDTH = 96
FFNO_WIDTH = 128
EXPECTED_PARAMETER_COUNTS = {
    "pcno": 10_780_401,
    "pcfno": 10_706_669,
    "ffno": 8_936_717,
}

# Exact integer levels for the seven released training conditions.  Equivalence
# ratio uses levels 0/1/2 for 0.8/1.0/1.2; temperature uses levels 0/1/4 for
# 290/300/330 K.  Fractions below normalize both axes to [0, 1] exactly.
TRAIN_PARAMETER_LEVELS = (
    (0, 1),
    (1, 4),
    (1, 1),
    (2, 1),
    (0, 4),
    (0, 0),
    (2, 0),
)

SCALING_SOURCE_PATHS = (
    "scripts/time_dependent_no/train_realm_planardet_pcno.py",
    "scripts/time_dependent_no/train_realm_planardet_scaling.py",
    "utility/time_dependent_no/realm_ffno.py",
    "utility/time_dependent_no/realm_planardet_scaling.py",
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,95}")


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _normalized_parameter_point(index: int) -> tuple[Fraction, Fraction]:
    phi_level, temperature_level = TRAIN_PARAMETER_LEVELS[index]
    return Fraction(phi_level, 2), Fraction(temperature_level, 4)


def maximin_train_indices(count: int) -> tuple[int, ...]:
    """Select the fixed train subset without using fields or validation results.

    For three trajectories, maximize the minimum squared distance in normalized
    physical-condition space, then the sum of pairwise squared distances, with
    lexicographically earlier release indices as the final deterministic tie
    break.  Seven trajectories retain the official released order.
    """

    if count not in TRAIN_TRAJECTORY_COUNTS:
        raise ValueError("train trajectory count must be 3 or 7")
    if count == len(PLANARDET_TRAIN_GROUPS):
        return tuple(range(count))

    def score(indices: tuple[int, ...]) -> tuple[Fraction, Fraction, tuple[int, ...]]:
        distances: list[Fraction] = []
        for left, right in itertools.combinations(indices, 2):
            left_point = _normalized_parameter_point(left)
            right_point = _normalized_parameter_point(right)
            distances.append(
                (left_point[0] - right_point[0]) ** 2
                + (left_point[1] - right_point[1]) ** 2
            )
        return min(distances), sum(distances), tuple(-value for value in indices)

    candidates = itertools.combinations(range(len(PLANARDET_TRAIN_GROUPS)), count)
    return max(candidates, key=score)


def active_train_groups(count: int) -> tuple[str, ...]:
    return tuple(
        PLANARDET_TRAIN_GROUPS[index] for index in maximin_train_indices(count)
    )


def balanced_case_indices(step: int, *, case_count: int, seed: int) -> tuple[int, ...]:
    """Return seven deterministic presentations balanced over active cases.

    Every three consecutive steps in the low-data arm present each active case
    exactly seven times.  With seven active cases this is byte-for-byte the
    D092 all-case permutation rule, so the existing seven-trajectory PCNO run
    remains a valid anchor for the sampling contract.
    """

    _require_plain_int(step, "step", minimum=1)
    _require_plain_int(seed, "seed")
    if case_count not in TRAIN_TRAJECTORY_COUNTS:
        raise ValueError("active case count must be 3 or 7")
    start = PRESENTATIONS_PER_STEP * (step - 1)
    indices = [
        (start + presentation) % case_count
        for presentation in range(PRESENTATIONS_PER_STEP)
    ]
    random.Random(seed + 10_000_019 + step * 97).shuffle(indices)
    return tuple(indices)


def scheduled_scaling_window(
    step: int,
    *,
    one_call_steps: int,
    seed: int,
    case_count: int,
    frame_count: int = 50,
) -> StepWindow:
    base = scheduled_step_window(
        step,
        one_call_steps=one_call_steps,
        seed=seed,
        frame_count=frame_count,
        case_count=PRESENTATIONS_PER_STEP,
    )
    return StepWindow(
        phase=base.phase,
        calls=base.calls,
        frame_start=base.frame_start,
        case_indices=balanced_case_indices(step, case_count=case_count, seed=seed),
        phase_cycle=base.phase_cycle,
    )


def validation_steps(contract: PlanarDetScalingContract) -> tuple[int, ...]:
    result = {1, contract.total_steps}
    result.update(
        range(
            contract.validation_interval,
            contract.total_steps + 1,
            contract.validation_interval,
        )
    )
    return tuple(sorted(result))


@dataclass(frozen=True)
class PlanarDetScalingContract:
    run_id: str
    architecture: str
    train_trajectory_count: int
    seed: int
    total_steps: int
    one_call_steps: int
    validation_interval: int
    checkpoint_eligible_from_step: int
    competence_npe_ceiling: float
    max_wall_seconds: float
    preflight_result_sha256: str
    normalizer_arrays_sha256: str
    data_audit_final_manifest_sha256: str
    open_manifest_payload_sha256: str
    persistence_npe: float
    linear_extrapolation_npe: float

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a safe lowercase artifact identifier")
        if self.architecture not in ARCHITECTURES:
            raise ValueError("architecture must be pcno, pcfno, or ffno")
        if self.train_trajectory_count not in TRAIN_TRAJECTORY_COUNTS:
            raise ValueError("train_trajectory_count must be 3 or 7")
        _require_plain_int(self.seed, "seed")
        _require_plain_int(self.total_steps, "total_steps", minimum=97)
        _require_plain_int(self.one_call_steps, "one_call_steps", minimum=49)
        if self.total_steps - self.one_call_steps < 48:
            raise ValueError("two-call phase must cover every legal window")
        _require_plain_int(self.validation_interval, "validation_interval", minimum=1)
        _require_plain_int(
            self.checkpoint_eligible_from_step,
            "checkpoint_eligible_from_step",
            minimum=1,
        )
        if self.checkpoint_eligible_from_step not in validation_steps(self):
            raise ValueError("checkpoint eligibility must begin at validation")
        if self.checkpoint_eligible_from_step - self.one_call_steps < 48:
            raise ValueError("checkpoint eligibility must follow a two-call cycle")
        for name in (
            "competence_npe_ceiling",
            "max_wall_seconds",
            "persistence_npe",
            "linear_extrapolation_npe",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.competence_npe_ceiling >= min(
            self.persistence_npe, self.linear_extrapolation_npe
        ):
            raise ValueError("competence ceiling must beat both frozen controls")
        for name in (
            "preflight_result_sha256",
            "normalizer_arrays_sha256",
            "data_audit_final_manifest_sha256",
            "open_manifest_payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanarDetScalingContract:
        expected = set(cls.__dataclass_fields__)
        expected.update({"schema", "canonical_payload_sha256"})
        if set(payload) != expected:
            raise ValueError("scaling preregistration keys differ")
        if payload.get("schema") != PREREGISTRATION_SCHEMA:
            raise ValueError("scaling preregistration schema differs")
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "canonical_payload_sha256"
        }
        if payload.get("canonical_payload_sha256") != canonical_json_sha256(unsigned):
            raise ValueError("scaling preregistration digest differs")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": PREREGISTRATION_SCHEMA, **asdict(self)}
        result["canonical_payload_sha256"] = canonical_json_sha256(result)
        return result

    @property
    def expected_parameter_count(self) -> int:
        return EXPECTED_PARAMETER_COUNTS[self.architecture]

    @property
    def active_train_indices(self) -> tuple[int, ...]:
        return maximin_train_indices(self.train_trajectory_count)

    @property
    def active_train_groups(self) -> tuple[str, ...]:
        return active_train_groups(self.train_trajectory_count)

    def frozen_training_config(self) -> dict[str, Any]:
        if self.architecture in {"pcno", "pcfno"}:
            model = {
                "family": "PCNO" if self.architecture == "pcno" else "PCFNO",
                "channels": CHANNELS,
                "mode_counts_xy": list(MODE_COUNTS_XY),
                "layers": [PCNO_WIDTH] * 5,
                "fc_dim": FC_DIM,
                "activation": "gelu",
                "gradient_branch": self.architecture == "pcno",
                "coordinate_policy": "physical_xy_plus_uniform_rho",
                "expected_trainable_parameters": self.expected_parameter_count,
            }
        else:
            model = {
                "family": "FFNO-M",
                "state_channels": CHANNELS,
                "coordinate_channels": 2,
                "output_channels": CHANNELS,
                "width": FFNO_WIDTH,
                "layers": 4,
                "modes_yx": [32, 32],
                "feedforward_factor": 4,
                "feedforward_layers": 2,
                "layer_norm": True,
                "head_width": 128,
                "coordinate_policy": "released_ffno_normalization",
                "expected_trainable_parameters": self.expected_parameter_count,
            }
        model.update(
            {
                "parameterization": PARAMETERIZATION,
                "zero_initialize_head": True,
                "raw_output_contract": "normalized_residual_increment",
            }
        )
        payload: dict[str, Any] = {
            "schema": TRAINING_CONFIG_SCHEMA,
            "run": asdict(self),
            "model": model,
            "precision": {
                "parameters_geometry_optimizer": "float32",
                "autocast": "bfloat16",
            },
            "data_exposure": {
                "active_train_indices": list(self.active_train_indices),
                "active_train_groups": list(self.active_train_groups),
                "unique_supervised_trajectories": self.train_trajectory_count,
                "presentations_per_optimizer_step": PRESENTATIONS_PER_STEP,
                "total_presentations": PRESENTATIONS_PER_STEP * self.total_steps,
                "subset_rule": "physical_condition_maximin_without_field_or_validation_access_v1",
                "three_case_balance": "each_case_seven_presentations_per_three_steps",
                "normalizer_fit_population": "all_seven_released_train_trajectories",
                "normalizer_control_is_transductive": self.train_trajectory_count == 3,
            },
            "training": {
                "microbatch_size": 1,
                "effective_presentations_per_step": PRESENTATIONS_PER_STEP,
                "one_call_steps": self.one_call_steps,
                "two_call_steps": self.total_steps - self.one_call_steps,
                "two_call_first_proposal": "detached",
                "loss": "sum_of_five_group_mean_squared_errors",
                "frame_schedule": "d092_phase_local_shuffled_cycles",
                "parameterization": PARAMETERIZATION,
            },
            "optimizer": {
                "name": "AdamW",
                "max_lr": MAX_LR,
                "weight_decay": WEIGHT_DECAY,
                "betas": list(ADAM_BETAS),
                "eps": ADAM_EPS,
                "gradient_clip": None,
            },
            "scheduler": {
                "name": "OneCycleLR",
                "total_steps": self.total_steps,
                "pct_start": ONECYCLE_PCT_START,
                "anneal_strategy": "cos",
                "cycle_momentum": False,
                "div_factor": ONECYCLE_DIV_FACTOR,
                "final_div_factor": ONECYCLE_FINAL_DIV_FACTOR,
            },
            "validation": {
                "population": "one_released_validation_trajectory",
                "horizon": VALIDATION_HORIZON,
                "steps": list(validation_steps(self)),
                "selection_metric": "teacher_forced_realm_npe_mean",
                "eligible_from_step": self.checkpoint_eligible_from_step,
            },
            "scope": {
                "test_objects_allowed": False,
                "architecture_regularization_is_matched": True,
                "sealed_test_ranking_allowed": False,
            },
        }
        payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
        return payload


__all__ = [
    "ARCHITECTURES",
    "EXPECTED_PARAMETER_COUNTS",
    "FFNO_WIDTH",
    "PCNO_WIDTH",
    "PREFLIGHT_SCHEMA",
    "PREREGISTRATION_SCHEMA",
    "PRESENTATIONS_PER_STEP",
    "SCALING_SOURCE_PATHS",
    "TRAIN_PARAMETER_LEVELS",
    "TRAIN_TRAJECTORY_COUNTS",
    "PlanarDetScalingContract",
    "active_train_groups",
    "balanced_case_indices",
    "maximin_train_indices",
    "scheduled_scaling_window",
    "validation_steps",
]

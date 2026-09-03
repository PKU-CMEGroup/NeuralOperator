"""Run the preregistered B2-NL nonlinear corrective-ODE stress test.

This entry point is deliberately independent of the completed affine ODE
packet.  It reuses only the frozen residual-MLP implementation, binds that
source in its own manifest, and never mutates either completed packet.

All solver-relative quantities are offline diagnostics.  Recurrent forcing is
applied after the complete learned map and is compared both with the unforced
clean path and with the trusted path under the identical forcing sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.time_dependent_no.run_corrective_ode_study import (
    ResidualMLP,
    chordal_error,
    lifted_coordinate_error,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    digest_array,
    git_state,
    sha256_file,
    write_csv,
)

STUDY_SCHEMA = "corrective_ode_nonlinear_stress_v1"
CONFIG_SCHEMA = "corrective_ode_nonlinear_config_v1"
PREDICTION_SCHEMA = "corrective_ode_nonlinear_predictions_v1"
MANIFEST_SCHEMA = "corrective_ode_nonlinear_manifest_v1"

AFFINE_PACKET_MANIFEST_SHA256 = (
    "54d65dfe6f7186768d270453be780f5f1fe07d74c8042f607b3278e9439edb99"
)
LANDSCAPE_PACKET_MANIFEST_SHA256 = (
    "256e175d8d77e352a2fb545e85ce4470e603a9af06cfb8370e1df03562f6ff0a"
)
AFFINE_PACKET_MANIFEST = Path(
    "artifacts/time_dependent_no/corrective_ode_study_20260830d/manifest.json"
)
LANDSCAPE_PACKET_MANIFEST = Path(
    "artifacts/time_dependent_no/corrective_ode_landscapes_20260830e/manifest.json"
)

SOURCE_FILES = (
    "docs/time_dependent_no/B2_NL_NONLINEAR_ODE_STRESS_PREREGISTRATION.md",
    "scripts/time_dependent_no/run_corrective_ode_nonlinear_stress.py",
    "tests/time_dependent_no/test_corrective_ode_nonlinear_stress.py",
    "scripts/time_dependent_no/run_corrective_ode_study.py",
    "utility/time_dependent_no/pcno_artifacts.py",
)
PROVENANCE_SNAPSHOT_NAME = "source_compatibility_snapshot.json"
PROVENANCE_SNAPSHOT_SCHEMA = "corrective_ode_nonlinear_provenance_snapshot_v1"


@dataclass(frozen=True)
class Regime:
    key: str
    label: str
    offset: float
    cosine_amplitude: float

    def validated(self) -> Regime:
        if not self.key or not self.label:
            raise ValueError("regime names must be nonempty")
        if not math.isfinite(self.offset) or not math.isfinite(self.cosine_amplitude):
            raise ValueError("regime coefficients must be finite")
        return self


REGIMES = (
    Regime("C", "contractive", -0.4, 0.2),
    Regime("M", "mixed", 0.0, 0.35),
    Regime("E", "expansive", 0.4, 0.2),
)


@dataclass(frozen=True)
class NonlinearStressConfig:
    step_size: float = 0.2
    solver_substeps: int = 32
    qualification_substeps: int = 64
    train_phases: int = 64
    training_radii: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
    inner_query_radii: tuple[float, ...] = (-0.1, -0.025, 0.025, 0.1)
    outer_query_radii: tuple[float, ...] = (-0.2, 0.2)
    hidden_width: int = 64
    hidden_layers: int = 3
    learning_rate: float = 2.0e-3
    batch_size: int = 128
    training_steps: int = 3000
    seeds: tuple[int, ...] = (17, 29, 43)
    response_epsilon: float = 1.0e-3
    local_secant_steps: int = 32
    max_normalized_inner_flow_defect: float = 0.15
    max_clean_lifted_error: float = 2.0e-3
    max_dyn_clean_error_ratio: float = 2.0
    noise_steps: int = 192
    noise_phases: int = 32
    noise_sequences: int = 16
    forcing_seed: int = 20260831
    noise_sigmas: tuple[float, ...] = (0.0, 1.0e-3, 5.0e-3)
    primary_noise_sigma: float = 5.0e-3
    tube_radius: float = 0.15
    safety_radius: float = 0.5
    min_common_prefix_steps: int = 16

    def validated(self) -> NonlinearStressConfig:
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if self.solver_substeps < 1 or self.qualification_substeps < 1:
            raise ValueError("solver substep counts must be positive")
        if self.qualification_substeps <= self.solver_substeps:
            raise ValueError("qualification solver must be finer than canonical solver")
        if self.train_phases < 16 or self.train_phases % 2:
            raise ValueError("train_phases must be an even integer at least 16")
        if self.noise_phases < 4 or self.train_phases % self.noise_phases:
            raise ValueError("noise_phases must divide train_phases")
        if self.noise_sequences < 2 or self.noise_sequences % 2:
            raise ValueError("noise_sequences must be positive and even")
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training_steps and batch_size must be positive")
        if self.hidden_width < 1 or self.hidden_layers < 1:
            raise ValueError("hidden dimensions must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if len(self.training_radii) < 2 or len(set(self.training_radii)) != len(
            self.training_radii
        ):
            raise ValueError("training_radii must contain distinct values")
        if not all(
            math.isfinite(value) and value != 0.0 for value in self.training_radii
        ):
            raise ValueError("training radii must be finite and nonzero")
        if not all(
            math.isfinite(value) and 0.0 < abs(value) < self.tube_radius
            for value in self.inner_query_radii
        ):
            raise ValueError("inner queries must lie strictly inside the tube")
        if not all(
            math.isfinite(value) and self.tube_radius < abs(value) < self.safety_radius
            for value in self.outer_query_radii
        ):
            raise ValueError("outer queries must lie between tube and safety radii")
        if (
            not 0.0
            < self.response_epsilon
            < min(abs(value) for value in self.inner_query_radii)
        ):
            raise ValueError("response_epsilon must be a small positive inner radius")
        if self.local_secant_steps < 2 or self.noise_steps < 2:
            raise ValueError("rollout horizons must be at least two")
        if (
            not math.isfinite(self.max_normalized_inner_flow_defect)
            or self.max_normalized_inner_flow_defect <= 0.0
        ):
            raise ValueError("normalized inner-flow threshold must be positive")
        if (
            not math.isfinite(self.max_clean_lifted_error)
            or self.max_clean_lifted_error <= 0.0
        ):
            raise ValueError("clean-error threshold must be positive")
        if (
            not math.isfinite(self.max_dyn_clean_error_ratio)
            or self.max_dyn_clean_error_ratio < 1.0
        ):
            raise ValueError("clean-error comparability ratio must be at least one")
        if not 0.0 < self.tube_radius < self.safety_radius:
            raise ValueError("tube radius must lie inside the safety radius")
        if not 1 <= self.min_common_prefix_steps <= self.noise_steps:
            raise ValueError("common-prefix threshold must lie within the rollout")
        if not self.noise_sigmas or self.noise_sigmas[0] != 0.0:
            raise ValueError("noise_sigmas must start with the zero-forcing control")
        if any(not math.isfinite(value) or value < 0.0 for value in self.noise_sigmas):
            raise ValueError("noise sigmas must be finite and nonnegative")
        if self.primary_noise_sigma not in self.noise_sigmas:
            raise ValueError("primary_noise_sigma must be registered in noise_sigmas")
        for regime in REGIMES:
            regime.validated()
        return self


def _finite_state(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise ValueError(f"{name} must end in dimension two")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class NonlinearFlow:
    regime: Regime
    step_size: float = 0.2
    substeps: int = 32

    def validated(self) -> NonlinearFlow:
        self.regime.validated()
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        return self

    def vector_field(self, state: np.ndarray) -> np.ndarray:
        state_array = _finite_state(state, name="state")
        theta = state_array[..., 0]
        radius = state_array[..., 1]
        result = np.empty_like(state_array)
        result[..., 0] = (
            1.0
            + 0.2 * np.sin(theta)
            + (0.5 + 0.2 * np.cos(theta)) * radius
            + 0.1 * radius * radius
        )
        kappa = self.regime.offset + self.regime.cosine_amplitude * np.cos(theta)
        result[..., 1] = kappa * radius - 4.0 * radius**3
        return result

    def advance(self, state: np.ndarray, *, substeps: int | None = None) -> np.ndarray:
        self.validated()
        state_array = _finite_state(state, name="state")
        steps = self.substeps if substeps is None else int(substeps)
        if steps < 1:
            raise ValueError("substeps must be positive")
        result = state_array.copy()
        dt = self.step_size / steps
        for _ in range(steps):
            k1 = self.vector_field(result)
            k2 = self.vector_field(result + 0.5 * dt * k1)
            k3 = self.vector_field(result + 0.5 * dt * k2)
            k4 = self.vector_field(result + dt * k3)
            result = result + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(result).all():
            raise FloatingPointError("trusted RK4 step produced a non-finite state")
        return result

    def variational_normal_multiplier(
        self, theta: np.ndarray, *, substeps: int = 256
    ) -> np.ndarray:
        """Integrate the clean normal variational equation independently."""

        theta_array = np.asarray(theta, dtype=np.float64)
        if not np.isfinite(theta_array).all() or substeps < 1:
            raise ValueError("invalid variational inputs")
        phase = theta_array.copy()
        log_gain = np.zeros_like(phase)
        dt = self.step_size / substeps

        def derivative(current_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            phase_rate = 1.0 + 0.2 * np.sin(current_phase)
            gain_rate = self.regime.offset + self.regime.cosine_amplitude * np.cos(
                current_phase
            )
            return phase_rate, gain_rate

        for _ in range(substeps):
            p1, g1 = derivative(phase)
            p2, g2 = derivative(phase + 0.5 * dt * p1)
            p3, g3 = derivative(phase + 0.5 * dt * p2)
            p4, g4 = derivative(phase + dt * p3)
            phase = phase + (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4)
            log_gain = log_gain + (dt / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
        return np.exp(log_gain)

    @property
    def exact_full_cycle_log_gain(self) -> float:
        return float(self.regime.offset * 2.0 * np.pi / math.sqrt(1.0 - 0.2**2))


def flows(config: NonlinearStressConfig) -> dict[str, NonlinearFlow]:
    config.validated()
    return {
        regime.key: NonlinearFlow(
            regime=regime,
            step_size=config.step_size,
            substeps=config.solver_substeps,
        ).validated()
        for regime in REGIMES
    }


def heldout_phases(config: NonlinearStressConfig) -> np.ndarray:
    config.validated()
    spacing = 2.0 * np.pi / config.train_phases
    return -np.pi + (np.arange(config.train_phases, dtype=np.float64) + 0.5) * spacing


def qualify_trusted_solver(config: NonlinearStressConfig) -> dict[str, Any]:
    config.validated()
    phase = heldout_phases(config)
    radii = np.asarray(
        sorted(
            {
                0.0,
                -config.safety_radius,
                config.safety_radius,
                *config.training_radii,
                *config.inner_query_radii,
                *config.outer_query_radii,
            }
        ),
        dtype=np.float64,
    )
    bank = np.concatenate(
        [np.column_stack((phase, np.full_like(phase, radius))) for radius in radii],
        axis=0,
    )
    clean = np.column_stack((phase, np.zeros_like(phase)))
    records: dict[str, Any] = {}
    clean_outputs: list[np.ndarray] = []
    all_pass = True
    epsilon = 1.0e-6
    safety_tolerance = 64.0 * np.finfo(np.float64).eps
    flow_by_key = flows(config)
    clean_flow = flow_by_key["C"]
    for key, flow in flow_by_key.items():
        canonical = flow.advance(bank)
        qualified = flow.advance(bank, substeps=config.qualification_substeps)
        difference = lifted_coordinate_error(canonical, qualified)
        plus = clean.copy()
        minus = clean.copy()
        plus[:, 1] = epsilon
        minus[:, 1] = -epsilon
        finite_difference = (
            flow.advance(plus, substeps=config.qualification_substeps)[:, 1]
            - flow.advance(minus, substeps=config.qualification_substeps)[:, 1]
        ) / (2.0 * epsilon)
        variational = flow.variational_normal_multiplier(phase)
        response_error = np.abs(finite_difference - variational)
        clean_output = flow.advance(clean)
        clean_outputs.append(clean_output)
        trusted_local_secant, trusted_local_secant_sign = (
            _clean_reference_local_secant_log_gain(
                flow.advance,
                clean_flow,
                clean,
                epsilon=config.response_epsilon,
                steps=config.local_secant_steps,
            )
        )
        trusted_local_secant_mean = float(np.mean(trusted_local_secant))
        max_abs_normal_32 = float(np.max(np.abs(canonical[:, 1])))
        max_abs_normal_64 = float(np.max(np.abs(qualified[:, 1])))
        record = {
            "max_32_vs_64_lifted_difference": float(np.max(difference)),
            "rms_32_vs_64_lifted_difference": float(np.sqrt(np.mean(difference**2))),
            "max_variational_response_error": float(np.max(response_error)),
            "max_clean_normal_residual": float(np.max(np.abs(clean_output[:, 1]))),
            "exact_full_cycle_log_gain": flow.exact_full_cycle_log_gain,
            "trusted_clean_reference_local_secant_log_gain_mean": (
                trusted_local_secant_mean
            ),
            "trusted_clean_reference_local_secant_negative_sign_fraction": float(
                np.mean(trusted_local_secant_sign < 0.0)
            ),
            "max_abs_normal_after_32_substep_map": max_abs_normal_32,
            "max_abs_normal_after_64_substep_map": max_abs_normal_64,
            "solver_difference_pass": bool(np.max(difference) <= 1.0e-8),
            "variational_response_pass": bool(np.max(response_error) <= 2.0e-7),
            "clean_invariance_pass": bool(np.max(np.abs(clean_output[:, 1])) == 0.0),
            "discrete_map_32_safety_pass": bool(
                max_abs_normal_32 <= config.safety_radius + safety_tolerance
            ),
            "discrete_map_64_safety_pass": bool(
                max_abs_normal_64 <= config.safety_radius + safety_tolerance
            ),
        }
        record["regime_sign_pass"] = bool(
            (key == "C" and flow.exact_full_cycle_log_gain < 0.0)
            or (key == "M" and flow.exact_full_cycle_log_gain == 0.0)
            or (key == "E" and flow.exact_full_cycle_log_gain > 0.0)
        )
        record["trusted_local_secant_threshold_pass"] = bool(
            (key == "C" and trusted_local_secant_mean < -0.5)
            or (key == "M" and abs(trusted_local_secant_mean) <= 0.5)
            or (key == "E" and trusted_local_secant_mean > 0.5)
        )
        record["pass"] = all(
            record[name]
            for name in (
                "solver_difference_pass",
                "variational_response_pass",
                "clean_invariance_pass",
                "discrete_map_32_safety_pass",
                "discrete_map_64_safety_pass",
                "regime_sign_pass",
                "trusted_local_secant_threshold_pass",
            )
        )
        all_pass &= bool(record["pass"])
        records[key] = record

    common_clean_bytes = all(
        np.array_equal(clean_outputs[0], output) for output in clean_outputs[1:]
    )
    phase_rate_lower_bound = 1.0 - 0.2 - 0.7 * config.safety_radius
    inward_margin = min(
        4.0 * config.safety_radius**2 - (regime.offset + abs(regime.cosine_amplitude))
        for regime in REGIMES
    )
    trusted_local_secant_means = {
        key: float(record["trusted_clean_reference_local_secant_log_gain_mean"])
        for key, record in records.items()
    }
    structural = {
        "common_clean_successor_bytes": common_clean_bytes,
        "phase_rate_lower_bound_on_safety_strip": phase_rate_lower_bound,
        "normal_inward_margin_at_safety_boundary": inward_margin,
        "phase_monotone_pass": bool(phase_rate_lower_bound > 0.0),
        "safety_boundary_inward_pass": bool(inward_margin > 0.0),
        "sampled_discrete_map_safety_pass": bool(
            all(
                record["discrete_map_32_safety_pass"]
                and record["discrete_map_64_safety_pass"]
                for record in records.values()
            )
        ),
        "trusted_clean_reference_local_secant_log_gain_means": (
            trusted_local_secant_means
        ),
        "trusted_local_secant_order_pass": bool(
            trusted_local_secant_means["C"]
            < trusted_local_secant_means["M"]
            < trusted_local_secant_means["E"]
        ),
    }
    all_pass &= all(
        structural[name]
        for name in (
            "common_clean_successor_bytes",
            "phase_monotone_pass",
            "safety_boundary_inward_pass",
            "sampled_discrete_map_safety_pass",
            "trusted_local_secant_order_pass",
        )
    )
    return {
        "schema": "corrective_ode_nonlinear_solver_qualification_v1",
        "status": "pass" if all_pass else "fail",
        "bank_digest": digest_array(bank),
        "regimes": records,
        "structural": structural,
    }


def build_training_contract(
    config: NonlinearStressConfig,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    config.validated()
    flow_by_key = flows(config)
    phase = np.linspace(-np.pi, np.pi, config.train_phases, endpoint=False)
    clean = np.column_stack((phase, np.zeros_like(phase)))
    clean_targets = [flow.advance(clean) for flow in flow_by_key.values()]
    if not all(np.array_equal(clean_targets[0], value) for value in clean_targets[1:]):
        raise AssertionError("clean targets differ across normal-response regimes")
    clean_target = clean_targets[0]
    repeats = len(config.training_radii)
    clean_block = np.tile(clean, (repeats, 1))
    clean_target_block = np.tile(clean_target, (repeats, 1))
    displaced = np.concatenate(
        [
            np.column_stack((phase, np.full_like(phase, radius)))
            for radius in config.training_radii
        ],
        axis=0,
    )
    common_inputs = np.concatenate((clean_block, displaced), axis=0)
    recovery_targets = np.concatenate((clean_target_block, clean_target_block), axis=0)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "CLEAN": (
            np.concatenate((clean_block, clean_block), axis=0),
            np.concatenate((clean_target_block, clean_target_block), axis=0),
        ),
        "RECOVERY": (common_inputs, recovery_targets),
    }
    for key, flow in flow_by_key.items():
        datasets[f"DYN_{key}"] = (
            common_inputs,
            np.concatenate((clean_target_block, flow.advance(displaced)), axis=0),
        )
    input_digests = {name: digest_array(values[0]) for name, values in datasets.items()}
    if len({input_digests[f"DYN_{key}"] for key in flow_by_key}) != 1:
        raise AssertionError("dynamics-relabeling inputs differ by regime")
    if input_digests["RECOVERY"] != input_digests["DYN_C"]:
        raise AssertionError("recovery and dynamics-relabeling inputs differ")
    rows = next(iter(datasets.values()))[0].shape[0]
    if not all(values[0].shape[0] == rows for values in datasets.values()):
        raise AssertionError("training row counts differ across arms")
    metadata = {
        "clean_digest": digest_array(clean),
        "clean_target_digest": digest_array(clean_target),
        "displaced_digest": digest_array(displaced),
        "common_input_digest": input_digests["RECOVERY"],
        "recovery_target_digest": digest_array(recovery_targets),
        "input_digests": input_digests,
        "target_digests": {
            name: digest_array(values[1]) for name, values in datasets.items()
        },
        "rows_per_arm": rows,
        "clean_rows_per_arm": clean_block.shape[0],
        "displaced_rows_per_response_arm": displaced.shape[0],
    }
    return datasets, metadata


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _batch_plan(config: NonlinearStressConfig, seed: int, rows: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000_019)
    return rng.integers(
        0,
        rows,
        size=(config.training_steps, config.batch_size),
        dtype=np.int64,
    )


def train_models(
    config: NonlinearStressConfig,
    *,
    checkpoint_dir: Path,
) -> tuple[dict[int, dict[str, ResidualMLP]], list[dict[str, Any]], dict[str, Any]]:
    config.validated()
    datasets, data_metadata = build_training_contract(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    models: dict[int, dict[str, ResidualMLP]] = {}
    rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {"data": data_metadata, "seeds": {}}
    arm_order = ("CLEAN", "RECOVERY", "DYN_C", "DYN_M", "DYN_E")

    for seed in config.seeds:
        torch.manual_seed(seed)
        template = ResidualMLP(
            width=config.hidden_width, hidden_layers=config.hidden_layers
        ).to(dtype=torch.float32)
        initial_state = {
            name: value.detach().clone()
            for name, value in template.state_dict().items()
        }
        initial_digest = _state_dict_digest(initial_state)
        row_count = next(iter(datasets.values()))[0].shape[0]
        batches = _batch_plan(config, seed, row_count)
        batch_digest = digest_array(batches)
        models[seed] = {}
        seed_record: dict[str, Any] = {
            "initial_parameter_digest": initial_digest,
            "batch_plan_digest": batch_digest,
            "arms": {},
        }
        for arm in arm_order:
            model = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            model.load_state_dict(initial_state)
            if _state_dict_digest(model.state_dict()) != initial_digest:
                raise AssertionError("paired initialization did not close")
            inputs_np, targets_np = datasets[arm]
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32)
            targets = torch.as_tensor(targets_np, dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            with torch.no_grad():
                initial_loss = float(torch.mean((model(inputs) - targets) ** 2).item())
            for batch in batches:
                indices = torch.from_numpy(batch)
                prediction = model(inputs[indices])
                loss = torch.mean((prediction - targets[indices]) ** 2)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {arm} loss at seed {seed}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final_loss = float(torch.mean((model(inputs) - targets) ** 2).item())
            checkpoint = checkpoint_dir / f"seed_{seed}_{arm.lower()}.pt"
            torch.save(model.state_dict(), checkpoint)
            reloaded = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            reloaded.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            reloaded.eval()
            if _state_dict_digest(reloaded.state_dict()) != _state_dict_digest(
                model.state_dict()
            ):
                raise AssertionError("checkpoint round trip changed parameters")
            models[seed][arm] = reloaded
            arm_record = {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_digest": _state_dict_digest(reloaded.state_dict()),
            }
            seed_record["arms"][arm] = arm_record
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "updates": config.training_steps,
                    "initial_parameter_digest": initial_digest,
                    "batch_plan_digest": batch_digest,
                    **arm_record,
                }
            )
        paired["seeds"][str(seed)] = seed_record
    return models, rows, paired


def model_step(
    model: ResidualMLP, state: np.ndarray, *, allow_nonfinite: bool = False
) -> np.ndarray:
    state_array = _finite_state(state, name="state")
    original_shape = state_array.shape
    flat = state_array.reshape(-1, 2)
    with torch.no_grad():
        output = model(torch.as_tensor(flat, dtype=torch.float32)).cpu().numpy()
    result = np.asarray(output, dtype=np.float64).reshape(original_shape)
    if not allow_nonfinite and not np.isfinite(result).all():
        raise FloatingPointError("learned map produced a non-finite state")
    return result


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("RMS values must be nonempty and finite")
    return float(np.sqrt(np.mean(array**2)))


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if array.shape != mask_array.shape:
        raise ValueError("masked RMS requires aligned values and mask")
    coverage = float(np.mean(mask_array))
    if not np.any(mask_array):
        return 0.0, coverage
    return _rms(array[mask_array]), coverage


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if array.shape != mask_array.shape:
        raise ValueError("masked mean requires aligned values and mask")
    coverage = float(np.mean(mask_array))
    if not np.any(mask_array):
        return 0.0, coverage
    subset = array[mask_array]
    if not np.isfinite(subset).all():
        raise ValueError("masked mean requires finite selected values")
    return float(np.mean(subset)), coverage


def _response_profile(
    step: Callable[[np.ndarray], np.ndarray],
    clean: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    plus = clean.copy()
    minus = clean.copy()
    plus[:, 1] = epsilon
    minus[:, 1] = -epsilon
    response = (step(plus) - step(minus)) / (2.0 * epsilon)
    return response[:, 0], response[:, 1]


def _clean_reference_local_secant_log_gain(
    step: Callable[[np.ndarray], np.ndarray],
    clean_flow: NonlinearFlow,
    initial: np.ndarray,
    *,
    epsilon: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    clean = initial.copy()
    log_gain = np.zeros(clean.shape[0], dtype=np.float64)
    sign = np.ones(clean.shape[0], dtype=np.float64)
    for _ in range(steps):
        _, local_gain = _response_profile(step, clean, epsilon)
        sign *= np.sign(local_gain)
        log_gain += np.log(np.maximum(np.abs(local_gain), 1.0e-12))
        clean = clean_flow.advance(clean)
    return log_gain, sign


def evaluate_response_bank(
    models: Mapping[int, Mapping[str, ResidualMLP]],
    config: NonlinearStressConfig,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    config.validated()
    phase = heldout_phases(config)
    clean = np.column_stack((phase, np.zeros_like(phase)))
    flow_by_key = flows(config)
    clean_flow = flow_by_key["C"]
    clean_target = clean_flow.advance(clean)
    metric_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        seed_models = models[seed]
        for regime_key, trusted_flow in flow_by_key.items():
            deployed = (
                ("CLEAN", seed_models["CLEAN"]),
                ("RECOVERY", seed_models["RECOVERY"]),
                ("DYN", seed_models[f"DYN_{regime_key}"]),
            )
            trusted_phase, trusted_normal = _response_profile(
                trusted_flow.advance, clean, config.response_epsilon
            )
            trusted_local_secant, trusted_local_secant_sign = (
                _clean_reference_local_secant_log_gain(
                    trusted_flow.advance,
                    clean_flow,
                    clean,
                    epsilon=config.response_epsilon,
                    steps=config.local_secant_steps,
                )
            )
            for arm, model in deployed:
                step = lambda value, current=model: model_step(current, value)
                clean_output = step(clean)
                clean_lifted = lifted_coordinate_error(clean_output, clean_target)
                clean_chordal = chordal_error(clean_output, clean_target)
                learned_phase, learned_normal = _response_profile(
                    step, clean, config.response_epsilon
                )
                learned_local_secant, learned_local_secant_sign = (
                    _clean_reference_local_secant_log_gain(
                        step,
                        clean_flow,
                        clean,
                        epsilon=config.response_epsilon,
                        steps=config.local_secant_steps,
                    )
                )
                for index, theta in enumerate(phase):
                    profile_rows.append(
                        {
                            "seed": seed,
                            "regime": regime_key,
                            "arm": arm,
                            "theta": float(theta),
                            "trusted_phase_response": float(trusted_phase[index]),
                            "learned_phase_response": float(learned_phase[index]),
                            "trusted_normal_gain": float(trusted_normal[index]),
                            "learned_normal_gain": float(learned_normal[index]),
                            "trusted_clean_reference_local_secant_log_gain": float(
                                trusted_local_secant[index]
                            ),
                            "learned_clean_reference_local_secant_log_gain": float(
                                learned_local_secant[index]
                            ),
                            "trusted_clean_reference_local_secant_sign": float(
                                trusted_local_secant_sign[index]
                            ),
                            "learned_clean_reference_local_secant_sign": float(
                                learned_local_secant_sign[index]
                            ),
                        }
                    )

                bank_metrics: dict[str, float] = {}
                for bank_name, radii in (
                    ("inner", config.inner_query_radii),
                    ("outer", config.outer_query_radii),
                ):
                    flow_defects = []
                    return_errors = []
                    trusted_displaced_responses = []
                    for radius in radii:
                        query = clean.copy()
                        query[:, 1] = radius
                        learned = step(query)
                        trusted = trusted_flow.advance(query)
                        flow_error = lifted_coordinate_error(learned, trusted)
                        return_error = lifted_coordinate_error(learned, clean_target)
                        trusted_displaced_response = lifted_coordinate_error(
                            trusted, clean_target
                        )
                        flow_defects.extend(flow_error.tolist())
                        return_errors.extend(return_error.tolist())
                        trusted_displaced_responses.extend(
                            trusted_displaced_response.tolist()
                        )
                        for index, theta in enumerate(phase):
                            query_rows.append(
                                {
                                    "seed": seed,
                                    "regime": regime_key,
                                    "arm": arm,
                                    "bank": bank_name,
                                    "theta": float(theta),
                                    "radius": float(radius),
                                    "trusted_flow_defect": float(flow_error[index]),
                                    "clean_return_error": float(return_error[index]),
                                    "trusted_displaced_response": float(
                                        trusted_displaced_response[index]
                                    ),
                                }
                            )
                    bank_metrics[f"{bank_name}_flow_defect_rms"] = _rms(
                        np.asarray(flow_defects)
                    )
                    bank_metrics[f"{bank_name}_return_error_rms"] = _rms(
                        np.asarray(return_errors)
                    )
                    bank_metrics[f"{bank_name}_trusted_displaced_response_rms"] = _rms(
                        np.asarray(trusted_displaced_responses)
                    )

                trusted_inner_scale = bank_metrics[
                    "inner_trusted_displaced_response_rms"
                ]
                if trusted_inner_scale <= 0.0:
                    raise AssertionError("trusted inner response has zero RMS scale")
                bank_metrics["normalized_inner_flow_defect"] = (
                    bank_metrics["inner_flow_defect_rms"] / trusted_inner_scale
                )

                metric_rows.append(
                    {
                        "seed": seed,
                        "regime": regime_key,
                        "arm": arm,
                        "clean_lifted_error_rms": _rms(clean_lifted),
                        "clean_chordal_error_rms": _rms(clean_chordal),
                        "fresh_normal_forcing_rms": _rms(clean_output[:, 1]),
                        "trusted_response_defect_rms": math.sqrt(
                            _rms(learned_phase - trusted_phase) ** 2
                            + _rms(learned_normal - trusted_normal) ** 2
                        ),
                        "normal_response_magnitude_rms": _rms(learned_normal),
                        "trusted_clean_reference_local_secant_log_gain_mean": float(
                            np.mean(trusted_local_secant)
                        ),
                        "learned_clean_reference_local_secant_log_gain_mean": float(
                            np.mean(learned_local_secant)
                        ),
                        "learned_clean_reference_local_secant_negative_sign_fraction": float(
                            np.mean(learned_local_secant_sign < 0.0)
                        ),
                        **bank_metrics,
                    }
                )

    lookup = {
        (int(row["seed"]), str(row["regime"]), str(row["arm"])): row
        for row in metric_rows
    }
    target_checks: dict[str, Any] = {}
    local_secant_checks: dict[str, Any] = {}
    clean_trace_checks: dict[str, Any] = {}
    for seed in config.seeds:
        target_checks[str(seed)] = {}
        for regime in ("C", "M", "E"):
            recovery = lookup[(seed, regime, "RECOVERY")]
            dynamic = lookup[(seed, regime, "DYN")]
            target_checks[str(seed)][regime] = {
                "dynamic_lower_trusted_response_defect": bool(
                    dynamic["trusted_response_defect_rms"]
                    < recovery["trusted_response_defect_rms"]
                ),
                "recovery_lower_inner_return_error": bool(
                    recovery["inner_return_error_rms"]
                    < dynamic["inner_return_error_rms"]
                ),
                "recovery_smaller_normal_response_magnitude": bool(
                    recovery["normal_response_magnitude_rms"]
                    < dynamic["normal_response_magnitude_rms"]
                ),
                "dynamic_absolute_inner_flow_fidelity": bool(
                    dynamic["normalized_inner_flow_defect"]
                    <= config.max_normalized_inner_flow_defect
                ),
                "dynamic_clean_trace_accuracy": bool(
                    dynamic["clean_lifted_error_rms"] <= config.max_clean_lifted_error
                ),
            }
        values = {
            regime: float(
                lookup[(seed, regime, "DYN")][
                    "learned_clean_reference_local_secant_log_gain_mean"
                ]
            )
            for regime in ("C", "M", "E")
        }
        local_secant_checks[str(seed)] = {
            "values": values,
            "ordered": bool(values["C"] < values["M"] < values["E"]),
            "contractive_negative": bool(values["C"] < -0.5),
            "mixed_near_zero": bool(abs(values["M"]) <= 0.5),
            "expansive_positive": bool(values["E"] > 0.5),
        }
        clean_values = {
            regime: float(lookup[(seed, regime, "DYN")]["clean_lifted_error_rms"])
            for regime in ("C", "M", "E")
        }
        smallest_clean_error = min(clean_values.values())
        largest_clean_error = max(clean_values.values())
        clean_error_ratio = (
            1.0
            if largest_clean_error == 0.0
            else largest_clean_error / max(smallest_clean_error, 1.0e-15)
        )
        clean_trace_checks[str(seed)] = {
            "values": clean_values,
            "largest_to_smallest_ratio": float(clean_error_ratio),
            "absolute_threshold_pass": bool(
                largest_clean_error <= config.max_clean_lifted_error
            ),
            "across_regime_comparability_pass": bool(
                clean_error_ratio <= config.max_dyn_clean_error_ratio
            ),
        }
    p1 = all(
        all(all(checks.values()) for checks in by_regime.values())
        for by_regime in target_checks.values()
    ) and all(
        checks["absolute_threshold_pass"] and checks["across_regime_comparability_pass"]
        for checks in clean_trace_checks.values()
    )
    p2 = all(
        all(value for key, value in checks.items() if key != "values")
        for checks in local_secant_checks.values()
    )
    checks = {
        "target_checks": target_checks,
        "clean_trace_checks": clean_trace_checks,
        "local_secant_checks": local_secant_checks,
        "P1_target_realization": p1,
        "P2_response_regime_ordering": p2,
        "response_qualification_pass": bool(p1 and p2),
    }
    return metric_rows, query_rows, profile_rows, checks


def forcing_bank(
    config: NonlinearStressConfig, sigma: float
) -> tuple[np.ndarray, np.ndarray, str]:
    config.validated()
    if sigma not in config.noise_sigmas:
        raise ValueError("sigma is not registered")
    phase = heldout_phases(config)[:: config.train_phases // config.noise_phases]
    if phase.size != config.noise_phases:
        raise AssertionError("noise phase selection has the wrong size")
    initial_phase = np.repeat(phase, config.noise_sequences)
    initial = np.column_stack((initial_phase, np.zeros_like(initial_phase)))
    rng = np.random.default_rng(config.forcing_seed)
    half = config.noise_sequences // 2
    base = rng.choice(
        np.asarray((-1.0, 1.0)),
        size=(config.noise_steps, config.noise_phases, half),
    )
    paired = np.concatenate((base, -base), axis=2)
    forcing = np.zeros((config.noise_steps, initial.shape[0], 2), dtype=np.float64)
    forcing[..., 1] = sigma * paired.reshape(config.noise_steps, -1)
    return initial, forcing, digest_array(paired)


def rollout_with_forcing(
    step: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    forcing: np.ndarray,
) -> np.ndarray:
    initial_array = _finite_state(initial, name="initial")
    forcing_array = np.asarray(forcing, dtype=np.float64)
    if forcing_array.ndim != 3 or forcing_array.shape[1:] != initial_array.shape:
        raise ValueError("forcing must have shape [steps, batch, 2]")
    if not np.isfinite(forcing_array).all():
        raise ValueError("forcing contains non-finite values")
    values = np.full(
        (forcing_array.shape[0] + 1, *initial_array.shape), np.nan, dtype=np.float64
    )
    values[0] = initial_array
    active = np.ones(initial_array.shape[0], dtype=bool)
    for index in range(forcing_array.shape[0]):
        if np.any(active):
            current = values[index, active]
            proposal = np.asarray(step(current), dtype=np.float64)
            if proposal.shape != current.shape:
                raise ValueError("step changed the state shape")
            values[index + 1, active] = proposal + forcing_array[index, active]
        active &= np.isfinite(values[index + 1]).all(axis=1)
    return values


def _first_exit_steps(radius: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(radius, dtype=np.float64)
    if values.ndim != 2 or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("invalid first-exit inputs")
    crossed = np.abs(values) > threshold
    has_exit = np.any(crossed, axis=0)
    first = np.argmax(crossed, axis=0)
    return np.where(has_exit, first, values.shape[0]).astype(np.int64)


def _first_nonfinite_steps(trajectory: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("trajectory must have shape [steps, batch, 2]")
    nonfinite = ~np.isfinite(values).all(axis=2)
    has_nonfinite = np.any(nonfinite, axis=0)
    first = np.argmax(nonfinite, axis=0)
    return np.where(has_nonfinite, first, values.shape[0]).astype(np.int64)


def _trajectory_summary(
    predicted: np.ndarray,
    clean_reference: np.ndarray,
    forced_reference: np.ndarray,
    *,
    initial: np.ndarray,
    sequences_per_phase: int,
    tube_radius: float,
    safety_radius: float,
) -> tuple[list[dict[str, float]], dict[str, float], list[dict[str, float]]]:
    if (
        predicted.shape != clean_reference.shape
        or predicted.shape != forced_reference.shape
    ):
        raise ValueError("trajectory arrays must have identical shapes")
    initial_array = _finite_state(initial, name="initial")
    if initial_array.shape != predicted.shape[1:]:
        raise ValueError("initial state shape does not match trajectory batch shape")
    if sequences_per_phase < 1:
        raise ValueError("sequences_per_phase must be positive")
    clean_phase_error = predicted[..., 0] - clean_reference[..., 0]
    clean_normal_error = predicted[..., 1] - clean_reference[..., 1]
    clean_error = np.sqrt(clean_phase_error**2 + clean_normal_error**2)
    forced_phase_error = predicted[..., 0] - forced_reference[..., 0]
    forced_normal_error = predicted[..., 1] - forced_reference[..., 1]
    forced_error = np.sqrt(forced_phase_error**2 + forced_normal_error**2)
    radius = np.abs(predicted[..., 1])
    finite_mask = np.isfinite(predicted).all(axis=2)
    tube_crossings = _first_exit_steps(np.where(finite_mask, radius, 0.0), tube_radius)
    safety_crossings = _first_exit_steps(
        np.where(finite_mask, radius, 0.0), safety_radius
    )
    nonfinite_steps = _first_nonfinite_steps(predicted)
    exits = np.minimum(tube_crossings, nonfinite_steps)
    safety_exits = np.minimum(safety_crossings, nonfinite_steps)
    horizon = predicted.shape[0] - 1
    residence_steps = np.clip(exits - 1, 0, horizon)
    residence_fraction = residence_steps / horizon
    curves: list[dict[str, float]] = []
    for step in range(predicted.shape[0]):
        step_mask = finite_mask[step]
        clean_lifted_rms, _ = _masked_rms(clean_error[step], step_mask)
        forced_lifted_rms, _ = _masked_rms(forced_error[step], step_mask)
        clean_phase_rms, _ = _masked_rms(clean_phase_error[step], step_mask)
        tube_distance_rms, _ = _masked_rms(radius[step], step_mask)
        curves.append(
            {
                "step": float(step),
                "clean_lifted_error_rms": clean_lifted_rms,
                "forced_truth_lifted_error_rms": forced_lifted_rms,
                "clean_phase_unwrapped_error_rms": clean_phase_rms,
                "tube_distance_rms": tube_distance_rms,
                "finite_fraction": float(np.mean(step_mask)),
                "nonfinite_fraction": float(np.mean(~step_mask)),
                "tube_survival": float(np.mean(exits > step)),
                "safety_survival": float(np.mean(safety_exits > step)),
            }
        )
    time = np.arange(predicted.shape[0])[:, None]
    post_initial_mask = finite_mask & (time >= 1)
    pre_exit_mask = post_initial_mask & (time < exits[None, :])
    pre_exit_clean_lifted_rms, pre_exit_sample_coverage = _masked_rms(
        clean_error, pre_exit_mask
    )
    pre_exit_forced_lifted_rms, _ = _masked_rms(forced_error, pre_exit_mask)
    pre_exit_clean_phase_rms, _ = _masked_rms(clean_phase_error, pre_exit_mask)
    max_tube_distance = (
        float(np.max(radius[finite_mask])) if np.any(finite_mask) else 0.0
    )
    summary = {
        "clean_lifted_error_auc": float(
            np.mean([row["clean_lifted_error_rms"] for row in curves[1:]])
        ),
        "forced_truth_lifted_error_auc": float(
            np.mean([row["forced_truth_lifted_error_rms"] for row in curves[1:]])
        ),
        "clean_phase_unwrapped_error_auc": float(
            np.mean([row["clean_phase_unwrapped_error_rms"] for row in curves[1:]])
        ),
        "pre_exit_clean_lifted_error_rms": pre_exit_clean_lifted_rms,
        "pre_exit_forced_truth_lifted_error_rms": pre_exit_forced_lifted_rms,
        "pre_exit_clean_phase_unwrapped_error_rms": pre_exit_clean_phase_rms,
        "pre_exit_sample_coverage": pre_exit_sample_coverage,
        "pre_exit_trajectory_coverage": float(np.mean(exits > 1)),
        "final_tube_survival": float(np.mean(exits >= predicted.shape[0])),
        "final_safety_survival": float(np.mean(safety_exits >= predicted.shape[0])),
        "final_nonfinite_fraction": float(
            np.mean(nonfinite_steps < predicted.shape[0])
        ),
        "median_first_nonfinite_step": float(np.median(nonfinite_steps)),
        "median_first_exit_step": float(np.median(exits)),
        "mean_tube_residence_fraction": float(np.mean(residence_fraction)),
        "mean_finite_fraction": float(
            np.mean([row["finite_fraction"] for row in curves[1:]])
        ),
        "max_tube_distance": max_tube_distance,
    }
    unit_rows: list[dict[str, float]] = []
    for index in range(predicted.shape[1]):
        unit_post_initial = post_initial_mask[:, index]
        unit_pre_exit = pre_exit_mask[:, index]
        clean_auc, _ = _masked_mean(clean_error[:, index], unit_post_initial)
        forced_auc, _ = _masked_mean(forced_error[:, index], unit_post_initial)
        phase_auc, _ = _masked_mean(
            np.abs(clean_phase_error[:, index]), unit_post_initial
        )
        pre_exit_clean, pre_exit_coverage = _masked_rms(
            clean_error[:, index], unit_pre_exit
        )
        pre_exit_forced, _ = _masked_rms(forced_error[:, index], unit_pre_exit)
        pre_exit_phase, _ = _masked_rms(clean_phase_error[:, index], unit_pre_exit)
        unit_radius = radius[:, index]
        finite_radius = unit_radius[np.isfinite(unit_radius)]
        unit_rows.append(
            {
                "trajectory_index": index,
                "phase_index": index // sequences_per_phase,
                "forcing_sequence_index": index % sequences_per_phase,
                "initial_theta": float(initial_array[index, 0]),
                "initial_radius": float(initial_array[index, 1]),
                "clean_lifted_error_auc": clean_auc,
                "forced_truth_lifted_error_auc": forced_auc,
                "clean_phase_unwrapped_error_auc": phase_auc,
                "pre_exit_clean_lifted_error_rms": pre_exit_clean,
                "pre_exit_forced_truth_lifted_error_rms": pre_exit_forced,
                "pre_exit_clean_phase_unwrapped_error_rms": pre_exit_phase,
                "pre_exit_sample_coverage": pre_exit_coverage,
                "first_tube_crossing_step": int(tube_crossings[index]),
                "first_exit_step": int(exits[index]),
                "first_safety_exit_step": int(safety_exits[index]),
                "first_nonfinite_step": int(nonfinite_steps[index]),
                "tube_residence_steps": int(residence_steps[index]),
                "tube_residence_fraction": float(residence_fraction[index]),
                "max_tube_distance": (
                    float(np.max(finite_radius)) if finite_radius.size else 0.0
                ),
            }
        )
    return curves, summary, unit_rows


def _common_prefix_errors(
    first: np.ndarray,
    second: np.ndarray,
    clean_reference: np.ndarray,
    forced_reference: np.ndarray,
    *,
    tube_radius: float,
    minimum_steps: int,
) -> dict[str, Any]:
    if (
        first.shape != second.shape
        or first.shape != clean_reference.shape
        or first.shape != forced_reference.shape
    ):
        raise ValueError("common-prefix arrays must have identical shapes")
    if first.ndim != 3 or first.shape[-1] != 2 or first.shape[0] < 2:
        raise ValueError("common-prefix arrays must have shape [steps, batch, 2]")
    if minimum_steps < 1 or minimum_steps > first.shape[0] - 1:
        raise ValueError("minimum common-prefix steps must lie within the rollout")
    first_finite = np.isfinite(first).all(axis=2)
    second_finite = np.isfinite(second).all(axis=2)
    truth_finite = np.isfinite(forced_reference).all(axis=2)
    first_exit = np.minimum(
        _first_exit_steps(
            np.where(first_finite, np.abs(first[..., 1]), 0.0), tube_radius
        ),
        _first_nonfinite_steps(first),
    )
    second_exit = np.minimum(
        _first_exit_steps(
            np.where(second_finite, np.abs(second[..., 1]), 0.0), tube_radius
        ),
        _first_nonfinite_steps(second),
    )
    truth_exit = np.minimum(
        _first_exit_steps(
            np.where(truth_finite, np.abs(forced_reference[..., 1]), 0.0), tube_radius
        ),
        _first_nonfinite_steps(forced_reference),
    )
    common_exit = np.minimum(np.minimum(first_exit, second_exit), truth_exit)
    time = np.arange(first.shape[0])[:, None]
    mask = (
        (time >= 1)
        & (time < common_exit[None, :])
        & first_finite
        & second_finite
        & truth_finite
    )
    first_clean_error = np.sqrt(
        (first[..., 0] - clean_reference[..., 0]) ** 2
        + (first[..., 1] - clean_reference[..., 1]) ** 2
    )
    first_forced_error = np.sqrt(
        (first[..., 0] - forced_reference[..., 0]) ** 2
        + (first[..., 1] - forced_reference[..., 1]) ** 2
    )
    second_clean_error = np.sqrt(
        (second[..., 0] - clean_reference[..., 0]) ** 2
        + (second[..., 1] - clean_reference[..., 1]) ** 2
    )
    second_forced_error = np.sqrt(
        (second[..., 0] - forced_reference[..., 0]) ** 2
        + (second[..., 1] - forced_reference[..., 1]) ** 2
    )
    horizon = first.shape[0] - 1
    prefix_steps = np.clip(common_exit - 1, 0, horizon)
    unit_rows: list[dict[str, Any]] = []
    for index in range(first.shape[1]):
        unit_mask = mask[:, index]
        row: dict[str, Any] = {
            "trajectory_index": index,
            "first_dyn_exit_step": int(first_exit[index]),
            "first_recovery_exit_step": int(second_exit[index]),
            "first_forced_reference_exit_step": int(truth_exit[index]),
            "common_prefix_steps": int(prefix_steps[index]),
            "common_prefix_fraction": float(prefix_steps[index] / horizon),
        }
        if np.any(unit_mask):
            row.update(
                {
                    "dyn_clean_error": _rms(first_clean_error[:, index][unit_mask]),
                    "dyn_forced_error": _rms(first_forced_error[:, index][unit_mask]),
                    "recovery_clean_error": _rms(
                        second_clean_error[:, index][unit_mask]
                    ),
                    "recovery_forced_error": _rms(
                        second_forced_error[:, index][unit_mask]
                    ),
                }
            )
        else:
            row.update(
                {
                    "dyn_clean_error": None,
                    "dyn_forced_error": None,
                    "recovery_clean_error": None,
                    "recovery_forced_error": None,
                }
            )
        unit_rows.append(row)

    eligible = [row for row in unit_rows if int(row["common_prefix_steps"]) > 0]
    if not eligible:
        aggregate = {
            "first_clean_error": None,
            "first_forced_error": None,
            "second_clean_error": None,
            "second_forced_error": None,
        }
    else:
        aggregate = {
            "first_clean_error": float(
                np.mean([float(row["dyn_clean_error"]) for row in eligible])
            ),
            "first_forced_error": float(
                np.mean([float(row["dyn_forced_error"]) for row in eligible])
            ),
            "second_clean_error": float(
                np.mean([float(row["recovery_clean_error"]) for row in eligible])
            ),
            "second_forced_error": float(
                np.mean([float(row["recovery_forced_error"]) for row in eligible])
            ),
        }
    median_steps = float(np.median(prefix_steps))
    return {
        **aggregate,
        "common_prefix_fraction": float(np.mean(prefix_steps / horizon)),
        "common_prefix_trajectory_coverage": float(len(eligible) / len(unit_rows)),
        "common_prefix_median_steps": median_steps,
        "common_prefix_required_coverage_pass": bool(median_steps >= minimum_steps),
        "unit_rows": unit_rows,
    }


def evaluate_forced_rollouts(
    models: Mapping[int, Mapping[str, ResidualMLP]],
    config: NonlinearStressConfig,
    *,
    training_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config.validated()
    flow_by_key = flows(config)
    clean_flow = flow_by_key["C"]
    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    common_prefix_rows: list[dict[str, Any]] = []
    pairwise_prefix_rows: list[dict[str, Any]] = []
    forcing_digests: dict[str, str] = {}
    checkpoint_lookup = {
        (int(row["seed"]), str(row["arm"])): {
            "checkpoint": str(row["checkpoint"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "parameter_digest": str(row["parameter_digest"]),
        }
        for row in training_rows
    }

    for sigma in config.noise_sigmas:
        initial, forcing, sign_digest = forcing_bank(config, sigma)
        forcing_digests[str(sigma)] = sign_digest
        clean_reference = rollout_with_forcing(
            clean_flow.advance, initial, np.zeros_like(forcing)
        )
        shared_predictions: dict[tuple[int, str], np.ndarray] = {}
        for seed in config.seeds:
            for arm in ("CLEAN", "RECOVERY"):
                model = models[seed][arm]
                shared_predictions[(seed, arm)] = rollout_with_forcing(
                    lambda value, current=model: model_step(
                        current, value, allow_nonfinite=True
                    ),
                    initial,
                    forcing,
                )

        for regime_key, trusted_flow in flow_by_key.items():
            forced_reference = rollout_with_forcing(
                trusted_flow.advance, initial, forcing
            )
            oracle_curves, oracle_summary, oracle_units = _trajectory_summary(
                forced_reference,
                clean_reference,
                forced_reference,
                initial=initial,
                sequences_per_phase=config.noise_sequences,
                tube_radius=config.tube_radius,
                safety_radius=config.safety_radius,
            )
            for row in oracle_curves:
                curve_rows.append(
                    {
                        "seed": -1,
                        "regime": regime_key,
                        "arm": "TRUSTED_ORACLE",
                        "sigma": sigma,
                        **row,
                    }
                )
            endpoint_rows.append(
                {
                    "seed": -1,
                    "regime": regime_key,
                    "arm": "TRUSTED_ORACLE",
                    "sigma": sigma,
                    "source_arm": "",
                    "checkpoint": "",
                    "checkpoint_sha256": "",
                    "parameter_digest": "",
                    "trajectory_digest": digest_array(forced_reference),
                    "clean_reference_digest": digest_array(clean_reference),
                    "forced_reference_digest": digest_array(forced_reference),
                    **oracle_summary,
                }
            )
            for row in oracle_units:
                unit_rows.append(
                    {
                        "seed": -1,
                        "regime": regime_key,
                        "arm": "TRUSTED_ORACLE",
                        "sigma": sigma,
                        "source_arm": "",
                        "checkpoint": "",
                        "checkpoint_sha256": "",
                        "parameter_digest": "",
                        "trajectory_digest": digest_array(forced_reference),
                        "clean_reference_digest": digest_array(clean_reference),
                        "forced_reference_digest": digest_array(forced_reference),
                        **row,
                    }
                )

            for seed in config.seeds:
                trajectories = {
                    "CLEAN": shared_predictions[(seed, "CLEAN")],
                    "RECOVERY": shared_predictions[(seed, "RECOVERY")],
                }
                dynamic_model = models[seed][f"DYN_{regime_key}"]
                trajectories["DYN"] = rollout_with_forcing(
                    lambda value, current=dynamic_model: model_step(
                        current, value, allow_nonfinite=True
                    ),
                    initial,
                    forcing,
                )
                for arm, predicted in trajectories.items():
                    source_arm = arm if arm != "DYN" else f"DYN_{regime_key}"
                    if (seed, source_arm) not in checkpoint_lookup:
                        raise KeyError(
                            f"missing checkpoint metadata for {seed}/{source_arm}"
                        )
                    curves, summary, units = _trajectory_summary(
                        predicted,
                        clean_reference,
                        forced_reference,
                        initial=initial,
                        sequences_per_phase=config.noise_sequences,
                        tube_radius=config.tube_radius,
                        safety_radius=config.safety_radius,
                    )
                    for row in curves:
                        curve_rows.append(
                            {
                                "seed": seed,
                                "regime": regime_key,
                                "arm": arm,
                                "sigma": sigma,
                                "source_arm": source_arm,
                                **row,
                            }
                        )
                    endpoint_rows.append(
                        {
                            "seed": seed,
                            "regime": regime_key,
                            "arm": arm,
                            "sigma": sigma,
                            "source_arm": source_arm,
                            **checkpoint_lookup[(seed, source_arm)],
                            "trajectory_digest": digest_array(predicted),
                            "clean_reference_digest": digest_array(clean_reference),
                            "forced_reference_digest": digest_array(forced_reference),
                            **summary,
                        }
                    )
                    for row in units:
                        unit_rows.append(
                            {
                                "seed": seed,
                                "regime": regime_key,
                                "arm": arm,
                                "sigma": sigma,
                                "source_arm": source_arm,
                                **checkpoint_lookup[(seed, source_arm)],
                                "trajectory_digest": digest_array(predicted),
                                "clean_reference_digest": digest_array(clean_reference),
                                "forced_reference_digest": digest_array(
                                    forced_reference
                                ),
                                **row,
                            }
                        )
                prefix = _common_prefix_errors(
                    trajectories["DYN"],
                    trajectories["RECOVERY"],
                    clean_reference,
                    forced_reference,
                    tube_radius=config.tube_radius,
                    minimum_steps=config.min_common_prefix_steps,
                )
                for row in prefix["unit_rows"]:
                    trajectory_index = int(row["trajectory_index"])
                    pairwise_prefix_rows.append(
                        {
                            "seed": seed,
                            "regime": regime_key,
                            "sigma": sigma,
                            "phase_index": (trajectory_index // config.noise_sequences),
                            "forcing_sequence_index": (
                                trajectory_index % config.noise_sequences
                            ),
                            "initial_theta": float(initial[trajectory_index, 0]),
                            "dyn_trajectory_digest": digest_array(trajectories["DYN"]),
                            "recovery_trajectory_digest": digest_array(
                                trajectories["RECOVERY"]
                            ),
                            "forced_reference_digest": digest_array(forced_reference),
                            "dyn_checkpoint_sha256": checkpoint_lookup[
                                (seed, f"DYN_{regime_key}")
                            ]["checkpoint_sha256"],
                            "recovery_checkpoint_sha256": checkpoint_lookup[
                                (seed, "RECOVERY")
                            ]["checkpoint_sha256"],
                            **row,
                        }
                    )
                common_prefix_rows.append(
                    {
                        "seed": seed,
                        "regime": regime_key,
                        "sigma": sigma,
                        "dyn_trajectory_digest": digest_array(trajectories["DYN"]),
                        "recovery_trajectory_digest": digest_array(
                            trajectories["RECOVERY"]
                        ),
                        "forced_reference_digest": digest_array(forced_reference),
                        "dyn_checkpoint_sha256": checkpoint_lookup[
                            (seed, f"DYN_{regime_key}")
                        ]["checkpoint_sha256"],
                        "recovery_checkpoint_sha256": checkpoint_lookup[
                            (seed, "RECOVERY")
                        ]["checkpoint_sha256"],
                        "dyn_clean_error": prefix["first_clean_error"],
                        "dyn_forced_error": prefix["first_forced_error"],
                        "recovery_clean_error": prefix["second_clean_error"],
                        "recovery_forced_error": prefix["second_forced_error"],
                        "common_prefix_fraction": prefix["common_prefix_fraction"],
                        "common_prefix_trajectory_coverage": prefix[
                            "common_prefix_trajectory_coverage"
                        ],
                        "common_prefix_median_steps": prefix[
                            "common_prefix_median_steps"
                        ],
                        "common_prefix_required_coverage_pass": prefix[
                            "common_prefix_required_coverage_pass"
                        ],
                    }
                )

    lookup: dict[tuple[int, str, str, float], Mapping[str, Any]] = {
        (
            int(row["seed"]),
            str(row["regime"]),
            str(row["arm"]),
            float(row["sigma"]),
        ): row
        for row in endpoint_rows
        if int(row["seed"]) >= 0
    }
    sigma = config.primary_noise_sigma
    dynamic_aggregate: dict[str, dict[str, float]] = {}
    for regime in ("C", "M", "E"):
        endpoint_subset = [
            lookup[(seed, regime, "DYN", sigma)] for seed in config.seeds
        ]
        unit_subset = [
            row
            for row in unit_rows
            if int(row["seed"]) >= 0
            and row["regime"] == regime
            and row["arm"] == "DYN"
            and float(row["sigma"]) == sigma
        ]
        dynamic_aggregate[regime] = {
            "mean_tube_residence_fraction": float(
                np.mean([row["tube_residence_fraction"] for row in unit_subset])
            ),
            "survival": float(
                np.mean([row["final_tube_survival"] for row in endpoint_subset])
            ),
            "final_nonfinite_fraction": float(
                np.mean([row["final_nonfinite_fraction"] for row in endpoint_subset])
            ),
        }
    recovery_rows = [
        row
        for row in unit_rows
        if int(row["seed"]) >= 0
        and row["regime"] == "E"
        and row["arm"] == "RECOVERY"
        and float(row["sigma"]) == sigma
    ]
    recovery_endpoints = [
        lookup[(seed, "E", "RECOVERY", sigma)] for seed in config.seeds
    ]
    recovery_residence = float(
        np.mean([row["tube_residence_fraction"] for row in recovery_rows])
    )
    recovery_survival = float(
        np.mean([row["final_tube_survival"] for row in recovery_endpoints])
    )
    residence_order = (
        dynamic_aggregate["C"]["mean_tube_residence_fraction"]
        >= dynamic_aggregate["M"]["mean_tube_residence_fraction"]
        >= dynamic_aggregate["E"]["mean_tube_residence_fraction"]
    )
    residence_separation = (
        dynamic_aggregate["C"]["mean_tube_residence_fraction"]
        - dynamic_aggregate["E"]["mean_tube_residence_fraction"]
        >= 0.20
    )
    dyn_interaction = bool(residence_order and residence_separation)
    recovery_beats_expansive = bool(
        recovery_residence - dynamic_aggregate["E"]["mean_tube_residence_fraction"]
        >= 0.20
    )
    primary_prefix = [
        row
        for row in pairwise_prefix_rows
        if row["regime"] == "E" and float(row["sigma"]) == sigma
    ]
    primary_eligible = [
        row for row in primary_prefix if int(row["common_prefix_steps"]) > 0
    ]
    primary_median_steps = float(
        np.median([row["common_prefix_steps"] for row in primary_prefix])
    )
    p4 = bool(
        primary_eligible
        and primary_median_steps >= config.min_common_prefix_steps
        and np.mean([row["dyn_forced_error"] for row in primary_eligible])
        < np.mean([row["recovery_forced_error"] for row in primary_eligible])
        and np.mean([row["recovery_clean_error"] for row in primary_eligible])
        < np.mean([row["dyn_clean_error"] for row in primary_eligible])
    )
    p5_by_sigma: dict[str, bool] = {}
    for forcing_sigma in config.noise_sigmas:
        if forcing_sigma == 0.0:
            continue
        contractive_rows = [
            lookup[(seed, "C", "DYN", forcing_sigma)] for seed in config.seeds
        ]
        contractive_prefix = [
            row
            for row in pairwise_prefix_rows
            if row["regime"] == "C" and float(row["sigma"]) == forcing_sigma
        ]
        contractive_eligible = [
            row for row in contractive_prefix if int(row["common_prefix_steps"]) > 0
        ]
        p5_by_sigma[str(forcing_sigma)] = bool(
            contractive_eligible
            and float(np.mean([row["final_tube_survival"] for row in contractive_rows]))
            >= 0.80
            and np.mean([row["dyn_forced_error"] for row in contractive_eligible])
            < np.mean([row["recovery_forced_error"] for row in contractive_eligible])
        )
    p5 = bool(p5_by_sigma and all(p5_by_sigma.values()))
    shared_arm_checks: dict[str, Any] = {}
    for forcing_sigma in config.noise_sigmas:
        sigma_rows = [
            row
            for row in endpoint_rows
            if int(row["seed"]) >= 0 and float(row["sigma"]) == forcing_sigma
        ]
        for seed in config.seeds:
            for arm in ("CLEAN", "RECOVERY"):
                selected = [
                    row
                    for row in sigma_rows
                    if int(row["seed"]) == seed and str(row["arm"]) == arm
                ]
                if len(selected) != len(REGIMES):
                    raise AssertionError(
                        "shared-arm rollout rows do not cover every regime label"
                    )
                trajectory_digests = sorted(
                    {str(row["trajectory_digest"]) for row in selected}
                )
                checkpoint_digests = sorted(
                    {str(row["checkpoint_sha256"]) for row in selected}
                )
                parameter_digests = sorted(
                    {str(row["parameter_digest"]) for row in selected}
                )
                shared_arm_checks[f"{seed}:{forcing_sigma}:{arm}"] = {
                    "trajectory_digests": trajectory_digests,
                    "checkpoint_sha256s": checkpoint_digests,
                    "parameter_digests": parameter_digests,
                    "trajectory_digest_reused": len(trajectory_digests) == 1,
                    "checkpoint_sha256_reused": len(checkpoint_digests) == 1,
                    "parameter_digest_reused": len(parameter_digests) == 1,
                }
    checks = {
        "forcing_sign_digests": forcing_digests,
        "primary_sigma": sigma,
        "dynamic_aggregate": dynamic_aggregate,
        "recovery_expansive": {
            "mean_tube_residence_fraction": recovery_residence,
            "survival": recovery_survival,
        },
        "P3_metric": "mean_tube_residence_fraction",
        "shared_arm_checks": shared_arm_checks,
        "P3_dynamic_regime_interaction": dyn_interaction,
        "P3_recovery_beats_dynamic_expansive": recovery_beats_expansive,
        "P4_common_prefix_median_steps": primary_median_steps,
        "P4_common_prefix_eligible_fraction": float(
            len(primary_eligible) / len(primary_prefix)
        ),
        "P4_dual_target_tradeoff": p4,
        "P5_by_sigma": p5_by_sigma,
        "P5_contractive_control": p5,
    }
    checks["rollout_predictions_pass"] = all(
        checks[name]
        for name in (
            "P3_dynamic_regime_interaction",
            "P3_recovery_beats_dynamic_expansive",
            "P4_dual_target_tradeoff",
            "P5_contractive_control",
        )
    )
    checks["common_prefix_rows"] = common_prefix_rows
    return curve_rows, endpoint_rows, unit_rows, pairwise_prefix_rows, checks


def registered_predictions() -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "outcome_blind": True,
        "primary_noise_sigma": 5.0e-3,
        "predictions": {
            "P1": "DYN realizes trusted in-tube response; RECOVERY realizes return.",
            "P2": (
                "DYN 32-step clean-reference local-secant log-gain is ordered "
                "C < M < E with registered signs."
            ),
            "P3": "DYN clean-path robustness degrades across C, M, E; RECOVERY beats DYN-E.",
            "P4": "DYN-E is closer to same-forcing truth while RECOVERY is closer to clean truth before common exit.",
            "P5": "DYN-C retains at least 0.80 tube survival and its forced-truth advantage.",
        },
        "negative_outcomes_retained": True,
        "ode_to_pde_transfer_claim": False,
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def _plot_summary(
    response_rows: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    rollout_checks: Mapping[str, Any],
    *,
    output_pdf: Path,
    output_png: Path,
    primary_sigma: float,
) -> None:
    import matplotlib.pyplot as plt

    def available_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else math.nan

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    regimes = ("C", "M", "E")
    x = np.arange(3, dtype=np.float64)

    trusted = []
    learned = []
    learned_std = []
    for regime in regimes:
        rows = [
            row
            for row in response_rows
            if row["regime"] == regime and row["arm"] == "DYN"
        ]
        trusted.append(
            float(
                np.mean(
                    [
                        row["trusted_clean_reference_local_secant_log_gain_mean"]
                        for row in rows
                    ]
                )
            )
        )
        values = np.asarray(
            [row["learned_clean_reference_local_secant_log_gain_mean"] for row in rows]
        )
        learned.append(float(np.mean(values)))
        learned_std.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
    axes[0, 0].plot(x, trusted, marker="o", color="black", label="trusted")
    axes[0, 0].errorbar(
        x, learned, yerr=learned_std, marker="s", capsize=3, label="DYN learned"
    )
    axes[0, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0, 0].set_xticks(x, regimes)
    axes[0, 0].set_ylabel("32-step clean-reference local-secant log-gain")
    axes[0, 0].set_title("(a) Registered response regimes")
    axes[0, 0].legend(frameon=False)

    width = 0.34
    for index, arm in enumerate(("RECOVERY", "DYN")):
        flow_values = []
        return_values = []
        for regime in regimes:
            rows = [
                row
                for row in response_rows
                if row["regime"] == regime and row["arm"] == arm
            ]
            flow_values.append(
                float(np.mean([row["inner_flow_defect_rms"] for row in rows]))
            )
            return_values.append(
                float(np.mean([row["inner_return_error_rms"] for row in rows]))
            )
        offset = (index - 0.5) * width
        axes[0, 1].bar(x + offset, flow_values, width, label=f"{arm}: flow")
        axes[0, 1].scatter(
            x + offset, return_values, marker="x", color="black", zorder=5
        )
    axes[0, 1].set_xticks(x, regimes)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("held-out in-tube RMS error")
    axes[0, 1].set_title("(b) Flow fidelity versus return")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for arm, marker in (("DYN", "o"), ("RECOVERY", "s"), ("TRUSTED_ORACLE", "^")):
        values = []
        for regime in regimes:
            rows = [
                row
                for row in endpoint_rows
                if row["regime"] == regime
                and row["arm"] == arm
                and float(row["sigma"]) == primary_sigma
            ]
            values.append(float(np.mean([row["final_tube_survival"] for row in rows])))
        axes[1, 0].plot(x, values, marker=marker, label=arm)
    axes[1, 0].set_xticks(x, regimes)
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylabel("tube survival at step 192")
    axes[1, 0].set_title(f"(c) Recurrent forcing, $\\sigma={primary_sigma:g}$")
    axes[1, 0].legend(frameon=False, fontsize=8)

    prefix = [
        row
        for row in rollout_checks["common_prefix_rows"]
        if row["regime"] == "E" and float(row["sigma"]) == primary_sigma
    ]
    labels = ("DYN", "RECOVERY")
    clean_values = (
        available_mean(prefix, "dyn_clean_error"),
        available_mean(prefix, "recovery_clean_error"),
    )
    forced_values = (
        available_mean(prefix, "dyn_forced_error"),
        available_mean(prefix, "recovery_forced_error"),
    )
    y = np.arange(2)
    axes[1, 1].bar(y - width / 2, clean_values, width, label="to clean path")
    axes[1, 1].bar(y + width / 2, forced_values, width, label="to forced truth")
    axes[1, 1].set_xticks(y, labels)
    axes[1, 1].set_ylabel("common-prefix lifted RMS error")
    axes[1, 1].set_title("(d) Expansive regime: two targets")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_name in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative_name] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _output_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "closeout.json"}:
            continue
        records[path.relative_to(output_dir).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _compatibility_anchor_specs() -> dict[str, dict[str, Any]]:
    return {
        "affine_packet": {
            "path": AFFINE_PACKET_MANIFEST.as_posix(),
            "expected_manifest_sha256": AFFINE_PACKET_MANIFEST_SHA256,
        },
        "landscape_packet": {
            "path": LANDSCAPE_PACKET_MANIFEST.as_posix(),
            "expected_manifest_sha256": LANDSCAPE_PACKET_MANIFEST_SHA256,
        },
    }


def _compatibility_anchors() -> dict[str, Any]:
    anchors = _compatibility_anchor_specs()
    for record in anchors.values():
        path = REPOSITORY_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        record["actual_manifest_sha256"] = actual
        record["verified"] = actual == record["expected_manifest_sha256"]
        if not record["verified"]:
            raise ValueError(f"compatibility anchor mismatch: {record['path']}")
    return anchors


def _validate_recorded_compatibility_anchors(value: Any) -> dict[str, Any]:
    expected = {
        name: {
            **record,
            "actual_manifest_sha256": record["expected_manifest_sha256"],
            "verified": True,
        }
        for name, record in _compatibility_anchor_specs().items()
    }
    if value != expected:
        raise ValueError("recorded compatibility anchor mismatch")
    return expected


def verify_packet(
    output_dir: str | Path, *, verify_current_sources: bool = True
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("B2-NL packet lacks manifest.json or closeout.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported B2-NL manifest schema")
    if (
        closeout.get("schema") != "corrective_ode_nonlinear_closeout_v1"
        or closeout.get("status") != "complete"
    ):
        raise ValueError("invalid B2-NL closeout record")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("closeout manifest hash mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_FILES):
        raise ValueError("source inventory mismatch")
    recorded_anchors = _validate_recorded_compatibility_anchors(
        manifest.get("compatibility_anchors")
    )
    snapshot_path = root / PROVENANCE_SNAPSHOT_NAME
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"B2-NL packet lacks {PROVENANCE_SNAPSHOT_NAME}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("schema") != PROVENANCE_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported B2-NL provenance snapshot schema")
    if snapshot.get("sources") != sources:
        raise ValueError("provenance snapshot source mismatch")
    if snapshot.get("compatibility_anchors") != recorded_anchors:
        raise ValueError("provenance snapshot compatibility mismatch")
    outputs = manifest.get("outputs")
    actual_outputs = _output_records(root)
    if not isinstance(outputs, Mapping) or set(outputs) != set(actual_outputs):
        raise ValueError("output inventory mismatch")
    for relative_name, record in outputs.items():
        path = root / relative_name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"output hash mismatch: {relative_name}")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"output size mismatch: {relative_name}")
    if verify_current_sources:
        for relative_name, record in sources.items():
            path = REPOSITORY_ROOT / relative_name
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"current source mismatch: {relative_name}")
            if int(path.stat().st_size) != int(record.get("bytes", -1)):
                raise ValueError(f"current source size mismatch: {relative_name}")
        if _compatibility_anchors() != recorded_anchors:
            raise ValueError("current compatibility anchor mismatch")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "outputs": len(manifest.get("outputs", {})),
        "sources": len(manifest.get("sources", {})),
        "current_sources_checked": verify_current_sources,
        "canonical_contract": bool(manifest.get("canonical_contract")),
        "scientific_classification": manifest.get("scientific_classification"),
    }


def classify_scientific_outcome(
    prediction_checks: Mapping[str, bool],
    *,
    response_qualification_pass: bool,
) -> str:
    """Classify rollout claims only after the response mechanism qualifies."""

    response_names = ("P1_target_realization", "P2_response_regime_ordering")
    missing = [name for name in response_names if name not in prediction_checks]
    if missing:
        raise ValueError(f"prediction checks lack response gates: {missing}")
    if not response_qualification_pass:
        return "response_qualification_failed"
    if not all(bool(prediction_checks[name]) for name in response_names):
        raise ValueError("response qualification conflicts with P1/P2 checks")

    rollout_names = tuple(
        name for name in prediction_checks if name not in response_names
    )
    if not rollout_names:
        raise ValueError("prediction checks lack rollout predictions")
    supported_count = sum(bool(prediction_checks[name]) for name in rollout_names)
    if supported_count == len(rollout_names):
        return "all_registered_predictions_supported"
    if supported_count:
        return "registered_predictions_partially_supported"
    return "registered_predictions_falsified"


def run_study(
    *,
    output_dir: str | Path,
    config: NonlinearStressConfig | None = None,
) -> dict[str, Any]:
    study_config = (config or NonlinearStressConfig()).validated()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        root / "config.json",
        {
            "schema": CONFIG_SCHEMA,
            "config": asdict(study_config),
            "regimes": [asdict(regime) for regime in REGIMES],
        },
    )
    atomic_write_json(root / "predictions.json", registered_predictions())
    qualification = qualify_trusted_solver(study_config)
    atomic_write_json(root / "solver_qualification.json", qualification)
    if qualification["status"] != "pass":
        raise AssertionError("trusted nonlinear ODE solver qualification failed")
    source_snapshot = _source_records()
    anchor_snapshot = _compatibility_anchors()
    atomic_write_json(
        root / PROVENANCE_SNAPSHOT_NAME,
        {
            "schema": PROVENANCE_SNAPSHOT_SCHEMA,
            "sources": source_snapshot,
            "compatibility_anchors": anchor_snapshot,
        },
    )

    torch.set_num_threads(1)
    models, training_rows, paired = train_models(
        study_config, checkpoint_dir=root / "checkpoints"
    )
    response_rows, query_rows, profile_rows, response_checks = evaluate_response_bank(
        models, study_config
    )
    (
        curve_rows,
        endpoint_rows,
        unit_rows,
        pairwise_prefix_rows,
        rollout_checks,
    ) = evaluate_forced_rollouts(models, study_config, training_rows=training_rows)
    rollout_checks["interpretation"] = (
        "mechanism_qualified"
        if response_checks["response_qualification_pass"]
        else "descriptive_only_response_qualification_failed"
    )
    prediction_checks = {
        "P1_target_realization": response_checks["P1_target_realization"],
        "P2_response_regime_ordering": response_checks["P2_response_regime_ordering"],
        "P3_dynamic_regime_interaction": rollout_checks[
            "P3_dynamic_regime_interaction"
        ],
        "P3_recovery_beats_dynamic_expansive": rollout_checks[
            "P3_recovery_beats_dynamic_expansive"
        ],
        "P4_dual_target_tradeoff": rollout_checks["P4_dual_target_tradeoff"],
        "P5_contractive_control": rollout_checks["P5_contractive_control"],
    }
    classification = classify_scientific_outcome(
        prediction_checks,
        response_qualification_pass=response_checks["response_qualification_pass"],
    )

    write_csv(root / "training.csv", training_rows)
    write_csv(root / "response_metrics.csv", response_rows)
    write_csv(root / "query_bank_metrics.csv", query_rows)
    write_csv(root / "response_profiles.csv", profile_rows)
    write_csv(root / "rollout_curves.csv", curve_rows)
    write_csv(root / "rollout_endpoints.csv", endpoint_rows)
    write_csv(root / "rollout_units.csv", unit_rows)
    write_csv(root / "common_prefix_units.csv", pairwise_prefix_rows)
    atomic_write_json(
        root / "learned_summary.json",
        {
            "schema": "corrective_ode_nonlinear_learned_v1",
            "response_checks": response_checks,
            "rollout_checks": rollout_checks,
            "paired_contract": paired,
            "response_aggregate": {
                f"{regime}_{arm}": _aggregate_rows(
                    [
                        row
                        for row in response_rows
                        if row["regime"] == regime and row["arm"] == arm
                    ],
                    (
                        "clean_lifted_error_rms",
                        "inner_flow_defect_rms",
                        "normalized_inner_flow_defect",
                        "inner_return_error_rms",
                        "trusted_response_defect_rms",
                        "normal_response_magnitude_rms",
                        "learned_clean_reference_local_secant_log_gain_mean",
                    ),
                )
                for regime in ("C", "M", "E")
                for arm in ("CLEAN", "RECOVERY", "DYN")
            },
        },
    )
    summary = {
        "schema": STUDY_SCHEMA,
        "status": "complete",
        "scientific_classification": classification,
        "prediction_checks": prediction_checks,
        "response_qualification_pass": response_checks["response_qualification_pass"],
        "scope": "supporting_nonlinear_ode_mechanism_stress_only",
        "ode_to_pde_transfer_claim": False,
    }
    atomic_write_json(root / "summary.json", summary)
    _plot_summary(
        response_rows,
        endpoint_rows,
        rollout_checks,
        output_pdf=root / "nonlinear_ode_regime_stress.pdf",
        output_png=root / "nonlinear_ode_regime_stress.png",
        primary_sigma=study_config.primary_noise_sigma,
    )
    if _source_records() != source_snapshot:
        raise ValueError("B2-NL sources changed during execution")
    if _compatibility_anchors() != anchor_snapshot:
        raise ValueError("B2-NL compatibility anchors changed during execution")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study_schema": STUDY_SCHEMA,
        "canonical_contract": asdict(study_config) == asdict(NonlinearStressConfig()),
        "scientific_classification": classification,
        "git": git_state(),
        "compatibility_anchors": anchor_snapshot,
        "sources": source_snapshot,
        "outputs": _output_records(root),
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(
        root / "closeout.json",
        {
            "schema": "corrective_ode_nonlinear_closeout_v1",
            "status": "complete",
            "manifest_sha256": sha256_file(root / "manifest.json"),
        },
    )
    verification = verify_packet(root)
    return {
        "output_dir": str(root),
        "summary": summary,
        "verification": verification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the bounded B2-NL nonlinear ODE stress packet. "
            "The command accesses no PDE data, model, solver, or checkpoint."
        )
    )
    parser.add_argument("--mode", choices=("run", "verify"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        result = run_study(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

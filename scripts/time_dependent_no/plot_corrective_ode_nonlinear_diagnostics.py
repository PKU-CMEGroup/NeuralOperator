"""Render read-only diagnostics from the frozen nonlinear ODE packet.

This script deliberately does not redefine, retrain, or repair B2-NL.  It first
verifies the exact parent manifest and all current parent sources, reloads all
15 frozen checkpoints, and then derives state-resolved visual diagnostics:

* trusted-flow defects and clean-return errors on a common ``theta-r`` grid;
* one prospectively fixed forced trajectory and its visited one-step defects.

The parent used four fixed signed radii rather than Gaussian input noise.  The
derived figures therefore provide intuition about that old protocol only.
They are not evidence for a Gaussian-noise intervention, are not online OOD
detectors, and are not paper evidence unless separately promoted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    CONFIG_SCHEMA,
    NonlinearStressConfig,
    ResidualMLP,
    build_training_contract,
    flows,
    forcing_bank,
    lifted_coordinate_error,
    model_step,
    rollout_with_forcing,
)
from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    verify_packet as verify_parent_packet,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    digest_array,
    git_state,
    sha256_file,
    write_csv,
)

PARENT_PACKET = Path(
    "artifacts/time_dependent_no/corrective_ode_nonlinear_stress_20260830a"
)
PARENT_MANIFEST_SHA256 = (
    "8c1458d44dedb5dfa170d7e2acd7b84efb74541c2d87eaa49e0276373abf9f40"
)
EXPECTED_CHECKPOINT_SHA256 = {
    "checkpoints/seed_17_clean.pt": (
        "b13301b839b318c0506c74a78f42c6e9e157574466570366aab93d6af7ceba6e"
    ),
    "checkpoints/seed_17_dyn_c.pt": (
        "bd98ae54dbbfd77206b848dc7d46d232afdac135dd626aabede72d275b068df3"
    ),
    "checkpoints/seed_17_dyn_e.pt": (
        "1ff804369021861638f927358400d4cdf2894c8b87246cffefe613838c3860d9"
    ),
    "checkpoints/seed_17_dyn_m.pt": (
        "314778551f02cce3ad3f5ac70be86499ee58634ceaf54afedb941207f3983303"
    ),
    "checkpoints/seed_17_recovery.pt": (
        "de4ff7653a5149d07c49da941344705713dff901df9040a26647a3cf1894021a"
    ),
    "checkpoints/seed_29_clean.pt": (
        "835d2e7a196ae26eafc35fbd7780627e5949b8847d695cbb166a71261b087836"
    ),
    "checkpoints/seed_29_dyn_c.pt": (
        "504d28df6f4a8fb1db50c0d95cc664d51045f35453967c228e1a1ef050d6bfc9"
    ),
    "checkpoints/seed_29_dyn_e.pt": (
        "142292b0cfaeb5e94cedfe94f1fb5ece8e2c54bf63a45d35dd062aa2b5b7d1f7"
    ),
    "checkpoints/seed_29_dyn_m.pt": (
        "0483711866192ea245e745c2d8de75523cdc06244cf92d663875e8c975f1fdbf"
    ),
    "checkpoints/seed_29_recovery.pt": (
        "7a68e0efe811a84aedc397800b87bbbb8062be6dc09ed572b8feb20de66c8dc0"
    ),
    "checkpoints/seed_43_clean.pt": (
        "6820681a1ae094b8fd3f9a938e33b9b67fb49d1ff8eaeeb4a3f517e3626cb0ba"
    ),
    "checkpoints/seed_43_dyn_c.pt": (
        "1f7719659c84449c0490c69d813274e70359bba11103a4aa2afba202668644b6"
    ),
    "checkpoints/seed_43_dyn_e.pt": (
        "4dfca555445eff251f277a267348e4b4ea98b5b36b1e570797d2c5b49130924e"
    ),
    "checkpoints/seed_43_dyn_m.pt": (
        "e15cea8f3260615e1d1d160619ce4ff2c8094876381ef20fe70e9caa13c62cf8"
    ),
    "checkpoints/seed_43_recovery.pt": (
        "ed422e9a89bb14a62d58ab810f258955a93c249c4df5e7e2e80cb1a930a74a10"
    ),
}

MANIFEST_SCHEMA = "corrective_ode_nonlinear_diagnostic_manifest_v1"
SUMMARY_SCHEMA = "corrective_ode_nonlinear_diagnostic_summary_v1"
CLOSEOUT_SCHEMA = "corrective_ode_nonlinear_diagnostic_closeout_v1"
SOURCE_FILES = (
    "scripts/time_dependent_no/plot_corrective_ode_nonlinear_diagnostics.py",
    "tests/time_dependent_no/test_corrective_ode_nonlinear_diagnostics.py",
)
REGIME_KEYS = ("C", "M", "E")
ARM_KEYS = ("CLEAN", "RECOVERY", "DYN")
CHECKPOINT_ARMS = ("CLEAN", "RECOVERY", "DYN_C", "DYN_M", "DYN_E")
PAPER_HEATMAP_CMAP = "magma"
PAPER_REFERENCE_COLOR = "#00E5FF"
PAPER_TRUSTED_COLOR = "#009E73"
PAPER_ARM_COLORS = {
    "CLEAN": "#4C78A8",
    "RECOVERY": "#F58518",
    "DYN": "#E45756",
}
TRAJECTORY_SEED = 17
TRAJECTORY_PHASE_INDEX = 0
TRAJECTORY_SEQUENCE_INDEX = 0


@dataclass(frozen=True)
class DiagnosticConfig:
    """Fixed choices for the retrospective visualization grid."""

    theta_points: int = 256
    radius_points: int = 241
    radius_limit: float = 0.30

    def validated(self, parent: NonlinearStressConfig) -> DiagnosticConfig:
        if self.theta_points < 16 or self.radius_points < 17:
            raise ValueError("diagnostic grids are too small")
        if self.radius_points % 2 == 0:
            raise ValueError("radius_points must be odd so r=0 is represented")
        if not math.isfinite(self.radius_limit):
            raise ValueError("radius_limit must be finite")
        if not parent.tube_radius < self.radius_limit <= parent.safety_radius:
            raise ValueError("radius_limit must lie beyond the parent tube and in safety")
        if self.radius_limit < max(abs(value) for value in parent.training_radii):
            raise ValueError("radius_limit does not contain the training support")
        return self


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _parent_config(parent_dir: Path) -> NonlinearStressConfig:
    payload = _json(parent_dir / "config.json")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported nonlinear parent configuration schema")
    values = dict(payload.get("config", {}))
    for key in (
        "training_radii",
        "inner_query_radii",
        "outer_query_radii",
        "seeds",
        "noise_sigmas",
    ):
        values[key] = tuple(values.get(key, ()))
    return NonlinearStressConfig(**values).validated()


def validate_parent_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    actual_manifest_sha256: str,
    expected_manifest_sha256: str = PARENT_MANIFEST_SHA256,
    expected_checkpoint_sha256: Mapping[str, str] = EXPECTED_CHECKPOINT_SHA256,
) -> dict[str, str]:
    """Refuse any parent identity other than the exact frozen packet."""

    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("nonlinear parent manifest hash mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError("nonlinear parent manifest lacks output records")
    actual = {
        str(name): str(record.get("sha256"))
        for name, record in outputs.items()
        if str(name).startswith("checkpoints/") and isinstance(record, Mapping)
    }
    if actual != dict(expected_checkpoint_sha256):
        raise ValueError("nonlinear parent checkpoint binding mismatch")
    return actual


def _validate_parent_contract(
    parent_dir: Path, *, verify_current_sources: bool = True
) -> tuple[NonlinearStressConfig, dict[str, Any]]:
    verification = verify_parent_packet(
        parent_dir, verify_current_sources=verify_current_sources
    )
    if verification.get("status") != "verified":
        raise ValueError("nonlinear parent packet did not verify")
    manifest_path = parent_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = _json(manifest_path)
    checkpoints = validate_parent_manifest_binding(
        manifest, actual_manifest_sha256=manifest_sha256
    )
    config = _parent_config(parent_dir)
    datasets, rebuilt_support = build_training_contract(config)
    learned = _json(parent_dir / "learned_summary.json")
    recorded_support = learned.get("paired_contract", {}).get("data")
    if rebuilt_support != recorded_support:
        raise ValueError("rebuilt nonlinear training support differs from parent")
    return config, {
        "verification": verification,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoints,
        "config_sha256": sha256_file(parent_dir / "config.json"),
        "training_support": rebuilt_support,
        "datasets": datasets,
    }


def _load_models(
    parent_dir: Path, config: NonlinearStressConfig
) -> dict[int, dict[str, ResidualMLP]]:
    models: dict[int, dict[str, ResidualMLP]] = {}
    for seed in config.seeds:
        models[seed] = {}
        for arm in CHECKPOINT_ARMS:
            relative_name = f"checkpoints/seed_{seed}_{arm.lower()}.pt"
            checkpoint = parent_dir / relative_name
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256.get(relative_name):
                raise ValueError(f"checkpoint hash mismatch: {relative_name}")
            model = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            model.eval()
            models[seed][arm] = model
    if sum(len(value) for value in models.values()) != len(
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise AssertionError("did not load all 15 frozen checkpoints")
    return models


def wrapped_phase(theta: np.ndarray | float) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def periodic_path_for_plot(
    theta: Sequence[float] | np.ndarray,
    radius: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap phase and insert NaNs rather than drawing across the seam."""

    wrapped = wrapped_phase(np.asarray(theta, dtype=np.float64).reshape(-1))
    normal = np.asarray(radius, dtype=np.float64).reshape(-1)
    if wrapped.shape != normal.shape or not wrapped.size:
        raise ValueError("trajectory coordinates must be nonempty and aligned")
    if not np.isfinite(wrapped).all() or not np.isfinite(normal).all():
        raise ValueError("trajectory coordinates must be finite")
    seam = {
        int(value)
        for value in np.flatnonzero(np.abs(np.diff(wrapped)) > np.pi)
    }
    if not seam:
        return wrapped, normal
    theta_output: list[float] = []
    radius_output: list[float] = []
    for index, (phase, value) in enumerate(zip(wrapped, normal, strict=True)):
        theta_output.append(float(phase))
        radius_output.append(float(value))
        if index in seam:
            theta_output.append(float("nan"))
            radius_output.append(float("nan"))
    return np.asarray(theta_output), np.asarray(radius_output)


def _expected_model_keys(config: NonlinearStressConfig) -> set[int]:
    return {int(seed) for seed in config.seeds}


def compute_landscape_arrays(
    models: Mapping[int, Mapping[str, torch.nn.Module]],
    parent: NonlinearStressConfig,
    diagnostic: DiagnosticConfig,
) -> dict[str, np.ndarray]:
    """Compute both one-step targets on one common, seam-safe grid."""

    diagnostic.validated(parent)
    if set(models) != _expected_model_keys(parent):
        raise ValueError("loaded model seeds do not match the parent configuration")
    theta = np.linspace(
        -np.pi, np.pi, diagnostic.theta_points, endpoint=False, dtype=np.float64
    )
    radius = np.linspace(
        -diagnostic.radius_limit,
        diagnostic.radius_limit,
        diagnostic.radius_points,
        dtype=np.float64,
    )
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="xy")
    states = np.column_stack((theta_grid.reshape(-1), radius_grid.reshape(-1)))
    clean_anchors = states.copy()
    clean_anchors[:, 1] = 0.0
    trusted_flows = flows(parent)
    clean_target = trusted_flows["C"].advance(clean_anchors)
    shape = theta_grid.shape
    flow_defect = np.empty(
        (len(parent.seeds), len(REGIME_KEYS), len(ARM_KEYS), *shape),
        dtype=np.float64,
    )
    return_error = np.empty_like(flow_defect)
    output_clean_set_distance = np.empty_like(flow_defect)

    for regime_index, regime in enumerate(REGIME_KEYS):
        flow_target = trusted_flows[regime].advance(states)
        for seed_index, seed in enumerate(parent.seeds):
            seed_models = models[seed]
            expected_arms = set(CHECKPOINT_ARMS)
            if set(seed_models) != expected_arms:
                raise ValueError(f"checkpoint arm mismatch for seed {seed}")
            for arm_index, arm in enumerate(ARM_KEYS):
                checkpoint_arm = f"DYN_{regime}" if arm == "DYN" else arm
                output = model_step(seed_models[checkpoint_arm], states)
                flow_defect[seed_index, regime_index, arm_index] = (
                    lifted_coordinate_error(output, flow_target).reshape(shape)
                )
                return_error[seed_index, regime_index, arm_index] = (
                    lifted_coordinate_error(output, clean_target).reshape(shape)
                )
                output_clean_set_distance[seed_index, regime_index, arm_index] = (
                    np.abs(output[:, 1]).reshape(shape)
                )
    if not all(
        np.isfinite(values).all()
        for values in (flow_defect, return_error, output_clean_set_distance)
    ):
        raise FloatingPointError("non-finite nonlinear diagnostic landscape")
    return {
        "theta": theta,
        "radius": radius,
        "theta_grid": theta_grid,
        "radius_grid": radius_grid,
        "seeds": np.asarray(parent.seeds, dtype=np.int64),
        "regimes": np.asarray(REGIME_KEYS),
        "arms": np.asarray(ARM_KEYS),
        "trusted_flow_defect_by_seed": flow_defect,
        "clean_return_error_by_seed": return_error,
        "output_clean_set_distance_by_seed": output_clean_set_distance,
        "trusted_flow_defect_mean": np.mean(flow_defect, axis=0),
        "clean_return_error_mean": np.mean(return_error, axis=0),
        "output_clean_set_distance_mean": np.mean(
            output_clean_set_distance, axis=0
        ),
    }


def select_fixed_forcing(
    parent: NonlinearStressConfig,
    *,
    phase_index: int = TRAJECTORY_PHASE_INDEX,
    sequence_index: int = TRAJECTORY_SEQUENCE_INDEX,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select one registered forcing path without inspecting any result."""

    initial, forcing, bank_digest = forcing_bank(
        parent, parent.primary_noise_sigma
    )
    if not 0 <= phase_index < parent.noise_phases:
        raise IndexError("phase_index is outside the parent forcing bank")
    if not 0 <= sequence_index < parent.noise_sequences:
        raise IndexError("sequence_index is outside the parent forcing bank")
    flat_index = phase_index * parent.noise_sequences + sequence_index
    return (
        initial[flat_index : flat_index + 1].copy(),
        forcing[:, flat_index : flat_index + 1].copy(),
        {
            "seed": TRAJECTORY_SEED,
            "phase_index": phase_index,
            "sequence_index": sequence_index,
            "flat_index": flat_index,
            "sigma": parent.primary_noise_sigma,
            "steps": parent.noise_steps,
            "full_forcing_bank_digest": bank_digest,
            "selected_initial_digest": digest_array(
                initial[flat_index : flat_index + 1]
            ),
            "selected_forcing_digest": digest_array(
                forcing[:, flat_index : flat_index + 1]
            ),
        },
    )


def canonical_selected_trajectory_bindings(
    parent_dir: Path, parent: NonlinearStressConfig
) -> dict[str, dict[str, str]]:
    """Read the nine canonical unit rows selected before visualization."""

    rows: dict[str, dict[str, str]] = {}
    path = parent_dir / "rollout_units.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["seed"]) != TRAJECTORY_SEED
                or float(row["sigma"]) != parent.primary_noise_sigma
                or int(row["phase_index"]) != TRAJECTORY_PHASE_INDEX
                or int(row["forcing_sequence_index"])
                != TRAJECTORY_SEQUENCE_INDEX
                or row["regime"] not in REGIME_KEYS
                or row["arm"] not in ARM_KEYS
            ):
                continue
            key = f"{row['regime']}_{row['arm']}"
            if key in rows:
                raise ValueError(f"duplicate canonical trajectory row: {key}")
            expected_arm = (
                f"DYN_{row['regime']}" if row["arm"] == "DYN" else row["arm"]
            )
            checkpoint_name = (
                f"checkpoints/seed_{TRAJECTORY_SEED}_{expected_arm.lower()}.pt"
            )
            if (
                row["checkpoint_sha256"]
                != EXPECTED_CHECKPOINT_SHA256[checkpoint_name]
            ):
                raise ValueError(f"canonical trajectory checkpoint mismatch: {key}")
            rows[key] = {
                "trajectory_digest": row["trajectory_digest"],
                "forced_reference_digest": row["forced_reference_digest"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
    expected = {f"{regime}_{arm}" for regime in REGIME_KEYS for arm in ARM_KEYS}
    if set(rows) != expected:
        raise ValueError("canonical selected trajectory rows are incomplete")
    return rows


def _one_step_errors(
    *,
    model: torch.nn.Module,
    trusted_flow: Any,
    clean_flow: Any,
    states: np.ndarray,
    safety_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    output = model_step(model, states)
    clean_anchor = states.copy()
    clean_anchor[:, 1] = 0.0
    return_target = clean_flow.advance(clean_anchor)
    return_error = lifted_coordinate_error(output, return_target)
    flow_error = np.full(states.shape[0], np.nan, dtype=np.float64)
    trusted_available = np.isfinite(states).all(axis=1) & (
        np.abs(states[:, 1]) <= safety_radius
    )
    if np.any(trusted_available):
        flow_target = trusted_flow.advance(states[trusted_available])
        flow_error[trusted_available] = lifted_coordinate_error(
            output[trusted_available], flow_target
        )
    return flow_error, return_error


def compute_trajectory_rows(
    models: Mapping[int, Mapping[str, torch.nn.Module]],
    parent: NonlinearStressConfig,
    *,
    canonical_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute the fixed 192-step path and visited defects."""

    if TRAJECTORY_SEED not in models:
        raise ValueError("fixed trajectory seed is absent from the parent models")
    if canonical_bindings is not None:
        torch.set_num_threads(1)
    selected_initial, selected_forcing, selection = select_fixed_forcing(parent)
    full_initial, full_forcing, full_bank_digest = forcing_bank(
        parent, parent.primary_noise_sigma
    )
    if full_bank_digest != selection["full_forcing_bank_digest"]:
        raise AssertionError("forcing bank changed between selection and replay")
    flat_index = int(selection["flat_index"])
    if not np.array_equal(
        selected_initial, full_initial[flat_index : flat_index + 1]
    ) or not np.array_equal(
        selected_forcing, full_forcing[:, flat_index : flat_index + 1]
    ):
        raise AssertionError("fixed forcing selection does not match the full bank")
    trusted_flows = flows(parent)
    clean_flow = trusted_flows["C"]
    rows: list[dict[str, Any]] = []
    verified_digests: dict[str, dict[str, str]] = {}
    for regime in REGIME_KEYS:
        trusted_flow = trusted_flows[regime]
        trusted_forced_full = rollout_with_forcing(
            trusted_flow.advance, full_initial, full_forcing
        )
        trusted_forced = trusted_forced_full[:, flat_index]
        forced_reference_digest = digest_array(trusted_forced_full)
        for arm in ARM_KEYS:
            checkpoint_arm = f"DYN_{regime}" if arm == "DYN" else arm
            model = models[TRAJECTORY_SEED][checkpoint_arm]
            step = lambda value, current=model: model_step(
                current, value, allow_nonfinite=True
            )
            predicted_full = rollout_with_forcing(
                step, full_initial, full_forcing
            )
            trajectory_digest = digest_array(predicted_full)
            binding_key = f"{regime}_{arm}"
            if canonical_bindings is not None:
                expected = canonical_bindings.get(binding_key)
                if expected is None:
                    raise ValueError(f"canonical trajectory binding absent: {binding_key}")
                if trajectory_digest != expected.get("trajectory_digest"):
                    raise ValueError(f"canonical trajectory replay mismatch: {binding_key}")
                if forced_reference_digest != expected.get("forced_reference_digest"):
                    raise ValueError(
                        f"canonical forced-reference replay mismatch: {binding_key}"
                    )
            verified_digests[binding_key] = {
                "trajectory_digest": trajectory_digest,
                "forced_reference_digest": forced_reference_digest,
            }
            predicted = predicted_full[:, flat_index]
            visited = predicted[:-1]
            flow_defect, return_error = _one_step_errors(
                model=model,
                trusted_flow=trusted_flow,
                clean_flow=clean_flow,
                states=visited,
                safety_radius=parent.safety_radius,
            )
            path_error = lifted_coordinate_error(predicted, trusted_forced)
            for index in range(parent.noise_steps):
                rows.append(
                    {
                        "regime": regime,
                        "arm": arm,
                        "seed": TRAJECTORY_SEED,
                        "phase_index": selection["phase_index"],
                        "sequence_index": selection["sequence_index"],
                        "forcing_sigma": selection["sigma"],
                        "step": index,
                        "theta_lifted": float(predicted[index, 0]),
                        "theta_wrapped": float(wrapped_phase(predicted[index, 0])),
                        "normal": float(predicted[index, 1]),
                        "next_theta_lifted": float(predicted[index + 1, 0]),
                        "next_theta_wrapped": float(
                            wrapped_phase(predicted[index + 1, 0])
                        ),
                        "next_normal": float(predicted[index + 1, 1]),
                        "normal_forcing": float(
                            full_forcing[index, flat_index, 1]
                        ),
                        "inside_parent_tube": bool(
                            abs(predicted[index, 1]) <= parent.tube_radius
                        ),
                        "trusted_flow_defect_available": bool(
                            np.isfinite(flow_defect[index])
                        ),
                        "visited_trusted_flow_defect": (
                            float(flow_defect[index])
                            if np.isfinite(flow_defect[index])
                            else None
                        ),
                        "visited_clean_return_error": float(return_error[index]),
                        "forced_path_error": float(path_error[index]),
                        "trusted_forced_theta_lifted": float(
                            trusted_forced[index, 0]
                        ),
                        "trusted_forced_normal": float(trusted_forced[index, 1]),
                        "trusted_forced_next_theta_lifted": float(
                            trusted_forced[index + 1, 0]
                        ),
                        "trusted_forced_next_normal": float(
                            trusted_forced[index + 1, 1]
                        ),
                    }
                )
    selection["parent_digest_semantics"] = (
        "trajectory digests cover the full canonical forcing batch; the plotted "
        "unit is phase_index=0, forcing_sequence_index=0"
    )
    selection["verified_parent_trajectory_digests"] = verified_digests
    return rows, selection


def _unique_support(inputs: np.ndarray) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    return np.unique(values, axis=0)


def _training_support(
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        "CLEAN": _unique_support(datasets["CLEAN"][0]),
        "RECOVERY": _unique_support(datasets["RECOVERY"][0]),
        "DYN": _unique_support(datasets["DYN_C"][0]),
    }


def _positive_limits(fields: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(field, dtype=np.float64).reshape(-1) for field in fields]
    )
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        return 1.0e-12, 1.0e-11
    lower = max(float(np.quantile(positive, 0.005)), 1.0e-12)
    upper = float(np.quantile(positive, 0.995))
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def _trajectory_subset(
    rows: Sequence[Mapping[str, Any]], *, regime: str, arm: str
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows if row["regime"] == regime and row["arm"] == arm
    ]


def _full_path(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("trajectory row selection is empty")
    theta = [float(row["theta_lifted"]) for row in rows]
    radius = [float(row["normal"]) for row in rows]
    theta.append(float(rows[-1]["next_theta_lifted"]))
    radius.append(float(rows[-1]["next_normal"]))
    return np.asarray(theta), np.asarray(radius)


def _trusted_full_path(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("trusted trajectory row selection is empty")
    theta = [float(row["trusted_forced_theta_lifted"]) for row in rows]
    radius = [float(row["trusted_forced_normal"]) for row in rows]
    theta.append(float(rows[-1]["trusted_forced_next_theta_lifted"]))
    radius.append(float(rows[-1]["trusted_forced_next_normal"]))
    return np.asarray(theta), np.asarray(radius)


def _plot_values_on_grid(
    axis: Any,
    theta: np.ndarray,
    radius: np.ndarray,
    *,
    radius_limit: float,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.1,
) -> None:
    outside = np.flatnonzero(np.abs(radius) > radius_limit)
    stop = int(outside[0]) if outside.size else len(radius)
    if stop:
        plot_theta, plot_radius = periodic_path_for_plot(theta[:stop], radius[:stop])
        axis.plot(
            plot_theta,
            plot_radius,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=7,
        )
    if outside.size:
        index = int(outside[0])
        axis.scatter(
            [wrapped_phase(theta[index])],
            [math.copysign(radius_limit, radius[index])],
            marker="x",
            color=color,
            s=30,
            linewidths=1.2,
            zorder=8,
        )


def _plot_path_on_grid(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    radius_limit: float,
    color: str,
    linewidth: float = 1.1,
) -> None:
    theta, radius = _full_path(rows)
    _plot_values_on_grid(
        axis,
        theta,
        radius,
        radius_limit=radius_limit,
        color=color,
        linewidth=linewidth,
    )


def _plot_landscape_atlas(
    *,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    support: Mapping[str, np.ndarray],
    parent: NonlinearStressConfig,
    regime_index: int,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D

    regime = REGIME_KEYS[regime_index]
    flow_fields = [
        arrays["trusted_flow_defect_mean"][regime_index, index]
        for index in range(len(ARM_KEYS))
    ]
    return_fields = [
        arrays["clean_return_error_mean"][regime_index, index]
        for index in range(len(ARM_KEYS))
    ]
    tube_fields = [
        arrays["output_clean_set_distance_mean"][regime_index, index]
        for index in range(len(ARM_KEYS))
    ]
    flow_limits = _positive_limits(flow_fields)
    return_limits = _positive_limits(return_fields)
    tube_upper = max(
        parent.tube_radius,
        float(
            np.quantile(
                np.concatenate([field.reshape(-1) for field in tube_fields]),
                0.995,
            )
        ),
    )
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.6, 8.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    row_images: list[Any] = [None, None, None]
    for arm_index, arm in enumerate(ARM_KEYS):
        fields_and_norms = (
            (
                flow_fields[arm_index],
                LogNorm(vmin=flow_limits[0], vmax=flow_limits[1]),
            ),
            (
                return_fields[arm_index],
                LogNorm(vmin=return_limits[0], vmax=return_limits[1]),
            ),
            (tube_fields[arm_index], Normalize(vmin=0.0, vmax=tube_upper)),
        )
        selected_rows = _trajectory_subset(rows, regime=regime, arm=arm)
        trusted_theta, trusted_radius = _trusted_full_path(selected_rows)
        for row_index, (field, norm) in enumerate(fields_and_norms):
            axis = axes[row_index, arm_index]
            image = axis.pcolormesh(
                arrays["theta"],
                arrays["radius"],
                field,
                shading="auto",
                cmap=PAPER_HEATMAP_CMAP,
                norm=norm,
                rasterized=True,
            )
            row_images[row_index] = image
            axis.axhline(
                0.0, color=PAPER_REFERENCE_COLOR, linewidth=1.0, zorder=5
            )
            for sign in (-1.0, 1.0):
                axis.axhline(
                    sign * parent.tube_radius,
                    color="white",
                    linestyle=":",
                    linewidth=0.7,
                    zorder=5,
                )
            values = support[arm]
            axis.scatter(
                values[:, 0],
                values[:, 1],
                s=3.2,
                color="white",
                edgecolors="black",
                linewidths=0.15,
                alpha=0.65,
                zorder=6,
            )
            _plot_path_on_grid(
                axis,
                selected_rows,
                radius_limit=float(arrays["radius"][-1]),
                color=PAPER_ARM_COLORS[arm],
                linewidth=1.0,
            )
            _plot_values_on_grid(
                axis,
                trusted_theta,
                trusted_radius,
                radius_limit=float(arrays["radius"][-1]),
                color=PAPER_TRUSTED_COLOR,
                linestyle="--",
                linewidth=1.4,
            )
            axis.set_xlim(-np.pi, np.pi)
            axis.set_ylim(float(arrays["radius"][0]), float(arrays["radius"][-1]))
            axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, arm_index].set_title(arm)
        axes[2, arm_index].set_xlabel(r"phase $\theta$ (display wrapped)")
    axes[0, 0].set_ylabel(r"trusted-flow defect; normal $r$")
    axes[1, 0].set_ylabel(r"clean-return error; normal $r$")
    axes[2, 0].set_ylabel(r"predicted $|r_{out}|$; input normal $r$")
    fig.colorbar(
        row_images[0], ax=list(axes[0]), shrink=0.82, label="mean lifted defect"
    )
    fig.colorbar(
        row_images[1], ax=list(axes[1]), shrink=0.82, label="mean lifted error"
    )
    fig.colorbar(
        row_images[2], ax=list(axes[2]), shrink=0.82, label=r"mean $|r_{out}|$"
    )
    fig.suptitle(
        f"Regime {regime}: retrospective fixed-support diagnostic (not Gaussian data)",
        fontsize=12,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=PAPER_REFERENCE_COLOR,
                label=r"clean set $r=0$",
            ),
            Line2D(
                [0], [0], color="white", linestyle=":", label="parent tube"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                linewidth=0,
                markersize=4,
                label="fixed support",
            ),
            *[
                Line2D([0], [0], color=PAPER_ARM_COLORS[key], label=key)
                for key in ARM_KEYS
            ],
            Line2D(
                [0],
                [0],
                color=PAPER_TRUSTED_COLOR,
                linestyle="--",
                label="trusted path",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_diagnostics(
    *,
    rows: Sequence[Mapping[str, Any]],
    parent: NonlinearStressConfig,
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = PAPER_ARM_COLORS
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 6.4),
        sharex=True,
        constrained_layout=True,
    )
    for regime_index, regime in enumerate(REGIME_KEYS):
        top = axes[0, regime_index]
        bottom = axes[1, regime_index]
        _, trusted_radius = _trusted_full_path(
            _trajectory_subset(rows, regime=regime, arm="CLEAN")
        )
        trusted_magnitude = np.abs(trusted_radius)
        trusted_magnitude[trusted_magnitude == 0.0] = np.nan
        top.semilogy(
            np.arange(trusted_magnitude.size),
            trusted_magnitude,
            color=PAPER_TRUSTED_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="trusted",
        )
        for arm in ARM_KEYS:
            selected = _trajectory_subset(rows, regime=regime, arm=arm)
            _, radius = _full_path(selected)
            magnitude = np.abs(radius)
            magnitude[magnitude == 0.0] = np.nan
            top.semilogy(
                np.arange(magnitude.size),
                magnitude,
                color=colors[arm],
                linewidth=1.2,
                alpha=0.92,
                label=arm,
            )
            steps = np.asarray([row["step"] for row in selected])
            flow_defect = np.asarray(
                [row["visited_trusted_flow_defect"] for row in selected],
                dtype=np.float64,
            )
            return_error = np.asarray(
                [row["visited_clean_return_error"] for row in selected]
            )
            bottom.semilogy(
                steps,
                np.maximum(flow_defect, np.finfo(np.float64).tiny),
                color=colors[arm],
                linewidth=1.0,
            )
            bottom.semilogy(
                steps,
                np.maximum(return_error, np.finfo(np.float64).tiny),
                color=colors[arm],
                linewidth=1.0,
                linestyle="--",
            )
            unavailable = np.flatnonzero(~np.isfinite(flow_defect))
            if unavailable.size:
                finite_flow = flow_defect[np.isfinite(flow_defect)]
                marker_y = (
                    float(np.max(finite_flow))
                    if finite_flow.size
                    else np.finfo(np.float64).tiny
                )
                bottom.scatter(
                    [int(unavailable[0])],
                    [max(marker_y, np.finfo(np.float64).tiny)],
                    marker="x",
                    color=colors[arm],
                    s=28,
                    linewidths=1.2,
                    zorder=7,
                )
        top.axhline(
            parent.tube_radius,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        top.axhline(
            parent.safety_radius,
            color="#333333",
            linestyle="-.",
            linewidth=1.0,
        )
        top.set_xlim(0, parent.noise_steps)
        top.set_title(f"Regime {regime}")
        bottom.set_xlim(0, parent.noise_steps)
        bottom.set_xlabel("visited rollout step")
        bottom.grid(axis="y", alpha=0.2, linewidth=0.6)
        for axis in (top, bottom):
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel(r"normal distance $|r_n|$")
    axes[1, 0].set_ylabel("visited one-step error")
    handles = [
        Line2D([0], [0], color=colors[arm], label=arm) for arm in ARM_KEYS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=PAPER_TRUSTED_COLOR,
                linestyle="--",
                label="same-forcing trusted path",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linestyle=":",
                label="tube radius",
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="-.",
                label="safety radius",
            ),
            Line2D([0], [0], color="black", label="trusted-flow defect"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="clean-return error",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                marker="x",
                linewidth=0,
                label="flow defect unavailable beyond safety",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Fixed replay: retention and visited defects under the old protocol",
        fontsize=12,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _summarize(
    *,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    parent: NonlinearStressConfig,
    diagnostic: DiagnosticConfig,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    landscapes: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIME_KEYS):
        for arm_index, arm in enumerate(ARM_KEYS):
            key = f"{regime}_{arm}"
            flow_field = arrays["trusted_flow_defect_mean"][
                regime_index, arm_index
            ]
            return_field = arrays["clean_return_error_mean"][
                regime_index, arm_index
            ]
            clean_set_field = arrays["output_clean_set_distance_mean"][
                regime_index, arm_index
            ]
            selected = _trajectory_subset(rows, regime=regime, arm=arm)
            _, radius = _full_path(selected)
            available_flow = [
                float(row["visited_trusted_flow_defect"])
                for row in selected
                if row["visited_trusted_flow_defect"] is not None
            ]
            unavailable_steps = [
                int(row["step"])
                for row in selected
                if row["visited_trusted_flow_defect"] is None
            ]
            landscapes[key] = {
                "mean_trusted_flow_defect": float(np.mean(flow_field)),
                "max_trusted_flow_defect": float(np.max(flow_field)),
                "mean_clean_return_error": float(np.mean(return_field)),
                "max_clean_return_error": float(np.max(return_field)),
                "mean_predicted_output_clean_set_distance": float(
                    np.mean(clean_set_field)
                ),
                "max_predicted_output_clean_set_distance": float(
                    np.max(clean_set_field)
                ),
            }
            trajectories[key] = {
                "max_abs_normal": float(np.max(np.abs(radius))),
                "tube_residence_fraction": float(
                    np.mean([bool(row["inside_parent_tube"]) for row in selected])
                ),
                "trusted_flow_defect_coverage": float(
                    len(available_flow) / len(selected)
                ),
                "first_trusted_flow_defect_unavailable_step": (
                    unavailable_steps[0] if unavailable_steps else None
                ),
                "mean_visited_trusted_flow_defect_when_available": (
                    float(np.mean(available_flow)) if available_flow else None
                ),
                "mean_visited_clean_return_error": float(
                    np.mean(
                        [row["visited_clean_return_error"] for row in selected]
                    )
                ),
            }
    return {
        "schema": SUMMARY_SCHEMA,
        "scope": "read_only_diagnostic_of_frozen_fixed_support_protocol",
        "paper_evidence": False,
        "gaussian_training_data": False,
        "defines_id_or_ood": False,
        "online_detector_or_corrector": False,
        "interpretation": (
            "These plots provide intuition for the old four-signed-radius B2-NL "
            "training support. They do not evaluate Gaussian input-noise training."
        ),
        "diagnostic": asdict(diagnostic),
        "parent_config": asdict(parent),
        "trajectory_selection": dict(selection),
        "landscapes": landscapes,
        "trajectories": trajectories,
    }


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


def generate_diagnostic_packet(
    *,
    parent_dir: str | Path,
    output_dir: str | Path,
    diagnostic: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    parent_root = Path(parent_dir).resolve()
    parent, contract = _validate_parent_contract(
        parent_root, verify_current_sources=True
    )
    source_snapshot = _source_records()
    selected = (diagnostic or DiagnosticConfig()).validated(parent)
    torch.set_num_threads(1)
    models = _load_models(parent_root, parent)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=False)

    arrays = compute_landscape_arrays(models, parent, selected)
    canonical_bindings = canonical_selected_trajectory_bindings(
        parent_root, parent
    )
    rows, selection = compute_trajectory_rows(
        models, parent, canonical_bindings=canonical_bindings
    )
    support = _training_support(contract["datasets"])
    np.savez_compressed(output_root / "nonlinear_diagnostic_arrays.npz", **arrays)
    write_csv(output_root / "selected_trajectory_diagnostics.csv", rows)
    atomic_write_json(
        output_root / "diagnostic_config.json",
        {
            "schema": "corrective_ode_nonlinear_diagnostic_config_v1",
            "diagnostic": asdict(selected),
            "trajectory_selection": selection,
        },
    )
    atomic_write_json(
        output_root / "summary.json",
        _summarize(
            arrays=arrays,
            rows=rows,
            parent=parent,
            diagnostic=selected,
            selection=selection,
        ),
    )
    for regime_index, regime in enumerate(REGIME_KEYS):
        stem = f"nonlinear_defect_landscape_{regime.lower()}"
        _plot_landscape_atlas(
            arrays=arrays,
            rows=rows,
            support=support,
            parent=parent,
            regime_index=regime_index,
            output_pdf=output_root / f"{stem}.pdf",
            output_png=output_root / f"{stem}.png",
        )
    _plot_trajectory_diagnostics(
        rows=rows,
        parent=parent,
        output_pdf=output_root / "nonlinear_selected_trajectory_diagnostics.pdf",
        output_png=output_root / "nonlinear_selected_trajectory_diagnostics.png",
    )

    _, close_contract = _validate_parent_contract(
        parent_root, verify_current_sources=True
    )
    if _source_records() != source_snapshot:
        raise ValueError("nonlinear diagnostic sources changed during generation")
    for key in (
        "manifest_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "training_support",
    ):
        if close_contract[key] != contract[key]:
            raise ValueError(f"nonlinear parent binding changed during generation: {key}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scope": "derived_read_only_fixed_support_diagnostic",
        "paper_evidence": False,
        "gaussian_training_data": False,
        "git": git_state(),
        "parent": {
            "packet_name": parent_root.name,
            "manifest_sha256": contract["manifest_sha256"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "config_sha256": contract["config_sha256"],
            "training_support": contract["training_support"],
            "selected_trajectory_bindings": canonical_bindings,
            "verification": contract["verification"],
        },
        "sources": source_snapshot,
        "outputs": _output_records(output_root),
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    atomic_write_json(
        output_root / "closeout.json",
        {
            "schema": CLOSEOUT_SCHEMA,
            "status": "complete",
            "manifest_sha256": sha256_file(output_root / "manifest.json"),
        },
    )
    return verify_diagnostic_packet(
        output_root,
        parent_dir=parent_root,
        verify_current_sources=True,
    )


def verify_diagnostic_packet(
    output_dir: str | Path,
    *,
    parent_dir: str | Path,
    verify_current_sources: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    parent_root = Path(parent_dir).resolve()
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("diagnostic packet lacks manifest or closeout")
    manifest = _json(manifest_path)
    closeout = _json(closeout_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported nonlinear diagnostic manifest schema")
    if closeout.get("schema") != CLOSEOUT_SCHEMA or closeout.get("status") != "complete":
        raise ValueError("invalid nonlinear diagnostic closeout")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("nonlinear diagnostic closeout hash mismatch")

    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_FILES):
        raise ValueError("nonlinear diagnostic source inventory mismatch")
    actual_outputs = _output_records(root)
    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, Mapping) or set(recorded_outputs) != set(
        actual_outputs
    ):
        raise ValueError("nonlinear diagnostic output inventory mismatch")
    for relative_name, record in recorded_outputs.items():
        path = root / relative_name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"nonlinear diagnostic output hash mismatch: {relative_name}")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"nonlinear diagnostic output size mismatch: {relative_name}")
    if verify_current_sources:
        for relative_name, record in sources.items():
            path = REPOSITORY_ROOT / relative_name
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"current nonlinear diagnostic source mismatch: {relative_name}")
            if int(path.stat().st_size) != int(record.get("bytes", -1)):
                raise ValueError(f"current nonlinear diagnostic source size mismatch: {relative_name}")

    parent, parent_contract = _validate_parent_contract(
        parent_root, verify_current_sources=verify_current_sources
    )
    canonical_bindings = canonical_selected_trajectory_bindings(
        parent_root, parent
    )
    recorded_parent = manifest.get("parent")
    if not isinstance(recorded_parent, Mapping):
        raise TypeError("diagnostic manifest lacks parent binding")
    if recorded_parent.get("packet_name") != parent_root.name:
        raise ValueError("nonlinear parent packet identity mismatch")
    for key in (
        "manifest_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "training_support",
    ):
        if recorded_parent.get(key) != parent_contract[key]:
            raise ValueError(f"nonlinear parent binding mismatch: {key}")
    if recorded_parent.get("selected_trajectory_bindings") != canonical_bindings:
        raise ValueError("nonlinear parent selected trajectory binding mismatch")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "outputs": len(recorded_outputs),
        "sources": len(sources),
        "current_sources_checked": verify_current_sources,
        "parent_manifest_sha256": parent_contract["manifest_sha256"],
        "checkpoints_verified": len(parent_contract["checkpoint_sha256"]),
        "scope": manifest.get("scope"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify a derived nonlinear ODE diagnostic packet. "
            "The exact frozen parent is read-only."
        )
    )
    parser.add_argument("--mode", choices=("generate", "verify"), required=True)
    parser.add_argument("--parent-dir", type=Path, default=PARENT_PACKET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta-points", type=int, default=256)
    parser.add_argument("--radius-points", type=int, default=241)
    parser.add_argument("--radius-limit", type=float, default=0.30)
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help="with --mode verify, skip checks against current source files",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "verify":
        result = verify_diagnostic_packet(
            args.output_dir,
            parent_dir=args.parent_dir,
            verify_current_sources=not args.packet_only,
        )
    else:
        if args.packet_only:
            raise ValueError("--packet-only is valid only with --mode verify")
        result = generate_diagnostic_packet(
            parent_dir=args.parent_dir,
            output_dir=args.output_dir,
            diagnostic=DiagnosticConfig(
                theta_points=args.theta_points,
                radius_points=args.radius_points,
                radius_limit=args.radius_limit,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

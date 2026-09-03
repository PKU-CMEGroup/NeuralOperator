#!/usr/bin/env python3
"""Render source-bound unit-circle diagnostics from the B2-GN packet.

The learned state is ``(theta, r)``.  For visualization only, this module uses
the tubular chart

    E(theta, r) = ((1 + r) cos(theta), (1 + r) sin(theta)),

so the clean training manifold ``r = 0`` is the unit circle.  This is not the
literal three-coordinate embedding used inside the MLP.
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

from scripts.time_dependent_no.run_corrective_ode_gaussian_normal_noise import (
    PAPER_ARM_COLORS,
    _config_from_payload,
    verify_packet as verify_gaussian_packet,
)
from scripts.time_dependent_no.run_corrective_ode_nonlinear_stress import (
    ResidualMLP,
    flows,
    forcing_bank,
    model_step,
    rollout_with_forcing,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    digest_array,
    git_state,
    sha256_file,
    write_csv,
)

SCHEMA = "corrective_ode_gaussian_xy_diagnostics_v1"
MANIFEST_SCHEMA = "corrective_ode_gaussian_xy_diagnostics_manifest_v1"
CLOSEOUT_SCHEMA = "corrective_ode_gaussian_xy_diagnostics_closeout_v1"
PARENT_PACKET = Path(
    "artifacts/time_dependent_no/corrective_ode_gaussian_normal_noise_20260830a"
)
PARENT_MANIFEST_SHA256 = (
    "8902430e479caae18e877407cfbc40c8ddd231bb55dc310bd765990a5af46172"
)
SOURCE_FILES = (
    "scripts/time_dependent_no/plot_corrective_ode_gaussian_xy_diagnostics.py",
    "tests/time_dependent_no/test_corrective_ode_gaussian_xy_diagnostics.py",
    "scripts/time_dependent_no/run_corrective_ode_gaussian_normal_noise.py",
    "scripts/time_dependent_no/run_corrective_ode_nonlinear_stress.py",
    "scripts/time_dependent_no/run_corrective_ode_study.py",
    "utility/time_dependent_no/pcno_artifacts.py",
)

SEED_COLORS = {17: "#FFFFFF", 29: "#56B4E9", 43: "#F0E442"}
TRUSTED_COLOR = "#009E73"
MANIFOLD_COLOR = "#00E5FF"
TUBE_COLOR = "#FFFFFF"
METHOD_ARMS = ("CLEAN", "RECOVERY", "DYN")
REGIMES = ("C", "M", "E")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Frozen display and replay choices; none is a scale-selection rule."""

    grid_points: int = 241
    field_radius_limit: float = 0.30
    axis_limit: float = 1.32
    method_display_sigma: float = 0.02
    forcing_sigma: float = 0.005
    selected_trajectory_index: int = 0
    central_gaussian_probability: float = 0.95

    def validated(self) -> DiagnosticConfig:
        if self.grid_points < 65 or self.grid_points % 2 == 0:
            raise ValueError("grid_points must be an odd integer at least 65")
        if not 0.0 < self.field_radius_limit < 1.0:
            raise ValueError("field_radius_limit must lie in (0, 1)")
        if self.axis_limit <= 1.0 + self.field_radius_limit:
            raise ValueError("axis_limit must contain the displayed annulus")
        if self.method_display_sigma <= 0.0 or self.forcing_sigma <= 0.0:
            raise ValueError("display and forcing scales must be positive")
        if self.selected_trajectory_index < 0:
            raise ValueError("selected_trajectory_index must be nonnegative")
        if self.central_gaussian_probability != 0.95:
            raise ValueError("the registered display uses the central 95% annulus")
        return self


def tubular_chart(state: np.ndarray) -> np.ndarray:
    """Map ``(theta, r)`` to the planar unit-circle tube chart."""

    value = np.asarray(state, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError("state must end in dimension two")
    radial_coordinate = 1.0 + value[..., 1]
    return np.stack(
        (
            radial_coordinate * np.cos(value[..., 0]),
            radial_coordinate * np.sin(value[..., 0]),
        ),
        axis=-1,
    )


def planar_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Euclidean error after applying the planar tube chart."""

    left_xy = tubular_chart(left)
    right_xy = tubular_chart(right)
    if left_xy.shape != right_xy.shape:
        raise ValueError("planar error inputs must be aligned")
    return np.linalg.norm(left_xy - right_xy, axis=-1)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        path = REPOSITORY_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return records


def _output_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "closeout.json"}:
            continue
        records[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return records


def _sigma_tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _same_float(value: str | float, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-14)
    except (TypeError, ValueError):
        return False


def load_training_index(
    parent_root: Path, parent_manifest: Mapping[str, Any]
) -> dict[tuple[int, str, float], dict[str, str]]:
    """Validate and index all 51 checkpoint rows from the parent packet."""

    rows = _read_csv(parent_root / "training.csv")
    outputs = parent_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError("parent manifest lacks an output inventory")
    index: dict[tuple[int, str, float], dict[str, str]] = {}
    checkpoint_names: set[str] = set()
    for row in rows:
        seed = int(row["seed"])
        arm = row["arm"]
        sigma = float(row["sigma_train"])
        key = (seed, arm, sigma)
        if key in index:
            raise ValueError(f"duplicate training row: {key}")
        checkpoint_name = row["checkpoint"]
        checkpoint_path = parent_root / "checkpoints" / checkpoint_name
        manifest_name = f"checkpoints/{checkpoint_name}"
        manifest_record = outputs.get(manifest_name)
        if not isinstance(manifest_record, Mapping):
            raise ValueError(f"checkpoint is absent from parent manifest: {manifest_name}")
        actual_hash = sha256_file(checkpoint_path)
        expected_hash = row["checkpoint_sha256"]
        if actual_hash != expected_hash or manifest_record.get("sha256") != actual_hash:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint_name}")
        if checkpoint_name in checkpoint_names:
            raise ValueError(f"checkpoint filename reused: {checkpoint_name}")
        checkpoint_names.add(checkpoint_name)
        index[key] = row
    if len(index) != 51 or len(checkpoint_names) != 51:
        raise ValueError("parent training index must bind exactly 51 checkpoints")
    return index


def _load_models(
    parent_root: Path,
    index: Mapping[tuple[int, str, float], Mapping[str, str]],
    *,
    width: int,
    hidden_layers: int,
) -> dict[tuple[int, str, float], ResidualMLP]:
    models: dict[tuple[int, str, float], ResidualMLP] = {}
    for key, row in index.items():
        model = ResidualMLP(width=width, hidden_layers=hidden_layers).to(
            dtype=torch.float32
        )
        checkpoint = parent_root / "checkpoints" / row["checkpoint"]
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        models[key] = model
    return models


def _model_key(seed: int, arm: str, sigma: float, regime: str = "") -> tuple[int, str, float]:
    if arm == "CLEAN":
        return seed, "CLEAN", 0.0
    if arm == "RECOVERY":
        return seed, "RECOVERY", sigma
    if arm == "DYN":
        if regime not in REGIMES:
            raise ValueError("DYN requires a registered regime")
        return seed, f"DYN_{regime}", sigma
    raise ValueError(f"unknown arm: {arm}")


def _build_grid(config: DiagnosticConfig) -> dict[str, np.ndarray]:
    outer = 1.0 + config.field_radius_limit
    coordinate = np.linspace(-outer, outer, config.grid_points, dtype=np.float64)
    x, y = np.meshgrid(coordinate, coordinate)
    physical_radius = np.hypot(x, y)
    normal = physical_radius - 1.0
    theta = np.arctan2(y, x)
    mask = np.abs(normal) <= config.field_radius_limit + 1.0e-12
    state = np.column_stack((theta[mask], normal[mask]))
    return {
        "coordinate": coordinate,
        "x": x,
        "y": y,
        "normal": normal,
        "mask": mask,
        "state": state,
    }


def _masked_field(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(mask.shape, np.nan, dtype=np.float64)
    result[mask] = np.asarray(values, dtype=np.float64)
    return result


def compute_fields(
    models: Mapping[tuple[int, str, float], ResidualMLP],
    *,
    base_config: Any,
    diagnostic: DiagnosticConfig,
    scales: Sequence[float],
) -> dict[str, np.ndarray]:
    """Compute by-seed planar defects and their arithmetic means."""

    grid = _build_grid(diagnostic)
    state = grid["state"]
    mask = grid["mask"]
    flow_by_regime = flows(base_config)
    trusted_xy = {
        regime: tubular_chart(flow.advance(state))
        for regime, flow in flow_by_regime.items()
    }
    clean_state = state.copy()
    clean_state[:, 1] = 0.0
    clean_successor_xy = tubular_chart(flow_by_regime["C"].advance(clean_state))

    prediction_xy = {
        key: tubular_chart(model_step(model, state)) for key, model in models.items()
    }
    arrays: dict[str, np.ndarray] = {
        "coordinate": grid["coordinate"],
        "x": grid["x"],
        "y": grid["y"],
        "normal": grid["normal"],
        "annulus_mask": mask,
    }

    def register(prefix: str, fields: Sequence[np.ndarray]) -> None:
        if len(fields) != len(base_config.seeds):
            raise ValueError("one field per registered seed is required")
        for seed, field in zip(base_config.seeds, fields, strict=True):
            arrays[f"{prefix}_seed_{seed}"] = _masked_field(field, mask).astype(
                np.float32
            )
        arrays[f"{prefix}_mean"] = _masked_field(
            np.mean(np.stack(fields, axis=0), axis=0), mask
        ).astype(np.float32)

    display_sigma = diagnostic.method_display_sigma
    for regime in REGIMES:
        for arm in METHOD_ARMS:
            fields = []
            for seed in base_config.seeds:
                key = _model_key(seed, arm, display_sigma, regime)
                fields.append(
                    np.linalg.norm(prediction_xy[key] - trusted_xy[regime], axis=-1)
                )
            register(f"method_{regime}_{arm}_flow", fields)

    for sigma in scales:
        tag = _sigma_tag(sigma)
        recovery_fields = []
        for seed in base_config.seeds:
            key = _model_key(seed, "RECOVERY", sigma)
            recovery_fields.append(
                np.linalg.norm(prediction_xy[key] - clean_successor_xy, axis=-1)
            )
        register(f"scale_RECOVERY_{tag}_return", recovery_fields)
        for regime in REGIMES:
            dynamic_fields = []
            for seed in base_config.seeds:
                key = _model_key(seed, "DYN", sigma, regime)
                dynamic_fields.append(
                    np.linalg.norm(
                        prediction_xy[key] - trusted_xy[regime], axis=-1
                    )
                )
            register(f"scale_DYN_{regime}_{tag}_flow", dynamic_fields)
    return arrays


def _endpoint_matches(
    row: Mapping[str, str],
    *,
    seed: int,
    regime: str,
    arm: str,
    sigma_train: float,
    sigma_force: float,
) -> bool:
    if (
        int(row["seed"]) != seed
        or row["regime"] != regime
        or row["arm"] != arm
        or not _same_float(row["sigma_force"], sigma_force)
    ):
        return False
    if arm == "TRUSTED_ORACLE":
        return _same_float(row["comparison_sigma_train"], sigma_train)
    if arm == "CLEAN":
        return _same_float(row["sigma_train"], 0.0) and _same_float(
            row["comparison_sigma_train"], sigma_train
        )
    return _same_float(row["sigma_train"], sigma_train)


def _unique_endpoint(
    rows: Sequence[Mapping[str, str]], **criteria: Any
) -> Mapping[str, str]:
    selected = [row for row in rows if _endpoint_matches(row, **criteria)]
    if len(selected) != 1:
        raise ValueError(f"expected one endpoint row, found {len(selected)}: {criteria}")
    return selected[0]


def replay_trajectories(
    models: Mapping[tuple[int, str, float], ResidualMLP],
    training_index: Mapping[tuple[int, str, float], Mapping[str, str]],
    endpoint_rows: Sequence[Mapping[str, str]],
    *,
    base_config: Any,
    diagnostic: DiagnosticConfig,
    scales: Sequence[float],
) -> tuple[dict[tuple[str, str, float, int], np.ndarray], dict[str, Any]]:
    """Replay full registered banks, verify digests, retain one plotted unit."""

    initial, forcing, sign_digest = forcing_bank(base_config, diagnostic.forcing_sigma)
    selected = diagnostic.selected_trajectory_index
    if selected >= initial.shape[0]:
        raise ValueError("selected trajectory index is outside the forcing bank")
    path_by_key: dict[tuple[str, str, float, int], np.ndarray] = {}
    replay_records: list[dict[str, Any]] = []
    flow_by_regime = flows(base_config)

    trusted_full: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        trajectory = rollout_with_forcing(flow_by_regime[regime].advance, initial, forcing)
        trusted_full[regime] = trajectory
        path_by_key[("TRUSTED", regime, 0.0, -1)] = trajectory[:, selected]
        actual_digest = digest_array(trajectory)
        for sigma in scales:
            expected = _unique_endpoint(
                endpoint_rows,
                seed=-1,
                regime=regime,
                arm="TRUSTED_ORACLE",
                sigma_train=sigma,
                sigma_force=diagnostic.forcing_sigma,
            )
            if actual_digest != expected["trajectory_digest"]:
                raise ValueError(f"trusted trajectory digest mismatch: {regime}")
        replay_records.append(
            {
                "source": "TRUSTED_ORACLE",
                "regime": regime,
                "trajectory_digest": actual_digest,
                "canonical_rows_matched": len(scales),
            }
        )

    for seed in base_config.seeds:
        clean_key = _model_key(seed, "CLEAN", 0.0)
        clean_full = rollout_with_forcing(
            lambda value, model=models[clean_key]: model_step(
                model, value, allow_nonfinite=True
            ),
            initial,
            forcing,
        )
        clean_digest = digest_array(clean_full)
        path_by_key[("CLEAN", "", 0.0, seed)] = clean_full[:, selected]
        for regime in REGIMES:
            for sigma in scales:
                expected = _unique_endpoint(
                    endpoint_rows,
                    seed=seed,
                    regime=regime,
                    arm="CLEAN",
                    sigma_train=sigma,
                    sigma_force=diagnostic.forcing_sigma,
                )
                if clean_digest != expected["trajectory_digest"]:
                    raise ValueError(f"CLEAN trajectory digest mismatch: seed {seed}")
        replay_records.append(
            {
                "source": "CLEAN",
                "seed": seed,
                "trajectory_digest": clean_digest,
                "canonical_rows_matched": len(REGIMES) * len(scales),
            }
        )

        for sigma in scales:
            recovery_key = _model_key(seed, "RECOVERY", sigma)
            recovery_full = rollout_with_forcing(
                lambda value, model=models[recovery_key]: model_step(
                    model, value, allow_nonfinite=True
                ),
                initial,
                forcing,
            )
            recovery_digest = digest_array(recovery_full)
            path_by_key[("RECOVERY", "", sigma, seed)] = recovery_full[:, selected]
            for regime in REGIMES:
                expected = _unique_endpoint(
                    endpoint_rows,
                    seed=seed,
                    regime=regime,
                    arm="RECOVERY",
                    sigma_train=sigma,
                    sigma_force=diagnostic.forcing_sigma,
                )
                if recovery_digest != expected["trajectory_digest"]:
                    raise ValueError(
                        f"RECOVERY trajectory digest mismatch: seed {seed}, sigma {sigma}"
                    )
            replay_records.append(
                {
                    "source": "RECOVERY",
                    "seed": seed,
                    "sigma_train": sigma,
                    "trajectory_digest": recovery_digest,
                    "canonical_rows_matched": len(REGIMES),
                }
            )

            for regime in REGIMES:
                dynamic_key = _model_key(seed, "DYN", sigma, regime)
                dynamic_full = rollout_with_forcing(
                    lambda value, model=models[dynamic_key]: model_step(
                        model, value, allow_nonfinite=True
                    ),
                    initial,
                    forcing,
                )
                dynamic_digest = digest_array(dynamic_full)
                path_by_key[("DYN", regime, sigma, seed)] = dynamic_full[:, selected]
                expected = _unique_endpoint(
                    endpoint_rows,
                    seed=seed,
                    regime=regime,
                    arm="DYN",
                    sigma_train=sigma,
                    sigma_force=diagnostic.forcing_sigma,
                )
                training_row = training_index[dynamic_key]
                if (
                    dynamic_digest != expected["trajectory_digest"]
                    or training_row["checkpoint"] != expected["checkpoint"]
                    or training_row["checkpoint_sha256"]
                    != expected["checkpoint_sha256"]
                ):
                    raise ValueError(
                        "DYN trajectory or checkpoint mismatch: "
                        f"seed {seed}, regime {regime}, sigma {sigma}"
                    )
                replay_records.append(
                    {
                        "source": f"DYN_{regime}",
                        "seed": seed,
                        "sigma_train": sigma,
                        "trajectory_digest": dynamic_digest,
                        "canonical_rows_matched": 1,
                    }
                )

    return path_by_key, {
        "schema": "corrective_ode_gaussian_xy_trajectory_replay_v1",
        "forcing_sigma": diagnostic.forcing_sigma,
        "forcing_sign_digest": sign_digest,
        "forcing_digest": digest_array(forcing),
        "initial_bank_digest": digest_array(initial),
        "selected_trajectory_index": selected,
        "selected_initial_state": initial[selected].tolist(),
        "full_bank_replayed": True,
        "all_replayed_digests_match_parent": True,
        "records": replay_records,
    }


def trajectory_rows(
    paths: Mapping[tuple[str, str, float, int], np.ndarray],
    *,
    radius_limit: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (arm, regime, sigma, seed), path in sorted(paths.items()):
        xy = tubular_chart(path)
        for step, (state, point) in enumerate(zip(path, xy, strict=True)):
            rows.append(
                {
                    "arm": arm,
                    "regime": regime,
                    "sigma_train": sigma,
                    "seed": seed,
                    "step": step,
                    "theta_lifted": float(state[0]),
                    "normal": float(state[1]),
                    "x1": float(point[0]),
                    "x2": float(point[1]),
                    "inside_display_annulus": bool(
                        np.isfinite(state).all() and abs(state[1]) <= radius_limit
                    ),
                }
            )
    return rows


def visible_path(path: np.ndarray, radius_limit: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the in-annulus prefix and a boundary marker for its first exit."""

    value = np.asarray(path, dtype=np.float64)
    finite = np.isfinite(value).all(axis=1)
    inside = finite & (np.abs(value[:, 1]) <= radius_limit)
    exits = np.flatnonzero(~inside)
    if not exits.size:
        return tubular_chart(value), None
    first = int(exits[0])
    prefix = tubular_chart(value[:first]) if first else np.empty((0, 2))
    if not finite[first]:
        return prefix, None
    boundary_state = value[first].copy()
    boundary_state[1] = math.copysign(radius_limit, boundary_state[1])
    return prefix, tubular_chart(boundary_state)


def positive_limits(fields: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(field, dtype=np.float64).reshape(-1) for field in fields]
    )
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        raise ValueError("logarithmic field contains no positive finite values")
    lower = max(float(np.quantile(positive, 0.01)), 1.0e-7)
    upper = float(np.quantile(positive, 0.99))
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def _draw_reference_geometry(axis: Any, *, tube_radius: float) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 721)
    for radius in (1.0 - tube_radius, 1.0 + tube_radius):
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            color=TUBE_COLOR,
            linestyle=":",
            linewidth=0.75,
            alpha=0.85,
            zorder=4,
        )
    axis.plot(
        np.cos(angle),
        np.sin(angle),
        color=MANIFOLD_COLOR,
        linewidth=1.25,
        zorder=5,
    )


def _draw_gaussian_annulus(axis: Any, sigma: float) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 721)
    quantile = 1.959963984540054
    for normal in (-quantile * sigma, quantile * sigma):
        radius = 1.0 + normal
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            color="#D9D9D9",
            linestyle="--",
            linewidth=0.7,
            alpha=0.9,
            zorder=4,
        )


def _draw_paths(
    axis: Any,
    paths: Mapping[tuple[str, str, float, int], np.ndarray],
    *,
    arm: str,
    regime: str,
    sigma: float,
    seeds: Sequence[int],
    radius_limit: float,
) -> None:
    for seed in seeds:
        key = (arm, "" if arm != "DYN" else regime, sigma, seed)
        if arm == "CLEAN":
            key = (arm, "", 0.0, seed)
        prefix, exit_point = visible_path(paths[key], radius_limit)
        if prefix.size:
            axis.plot(
                prefix[:, 0],
                prefix[:, 1],
                color=SEED_COLORS[seed],
                linewidth=1.05,
                alpha=0.95,
                zorder=7,
            )
        if exit_point is not None:
            axis.scatter(
                [exit_point[0]],
                [exit_point[1]],
                color=SEED_COLORS[seed],
                marker="x",
                s=20,
                linewidths=1.0,
                zorder=9,
            )
    trusted_prefix, trusted_exit = visible_path(
        paths[("TRUSTED", regime, 0.0, -1)], radius_limit
    )
    if trusted_prefix.size:
        axis.plot(
            trusted_prefix[:, 0],
            trusted_prefix[:, 1],
            color=TRUSTED_COLOR,
            linestyle="--",
            linewidth=1.2,
            zorder=8,
        )
    if trusted_exit is not None:
        axis.scatter(
            [trusted_exit[0]],
            [trusted_exit[1]],
            color=TRUSTED_COLOR,
            marker="x",
            s=22,
            linewidths=1.0,
            zorder=9,
        )


def _format_axis(axis: Any, config: DiagnosticConfig) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-config.axis_limit, config.axis_limit)
    axis.set_ylim(-config.axis_limit, config.axis_limit)
    axis.set_xticks((-1.0, 0.0, 1.0))
    axis.set_yticks((-1.0, 0.0, 1.0))
    axis.spines[["top", "right"]].set_visible(False)


def _legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        *[
            Line2D(
                [0],
                [0],
                color=SEED_COLORS[seed],
                lw=1.3,
                marker="o",
                markersize=4,
                markeredgecolor="black" if seed == 17 else SEED_COLORS[seed],
                label=f"seed {seed}",
            )
            for seed in SEED_COLORS
        ],
        Line2D(
            [0], [0], color=TRUSTED_COLOR, lw=1.3, ls="--", label="trusted flow"
        ),
        Line2D([0], [0], color=MANIFOLD_COLOR, lw=1.3, label="training manifold"),
        Line2D([0], [0], color="#777777", lw=0.9, ls=":", label="tube boundary"),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="x",
            linestyle="none",
            markersize=5,
            label="first display exit",
        ),
    ]


def plot_method_comparison(
    arrays: Mapping[str, np.ndarray],
    paths: Mapping[tuple[str, str, float, int], np.ndarray],
    *,
    base_config: Any,
    diagnostic: DiagnosticConfig,
    output_pdf: Path,
    output_png: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(3, 3, figsize=(9.2, 8.5), constrained_layout=True)
    fig.suptitle(
        rf"trajectory overlays: normal forcing $\sigma_{{\rm force}}={diagnostic.forcing_sigma:g}$",
        fontsize=8,
    )
    panel = 0
    limits: dict[str, list[float]] = {}
    for row, regime in enumerate(REGIMES):
        row_fields = [arrays[f"method_{regime}_{arm}_flow_mean"] for arm in METHOD_ARMS]
        vmin, vmax = positive_limits(row_fields)
        limits[regime] = [vmin, vmax]
        images = []
        for column, arm in enumerate(METHOD_ARMS):
            axis = axes[row, column]
            image = axis.pcolormesh(
                arrays["x"],
                arrays["y"],
                np.ma.masked_invalid(row_fields[column]),
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
                shading="auto",
                rasterized=True,
            )
            images.append(image)
            _draw_reference_geometry(axis, tube_radius=base_config.tube_radius)
            _draw_paths(
                axis,
                paths,
                arm=arm,
                regime=regime,
                sigma=diagnostic.method_display_sigma,
                seeds=base_config.seeds,
                radius_limit=diagnostic.field_radius_limit,
            )
            _format_axis(axis, diagnostic)
            panel += 1
            axis.text(
                0.03,
                0.96,
                f"({chr(96 + panel)})",
                transform=axis.transAxes,
                va="top",
                color="black",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.7,
                },
                zorder=10,
            )
            if row == 0:
                axis.set_title(
                    {
                        "CLEAN": "CLEAN\nzero-noise control",
                        "RECOVERY": "RECOVERY\n"
                        + rf"$\sigma_{{\rm train}}={diagnostic.method_display_sigma:g}$",
                        "DYN": "DYN-RELABEL\n"
                        + rf"$\sigma_{{\rm train}}={diagnostic.method_display_sigma:g}$",
                    }[arm],
                    color=PAPER_ARM_COLORS[arm],
                    fontsize=10,
                )
            if column == 0:
                axis.set_ylabel(f"{regime}: $x_2$")
            if row == 2:
                axis.set_xlabel("$x_1$")
        colorbar = fig.colorbar(images[-1], ax=axes[row, :].tolist(), pad=0.012, shrink=0.82)
        colorbar.set_label("planar one-step defect", fontsize=8)
    fig.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"row_color_limits": limits}


def plot_scale_comparison(
    arrays: Mapping[str, np.ndarray],
    paths: Mapping[tuple[str, str, float, int], np.ndarray],
    *,
    base_config: Any,
    diagnostic: DiagnosticConfig,
    scales: Sequence[float],
    output_pdf: Path,
    output_png: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    row_specs = (
        ("RECOVERY", "C", "clean-return error"),
        ("DYN", "C", "trusted-flow defect"),
        ("DYN", "M", "trusted-flow defect"),
        ("DYN", "E", "trusted-flow defect"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(11.2, 10.4), constrained_layout=True)
    fig.suptitle(
        rf"trajectory overlays: normal forcing $\sigma_{{\rm force}}={diagnostic.forcing_sigma:g}$",
        fontsize=8,
    )
    limits: dict[str, list[float]] = {}
    panel = 0
    for row, (arm, regime, quantity) in enumerate(row_specs):
        fields = []
        for sigma in scales:
            tag = _sigma_tag(sigma)
            key = (
                f"scale_RECOVERY_{tag}_return_mean"
                if arm == "RECOVERY"
                else f"scale_DYN_{regime}_{tag}_flow_mean"
            )
            fields.append(arrays[key])
        vmin, vmax = positive_limits(fields)
        row_name = "RECOVERY" if arm == "RECOVERY" else f"DYN-{regime}"
        limits[row_name] = [vmin, vmax]
        images = []
        for column, sigma in enumerate(scales):
            axis = axes[row, column]
            image = axis.pcolormesh(
                arrays["x"],
                arrays["y"],
                np.ma.masked_invalid(fields[column]),
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
                shading="auto",
                rasterized=True,
            )
            images.append(image)
            _draw_reference_geometry(axis, tube_radius=base_config.tube_radius)
            _draw_gaussian_annulus(axis, sigma)
            _draw_paths(
                axis,
                paths,
                arm=arm,
                regime=regime,
                sigma=sigma,
                seeds=base_config.seeds,
                radius_limit=diagnostic.field_radius_limit,
            )
            _format_axis(axis, diagnostic)
            panel += 1
            axis.text(
                0.03,
                0.96,
                f"({chr(96 + panel)})",
                transform=axis.transAxes,
                va="top",
                color="black",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.6,
                },
                zorder=10,
            )
            if row == 0:
                axis.set_title(rf"$\sigma_{{\rm train}}={sigma:g}$", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{row_name}\n$x_2$")
            if row == 3:
                axis.set_xlabel("$x_1$")
        colorbar = fig.colorbar(images[-1], ax=axes[row, :].tolist(), pad=0.01, shrink=0.8)
        colorbar.set_label(quantity, fontsize=8)
    gaussian_handle = Line2D(
        [0],
        [0],
        color="#777777",
        lw=0.9,
        ls="--",
        label="central 95% training-noise annulus",
    )
    fig.legend(
        handles=[*_legend_handles(), gaussian_handle],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"row_color_limits": limits}


def verify_packet(output_dir: str | Path, *, verify_current_sources: bool = True) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("derived packet lacks manifest.json or closeout.json")
    manifest = _load_json(manifest_path)
    closeout = _load_json(closeout_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported derived manifest schema")
    if closeout.get("schema") != CLOSEOUT_SCHEMA or closeout.get("status") != "complete":
        raise ValueError("invalid derived closeout record")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("derived closeout does not bind the manifest")
    if manifest.get("outputs") != _output_records(root):
        raise ValueError("derived output inventory or hash mismatch")
    if verify_current_sources and manifest.get("sources") != _source_records():
        raise ValueError("current sources differ from the derived packet")
    parent_manifest = REPOSITORY_ROOT / PARENT_PACKET / "manifest.json"
    if (
        sha256_file(parent_manifest) != PARENT_MANIFEST_SHA256
        or manifest.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
    ):
        raise ValueError("derived packet parent binding mismatch")
    verify_gaussian_packet(
        REPOSITORY_ROOT / PARENT_PACKET,
        verify_current_sources=verify_current_sources,
    )
    replay = _load_json(root / "trajectory_replay.json")
    if replay.get("all_replayed_digests_match_parent") is not True:
        raise ValueError("trajectory replay did not close")
    contract = _load_json(root / "diagnostic_contract.json")
    if (
        contract.get("scale_optimization") is not False
        or contract.get("all_registered_scales_shown") is not True
        or contract.get("state_chart") != "planar_unit_circle_tube_chart"
    ):
        raise ValueError("derived diagnostic contract changed semantics")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "output_count": len(manifest["outputs"]),
        "current_sources_verified": verify_current_sources,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
    }


def run(
    output_dir: str | Path,
    *,
    diagnostic: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    config = (diagnostic or DiagnosticConfig()).validated()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    parent_root = REPOSITORY_ROOT / PARENT_PACKET
    parent_verification = verify_gaussian_packet(parent_root)
    parent_manifest_path = parent_root / "manifest.json"
    if sha256_file(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("canonical B2-GN parent manifest hash mismatch")
    parent_manifest = _load_json(parent_manifest_path)
    gaussian_config = _config_from_payload(_load_json(parent_root / "config.json"))
    scales = gaussian_config.train_noise_stds
    if config.method_display_sigma not in scales:
        raise ValueError("method display scale is not registered")
    if config.forcing_sigma != gaussian_config.base.primary_noise_sigma:
        raise ValueError("trajectory forcing scale differs from the registered primary assay")
    source_snapshot = _source_records()
    training_index = load_training_index(parent_root, parent_manifest)
    torch.set_num_threads(1)
    models = _load_models(
        parent_root,
        training_index,
        width=gaussian_config.base.hidden_width,
        hidden_layers=gaussian_config.base.hidden_layers,
    )
    arrays = compute_fields(
        models,
        base_config=gaussian_config.base,
        diagnostic=config,
        scales=scales,
    )
    np.savez_compressed(root / "planar_defect_fields.npz", **arrays)
    endpoint_rows = _read_csv(parent_root / "rollout_endpoints.csv")
    paths, replay = replay_trajectories(
        models,
        training_index,
        endpoint_rows,
        base_config=gaussian_config.base,
        diagnostic=config,
        scales=scales,
    )
    write_csv(
        root / "selected_trajectories.csv",
        trajectory_rows(paths, radius_limit=config.field_radius_limit),
    )
    atomic_write_json(root / "trajectory_replay.json", replay)
    method_plot = plot_method_comparison(
        arrays,
        paths,
        base_config=gaussian_config.base,
        diagnostic=config,
        output_pdf=root / "ode_gaussian_intervention_comparison_xy.pdf",
        output_png=root / "ode_gaussian_intervention_comparison_xy.png",
    )
    scale_plot = plot_scale_comparison(
        arrays,
        paths,
        base_config=gaussian_config.base,
        diagnostic=config,
        scales=scales,
        output_pdf=root / "ode_gaussian_scale_comparison_xy.pdf",
        output_png=root / "ode_gaussian_scale_comparison_xy.png",
    )
    contract = {
        "schema": SCHEMA,
        "scope": "derived_read_only_planar_unit_circle_diagnostic",
        "state_chart": "planar_unit_circle_tube_chart",
        "chart_definition": "E(theta,r)=((1+r)cos(theta),(1+r)sin(theta))",
        "literal_mlp_input_embedding": "(cos(theta),sin(theta),r)",
        "chart_is_visualization_only": True,
        "training_manifold": "unit_circle_r_equals_zero",
        "method_comparison": {
            "corrected_arm_display_sigma": config.method_display_sigma,
            "clean_arm_sigma": 0.0,
            "display_scale_is_not_selected_as_best": True,
            "rows": list(REGIMES),
            "columns": list(METHOD_ARMS),
            "background": "mean_seed_planar_trusted_flow_defect",
        },
        "scale_comparison": {
            "scales": list(scales),
            "all_scales_reported": True,
            "recovery_background": "mean_seed_planar_clean_return_error",
            "dynamic_background": "mean_seed_planar_trusted_flow_defect",
        },
        "trajectory_assay": {
            "forcing_sigma": config.forcing_sigma,
            "forcing_law": "post_map_paired_rademacher_normal_only",
            "selected_trajectory_index": config.selected_trajectory_index,
            "all_parent_banks_replayed_for_digest_verification": True,
        },
        "diagnostic": asdict(config),
        "palette": {
            "arms": PAPER_ARM_COLORS,
            "seeds": SEED_COLORS,
            "trusted": TRUSTED_COLOR,
            "training_manifold": MANIFOLD_COLOR,
            "heatmap": "magma",
        },
        "method_plot": method_plot,
        "scale_plot": scale_plot,
        "seed_aggregation": "arithmetic_mean_of_planar_error_magnitudes",
        "by_seed_fields_retained": True,
        "all_registered_scales_shown": True,
        "scale_optimization": False,
        "defines_id_or_ood": False,
        "online_detector_or_corrector": False,
        "rollout_mechanism_qualified": False,
        "scientific_classification": "descriptive_supporting_diagnostic_only",
        "paper_layout_candidate": True,
        "paper_evidence": False,
        "new_training_or_scientific_result": False,
        "ode_to_pde_transfer_claim": False,
    }
    atomic_write_json(root / "diagnostic_contract.json", contract)
    if _source_records() != source_snapshot:
        raise ValueError("diagnostic sources changed during rendering")
    if sha256_file(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("parent packet changed during rendering")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "git": git_state(),
        "parent_packet": PARENT_PACKET.as_posix(),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_verification": parent_verification,
        "checkpoint_count": len(training_index),
        "sources": source_snapshot,
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
    return {"output_dir": str(root), "verification": verify_packet(root)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or verify B2-GN unit-circle defect and trajectory diagnostics."
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
    if args.mode == "run":
        if args.packet_only:
            raise ValueError("--packet-only is valid only with --mode verify")
        result = run(args.output_dir)
    else:
        result = verify_packet(
            args.output_dir, verify_current_sources=not args.packet_only
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

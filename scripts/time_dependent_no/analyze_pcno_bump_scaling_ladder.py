#!/usr/bin/env python3
"""Analyze the completed D094 seed-0 bump scaling ladder.

The analysis keeps selected checkpoints, terminal checkpoints, and the
outside-selection audit distinct.  It also reports exact matched-exposure
slices only when a recorded rollout row exists; it never interpolates a model
state or a missing fixed-bank metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_scaling import (
    _load_json,
    _load_jsonl,
    _metric_snapshot,
    _overoptimization_event,
    _snapshot_integrity,
)

SCHEMA = "d094_bump_scaling_ladder_analysis_v1"
ARTIFACT_SCHEMA = "d094_bump_scaling_ladder_analysis_artifacts_v1"
AUDIT_SCHEMA = "d094_bump_scaling_ladder_outside_selection_audit_v1"
FRESH_SOURCE_SET_DIGEST = (
    "c9ecfd93f75f61a69a1333f2f25b0778f4436a33660b40bd3dbc38fc862a8e61"
)
B1A_SOURCE_SET_DIGEST = (
    "6c510fbdca8f50d2bfacd40239574e7ac4496bdb0ba575c5ce69bb744568fca5"
)
SPLIT_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
COUNTS = (8, 16, 32, 64, 128, 256)
ARCHITECTURES = ("pcno", "pcfno")
OPTIMIZER_STEPS = 20_480
EPOCHS = 80
STEPS_PER_EPOCH = 256
WINDOWS_PER_TRAJECTORY = 79
MATCHED_EXPOSURE_STEP_MULTIPLIERS = (80, 160, 320, 640, 1_280)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-ladder-root", type=Path, required=True)
    parser.add_argument("--n256-gate-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_821)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV output requires at least one row")
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("audit artifact manifest lacks files")
    checked = 0
    for name, record in records.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise TypeError("audit artifact record is malformed")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"audit artifact byte count changed: {name}")
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"audit artifact digest changed: {name}")
        checked += 1
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "checked_files": checked,
    }


def _check_completion_receipts(fresh_root: Path, gate_root: Path) -> None:
    receipts = (
        (fresh_root / "ladder.exit", "0"),
        (gate_root / "gate.exit", "0"),
    )
    markers = (fresh_root / "ladder.completed", gate_root / "gate.completed")
    for path, expected in receipts:
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != expected:
            raise ValueError(f"completion receipt failed: {path}")
    for path in markers:
        if not path.is_file():
            raise FileNotFoundError(path)


def _scope_matches(snapshot: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    for name, expected in scope.items():
        actual = snapshot.get(name)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if actual is None or not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-14
            ):
                return False
        elif actual != expected:
            return False
    return True


def _load_history(
    run_dir: Path,
    audit_cell: Mapping[str, Any],
    *,
    expected_source_set_digest: str,
) -> dict[str, Any]:
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    if len(rows) != EPOCHS:
        raise ValueError(f"training history does not have {EPOCHS} rows: {run_dir}")
    snapshots = [_metric_snapshot(row) for row in rows]
    expected_steps = [(index + 1) * STEPS_PER_EPOCH for index in range(EPOCHS)]
    if [int(row["optimizer_step"]) for row in snapshots] != expected_steps:
        raise ValueError(f"optimizer-step history drifted: {run_dir}")
    scopes = audit_cell.get("training_metric_scopes")
    if not isinstance(scopes, Mapping):
        raise TypeError("audit cell lacks training metric scopes")
    selected_scope = scopes.get("selected_checkpoint")
    terminal_scope = scopes.get("terminal_checkpoint")
    if not isinstance(selected_scope, Mapping) or not isinstance(
        terminal_scope, Mapping
    ):
        raise TypeError("audit cell training scopes are malformed")
    selected_rows = [
        snapshot
        for snapshot in snapshots
        if int(snapshot["epoch"]) == int(selected_scope["epoch"])
    ]
    if len(selected_rows) != 1 or not _scope_matches(selected_rows[0], selected_scope):
        raise ValueError("selected audit metrics differ from retained history")
    if not _scope_matches(snapshots[-1], terminal_scope):
        raise ValueError("terminal audit metrics differ from retained history")

    summary = _load_json(run_dir / "summary.json")
    contract = _load_json(run_dir / "bump_scaling_contract.json")
    source_manifest = _load_json(run_dir / "source_snapshot" / "manifest.json")
    if source_manifest.get("source_set_digest") != expected_source_set_digest:
        raise ValueError("training source-set digest changed")
    checked_source_files = _snapshot_integrity(run_dir, source_manifest)
    if contract.get("historical_test_population_accessed") is not False:
        raise ValueError("training cell accessed the historical test population")
    if contract.get("split_partition_digest") != SPLIT_PARTITION_DIGEST:
        raise ValueError("training cell split binding changed")
    if int(summary.get("actual_optimizer_steps", -1)) != OPTIMIZER_STEPS:
        raise ValueError("training cell optimizer budget changed")
    if int(summary.get("completed_epochs", -1)) != EPOCHS:
        raise ValueError("training cell epoch count changed")
    rollout_snapshots = [
        row
        for row in snapshots
        if row["rollout_all_call_mean_relative_l2"] is not None
        and row["fixed_seen_train_one_step_relative_l2"] is not None
    ]
    return {
        "snapshots": snapshots,
        "selected": selected_rows[0],
        "terminal": snapshots[-1],
        "overoptimization": _overoptimization_event(rollout_snapshots),
        "normalization_digest": str(summary["normalization_digest"]),
        "initial_full_state_sha256": str(
            summary["initialization_control"]["differential_branch"][
                "initial_full_state_sha256"
            ]
        ),
        "initial_nondifferential_state_sha256": str(
            summary["initialization_control"]["differential_branch"][
                "initial_nondifferential_state_sha256"
            ]
        ),
        "presentation_stream_digest": _mapping_digest(
            contract["presentation_stream_sha256_by_epoch"]
        ),
        "fixed_seen_pair_bank_sha256": str(contract["fixed_seen_pair_bank_sha256"]),
        "fixed_validation_pair_bank_sha256": str(
            contract["fixed_validation_pair_bank_sha256"]
        ),
        "checked_source_files": checked_source_files,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def shared_bootstrap_indices(
    population_size: int, draws: int, seed: int
) -> list[tuple[int, ...]]:
    if population_size <= 0 or draws <= 0:
        raise ValueError("bootstrap population and draws must be positive")
    generator = random.Random(seed)
    return [
        tuple(generator.randrange(population_size) for _ in range(population_size))
        for _ in range(draws)
    ]


def paired_ratio_bootstrap(
    numerator: Sequence[float],
    denominator: Sequence[float],
    index_draws: Sequence[Sequence[int]],
) -> dict[str, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap inputs must be nonempty and equal length")
    observed = statistics.fmean(numerator) / statistics.fmean(denominator)
    ratios = []
    for indices in index_draws:
        if len(indices) != len(numerator):
            raise ValueError("bootstrap draw has the wrong population size")
        left = statistics.fmean(numerator[index] for index in indices)
        right = statistics.fmean(denominator[index] for index in indices)
        ratios.append(left / right)
    return {
        "ratio": observed,
        "ci95_low": _percentile(ratios, 0.025),
        "ci95_high": _percentile(ratios, 0.975),
    }


def _trajectory_errors(cell: Mapping[str, Any]) -> dict[str, float]:
    rollout = cell.get("outside_selection_rollout")
    if not isinstance(rollout, Mapping):
        raise TypeError("audit cell lacks outside-selection rollout")
    trajectories = rollout.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise TypeError("audit rollout lacks trajectory rows")
    result = {}
    for row in trajectories:
        if not isinstance(row, Mapping):
            raise TypeError("audit trajectory row is malformed")
        result[str(row["trajectory"])] = float(row["mean_prefix_relative_l2"])
    if len(result) != len(trajectories):
        raise ValueError("audit rollout contains duplicate trajectories")
    return result


def _curve_rows(
    histories: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for trajectory_count in COUNTS:
        for architecture in ARCHITECTURES:
            history = histories[(trajectory_count, architecture)]
            for snapshot in history["snapshots"]:
                rows.append(
                    {
                        "trajectory_count": trajectory_count,
                        "architecture": architecture,
                        "epoch": snapshot["epoch"],
                        "optimizer_step": snapshot["optimizer_step"],
                        "window_equivalent_exposure": (
                            float(snapshot["optimizer_step"])
                            / (WINDOWS_PER_TRAJECTORY * trajectory_count)
                        ),
                        **{
                            name: snapshot[name]
                            for name in (
                                "learning_rate",
                                "online_train_one_step_relative_l2",
                                "fixed_seen_train_one_step_relative_l2",
                                "fixed_validation_one_step_relative_l2",
                                "rollout_all_call_mean_relative_l2",
                                "rollout_h79_relative_l2",
                                "rollout_completion_rate",
                                "rollout_hard_failure_count",
                                "physical_admissibility_rate",
                            )
                        },
                    }
                )
    return rows


def matched_exposure_rows(
    histories: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for multiplier in MATCHED_EXPOSURE_STEP_MULTIPLIERS:
        group = []
        for trajectory_count in COUNTS:
            step = multiplier * trajectory_count
            if step > OPTIMIZER_STEPS:
                continue
            for architecture in ARCHITECTURES:
                snapshots = histories[(trajectory_count, architecture)]["snapshots"]
                matches = [row for row in snapshots if int(row["optimizer_step"]) == step]
                if len(matches) != 1:
                    continue
                snapshot = matches[0]
                if (
                    snapshot["rollout_all_call_mean_relative_l2"] is None
                    or snapshot["fixed_seen_train_one_step_relative_l2"] is None
                ):
                    continue
                group.append(
                    {
                        "step_multiplier_per_trajectory": multiplier,
                        "window_equivalent_exposure": (
                            multiplier / WINDOWS_PER_TRAJECTORY
                        ),
                        "trajectory_count": trajectory_count,
                        "architecture": architecture,
                        "optimizer_step": step,
                        **{
                            name: snapshot[name]
                            for name in (
                                "online_train_one_step_relative_l2",
                                "fixed_seen_train_one_step_relative_l2",
                                "fixed_validation_one_step_relative_l2",
                                "rollout_all_call_mean_relative_l2",
                                "rollout_h79_relative_l2",
                                "physical_admissibility_rate",
                            )
                        },
                    }
                )
        counts = {int(row["trajectory_count"]) for row in group}
        if len(counts) >= 2 and len(group) == 2 * len(counts):
            rows.extend(group)
    return rows


def _cell_row(
    cell: Mapping[str, Any], history: Mapping[str, Any]
) -> dict[str, Any]:
    selected = history["selected"]
    terminal = history["terminal"]
    rollout = cell["outside_selection_rollout"]
    structure = cell["outside_selection_h79_structure_means"]
    selected_seen = float(selected["fixed_seen_train_one_step_relative_l2"])
    terminal_seen = float(terminal["fixed_seen_train_one_step_relative_l2"])
    return {
        "trajectory_count": int(cell["trajectory_count"]),
        "architecture": str(cell["architecture"]),
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "selected_online_train_one_step_relative_l2": selected[
            "online_train_one_step_relative_l2"
        ],
        "selected_fixed_seen_train_one_step_relative_l2": selected_seen,
        "selected_fixed_validation_one_step_relative_l2": selected[
            "fixed_validation_one_step_relative_l2"
        ],
        "selected_fixed_validation_to_seen_ratio": (
            float(selected["fixed_validation_one_step_relative_l2"]) / selected_seen
        ),
        "selected_rollout_all_call_mean_relative_l2": selected[
            "rollout_all_call_mean_relative_l2"
        ],
        "selected_rollout_h79_relative_l2": selected["rollout_h79_relative_l2"],
        "terminal_optimizer_step": int(terminal["optimizer_step"]),
        "terminal_fixed_seen_train_one_step_relative_l2": terminal_seen,
        "terminal_fixed_validation_one_step_relative_l2": terminal[
            "fixed_validation_one_step_relative_l2"
        ],
        "terminal_fixed_validation_to_seen_ratio": (
            float(terminal["fixed_validation_one_step_relative_l2"]) / terminal_seen
        ),
        "terminal_rollout_all_call_mean_relative_l2": terminal[
            "rollout_all_call_mean_relative_l2"
        ],
        "terminal_rollout_h79_relative_l2": terminal["rollout_h79_relative_l2"],
        "selected_to_terminal_fixed_seen_change": terminal_seen / selected_seen - 1.0,
        "selected_to_terminal_fixed_validation_change": (
            float(terminal["fixed_validation_one_step_relative_l2"])
            / float(selected["fixed_validation_one_step_relative_l2"])
            - 1.0
        ),
        "selected_to_terminal_rollout_change": (
            float(terminal["rollout_all_call_mean_relative_l2"])
            / float(selected["rollout_all_call_mean_relative_l2"])
            - 1.0
        ),
        "one_step_rollout_divergence_after_selection": bool(
            terminal_seen < selected_seen
            and float(terminal["fixed_validation_one_step_relative_l2"])
            < float(selected["fixed_validation_one_step_relative_l2"])
            and float(terminal["rollout_all_call_mean_relative_l2"])
            > float(selected["rollout_all_call_mean_relative_l2"])
        ),
        "registered_one_step_overoptimization_detected": bool(
            history["overoptimization"]["detected"]
        ),
        "audit_rollout_all_call_mean_relative_l2": rollout[
            "mean_full_horizon_relative_l2"
        ],
        "audit_rollout_h79_relative_l2": rollout["mean_endpoint_relative_l2"][
            "79"
        ],
        "audit_rollout_physical_admissibility_rate": rollout[
            "physical_admissibility_rate"
        ],
        **{f"audit_h79_{name}": value for name, value in structure.items()},
    }


def analyze_ladder(
    fresh_root: Path,
    n256_gate_root: Path,
    audit_root: Path,
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _check_completion_receipts(fresh_root, n256_gate_root)
    audit_verification = verify_artifact_manifest(audit_root)
    summary = _load_json(audit_root / "summary.json")
    if summary.get("schema") != AUDIT_SCHEMA or summary.get("status") != "complete":
        raise ValueError("unsupported or incomplete D094 ladder audit")
    if summary.get("historical_test_population_accessed") is not False:
        raise ValueError("D094 audit accessed the historical test population")
    if summary.get("split_partition_digest") != SPLIT_PARTITION_DIGEST:
        raise ValueError("D094 audit split binding changed")
    if summary.get("checkpoint_source_set_digests") != {
        "fresh_n8_to_n128": FRESH_SOURCE_SET_DIGEST,
        "reused_b1a_n256": B1A_SOURCE_SET_DIGEST,
    }:
        raise ValueError("D094 audit checkpoint source identities changed")
    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise ValueError("D094 audit does not contain 12 cells")
    by_identity = {
        (int(cell["trajectory_count"]), str(cell["architecture"])): cell
        for cell in cells
    }
    expected = {(count, architecture) for count in COUNTS for architecture in ARCHITECTURES}
    if set(by_identity) != expected:
        raise ValueError("D094 audit cell matrix changed")

    histories: dict[tuple[int, str], dict[str, Any]] = {}
    paired_control_fields = (
        "normalization_digest",
        "initial_full_state_sha256",
        "initial_nondifferential_state_sha256",
        "presentation_stream_digest",
        "fixed_seen_pair_bank_sha256",
        "fixed_validation_pair_bank_sha256",
    )
    paired_controls: dict[int, dict[str, set[str]]] = {
        count: {name: set() for name in paired_control_fields} for count in COUNTS
    }
    for identity in sorted(expected):
        trajectory_count, _architecture = identity
        cell = by_identity[identity]
        base = fresh_root if trajectory_count < 256 else n256_gate_root
        run_dir = base / "outputs" / str(cell["run"])
        expected_source = (
            FRESH_SOURCE_SET_DIGEST
            if trajectory_count < 256
            else B1A_SOURCE_SET_DIGEST
        )
        history = _load_history(
            run_dir, cell, expected_source_set_digest=expected_source
        )
        histories[identity] = history
        for name in paired_control_fields:
            paired_controls[trajectory_count][name].add(str(history[name]))
    drifted_controls = {
        count: [name for name, values in fields.items() if len(values) != 1]
        for count, fields in paired_controls.items()
    }
    drifted_controls = {
        count: names for count, names in drifted_controls.items() if names
    }
    if drifted_controls:
        raise ValueError(f"paired architecture controls changed: {drifted_controls}")

    outside_keys = [str(key) for key in summary["outside_selection_keys"]]
    index_draws = shared_bootstrap_indices(
        len(outside_keys), bootstrap_draws, bootstrap_seed
    )
    architecture_rows = []
    for trajectory_count in COUNTS:
        pcno_cell = by_identity[(trajectory_count, "pcno")]
        pcfno_cell = by_identity[(trajectory_count, "pcfno")]
        pcno_errors = _trajectory_errors(pcno_cell)
        pcfno_errors = _trajectory_errors(pcfno_cell)
        if set(pcno_errors) != set(outside_keys) or set(pcfno_errors) != set(
            outside_keys
        ):
            raise ValueError("outside-selection trajectory identities changed")
        pcno_values = [pcno_errors[key] for key in outside_keys]
        pcfno_values = [pcfno_errors[key] for key in outside_keys]
        bootstrap = paired_ratio_bootstrap(pcno_values, pcfno_values, index_draws)
        pcno_history = histories[(trajectory_count, "pcno")]
        pcfno_history = histories[(trajectory_count, "pcfno")]
        pcno_selected = pcno_history["selected"]
        pcfno_selected = pcfno_history["selected"]
        pcno_terminal = pcno_history["terminal"]
        pcfno_terminal = pcfno_history["terminal"]
        pcno_structure = pcno_cell["outside_selection_h79_structure_means"]
        pcfno_structure = pcfno_cell["outside_selection_h79_structure_means"]
        architecture_rows.append(
            {
                "trajectory_count": trajectory_count,
                "selected_fixed_seen_pcno_over_pcfno": (
                    float(pcno_selected["fixed_seen_train_one_step_relative_l2"])
                    / float(pcfno_selected["fixed_seen_train_one_step_relative_l2"])
                ),
                "selected_fixed_validation_pcno_over_pcfno": (
                    float(pcno_selected["fixed_validation_one_step_relative_l2"])
                    / float(pcfno_selected["fixed_validation_one_step_relative_l2"])
                ),
                "selection_rollout_pcno_over_pcfno": (
                    float(pcno_selected["rollout_all_call_mean_relative_l2"])
                    / float(pcfno_selected["rollout_all_call_mean_relative_l2"])
                ),
                "terminal_rollout_pcno_over_pcfno": (
                    float(pcno_terminal["rollout_all_call_mean_relative_l2"])
                    / float(pcfno_terminal["rollout_all_call_mean_relative_l2"])
                ),
                "audit_rollout_pcno_over_pcfno": bootstrap["ratio"],
                "audit_rollout_ratio_ci95_low": bootstrap["ci95_low"],
                "audit_rollout_ratio_ci95_high": bootstrap["ci95_high"],
                "audit_pcno_trajectory_wins": sum(
                    left < right
                    for left, right in zip(pcno_values, pcfno_values, strict=True)
                ),
                "audit_trajectory_count": len(outside_keys),
                "audit_h79_pcno_over_pcfno": (
                    float(
                        pcno_cell["outside_selection_rollout"][
                            "mean_endpoint_relative_l2"
                        ]["79"]
                    )
                    / float(
                        pcfno_cell["outside_selection_rollout"][
                            "mean_endpoint_relative_l2"
                        ]["79"]
                    )
                ),
                "audit_admissibility_pcno_minus_pcfno": (
                    float(
                        pcno_cell["outside_selection_rollout"][
                            "physical_admissibility_rate"
                        ]
                    )
                    - float(
                        pcfno_cell["outside_selection_rollout"][
                            "physical_admissibility_rate"
                        ]
                    )
                ),
                "audit_front_centroid_pcno_over_pcfno": (
                    float(pcno_structure["front_centroid_distance"])
                    / float(pcfno_structure["front_centroid_distance"])
                ),
                "audit_front_iou_pcno_minus_pcfno": (
                    float(pcno_structure["front_iou"])
                    - float(pcfno_structure["front_iou"])
                ),
                "audit_front_chamfer_pcno_over_pcfno": (
                    float(pcno_structure["front_symmetric_chamfer"])
                    / float(pcfno_structure["front_symmetric_chamfer"])
                ),
                "audit_shock_thickness_pcno_over_pcfno": (
                    float(pcno_structure["shock_thickness_log_error"])
                    / float(pcfno_structure["shock_thickness_log_error"])
                ),
                "audit_shock_strength_pcno_over_pcfno": (
                    float(pcno_structure["shock_strength_log_error"])
                    / float(pcfno_structure["shock_strength_log_error"])
                ),
                "audit_smooth_highpass_pcno_over_pcfno": (
                    float(
                        pcno_structure[
                            "smooth_highpass_energy_reconstructed_weight_proxy"
                        ]
                    )
                    / float(
                        pcfno_structure[
                            "smooth_highpass_energy_reconstructed_weight_proxy"
                        ]
                    )
                ),
                "audit_proxy_total_pcno_over_pcfno": (
                    float(
                        pcno_structure[
                            "conserved_total_scaled_rmse_reconstructed_weight_proxy"
                        ]
                    )
                    / float(
                        pcfno_structure[
                            "conserved_total_scaled_rmse_reconstructed_weight_proxy"
                        ]
                    )
                ),
            }
        )

    cell_rows = [
        _cell_row(by_identity[identity], histories[identity])
        for identity in sorted(expected)
    ]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": "single_seed_bump_development_scaling_analysis",
        "historical_test_population_accessed": False,
        "counts": list(COUNTS),
        "architectures": list(ARCHITECTURES),
        "optimizer_steps": OPTIMIZER_STEPS,
        "window_equivalent_exposure_definition": "optimizer_step / (79 * n)",
        "audit_artifact_verification": audit_verification,
        "audit_summary_sha256": _sha256_file(audit_root / "summary.json"),
        "training_source_set_digests": {
            "fresh_n8_to_n128": FRESH_SOURCE_SET_DIGEST,
            "reused_b1a_n256": B1A_SOURCE_SET_DIGEST,
        },
        "split_partition_digest": SPLIT_PARTITION_DIGEST,
        "paired_training_controls_verified_for_every_n": list(
            paired_control_fields
        ),
        "bootstrap": {
            "unit": "outside_selection_trajectory",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "shared_resample_stream_across_counts": True,
            "status": "descriptive_post_hoc",
        },
        "cell_rows": cell_rows,
        "architecture_rows": architecture_rows,
        "matched_exposure_rows": matched_exposure_rows(histories),
        "training_curves": _curve_rows(histories),
        "claims_not_supported": [
            "multi-seed architecture or scaling law",
            "causal gradient-path attribution",
            "physical conservation from reconstructed proxy weights",
            "untouched holdout or historical-test performance",
            "capacity limitation without compute and capacity controls",
        ],
    }


def write_analysis(analysis: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("analysis output directory must not already exist")
    output_dir.mkdir(parents=True)
    payload = dict(analysis)
    curve_rows = payload.pop("training_curves")
    cell_rows = payload.pop("cell_rows")
    architecture_rows = payload.pop("architecture_rows")
    exposure_rows = payload.pop("matched_exposure_rows")
    _write_json(output_dir / "analysis.json", payload)
    _write_csv(output_dir / "training_curves.csv", curve_rows)
    _write_csv(output_dir / "selected_terminal_audit.csv", cell_rows)
    _write_csv(output_dir / "architecture_ratios.csv", architecture_rows)
    _write_csv(output_dir / "matched_exposure.csv", exposure_rows)
    names = (
        "analysis.json",
        "training_curves.csv",
        "selected_terminal_audit.csv",
        "architecture_ratios.csv",
        "matched_exposure.csv",
    )
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256_file(output_dir / name),
            }
            for name in names
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    analysis = analyze_ladder(
        args.fresh_ladder_root,
        args.n256_gate_root,
        args.audit_root,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    manifest = write_analysis(analysis, args.output_dir)
    return {"analysis": analysis, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    result = run(argv)
    analysis = result["analysis"]
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "counts": analysis["counts"],
                "architectures": analysis["architectures"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

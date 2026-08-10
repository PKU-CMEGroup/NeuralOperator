#!/usr/bin/env python3
"""Continue finite-inadmissible bump PCNO rollouts under the frozen contract.

The evaluator is diagnostic and read-only with respect to checkpoints.  It
does not floor, clip, smooth, limit, or change the registered causal nodal
boundary policy.  Each call records the input projection, raw model proposal,
and deployed output projection separately.  A rollout continues after finite
Euler inadmissibility and stops only after a deployed nonfinite state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_RESIDUAL_NONE,
    BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    build_graph_causal_boundary_policy,
)
from utility.time_dependent_no.pcno_inadmissibility import (
    BLOWUP_AMPLITUDE_RATIO,
    BLOWUP_RELATIVE_L2,
    stage_transition_label,
    state_diagnostics,
    trajectory_event_summary,
    weighted_rms,
)
from utility.time_dependent_no.pcno_rollout import (
    boundary_outflow_normal_mach,
    contract_forward_sample,
)
from utility.time_dependent_no.pcno_runtime import (
    autocast_context,
    build_checkpoint_model,
    load_checkpoint,
    select_device,
)

SCHEMA = "pcno_inadmissibility_continuation_v1"
SEMANTIC_NAMES = ("wall", "outflow", "inflow")
DEFAULT_SAVE_CASES = ("23", "54", "128")
SAVE_VARIANTS = ("N0_correct", "D082_correct", "D082_zero_inflow")
SOURCE_BINDING_FILES = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_rollout.py",
    "utility/time_dependent_no/pcno_runtime.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--d082-run-dir", type=Path, required=True)
    parser.add_argument("--n0-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--num-steps", type=int, default=79)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument(
        "--save-case-keys",
        nargs="*",
        default=list(DEFAULT_SAVE_CASES),
        help="Predeclared open-validation cases whose all-frame arrays are retained.",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.num_steps != 79:
        raise ValueError("this registered diagnostic requires exactly 79 model calls")
    if args.repeat_count != 2:
        raise ValueError("this registered diagnostic requires exactly two repeats")
    digest = str(args.expected_data_manifest_digest).lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("--expected-data-manifest-digest must be a SHA-256")
    args.expected_data_manifest_digest = digest
    args.save_case_keys = [str(value) for value in args.save_case_keys]
    return args


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(payload)


def selected_history_row(run_dir: Path, best_epoch: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row["epoch"]) == best_epoch]
    if len(selected) != 1 or not isinstance(selected[0].get("rollout"), Mapping):
        raise ValueError(f"{run_dir} lacks one selected rollout row")
    return dict(selected[0])


class FieldInterventionModel(nn.Module):
    """Delegate to frozen D082 after zeroing selected semantic fields."""

    def __init__(
        self, base: PCNOEuler2DResidual, zero_field_indices: Sequence[int]
    ) -> None:
        super().__init__()
        self.base = base
        self.zero_field_indices = tuple(int(index) for index in zero_field_indices)
        if not self.zero_field_indices:
            raise ValueError("an intervention must zero at least one field")
        if any(
            index < 0 or index >= len(SEMANTIC_NAMES)
            for index in self.zero_field_indices
        ):
            raise ValueError("semantic field index is out of range")
        self.model_node_type_input = str(base.model_node_type_input)

    @property
    def gamma(self) -> float:
        return float(self.base.gamma)

    @property
    def state_scale(self) -> torch.Tensor:
        return self.base.state_scale

    def forward(
        self, current_conservative: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        fields = kwargs.pop("boundary_features", None)
        if fields is None:
            raise ValueError("D082 field intervention requires boundary_features")
        intervened = fields.clone()
        intervened[..., list(self.zero_field_indices)] = 0.0
        return self.base(
            current_conservative,
            boundary_features=intervened,
            **kwargs,
        )


def verify_bound_source(d082_run_dir: Path) -> dict[str, Any]:
    manifest_path = d082_run_dir / "source_snapshot" / "manifest.json"
    manifest = read_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        raise TypeError("D082 source snapshot lacks file records")
    checks: dict[str, Any] = {}
    incompatible: list[str] = []
    for relative_name in SOURCE_BINDING_FILES:
        expected = str(records[relative_name]["sha256"])
        current_path = ROOT / relative_name
        observed = sha256_file(current_path)
        checks[relative_name] = {"expected": expected, "observed": observed}
        if observed != expected:
            incompatible.append(relative_name)
    if incompatible:
        raise ValueError(
            f"active model/rollout source differs from D082 snapshot: {incompatible}"
        )
    return {
        "manifest_name": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "source_set_digest": manifest.get("source_set_digest"),
        "files": checks,
    }


def load_bound_model(
    run_dir: Path,
    *,
    device: torch.device,
    expected_manifest: str,
    expected_side_mode: str,
) -> tuple[PCNOEuler2DResidual, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    summary = read_json(run_dir / "summary.json")
    split = read_json(run_dir / "split.json")
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != str(summary["artifact_sha256"]["best_checkpoint"]):
        raise ValueError(f"{run_dir.name} checkpoint hash differs")
    if str(summary["data_manifest_digest"]) != expected_manifest:
        raise ValueError(f"{run_dir.name} data manifest differs")
    if split.get("test_keys") != []:
        raise ValueError(f"{run_dir.name} exposes a sealed/test population")
    checkpoint = load_checkpoint(checkpoint_path)
    if str(checkpoint["data_manifest_digest"]) != expected_manifest:
        raise ValueError(f"{run_dir.name} checkpoint data manifest differs")
    model, _ = build_checkpoint_model(
        checkpoint,
        device,
        model_node_type_input=str(checkpoint["model_node_type_input"]),
    )
    if model.boundary_residual_mode != expected_side_mode:
        raise ValueError(f"{run_dir.name} boundary-residual mode differs")
    if str(summary["boundary_mode"]) != "causal_nodal_physical":
        raise ValueError(f"{run_dir.name} boundary policy differs")
    if bool(summary["raw_recurrence"]):
        raise ValueError(f"{run_dir.name} recurrence differs")
    selected = selected_history_row(run_dir, int(summary["best_epoch"]))
    return model, summary, split, selected, checkpoint_hash


def _safe_minimum(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    finite = value[torch.isfinite(value)]
    return None if finite.numel() == 0 else float(finite.min().detach().cpu())


def _prefix(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _first_invalid_location(
    invalid: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    nodes = torch.nonzero(invalid[0], as_tuple=False).reshape(-1)
    if nodes.numel() == 0:
        return {
            "first_invalid_node": None,
            "first_invalid_node_type": None,
            "first_invalid_x": None,
            "first_invalid_y": None,
            "first_invalid_in_union_collar": None,
        }
    index = int(nodes[0].detach().cpu())
    node_type = sample["node_type"]
    if node_type.ndim == 3:
        node_type = node_type[..., 0]
    position = sample["nodes"][0, index]
    fields = sample.get("boundary_features")
    in_collar = None
    if fields is not None:
        in_collar = bool(float(fields[0, index].max().detach().cpu()) > 0.0)
    return {
        "first_invalid_node": index,
        "first_invalid_node_type": int(node_type[0, index].detach().cpu()),
        "first_invalid_x": float(position[0].detach().cpu()),
        "first_invalid_y": float(position[1].detach().cpu()),
        "first_invalid_in_union_collar": in_collar,
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


@torch.inference_mode()
def run_case_once(
    model: nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    variant: str,
    repeat: int,
    num_steps: int,
    device: torch.device,
    amp: str,
    boundary_policy: Mapping[str, Any],
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    residual_scale: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    states = store.states(key)
    if states.shape[0] < num_steps + 1:
        raise ValueError(f"trajectory {key} has fewer than {num_steps + 1} frames")
    reference = np.asarray(states[: num_steps + 1], dtype=np.float32)
    if not np.isfinite(reference).all():
        raise ValueError(f"trajectory {key} reference is nonfinite")
    reference_max_abs = float(np.max(np.abs(reference)))
    sample = store.tensor_sample(key, 0, step_stride=1, device=device)
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
    current = sample["current"]
    node_count = reference.shape[1]
    deployed_states = np.full((num_steps + 1, node_count, 4), np.nan, np.float32)
    raw_proposals = np.full((num_steps, node_count, 4), np.nan, np.float32)
    model_currents = np.full((num_steps, node_count, 4), np.nan, np.float32)
    deployed_states[0] = current[0].detach().float().cpu().numpy()
    node_type = sample["node_type"]
    if node_type.ndim == 3:
        node_type = node_type[..., 0]
    boundary_mask = (node_type != 0).to(dtype=sample["node_mask"].dtype).unsqueeze(-1)
    boundary_mask = boundary_mask * sample["node_mask"]
    rows: list[dict[str, Any]] = []
    model.eval()

    for call_index in range(num_steps):
        call = call_index + 1
        previous_reference = reference_tensor[call_index].unsqueeze(0)
        target = reference_tensor[call].unsqueeze(0)
        with autocast_context(device, amp):
            deployed, raw, model_current = contract_forward_sample(
                model,
                sample,
                current,
                boundary_policy=boundary_policy,
            )
        current_metrics, _ = state_diagnostics(
            current,
            target=previous_reference,
            node_weights=sample["node_weights"],
            node_mask=sample["node_mask"],
            state_mean=state_mean,
            state_scale=state_scale,
            reference_max_abs=reference_max_abs,
            gamma=model.gamma,
        )
        model_current_metrics, model_current_invalid = state_diagnostics(
            model_current,
            target=previous_reference,
            node_weights=sample["node_weights"],
            node_mask=sample["node_mask"],
            state_mean=state_mean,
            state_scale=state_scale,
            reference_max_abs=reference_max_abs,
            gamma=model.gamma,
        )
        raw_metrics, raw_invalid = state_diagnostics(
            raw,
            target=target,
            node_weights=sample["node_weights"],
            node_mask=sample["node_mask"],
            state_mean=state_mean,
            state_scale=state_scale,
            reference_max_abs=reference_max_abs,
            gamma=model.gamma,
        )
        deployed_metrics, deployed_invalid = state_diagnostics(
            deployed,
            target=target,
            node_weights=sample["node_weights"],
            node_mask=sample["node_mask"],
            state_mean=state_mean,
            state_scale=state_scale,
            reference_max_abs=reference_max_abs,
            gamma=model.gamma,
        )

        reference_increment = target - previous_reference
        raw_increment = raw - model_current
        deployed_increment = deployed - current
        reference_increment_rms = weighted_rms(
            reference_increment,
            sample["node_weights"],
            sample["node_mask"],
            component_scale=residual_scale,
        )
        raw_increment_rms = weighted_rms(
            raw_increment,
            sample["node_weights"],
            sample["node_mask"],
            component_scale=residual_scale,
        )
        deployed_increment_rms = weighted_rms(
            deployed_increment,
            sample["node_weights"],
            sample["node_mask"],
            component_scale=residual_scale,
        )
        raw_increment_error = weighted_rms(
            raw_increment - reference_increment,
            sample["node_weights"],
            sample["node_mask"],
            component_scale=residual_scale,
        )
        deployed_increment_error = weighted_rms(
            deployed_increment - reference_increment,
            sample["node_weights"],
            sample["node_mask"],
            component_scale=residual_scale,
        )
        input_transition = stage_transition_label(
            str(current_metrics["failure_cause"]),
            str(model_current_metrics["failure_cause"]),
            stage="input_projection",
        )
        model_transition = stage_transition_label(
            str(model_current_metrics["failure_cause"]),
            str(raw_metrics["failure_cause"]),
            stage="model",
        )
        output_transition = stage_transition_label(
            str(raw_metrics["failure_cause"]),
            str(deployed_metrics["failure_cause"]),
            stage="output_projection",
        )
        recurrence_reentry = (
            current_metrics["failure_cause"] != "admissible"
            and deployed_metrics["failure_cause"] == "admissible"
        )
        recovery_stages = [
            name
            for name, transition in (
                ("input_projection", input_transition),
                ("model", model_transition),
                ("output_projection", output_transition),
            )
            if transition.endswith("_recovery")
        ]
        raw_outflow = _safe_minimum(
            boundary_outflow_normal_mach(raw.float(), boundary_policy)
        )
        deployed_outflow = _safe_minimum(
            boundary_outflow_normal_mach(deployed.float(), boundary_policy)
        )
        row = {
            "trajectory": key,
            "variant": variant,
            "repeat": repeat,
            "call": call,
            "physical_time": float(call * float(store.manifest["dt"])),
            "input_projection_transition": input_transition,
            "model_transition": model_transition,
            "output_projection_transition": output_transition,
            "recurrence_reentry": recurrence_reentry,
            "recurrence_reentry_recovery_stages": recovery_stages,
            "reference_increment_proxy_scaled_rms": reference_increment_rms,
            "raw_increment_proxy_scaled_rms": raw_increment_rms,
            "deployed_increment_proxy_scaled_rms": deployed_increment_rms,
            "raw_increment_to_reference_ratio": _ratio(
                raw_increment_rms, reference_increment_rms
            ),
            "deployed_increment_to_reference_ratio": _ratio(
                deployed_increment_rms, reference_increment_rms
            ),
            "raw_increment_error_proxy_scaled_rms": raw_increment_error,
            "deployed_increment_error_proxy_scaled_rms": deployed_increment_error,
            "raw_increment_error_to_reference_ratio": _ratio(
                raw_increment_error, reference_increment_rms
            ),
            "deployed_increment_error_to_reference_ratio": _ratio(
                deployed_increment_error, reference_increment_rms
            ),
            "input_boundary_projection_proxy_scaled_rms": weighted_rms(
                model_current - current,
                sample["node_weights"],
                boundary_mask,
                component_scale=state_scale,
            ),
            "output_boundary_projection_proxy_scaled_rms": weighted_rms(
                deployed - raw,
                sample["node_weights"],
                boundary_mask,
                component_scale=state_scale,
            ),
            "raw_outflow_normal_mach_min": raw_outflow,
            "deployed_outflow_normal_mach_min": deployed_outflow,
            **_prefix("current", current_metrics),
            **_prefix("model_current", model_current_metrics),
            **_prefix("raw", raw_metrics),
            **_prefix("deployed", deployed_metrics),
            **_prefix(
                "model_current", _first_invalid_location(model_current_invalid, sample)
            ),
            **_prefix("raw", _first_invalid_location(raw_invalid, sample)),
            **_prefix("deployed", _first_invalid_location(deployed_invalid, sample)),
        }
        rows.append(row)
        model_currents[call_index] = model_current[0].detach().float().cpu().numpy()
        raw_proposals[call_index] = raw[0].detach().float().cpu().numpy()
        deployed_states[call] = deployed[0].detach().float().cpu().numpy()
        current = deployed
        if deployed_metrics["failure_cause"] == "nonfinite_state":
            break

    events = trajectory_event_summary(rows)
    strict_failure_call: int | None = None
    strict_failure_cause = "completed"
    for row in rows:
        if row["deployed_failure_cause"] != "admissible":
            strict_failure_call = int(row["call"])
            strict_failure_cause = str(row["deployed_failure_cause"])
            break
        outflow = row["deployed_outflow_normal_mach_min"]
        if outflow is None or not math.isfinite(float(outflow)):
            strict_failure_call = int(row["call"])
            strict_failure_cause = "nonfinite_outflow_normal_mach"
            break
        if float(outflow) <= 1.0:
            strict_failure_call = int(row["call"])
            strict_failure_cause = "non_supersonic_outflow"
            break
    strict_valid_length = (
        num_steps if strict_failure_call is None else strict_failure_call - 1
    )
    strict_errors = [
        float(row["deployed_proxy_scaled_relative_l2"])
        for row in rows[:strict_valid_length]
        if row["deployed_proxy_scaled_relative_l2"] is not None
    ]
    first_row = (
        None
        if events["first_inadmissible_call"] is None
        else rows[int(events["first_inadmissible_call"]) - 1]
    )
    events.update(
        {
            "trajectory": key,
            "variant": variant,
            "repeat": repeat,
            "strict_valid_length": strict_valid_length,
            "strict_completed": strict_valid_length == num_steps,
            "strict_failure_call": strict_failure_call,
            "strict_failure_cause": strict_failure_cause,
            "strict_survival_fraction": strict_valid_length / num_steps,
            "strict_mean_prefix_relative_l2": (
                None if not strict_errors else float(np.mean(strict_errors))
            ),
            "strict_final_relative_l2": (
                None if not strict_errors else strict_errors[-1]
            ),
            "first_inadmissible_node": (
                None if first_row is None else first_row["deployed_first_invalid_node"]
            ),
            "first_inadmissible_node_type": (
                None
                if first_row is None
                else first_row["deployed_first_invalid_node_type"]
            ),
            "first_inadmissible_x": (
                None if first_row is None else first_row["deployed_first_invalid_x"]
            ),
            "first_inadmissible_y": (
                None if first_row is None else first_row["deployed_first_invalid_y"]
            ),
            "first_inadmissible_in_union_collar": (
                None
                if first_row is None
                else first_row["deployed_first_invalid_in_union_collar"]
            ),
            "recurrence_reentry_calls": [
                int(row["call"]) for row in rows if row["recurrence_reentry"]
            ],
        }
    )
    arrays = {
        "deployed_states": deployed_states,
        "raw_proposals": raw_proposals,
        "model_currents": model_currents,
    }
    return rows, events, arrays


def compare_repeats(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    first_events: Mapping[str, Any],
    second_events: Mapping[str, Any],
) -> dict[str, Any]:
    left = np.asarray(first["deployed_states"], dtype=np.float32)
    right = np.asarray(second["deployed_states"], dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("repeat rollout shapes differ")
    common_finite = np.isfinite(left) & np.isfinite(right)
    max_abs = (
        None
        if not np.any(common_finite)
        else float(np.max(np.abs(left[common_finite] - right[common_finite])))
    )
    return {
        "trajectory": str(first_events["trajectory"]),
        "variant": str(first_events["variant"]),
        "finite_mask_exact": bool(
            np.array_equal(np.isfinite(left), np.isfinite(right))
        ),
        "deployed_states_exact_equal_nan": bool(
            np.array_equal(left, right, equal_nan=True)
        ),
        "deployed_states_max_abs_common_finite": max_abs,
        "first_inadmissible_call_equal": (
            first_events["first_inadmissible_call"]
            == second_events["first_inadmissible_call"]
        ),
        "registered_blowup_equal": (
            first_events["registered_blowup"] == second_events["registered_blowup"]
        ),
        "strict_valid_length_equal": (
            first_events["strict_valid_length"] == second_events["strict_valid_length"]
        ),
        "repeat_0_first_inadmissible_call": first_events["first_inadmissible_call"],
        "repeat_1_first_inadmissible_call": second_events["first_inadmissible_call"],
        "repeat_0_strict_valid_length": first_events["strict_valid_length"],
        "repeat_1_strict_valid_length": second_events["strict_valid_length"],
    }


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_events(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty event population")
    invalid = [row for row in rows if bool(row["ever_inadmissible"])]
    never_invalid = [row for row in rows if not bool(row["ever_inadmissible"])]
    blowup = [row for row in rows if bool(row["registered_blowup"])]
    recovered = [row for row in rows if int(row["recovered_episode_count"]) > 0]
    terminal_invalid = [row for row in rows if not bool(row["terminal_admissible"])]
    strict_completed = [row for row in rows if bool(row["strict_completed"])]
    type_counts = Counter(
        str(int(row["first_inadmissible_node_type"]))
        for row in invalid
        if row["first_inadmissible_node_type"] is not None
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    invalid_blowup = sum(bool(row["registered_blowup"]) for row in invalid)
    never_invalid_blowup = sum(bool(row["registered_blowup"]) for row in never_invalid)
    prefix_errors = [
        float(row["strict_mean_prefix_relative_l2"])
        for row in rows
        if row["strict_mean_prefix_relative_l2"] is not None
    ]
    final_errors = [
        float(row["final_proxy_scaled_relative_l2"])
        for row in rows
        if row["final_proxy_scaled_relative_l2"] is not None
    ]
    return {
        "trajectories": len(rows),
        "strict_completion_count": len(strict_completed),
        "strict_completion_rate": len(strict_completed) / len(rows),
        "mean_strict_survival_fraction": _mean(
            [float(row["strict_survival_fraction"]) for row in rows]
        ),
        "mean_strict_selection_relative_l2": _mean(prefix_errors),
        "ever_inadmissible_count": len(invalid),
        "ever_inadmissible_cases": [str(row["trajectory"]) for row in invalid],
        "recovered_inadmissibility_count": len(recovered),
        "recovered_inadmissibility_cases": [
            str(row["trajectory"]) for row in recovered
        ],
        "terminal_inadmissible_count": len(terminal_invalid),
        "terminal_inadmissible_cases": [
            str(row["trajectory"]) for row in terminal_invalid
        ],
        "registered_blowup_count": len(blowup),
        "registered_blowup_cases": [str(row["trajectory"]) for row in blowup],
        "invalid_without_registered_blowup_count": len(invalid) - invalid_blowup,
        "registered_blowup_given_ever_inadmissible": rate(invalid_blowup, len(invalid)),
        "registered_blowup_given_never_inadmissible": rate(
            never_invalid_blowup, len(never_invalid)
        ),
        "first_inadmissible_node_type_counts": dict(sorted(type_counts.items())),
        "first_inadmissible_in_union_collar_count": sum(
            row["first_inadmissible_in_union_collar"] is True for row in invalid
        ),
        "input_projection_recovery_calls": sum(
            int(row["input_projection_recovery_calls"]) for row in rows
        ),
        "model_recovery_calls": sum(int(row["model_recovery_calls"]) for row in rows),
        "output_projection_recovery_calls": sum(
            int(row["output_projection_recovery_calls"]) for row in rows
        ),
        "mean_final_proxy_scaled_relative_l2": _mean(final_errors),
        "maximum_proxy_scaled_relative_l2": max(
            (
                float(row["maximum_proxy_scaled_relative_l2"])
                for row in rows
                if row["maximum_proxy_scaled_relative_l2"] is not None
            ),
            default=None,
        ),
        "maximum_amplitude_ratio": max(
            (
                float(row["maximum_amplitude_ratio"])
                for row in rows
                if row["maximum_amplitude_ratio"] is not None
            ),
            default=None,
        ),
    }


def reproduction_audit(
    observed_events: Sequence[Mapping[str, Any]],
    selected_history: Mapping[str, Any],
) -> dict[str, Any]:
    selected_rollout = selected_history["rollout"]
    selected_rows = {
        str(row["trajectory"]): row for row in selected_rollout["trajectories"]
    }
    observed_rows = {str(row["trajectory"]): row for row in observed_events}
    if set(selected_rows) != set(observed_rows):
        raise ValueError("selected-history and diagnostic trajectory keys differ")
    case_rows = []
    for key in sorted(observed_rows, key=lambda value: int(value)):
        selected = selected_rows[key]
        observed = observed_rows[key]
        case_rows.append(
            {
                "trajectory": key,
                "selected_valid_length": int(selected["valid_length"]),
                "observed_valid_length": int(observed["strict_valid_length"]),
                "valid_length_delta": int(observed["strict_valid_length"])
                - int(selected["valid_length"]),
                "failure_cause_equal": str(selected["failure_cause"])
                == str(observed["strict_failure_cause"]),
            }
        )
    aggregate = aggregate_events(observed_events)
    observed_survival = float(aggregate["mean_strict_survival_fraction"])
    observed_error = float(aggregate["mean_strict_selection_relative_l2"])
    return {
        "selected_completion_rate": float(selected_rollout["completion_rate"]),
        "observed_completion_rate": aggregate["strict_completion_rate"],
        "completion_rate_delta": aggregate["strict_completion_rate"]
        - float(selected_rollout["completion_rate"]),
        "selected_mean_survival_fraction": float(
            selected_rollout["mean_survival_fraction"]
        ),
        "observed_mean_survival_fraction": observed_survival,
        "mean_survival_fraction_delta": observed_survival
        - float(selected_rollout["mean_survival_fraction"]),
        "selected_mean_selection_relative_l2": float(
            selected_rollout["mean_selection_relative_l2"]
        ),
        "observed_mean_selection_relative_l2": observed_error,
        "mean_selection_relative_l2_delta": observed_error
        - float(selected_rollout["mean_selection_relative_l2"]),
        "exact_valid_length_case_count": sum(
            int(row["valid_length_delta"]) == 0 for row in case_rows
        ),
        "exact_failure_cause_case_count": sum(
            bool(row["failure_cause_equal"]) for row in case_rows
        ),
        "case_rows": case_rows,
    }


def save_case_bundle(
    path: Path,
    store: PCNOEuler2DShardStore,
    key: str,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    events: Sequence[Mapping[str, Any]],
    *,
    num_steps: int,
    checkpoint_hashes: Mapping[str, str],
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    residual_scale: torch.Tensor,
) -> None:
    geometry = store.geometry_numpy(key)
    payload: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "trajectory": np.asarray(key),
        "reference_states": np.asarray(
            store.states(key)[: num_steps + 1], dtype=np.float32
        ),
        "positions": np.asarray(geometry["nodes"], dtype=np.float32),
        "node_type": np.asarray(geometry["node_type"], dtype=np.int64),
        "node_weights_proxy": np.asarray(geometry["node_weights"], dtype=np.float32),
        "boundary_features": np.asarray(
            geometry["boundary_features"], dtype=np.float32
        ),
        "physical_times": np.arange(num_steps + 1, dtype=np.float64)
        * float(store.manifest["dt"]),
        "state_mean": state_mean.detach().float().cpu().numpy().reshape(4),
        "state_scale": state_scale.detach().float().cpu().numpy().reshape(4),
        "residual_scale": residual_scale.detach().float().cpu().numpy().reshape(4),
        "checkpoint_hashes_json": np.asarray(
            json.dumps(dict(checkpoint_hashes), sort_keys=True)
        ),
        "event_summaries_json": np.asarray(json.dumps(list(events), sort_keys=True)),
        "all_temporal_frames_retained": np.asarray(True),
        "unavailable_post_nonfinite_frames_are_nan": np.asarray(True),
    }
    for variant, record in arrays.items():
        payload[f"{variant}_deployed_states"] = record["deployed_states"]
        if variant in {"D082_correct", "D082_zero_inflow"}:
            payload[f"{variant}_raw_proposals"] = record["raw_proposals"]
            payload[f"{variant}_model_currents"] = record["model_currents"]
    np.savez_compressed(path, **payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = perf_counter()
    device = select_device(args.device)
    source_binding = verify_bound_source(args.d082_run_dir)
    store = PCNOEuler2DShardStore(args.data_dir)
    if store.manifest_digest != args.expected_data_manifest_digest:
        raise ValueError("active store manifest differs")

    d082, d082_summary, d082_split, d082_selected, d082_hash = load_bound_model(
        args.d082_run_dir,
        device=device,
        expected_manifest=args.expected_data_manifest_digest,
        expected_side_mode=BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
    )
    n0, n0_summary, n0_split, n0_selected, n0_hash = load_bound_model(
        args.n0_run_dir,
        device=device,
        expected_manifest=args.expected_data_manifest_digest,
        expected_side_mode=BOUNDARY_RESIDUAL_NONE,
    )
    if d082_split != n0_split:
        raise ValueError("D082 and N0 split contracts differ")
    if d082_summary["normalization_digest"] != n0_summary["normalization_digest"]:
        raise ValueError("D082 and N0 normalization digests differ")
    if d082_summary["normalization"] != n0_summary["normalization"]:
        raise ValueError("D082 and N0 normalizers differ")
    if d082_summary["boundary_contract"] != n0_summary["boundary_contract"]:
        raise ValueError("D082 and N0 boundary contracts differ")
    if tuple(d082.boundary_residual_names) != SEMANTIC_NAMES:
        raise ValueError("D082 semantic order differs")
    if tuple(store.boundary_field_names) != SEMANTIC_NAMES:
        raise ValueError("active shard semantic-field order differs")

    keys = [str(key) for key in d082_split["rollout_keys"]]
    if len(keys) != 30:
        raise ValueError("expected exactly 30 open-validation rollout cases")
    if d082_split.get("test_keys") != []:
        raise ValueError("sealed/test population is not empty")
    missing_save_cases = sorted(set(args.save_case_keys) - set(keys))
    if missing_save_cases:
        raise ValueError(
            f"save cases are outside open validation: {missing_save_cases}"
        )

    contract = d082_summary["boundary_contract"]
    expected_policy_digests = {
        str(key): str(value) for key, value in contract["policy_digests"].items()
    }
    policies: dict[str, dict[str, Any]] = {}
    policy_records: dict[str, dict[str, Any]] = {}
    for key in keys:
        policy, record = build_graph_causal_boundary_policy(
            store,
            key,
            device=device,
            max_source_hops=int(contract["max_source_hops"]),
            rho_inf=float(contract["rho_inf"]),
            p_inf=float(contract["p_inf"]),
        )
        if record["policy_digest"] != expected_policy_digests[key]:
            raise ValueError(f"boundary policy digest differs for trajectory {key}")
        policies[key] = policy
        policy_records[key] = record

    variants: dict[str, nn.Module] = {
        "N0_correct": n0,
        "D082_correct": d082,
        "D082_zero_all": FieldInterventionModel(d082, (0, 1, 2)),
        "D082_zero_wall": FieldInterventionModel(d082, (0,)),
        "D082_zero_outflow": FieldInterventionModel(d082, (1,)),
        "D082_zero_inflow": FieldInterventionModel(d082, (2,)),
    }
    state_mean = d082.state_mean
    state_scale = d082.state_scale
    residual_scale = d082.residual_scale
    checkpoint_hashes = {"D082": d082_hash, "N0": n0_hash}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle_dir = args.output_dir / "cases"
    bundle_dir.mkdir()
    call_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    repeatability_rows: list[dict[str, Any]] = []
    bundle_records: list[dict[str, Any]] = []

    for case_index, key in enumerate(keys, start=1):
        saved_arrays: dict[str, Mapping[str, np.ndarray]] = {}
        case_events: list[Mapping[str, Any]] = []
        for variant, model in variants.items():
            repeat_outputs = []
            for repeat in range(args.repeat_count):
                rows, events, arrays = run_case_once(
                    model,
                    store,
                    key,
                    variant=variant,
                    repeat=repeat,
                    num_steps=args.num_steps,
                    device=device,
                    amp=args.amp,
                    boundary_policy=policies[key],
                    state_mean=state_mean,
                    state_scale=state_scale,
                    residual_scale=residual_scale,
                )
                call_rows.extend(rows)
                event_rows.append(events)
                repeat_outputs.append((events, arrays))
                if repeat == 0:
                    case_events.append(events)
                    if key in args.save_case_keys and variant in SAVE_VARIANTS:
                        saved_arrays[variant] = arrays
            repeatability_rows.append(
                compare_repeats(
                    repeat_outputs[0][1],
                    repeat_outputs[1][1],
                    repeat_outputs[0][0],
                    repeat_outputs[1][0],
                )
            )
            print(
                json.dumps(
                    {
                        "case": key,
                        "case_index": case_index,
                        "case_count": len(keys),
                        "variant": variant,
                        "first_inadmissible_calls": [
                            item[0]["first_inadmissible_call"]
                            for item in repeat_outputs
                        ],
                        "registered_blowup": [
                            item[0]["registered_blowup"] for item in repeat_outputs
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if key in args.save_case_keys:
            bundle_path = bundle_dir / f"case_{key}.npz"
            save_case_bundle(
                bundle_path,
                store,
                key,
                saved_arrays,
                case_events,
                num_steps=args.num_steps,
                checkpoint_hashes=checkpoint_hashes,
                state_mean=state_mean,
                state_scale=state_scale,
                residual_scale=residual_scale,
            )
            bundle_records.append(
                {
                    "trajectory": key,
                    "file": str(bundle_path.relative_to(args.output_dir)),
                    "sha256": sha256_file(bundle_path),
                    "bytes": int(bundle_path.stat().st_size),
                    "temporal_frames": args.num_steps + 1,
                    "variants": list(saved_arrays),
                }
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    repeat_zero_events = [row for row in event_rows if int(row["repeat"]) == 0]
    aggregates = {
        variant: aggregate_events(
            [row for row in repeat_zero_events if row["variant"] == variant]
        )
        for variant in variants
    }
    d082_reproduction = reproduction_audit(
        [row for row in repeat_zero_events if row["variant"] == "D082_correct"],
        d082_selected,
    )
    n0_reproduction = reproduction_audit(
        [row for row in repeat_zero_events if row["variant"] == "N0_correct"],
        n0_selected,
    )
    repeatability = {
        "comparisons": len(repeatability_rows),
        "exact_deployed_state_count": sum(
            bool(row["deployed_states_exact_equal_nan"]) for row in repeatability_rows
        ),
        "exact_finite_mask_count": sum(
            bool(row["finite_mask_exact"]) for row in repeatability_rows
        ),
        "exact_first_inadmissible_call_count": sum(
            bool(row["first_inadmissible_call_equal"]) for row in repeatability_rows
        ),
        "exact_strict_valid_length_count": sum(
            bool(row["strict_valid_length_equal"]) for row in repeatability_rows
        ),
        "maximum_deployed_state_repeat_difference": max(
            (
                float(row["deployed_states_max_abs_common_finite"])
                for row in repeatability_rows
                if row["deployed_states_max_abs_common_finite"] is not None
            ),
            default=None,
        ),
    }

    write_csv_with_paths(args.output_dir / "call_metrics.csv", call_rows)
    write_csv_with_paths(args.output_dir / "trajectory_events.csv", event_rows)
    write_csv_with_paths(args.output_dir / "repeatability.csv", repeatability_rows)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "elapsed_seconds": perf_counter() - started,
        "runtime": runtime_environment(device),
        "provenance": {
            "data_manifest_digest": store.manifest_digest,
            "open_validation_only": True,
            "sealed_test_keys": [],
            "d082_checkpoint": {
                "sha256": d082_hash,
                "best_epoch": int(d082_summary["best_epoch"]),
                "config_digest": d082_summary["config_digest"],
            },
            "n0_checkpoint": {
                "sha256": n0_hash,
                "best_epoch": int(n0_summary["best_epoch"]),
                "config_digest": n0_summary["config_digest"],
            },
            "normalization_digest": d082_summary["normalization_digest"],
            "source_binding": source_binding,
            "diagnostic_source": {
                "evaluator_sha256": sha256_file(Path(__file__).resolve()),
                "utility_sha256": sha256_file(
                    ROOT / "utility" / "time_dependent_no" / "pcno_inadmissibility.py"
                ),
            },
            "boundary_policy_set_digest": contract["policy_set_digest"],
            "boundary_policy_count": len(policy_records),
            "all_policy_digests_reproduced": True,
        },
        "evaluation_contract": {
            "trajectory_keys": keys,
            "trajectory_count": len(keys),
            "num_steps": args.num_steps,
            "state_frames": args.num_steps + 1,
            "step_stride": 1,
            "physical_delta_t": float(store.manifest["dt"]),
            "amp": args.amp,
            "batch_size": 1,
            "repeat_count": args.repeat_count,
            "variants": {
                "N0_correct": "frozen no-boundary-channel control",
                "D082_correct": "frozen semantic-collar residual checkpoint",
                "D082_zero_all": "zero wall, outflow, and inflow fields every call",
                "D082_zero_wall": "zero wall field every call",
                "D082_zero_outflow": "zero outflow field every call",
                "D082_zero_inflow": "zero inflow field every call",
            },
            "boundary_policy": (
                "unchanged checkpoint-native causal nodal physical closure on model "
                "input and raw output"
            ),
            "clipping_floors_smoothing_limiter": False,
            "future_reference_state_used_by_recurrence": False,
            "continuation_rule": (
                "continue finite inadmissible deployed states; stop only after a "
                "deployed nonfinite state"
            ),
            "instrumentation": (
                "explicit standard contract return values only; no forward hooks"
            ),
            "hook_equivalence_test": "not_applicable_no_hooks_registered",
            "save_case_keys": args.save_case_keys,
            "save_case_rule": (
                "predeclared known D082 failure case 54, D082 rescue of N0 case 23, "
                "and prior cross-encoding visualization case 128"
            ),
        },
        "event_contract": {
            "inadmissible": (
                "any nonfinite conservative/derived primitive component, nonpositive "
                "density, nonpositive internal energy, or nonpositive pressure"
            ),
            "definitive_numerical_blowup": "any nonfinite deployed conservative state",
            "severe_amplitude_explosion": (
                f"deployed max absolute conservative component reaches "
                f"{BLOWUP_AMPLITUDE_RATIO:g} times the case reference maximum"
            ),
            "severe_global_error_explosion": (
                "proxy-weighted state-scaled relative L2 reaches "
                f"{BLOWUP_RELATIVE_L2:g}"
            ),
            "thresholds_are_diagnostic_not_physical": True,
            "normalizer_excursions_are_not_training_distribution_membership_tests": True,
            "bump_weights_are_quadrature_proxies_not_validated_physical_volumes": True,
        },
        "aggregates_repeat_0": aggregates,
        "repeatability": repeatability,
        "historical_strict_reproduction": {
            "D082_correct": d082_reproduction,
            "N0_correct": n0_reproduction,
        },
        "case_bundles": bundle_records,
        "claim_boundary": {
            "verified": (
                "temporal association, recovery, persistence, and registered explosion "
                "under frozen checkpoint recurrence on the 30 open bump validation cases"
            ),
            "exact_stage_attribution": (
                "raw-invalid to deployed-admissible is output-projection recovery; "
                "model-current-invalid to raw-admissible is model recovery"
            ),
            "checkpoint_local_causal_interventions": (
                "D082 semantic-field zeroing changes only named frozen input fields"
            ),
            "not_claimed": (
                "inadmissibility is randomized or independently causal, general Euler "
                "stability, exact DG boundary replay, or physical conservation"
            ),
        },
    }
    atomic_write_json_with_paths(args.output_dir / "summary.json", payload)
    files = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(args.output_dir)),
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    atomic_write_json_with_paths(
        args.output_dir / "artifact_manifest.json",
        {"schema": f"{SCHEMA}_artifact_manifest", "files": files},
    )
    store.close()
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "elapsed_seconds": payload["elapsed_seconds"],
                "aggregates": aggregates,
                "repeatability": repeatability,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

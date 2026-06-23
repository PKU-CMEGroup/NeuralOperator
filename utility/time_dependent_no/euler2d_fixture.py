"""End-to-end smoke diagnostics for the synthetic 2D Euler fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from utility.time_dependent_no.euler2d import inspect_cpg_trajectory, load_cpg_primitive_sequence
from utility.time_dependent_no.euler2d_synthetic import (
    DegradationMode,
    SyntheticEuler2DConfig,
    make_degraded_synthetic_euler2d_prediction,
    make_synthetic_cpg_trajectory,
)
from utility.time_dependent_no.euler2d_metrics import (
    boundary_leakage_metrics,
    near_shock_error,
    positivity_metrics,
    rollout_relative_l2_by_time,
    shock_centroid_distance,
    shock_indicator,
    shock_smearing_metrics,
    summarize_euler2d_rollout,
)


DEFAULT_FIXTURE_CASES: tuple[DegradationMode, ...] = (
    "perfect",
    "lagged_shock",
    "smeared_shock",
    "boundary_leak",
    "positivity_failure",
)


def run_euler2d_fixture_diagnostics(
    config: SyntheticEuler2DConfig | None = None,
    *,
    cases: tuple[DegradationMode, ...] = DEFAULT_FIXTURE_CASES,
) -> dict[str, Any]:
    """Run a deterministic no-model diagnostic smoke test.

    This is a code-path check, not research evidence. It verifies that a
    CPG-style trajectory can be inspected and that controlled forecast defects
    are visible to the Track 1 Euler diagnostic metrics.
    """

    if config is None:
        config = SyntheticEuler2DConfig()
    group = make_synthetic_cpg_trajectory(config)
    target = load_cpg_primitive_sequence(group)
    edges = group["edges"]
    positions = group["pos"]
    node_type = group["node_type"][:, 0]
    node_weights = group["node_weights"]
    final_shock_mask = shock_indicator(target[-1], edges, quantile=0.85)

    case_payloads: dict[str, dict[str, Any]] = {}
    for case in cases:
        prediction = make_degraded_synthetic_euler2d_prediction(
            config,
            case,
            target=target,
        )
        summary = summarize_euler2d_rollout(
            prediction,
            target,
            node_type=node_type,
            edges=edges,
            positions=positions,
            node_weights=node_weights,
            gamma=config.gamma,
        )
        case_payloads[case] = {
            "summary": _json_ready(summary),
            "relative_l2_by_time": _json_ready(
                rollout_relative_l2_by_time(prediction, target)
            ),
            "final_shock_centroid_distance": _json_ready(
                shock_centroid_distance(prediction[-1], target[-1], positions, edges)
            ),
            "final_shock_smearing": _json_ready(
                shock_smearing_metrics(prediction[-1], target[-1], edges)
            ),
            "final_near_shock_error": _json_ready(
                near_shock_error(prediction[-1], target[-1], final_shock_mask)
            ),
            "positivity": _json_ready(positivity_metrics(prediction)),
            "boundary": _json_ready(
                boundary_leakage_metrics(prediction, target, node_type)
            ),
        }

    return {
        "kind": "synthetic_euler2d_fixture_diagnostics",
        "evidence_class": "local_smoke_only",
        "config": config.to_dict(),
        "schema": inspect_cpg_trajectory(group, name="synthetic_fixture").to_dict(),
        "cases": case_payloads,
        "checks": _diagnostic_checks(case_payloads),
    }


def write_euler2d_fixture_diagnostics_json(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a fixture diagnostic payload as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def _diagnostic_checks(cases: dict[str, dict[str, Any]]) -> dict[str, bool]:
    perfect = cases["perfect"]
    lagged = cases["lagged_shock"]
    smeared = cases["smeared_shock"]
    boundary = cases["boundary_leak"]
    positivity = cases["positivity_failure"]
    return {
        "perfect_has_zero_error": perfect["summary"]["mean_relative_l2"] == 0.0,
        "lagged_shock_moves_centroid": lagged["final_shock_centroid_distance"]
        > perfect["final_shock_centroid_distance"],
        "smeared_shock_weakens_gradient": smeared["summary"][
            "final_shock_strength_ratio"
        ]
        < 1.0,
        "boundary_leak_increases_boundary_error": boundary["boundary"]["boundary_rmse"]
        > perfect["boundary"]["boundary_rmse"],
        "positivity_case_is_detected": not positivity["positivity"]["all_positive"],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


"""Build a conservative restriction-consistent 1D Euler resolution family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.time_dependent_no.evaluate_euler1d_resolution_transfer import (
        validate_resolution_family,
    )
else:
    from evaluate_euler1d_resolution_transfer import validate_resolution_family
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    load_euler1d_npz,
    primitive_to_conservative_np,
    restrict_euler1d_source,
    restriction_commutation_metrics,
    save_euler1d_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coarse-cells",
        type=int,
        nargs="+",
        default=[128, 256],
    )
    parser.add_argument("--commutation-tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def _metadata_scalar(source: Euler1DNPZ, key: str) -> Any:
    if key not in source.metadata:
        raise ValueError(f"fine dataset is missing metadata: {key}")
    value = np.asarray(source.metadata[key])
    return value.item() if value.shape == () else value


def exact_initial_average_error(source: Euler1DNPZ) -> dict[str, float]:
    """Compare frame zero with analytic Riemann cell averages in conserved form."""

    domains = np.asarray(
        source.metadata.get("domains_exact", _metadata_scalar(source, "domains")),
        dtype=np.float64,
    )
    discontinuities = np.asarray(
        source.metadata.get("x_disc_exact", _metadata_scalar(source, "x_disc")),
        dtype=np.float64,
    )
    edges_fraction = np.arange(source.num_cells + 1, dtype=np.float64)
    edges = domains[:, :1] + (
        (domains[:, 1:] - domains[:, :1]) * edges_fraction[None, :] / source.num_cells
    )
    dx = (domains[:, 1] - domains[:, 0]) / source.num_cells
    left_fraction = np.clip(
        (discontinuities[:, None] - edges[:, :-1]) / dx[:, None],
        0.0,
        1.0,
    )
    left = primitive_to_conservative_np(
        np.asarray(
            source.metadata.get("left_states_exact", source.left_states),
            dtype=np.float64,
        ),
        source.gamma,
    )
    right = primitive_to_conservative_np(
        np.asarray(
            source.metadata.get("right_states_exact", source.right_states),
            dtype=np.float64,
        ),
        source.gamma,
    )
    expected = (
        left_fraction[..., None] * left[:, None, :]
        + (1.0 - left_fraction[..., None]) * right[:, None, :]
    )
    actual = primitive_to_conservative_np(
        source.data[:, 0].astype(np.float64), source.gamma
    )
    difference = (actual - expected).reshape(source.num_cases, -1)
    denominator = np.linalg.norm(expected.reshape(source.num_cases, -1), axis=1)
    relative = np.linalg.norm(difference, axis=1) / np.maximum(denominator, 1.0e-12)
    return {
        "relative_l2_mean": float(relative.mean()),
        "relative_l2_max": float(relative.max()),
        "max_abs": float(np.max(np.abs(actual - expected))),
    }


def build_family(
    fine_path: Path,
    output_dir: Path,
    coarse_cells: list[int],
    tolerance: float,
) -> dict[str, Any]:
    if tolerance <= 0.0:
        raise ValueError("commutation tolerance must be positive")
    fine = load_euler1d_npz(fine_path)
    initialization_mode = str(_metadata_scalar(fine, "initialization_mode"))
    if initialization_mode != "exact_cell_average":
        raise ValueError(
            "restriction family requires initialization_mode=exact_cell_average"
        )
    requested = sorted(set(coarse_cells))
    if not requested:
        raise ValueError("at least one coarse resolution is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Euler1DNPZ] = {f"nx{fine.num_cells}": fine}
    paths: dict[str, Path] = {f"nx{fine.num_cells}": fine_path}
    for cells in requested:
        coarse = restrict_euler1d_source(fine, cells)
        path = output_dir / f"restriction_nx{cells}_from_nx{fine.num_cells}.npz"
        save_euler1d_npz(path, coarse)
        label = f"nx{cells}"
        sources[label] = load_euler1d_npz(path)
        paths[label] = path

    contract = validate_resolution_family(sources)
    ordered = sorted(sources.items(), key=lambda item: item[1].num_cells)
    initialization = {
        label: exact_initial_average_error(source) for label, source in ordered
    }
    commutation = [
        restriction_commutation_metrics(coarse, fine_source, strides=(1, 2))
        for (_, coarse), (_, fine_source) in zip(ordered, ordered[1:])
    ]
    maximum_error = max(
        [item["state_global_relative_l2"] for item in commutation]
        + [
            update["global_relative_l2"]
            for item in commutation
            for update in item["updates"]
        ]
        + [item["relative_l2_max"] for item in initialization.values()]
    )
    if maximum_error > tolerance:
        raise ValueError(
            f"restriction contract error {maximum_error:.6e} exceeds {tolerance:.6e}"
        )

    payload = {
        "status": "ok",
        "fine_data": fine_path,
        "paths": paths,
        "contract": contract,
        "initialization": initialization,
        "commutation": commutation,
        "maximum_relative_error": maximum_error,
        "local_relative_error_diagnostic_max": max(
            [item["state_relative_l2_max"] for item in commutation]
            + [
                update["relative_l2_max"]
                for item in commutation
                for update in item["updates"]
            ]
        ),
        "tolerance": tolerance,
    }
    (output_dir / "contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = build_family(
        args.fine_data,
        args.output_dir,
        args.coarse_cells,
        args.commutation_tolerance,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

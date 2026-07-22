"""Generate the fixed, two-group OOD gate for the Euler flow-map frontier.

This is not a configurable parameter sweep.  It preserves the training
solver, grid, saved times, boundaries, and domain sampling while moving either
the inflow state just above the training support or the ambient right state
just below it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__:
    from scripts.time_dependent_no.euler1d_weno_hllc_ader_dataset import (
        SampleConfig,
        generate_dataset,
    )
else:
    from euler1d_weno_hllc_ader_dataset import SampleConfig, generate_dataset


TRAINING_SUPPORT = SampleConfig(
    domain_length_range=(1.0, 1.0),
    domain_left_range=(0.10, 0.20),
    discontinuity_fraction_range=(0.15, 0.35),
    left_rho_range=(1.0, 1.50),
    left_u_range=(0.60, 2.40),
    left_p_range=(0.90, 4.00),
    right_rho_range=(0.10, 0.30),
    right_p_range=(0.10, 0.40),
)

OOD_REGIMES = {
    "high_inflow": SampleConfig(
        domain_length_range=(1.0, 1.0),
        domain_left_range=(0.10, 0.20),
        discontinuity_fraction_range=(0.15, 0.35),
        left_rho_range=(1.50, 1.65),
        left_u_range=(2.40, 2.65),
        left_p_range=(4.00, 4.50),
        right_rho_range=(0.10, 0.30),
        right_p_range=(0.10, 0.40),
    ),
    "low_ambient": SampleConfig(
        domain_length_range=(1.0, 1.0),
        domain_left_range=(0.10, 0.20),
        discontinuity_fraction_range=(0.15, 0.35),
        left_rho_range=(1.0, 1.50),
        left_u_range=(0.60, 2.40),
        left_p_range=(0.90, 4.00),
        right_rho_range=(0.07, 0.10),
        right_p_range=(0.07, 0.10),
    ),
}

CASE_AXIS_KEYS = {
    "data",
    "x",
    "t",
    "left_states",
    "right_states",
    "domains",
    "x_disc",
    "t_final",
    "left_states_exact",
    "right_states_exact",
    "domains_exact",
    "x_disc_exact",
    "t_final_exact",
    "fallback_counts",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-regime", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _config_dict(config: SampleConfig) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in getattr(config, name)]
        for name in config.__dataclass_fields__
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_regime_files(
    paths: Sequence[Path],
    labels: Sequence[str],
    output: Path,
    *,
    cases_per_regime: int,
    seed: int,
) -> None:
    if len(paths) != len(labels) or not paths:
        raise ValueError("paths and labels must be nonempty and aligned")
    loaded: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            loaded.append({key: np.asarray(arrays[key]) for key in arrays.files})
    common_keys = set(loaded[0])
    if any(set(item) != common_keys for item in loaded[1:]):
        raise ValueError("regime files do not share the same dataset schema")

    payload: dict[str, Any] = {}
    for key in sorted(common_keys):
        values = [item[key] for item in loaded]
        if key in CASE_AXIS_KEYS:
            if any(value.shape[0] != cases_per_regime for value in values):
                raise ValueError(f"unexpected case axis for {key}")
            payload[key] = np.concatenate(values, axis=0)
        elif key == "n_cases":
            payload[key] = np.array(
                cases_per_regime * len(paths),
                dtype=np.int32,
            )
        else:
            if any(not np.array_equal(values[0], value) for value in values[1:]):
                raise ValueError(f"regime metadata differs for {key}")
            payload[key] = values[0]

    payload["ood_regime"] = np.repeat(
        np.asarray(labels),
        cases_per_regime,
    )
    payload["ood_contract_json"] = np.array(
        json.dumps(
            {
                "label": "mild_support_extrapolation_v1",
                "seed": seed,
                "cases_per_regime": cases_per_regime,
                "training_support": _config_dict(TRAINING_SUPPORT),
                "regimes": {
                    label: _config_dict(config) for label, config in OOD_REGIMES.items()
                },
                "fixed_contract": {
                    "nx": 256,
                    "n_steps": 100,
                    "t_final": 0.5,
                    "gamma": 1.4,
                    "cfl": 0.2,
                    "initialization_mode": "cell_center",
                    "boundary": (
                        "left fixed primitive inflow per case; right reflective wall"
                    ),
                },
            },
            sort_keys=True,
        )
    )
    payload["ood_generator_source_sha256"] = np.array(_sha256(Path(__file__).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cases_per_regime < 1:
        raise ValueError("--cases-per-regime must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing OOD data: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(OOD_REGIMES)
    with tempfile.TemporaryDirectory(
        prefix="euler1d_flow_map_ood_",
        dir=args.output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        paths: list[Path] = []
        for regime_id, label in enumerate(labels):
            path = temporary_root / f"{label}.npz"
            generate_dataset(
                path,
                n_cases=args.cases_per_regime,
                n_steps=100,
                nx=256,
                t_final=0.5,
                gamma=1.4,
                cfl=0.2,
                use_shock_flattening=True,
                use_hlle_on_troubled_faces=True,
                shock_sensor_threshold=0.05,
                shock_flatten_radius=4,
                seed=args.seed + 1000 * regime_id,
                ng=3,
                rho_floor=1.0e-12,
                p_floor=1.0e-12,
                sample_config=OOD_REGIMES[label],
                verbose=not args.quiet,
                save_face_flux_integral=False,
                initialization_mode="cell_center",
                num_workers=args.workers,
                storage_dtype="float32",
            )
            paths.append(path)
        merge_regime_files(
            paths,
            labels,
            args.output,
            cases_per_regime=args.cases_per_regime,
            seed=args.seed,
        )
    with np.load(args.output, allow_pickle=False) as arrays:
        data = np.asarray(arrays["data"])
        if not np.all(np.isfinite(data)):
            raise RuntimeError("generated OOD data contains nonfinite values")
        if np.any(data[..., 0] <= 0.0) or np.any(data[..., 2] <= 0.0):
            raise RuntimeError("generated OOD data is not admissible")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": _sha256(args.output),
                    "shape": list(data.shape),
                    "regime_counts": {
                        label: int(np.sum(np.asarray(arrays["ood_regime"]) == label))
                        for label in labels
                    },
                    "fallback_count_total": int(
                        np.sum(np.asarray(arrays["fallback_counts"]))
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

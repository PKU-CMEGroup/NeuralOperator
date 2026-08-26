"""Run the M1 Kolmogorov reference map on synthetic states only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic-only A1 preflight for the restartable Kolmogorov "
            "reference map. This command does not read datasets or checkpoints."
        )
    )
    parser.add_argument("--resolution", type=int, default=18)
    parser.add_argument("--forcing-wavenumber", type=int, default=2)
    parser.add_argument("--macro-dt", type=float, default=0.01)
    parser.add_argument("--dt-max", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = KolmogorovReferenceConfig(
        resolution=args.resolution,
        viscosity=0.02,
        linear_drag=0.1,
        forcing_amplitude=0.2,
        forcing_wavenumber=args.forcing_wavenumber,
        macro_dt=args.macro_dt,
        dt_max=args.dt_max,
    ).validated()
    stepper = KolmogorovReferenceStepper(config)
    rng = np.random.default_rng(args.seed)
    initial = stepper.canonicalize(rng.normal(size=stepper.state_shape))

    first_a = stepper.advance_canonical(initial)
    first_b = stepper.advance_canonical(initial.copy())
    rollout, records = stepper.rollout_canonical(initial, args.steps)
    laminar = stepper.laminar_vorticity()
    laminar_next = stepper.advance_canonical(laminar)
    laminar_relative_l2 = float(
        np.linalg.norm(laminar_next.state - laminar)
        / max(np.linalg.norm(laminar), np.finfo(np.float64).tiny)
    )
    deterministic = np.array_equal(first_a.state, first_b.state)
    first_step_closure = np.array_equal(first_a.state, rollout[1])
    output_projection = stepper.projection_relative_l2(rollout[-1])

    if not deterministic:
        raise RuntimeError("same-process reference replay is not bitwise deterministic")
    if not first_step_closure:
        raise RuntimeError("single-step and repeated-rollout paths do not close")
    if laminar_relative_l2 > 1.0e-10:
        raise RuntimeError("configured laminar steady state does not close")
    if output_projection > config.canonical_tolerance:
        raise RuntimeError("reference output left the canonical state space")

    initial_diagnostics = stepper.diagnostics_canonical(initial)
    final_diagnostics = stepper.diagnostics_canonical(rollout[-1])
    payload = {
        "status": "pass",
        "scope": "synthetic_a1_only",
        "scientific_data_access": False,
        "checkpoint_access": False,
        "config": asdict(config),
        "state_shape": list(stepper.state_shape),
        "rollout_steps": args.steps,
        "deterministic_same_process": deterministic,
        "first_step_closure": first_step_closure,
        "laminar_relative_l2": laminar_relative_l2,
        "output_projection_relative_l2": output_projection,
        "substeps": [record.substeps for record in records],
        "initial_diagnostics": asdict(initial_diagnostics),
        "final_diagnostics": asdict(final_diagnostics),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

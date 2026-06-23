from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d_fixture import (
    run_euler2d_fixture_diagnostics,
    write_euler2d_fixture_diagnostics_json,
)
from utility.time_dependent_no.euler2d_synthetic import SyntheticEuler2DConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic 2D Euler schema and diagnostic smoke checks."
    )
    parser.add_argument("--nx", type=int, default=16, help="Fixture grid columns.")
    parser.add_argument("--ny", type=int, default=12, help="Fixture grid rows.")
    parser.add_argument(
        "--num-steps", type=int, default=8, help="Fixture rollout time steps."
    )
    parser.add_argument("--mach", type=float, default=2.0, help="Fixture Mach field.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "time_dependent_no" / "fixture_diagnostics.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticEuler2DConfig(
        nx=args.nx,
        ny=args.ny,
        num_steps=args.num_steps,
        mach=args.mach,
    )
    payload = run_euler2d_fixture_diagnostics(config)
    write_euler2d_fixture_diagnostics_json(payload, args.output)

    print(f"saved: {args.output}")
    print("checks:")
    for name, passed in payload["checks"].items():
        print(f"- {name}: {passed}")
    print("case summaries:")
    for case, result in payload["cases"].items():
        summary = result["summary"]
        print(
            f"- {case}: final_rel_l2={summary['final_relative_l2']:.6f}, "
            f"shock_dist={result['final_shock_centroid_distance']:.6f}, "
            f"strength_ratio={summary['final_shock_strength_ratio']:.6f}, "
            f"positive={summary['all_density_pressure_positive']}"
        )


if __name__ == "__main__":
    main()

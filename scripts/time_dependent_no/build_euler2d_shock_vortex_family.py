#!/usr/bin/env python3
"""Build or audit the frozen 2D shock--vortex perturbation family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    audit_shock_vortex_family_artifacts,
    build_shock_vortex_family_manifest,
    load_shock_vortex_family_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="write the immutable family manifest")
    build.add_argument("--output-path", type=Path, required=True)

    audit = subparsers.add_parser(
        "audit", help="validate generated case directories against the manifest"
    )
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--artifact-root", type=Path, required=True)
    audit.add_argument("--output-path", type=Path, required=True)
    audit.add_argument(
        "--scope",
        choices=("smoke", "complete"),
        default="smoke",
        help="Audit the three prespecified smoke cases or all 135 family cases.",
    )
    return parser.parse_args(argv)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"output path already exists; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        payload = build_shock_vortex_family_manifest()
        _write_new_json(args.output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return

    manifest = load_shock_vortex_family_manifest(args.manifest)
    case_ids = (
        manifest["smoke_case_ids"]
        if args.scope == "smoke"
        else [case["case_id"] for case in manifest["cases"]]
    )
    payload = audit_shock_vortex_family_artifacts(
        manifest,
        args.artifact_root,
        case_ids,
    )
    _write_new_json(args.output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "passed":
        raise RuntimeError("shock-vortex family artifact audit failed")


if __name__ == "__main__":
    main()

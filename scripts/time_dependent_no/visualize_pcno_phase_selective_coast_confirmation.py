#!/usr/bin/env python3
"""Render the registered A44 frozen-A43 confirmation bundles."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_phase_selective_coast_confirmation import (
    ANIMATION_CONTRACT,
    ANIMATION_SCHEMA,
    CASE_TO_GROUP,
    GROUP_PROVENANCE_IDS,
    RESULT_SCHEMA,
    WORKING_ID,
)
from scripts.time_dependent_no.visualize_pcno_response_filtered_block import (
    DPI,
    FIELDS,
    FPS,
    _render_one,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)

RENDER_SCHEMA = "pcno_phase_selective_coast_confirmation_animation_render_v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _load_contract(
    rollout_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    rollout = _read_json(rollout_path)
    manifest = _read_json(manifest_path)
    verify_payload_sha256(rollout)
    verify_payload_sha256(manifest)
    if (
        rollout.get("schema") != RESULT_SCHEMA
        or rollout.get("working_id") != WORKING_ID
        or rollout.get("recurrence_executed") is not True
        or rollout.get("animations_rendered") is not False
    ):
        raise ValueError("rollout is not the registered unrendered A44 execution")
    if (
        manifest.get("schema") != ANIMATION_SCHEMA
        or manifest.get("working_id") != WORKING_ID
        or manifest.get("contract") != ANIMATION_CONTRACT
        or manifest.get("inference_or_gate_input") is not False
        or rollout.get("animation_bundle_manifest") != manifest
        or rollout.get("animation_bundle_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("animation manifest differs from the registered A44 contract")
    return rollout, manifest


def _load_confirmation_bundle(
    path: Path, expected: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"animation bundle hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    case_id = str(payload["case_id"].item())
    expected_case_id = str(expected["case_id"])
    if expected_case_id not in CASE_TO_GROUP:
        raise ValueError(f"invalid registered A44 animation case: {expected_case_id}")
    expected_group_id = GROUP_PROVENANCE_IDS[CASE_TO_GROUP[expected_case_id]]
    group_id = str(payload["split_group_id"].item())
    resolution = tuple(int(value) for value in payload["native_resolution"])
    calls = [int(value) for value in payload["output_calls"]]
    times = [float(value) for value in payload["physical_times"]]
    frame_count = len(contract["output_calls"])
    expected_shape = (frame_count, resolution[0] * resolution[1], 4)
    if (
        case_id != expected_case_id
        or expected.get("split_group_id") != expected_group_id
        or group_id != expected_group_id
        or int(expected.get("frames", -1)) != frame_count
        or resolution != tuple(contract["native_resolution"])
        or calls != contract["output_calls"]
        or not np.allclose(times, contract["physical_times"], atol=0.0, rtol=0.0)
        or payload["nodes"].shape != (resolution[0] * resolution[1], 2)
        or payload["volumes"].shape != (resolution[0] * resolution[1],)
        or any(
            payload[name].shape != expected_shape
            for name in (
                "truth_conservative",
                "raw_shadow_conservative",
                "corrected_conservative",
            )
        )
        or not all(
            np.isfinite(payload[name]).all()
            for name in (
                "truth_conservative",
                "raw_shadow_conservative",
                "corrected_conservative",
            )
        )
    ):
        raise ValueError(f"animation bundle contract mismatch: {path.name}")
    return payload


def render(rollout_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rollout, manifest = _load_contract(rollout_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    generated: list[Path] = []
    for bundle_row in manifest["bundles"]:
        bundle_path = manifest_path.parent / bundle_row["path"]
        bundle = _load_confirmation_bundle(
            bundle_path, bundle_row, manifest["contract"]
        )
        for field in FIELDS:
            paths, saturation = _render_one(
                bundle,
                field=field,
                contract=manifest["contract"],
                output_dir=output_dir,
            )
            generated.extend(paths)
            rows.append(
                {
                    "case_id": bundle_row["case_id"],
                    "field": field,
                    "saturation": saturation,
                    "files": [
                        {
                            "path": path.name,
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in paths
                    ],
                }
            )
    payload = with_payload_sha256(
        {
            "schema": RENDER_SCHEMA,
            "working_id": WORKING_ID,
            "rollout_sha256": sha256_file(rollout_path),
            "rollout_payload_sha256": rollout["payload_sha256"],
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "bundle_manifest_payload_sha256": manifest["payload_sha256"],
            "renderer_source_sha256": sha256_file(Path(__file__)),
            "renderer_patch": "a44_training_support_group_binding_fix1",
            "contract": manifest["contract"],
            "fps": FPS,
            "dpi": DPI,
            "rendered": rows,
            "generated_file_count": len(generated),
            "selection_or_gate_effect": False,
            "interpretation": (
                "Positive signed relative improvement means the frozen A43 "
                "candidate has lower pointwise absolute field error than raw."
            ),
        }
    )
    atomic_write_json(output_dir / "animation_render_manifest.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = render(args.rollout, args.bundle_manifest, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

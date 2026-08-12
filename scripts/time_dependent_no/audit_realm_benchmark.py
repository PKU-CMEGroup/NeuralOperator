"""Audit the frozen IgnitHIT open manifest and optionally replay open data.

Metadata-only mode opens only the supplied JSON inventory. Supplying both an
exact data root and a closed report directory enables the P1b train/validation
reference replay. Neither mode performs network access, and both reject every
path with a ``test`` component.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_OPEN_MANIFEST_SHA256,
    canonical_json_sha256,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)
from utility.time_dependent_no.realm_ignithit import (
    analyze_ignithit_open_dataset,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Local normalized JSON manifest for the pinned open population.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON summary path; otherwise print to stdout.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Exact acquired open tree; enables the P1b reference replay.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Closed P1b report directory; required with --data-root.",
    )
    return parser


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("manifest root must be a JSON object")
    return payload


def audit_manifest(path: Path) -> dict[str, Any]:
    payload = _load_payload(path)
    repository, revision, entries = parse_manifest_payload(payload)
    summary = validate_ignithit_open_manifest(repository, revision, entries)
    summary["manifest_payload_sha256"] = canonical_json_sha256(payload)
    summary["audit_scope"] = "metadata_only_no_object_open"
    return summary


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "time_dependent_no" / "acquire_realm_ignithit.py",
    )
    rows = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_source_manifest_v1",
        "files": rows,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _runtime_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_cpu_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "execution_device": "cpu",
        "model_or_checkpoint_loaded": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _closed_report_files(report_dir: Path) -> tuple[str, ...]:
    return (
        "schema_report.json",
        "train_statistics.json",
        "reference_replay.json",
        "normalizer_arrays.npz",
        "source_manifest.json",
        "runtime_manifest.json",
    )


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def run_p1b_replay(
    manifest_path: Path,
    data_root: Path,
    report_dir: Path,
) -> dict[str, Any]:
    payload = _load_payload(manifest_path)
    repository, revision, entries = parse_manifest_payload(payload)
    manifest_summary = validate_ignithit_open_manifest(repository, revision, entries)
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("report directory must be absent or empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = analyze_ignithit_open_dataset(data_root, entries)

    schema_report = {
        "schema": report["schema"],
        "scope": report["scope"],
        "open_manifest": manifest_summary,
        "normalized_manifest_payload_sha256": canonical_json_sha256(payload),
        "metadata": report["metadata"],
        "inventory": report["inventory"],
        "trajectory_rows": report["trajectory_rows"],
        "schema_gates": {
            name: report["gates"][name]
            for name in (
                "exact_file_inventory",
                "metadata_schema",
                "all_open_trajectories_finite_and_admissible",
                "sealed_test_objects_absent",
            )
        },
    }
    train_statistics = {
        "schema": "d088_ignithit_train_statistics_v1",
        "field_order": list(report["metadata"]["field_order"]),
        "primary": report["train_statistics_primary"],
        "source_sensitivity": report["train_statistics_source_sensitivity"],
        "raw": report["train_statistics_raw"],
        "boundedness": report["boundedness"],
        "front_thresholds_train_only": report["front_thresholds_train_only"],
    }
    reference_replay = {
        "schema": "d088_ignithit_reference_replay_v1",
        "normalizer_replay": report["normalizer_replay"],
        "validation_identity_metric_replay": report[
            "validation_identity_metric_replay"
        ],
        "gates": report["gates"],
        "all_gates_pass": report["all_gates_pass"],
        "anti_claims": report["anti_claims"],
    }
    _write_json(report_dir / "schema_report.json", schema_report)
    _write_json(report_dir / "train_statistics.json", train_statistics)
    _write_json(report_dir / "reference_replay.json", reference_replay)
    np.savez_compressed(report_dir / "normalizer_arrays.npz", **arrays)
    _write_json(report_dir / "source_manifest.json", _source_manifest())
    _write_json(report_dir / "runtime_manifest.json", _runtime_manifest())

    file_hashes = {
        name: sha256_file(report_dir / name)
        for name in _closed_report_files(report_dir)
    }
    final_manifest = {
        "schema": "d088_ignithit_p1b_final_hash_manifest_v1",
        "open_manifest_sha256": IGNITHIT_OPEN_MANIFEST_SHA256,
        "files": file_hashes,
        "self_hash_excluded": True,
    }
    _write_json(report_dir / "final_hash_manifest.json", final_manifest)
    if any(
        sha256_file(report_dir / name) != digest for name, digest in file_hashes.items()
    ):
        raise RuntimeError("post-write P1b artifact verification failed")
    return {
        "schema": "d088_ignithit_p1b_compact_summary_v1",
        "all_gates_pass": report["all_gates_pass"],
        "open_manifest_sha256": IGNITHIT_OPEN_MANIFEST_SHA256,
        "report_files": [*_closed_report_files(report_dir), "final_hash_manifest.json"],
        "final_hash_manifest_sha256": sha256_file(
            report_dir / "final_hash_manifest.json"
        ),
        "model_or_checkpoint_loaded": False,
        "test_object_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.output.resolve(strict=False) == (
            args.manifest.resolve(strict=False)
        ):
            raise ValueError("output must not overwrite the input manifest")
        if (args.data_root is None) != (args.report_dir is None):
            raise ValueError("--data-root and --report-dir must be supplied together")
        if args.data_root is None:
            summary = audit_manifest(args.manifest)
        else:
            if _is_within(args.report_dir, args.data_root):
                raise ValueError("report directory must be outside the exact data root")
            if args.output is not None and (
                _is_within(args.output, args.data_root)
                or _is_within(args.output, args.report_dir)
            ):
                raise ValueError(
                    "output must be outside the exact data root and closed report directory"
                )
            summary = run_p1b_replay(args.manifest, args.data_root, args.report_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if summary.get("all_gates_pass", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

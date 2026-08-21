"""Audit the pinned open PlanarDet train/validation release without a model."""

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
    canonical_json_sha256,
    parse_manifest_payload,
)
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_FIELDS,
    PLANARDET_OPEN_MANIFEST_SHA256,
    analyze_planardet_open_dataset,
    sha256_file,
    validate_planardet_open_manifest,
)

SCHEMA = "w26_l4_planardet_pd0_a2_data_audit_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "__init__.py",
        REPO_ROOT / "utility" / "adam.py",
        REPO_ROOT / "utility" / "losses.py",
        REPO_ROOT / "utility" / "normalizer.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "__init__.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_planardet.py",
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "time_dependent_no" / "acquire_realm_planardet.py",
    )
    rows = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a2_data_source_manifest_v1",
        "files": rows,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _runtime_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a2_cpu_runtime_v1",
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


def run_audit(
    manifest_path: Path,
    data_root: Path,
    report_dir: Path,
) -> dict[str, Any]:
    payload = _load_payload(manifest_path)
    repository, revision, entries = parse_manifest_payload(payload)
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("report directory must be absent or empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = analyze_planardet_open_dataset(data_root, entries)

    schema_report = {
        "schema": SCHEMA,
        "scope": report["scope"],
        "open_manifest": manifest_summary,
        "normalized_manifest_payload_sha256": canonical_json_sha256(payload),
        "metadata": report["metadata"],
        "inventory": report["inventory"],
        "trajectory_rows": report["trajectory_rows"],
        "pMax_operational_semantics": report["pMax_operational_semantics"],
    }
    train_statistics = {
        "schema": "w26_l4_planardet_pd0_train_statistics_v1",
        "field_order": list(PLANARDET_FIELDS),
        "primary": report["train_statistics_primary"],
        "source_sensitivity": report["train_statistics_source_sensitivity"],
        "raw": report["train_statistics_raw"],
        "primary_clamp_counts_by_species": report["primary_clamp_counts_by_species"],
        "primary_decode_max_abs_error_by_channel": report[
            "primary_decode_max_abs_error_by_channel"
        ],
        "boundedness": report["boundedness"],
    }
    reference_replay = {
        "schema": "w26_l4_planardet_pd0_reference_replay_v1",
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

    retained = (
        "schema_report.json",
        "train_statistics.json",
        "reference_replay.json",
        "normalizer_arrays.npz",
        "source_manifest.json",
        "runtime_manifest.json",
    )
    file_hashes = {name: sha256_file(report_dir / name) for name in retained}
    final_manifest = {
        "schema": "w26_l4_planardet_pd0_a2_data_final_hash_manifest_v1",
        "open_manifest_sha256": PLANARDET_OPEN_MANIFEST_SHA256,
        "files": file_hashes,
        "self_hash_excluded": True,
    }
    _write_json(report_dir / "final_hash_manifest.json", final_manifest)
    if any(
        sha256_file(report_dir / name) != digest for name, digest in file_hashes.items()
    ):
        raise RuntimeError("post-write PlanarDet audit verification failed")
    return {
        "schema": "w26_l4_planardet_pd0_a2_data_summary_v1",
        "all_gates_pass": report["all_gates_pass"],
        "open_manifest_sha256": PLANARDET_OPEN_MANIFEST_SHA256,
        "report_files": [*retained, "final_hash_manifest.json"],
        "final_hash_manifest_sha256": sha256_file(
            report_dir / "final_hash_manifest.json"
        ),
        "model_or_checkpoint_loaded": False,
        "test_object_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if _is_within(args.report_dir, args.data_root):
            raise ValueError("report directory must be outside the exact data root")
        if _is_within(args.summary, args.data_root) or _is_within(
            args.summary, args.report_dir
        ):
            raise ValueError(
                "summary must be outside the exact data root and report directory"
            )
        summary = run_audit(args.manifest, args.data_root, args.report_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

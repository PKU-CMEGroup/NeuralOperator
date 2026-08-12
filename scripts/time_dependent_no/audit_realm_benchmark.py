"""Audit the frozen IgnitHIT open-population manifest without opening data.

The input is a local JSON inventory assembled from public repository metadata.
This command performs no network access and rejects every path with a ``test``
component before it reports a valid train/validation manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Local normalized JSON manifest; no repository object is opened.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON summary path; otherwise print to stdout.",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = audit_manifest(args.manifest)
    except (TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

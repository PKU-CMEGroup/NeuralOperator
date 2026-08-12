"""Acquire only the pinned D088 IgnitHIT train/validation object manifest.

The command consumes a previously frozen normalized metadata manifest, rejects
every test path, downloads by immutable revision, verifies every LFS SHA-256 or
Git blob ID, and then verifies that the destination contains exactly the open
manifest. It never discovers or expands the population on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_REPOSITORY,
    IGNITHIT_REVISION,
    ManifestEntry,
    canonical_json_sha256,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)
from utility.time_dependent_no.realm_ignithit import (
    validate_local_open_tree,
    verify_manifest_file,
)

SCHEMA = "d088_ignithit_open_acquisition_v1"
DEFAULT_BASE_URL = "https://huggingface.co/datasets"
OpenUrl = Callable[[str], BinaryIO]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], tuple[ManifestEntry, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("manifest root must be a JSON object")
    repository, revision, entries = parse_manifest_payload(payload)
    validate_ignithit_open_manifest(repository, revision, entries)
    return payload, entries


def object_url(entry: ManifestEntry, *, base_url: str = DEFAULT_BASE_URL) -> str:
    base = base_url.rstrip("/")
    repository = quote(IGNITHIT_REPOSITORY, safe="/")
    revision = quote(IGNITHIT_REVISION, safe="")
    path = quote(entry.path, safe="/=_-.")
    return f"{base}/{repository}/resolve/{revision}/{path}"


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _preflight_acquisition_root(
    root: Path,
    entries: Sequence[ManifestEntry],
) -> None:
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError("data root must be a directory and not a symlink")
    expected_files = {entry.path for entry in entries}
    expected_directories = {
        PurePosixPath(*path.parts[:index]).as_posix()
        for entry in entries
        for path in (PurePosixPath(entry.path),)
        for index in range(1, len(path.parts))
    }
    for path in root.rglob("*"):
        relative = PurePosixPath(*path.relative_to(root).parts)
        if any(part.casefold() == "test" for part in relative.parts):
            raise ValueError(f"sealed test path exists before acquisition: {relative}")
        if path.is_symlink():
            raise ValueError(f"symlink exists before acquisition: {relative}")
        if path.is_file() and relative.as_posix() not in expected_files:
            raise ValueError(f"unexpected file exists before acquisition: {relative}")
        if path.is_dir() and relative.as_posix() not in expected_directories:
            raise ValueError(
                f"unexpected directory exists before acquisition: {relative}"
            )


def _open_official_url(url: str) -> BinaryIO:
    request = Request(
        url,
        headers={"User-Agent": "NeuralOperator-D088-metadata-bound-acquisition/1"},
    )
    return urlopen(request, timeout=120)


def acquire_entry(
    root: Path,
    entry: ManifestEntry,
    *,
    open_url: OpenUrl = _open_official_url,
) -> tuple[dict[str, Any], bool]:
    target = root.joinpath(*PurePosixPath(entry.path).parts)
    if target.exists() or target.is_symlink():
        return verify_manifest_file(root, entry), True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"temporary acquisition path already exists: {entry.path}")
    try:
        content_sha256 = hashlib.sha256()
        git_oid = hashlib.sha1(usedforsecurity=False)
        git_oid.update(f"blob {entry.size}\0".encode())
        with open_url(object_url(entry)) as response, temporary.open("xb") as handle:
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > entry.size:
                    raise ValueError(f"download exceeds registered size: {entry.path}")
                handle.write(chunk)
                content_sha256.update(chunk)
                git_oid.update(chunk)
        if copied != entry.size:
            raise ValueError(f"download size mismatch: {entry.path}")
        actual_oid = content_sha256.hexdigest() if entry.lfs else git_oid.hexdigest()
        if actual_oid != entry.oid:
            raise ValueError(f"downloaded object mismatch: {entry.path}")
        os.replace(temporary, target)
        return {
            "path": entry.path,
            "size": entry.size,
            "oid": entry.oid,
            "lfs": entry.lfs,
            "content_sha256": content_sha256.hexdigest(),
        }, False
    except BaseException:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def acquire_open_manifest(
    root: Path,
    entries: Sequence[ManifestEntry],
    *,
    open_url: OpenUrl = _open_official_url,
) -> dict[str, Any]:
    _preflight_acquisition_root(root, entries)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reused_count = 0
    downloaded_bytes = 0
    for entry in sorted(entries):
        row, reused = acquire_entry(root, entry, open_url=open_url)
        row["reused_verified_file"] = reused
        rows.append(row)
        if reused:
            reused_count += 1
        else:
            downloaded_bytes += entry.size
    final_rows = validate_local_open_tree(root, entries)
    return {
        "schema": SCHEMA,
        "repository": IGNITHIT_REPOSITORY,
        "revision": IGNITHIT_REVISION,
        "entry_count": len(entries),
        "total_bytes": sum(entry.size for entry in entries),
        "downloaded_bytes_this_invocation": downloaded_bytes,
        "reused_verified_file_count": reused_count,
        "files": rows,
        "final_tree": final_rows,
        "sealed_test_objects_absent": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload, entries = _load_manifest(args.manifest)
        if _is_within(args.summary, args.data_root):
            raise ValueError("summary must be outside the exact data root")
        summary = acquire_open_manifest(args.data_root, entries)
        summary["normalized_manifest_payload_sha256"] = canonical_json_sha256(payload)
    except (OSError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

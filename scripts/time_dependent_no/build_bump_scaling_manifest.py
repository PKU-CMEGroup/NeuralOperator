"""Build the field-blind D094 bump trajectory-scaling split manifest.

The input is the small JSON manifest produced by the maintained bump shard
preprocessor.  This script never opens state or geometry arrays.  It reserves a
stratified open-validation population, then constructs nested maximin training
subsets from condition and coarse graph metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "d094_bump_trajectory_scaling_split_v1"
DEFAULT_EXPOSURE_COUNTS = (8, 16, 32, 64, 128, 256)
DEFAULT_SPLIT_SEED = 20260820
DEFAULT_VALIDATION_COUNT = 44
EXPECTED_SOURCE_COUNT = 300
SOURCE_FIELDS_USED = (
    "key",
    "mach",
    "num_nodes",
    "num_directed_edges",
    "num_steps",
    "geometry_digest",
)
SELECTION_FEATURES = ("mach", "num_nodes", "directed_edges_per_node")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _source_records(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_trajectories = source.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("source manifest must contain a nonempty trajectory list")

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_geometry: set[str] = set()
    for source_index, raw in enumerate(raw_trajectories):
        if not isinstance(raw, Mapping):
            raise TypeError("every source trajectory must be a mapping")
        missing = set(SOURCE_FIELDS_USED) - set(raw)
        if missing:
            raise ValueError(f"source trajectory is missing metadata: {sorted(missing)}")
        key = str(raw["key"])
        geometry_digest = str(raw["geometry_digest"])
        if not key or key in seen_keys:
            raise ValueError("source trajectory keys must be nonempty and unique")
        if not geometry_digest or geometry_digest in seen_geometry:
            raise ValueError("geometry digests must be nonempty and unique")
        mach = float(raw["mach"])
        if not math.isfinite(mach) or mach <= 0.0:
            raise ValueError("Mach metadata must be finite and positive")
        num_nodes = _require_int(raw["num_nodes"], "num_nodes")
        num_directed_edges = _require_int(
            raw["num_directed_edges"], "num_directed_edges"
        )
        num_steps = _require_int(raw["num_steps"], "num_steps")
        records.append(
            {
                "source_index": source_index,
                "key": key,
                "mach": mach,
                "num_nodes": num_nodes,
                "num_directed_edges": num_directed_edges,
                "directed_edges_per_node": num_directed_edges / num_nodes,
                "num_steps": num_steps,
                "geometry_digest": geometry_digest,
            }
        )
        seen_keys.add(key)
        seen_geometry.add(geometry_digest)
    return records


def _array_split(values: Sequence[int], count: int) -> list[list[int]]:
    quotient, remainder = divmod(len(values), count)
    chunks: list[list[int]] = []
    start = 0
    for chunk_index in range(count):
        size = quotient + int(chunk_index < remainder)
        chunks.append(list(values[start : start + size]))
        start += size
    if start != len(values) or any(not chunk for chunk in chunks):
        raise RuntimeError("stratification chunks did not close")
    return chunks


def _validation_indices(
    records: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> list[int]:
    if count < 1 or count >= len(records):
        raise ValueError("validation count must lie between one and N-1")
    ordered = sorted(
        range(len(records)),
        key=lambda index: (
            float(records[index]["mach"]),
            int(records[index]["num_nodes"]),
            int(records[index]["source_index"]),
        ),
    )
    selected: list[int] = []
    for chunk in _array_split(ordered, count):
        selected.append(
            min(
                chunk,
                key=lambda index: (
                    hashlib.sha256(
                        f"{seed}:{records[index]['key']}".encode()
                    ).hexdigest(),
                    int(records[index]["source_index"]),
                ),
            )
        )
    return sorted(selected)


def _normalized_features(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[int, tuple[float, ...]]:
    columns = [
        [float(records[index][feature]) for index in indices]
        for feature in SELECTION_FEATURES
    ]
    limits = [(min(column), max(column)) for column in columns]
    normalized: dict[int, tuple[float, ...]] = {}
    for index in indices:
        values: list[float] = []
        for feature, (lower, upper) in zip(
            SELECTION_FEATURES, limits, strict=True
        ):
            value = float(records[index][feature])
            values.append(0.0 if upper == lower else (value - lower) / (upper - lower))
        normalized[index] = tuple(values)
    return normalized


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _maximin_order(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> list[int]:
    if not indices:
        raise ValueError("maximin selection requires candidates")
    candidates = sorted(indices)
    if len(candidates) == 1:
        return candidates
    features = _normalized_features(records, candidates)

    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    for left_position, left in enumerate(candidates[:-1]):
        for right in candidates[left_position + 1 :]:
            distance = _squared_distance(features[left], features[right])
            pair = (left, right)
            if distance > best_distance or (
                distance == best_distance and (best_pair is None or pair < best_pair)
            ):
                best_distance = distance
                best_pair = pair
    if best_pair is None:
        raise RuntimeError("maximin initial pair was not resolved")

    selected = list(best_pair)
    selected_set = set(selected)
    while len(selected) < len(candidates):
        next_index: int | None = None
        next_score = -1.0
        for candidate in candidates:
            if candidate in selected_set:
                continue
            score = min(
                _squared_distance(features[candidate], features[prior])
                for prior in selected
            )
            if score > next_score or (
                score == next_score
                and (next_index is None or candidate < next_index)
            ):
                next_score = score
                next_index = candidate
        if next_index is None:
            raise RuntimeError("maximin addition did not resolve a candidate")
        selected.append(next_index)
        selected_set.add(next_index)
    return selected


def build_scaling_manifest(
    source: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    expected_source_count: int = EXPECTED_SOURCE_COUNT,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    split_seed: int = DEFAULT_SPLIT_SEED,
    exposure_counts: Sequence[int] = DEFAULT_EXPOSURE_COUNTS,
) -> dict[str, Any]:
    if len(source_manifest_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in source_manifest_sha256
    ):
        raise ValueError("source manifest SHA-256 must be lowercase hexadecimal")
    records = _source_records(source)
    if len(records) != expected_source_count:
        raise ValueError("source trajectory count differs from the frozen contract")
    validation = _validation_indices(records, count=validation_count, seed=split_seed)
    validation_set = set(validation)
    training = [index for index in range(len(records)) if index not in validation_set]

    counts = tuple(int(value) for value in exposure_counts)
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or counts[0] < 2
        or counts[-1] != len(training)
    ):
        raise ValueError(
            "exposure counts must be unique, increasing, and end at the train pool"
        )
    addition_order = _maximin_order(records, training)
    subsets: dict[str, list[str]] = {}
    for count in counts:
        selected = set(addition_order[:count])
        subsets[str(count)] = [
            str(records[index]["key"]) for index in training if index in selected
        ]

    field_blind_metadata = [
        {name: record[name] for name in SOURCE_FIELDS_USED}
        for record in records
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "source_schema_version": source.get("schema_version"),
        "source_trajectory_count": len(records),
        "source_fields_used": list(SOURCE_FIELDS_USED),
        "source_metadata_digest": canonical_json_sha256(field_blind_metadata),
        "state_arrays_opened": False,
        "historical_test_population_opened": False,
        "split": {
            "algorithm": (
                "sort by Mach,node count,source index; array-split into validation "
                "strata; select each stratum by SHA256(seed:key)"
            ),
            "seed": int(split_seed),
            "train_pool_count": len(training),
            "open_validation_count": len(validation),
            "train_pool_keys": [str(records[index]["key"]) for index in training],
            "open_validation_keys": [
                str(records[index]["key"]) for index in validation
            ],
        },
        "nested_exposure": {
            "algorithm": "greedy maximin in min-max-normalized metadata",
            "features": list(SELECTION_FEATURES),
            "addition_order": [
                str(records[index]["key"]) for index in addition_order
            ],
            "counts": list(counts),
            "subsets": subsets,
        },
        "metadata_ranges": {
            "mach": [
                min(float(record["mach"]) for record in records),
                max(float(record["mach"]) for record in records),
            ],
            "num_nodes": [
                min(int(record["num_nodes"]) for record in records),
                max(int(record["num_nodes"]) for record in records),
            ],
            "num_directed_edges": [
                min(int(record["num_directed_edges"]) for record in records),
                max(int(record["num_directed_edges"]) for record in records),
            ],
            "num_steps": sorted({int(record["num_steps"]) for record in records}),
            "unique_geometry_digests": len(
                {str(record["geometry_digest"]) for record in records}
            ),
        },
    }
    payload["partition_digest"] = canonical_json_sha256(
        {
            "split": payload["split"],
            "nested_exposure": payload["nested_exposure"],
        }
    )
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-count", type=int, default=EXPECTED_SOURCE_COUNT)
    parser.add_argument("--validation-count", type=int, default=DEFAULT_VALIDATION_COUNT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--exposure-counts", type=int, nargs="+", default=DEFAULT_EXPOSURE_COUNTS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with args.source_manifest.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    payload = build_scaling_manifest(
        source,
        source_manifest_sha256=sha256_file(args.source_manifest),
        expected_source_count=args.expected_source_count,
        validation_count=args.validation_count,
        split_seed=args.split_seed,
        exposure_counts=args.exposure_counts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "source_trajectory_count": payload["source_trajectory_count"],
                "train_pool_count": payload["split"]["train_pool_count"],
                "open_validation_count": payload["split"]["open_validation_count"],
                "exposure_counts": payload["nested_exposure"]["counts"],
                "partition_digest": payload["partition_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

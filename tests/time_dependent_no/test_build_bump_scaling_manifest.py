from __future__ import annotations

import copy

import pytest

from scripts.time_dependent_no.build_bump_scaling_manifest import (
    build_scaling_manifest,
)


def _source(count: int = 12) -> dict[str, object]:
    return {
        "schema_version": 1,
        "trajectories": [
            {
                "key": str(index),
                "mach": 2.6 + 0.07 * index,
                "num_nodes": 100 + (index % 5) * 11,
                "num_directed_edges": 400 + (index % 4) * 17,
                "num_steps": 80,
                "geometry_digest": f"geometry-{index:03d}",
                "state_digest": f"state-{index:03d}",
                "folder": f"private-folder-{index}",
            }
            for index in range(count)
        ],
    }


def _build(source: dict[str, object]) -> dict[str, object]:
    return build_scaling_manifest(
        source,
        source_manifest_sha256="a" * 64,
        expected_source_count=12,
        validation_count=4,
        split_seed=17,
        exposure_counts=(2, 4, 8),
    )


def test_split_is_deterministic_disjoint_nested_and_field_blind() -> None:
    source = _source()
    result = _build(source)
    repeated = _build(copy.deepcopy(source))

    assert result == repeated
    assert result["state_arrays_opened"] is False
    assert result["historical_test_population_opened"] is False
    split = result["split"]
    train = set(split["train_pool_keys"])
    validation = set(split["open_validation_keys"])
    assert len(train) == 8
    assert len(validation) == 4
    assert train.isdisjoint(validation)
    assert train | validation == {str(index) for index in range(12)}

    subsets = result["nested_exposure"]["subsets"]
    assert [len(subsets[str(count)]) for count in (2, 4, 8)] == [2, 4, 8]
    assert set(subsets["2"]) < set(subsets["4"]) < set(subsets["8"])
    assert set(subsets["8"]) == train

    changed_states = copy.deepcopy(source)
    for trajectory in changed_states["trajectories"]:
        trajectory["state_digest"] += "-changed"
        trajectory["folder"] += "-changed"
    changed = _build(changed_states)
    assert changed["source_metadata_digest"] == result["source_metadata_digest"]
    assert changed["split"] == result["split"]
    assert changed["nested_exposure"] == result["nested_exposure"]


def test_split_changes_reproducibly_with_seed() -> None:
    source = _source()
    first = _build(source)
    second = build_scaling_manifest(
        source,
        source_manifest_sha256="a" * 64,
        expected_source_count=12,
        validation_count=4,
        split_seed=18,
        exposure_counts=(2, 4, 8),
    )

    assert first["split"]["open_validation_keys"] != second["split"][
        "open_validation_keys"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source["trajectories"].pop(), "trajectory count"),
        (
            lambda source: source["trajectories"][1].update(key="0"),
            "keys must be nonempty and unique",
        ),
        (
            lambda source: source["trajectories"][1].update(
                geometry_digest="geometry-000"
            ),
            "geometry digests",
        ),
        (
            lambda source: source["trajectories"][0].update(mach=float("nan")),
            "Mach metadata",
        ),
    ],
)
def test_source_contract_rejects_drift(mutation, message: str) -> None:
    source = _source()
    mutation(source)
    with pytest.raises((TypeError, ValueError), match=message):
        _build(source)


def test_exposure_contract_must_end_at_full_training_pool() -> None:
    with pytest.raises(ValueError, match="end at the train pool"):
        build_scaling_manifest(
            _source(),
            source_manifest_sha256="a" * 64,
            expected_source_count=12,
            validation_count=4,
            split_seed=17,
            exposure_counts=(2, 4),
        )


def test_source_hash_must_be_canonical_lowercase_hex() -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        build_scaling_manifest(
            _source(),
            source_manifest_sha256="Z" * 64,
            expected_source_count=12,
            validation_count=4,
            split_seed=17,
            exposure_counts=(2, 4, 8),
        )

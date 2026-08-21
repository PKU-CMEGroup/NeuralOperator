from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.time_dependent_no.evaluate_realm_planardet_scaling import (
    BASE_EVALUATOR_PATH,
    EVALUATION_SOURCE_PATHS,
    EVALUATOR_PATH,
    _history_best,
    _regular_files,
    _validate_signed_payload,
    build_parser,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256
from utility.time_dependent_no.realm_planardet_scaling import SCALING_SOURCE_PATHS


def _history_row(step: int, score: float, *, eligible: bool = True) -> dict[str, object]:
    return {
        "completed_step": step,
        "checkpoint_eligible": eligible,
        "validation": {"realm_npe_mean": score},
    }


def test_history_best_uses_only_eligible_rows() -> None:
    rows = [
        _history_row(1, 0.01, eligible=False),
        _history_row(50, 0.4),
        _history_row(100, 0.2),
        _history_row(150, 0.3),
    ]

    recovered, best = _history_best({"rows": rows})

    assert recovered == rows
    assert best["completed_step"] == 100
    assert best["validation"]["realm_npe_mean"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "rows",
    (
        [],
        [_history_row(50, 0.2)],
        [_history_row(1, 0.2), _history_row(1, 0.1)],
        [_history_row(1, 0.2, eligible=False)],
    ),
)
def test_history_best_rejects_incomplete_or_ambiguous_sequences(
    rows: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        _history_best({"rows": rows})


def test_signed_payload_validation_is_fail_closed() -> None:
    payload = {"schema": "synthetic", "value": 3}
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    _validate_signed_payload(payload, name="synthetic")

    payload["value"] = 4
    with pytest.raises(ValueError, match="canonical payload digest differs"):
        _validate_signed_payload(payload, name="synthetic")


def test_regular_files_rejects_nested_or_symlink_objects(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(json.dumps({"rows": []}))
    assert _regular_files(tmp_path) == {"history.json"}

    (tmp_path / "nested").mkdir()
    with pytest.raises(ValueError, match="non-regular"):
        _regular_files(tmp_path)


def test_evaluator_source_inventory_includes_both_evaluators_and_scaling() -> None:
    assert EVALUATOR_PATH in EVALUATION_SOURCE_PATHS
    assert BASE_EVALUATOR_PATH in EVALUATION_SOURCE_PATHS
    assert set(SCALING_SOURCE_PATHS).issubset(EVALUATION_SOURCE_PATHS)
    assert len(EVALUATION_SOURCE_PATHS) == len(set(EVALUATION_SOURCE_PATHS))


def test_parser_exposes_no_test_or_checkpoint_selection_argument() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "test" not in destinations
    assert "test_root" not in destinations
    assert "checkpoint" not in destinations
    assert "training_dir" in destinations
    assert "training_log" in destinations
    assert "training_exit_file" in destinations

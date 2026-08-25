from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.time_dependent_no import evaluate_pcno_bump_b1_c3_r1 as r1


def test_registered_sha256_values_have_canonical_shape() -> None:
    for value in (
        r1.EXPECTED_SOURCE_ARCHIVE_SHA256,
        r1.EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        r1.EXPECTED_CLOSEOUT_MATRIX_RECEIPT_SHA256,
        r1.EXPECTED_CLOSEOUT_ARTIFACT_MANIFEST_SHA256,
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", value)


def _receipt() -> dict:
    cells = []
    for count in r1.TRAJECTORY_COUNTS:
        for architecture in r1.ARCHITECTURES:
            name = (
                f"b1_c3_r1_s{r1.SEED}_n{count}_{architecture}_"
                "stretched_20480_20260825a"
            )
            step = r1.matched_exposure_step(count)
            cells.append(
                {
                    "trajectory_count": count,
                    "architecture": architecture,
                    "cell": name,
                    "sentinel_step": step,
                    "files": {
                        f"sentinels/step_{step:09d}.pt": {
                            "bytes": 1,
                            "sha256": "checkpoint",
                        }
                    },
                }
            )
    return {"cells": cells}


def test_receipt_file_records_preserves_nested_cell_paths() -> None:
    records = r1._receipt_file_records(_receipt())
    assert len(records) == 12
    assert (
        "results/b1_c3_r1_s20260718_n8_pcno_stretched_20480_20260825a/"
        "sentinels/step_000000512.pt"
    ) in records


def test_receipt_file_records_rejects_cell_inventory_drift() -> None:
    receipt = _receipt()
    receipt["cells"].pop()
    with pytest.raises(ValueError, match="cell inventory changed"):
        r1._receipt_file_records(receipt)


def test_seed0_cohorts_must_match_prior_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [str(index) for index in range(44)]
    selection = validation[:16]
    outside = validation[16:]
    common = outside[:9]
    prior = {
        "selection_keys_by_seed": {str(r1.SEED): selection},
        "outside_selection_keys_by_seed": {str(r1.SEED): outside},
        "common_outside_selection_keys": common,
    }
    monkeypatch.setattr(
        r1,
        "outside_selection_keys",
        lambda _manifest, _splits: (validation, selection, outside),
    )
    monkeypatch.setattr(
        r1,
        "_ordered_key_digest",
        lambda keys: (
            r1.EXPECTED_VALIDATION_KEY_DIGEST
            if len(keys) == 44
            else r1.EXPECTED_COMMON_OUTSIDE_KEY_DIGEST
        ),
    )
    cohorts = r1._seed0_cohorts(
        {"split": {}}, [{"split": {}} for _ in range(12)], prior
    )
    assert cohorts["selection_keys"] == selection
    assert cohorts["outside_selection_keys"] == outside
    assert cohorts["common_outside_selection_keys"] == common


def test_seed0_cohorts_rejects_changed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = [str(index) for index in range(44)]
    selection = validation[:16]
    outside = validation[16:]
    prior = {
        "selection_keys_by_seed": {str(r1.SEED): list(reversed(selection))},
        "outside_selection_keys_by_seed": {str(r1.SEED): outside},
        "common_outside_selection_keys": outside[:9],
    }
    monkeypatch.setattr(
        r1,
        "outside_selection_keys",
        lambda _manifest, _splits: (validation, selection, outside),
    )
    monkeypatch.setattr(
        r1, "_ordered_key_digest", lambda _keys: r1.EXPECTED_VALIDATION_KEY_DIGEST
    )
    with pytest.raises(ValueError, match="cohort changed"):
        r1._seed0_cohorts({"split": {}}, [{"split": {}}], prior)


def test_discover_sentinels_binds_registered_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    for cell in receipt["cells"]:
        root = tmp_path / "results" / cell["cell"]
        root.mkdir(parents=True)
        architecture = cell["architecture"]
        count = cell["trajectory_count"]
        mode = "full" if architecture == "pcno" else "no_gradient"
        summary = {
            "config_digest": f"config-{count}-{architecture}",
            "normalization_digest": f"normalization-{count}",
        }
        contract = {
            "registered_stage": r1.REGISTERED_STAGE,
            "initialization_seed": r1.SEED,
            "trajectory_count": count,
            "differential_branch_mode": mode,
            "actual_optimizer_steps": 20_480,
        }
        cell["config_digest"] = summary["config_digest"]
        cell["normalization_digest"] = summary["normalization_digest"]
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "bump_scaling_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        (root / "split.json").write_text(
            json.dumps({"val_keys": ["v"]}), encoding="utf-8"
        )

    seen = []

    def fake_descriptor(base: dict, **keywords: object) -> dict:
        seen.append((base["trajectory_count"], base["architecture"], keywords))
        return dict(base)

    monkeypatch.setattr(r1, "_sentinel_descriptor", fake_descriptor)
    descriptors = r1.discover_sentinels(tmp_path, receipt)
    assert len(descriptors) == 12
    assert all(
        call[2]["expected_source_set_digest"]
        == r1.EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
        for call in seen
    )


def test_parser_requires_explicit_precision() -> None:
    parser = r1.build_parser()
    base = [
        "--attempt-root",
        "attempt",
        "--closeout-root",
        "closeout",
        "--b1-c2-audit-root",
        "prior",
        "--data-dir",
        "data",
        "--output-dir",
        "output",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    assert parser.parse_args([*base, "--amp", "bf16"]).amp == "bf16"

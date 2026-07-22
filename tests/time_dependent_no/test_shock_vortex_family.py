from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.build_euler2d_shock_vortex_family import (
    main as build_family_main,
)
from scripts.time_dependent_no.generate_euler2d_shock_vortex_family_case import (
    parse_args as parse_family_case_args,
)
from utility.time_dependent_no.shock_vortex_family import (
    CANONICAL_CASE_ID,
    FAMILY_ID,
    FAMILY_SCHEMA,
    SMOKE_CASE_IDS,
    audit_shock_vortex_family_artifacts,
    build_shock_vortex_family_manifest,
    canonical_json_digest,
    config_for_family_case,
    family_case_by_id,
    family_case_provenance,
    load_shock_vortex_family_manifest,
    validate_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_fv import (
    shock_vortex_initial_primitive,
)

EXPECTED_MANIFEST_DIGEST = (
    "150c589af9c7291674f502dfe30ca77930d9429fb2115b3f2a0f22042a08c69e"
)


def test_frozen_family_manifest_has_grouped_ood_split_and_canonical_case() -> None:
    manifest = build_shock_vortex_family_manifest()

    assert manifest["schema"] == FAMILY_SCHEMA
    assert manifest["family_id"] == FAMILY_ID
    assert manifest["manifest_digest_sha256"] == EXPECTED_MANIFEST_DIGEST
    assert manifest["split_contract"]["split_counts"] == {
        "train": 84,
        "validation": 24,
        "test": 27,
    }
    assert manifest["split_contract"]["split_group_counts"] == {
        "train": 12,
        "validation": 2,
        "test": 3,
    }
    assert len(manifest["cases"]) == 135
    assert len({case["case_id"] for case in manifest["cases"]}) == 135
    assert [case["trajectory_index"] for case in manifest["cases"]] == list(range(135))
    assert tuple(manifest["smoke_case_ids"]) == SMOKE_CASE_IDS

    canonical = family_case_by_id(manifest, CANONICAL_CASE_ID)
    assert canonical["split"] == "train"
    assert canonical["canonical_physical_case"] is True
    assert canonical["parameters"] == {
        "vortex_epsilon": 0.3,
        "vortex_y": 0.5,
    }

    for case in manifest["cases"]:
        epsilon_index = case["epsilon_index"]
        y_index = case["y_index"]
        if epsilon_index >= 12:
            assert case["split"] == "test"
        elif y_index in (0, 8):
            assert case["split"] == "validation"
        else:
            assert case["split"] == "train"


def test_family_manifest_validation_fails_closed_on_mutation() -> None:
    manifest = build_shock_vortex_family_manifest()
    mutated = json.loads(json.dumps(manifest))
    mutated["cases"][0]["split"] = "train"

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_shock_vortex_family_manifest(mutated)

    unsigned = dict(mutated)
    unsigned.pop("manifest_digest_sha256")
    mutated["manifest_digest_sha256"] = canonical_json_digest(unsigned)
    with pytest.raises(ValueError, match="differs from the frozen contract"):
        validate_shock_vortex_family_manifest(mutated)


def test_family_case_config_and_provenance_are_exact() -> None:
    manifest = build_shock_vortex_family_manifest()
    config = config_for_family_case(manifest, "sv_e13_y04")
    provenance = family_case_provenance(manifest, "sv_e13_y04")

    assert (config.nx, config.ny) == (1000, 400)
    assert (config.coarse_nx, config.coarse_ny) == (250, 100)
    assert config.output_times == tuple(index / 100.0 for index in range(61))
    assert config.vortex_epsilon == pytest.approx(0.375)
    assert config.vortex_y == pytest.approx(0.5)
    assert provenance == {
        "manifest_schema": FAMILY_SCHEMA,
        "family_id": FAMILY_ID,
        "manifest_digest_sha256": EXPECTED_MANIFEST_DIGEST,
        "case_id": "sv_e13_y04",
        "trajectory_index": 121,
        "split": "test",
        "split_group_id": "strength_ood_e13",
        "parameters": {"vortex_epsilon": 0.375, "vortex_y": 0.5},
        "canonical_physical_case": False,
    }


@pytest.mark.parametrize("case_id", ("sv_e00_y00", "sv_e14_y08"))
def test_family_parameter_extremes_have_admissible_initial_states(case_id: str) -> None:
    manifest = build_shock_vortex_family_manifest()
    config = config_for_family_case(manifest, case_id)
    x = torch.linspace(config.x_min, config.shock_x, 81, dtype=torch.float64)
    y = torch.linspace(config.y_min, config.y_max, 41, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    primitive = shock_vortex_initial_primitive(points, config)

    assert torch.all(torch.isfinite(primitive))
    assert float(torch.min(primitive[:, 0])) > 0.0
    assert float(torch.min(primitive[:, 3])) > 0.0


def test_manifest_entrypoint_writes_once_and_case_cli_is_manifest_only(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "family.json"
    build_family_main(["build", "--output-path", str(manifest_path)])
    loaded = load_shock_vortex_family_manifest(manifest_path)
    assert loaded["manifest_digest_sha256"] == EXPECTED_MANIFEST_DIGEST

    with pytest.raises(FileExistsError, match="already exists"):
        build_family_main(["build", "--output-path", str(manifest_path)])

    args = parse_family_case_args(
        [
            "--manifest",
            str(manifest_path),
            "--case-id",
            CANONICAL_CASE_ID,
            "--output-dir",
            str(tmp_path / CANONICAL_CASE_ID),
            "--device",
            "cpu",
        ]
    )
    assert args.case_id == CANONICAL_CASE_ID
    assert args.device == "cpu"


def test_family_artifact_audit_reports_missing_smoke_cases(tmp_path: Path) -> None:
    manifest = build_shock_vortex_family_manifest()
    audit = audit_shock_vortex_family_artifacts(
        manifest,
        tmp_path,
        manifest["smoke_case_ids"],
    )

    assert audit["status"] == "failed"
    assert audit["case_count"] == 3
    assert audit["passed_case_count"] == 0
    assert audit["failed_case_count"] == 3
    assert audit["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert all(
        set(row["failed_checks"])
        == {
            "summary_missing",
            "reference_artifact_missing",
        }
        for row in audit["rows"]
    )


def test_family_manifest_rejects_unknown_case() -> None:
    manifest = build_shock_vortex_family_manifest()
    with pytest.raises(ValueError, match="not unique and present"):
        config_for_family_case(manifest, "not-a-case")
    with pytest.raises(ValueError, match="unknown family case_ids"):
        audit_shock_vortex_family_artifacts(manifest, Path("."), ["not-a-case"])


def test_manifest_parameters_are_finite_and_unique() -> None:
    manifest = build_shock_vortex_family_manifest()
    parameters = np.asarray(
        [
            (
                case["parameters"]["vortex_epsilon"],
                case["parameters"]["vortex_y"],
            )
            for case in manifest["cases"]
        ],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(parameters))
    assert np.unique(parameters, axis=0).shape[0] == parameters.shape[0]

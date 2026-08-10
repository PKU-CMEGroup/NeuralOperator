from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.time_dependent_no.evaluate_pcno_deterministic_response_controller as d078
import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076
import scripts.time_dependent_no.evaluate_pcno_strength_grouped_response_controller as d077
from scripts.time_dependent_no.evaluate_pcno_native_residual_correction import (
    Candidate,
)


def _cases(energies=(1, 2, 3)):
    cases = []
    for energy in energies:
        for position, vortex_y in (("y00", 0.1), ("y08", 0.9)):
            case_id = f"sv_e{energy:02d}_{position}"
            cases.append(
                SimpleNamespace(
                    case_id=case_id,
                    provenance={
                        "case_id": case_id,
                        "split": "validation",
                        "parameters": {
                            "vortex_epsilon": float(energy),
                            "vortex_y": vortex_y,
                        },
                    },
                    reference_states=np.zeros((3, 4, 4), dtype=np.float64),
                )
            )
    return cases


def _features(amplitude: float, centroid: float, gain: float) -> dict[str, float]:
    return {
        "initial_vortex_amplitude": amplitude,
        "initial_vortex_centroid_y": centroid,
        "response_state_rms": 0.1 * gain,
        "response_integral__rho": 0.01 * gain,
        "response_integral__rho_u": 0.02 * gain,
        "response_integral__rho_v": 0.03 * gain,
        "response_integral__energy": 0.04 * gain,
        "response_amplification": 0.8,
        "response_correction_cosine": 0.9,
    }


def test_strength_group_builder_binds_complete_manifest_pairs() -> None:
    cases = _cases(d077.CALIBRATION_ENERGY_INDICES)
    observed = d077.build_strength_group_map(cases)
    assert set(observed) == set(d076.parent.DYNAMIC_CALIBRATION_CASES)
    assert set(observed.values()) == set(d077.EXPECTED_GROUP_MEMBERS)
    for group_id, expected_members in d077.EXPECTED_GROUP_MEMBERS.items():
        assert {
            case_id for case_id, value in observed.items() if value == group_id
        } == set(expected_members)


def test_strength_group_builder_rejects_incomplete_or_mixed_pairs() -> None:
    with pytest.raises(ValueError, match="complete y00/y08 pair"):
        d077.build_strength_group_map(_cases()[:-1])

    cases = _cases()
    cases[-1].provenance["parameters"]["vortex_epsilon"] = 99.0
    with pytest.raises(ValueError, match="mixes strength"):
        d077.build_strength_group_map(cases)


def test_d077_smoke_keeps_three_complete_groups_and_four_training_rows() -> None:
    smoke_ids = d076.parent.DYNAMIC_CALIBRATION_CASES[
        : d077.D077_SPEC.smoke_calibration_case_count
    ]
    cases = _cases((1, 2, 3))
    observed = d077.build_strength_group_map(cases)
    assert tuple(case.case_id for case in cases) == smoke_ids
    assert len(set(observed.values())) == 3
    for case_id, group_id in observed.items():
        assert (
            sum(training_group != group_id for training_group in observed.values()) == 4
        )
        assert case_id in d077.EXPECTED_GROUP_MEMBERS[group_id]


def test_grouped_crossfit_excludes_pair_from_coefficients_and_policy(
    monkeypatch,
) -> None:
    cases = _cases()
    candidates = (
        Candidate(key="zero", rank=0, gain=0.0),
        Candidate(key="rank8_gain0p125_raw", rank=8, gain=0.125),
        Candidate(key="rank8_gain0p5_raw", rank=8, gain=0.5),
    )
    case_ids = tuple(sorted(case.case_id for case in cases))
    coefficient_value = {
        case_id: float(index) for index, case_id in enumerate(case_ids, start=1)
    }
    observed_frozen: dict[str, float] = {}

    monkeypatch.setattr(
        d076,
        "_initial_vortex_features",
        lambda case, state: (
            float(case.provenance["parameters"]["vortex_epsilon"]),
            float(case.provenance["parameters"]["vortex_y"]),
        ),
    )
    monkeypatch.setattr(
        d076.parent,
        "_fit_case_coefficients",
        lambda model, case, ranks: {
            8: np.full(
                (2, 1, 1),
                coefficient_value[case.case_id],
                dtype=np.float64,
            )
        },
    )

    def fake_bias_sequence(case, frozen, *, rank):
        assert rank == 8
        observed_frozen[case.case_id] = float(np.asarray(frozen).reshape(-1)[0])
        return np.zeros((1, 1)), np.zeros((2, 4, 4), dtype=np.float64)

    monkeypatch.setattr(d076.parent, "_bias_sequence", fake_bias_sequence)
    monkeypatch.setattr(
        d076.parent,
        "_rollout_candidate",
        lambda model, case, candidate, *, bias_sequence, shock_quantile: (
            SimpleNamespace(candidate=candidate, complete=True, summary={})
        ),
    )
    monkeypatch.setattr(
        d076.parent,
        "_selector_rollout_summary",
        lambda result: result,
    )
    monkeypatch.setattr(
        d076,
        "_response_probe_features",
        lambda case, baseline, corrected, *, probe_calls: _features(
            float(case.provenance["parameters"]["vortex_epsilon"]),
            float(case.provenance["parameters"]["vortex_y"]),
            float(corrected.candidate.gain),
        ),
    )

    def fake_select_candidate(
        observed_candidates,
        observed_case_ids,
        compact,
        *,
        family,
    ):
        assert family == "dynamic_fv"
        assert set(compact) == {candidate.key for candidate in observed_candidates}
        rows = []
        for candidate in observed_candidates:
            for case_id in observed_case_ids:
                rows.append(
                    {
                        "case_id": case_id,
                        "candidate": candidate.key,
                        "eligible": True,
                        "complete": True,
                        "endpoint_state_ratio": (
                            1.0 if candidate.is_zero else 0.9 - 0.05 * candidate.gain
                        ),
                        "residual_rms_ratio": (1.0 if candidate.is_zero else 0.99),
                        "maximum_control_ratio": 1.0,
                    }
                )
        return observed_candidates[1], rows, [], []

    monkeypatch.setattr(d076.parent, "select_candidate", fake_select_candidate)
    group_by_case = d077.build_strength_group_map(cases)
    result = d076._crossfit_response_calibration(
        object(),
        cases,
        candidates,
        shock_quantile=0.9,
        probe_calls=1,
        smoke=True,
        group_by_case=group_by_case,
        require_group_amplitude_match=True,
        require_highest_group_abstention=True,
    )

    highest_members = d077.EXPECTED_GROUP_MEMBERS["sv_e03"]
    expected_highest_fold_mean = np.mean(
        [
            coefficient_value[case_id]
            for case_id in case_ids
            if case_id not in highest_members
        ]
    )
    for case_id in highest_members:
        assert observed_frozen[case_id] == pytest.approx(expected_highest_fold_mean)

    for row in result["fold_inventory_rows"]:
        excluded = set(json.loads(row["excluded_case_ids_json"]))
        coefficient_training = set(
            json.loads(row["coefficient_training_case_ids_json"])
        )
        policy_training = set(json.loads(row["policy_training_case_ids_json"]))
        assert excluded == set(d077.EXPECTED_GROUP_MEMBERS[row["query_group_id"]])
        assert excluded.isdisjoint(coefficient_training)
        assert excluded.isdisjoint(policy_training)
        assert coefficient_training == policy_training

    for row in result["controller_neighbor_rows"]:
        assert row["query_group_id"] != row["neighbor_group_id"]
        assert row["neighbor_case_id"] not in set(
            json.loads(row["excluded_case_ids_json"])
        )

    highest_selections = [
        row
        for row in result["controller_selection_rows"]
        if row["query_group_id"] == "sv_e03"
    ]
    assert len(highest_selections) == 2
    assert all(row["selected_candidate"] == "zero" for row in highest_selections)
    assert result["group_contract_checks"]["highest_group_abstention_pass"]
    assert result["group_contract_checks"]["passed"]
    assert result["qualification"]["target_loading_authorized"]


def test_d077_not_run_checks_are_null_while_d076_defaults_stay_legacy() -> None:
    grouped = d076._initial_evaluation_checks(d077.D077_SPEC)
    assert grouped["evaluation_ran"] is False
    for name in (
        "evaluation_inventory_pass",
        "all_evaluation_complete",
        "probe_prefix_replay_pass",
        "closure_pass",
        "maximum_recurrence_closure_rms",
    ):
        assert grouped[name] is None

    legacy = d076._initial_evaluation_checks(d076.D076_SPEC)
    assert "evaluation_ran" not in legacy
    assert legacy["evaluation_inventory_pass"] is True
    assert legacy["all_evaluation_complete"] is True
    assert legacy["closure_pass"] is True
    assert legacy["maximum_recurrence_closure_rms"] == 0.0


def test_d077_parse_args_preserves_frozen_dynamic_contract(tmp_path) -> None:
    required = d076.parent.REQUIRED_ARTIFACTS["dynamic_fv"]
    args = d077.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--normalization-json",
            str(tmp_path / "normalization.json"),
            "--split-json",
            str(tmp_path / "split.json"),
            "--data-dir",
            str(tmp_path / "shards"),
            "--family-root",
            str(tmp_path / "family"),
            "--multires-reference-root",
            str(tmp_path / "references"),
            "--output-dir",
            str(tmp_path / "output"),
            "--expected-checkpoint-sha256",
            required["checkpoint"],
            "--expected-normalization-sha256",
            required["normalization"],
            "--expected-split-sha256",
            required["split"],
            "--expected-data-manifest-digest",
            required["data_manifest"],
            "--expected-family-manifest-sha256",
            required["family_manifest"],
            "--expected-source-base-git-head",
            "b2193946dd20a350053520e5406b98bfee3c4ae3",
            "--source-manifest",
            str(tmp_path / "manifest.json"),
            "--expected-source-manifest-sha256",
            "a" * 64,
            "--smoke",
        ]
    )
    assert args.family == "dynamic_fv"
    assert args.experiment_contract == d077.EXPERIMENT_CONTRACT
    assert args.resolutions == ("250x100",)
    assert args.training_resolution == "250x100"
    assert args.rollout_calls == 2
    assert args.probe_calls == 1
    assert args.deterministic_algorithms is False
    assert d077.D077_SPEC.smoke_calibration_case_count == 6


def test_d078_changes_only_runtime_and_experiment_identity() -> None:
    assert d076.D076_SPEC.deterministic_algorithms is False
    assert d077.D077_SPEC.deterministic_algorithms is False
    assert d078.D078_SPEC.deterministic_algorithms is True
    for name in (
        "smoke_calibration_case_count",
        "group_builder",
        "group_contract",
        "require_group_amplitude_match",
        "require_highest_group_abstention",
        "null_not_run_evaluation",
    ):
        assert getattr(d078.D078_SPEC, name) == getattr(d077.D077_SPEC, name)
    assert d078.D078_SPEC.schema == d078.SCHEMA
    assert d078.D078_SPEC.experiment_contract == d078.EXPERIMENT_CONTRACT
    assert d078.D078_SPEC.extra_source_paths[:-1] == (
        *d077.D077_SPEC.extra_source_paths,
    )


def test_d078_direct_entry_point_resolves_repo_namespace(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(d078.__file__), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout

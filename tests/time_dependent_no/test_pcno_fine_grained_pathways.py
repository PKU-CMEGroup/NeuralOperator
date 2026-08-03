from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.analyze_pcno_fine_grained_pathways as pathway_evaluator
from scripts.time_dependent_no.analyze_pcno_fine_grained_pathways import (
    CASE_IDS,
    CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT,
    D067_INHERITED_SOURCE_CONTRACT,
    D067_INHERITED_SOURCE_PATHS,
    HISTORICAL_SOURCE_CONTRACT,
    _band_fields,
    _commutator_row,
    _contract_checks_pass,
    _d070c_contract_checks,
    _historical_compatibility_checks,
    _inventory_checks,
    _mechanism_selection,
    _model_state_sha256,
    _recurrence_row,
    _typed_closure_row,
    _verify_d067,
    parse_args,
)
from utility.time_dependent_no.pcno_resolution_pathways import _bases_for_modes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d067_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, dict[str, str], dict[str, str]]:
    project_root = tmp_path / "project"
    registered_sources = {}
    for relative in D067_INHERITED_SOURCE_PATHS:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
        registered_sources[relative] = _sha256(path)
    output_hashes = {}
    summary_root = tmp_path / "d067"
    for case_id in CASE_IDS:
        path = summary_root / "bundles" / f"{case_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(case_id.encode("utf-8"))
        output_hashes[f"bundles/{case_id}.npz"] = _sha256(path)
    summary_path = summary_root / "summary.json"
    payload = {
        "schema": "synthetic_d067",
        "status": "complete",
        "checkpoint_sha256": "a" * 64,
        "normalization_digest": "normalization",
        "source_hashes": registered_sources,
        "output_hashes": output_hashes,
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pathway_evaluator, "ROOT", project_root)
    args = SimpleNamespace(
        d067_summary=summary_path,
        expected_d067_summary_sha256=_sha256(summary_path),
        expected_checkpoint_sha256="a" * 64,
        d067_source_contract=D067_INHERITED_SOURCE_CONTRACT,
    )
    return args, payload, registered_sources


def test_d070_binds_all_exact_registered_d067_inherited_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, registered = _d067_fixture(tmp_path, monkeypatch)
    summary, binding = _verify_d067(
        args,
        {"normalization_digest": "normalization"},
    )
    assert summary["status"] == "complete"
    assert binding["active_source_sha256"] == registered
    assert binding["registered_source_sha256"] == registered
    assert binding["source_mismatches"] == {}
    assert binding["exact_replay_claimed"]
    assert D067_INHERITED_SOURCE_CONTRACT == ("exact_registered_d067_inherited_sources")


def test_d070_rejects_one_d067_inherited_source_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _d067_fixture(tmp_path, monkeypatch)
    (pathway_evaluator.ROOT / D067_INHERITED_SOURCE_PATHS[0]).write_text(
        "changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        _verify_d067(args, {"normalization_digest": "normalization"})


def test_d070_historical_compatibility_records_source_mismatch_without_exact_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _d067_fixture(tmp_path, monkeypatch)
    args.d067_source_contract = HISTORICAL_SOURCE_CONTRACT
    changed = pathway_evaluator.ROOT / D067_INHERITED_SOURCE_PATHS[0]
    changed.write_text("changed\n", encoding="utf-8")
    _, binding = _verify_d067(args, {"normalization_digest": "normalization"})
    assert D067_INHERITED_SOURCE_PATHS[0] in binding["source_mismatches"]
    assert not binding["exact_replay_claimed"]
    assert binding["source_contract"] == HISTORICAL_SOURCE_CONTRACT


def test_d070c_records_source_mismatch_without_cross_version_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _d067_fixture(tmp_path, monkeypatch)
    args.d067_source_contract = CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT
    changed = pathway_evaluator.ROOT / D067_INHERITED_SOURCE_PATHS[0]
    changed.write_text("changed\n", encoding="utf-8")
    _, binding = _verify_d067(args, {"normalization_digest": "normalization"})
    assert D067_INHERITED_SOURCE_PATHS[0] in binding["source_mismatches"]
    assert not binding["exact_replay_claimed"]
    assert not binding["cross_version_comparability_claimed"]
    assert binding["source_contract"] == CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_d070_rejects_changed_d067_inherited_source_inventory(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, payload, _ = _d067_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        payload["source_hashes"].pop(D067_INHERITED_SOURCE_PATHS[0])
    else:
        payload["source_hashes"]["unexpected.py"] = "f" * 64
    args.d067_summary.write_text(json.dumps(payload), encoding="utf-8")
    args.expected_d067_summary_sha256 = _sha256(args.d067_summary)
    with pytest.raises(ValueError, match="source inventory mismatch"):
        _verify_d067(args, {"normalization_digest": "normalization"})


def _compatibility_row() -> dict[str, float]:
    return {
        "case_id": "case",
        "pair": "125x50->250x100",
        "call": 1,
        "coarse_state_max_abs_physical": 1.0e-6,
        "restricted_fine_state_max_abs_physical": 2.0e-6,
        "coarse_state_relative_l2": 1.0e-8,
        "restricted_fine_state_relative_l2": 2.0e-8,
        "coarse_increment_drift_residual_scaled_rms": 1.0e-5,
        "restricted_fine_increment_drift_residual_scaled_rms": 2.0e-5,
        "commutator_drift_residual_scaled_rms": 1.0e-5,
        "baseline_large_mesh_defect_residual_scaled_rms": 1.0e-2,
        "large_commutator_drift_residual_scaled_rms": 1.0e-5,
        "large_commutator_drift_to_baseline_ratio": 1.0e-3,
        "band_reconstruction_closure": 1.0e-8,
    }


def test_d070_historical_compatibility_requires_complete_finite_rows() -> None:
    row = _compatibility_row()
    key = ("case", "125x50->250x100", 1)
    checks = _historical_compatibility_checks([row], expected_keys={key})
    assert checks["key_matrix_complete"]
    assert checks["all_metrics_finite"]
    assert checks["complete_and_finite"]
    assert checks["state_absolute_pass"]
    assert checks["state_relative_pass"]
    assert checks["science_scale_pass"]
    assert checks["band_reconstruction_pass"]
    assert not _historical_compatibility_checks(
        [row],
        expected_keys={key, ("other", "125x50->250x100", 1)},
    )["complete_and_finite"]


def test_d070_historical_compatibility_rejects_duplicate_key() -> None:
    row = _compatibility_row()
    key = ("case", "125x50->250x100", 1)
    checks = _historical_compatibility_checks(
        [row, dict(row)],
        expected_keys={key},
    )
    assert not checks["key_matrix_complete"]
    assert checks["duplicate_keys"] == [list(key)]
    assert not checks["complete_and_finite"]


@pytest.mark.parametrize(
    "field",
    (
        "coarse_increment_drift_residual_scaled_rms",
        "restricted_fine_increment_drift_residual_scaled_rms",
        "commutator_drift_residual_scaled_rms",
        "baseline_large_mesh_defect_residual_scaled_rms",
        "large_commutator_drift_residual_scaled_rms",
    ),
)
def test_d070_historical_compatibility_rejects_nonfinite_drift_field(
    field: str,
) -> None:
    row = _compatibility_row()
    row[field] = float("nan")
    checks = _historical_compatibility_checks(
        [row],
        expected_keys={("case", "125x50->250x100", 1)},
    )
    assert not checks["all_metrics_finite"]
    assert not checks["complete_and_finite"]


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("coarse_state_max_abs_physical", 2.1e-5, "state_absolute_pass"),
        ("coarse_state_relative_l2", 1.1e-7, "state_relative_pass"),
        (
            "large_commutator_drift_to_baseline_ratio",
            0.011,
            "science_scale_pass",
        ),
        ("band_reconstruction_closure", 2.1e-5, "band_reconstruction_pass"),
    ),
)
def test_d070_historical_compatibility_rejects_each_guard(
    field: str,
    value: float,
    failed_check: str,
) -> None:
    row = _compatibility_row()
    row[field] = value
    checks = _historical_compatibility_checks(
        [row],
        expected_keys={("case", "125x50->250x100", 1)},
    )
    assert not checks[failed_check]


def test_d070_provenance_failure_precedes_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    built = False

    def fail_provenance(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise RuntimeError("provenance stop")

    def build_model(*args: object, **kwargs: object) -> tuple[object, object]:
        nonlocal built
        del args, kwargs
        built = True
        return object(), object()

    monkeypatch.setattr(
        pathway_evaluator, "select_device", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        pathway_evaluator,
        "PCNOEuler2DShardStore",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        pathway_evaluator,
        "load_resolution_checkpoint",
        lambda _: {},
    )
    monkeypatch.setattr(pathway_evaluator, "_verify_common_provenance", fail_provenance)
    monkeypatch.setattr(
        pathway_evaluator,
        "build_resolution_checkpoint_model",
        build_model,
    )
    args = SimpleNamespace(
        device="cpu",
        data_dir=tmp_path,
        checkpoint=tmp_path / "checkpoint.pt",
    )
    with pytest.raises(RuntimeError, match="provenance stop"):
        pathway_evaluator.run(args)
    assert not built


def test_d070_band_fields_close_on_constant_large_scale_field() -> None:
    field = np.ones((8, 4), dtype=np.float64)
    bands, closure = _band_fields(
        field,
        resolution=(4, 2),
        residual_scale=np.ones(4),
    )
    np.testing.assert_allclose(bands["large"], field, atol=1.0e-12)
    np.testing.assert_allclose(bands["transition"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(bands["local"], 0.0, atol=1.0e-12)
    assert max(closure.values()) < 1.0e-12


def _identity_context(state: np.ndarray, increment: float) -> dict:
    state = np.asarray(state, dtype=np.float64)
    predicted = state + float(increment)
    return {
        "state": state,
        "prediction": torch.as_tensor(predicted[None], dtype=torch.float32),
        "increment": predicted - state,
        "normalized": torch.full(
            (1, state.shape[0], state.shape[1]),
            float(increment),
            dtype=torch.float32,
        ),
    }


def _rounded_identity_context(state: np.ndarray, head: float) -> dict:
    state = np.asarray(state, dtype=np.float64)
    current = torch.as_tensor(state, dtype=torch.float32)
    normalized = torch.full_like(current, float(head)).unsqueeze(0)
    prediction = (current + normalized[0]).unsqueeze(0)
    predicted = prediction[0].detach().cpu().numpy().astype(np.float64)
    return {
        "state": state,
        "prediction": prediction,
        "increment": predicted - state,
        "normalized": normalized,
    }


@pytest.mark.parametrize(
    ("mode", "coarse_offset", "paired_applicable"),
    (("teacher_forced", 0.0, True), ("free_rollout", 1.0, False)),
)
def test_d070c_generalized_commutator_identity_for_paired_and_unpaired_inputs(
    mode: str,
    coarse_offset: float,
    paired_applicable: bool,
) -> None:
    fine_state = np.zeros((16, 4), dtype=np.float64)
    coarse_state = np.full((4, 4), coarse_offset, dtype=np.float64)
    row = _commutator_row(
        case_id="case",
        mode=mode,
        call=1,
        coarse=(2, 2),
        fine=(4, 4),
        coarse_context=_identity_context(coarse_state, 0.5),
        fine_context=_identity_context(fine_state, 0.25),
        paired_coarse_context=_identity_context(
            np.zeros((4, 4), dtype=np.float64), 0.125
        ),
        residual_scale=np.ones(4),
    )
    assert row["generalized_closure_max_abs_physical"] == 0.0
    assert row["generalized_closure_residual_scaled_rms"] == 0.0
    assert row["mesh_state_closure_max_abs_physical"] == 0.0
    assert row["mesh_state_closure_residual_scaled_rms"] == 0.0
    assert row["paired_input_identity_applicable"] is paired_applicable
    if paired_applicable:
        assert row["paired_identity_closure_max_abs_physical"] == 0.0
    else:
        assert row["paired_identity_closure_max_abs_physical"] == coarse_offset


def test_d070c_generalized_identity_does_not_replace_residual_head_bridge() -> None:
    row = _commutator_row(
        case_id="case",
        mode="free_rollout",
        call=1,
        coarse=(2, 2),
        fine=(4, 4),
        coarse_context=_rounded_identity_context(np.full((4, 4), 100_000_000.25), 0.5),
        fine_context=_rounded_identity_context(np.zeros((16, 4)), 0.25),
        paired_coarse_context=_rounded_identity_context(np.zeros((4, 4)), 0.125),
        residual_scale=np.ones(4),
    )
    assert row["generalized_closure_max_abs_physical"] == 0.0
    assert row["increment_head_bridge_max_abs_physical"] == pytest.approx(0.75)
    assert row["increment_head_bridge_large_band_ratio"] > 0.01


def test_d070c_free_recurrence_identity_and_failure_signal() -> None:
    fine_state = np.zeros((8, 4), dtype=np.float64)
    coarse_state = np.ones((2, 4), dtype=np.float64)
    coarse_context = _identity_context(coarse_state, 0.5)
    fine_context = _identity_context(fine_state, 0.25)
    row = _recurrence_row(
        case_id="case",
        call=1,
        coarse=(2, 1),
        fine=(4, 2),
        coarse_state=coarse_state,
        fine_state=fine_state,
        coarse_context=coarse_context,
        fine_context=fine_context,
        residual_scale=np.ones(4),
    )
    assert row["closure_max_abs_physical"] == 0.0
    broken_context = dict(coarse_context)
    broken_context["increment"] = coarse_context["increment"] + 0.125
    broken = _recurrence_row(
        case_id="case",
        call=1,
        coarse=(2, 1),
        fine=(4, 2),
        coarse_state=coarse_state,
        fine_state=fine_state,
        coarse_context=broken_context,
        fine_context=fine_context,
        residual_scale=np.ones(4),
    )
    assert broken["closure_max_abs_physical"] == pytest.approx(0.125)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "extra", "nonfinite"))
def test_d070c_inventory_is_fail_closed(mutation: str) -> None:
    row = {"kind": "arm", "layer": 0, "metric": 1.0}
    expected = {("arm", 0), ("arm", 1)}
    rows = [dict(row), {"kind": "arm", "layer": 1, "metric": 1.0}]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(row))
    elif mutation == "extra":
        rows.append({"kind": "arm", "layer": 2, "metric": 1.0})
    else:
        rows[0]["metric"] = float("nan")
    checks = _inventory_checks(
        rows,
        key_fields=("kind", "layer"),
        expected_keys=expected,
        finite_fields=("metric",),
    )
    assert not checks["complete_and_finite"]


@pytest.mark.parametrize(
    ("numerator", "denominator", "absolute_limit", "relative_limit", "passed"),
    (
        (1.0e-6, 1.0, 2.0e-5, 2.0e-5, True),
        (3.0e-5, 1.0, 2.0e-5, 2.0e-5, False),
        (1.0e-6, 1.0e-2, 2.0e-5, 2.0e-5, False),
    ),
)
def test_d070c_typed_closure_applies_absolute_and_relative_limits(
    numerator: float,
    denominator: float,
    absolute_limit: float,
    relative_limit: float,
    passed: bool,
) -> None:
    row = _typed_closure_row(
        case_id="case",
        mode="teacher_forced",
        pair="pair",
        call=1,
        layer=0,
        closure_kind="latent_fourier",
        numerator=numerator,
        denominator=denominator,
        units="normalized_hidden_rms",
        absolute_tolerance=absolute_limit,
        relative_tolerance=relative_limit,
    )
    assert row["passed"] is passed


def test_d070c_state_digest_detects_mutation() -> None:
    model = torch.nn.Linear(3, 2)
    model.register_buffer("scalar_state", torch.tensor(1.0))
    before = _model_state_sha256(model)
    assert _model_state_sha256(model) == before
    with torch.no_grad():
        model.weight[0, 0] += 1.0
    assert _model_state_sha256(model) != before


def test_d070c_contract_ignores_d067_compatibility_values() -> None:
    d070c = {
        name: True
        for name in (
            "inventories_pass",
            "selector_denominators_pass",
            "generalized_commutator_identity_pass",
            "mesh_state_identity_pass",
            "residual_head_to_increment_bridge_pass",
            "teacher_input_restriction_floor_pass",
            "paired_input_commutator_identity_pass",
            "free_recurrence_identity_pass",
            "trace_replay_pass",
            "typed_closures_pass",
            "reference_binding_pass",
            "model_state_immutable_pass",
        )
    }
    checks = {
        "d067_source_compatibility_pass": False,
        "pathway_closure_pass": False,
        "all_baseline_outputs_admissible": True,
        "hook_equivalence_pass": True,
        "historical_output_compatibility": {"science_scale_pass": False},
    }
    assert _contract_checks_pass(CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT, checks, d070c)
    for name in d070c:
        broken = dict(d070c)
        broken[name] = False
        assert not _contract_checks_pass(
            CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT, checks, broken
        ), name
    for name in ("all_baseline_outputs_admissible", "hook_equivalence_pass"):
        broken_checks = dict(checks)
        broken_checks[name] = False
        assert not _contract_checks_pass(
            CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT, broken_checks, d070c
        ), name
    checks["d067_source_compatibility_pass"] = True
    checks["historical_output_compatibility"] = {"science_scale_pass": True}
    assert _contract_checks_pass(CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT, checks, d070c)


def _d070c_synthetic_rows() -> dict[str, list[dict]]:
    case_id = CASE_IDS[0]
    pairs = ("125x50->250x100", "250x100->500x200")
    contexts = [
        (case_id, mode, pair, 1)
        for mode in ("teacher_forced", "free_rollout")
        for pair in pairs
    ]
    rows: dict[str, list[dict]] = {
        name: []
        for name in (
            "arm",
            "pathway",
            "closure",
            "direct",
            "commutator",
            "recurrence",
            "trace",
            "completion",
            "reference",
        )
    }
    for context_case, mode, pair, call in contexts:
        for arm in pathway_evaluator.SAME_HIDDEN_ARMS:
            for band in ("total", "large", "transition", "local"):
                rows["arm"].append(
                    {
                        "record_kind": "same_hidden_single_layer",
                        "case_id": context_case,
                        "mode": mode,
                        "pair": pair,
                        "call": call,
                        "layer": 0,
                        "arm": arm,
                        "band": band,
                        "baseline_defect_rms": 1.0,
                        "arm_defect_rms": 0.8,
                        "defect_norm_ratio": 0.8,
                        "defect_reduction": 0.2,
                        "response_rms": 0.2,
                    }
                )
        for branch in pathway_evaluator.BRANCH_NAMES:
            for band in ("total", "large", "transition", "local"):
                rows["arm"].append(
                    {
                        "record_kind": "native_all_layer_branch",
                        "case_id": context_case,
                        "mode": mode,
                        "pair": pair,
                        "call": call,
                        "layer": "all",
                        "arm": f"{branch}_all_layers",
                        "band": band,
                        "baseline_defect_rms": 1.0,
                        "arm_defect_rms": 0.8,
                        "defect_norm_ratio": 0.8,
                        "defect_reduction": 0.2,
                        "response_rms": 0.2,
                    }
                )
        for pathway, terms in pathway_evaluator.PATHWAY_TERMS.items():
            for term in terms:
                rows["pathway"].append(
                    {
                        "case_id": context_case,
                        "mode": mode,
                        "pair": pair,
                        "call": call,
                        "layer": 0,
                        "pathway": pathway,
                        "term": term,
                        "term_rms": 0.5,
                        "mesh_gap_rms": 1.0,
                        "term_to_mesh_ratio": 0.5,
                    }
                )
            rows["closure"].append(
                _typed_closure_row(
                    case_id=context_case,
                    mode=mode,
                    pair=pair,
                    call=call,
                    layer=0,
                    closure_kind=f"latent_{pathway}",
                    numerator=0.0,
                    denominator=1.0,
                    units="normalized_hidden_rms",
                    absolute_tolerance=2.0e-5,
                    relative_tolerance=2.0e-5,
                )
            )
        for closure_kind in ("dct_reconstruction", "dct_energy"):
            rows["closure"].append(
                _typed_closure_row(
                    case_id=context_case,
                    mode=mode,
                    pair=pair,
                    call=call,
                    layer="all",
                    closure_kind=closure_kind,
                    numerator=0.0,
                    denominator=1.0,
                    units="scaled",
                    absolute_tolerance=1.0e-10,
                    relative_tolerance=1.0e-10,
                )
            )
        for output_role in ("coarse", "restricted_fine"):
            rows["direct"].append(
                {
                    "case_id": context_case,
                    "mode": mode,
                    "pair": pair,
                    "call": call,
                    "output_role": output_role,
                    "output_residual_scaled_rms": 1.0,
                    "large_band_residual_scaled_rms": 0.5,
                }
            )
        rows["commutator"].append(
            {
                "case_id": context_case,
                "mode": mode,
                "pair": pair,
                "call": call,
                "input_gap_max_abs_physical": 0.0,
                "input_gap_residual_scaled_rms": 0.0,
                "output_gap_residual_scaled_rms": 1.0,
                "increment_gap_residual_scaled_rms": 1.0,
                "head_defect_residual_scaled_rms": 1.0,
                "increment_head_bridge_max_abs_physical": 0.0,
                "increment_head_bridge_residual_scaled_rms": 0.0,
                "increment_defect_large_band_residual_scaled_rms": 1.0,
                "increment_head_bridge_large_band_residual_scaled_rms": 0.0,
                "increment_head_bridge_large_band_ratio": 0.0,
                "generalized_closure_max_abs_physical": 0.0,
                "generalized_closure_residual_scaled_rms": 0.0,
                "paired_input_identity_applicable": mode == "teacher_forced",
                "paired_identity_closure_max_abs_physical": 0.0,
                "paired_identity_closure_residual_scaled_rms": 0.0,
                "delta_mesh_residual_scaled_rms": 0.5,
                "mesh_head_defect_residual_scaled_rms": 0.5,
                "mesh_head_bridge_max_abs_physical": 0.0,
                "mesh_head_bridge_residual_scaled_rms": 0.0,
                "delta_mesh_large_band_residual_scaled_rms": 0.5,
                "mesh_head_bridge_large_band_residual_scaled_rms": 0.0,
                "mesh_head_bridge_large_band_ratio": 0.0,
                "delta_state_residual_scaled_rms": 0.5,
                "mesh_state_closure_max_abs_physical": 0.0,
                "mesh_state_closure_residual_scaled_rms": 0.0,
            }
        )
        trace_specs = [
            ("backbone_direct_coarse", "all", "none"),
            ("backbone_direct_restricted_fine", "all", "none"),
            ("same_hidden_restricted_fine", 0, "none"),
        ] + [
            (comparison, "all", branch)
            for comparison in (
                "no_replacement_coarse",
                "no_replacement_restricted_fine",
            )
            for branch in pathway_evaluator.BRANCH_NAMES
        ]
        for comparison, layer, branch in trace_specs:
            rows["trace"].append(
                {
                    "case_id": context_case,
                    "mode": mode,
                    "pair": pair,
                    "call": call,
                    "comparison": comparison,
                    "layer": layer,
                    "branch": branch,
                    "max_abs_physical": 0.0,
                    "error_residual_scaled_rms": 0.0,
                    "reference_residual_scaled_rms": 1.0,
                    "relative_residual_scaled_rms": 0.0,
                    "large_band_error_residual_scaled_rms": 0.0,
                    "baseline_large_band_residual_scaled_rms": 1.0,
                    "large_band_error_to_baseline_ratio": 0.0,
                    "passed": True,
                }
            )
    for pair in pairs:
        rows["recurrence"].append(
            {
                "case_id": case_id,
                "pair": pair,
                "call": 1,
                "error_before_residual_scaled_rms": 0.0,
                "delta_residual_scaled_rms": 1.0,
                "error_after_residual_scaled_rms": 1.0,
                "closure_max_abs_physical": 0.0,
                "closure_residual_scaled_rms": 0.0,
            }
        )
    rows["completion"].append(
        {
            "case_id": case_id,
            "calls": 1,
            "all_baseline_outputs_admissible": True,
            "seconds": 1.0,
        }
    )
    rows["reference"].append(
        {"case_id": case_id, "restriction_crosscheck_max_abs": 0.0}
    )
    return rows


def _synthetic_d070c_checks(rows: dict[str, list[dict]]) -> dict:
    return _d070c_contract_checks(
        active_case_ids=(CASE_IDS[0],),
        active_calls=1,
        diagnostic_calls=(1,),
        layer_count=1,
        arm_rows=rows["arm"],
        pathway_rows=rows["pathway"],
        closure_rows=rows["closure"],
        direct_output_rows=rows["direct"],
        commutator_rows=rows["commutator"],
        recurrence_rows=rows["recurrence"],
        trace_rows=rows["trace"],
        completion_rows=rows["completion"],
        reference_checks=rows["reference"],
        model_state_before="same",
        model_state_after="same",
    )


def test_d070c_complete_synthetic_contract_and_denominator_gate() -> None:
    rows = _d070c_synthetic_rows()
    checks = _synthetic_d070c_checks(rows)
    assert checks["inventories_pass"]
    assert checks["selector_denominators_pass"]
    assert all(
        inventory["complete_and_finite"] for inventory in checks["inventories"].values()
    )
    for row in rows["arm"]:
        if row["record_kind"] == "same_hidden_single_layer":
            row["baseline_defect_rms"] = 1.0e-9
    assert not _synthetic_d070c_checks(rows)["selector_denominators_pass"]


def test_d070c_entirely_absent_arm_or_layer_fails_inventory() -> None:
    rows = _d070c_synthetic_rows()
    rows["arm"] = [
        row
        for row in rows["arm"]
        if row["arm"] != pathway_evaluator.SAME_HIDDEN_ARMS[0]
    ]
    assert not _synthetic_d070c_checks(rows)["inventories"]["arm_metrics"][
        "matrix_complete"
    ]
    rows = _d070c_synthetic_rows()
    checks = _d070c_contract_checks(
        active_case_ids=(CASE_IDS[0],),
        active_calls=1,
        diagnostic_calls=(1,),
        layer_count=2,
        arm_rows=rows["arm"],
        pathway_rows=rows["pathway"],
        closure_rows=rows["closure"],
        direct_output_rows=rows["direct"],
        commutator_rows=rows["commutator"],
        recurrence_rows=rows["recurrence"],
        trace_rows=rows["trace"],
        completion_rows=rows["completion"],
        reference_checks=rows["reference"],
        model_state_before="same",
        model_state_after="same",
    )
    assert not checks["inventories"]["arm_metrics"]["matrix_complete"]
    assert not checks["inventories"]["pathway_terms"]["matrix_complete"]
    assert not checks["inventories"]["trace_replay"]["matrix_complete"]


def test_d070c_smoke_rejects_extra_reference_row() -> None:
    rows = _d070c_synthetic_rows()
    rows["reference"].append(
        {"case_id": CASE_IDS[1], "restriction_crosscheck_max_abs": 0.0}
    )
    reference_inventory = _synthetic_d070c_checks(rows)["inventories"][
        "reference_checks"
    ]
    assert not reference_inventory["matrix_complete"]
    assert reference_inventory["extra_keys"] == [[CASE_IDS[1]]]


@pytest.mark.parametrize(
    ("family", "field"),
    (
        ("arm", "baseline_defect_rms"),
        ("pathway", "term_rms"),
        ("closure", "closure_numerator"),
        ("direct", "output_residual_scaled_rms"),
        ("commutator", "generalized_closure_max_abs_physical"),
        ("recurrence", "closure_max_abs_physical"),
        ("trace", "max_abs_physical"),
        ("completion", "seconds"),
        ("reference", "restriction_crosscheck_max_abs"),
    ),
)
def test_d070c_each_row_family_rejects_nonfinite_metric(
    family: str, field: str
) -> None:
    rows = _d070c_synthetic_rows()
    rows[family][0][field] = float("nan")
    assert not _synthetic_d070c_checks(rows)["inventories_pass"]


def test_d070_manual_fourier_bases_follow_active_geometry_dtype_and_device() -> None:
    nodes = torch.zeros((1, 4, 2), dtype=torch.float64)
    weights = torch.ones((1, 4, 1), dtype=torch.float64)
    placeholder = torch.empty(0)
    spectral = type(
        "SpectralStub",
        (),
        {"modes": torch.ones((3, 2, 1), dtype=torch.float32)},
    )()
    bases = _bases_for_modes(
        spectral,
        (placeholder, nodes, weights, placeholder, placeholder),
    )
    assert all(value.dtype == nodes.dtype for value in bases)
    assert all(value.device == nodes.device for value in bases)


def _selector_rows(arms: tuple[str, ...] = ("fourier_analysis",)) -> list[dict]:
    pairs = ("125x50->250x100", "250x100->500x200")
    rows = []
    for arm in arms:
        for case_id in CASE_IDS:
            for pair in pairs:
                for mode in ("teacher_forced", "free_rollout"):
                    for call in (1, 5, 15, 30):
                        for band, ratio in (
                            ("large", 0.7),
                            ("total", 0.9),
                            ("transition", 1.0),
                            ("local", 1.0),
                        ):
                            rows.append(
                                {
                                    "record_kind": "same_hidden_single_layer",
                                    "case_id": case_id,
                                    "pair": pair,
                                    "mode": mode,
                                    "call": call,
                                    "arm": arm,
                                    "layer": 1,
                                    "band": band,
                                    "defect_norm_ratio": ratio,
                                    "defect_reduction": 1.0 - ratio,
                                }
                            )
    return rows


def test_d070_selector_requires_complete_mode_call_matrix() -> None:
    pairs = ("125x50->250x100", "250x100->500x200")
    rows = _selector_rows()
    selected = _mechanism_selection(rows, pairs)
    assert selected["decision"] == "selected"
    assert selected["selected"]["arm"] == "fourier_analysis"
    assert all(
        mode["matrix_complete"]
        for pair in selected["selected"]["pairs"]
        for mode in pair["modes"]
    )
    assert _mechanism_selection(rows[:-1], pairs)["decision"] == (
        "composite_or_unresolved"
    )
    for row in rows:
        if (
            row["pair"] == pairs[1]
            and row["case_id"] in CASE_IDS[:3]
            and row["mode"] == "free_rollout"
            and row["band"] == "large"
        ):
            row["defect_norm_ratio"] = 1.2
            row["defect_reduction"] = -0.2
    assert _mechanism_selection(rows, pairs)["decision"] == "composite_or_unresolved"


def test_d070_selector_excludes_native_rows_and_reports_multiple_support() -> None:
    pairs = ("125x50->250x100", "250x100->500x200")
    rows = _selector_rows()
    native = dict(rows[0])
    native["record_kind"] = "native_all_layer_branch"
    native["arm"] = "native_fine_fourier_all_layers"
    rows.append(native)
    assert _mechanism_selection(rows, pairs)["decision"] == "selected"

    multiple = _mechanism_selection(
        _selector_rows(("fourier_analysis", "fourier_quadrature")), pairs
    )
    assert multiple["decision"] == "multiple_supported"
    assert {row["arm"] for row in multiple["qualified"]} == {
        "fourier_analysis",
        "fourier_quadrature",
    }


def test_d070_parser_freezes_training_resolution(tmp_path: Path) -> None:
    argv = [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--normalization-json",
        str(tmp_path / "normalization.json"),
        "--split-json",
        str(tmp_path / "split.json"),
        "--data-dir",
        str(tmp_path / "data"),
        "--family-root",
        str(tmp_path / "family"),
        "--multires-reference-root",
        str(tmp_path / "multires"),
        "--d067-summary",
        str(tmp_path / "d067.json"),
        "--d067-source-contract",
        HISTORICAL_SOURCE_CONTRACT,
        "--output-dir",
        str(tmp_path / "output"),
        "--expected-checkpoint-sha256",
        "b" * 64,
        "--expected-normalization-sha256",
        "c" * 64,
        "--expected-split-sha256",
        "d" * 64,
        "--expected-data-manifest-digest",
        "e" * 64,
        "--expected-family-manifest-sha256",
        "f" * 64,
        "--expected-d067-summary-sha256",
        "1" * 64,
        "--expected-source-base-git-head",
        "2" * 40,
        "--source-manifest",
        str(tmp_path / "source_manifest.json"),
        "--expected-source-manifest-sha256",
        "3" * 64,
        "--training-resolution",
        "125x50",
    ]
    with pytest.raises(ValueError, match="training resolution"):
        parse_args(argv)

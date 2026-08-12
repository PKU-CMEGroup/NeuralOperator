from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.time_dependent_no import train_realm_ignithit_ffno as parent
from scripts.time_dependent_no.train_realm_ignithit_domain_linked_ffno import (
    BASE_FLOOR,
    BEST_CHECKPOINT_SCHEMA,
    LAST_CHECKPOINT_SCHEMA,
    PARENT_NORMALIZER_ARRAYS_SHA256,
    PARENT_SOURCE_SHA256,
    RUN_ID,
    activated_contract,
    build_domain_link,
    build_model,
    domain_link_diagnostics,
    main,
    require_eligible_validation,
    source_manifest,
)
from scripts.time_dependent_no.train_realm_ignithit_domain_linked_ffno import (
    frozen_training_contract as d089_training_contract,
)
from utility.time_dependent_no.realm_domain_link import (
    BoxCoxDomainLink,
    DomainLinkedMap,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    grouped_next_state_mse,
    trainable_parameter_count,
)


def _normalizer_state(dtype: torch.dtype = torch.float64) -> dict[str, object]:
    return {
        "mean": torch.tensor(
            [-8.0, -4.0, 1.0, -7.5, 0.5, -2.0, -6.0, -1.0, 3.0, 4.0, 5.0, 6.0],
            dtype=dtype,
        ),
        "scale": torch.tensor(
            [2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 0.75, 1.25, 2.0, 2.0, 2.0, 2.0],
            dtype=dtype,
        ),
        "transformed_channels": tuple(range(8)),
        "box_cox_lambda": 0.1,
        "box_cox_epsilon": 1.0e-8,
        "std_correction": 1,
        "scale_stabilizer": 1.0e-10,
        "source_sha256": PARENT_NORMALIZER_ARRAYS_SHA256,
    }


def _small_config() -> RealmFFNOConfig:
    return RealmFFNOConfig(
        width=8,
        layers=2,
        modes_y=3,
        modes_x=3,
        feedforward_factor=2,
        feedforward_layers=2,
        head_width=8,
    )


def test_domain_link_guarantees_strict_base_and_preserves_other_channels() -> None:
    state = _normalizer_state()
    link = build_domain_link(state)
    raw = torch.linspace(-1.0e3, 1.0e3, 2 * 12 * 3 * 2, dtype=torch.float64).reshape(
        2, 12, 3, 2
    )

    linked = link(raw)
    base = link.inverse_box_cox_base(linked)

    assert linked.shape == raw.shape
    assert torch.isfinite(linked).all()
    assert torch.all(base > 0.0)
    assert float(base.min()) == pytest.approx(BASE_FLOOR)
    assert torch.equal(linked[:, 8:], raw[:, 8:])
    assert link.transformed_channels == tuple(range(8))


def test_domain_link_is_zero_centered_with_finite_positive_local_gradient() -> None:
    link = build_domain_link(_normalizer_state())
    raw = torch.zeros(1, 12, 2, 2, dtype=torch.float64, requires_grad=True)

    linked = link(raw)
    linked.sum().backward()

    assert torch.allclose(linked, torch.zeros_like(linked), atol=1.0e-14, rtol=0.0)
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad[:, :8] > 0.0)
    assert torch.all(raw.grad[:, :8] <= 1.0)
    assert torch.equal(raw.grad[:, 8:], torch.ones_like(raw.grad[:, 8:]))


def test_domain_link_has_finite_nonzero_gradients_near_and_below_domain() -> None:
    link = build_domain_link(_normalizer_state())
    raw = torch.full((1, 12, 2, 2), -40.0, dtype=torch.float64, requires_grad=True)
    raw.data[:, 8:] = 2.0
    target = torch.zeros_like(raw)

    prediction = link(raw)
    loss, _ = grouped_next_state_mse(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad[:, :8] != 0.0)


def test_domain_link_rejects_nonfinite_raw_output_instead_of_repairing_it() -> None:
    link = build_domain_link(_normalizer_state())
    for value in (torch.nan, torch.inf, -torch.inf):
        raw = torch.zeros(1, 12, 2, 2, dtype=torch.float64)
        raw[:, 0, 0, 0] = value
        with pytest.raises(RuntimeError, match="raw normalized output must be finite"):
            link(raw)


def test_domain_link_validates_statistics_and_contract_shape() -> None:
    state = _normalizer_state()
    with pytest.raises(ValueError, match="positive"):
        BoxCoxDomainLink(
            state["mean"],
            -state["scale"],
            transformed_channels=tuple(range(8)),
            box_cox_lambda=0.1,
            base_floor=BASE_FLOOR,
        )
    with pytest.raises(ValueError, match="unique"):
        BoxCoxDomainLink(
            state["mean"],
            state["scale"],
            transformed_channels=(0, 0),
            box_cox_lambda=0.1,
            base_floor=BASE_FLOOR,
        )
    link = build_domain_link(state)
    with pytest.raises(ValueError, match="channel count"):
        link(torch.zeros(1, 11, 2, 2, dtype=torch.float64))
    wrong_source = dict(state, source_sha256="a" * 64)
    with pytest.raises(ValueError, match="source digest"):
        build_domain_link(wrong_source)
    wrong_channels = dict(state, transformed_channels=tuple(range(7)))
    with pytest.raises(ValueError, match="transformed channels"):
        build_domain_link(wrong_channels)


def test_small_domain_linked_ffno_forward_backward_and_recurrence_are_finite() -> None:
    torch.manual_seed(19)
    state = _normalizer_state(dtype=torch.float32)
    model = DomainLinkedMap(
        RealmFFNO2d(_small_config()),
        build_domain_link(state),
    )
    coordinates = torch.randn(1, 2, 8, 8)
    current = torch.randn(2, 12, 8, 8)
    truth = torch.randn_like(current)

    for _ in range(3):
        raw = model.forward_raw(current, coordinates)
        current = model(current, coordinates)
        assert raw.shape == current.shape
        assert torch.isfinite(current).all()
        assert torch.all(model.output_link.inverse_box_cox_base(current) >= BASE_FLOOR)
    loss, _ = grouped_next_state_mse(current, truth)
    loss.backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_domain_link_adds_no_trainable_parameters() -> None:
    model = build_model(_normalizer_state(dtype=torch.float32))
    assert trainable_parameter_count(model) == 8_936_460
    assert len(tuple(model.output_link.parameters())) == 0
    diagnostics = domain_link_diagnostics(model)["domain_link"]
    assert diagnostics["transformed_channels"] == list(range(8))
    assert diagnostics["physical_species_floor_configured"] == 1.0e-8
    assert diagnostics["normalized_zero_mapping_max_abs_error"] <= 1.0e-6
    assert diagnostics["raw_nonfinite_policy"] == "raise_before_link"


def test_d089_contract_changes_only_registered_factor_and_binds_sources() -> None:
    direct_contract = parent.frozen_training_contract()
    amended = d089_training_contract()
    changed_parent_fields = {
        "canonical_payload_sha256",
        "checkpoint_schemas",
        "claim_kind",
        "run_id",
        "schema",
    }
    for key, value in direct_contract.items():
        if key not in changed_parent_fields:
            assert amended[key] == value
    assert amended["run_id"] == RUN_ID
    assert (
        amended["parent_config_digest"] == direct_contract["canonical_payload_sha256"]
    )
    assert amended["parent_source_sha256"] == PARENT_SOURCE_SHA256
    assert (
        amended["selection_requires_all_decoded_finite_admissible_and_bounded"] is True
    )
    assert amended["output_parameterization"]["trainable_parameters"] == 0
    source = source_manifest()
    source_paths = {entry["path"] for entry in source["files"]}
    assert "utility/time_dependent_no/realm_domain_link.py" in source_paths
    assert (
        "scripts/time_dependent_no/train_realm_ignithit_domain_linked_ffno.py"
        in source_paths
    )
    parent_row = next(
        row
        for row in source["files"]
        if row["path"].endswith("train_realm_ignithit_ffno.py")
    )
    assert parent_row["sha256"] == PARENT_SOURCE_SHA256


def test_activated_contract_is_distinct_and_restores_parent_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _normalizer_state(dtype=torch.float32)
    normalizer_result = (object(), object(), torch.ones(12), state)
    original_normalizer_loader = parent._normalizers_from_arrays

    def synthetic_normalizer_loader(path: Path):
        return normalizer_result

    monkeypatch.setattr(
        "scripts.time_dependent_no.train_realm_ignithit_domain_linked_ffno."
        "_PARENT_NORMALIZERS_FROM_ARRAYS",
        synthetic_normalizer_loader,
    )
    monkeypatch.setattr(parent, "_normalizers_from_arrays", synthetic_normalizer_loader)
    original = {
        "RUN_ID": parent.RUN_ID,
        "BEST_CHECKPOINT_SCHEMA": parent.BEST_CHECKPOINT_SCHEMA,
        "LAST_CHECKPOINT_SCHEMA": parent.LAST_CHECKPOINT_SCHEMA,
        "RealmFFNO2d": parent.RealmFFNO2d,
        "frozen_training_contract": parent.frozen_training_contract,
        "_source_manifest": parent._source_manifest,
        "_runtime_manifest": parent._runtime_manifest,
        "_normalizers_from_arrays": synthetic_normalizer_loader,
        "_run_validation": parent._run_validation,
        "_write_json_atomic": parent._write_json_atomic,
    }

    with (
        pytest.raises(RuntimeError, match="synthetic execution failure"),
        activated_contract(),
    ):
        assert parent.RUN_ID == RUN_ID
        assert parent.BEST_CHECKPOINT_SCHEMA == BEST_CHECKPOINT_SCHEMA
        assert parent.LAST_CHECKPOINT_SCHEMA == LAST_CHECKPOINT_SCHEMA
        with pytest.raises(RuntimeError, match="preceded normalizer preflight"):
            parent.RealmFFNO2d()
        assert (
            parent._normalizers_from_arrays(Path("synthetic.npz")) is normalizer_result
        )
        assert isinstance(parent.RealmFFNO2d(), DomainLinkedMap)
        assert parent.frozen_training_contract()["run_id"] == RUN_ID
        raise RuntimeError("synthetic execution failure")

    for name, value in original.items():
        assert getattr(parent, name) is value or getattr(parent, name) == value
    monkeypatch.setattr(parent, "_normalizers_from_arrays", original_normalizer_loader)


def test_validation_eligibility_is_fail_closed_before_selection() -> None:
    eligible = {
        "all_normalized_finite": True,
        "all_decoded_finite": True,
        "all_released_state_admissible": True,
        "all_bounded_10x_train_max": True,
        "realm_npe_mean": 1.25,
    }
    assert require_eligible_validation(eligible) == eligible
    for flag in tuple(key for key in eligible if key.startswith("all_")):
        failed = dict(eligible)
        failed[flag] = False
        with pytest.raises(RuntimeError, match="finite/admissible/bounded"):
            require_eligible_validation(failed)
    missing = dict(eligible)
    missing.pop("all_decoded_finite")
    with pytest.raises(ValueError, match="missing"):
        require_eligible_validation(missing)
    nonfinite = dict(eligible, realm_npe_mean=torch.inf)
    with pytest.raises(RuntimeError, match="selection metric"):
        require_eligible_validation(nonfinite)


def test_activated_contract_rejects_preexisting_parent_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parent, "RUN_ID", "unexpected")
    with (
        pytest.raises(RuntimeError, match="changed before D089 activation"),
        activated_contract(),
    ):
        pass


def test_domain_link_cli_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    rendered = capsys.readouterr().out
    assert "--manifest" in rendered
    assert "--data-root" in rendered
    assert "--normalizer-arrays" in rendered
    assert "--output-dir" in rendered
    assert "--stop-after-step" in rendered
    assert "--resume-checkpoint" in rendered

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from scripts.time_dependent_no.audit_realm_benchmark import main as audit_main
from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_FIELDS,
    IGNITHIT_GROUPS,
    MANIFEST_SCHEMA,
    ChannelGroups,
    ManifestEntry,
    RealmNormalizer,
    apply_parameterization,
    boundary_band_error,
    box_cox,
    canonical_json_sha256,
    canonical_manifest_bytes,
    cartesian_boundary_band_mask,
    cartesian_front_metrics,
    decoded_admissibility,
    decoded_boundedness,
    decoded_spatial_pearson,
    fit_train_magnitude_envelope,
    grouped_normalized_prediction_error,
    inverse_box_cox,
    manifest_sha256,
    parse_manifest_payload,
    physical_spectrum_2d,
    predict_one_call,
    predict_two_call_final,
    validate_ignithit_open_manifest,
    validate_open_manifest_contract,
)


def _entry(
    path: str = "data/train/trajectory.npy",
    *,
    size: int = 17,
    lfs: bool = True,
    marker: str = "a",
) -> ManifestEntry:
    return ManifestEntry(
        path=path,
        size=size,
        oid=marker * (64 if lfs else 40),
        lfs=lfs,
    )


def _ignithit_state(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    state = torch.zeros((len(IGNITHIT_FIELDS), 3, 4), dtype=dtype)
    state[8] = 900.0
    state[9] = 1.2
    state[10] = 4.0
    state[11] = -0.5
    return state


def test_box_cox_float64_round_trip_and_zero_clamp() -> None:
    values = torch.tensor([0.0, 1.0e-20, 0.2, 1.0, 3.0], dtype=torch.float64)
    transformed = box_cox(values, lam=0.1, epsilon=1.0e-40)
    recovered = inverse_box_cox(
        transformed,
        lam=0.1,
        epsilon=1.0e-40,
    )

    expected = values.clamp_min(1.0e-40)
    assert torch.allclose(recovered, expected, rtol=2.0e-12, atol=1.0e-42)
    assert recovered[0].item() > 0.0


def test_log_limit_round_trip() -> None:
    values = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)
    encoded = box_cox(values, lam=0.0, epsilon=1.0e-8)
    assert torch.allclose(
        inverse_box_cox(encoded, lam=0.0, epsilon=1.0e-8),
        values,
        rtol=1.0e-14,
        atol=0.0,
    )


def test_inverse_box_cox_domain_is_fail_closed_or_reason_coded_nan() -> None:
    transformed = torch.tensor([-10.1, -10.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="outside the real domain"):
        inverse_box_cox(transformed, lam=0.1, epsilon=1.0e-8)

    recovered = inverse_box_cox(
        transformed,
        lam=0.1,
        epsilon=1.0e-8,
        domain_policy="nan",
    )
    assert torch.isnan(recovered[0])
    assert recovered[1].item() == 0.0
    assert recovered[2].item() == 1.0


def test_normalizer_uses_every_nonchannel_train_axis_and_zero_variance_policy() -> None:
    train = torch.tensor(
        [
            [[[[-1.0, 1.0]], [[5.0, 5.0]], [[0.0, 2.0]]]],
            [[[[3.0, 5.0]], [[5.0, 5.0]], [[4.0, 6.0]]]],
        ],
        dtype=torch.float64,
    )
    normalizer = RealmNormalizer.fit_train_only(
        train,
        transformed_channels=(),
        channel_axis=2,
        box_cox_epsilon=1.0e-8,
        std_correction=1,
    )

    flattened = train.movedim(2, 0).reshape(3, -1)
    expected_std = flattened.std(dim=1, correction=1)
    assert torch.equal(normalizer.mean, flattened.mean(dim=1))
    assert normalizer.scale[0].item() == pytest.approx(expected_std[0].item() + 1.0e-10)
    assert normalizer.scale[1].item() == 1.0
    assert normalizer.scale[2].item() == pytest.approx(expected_std[2].item() + 1.0e-10)
    assert torch.allclose(
        normalizer.decode(normalizer.encode(train)), train, rtol=0.0, atol=1.0e-14
    )


def test_normalizer_statistics_do_not_depend_on_validation_values() -> None:
    train = torch.tensor([[[[[1.0, 3.0]], [[7.0, 7.0]]]]], dtype=torch.float64)
    validation = torch.full_like(train, 1.0e9)
    normalizer = RealmNormalizer.fit_train_only(
        train,
        transformed_channels=(),
        channel_axis=2,
        box_cox_epsilon=1.0e-8,
    )

    assert normalizer.mean.tolist() == pytest.approx([2.0, 7.0])
    assert not torch.allclose(
        normalizer.encode(validation), torch.zeros_like(validation)
    )


def test_normalizer_applies_box_cox_only_to_registered_channels() -> None:
    train = torch.tensor([[[[[0.0, 1.0]], [[0.0, 1.0]]]]], dtype=torch.float64)
    normalizer = RealmNormalizer.fit_train_only(
        train,
        transformed_channels=(0,),
        channel_axis=2,
        box_cox_epsilon=1.0e-8,
    )

    recovered = normalizer.decode(normalizer.encode(train))
    assert recovered[0, 0, 0, 0, 0].item() == pytest.approx(1.0e-8)
    assert recovered[0, 0, 1, 0, 0].item() == 0.0


def test_ignithit_channel_groups_are_exact_and_pressure_is_absent() -> None:
    assert len(IGNITHIT_FIELDS) == 12
    assert IGNITHIT_GROUPS.slices(expected_channels=12) == {
        "chem": slice(0, 8),
        "T": slice(8, 9),
        "rho": slice(9, 10),
        "u": slice(10, 12),
        "p": slice(12, 12),
    }
    with pytest.raises(ValueError, match="does not match"):
        IGNITHIT_GROUPS.slices(expected_channels=13)
    with pytest.raises(ValueError, match="nonnegative"):
        ChannelGroups(-1, 1, 1, 2, 0)


def test_paper_average_and_released_source_sum_are_separately_exact() -> None:
    truth = torch.zeros((2, 2, 12, 2, 2), dtype=torch.float64)
    prediction = torch.zeros_like(truth)
    prediction[0, 0] = 1.0
    prediction[0, 1] = 2.0

    metrics = grouped_normalized_prediction_error(prediction, truth)

    assert set(metrics.grouped_per_call) == {"chem", "T", "rho", "u"}
    assert metrics.total_per_call.tolist() == [[4.0, 16.0], [0.0, 0.0]]
    assert metrics.per_case_mean.tolist() == [10.0, 0.0]
    assert metrics.per_case_sum.tolist() == [20.0, 0.0]
    assert metrics.realm_npe_mean == 5.0
    assert metrics.realm_npe_sum_source == 10.0


def test_normalized_error_is_case_first_and_propagates_prediction_nonfiniteness() -> (
    None
):
    truth = torch.zeros((2, 2, 12, 2, 2), dtype=torch.float64)
    prediction = torch.zeros_like(truth)
    prediction[0, 0, 0, 0, 0] = torch.nan

    metrics = grouped_normalized_prediction_error(prediction, truth)

    assert math.isinf(metrics.grouped_per_call["chem"][0, 0].item())
    assert math.isinf(metrics.realm_npe_mean)
    truth[1, 1, 2] = torch.nan
    with pytest.raises(ValueError, match="truth must be finite"):
        grouped_normalized_prediction_error(prediction, truth)


def test_decoded_pearson_reason_codes_constant_and_missing_channels() -> None:
    base = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    truth = torch.stack((base, torch.ones_like(base), base)).reshape(1, 1, 3, 2, 2)
    prediction = torch.stack((2.0 * base, base, torch.ones_like(base))).reshape(
        1, 1, 3, 2, 2
    )

    summary = decoded_spatial_pearson(prediction, truth)

    assert summary.values[0, 0, 0].item() == pytest.approx(1.0)
    assert summary.statuses[0][0] == (
        "ok",
        "constant_truth",
        "constant_prediction",
    )
    assert summary.population_case_first_mean == pytest.approx(1.0)


def test_decoded_pearson_reason_codes_nonfinite_inputs() -> None:
    truth = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2)
    prediction = truth.clone()
    truth[0, 0, 0, 0, 0] = torch.inf
    prediction[0, 0, 1, 0, 0] = torch.nan

    summary = decoded_spatial_pearson(prediction, truth)

    assert summary.statuses[0][0] == ("truth_nonfinite", "prediction_nonfinite")
    assert math.isnan(summary.population_case_first_mean)


def test_decoded_pearson_does_not_drop_an_unobservable_case() -> None:
    truth = torch.arange(8, dtype=torch.float64).reshape(2, 1, 1, 2, 2)
    prediction = truth.clone()
    prediction[1] = torch.nan

    summary = decoded_spatial_pearson(prediction, truth)

    assert summary.per_case_mean[0].item() == pytest.approx(1.0)
    assert torch.isnan(summary.per_case_mean[1])
    assert math.isnan(summary.population_case_first_mean)


def test_direct_and_zero_residual_recurrence_identities() -> None:
    current = torch.tensor([1.0, 2.0], dtype=torch.float64)
    static = torch.tensor([0.0], dtype=torch.float64)

    direct = predict_one_call(
        lambda state, coordinates: state * 0.0 + coordinates + 3.0,
        current,
        static,
        parameterization="direct",
    )
    residual = predict_one_call(
        lambda state, coordinates: torch.zeros_like(state) + 0.0 * coordinates,
        current,
        static,
        parameterization="residual",
    )

    assert torch.equal(direct, torch.tensor([3.0, 3.0], dtype=torch.float64))
    assert torch.equal(residual, current)
    with pytest.raises(ValueError, match="parameterization"):
        apply_parameterization(current, current, parameterization="unknown")  # type: ignore[arg-type]


def test_true_two_call_final_detaches_first_call_and_reuses_static_coordinates() -> (
    None
):
    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.grad_modes: list[bool] = []
            self.static_ids: list[int] = []
            self.input_requires_grad: list[bool] = []

        def forward(self, state: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
            self.grad_modes.append(torch.is_grad_enabled())
            self.static_ids.append(id(static))
            self.input_requires_grad.append(state.requires_grad)
            return self.weight * state + 0.0 * static

    model = ScalarModel()
    current = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    static = torch.tensor([5.0], dtype=torch.float64)

    final = predict_two_call_final(
        model,
        current,
        static,
        parameterization="direct",
    )
    final.sum().backward()

    assert final.item() == 12.0
    assert model.grad_modes == [False, True]
    assert model.input_requires_grad == [True, False]
    assert model.static_ids == [id(static), id(static)]
    assert model.weight.grad.item() == 6.0
    assert current.grad is None


def test_recurrence_apis_have_no_future_truth_argument() -> None:
    for function in (predict_one_call, predict_two_call_final):
        names = set(inspect.signature(function).parameters)
        assert names == {"model", "current", "static_coordinates", "parameterization"}
        assert all("truth" not in name and "target" not in name for name in names)


def test_manifest_serialization_is_sorted_utf8_and_deterministic() -> None:
    entries = (
        _entry("data/val/b.npy", marker="b"),
        _entry("README.md", size=3, lfs=False, marker="c"),
    )
    expected = (f"README.md\t3\t{'c' * 40}\ndata/val/b.npy\t17\t{'b' * 64}\n").encode()

    assert canonical_manifest_bytes(entries) == expected
    assert manifest_sha256(entries) == manifest_sha256(tuple(reversed(entries)))
    assert (
        manifest_sha256(entries) == __import__("hashlib").sha256(expected).hexdigest()
    )


@pytest.mark.parametrize(
    "path", ["/absolute", "C:/absolute", "a\\b", "a//b", "a/../b", "./a"]
)
def test_manifest_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX"):
        _entry(path)


def test_manifest_rejects_duplicate_paths_bad_sizes_and_bad_object_ids() -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="unique"):
        canonical_manifest_bytes((entry, entry))
    with pytest.raises(ValueError, match="nonnegative integer"):
        _entry(size=-1)
    with pytest.raises(ValueError, match="64-hex"):
        ManifestEntry(path="data/train/a", size=1, oid="a" * 40, lfs=True)
    with pytest.raises(ValueError, match="40-hex"):
        ManifestEntry(path="README.md", size=1, oid="a" * 64, lfs=False)


def test_parse_manifest_payload_has_an_exact_schema() -> None:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "repository": "owner/repo",
        "revision": "d" * 40,
        "entries": [{"path": "README.md", "size": 2, "oid": "e" * 40, "lfs": False}],
    }

    repository, revision, entries = parse_manifest_payload(payload)
    assert repository == "owner/repo"
    assert revision == "d" * 40
    assert entries == (_entry("README.md", size=2, lfs=False, marker="e"),)
    payload["extra"] = True
    with pytest.raises(ValueError, match="root must contain exactly"):
        parse_manifest_payload(payload)
    del payload["extra"]
    payload["entries"][0]["extra"] = True
    with pytest.raises(ValueError, match="path, size, oid, and lfs"):
        parse_manifest_payload(payload)


def test_open_manifest_accepts_only_exact_identity_bytes_and_digest() -> None:
    entries = (_entry("data/train/a.npy", size=11), _entry("data/val/b.npy", size=13))
    digest = manifest_sha256(entries)

    summary = validate_open_manifest_contract(
        repository="owner/repo",
        revision="f" * 40,
        entries=entries,
        expected_repository="owner/repo",
        expected_revision="f" * 40,
        expected_sha256=digest,
        expected_bytes=24,
    )
    assert summary["manifest_sha256"] == digest
    assert summary["total_bytes"] == 24
    assert summary["sealed_test_paths_present"] is False
    with pytest.raises(ValueError, match="byte total"):
        validate_open_manifest_contract(
            repository="owner/repo",
            revision="f" * 40,
            entries=entries,
            expected_repository="owner/repo",
            expected_revision="f" * 40,
            expected_sha256=digest,
            expected_bytes=25,
        )


def test_every_test_path_is_rejected_before_manifest_acceptance() -> None:
    entries = (_entry("data/test/a.npy"),)
    with pytest.raises(ValueError, match="sealed test paths"):
        validate_open_manifest_contract(
            repository="owner/repo",
            revision="f" * 40,
            entries=entries,
            expected_repository="owner/repo",
            expected_revision="f" * 40,
            expected_sha256=manifest_sha256(entries),
            expected_bytes=17,
        )


def test_unregistered_trajectory_split_is_rejected_before_digest_acceptance() -> None:
    entries = (_entry("data/dev/a.npy"),)
    with pytest.raises(ValueError, match="exactly train or val"):
        validate_open_manifest_contract(
            repository="owner/repo",
            revision="f" * 40,
            entries=entries,
            expected_repository="owner/repo",
            expected_revision="f" * 40,
            expected_sha256=manifest_sha256(entries),
            expected_bytes=17,
        )


def test_exact_ignithit_wrapper_rejects_an_unregistered_manifest() -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_ignithit_open_manifest(
            "realm-bench/realm-bench-IgnitHIT",
            "a0736b4d8c6c58a2688127e32addc30085e824c3",
            (_entry(),),
        )


def test_canonical_json_hash_binds_source_data_and_runtime_mappings() -> None:
    left = {"runtime": {"torch": "x", "tf32": False}, "seed": 7}
    right = {"seed": 7, "runtime": {"tf32": False, "torch": "x"}}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    with pytest.raises(ValueError):
        canonical_json_sha256({"invalid": math.nan})


def test_decoded_admissibility_is_limited_to_released_state_conditions() -> None:
    state = _ignithit_state()
    valid = decoded_admissibility(state)
    assert valid.finite and valid.admissible
    assert valid.min_pressure is None

    state[0, 0, 0] = -1.0
    state[8, 0, 1] = 0.0
    state[9, 0, 2] = 0.0
    invalid = decoded_admissibility(state)
    assert invalid.violations == (
        "negative_released_species",
        "nonpositive_temperature",
        "nonpositive_density",
    )
    state[10, 0, 0] = torch.nan
    assert decoded_admissibility(state).violations == ("native_nonfinite",)


def test_train_magnitude_envelope_has_inclusive_boundedness() -> None:
    train = torch.tensor(
        [[[[[0.0, 2.0], [-1.0, 1.0]], [[-4.0, 1.0], [2.0, 0.0]]]]],
        dtype=torch.float64,
    )
    envelope = fit_train_magnitude_envelope(train, quantile=1.0, channel_axis=2)
    state = torch.tensor([[[2.0]], [[-4.0]]], dtype=torch.float64)

    assert decoded_boundedness(
        state, envelope, expansion_factor=1.0, channel_axis=0
    ).bounded
    state[0, 0, 0] = 2.01
    assert not decoded_boundedness(
        state, envelope, expansion_factor=1.0, channel_axis=0
    ).bounded
    state[0, 0, 0] = torch.inf
    result = decoded_boundedness(state, envelope, expansion_factor=1.0, channel_axis=0)
    assert not result.finite and not result.bounded


def test_fixed_physical_boundary_band_separates_boundary_and_interior_error() -> None:
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    y = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    mask = cartesian_boundary_band_mask(x, y, band_width=0.2)
    truth = torch.zeros((1, 5, 5), dtype=torch.float64)
    prediction = torch.ones_like(truth)
    prediction[:, mask] = 2.0

    summary = boundary_band_error(prediction, truth, mask)
    assert mask.sum().item() == 16
    assert summary.boundary_mse == 4.0
    assert summary.interior_mse == 1.0
    assert summary.boundary_to_interior_ratio == 4.0


def test_cartesian_front_metrics_preserve_translation_invariants() -> None:
    left = torch.zeros((6, 8), dtype=torch.float64)
    left[:, 4:] = 1.0
    right = torch.zeros_like(left)
    right[:, 5:] = 1.0

    first = cartesian_front_metrics(
        left, dx=1.0, dy=1.0, low_threshold=0.25, high_threshold=0.75
    )
    second = cartesian_front_metrics(
        right, dx=1.0, dy=1.0, low_threshold=0.25, high_threshold=0.75
    )

    assert first.interface_length == second.interface_length == 6.0
    assert first.high_area == 24.0
    assert second.high_area == 18.0
    assert first.transition_area == first.thickness_proxy == 0.0
    assert first.centroid_x is not None and second.centroid_x is not None
    assert second.centroid_x - first.centroid_x == pytest.approx(1.0)
    assert first.jump_strength == pytest.approx(1.0)


def test_cartesian_front_reason_codes_a_missing_interface() -> None:
    result = cartesian_front_metrics(
        torch.ones((3, 4), dtype=torch.float64),
        dx=0.5,
        dy=0.25,
        low_threshold=0.25,
        high_threshold=0.75,
    )
    assert result.status == "no_interface"
    assert result.centroid_x is None
    assert result.jump_strength is None


def test_physical_spectrum_closes_parseval_and_localizes_a_sinusoid() -> None:
    x = torch.arange(8, dtype=torch.float64)
    field = torch.sin(2.0 * torch.pi * 2.0 * x / 8.0).repeat(8, 1)
    spectrum = physical_spectrum_2d(
        field,
        dx=1.0,
        dy=1.0,
        band_edges=(0.0, 0.2, 0.4, 0.8),
    )

    assert spectrum.spectral_energy == pytest.approx(
        spectrum.physical_energy, rel=2.0e-15
    )
    assert sum(spectrum.band_energy) == pytest.approx(
        spectrum.spectral_energy, rel=2.0e-15
    )
    assert spectrum.band_energy[1] == pytest.approx(spectrum.spectral_energy)


def test_demeaned_constant_spectrum_is_zero_and_bands_cover_grid() -> None:
    field = torch.ones((4, 4), dtype=torch.float64)
    spectrum = physical_spectrum_2d(
        field,
        dx=1.0,
        dy=1.0,
        band_edges=(0.0, 0.8),
        demean=True,
    )
    assert spectrum.physical_energy == spectrum.spectral_energy == 0.0
    with pytest.raises(ValueError, match="maximum grid frequency"):
        physical_spectrum_2d(
            field,
            dx=1.0,
            dy=1.0,
            band_edges=(0.0, 0.5),
        )


def test_manifest_audit_cli_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        audit_main(["--help"])
    assert exc_info.value.code == 0
    assert "--manifest" in capsys.readouterr().out


def test_manifest_audit_cli_rejects_a_nonobject_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        audit_main(["--manifest", str(manifest)])

    assert exc_info.value.code == 2
    assert "manifest root must be a JSON object" in capsys.readouterr().err


def test_manifest_payload_is_plain_json_serializable() -> None:
    entry = _entry("README.md", size=2, lfs=False)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "repository": "owner/repo",
        "revision": "b" * 40,
        "entries": [
            {"path": entry.path, "size": entry.size, "oid": entry.oid, "lfs": entry.lfs}
        ],
    }
    assert json.loads(json.dumps(payload)) == payload

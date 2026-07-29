from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import main
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_geometry,
    commutator_metrics,
    node_type_scaling_summary,
    node_types_for_protocol,
    parse_resolution,
    restrict_nested_state,
)
from utility.time_dependent_no.shock_vortex_fv import ShockVortexFVConfig

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_resolution_parsing_is_explicit() -> None:
    assert parse_resolution("125x50") == (125, 50)
    assert parse_resolution("12X8") == (12, 8)
    with pytest.raises(ValueError, match="NXxNY"):
        parse_resolution("125")
    with pytest.raises(ValueError, match="at least two"):
        parse_resolution("1x8")


def test_dynamic_node_types_have_expected_counts_and_volume_scaling() -> None:
    base = ShockVortexFVConfig()
    config, geometry = build_resolution_geometry(base, (10, 8))
    physical = node_types_for_protocol(
        geometry,
        config,
        "physical",
        training_resolution=(20, 16),
    )
    summary = node_type_scaling_summary(geometry, config, physical)

    assert summary["counts"] == {"0": 48, "1": 16, "2": 12, "3": 4}
    assert summary["tagged_count"] == 32
    assert summary["tagged_fraction"] == pytest.approx(0.4)
    assert summary["tagged_physical_volume"] == pytest.approx(0.8)
    assert summary["tagged_normalized_volume_mass"] == pytest.approx(0.4)
    assert summary["x_tagged_width_per_side"] == pytest.approx(0.2)
    assert summary["y_tagged_width_per_side"] == pytest.approx(0.125)
    assert summary["differential_input_support_hops_upper_bound"] == 3
    assert summary["differential_input_support_physical_upper_bound"] == (
        pytest.approx(0.6)
    )

    all_normal = node_types_for_protocol(
        geometry,
        config,
        "all_normal",
        training_resolution=(20, 16),
    )
    assert not bool(all_normal.any())


def test_training_band_keeps_width_when_target_grid_can_resolve_it() -> None:
    base = ShockVortexFVConfig()
    training_resolution = (20, 16)
    training_config, training_geometry = build_resolution_geometry(
        base, training_resolution
    )
    fine_config, fine_geometry = build_resolution_geometry(base, (40, 32))
    training_types = node_types_for_protocol(
        training_geometry,
        training_config,
        "training_band",
        training_resolution=training_resolution,
    )
    fine_types = node_types_for_protocol(
        fine_geometry,
        fine_config,
        "training_band",
        training_resolution=training_resolution,
    )
    training_summary = node_type_scaling_summary(
        training_geometry, training_config, training_types
    )
    fine_summary = node_type_scaling_summary(fine_geometry, fine_config, fine_types)

    assert training_summary["x_tagged_width_per_side"] == pytest.approx(0.1)
    assert fine_summary["x_tagged_width_per_side"] == pytest.approx(0.1)
    assert training_summary["y_tagged_width_per_side"] == pytest.approx(0.0625)
    assert fine_summary["y_tagged_width_per_side"] == pytest.approx(0.0625)
    assert fine_summary["tagged_physical_volume"] == pytest.approx(
        training_summary["tagged_physical_volume"]
    )
    assert fine_summary["counts"]["2"] > training_summary["counts"]["2"]


def test_conservative_restriction_and_update_commutator() -> None:
    fine_resolution = (4, 4)
    coarse_resolution = (2, 2)
    fine_current = np.arange(4 * 4 * 2, dtype=np.float64).reshape(16, 2)
    coarse_current = restrict_nested_state(
        fine_current,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    update = np.asarray([0.25, -0.5], dtype=np.float64)
    fine_prediction = fine_current + update
    coarse_prediction = coarse_current + update
    metrics = commutator_metrics(
        coarse_current=coarse_current,
        coarse_prediction=coarse_prediction,
        fine_current=fine_current,
        fine_prediction=fine_prediction,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        coarse_volumes=np.full((4, 1), 0.5),
        state_scale=np.ones(2),
        residual_scale=np.ones(2),
    )

    assert metrics["input_restriction_gap_scaled_rms"] == pytest.approx(0.0)
    assert metrics["prediction_commutator_scaled_rms"] == pytest.approx(0.0)
    assert metrics["update_commutator_scaled_rms"] == pytest.approx(0.0)
    assert metrics["update_commutator_relative_to_fine_update"] == pytest.approx(
        0.0
    )


def test_frozen_checkpoint_runs_on_two_node_and_edge_counts(tmp_path: Path) -> None:
    torch.manual_seed(7)
    normalization = Euler2DNormalization(
        state_mean=np.asarray([1.0, 1.0, 0.0, 3.0]),
        state_scale=np.asarray([0.1, 0.1, 0.1, 0.2]),
        residual_scale=np.asarray([1.0e-3] * 4),
        mach_mean=1.1,
        mach_scale=0.1,
        weight_provenance="validated_physical_cell_volume_normalized",
    )
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=1,
        domain_lengths=(2.0, 1.0),
        layers=(8, 8, 8),
        fc_dim=8,
        nmeasures=1,
        zero_initialize=False,
    )
    checkpoint = {
        "checkpoint_schema_version": 4,
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "normalization_digest": "synthetic-normalization",
        "data_manifest_digest": "synthetic-data",
        "data_contract": {
            "dataset": "synthetic_resolution_fixture",
            "step_stride": 1,
        },
        "config_digest": "synthetic-config",
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "inference_interventions": {"boundary_rebinding": False},
        "training_args": {"amp": "none"},
    }
    checkpoint_path = tmp_path / "synthetic.pt"
    torch.save(checkpoint, checkpoint_path)
    output_dir = tmp_path / "resolution"
    manifest = (
        ROOT
        / "artifacts"
        / "time_dependent_no"
        / "shock_vortex_family_frozen_20260720a"
        / "family_manifest.json"
    )

    assert (
        main(
            [
                "--family-manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                _sha256(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--case-ids",
                "sv_e00_y04",
                "--resolutions",
                "6x6",
                "12x12",
                "--training-resolution",
                "6x6",
                "--protocols",
                "physical",
                "all_normal",
                "training_band",
                "--expected-k-max",
                "1",
                "--repeat-forward",
                "2",
                "--device",
                "cpu",
                "--amp",
                "none",
            ]
        )
        == 0
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["scientific_scope"]["supports_pde_accuracy_claim"] is False
    assert summary["translation_contract"]["resolutions"] == ["6x6", "12x12"]
    assert len(summary["commutators"]) == 3
    assert len(summary["predictions"]) == 6
    assert summary["execution"]["maximum_repeat_abs_difference"] == pytest.approx(0.0)
    assert len(summary["execution"]["gradient_layer_softsign_prefactors"]) == 2
    assert {row["num_nodes"] for row in summary["predictions"]} == {36, 144}
    assert all(row["finite"] for row in summary["predictions"])

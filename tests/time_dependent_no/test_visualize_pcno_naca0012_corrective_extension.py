from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import (
    visualize_pcno_naca0012_corrective_extension as visualization,
)


def _synthetic_bundle() -> visualization.ReplayBundle:
    node_count = 4
    horizons = np.asarray(visualization.FULL_HORIZONS, dtype=np.int64)
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    reference = np.stack(
        [np.full(node_count, 1.0 + horizon / 1000.0) for horizon in horizons]
    ).astype(np.float32)
    predicted = np.broadcast_to(
        reference,
        (
            len(visualization.METHODS),
            len(visualization.SEEDS),
            len(horizons),
            node_count,
        ),
    ).copy()
    for method_offset in range(len(visualization.METHODS)):
        for seed_offset in range(len(visualization.SEEDS)):
            predicted[method_offset, seed_offset, 1:] += np.float32(
                0.001 * (method_offset + 1) * (seed_offset + 1)
            )
    static_offsets = np.asarray(visualization.STATIC_HORIZONS)
    exact_reference = reference[static_offsets].astype(np.float64)
    exact_prediction = predicted[:, :, static_offsets].copy()
    refiner_sampler = np.empty(
        (
            len(visualization.SEEDS),
            len(visualization.REFINER_SAMPLER_SEEDS),
            node_count,
        ),
        dtype=np.float32,
    )
    for seed_offset in range(len(visualization.SEEDS)):
        for sampler_offset in range(len(visualization.REFINER_SAMPLER_SEEDS)):
            refiner_sampler[seed_offset, sampler_offset] = exact_reference[
                -1
            ] + np.float32(0.002 * (seed_offset + 1) * (sampler_offset + 1))
    metrics = np.arange(
        len(visualization.METHODS)
        * len(visualization.SEEDS)
        * len(visualization.METRIC_SPECS),
        dtype=np.float64,
    ).reshape(
        len(visualization.METHODS),
        len(visualization.SEEDS),
        len(visualization.METRIC_SPECS),
    )
    metrics += 1.0
    metrics[0, 0, 0] = np.inf
    return visualization.ReplayBundle(
        horizons=horizons,
        frame_indices=visualization.ANCHOR + horizons,
        coordinates=coordinates,
        quads=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        airfoil_mask=np.ones(node_count, dtype=np.uint8),
        reference_density=reference,
        predicted_density=predicted,
        valid_mask=np.ones(predicted.shape[:-1], dtype=np.bool_),
        static_horizons=np.asarray(visualization.STATIC_HORIZONS, dtype=np.int64),
        exact_reference_density=exact_reference,
        exact_predicted_density=exact_prediction,
        exact_valid_mask=np.ones(exact_prediction.shape[:-1], dtype=np.bool_),
        refiner_sampler_density_h208=refiner_sampler,
        refiner_sampler_valid_h208=np.ones(refiner_sampler.shape[:-1], dtype=np.bool_),
        density_state_scale=2.0,
        per_seed_summary_metrics=metrics,
        manifest={},
    )


def _synthetic_arrays(bundle: visualization.ReplayBundle) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(visualization.REPLAY_SCHEMA),
        "qualitative_contract_sha256": np.asarray(
            visualization.QUALITATIVE_CONTRACT_SHA256
        ),
        "horizons": bundle.horizons,
        "frame_indices": bundle.frame_indices,
        "method_slugs": np.asarray([method.slug for method in visualization.METHODS]),
        "seeds": np.asarray(visualization.SEEDS, dtype=np.int64),
        "coordinates": bundle.coordinates,
        "quads": bundle.quads,
        "airfoil_mask": bundle.airfoil_mask,
        "reference_density": bundle.reference_density,
        "predicted_density": bundle.predicted_density,
        "valid_mask": bundle.valid_mask,
        "static_horizons": bundle.static_horizons,
        "exact_reference_density": bundle.exact_reference_density,
        "exact_predicted_density": bundle.exact_predicted_density,
        "exact_valid_mask": bundle.exact_valid_mask,
        "refiner_sampler_density_h208": bundle.refiner_sampler_density_h208,
        "refiner_sampler_valid_h208": bundle.refiner_sampler_valid_h208,
        "density_state_scale": np.asarray(bundle.density_state_scale, dtype=np.float64),
        "metric_names": np.asarray(
            [metric for metric, _ in visualization.METRIC_SPECS]
        ),
        "per_seed_summary_metrics": bundle.per_seed_summary_metrics,
    }


def _exact_snapshot_inputs(node_count: int) -> dict[str, np.ndarray]:
    base = np.arange(node_count * 5, dtype=np.float64).reshape(node_count, 5)
    return {
        f"reference_a{visualization.ANCHOR}_h{horizon}": base + horizon
        for horizon in visualization.STATIC_HORIZONS
    }


def _write_synthetic_replay(root: Path) -> None:
    root.mkdir()
    arrays = _synthetic_arrays(_synthetic_bundle())
    np.savez_compressed(root / visualization.REPLAY_FILE, **arrays)
    manifest: dict[str, object] = {
        "schema": visualization.REPLAY_SCHEMA,
        "status": "complete",
        "classification": "VISUALIZATION_ONLY_REPLAY",
        "experiment_id": visualization.EXPERIMENT_ID,
        "qualitative_contract": visualization.QUALITATIVE_CONTRACT,
        "qualitative_contract_sha256": visualization.QUALITATIVE_CONTRACT_SHA256,
        "population_role": "development",
        "anchor": visualization.ANCHOR,
        "seeds": list(visualization.SEEDS),
        "methods": visualization._method_records(),
        "first_nonfinite_horizon": {
            f"{method.slug}:{seed}": None
            for method in visualization.METHODS
            for seed in visualization.SEEDS
        },
        "online_solver_calls": False,
        "online_defect_trigger": False,
        "prospective_opened": False,
        "sealed_opened": False,
        "source": {
            "visualization_script": dict(visualization.VISUALIZATION_SOURCE_AT_IMPORT),
            "evaluator_source_set_sha256": visualization.evaluator.SOURCE_SET_SHA256,
        },
        "inputs": {
            "extension_contract_sha256": (
                visualization.evaluator.EXTENSION_CONTRACT_SHA256
            ),
            "successor_contract_sha256": (
                visualization.evaluator.trainer.SUCCESSOR_CONTRACT_SHA256
            ),
            "dataset_final_hash_manifest_sha256": (
                visualization.evaluator.trainer.DATASET_FINAL_SHA256
            ),
            "calibration_final_hash_manifest_sha256": (
                visualization.evaluator.trainer.SUCCESSOR_CALIBRATION_SHA256
            ),
            "relabel_pilot_final_hash_manifest_sha256": "a" * 64,
            "paired_bank_final_hash_manifest_sha256": "b" * 64,
            "evaluation_final_hash_manifest_sha256": "c" * 64,
            "extension_training_packets": [
                {
                    "arm": arm,
                    "seed": seed,
                    "final_hash_manifest_sha256": "d" * 64,
                    "checkpoint_sha256": "e" * 64,
                    "source_set_sha256": (
                        visualization.evaluator.trainer.SOURCE_SET_SHA256
                    ),
                }
                for arm in visualization.evaluator.LEARNED_ARMS
                for seed in visualization.SEEDS
            ],
            "inherited_training_packets": [
                {
                    "arm": arm,
                    "seed": seed,
                    "final_hash_manifest_sha256": "f" * 64,
                    "checkpoint_sha256": "0" * 64,
                    "source_set_sha256": "1" * 64,
                }
                for arm in ("CLEAN", "DETACHED_PUSHFORWARD")
                for seed in visualization.SEEDS
            ],
        },
        "replay_file": visualization.evaluator.parent._file_record(
            root / visualization.REPLAY_FILE, root
        ),
        "array_sha256": {
            key: visualization._array_sha256(value) for key, value in arrays.items()
        },
        "claim_boundary": visualization._claim_boundary(),
    }
    manifest["canonical_payload_sha256"] = visualization.evaluator._canonical_sha256(
        manifest
    )
    visualization.evaluator._write_json(
        root / visualization.REPLAY_MANIFEST_FILE, manifest
    )


def test_frozen_visualization_contract_and_primary_mapping() -> None:
    assert visualization.ANCHOR == 1234
    assert visualization.SEEDS == (17, 29, 43)
    assert visualization.STATIC_HORIZONS == (35, 104, 208)
    assert [
        (method.arm, method.deployment, method.state_stage, method.sampler_seed)
        for method in visualization.METHODS
    ] == [
        ("CLEAN", "raw", "raw", None),
        ("CLEAN_EMA", "ema", "raw", None),
        ("DETACHED_PUSHFORWARD", "raw", "raw", None),
        ("MP_PDE_PUSHFORWARD_M01", "online", "raw", None),
        ("CURRICULUM_EMA_PUSHFORWARD_K13", "ema", "raw", None),
        ("PAIRED_RECOVERY", "online", "raw", None),
        ("DYNAMICS_RELABEL", "online", "raw", None),
        ("PCNO_PDEREFINER_K3_VPRED", "ema", "raw", 101),
        ("PATH_PROJECTION", "path_projection", "corrected", None),
    ]
    assert visualization.QUALITATIVE_CONTRACT["population_role"] == "development"
    assert (
        visualization.QUALITATIVE_CONTRACT["animation_is_exact_evaluator_reproduction"]
        is False
    )
    assert visualization.METHOD_COLORS[2] == "#009E73"
    assert visualization.METHOD_COLORS[8] == "#CC79A7"
    assert visualization.SIGNED_ERROR_LINEAR_THRESHOLD == pytest.approx(1.0e-2)


@pytest.mark.parametrize("role", ["prospective", "sealed", "test"])
def test_replay_refuses_protected_roles_before_reading_paths(role: str) -> None:
    with pytest.raises(visualization.evaluator.parent.ProtectedPopulationError):
        visualization.replay(Namespace(population_role=role))


def test_snapshot_keys_match_common_evaluator_members() -> None:
    methods = {method.slug: method for method in visualization.METHODS}
    expected_h35 = {
        "clean": "clean_seed17_raw_a1234_h35",
        "clean_ema": "clean_ema_seed17_raw_a1234_h35_ema",
        "detached": "detached_pushforward_seed17_raw_a1234_h35",
        "mp_pde": "mp_pde_pushforward_m01_seed17_raw_a1234_h35_online",
        "curriculum": ("curriculum_ema_pushforward_k13_seed17_raw_a1234_h35_ema"),
        "recovery": "paired_recovery_seed17_raw_a1234_h35_online",
        "relabel": "dynamics_relabel_seed17_raw_a1234_h35_online",
        "pderefiner": ("pcno_pderefiner_k3_vpred_seed17_raw_a1234_h35_ema_sampler101"),
        "path_projection": "path_projection_seed17_corrected_a1234_h35",
    }
    assert {
        slug: visualization._snapshot_key(method, 17, 35)
        for slug, method in methods.items()
    } == expected_h35
    assert (
        visualization._snapshot_key(methods["pderefiner"], 43, 208, sampler_seed=307)
        == "pcno_pderefiner_k3_vpred_seed43_raw_a1234_h208_ema_sampler307"
    )


def test_exact_snapshot_loader_treats_all_nan_predictions_as_unavailable() -> None:
    node_count = 3
    snapshots = _exact_snapshot_inputs(node_count)
    sentinel = np.full((node_count, 5), np.nan, dtype=np.float32)
    finite = np.full((node_count, 5), 2.0, dtype=np.float32)
    snapshots[
        visualization._snapshot_key(
            visualization.METHODS[0], visualization.SEEDS[0], 35
        )
    ] = sentinel
    snapshots[
        visualization._snapshot_key(
            visualization.METHODS[1], visualization.SEEDS[0], 35
        )
    ] = finite
    refiner = next(
        method
        for method in visualization.METHODS
        if method.arm == visualization.evaluator.REFINER_ARM
    )
    snapshots[
        visualization._snapshot_key(
            refiner,
            visualization.SEEDS[0],
            208,
            sampler_seed=visualization.REFINER_SAMPLER_SEEDS[1],
        )
    ] = sentinel.copy()
    snapshots[
        visualization._snapshot_key(
            refiner,
            visualization.SEEDS[0],
            208,
            sampler_seed=visualization.REFINER_SAMPLER_SEEDS[2],
        )
    ] = finite.copy()

    _, predictions, valid, sampler_density, sampler_valid = (
        visualization._load_exact_snapshots(snapshots, node_count)
    )

    assert not valid[0, 0, 0]
    assert np.all(np.isnan(predictions[0, 0, 0]))
    assert valid[1, 0, 0]
    np.testing.assert_array_equal(predictions[1, 0, 0], 2.0)
    assert not sampler_valid[0, 1]
    assert np.all(np.isnan(sampler_density[0, 1]))
    assert sampler_valid[0, 2]
    np.testing.assert_array_equal(sampler_density[0, 2], 2.0)


@pytest.mark.parametrize(
    ("location", "error_match"),
    [
        ("prediction", "exact evaluator prediction snapshot differs"),
        ("sampler", "exact PDE-Refiner sampler snapshot differs"),
    ],
)
@pytest.mark.parametrize("nonfinite_kind", ["mixed_nan", "inf", "nan_and_inf"])
def test_exact_snapshot_loader_rejects_non_sentinel_nonfinite_predictions(
    location: str, error_match: str, nonfinite_kind: str
) -> None:
    node_count = 3
    snapshots = _exact_snapshot_inputs(node_count)
    if nonfinite_kind == "nan_and_inf":
        candidate = np.full((node_count, 5), np.nan, dtype=np.float32)
        candidate[0, 0] = np.inf
    else:
        candidate = np.ones((node_count, 5), dtype=np.float32)
        candidate[0, 0] = np.nan if nonfinite_kind == "mixed_nan" else np.inf
    if location == "prediction":
        key = visualization._snapshot_key(
            visualization.METHODS[0], visualization.SEEDS[0], 35
        )
    else:
        refiner = next(
            method
            for method in visualization.METHODS
            if method.arm == visualization.evaluator.REFINER_ARM
        )
        key = visualization._snapshot_key(
            refiner,
            visualization.SEEDS[0],
            208,
            sampler_seed=visualization.REFINER_SAMPLER_SEEDS[1],
        )
    snapshots[key] = candidate

    with pytest.raises(ValueError, match=error_match):
        visualization._load_exact_snapshots(snapshots, node_count)


def test_summary_aggregation_is_median_eight_anchors_then_retains_seeds() -> None:
    rows = []
    for method_offset, method in enumerate(visualization.METHODS):
        for seed_offset, seed in enumerate(visualization.SEEDS):
            for anchor_offset, anchor in enumerate(
                visualization.evaluator.DEVELOPMENT_ANCHORS
            ):
                base = 100.0 * method_offset + 10.0 * seed_offset + anchor_offset
                rows.append(
                    {
                        "arm": method.arm,
                        "seed": str(seed),
                        "deployment": method.deployment,
                        "state_stage": method.state_stage,
                        "sampler_seed": (
                            ""
                            if method.sampler_seed is None
                            else str(method.sampler_seed)
                        ),
                        "anchor": str(anchor),
                        **{
                            metric: str(base + metric_offset)
                            for metric_offset, (metric, _) in enumerate(
                                visualization.METRIC_SPECS
                            )
                        },
                    }
                )
    values = visualization._aggregate_summary_rows(rows)
    assert values.shape == (9, 3, 3)
    assert values[2, 1, 0] == pytest.approx(213.5)
    assert values[8, 2, 2] == pytest.approx(825.5)


def test_path_projection_replay_feeds_back_the_corrected_state(monkeypatch) -> None:
    class Geometry:
        def expand(self, batch_size: int, device: torch.device) -> object:
            assert batch_size == 2
            assert device.type == "cpu"
            return object()

    class Transition(torch.nn.Module):
        def prepare_fourier_tensors(self, geometry_batch: object) -> None:
            assert geometry_batch is not None

    class Projector:
        def to(self, device: torch.device, dtype: torch.dtype):
            assert device.type == "cpu"
            assert dtype == torch.float32
            return self

        def project(self, raw: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(corrected_state=0.5 * raw)

    def corrected_step(
        model,
        previous,
        current,
        geometry_batch,
        normalization,
        corrector,
        *,
        fourier_tensors,
    ):
        del model, previous, geometry_batch, normalization, fourier_tensors
        raw = current + 2.0
        return SimpleNamespace(
            corrected_next_state=corrector(raw), recurrent_previous=current
        )

    monkeypatch.setattr(visualization, "corrected_recurrent_step", corrected_step)
    method = next(
        method for method in visualization.METHODS if method.arm == "PATH_PROJECTION"
    )
    density, valid, failed = visualization._rollout_selected(
        method=method,
        transition=Transition(),
        previous=torch.zeros((2, 4, 5), dtype=torch.float32),
        current=torch.ones((2, 4, 5), dtype=torch.float32),
        selected_offset=0,
        geometry=Geometry(),
        normalization=object(),
        projector=Projector(),
        device=torch.device("cpu"),
    )
    assert failed is None
    assert np.all(valid)
    assert density[1, 0] == pytest.approx(1.5)
    assert density[2, 0] == pytest.approx(1.75)


def test_load_replay_and_frozen_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = tmp_path / "replay"
    _write_synthetic_replay(root)
    bundle = visualization.load_replay(root)
    quads = visualization._visible_quads(bundle)
    exact_limits = visualization._color_limits(bundle, quads, exact=True)
    replay_limits = visualization._color_limits(bundle, quads, exact=False)
    assert exact_limits["signed_error"][0] == -exact_limits["signed_error"][1]
    assert replay_limits["signed_error"][0] == -replay_limits["signed_error"][1]
    assert exact_limits["signed_error"][1] == pytest.approx(0.009)
    assert replay_limits["signed_error"][1] == pytest.approx(0.0135, abs=1.0e-7)

    comparison = visualization._method_comparison_figure(
        bundle, horizon=35, quads=quads, limits=exact_limits
    )
    sampler = visualization._refiner_sampler_figure(
        bundle, quads=quads, limits=exact_limits
    )
    summary = visualization._rollout_path_roughness_figure(bundle)
    animation, update = visualization._animation_figure(
        bundle, seed_offset=0, quads=quads, limits=replay_limits
    )
    update(35)
    assert comparison._suptitle is None
    assert sampler._suptitle is None
    assert summary._suptitle is None
    assert animation._suptitle is None
    assert any(
        "empirical train-path proximity proxy" in axis.get_xlabel()
        for axis in summary.axes
    )
    assert any(
        "Independent qualitative replay" in text.get_text() for text in animation.texts
    )
    comparison.savefig(tmp_path / "comparison.pdf")
    sampler.savefig(tmp_path / "sampler.pdf")
    summary.savefig(tmp_path / "summary.pdf")
    plt.close("all")


def test_load_replay_rejects_rehashed_method_order_tamper(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _write_synthetic_replay(root)
    replay_path = root / visualization.REPLAY_FILE
    with np.load(replay_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["method_slugs"] = arrays["method_slugs"][::-1]
    np.savez_compressed(replay_path, **arrays)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["replay_file"] = visualization.evaluator.parent._file_record(
        replay_path, root
    )
    manifest["array_sha256"] = {
        key: visualization._array_sha256(value) for key, value in arrays.items()
    }
    manifest["canonical_payload_sha256"] = visualization.evaluator._canonical_sha256(
        manifest
    )
    visualization.evaluator._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="selection differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_rehashed_upstream_lineage_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _write_synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["inputs"]["dataset_final_hash_manifest_sha256"] = "1" * 64
    manifest["canonical_payload_sha256"] = visualization.evaluator._canonical_sha256(
        manifest
    )
    visualization.evaluator._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="fixed input binding differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_rehashed_evaluator_source_tamper(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _write_synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["source"]["evaluator_source_set_sha256"] = "1" * 64
    manifest["canonical_payload_sha256"] = visualization.evaluator._canonical_sha256(
        manifest
    )
    visualization.evaluator._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source binding differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_rehashed_inherited_source_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _write_synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["inputs"]["inherited_training_packets"][0]["source_set_sha256"] = "2" * 64
    manifest["canonical_payload_sha256"] = visualization.evaluator._canonical_sha256(
        manifest
    )
    visualization.evaluator._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="inherited source identity differs"):
        visualization.load_replay(root)


def test_render_rejects_existing_or_dangling_original_output(
    tmp_path: Path, monkeypatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        visualization.render(Namespace(output_dir=existing))

    dangling = tmp_path / "dangling-output"
    path_type = type(dangling)
    original_is_symlink = path_type.is_symlink

    def simulated_dangling_symlink(path: Path) -> bool:
        return path == dangling or original_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", simulated_dangling_symlink)
    with pytest.raises(FileExistsError, match="output already exists"):
        visualization.render(Namespace(output_dir=dangling))


def test_render_detects_replay_mutation_immediately_after_load(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.animation import FFMpegWriter

    replay_root = tmp_path / "replay"
    output_root = tmp_path / "render"
    _write_synthetic_replay(replay_root)
    original_load_replay = visualization.load_replay

    def load_then_mutate(*args, **kwargs):
        bundle = original_load_replay(*args, **kwargs)
        replay_file = replay_root / visualization.REPLAY_FILE
        replay_file.write_bytes(replay_file.read_bytes() + b"swap")
        return bundle

    monkeypatch.setattr(visualization, "load_replay", load_then_mutate)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", staticmethod(lambda: True))
    with pytest.raises(ValueError, match="changed immediately after loading"):
        visualization.render(
            Namespace(
                replay_dir=replay_root,
                output_dir=output_root,
                ffmpeg=None,
                expected_replay_producer_sha256=(
                    visualization.VISUALIZATION_SOURCE_AT_IMPORT["sha256"]
                ),
                fps=2,
                frame_stride=208,
                dpi=40,
                bitrate=300,
            )
        )
    assert not output_root.exists()


def test_synthetic_render_smoke_emits_frozen_inventory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")

    replay_root = tmp_path / "replay"
    output_root = tmp_path / "render"
    _write_synthetic_replay(replay_root)
    manifest = visualization.render(
        Namespace(
            replay_dir=replay_root,
            output_dir=output_root,
            ffmpeg=Path(ffmpeg),
            expected_replay_producer_sha256=visualization.VISUALIZATION_SOURCE_AT_IMPORT[
                "sha256"
            ],
            fps=2,
            frame_stride=208,
            dpi=40,
            bitrate=300,
        )
    )

    expected_outputs = {
        "corrective_exposure_targets_h035.pdf",
        "corrective_exposure_targets_h104.pdf",
        "corrective_exposure_targets_h208.pdf",
        "rollout_path_roughness.pdf",
        "pderefiner_sampler_h208.pdf",
        "signed_density_error_seed17.mp4",
        "signed_density_error_seed29.mp4",
        "signed_density_error_seed43.mp4",
    }
    assert set(manifest["outputs"]) == expected_outputs
    assert {path.name for path in output_root.iterdir()} == expected_outputs | {
        visualization.RENDER_MANIFEST_FILE
    }
    assert all((output_root / name).stat().st_size > 0 for name in expected_outputs)
    assert manifest["rendering"]["animation_horizons"] == [0, 208]
    assert manifest["rendering"]["signed_error_limit_basis"] == (
        "maximum_absolute_rendered_native_quadrilateral_value"
    )
    assert manifest["rendering"]["signed_error_clipping"] is False
    assert manifest["rendering"]["static_signed_error_linthresh"] > 0.0
    assert manifest["rendering"]["animation_signed_error_linthresh"] > 0.0
    assert manifest["online_solver_calls"] is False
    assert manifest["online_defect_trigger"] is False
    assert manifest["prospective_opened"] is False
    assert manifest["sealed_opened"] is False

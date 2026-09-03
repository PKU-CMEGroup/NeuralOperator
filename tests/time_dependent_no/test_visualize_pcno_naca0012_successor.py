from __future__ import annotations

import json
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.time_dependent_no.visualize_pcno_naca0012_successor as visualization


def _aggregate(value: float) -> dict[str, float | int]:
    return {
        "count": 8,
        "median": value,
        "q90_nearest_rank": value,
        "maximum": value,
    }


def _synthetic_result() -> dict[str, object]:
    offline: dict[str, object] = {}
    methods: dict[str, object] = {}
    for method_offset, deployment in enumerate(visualization.DEPLOYMENTS):
        for seed_offset, seed in enumerate(visualization.SEEDS):
            value = float(1 + method_offset + seed_offset / 10.0)
            methods[visualization._primary_variant_key(deployment, seed)] = {
                "arm": deployment,
                "seed": seed,
                "state_stage": (
                    "corrected" if deployment == "PATH_PROJECTION" else "raw"
                ),
                "normalized_state_error_auc": _aggregate(value),
                "late_window_normalized_state_error_median": _aggregate(value + 1),
                "late_window_path_distance_median": _aggregate(value + 2),
                "late_window_projected_path_discrepancy_median": _aggregate(value + 3),
            }
            if deployment != "PATH_PROJECTION":
                offline[f"{deployment}:{seed}"] = {
                    "one_prefix_response_gain": _aggregate(value + 4)
                }
    return {"offline_diagnostics": offline, "method_summaries": methods}


def _synthetic_replay(root: Path) -> None:
    root.mkdir()
    node_count = 4
    horizons = np.asarray(visualization.FULL_HORIZONS, dtype=np.int64)
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    quads = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    reference = np.stack(
        [np.full(node_count, 1.0 + horizon / 1000.0) for horizon in horizons]
    ).astype(np.float32)
    predicted = np.broadcast_to(
        reference,
        (
            len(visualization.DEPLOYMENTS),
            len(visualization.SEEDS),
            len(horizons),
            node_count,
        ),
    ).copy()
    for method_offset in range(len(visualization.DEPLOYMENTS)):
        for seed_offset in range(len(visualization.SEEDS)):
            predicted[method_offset, seed_offset, 1:] += np.float32(
                1.0e-5 * (method_offset + 1) * (seed_offset + 1)
            )
    valid = np.ones(predicted.shape[:-1], dtype=np.bool_)
    registered_horizons = np.asarray(visualization.EVALUATOR_HORIZONS, dtype=np.int64)
    exact_reference = reference[registered_horizons].astype(np.float64)
    state_fields = 2
    qualitative_states = np.full(
        (
            len(visualization.DEPLOYMENTS),
            len(visualization.SEEDS),
            len(registered_horizons),
            node_count,
            state_fields,
        ),
        np.nan,
        dtype=np.float32,
    )
    for horizon_offset, horizon in enumerate(registered_horizons):
        qualitative_states[:, :, horizon_offset, :, 0] = predicted[:, :, horizon]
        qualitative_states[:, :, horizon_offset, :, 1] = 2.0 * predicted[:, :, horizon]
    evaluator_states = qualitative_states.copy()
    for method_offset in range(len(visualization.DEPLOYMENTS)):
        for seed_offset in range(len(visualization.SEEDS)):
            for horizon_offset, horizon in enumerate(registered_horizons):
                if horizon != 1:
                    evaluator_states[
                        method_offset, seed_offset, horizon_offset, :, 0
                    ] = exact_reference[horizon_offset] + np.float32(
                        0.02 * (method_offset + 1) * (seed_offset + 1)
                    )
    exact_predicted = evaluator_states[..., visualization.FIELD_INDEX].copy()
    exact_valid = np.ones(exact_predicted.shape[:-1], dtype=np.bool_)
    qualitative_registered_valid = np.ones_like(exact_valid)
    scale = 2.0
    errors = np.sqrt(
        np.mean(
            np.square((predicted.astype(np.float64) - reference[None, None]) / scale),
            axis=3,
        )
    )
    summary = visualization._extract_summary_arrays(_synthetic_result())
    training_records = [
        {
            "arm": deployment,
            "seed": seed,
            "final_hash_manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
        }
        for deployment in visualization.LEARNED_ARMS
        for seed in visualization.SEEDS
    ]
    first_nonfinite = {
        f"{deployment}:{seed}": None
        for deployment in visualization.DEPLOYMENTS
        for seed in visualization.SEEDS
    }
    replay_checks = []
    for method_offset, deployment in enumerate(visualization.DEPLOYMENTS):
        for seed_offset, seed in enumerate(visualization.SEEDS):
            for horizon_offset, horizon in enumerate(visualization.EVALUATOR_HORIZONS):
                replay_checks.append(
                    visualization._snapshot_check_record(
                        deployment=deployment,
                        seed=seed,
                        horizon=horizon,
                        observed=qualitative_states[
                            method_offset, seed_offset, horizon_offset
                        ],
                        expected=evaluator_states[
                            method_offset, seed_offset, horizon_offset
                        ],
                    )
                )
    np.savez_compressed(
        root / visualization.REPLAY_FILE,
        schema=np.asarray(visualization.REPLAY_SCHEMA),
        qualitative_contract_sha256=np.asarray(
            visualization.QUALITATIVE_CONTRACT_SHA256
        ),
        anchor=np.asarray(visualization.ANCHOR, dtype=np.int64),
        horizons=horizons,
        frame_indices=visualization.ANCHOR + horizons,
        seeds=np.asarray(visualization.SEEDS, dtype=np.int64),
        deployments=np.asarray(visualization.DEPLOYMENTS),
        coordinates=coordinates,
        quads=quads,
        airfoil_mask=np.asarray([1, 1, 1, 1], dtype=np.uint8),
        reference_density=reference,
        predicted_density=predicted,
        valid_mask=valid,
        registered_horizons=registered_horizons,
        exact_reference_density=exact_reference,
        exact_predicted_density=exact_predicted,
        exact_valid_mask=exact_valid,
        evaluator_prediction_states=evaluator_states,
        qualitative_registered_prediction_states=qualitative_states,
        qualitative_registered_valid_mask=qualitative_registered_valid,
        density_state_scale=np.asarray(scale, dtype=np.float64),
        density_error_rmse=errors,
        **summary,
    )
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
        "deployments": list(visualization.DEPLOYMENTS),
        "field": visualization.FIELD,
        "prospective_opened": False,
        "sealed_opened": False,
        "successor_contract_sha256": "1" * 64,
        "inherited_r0_contract_sha256": "6" * 64,
        "preregistration_sha256": "7" * 64,
        "dataset_final_hash_manifest_sha256": "2" * 64,
        "dataset_manifest_payload_sha256": "8" * 64,
        "calibration_final_hash_manifest_sha256": "3" * 64,
        "calibration_payload_sha256": "9" * 64,
        "evaluation_final_hash_manifest_sha256": "4" * 64,
        "training_packets": training_records,
        "source": {
            "visualization_script": dict(visualization.VISUALIZATION_SOURCE_AT_IMPORT),
            "evaluator_source_set_sha256": "5" * 64,
        },
        "first_nonfinite_horizon": first_nonfinite,
        "registered_snapshot_replay_checks": replay_checks,
        "registered_snapshot_check_summary": visualization._snapshot_check_summary(
            replay_checks
        ),
        "exact_static_snapshot_provenance": {
            "source": "verified_evaluation_rollout_snapshots",
            "evaluation_final_hash_manifest_sha256": "4" * 64,
            "evaluation_snapshot_file": {
                "relative_path": "rollout_snapshots.npz",
                "bytes": 123,
                "sha256": "d" * 64,
            },
            "registered_horizons": list(visualization.EVALUATOR_HORIZONS),
            "field": visualization.FIELD,
            "path_projection_state_stage": "corrected",
            "continuous_qualitative_replay_used_for_static_pdfs": False,
            "array_sha256": {
                "registered_horizons": visualization._array_sha256(registered_horizons),
                "exact_reference_density": visualization._array_sha256(exact_reference),
                "exact_predicted_density": visualization._array_sha256(exact_predicted),
                "exact_valid_mask": visualization._array_sha256(exact_valid),
                "evaluator_prediction_states": visualization._array_sha256(
                    evaluator_states
                ),
            },
        },
        "replay_file": visualization._file_record(
            root / visualization.REPLAY_FILE, root
        ),
        "arrays": {
            "reference_density": list(reference.shape),
            "predicted_density": list(predicted.shape),
            "valid_mask": list(valid.shape),
            "registered_horizons": list(registered_horizons.shape),
            "exact_reference_density": list(exact_reference.shape),
            "exact_predicted_density": list(exact_predicted.shape),
            "exact_valid_mask": list(exact_valid.shape),
            "evaluator_prediction_states": list(evaluator_states.shape),
            "qualitative_registered_prediction_states": list(qualitative_states.shape),
            "qualitative_registered_valid_mask": list(
                qualitative_registered_valid.shape
            ),
            "density_error_rmse": list(errors.shape),
            "coordinates": list(coordinates.shape),
            "quads": list(quads.shape),
        },
        "claim_boundary": visualization._claim_boundary(),
    }
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(root / visualization.REPLAY_MANIFEST_FILE, manifest)


def test_qualitative_selection_is_frozen_before_results() -> None:
    assert visualization.ANCHOR == 1234
    assert visualization.SEEDS == (17, 29, 43)
    assert visualization.DEPLOYMENTS == (
        "CLEAN",
        "IID_RECOVERY",
        "ERROR_SUBSPACE_RECOVERY",
        "DETACHED_PUSHFORWARD",
        "PATH_PROJECTION",
    )
    assert visualization.STATIC_HORIZONS == (35, 104, 208)
    assert visualization.QUALITATIVE_CONTRACT["population_role"] == "development"
    assert (
        visualization.QUALITATIVE_CONTRACT["paper_static_source"]
        == "exact_evaluator_registered_snapshots"
    )
    assert (
        visualization.QUALITATIVE_CONTRACT[
            "later_registered_checks_are_diagnostic_only"
        ]
        is True
    )
    assert visualization.QUALITATIVE_CONTRACT["prospective_opened"] is False
    assert visualization.QUALITATIVE_CONTRACT["sealed_opened"] is False


@pytest.mark.parametrize("role", ["prospective", "sealed"])
def test_replay_refuses_protected_role_before_resolving_paths(role: str) -> None:
    with pytest.raises(visualization.ProtectedPopulationError):
        visualization.replay(Namespace(population_role=role))


def test_path_projection_recurrence_feeds_back_corrected_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGeometry:
        def expand(self, batch_size: int, device: torch.device) -> object:
            assert batch_size == 2
            assert device.type == "cpu"
            return object()

    class FakeModel(torch.nn.Module):
        def prepare_fourier_tensors(self, geometry_batch: object) -> None:
            assert geometry_batch is not None

    class FakeProjector:
        def to(self, device: torch.device, dtype: torch.dtype) -> FakeProjector:
            assert device.type == "cpu"
            assert dtype == torch.float32
            return self

        def project(self, raw_state: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(corrected_state=0.5 * raw_state)

    def fake_corrected_step(
        model: torch.nn.Module,
        previous: torch.Tensor,
        current: torch.Tensor,
        geometry_batch: object,
        normalization: object,
        corrector: object,
        *,
        fourier_tensors: object,
    ) -> SimpleNamespace:
        del model, geometry_batch, normalization, fourier_tensors
        raw = current + 2.0
        corrected = corrector(raw)
        return SimpleNamespace(
            raw_next_state=raw,
            corrected_next_state=corrected,
            recurrent_previous=current,
        )

    monkeypatch.setattr(visualization, "corrected_recurrent_step", fake_corrected_step)
    previous = torch.zeros((2, 2, 5), dtype=torch.float32)
    current = torch.ones((2, 2, 5), dtype=torch.float32)
    density, valid, checkpoints, failed = visualization._rollout_deployment(
        deployment="PATH_PROJECTION",
        model=FakeModel(),
        previous=previous,
        current=current,
        selected_offset=0,
        geometry=FakeGeometry(),
        normalization=object(),
        projector=FakeProjector(),
        device=torch.device("cpu"),
    )
    assert np.all(valid)
    assert failed is None
    assert density[0, 0] == pytest.approx(1.0)
    assert density[1, 0] == pytest.approx(1.5)
    assert density[2, 0] == pytest.approx(1.75)
    assert checkpoints[35].shape == (2, 5)


def test_snapshot_keys_match_evaluator_semantics() -> None:
    assert visualization._snapshot_key("CLEAN", 17, 35) == "clean_seed17_raw_a1234_h35"
    assert (
        visualization._snapshot_key("PATH_PROJECTION", 43, 208)
        == "path_projection_seed43_corrected_a1234_h208"
    )


def test_snapshot_replay_rejects_first_transition_mismatch() -> None:
    references = {
        horizon: np.zeros((2, 5), dtype=np.float64)
        for horizon in visualization.EVALUATOR_HORIZONS
    }
    checkpoints = {
        horizon: np.ones((2, 5), dtype=np.float32)
        for horizon in visualization.EVALUATOR_HORIZONS
    }
    snapshots: dict[str, np.ndarray] = {}
    for horizon in visualization.EVALUATOR_HORIZONS:
        snapshots[f"reference_a1234_h{horizon}"] = references[horizon]
        snapshots[visualization._snapshot_key("CLEAN", 17, horizon)] = np.zeros(
            (2, 5), dtype=np.float32
        )
    with pytest.raises(
        ValueError, match="first-transition implementation-consistency gate failed"
    ):
        visualization._snapshot_replay_checks(
            deployment="CLEAN",
            seed=17,
            checkpoints=checkpoints,
            snapshots=snapshots,
            references=references,
        )


def test_snapshot_replay_records_later_divergence_without_rejecting() -> None:
    references = {
        horizon: np.zeros((2, 5), dtype=np.float64)
        for horizon in visualization.EVALUATOR_HORIZONS
    }
    snapshots: dict[str, np.ndarray] = {}
    checkpoints: dict[int, np.ndarray] = {}
    for horizon in visualization.EVALUATOR_HORIZONS:
        snapshots[f"reference_a1234_h{horizon}"] = references[horizon]
        snapshots[visualization._snapshot_key("CLEAN", 17, horizon)] = np.zeros(
            (2, 5), dtype=np.float32
        )
        checkpoints[horizon] = np.full(
            (2, 5), 0.0 if horizon == 1 else 1.0, dtype=np.float32
        )
    checks = visualization._snapshot_replay_checks(
        deployment="CLEAN",
        seed=17,
        checkpoints=checkpoints,
        snapshots=snapshots,
        references=references,
    )
    assert checks[0]["gate_role"] == "first_transition_implementation_gate"
    assert checks[0]["prediction_tolerance_pass"] is True
    assert all(check["gate_role"] == "diagnostic_only" for check in checks[1:])
    assert all(check["prediction_tolerance_pass"] is False for check in checks[1:])
    assert all(check["prediction_mismatch_count"] == 10 for check in checks[1:])


def test_summary_extraction_keeps_operational_corrector_response_undefined() -> None:
    arrays = visualization._extract_summary_arrays(_synthetic_result())
    assert arrays["response_gain"].shape == (5, 3)
    assert np.all(np.isfinite(arrays["response_gain"][:-1]))
    assert np.all(np.isnan(arrays["response_gain"][-1]))
    assert arrays["rollout_auc"][0, 0] == pytest.approx(1.0)
    assert arrays["late_path_distance"][-1, -1] == pytest.approx(7.2)


def test_training_record_validation_requires_one_source_set() -> None:
    records = [
        {
            "arm": deployment,
            "seed": seed,
            "final_hash_manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "source_set_sha256": "c" * 64,
        }
        for deployment in visualization.LEARNED_ARMS
        for seed in visualization.SEEDS
    ]
    records[-1]["source_set_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="one source set"):
        visualization._validate_training_records(records)


def test_nonfinite_bookkeeping_rejects_boolean_horizon(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    manifest = json.loads(
        (root / visualization.REPLAY_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    with np.load(root / visualization.REPLAY_FILE, allow_pickle=False) as archive:
        valid = np.array(archive["valid_mask"], copy=True)
        evaluator_states = np.array(archive["evaluator_prediction_states"], copy=True)
        exact_valid = np.array(archive["exact_valid_mask"], copy=True)
        qualitative_states = np.array(
            archive["qualitative_registered_prediction_states"], copy=True
        )
        qualitative_valid = np.array(
            archive["qualitative_registered_valid_mask"], copy=True
        )
    valid[0, 0, 1:] = False
    manifest["first_nonfinite_horizon"]["CLEAN:17"] = True
    with pytest.raises(ValueError, match="first-nonfinite"):
        visualization._validate_replay_bookkeeping(
            manifest,
            valid,
            evaluator_prediction_states=evaluator_states,
            evaluator_valid_mask=exact_valid,
            qualitative_prediction_states=qualitative_states,
            qualitative_valid_mask=qualitative_valid,
        )


def test_load_replay_and_plot_contract(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = tmp_path / "replay"
    _synthetic_replay(root)
    bundle = visualization.load_replay(root)
    assert bundle.anchor == 1234
    assert bundle.deployments == visualization.DEPLOYMENTS
    quads = visualization._visible_quads(bundle)
    limits = visualization.color_limits(bundle, quads)
    static_limits = visualization.static_color_limits(bundle, quads)
    assert limits["signed_error"][0] == -limits["signed_error"][1]
    assert static_limits["signed_error"][0] == -static_limits["signed_error"][1]
    reference, predictions, validity = visualization._exact_static_density(
        bundle, horizon=35, seed_offset=0
    )
    assert np.array_equal(
        reference,
        bundle.exact_reference_density[
            list(visualization.EVALUATOR_HORIZONS).index(35)
        ],
    )
    assert np.array_equal(
        predictions,
        bundle.exact_predicted_density[
            :, 0, list(visualization.EVALUATOR_HORIZONS).index(35)
        ],
    )
    assert np.all(validity)
    density_figure = visualization._density_snapshot(
        bundle, horizon=35, quads=quads, limits=static_limits
    )
    error_figure = visualization._error_snapshot(
        bundle, horizon=35, quads=quads, limits=static_limits
    )
    expected_density_face = visualization._quad_face_values(predictions[0], quads)
    expected_error_face = visualization._quad_face_values(
        (predictions[0] - reference) / bundle.density_state_scale, quads
    )
    assert np.allclose(
        np.asarray(density_figure.axes[1].collections[0].get_array()),
        expected_density_face,
    )
    assert np.allclose(
        np.asarray(error_figure.axes[0].collections[0].get_array()),
        expected_error_face,
    )
    trace_figure = visualization._error_trace_panel(bundle)
    tradeoff_figure, omitted = visualization._tradeoff_panel(bundle)
    assert density_figure._suptitle is None
    assert error_figure._suptitle is None
    assert trace_figure._suptitle is None
    assert tradeoff_figure._suptitle is None
    assert omitted == {
        "response_rollout": [],
        "path_state_accuracy": [],
        "path_phase_accuracy": [],
    }
    density_figure.savefig(tmp_path / "density.pdf")
    error_figure.savefig(tmp_path / "error.pdf")
    trace_figure.savefig(tmp_path / "trace.pdf")
    tradeoff_figure.savefig(tmp_path / "tradeoff.pdf")
    plt.close("all")


def test_load_replay_accepts_historical_visualization_source(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    historical_source = {
        "bytes": 99695,
        "sha256": "816463686ff7e53df8787a5cc31dc3cac8d731d40858820b2d6f986935a640bd",
    }
    manifest["source"]["visualization_script"] = historical_source
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(manifest_path, manifest)

    bundle = visualization.load_replay(
        root, expected_visualization_source_sha256=historical_source["sha256"]
    )

    assert bundle.manifest["source"]["visualization_script"] == historical_source

    with pytest.raises(ValueError, match="source closure differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_malformed_historical_visualization_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["source"]["visualization_script"] = {
        "bytes": 0,
        "sha256": "816463686ff7e53df8787a5cc31dc3cac8d731d40858820b2d6f986935a640bd",
    }
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source closure differs"):
        visualization.load_replay(
            root,
            expected_visualization_source_sha256=manifest["source"][
                "visualization_script"
            ]["sha256"],
        )


def test_load_replay_rejects_rehashed_selection_tamper(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["anchor"] = 1238
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="selection or claim boundary"):
        visualization.load_replay(root)


def test_load_replay_rejects_rehashed_snapshot_diagnostic_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["registered_snapshot_replay_checks"][1]["prediction_mismatch_count"] = 0
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="snapshot diagnostic record differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_rehashed_exact_static_array_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    replay_path = root / visualization.REPLAY_FILE
    with np.load(replay_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    horizon_offset = list(visualization.EVALUATOR_HORIZONS).index(35)
    arrays["exact_predicted_density"][0, 0, horizon_offset, 0] += np.float32(0.5)
    arrays["evaluator_prediction_states"][
        0, 0, horizon_offset, 0, visualization.FIELD_INDEX
    ] += np.float32(0.5)
    np.savez_compressed(replay_path, **arrays)
    manifest_path = root / visualization.REPLAY_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("canonical_payload_sha256")
    manifest["replay_file"] = visualization._file_record(replay_path, root)
    manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
    visualization._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="provenance binding differs"):
        visualization.load_replay(root)


def test_load_replay_rejects_extra_packet_inventory(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    (root / "unregistered.tmp").write_bytes(b"not part of the replay packet")
    with pytest.raises(ValueError, match="replay inventory differs"):
        visualization.load_replay(root)


def test_load_replay_rehashes_packet_after_array_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "replay"
    _synthetic_replay(root)
    replay_path = root / visualization.REPLAY_FILE
    real_validate = visualization._validate_replay_bookkeeping

    def validate_then_mutate(*args, **kwargs) -> None:
        real_validate(*args, **kwargs)
        with replay_path.open("ab") as stream:
            stream.write(b"late mutation")

    monkeypatch.setattr(
        visualization, "_validate_replay_bookkeeping", validate_then_mutate
    )
    with pytest.raises(ValueError, match="changed while rendering"):
        visualization.load_replay(root)


def test_render_input_record_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    record = visualization._file_record(path, tmp_path)
    path.write_bytes(b"after")
    with pytest.raises(ValueError, match="changed while rendering"):
        visualization._reverify_file_record(path, tmp_path, record)


def test_replay_output_overlap_is_refused_before_source_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    arguments = Namespace(
        population_role="development",
        contract=tmp_path / "r0.json",
        successor_contract=tmp_path / "successor.json",
        preregistration=tmp_path / "preregistration.md",
        dataset_dir=dataset,
        calibration_dir=tmp_path / "calibration",
        evaluation_dir=tmp_path / "evaluation",
        training_dir=[tmp_path / "training"],
        output_dir=dataset / "replay-output",
    )
    monkeypatch.setattr(
        visualization,
        "_reverify_visualization_source",
        lambda: pytest.fail("visualization source read occurred after overlap"),
    )
    with pytest.raises(ValueError, match="replay output overlaps dataset packet"):
        visualization.replay(arguments)


def test_render_output_overlap_is_refused_before_source_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replay = tmp_path / "replay"
    arguments = Namespace(
        replay_dir=replay,
        output_dir=replay / "render-output",
    )
    monkeypatch.setattr(
        visualization,
        "_reverify_visualization_source",
        lambda: pytest.fail("visualization source read occurred after overlap"),
    )
    with pytest.raises(ValueError, match="render output overlaps.*replay input"):
        visualization.render(arguments)


def test_atomic_visualization_publication_cleans_late_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"

    def build(root: Path) -> dict:
        (root / "partial.pdf").write_bytes(b"complete staged bytes")
        (root / visualization.RENDER_MANIFEST_FILE).write_text("{}", encoding="utf-8")
        return {"status": "complete"}

    def fail_after_complete_build(root: Path, manifest: dict) -> None:
        del root, manifest
        raise RuntimeError("synthetic late publication failure")

    with pytest.raises(RuntimeError, match="late publication failure"):
        visualization._publish_directory_atomically(
            output, build, before_publish=fail_after_complete_build
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".published.render-staging-*"))
    assert list(tmp_path.iterdir()) == []


def test_atomic_visualization_publication_never_clobbers_competing_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "published"

    def build(root: Path) -> dict:
        (root / "figure.pdf").write_bytes(b"staged figure")
        return {"status": "complete"}

    real_publish = visualization._rename_directory_no_replace

    def competing_publish(source: Path, destination: Path) -> None:
        destination = Path(destination)
        destination.mkdir()
        real_publish(Path(source), destination)

    monkeypatch.setattr(
        visualization, "_rename_directory_no_replace", competing_publish
    )
    with pytest.raises(OSError):
        visualization._publish_directory_atomically(output, build)
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not (output / "figure.pdf").exists()
    assert not list(tmp_path.glob(".published.render-staging-*"))


def test_atomic_visualization_publication_concurrent_loser_cleans_only_own_staging(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    barrier = Barrier(2)

    def publish(owner: str) -> bool:
        def build(root: Path) -> dict:
            (root / "owner.txt").write_text(owner, encoding="utf-8")
            barrier.wait(timeout=10.0)
            return {"status": "complete"}

        try:
            visualization._publish_directory_atomically(output, build)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, owner) for owner in ("first", "second")]
        outcomes = [future.result() for future in futures]
    assert sorted(outcomes) == [False, True]
    assert (output / "owner.txt").read_text(encoding="utf-8") in {"first", "second"}
    assert not list(tmp_path.glob(".published.render-staging-*"))


@pytest.mark.parametrize("mutation", ("replay", "manifest", "inventory"))
def test_final_staged_packet_verification_rejects_late_mutation(
    mutation: str, tmp_path: Path
) -> None:
    output = tmp_path / "published"
    records: dict[str, dict] = {}

    def build(root: Path) -> dict:
        (root / visualization.REPLAY_MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
        (root / visualization.REPLAY_FILE).write_bytes(b"verified replay")
        (root / "figure.pdf").write_bytes(b"verified figure")
        records["replay_manifest"] = visualization._file_record(
            root / visualization.REPLAY_MANIFEST_FILE, root
        )
        records["replay_file"] = visualization._file_record(
            root / visualization.REPLAY_FILE, root
        )
        manifest = {
            "replay_manifest": records["replay_manifest"],
            "replay_file": records["replay_file"],
            "outputs": {
                "figure.pdf": visualization._file_record(root / "figure.pdf", root)
            },
        }
        manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
        visualization._write_json(root / visualization.RENDER_MANIFEST_FILE, manifest)
        return manifest

    def mutate_then_verify(root: Path, manifest: dict) -> None:
        if mutation == "replay":
            (root / visualization.REPLAY_FILE).write_bytes(b"late mutation")
        elif mutation == "manifest":
            manifest["canonical_payload_sha256"] = "0" * 64
            visualization._write_json(
                root / visualization.RENDER_MANIFEST_FILE, manifest
            )
        elif mutation == "inventory":
            (root / "unregistered.tmp").write_bytes(b"late mutation")
        else:  # pragma: no cover - the parameterization is closed above
            raise AssertionError("unknown mutation")
        visualization._verify_staged_render_packet(
            root,
            manifest,
            replay_manifest_record=records["replay_manifest"],
            replay_file_record=records["replay_file"],
        )

    with pytest.raises(ValueError):
        visualization._publish_directory_atomically(
            output, build, before_publish=mutate_then_verify
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".published.render-staging-*"))
    assert list(tmp_path.iterdir()) == []


def test_atomic_visualization_publication_publishes_complete_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    records: dict[str, dict] = {}

    def build(root: Path) -> dict:
        (root / "figure.pdf").write_bytes(b"figure")
        (root / visualization.REPLAY_FILE).write_bytes(b"replay")
        (root / visualization.REPLAY_MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
        records["replay_manifest"] = visualization._file_record(
            root / visualization.REPLAY_MANIFEST_FILE, root
        )
        records["replay_file"] = visualization._file_record(
            root / visualization.REPLAY_FILE, root
        )
        manifest = {
            "replay_manifest": records["replay_manifest"],
            "replay_file": records["replay_file"],
            "outputs": {
                "figure.pdf": visualization._file_record(root / "figure.pdf", root)
            },
        }
        manifest["canonical_payload_sha256"] = visualization._canonical_sha256(manifest)
        visualization._write_json(root / visualization.RENDER_MANIFEST_FILE, manifest)
        return manifest

    def verify(root: Path, manifest: dict) -> None:
        visualization._verify_staged_render_packet(
            root,
            manifest,
            replay_manifest_record=records["replay_manifest"],
            replay_file_record=records["replay_file"],
        )

    result = visualization._publish_directory_atomically(
        output, build, before_publish=verify
    )
    assert result["canonical_payload_sha256"] == visualization._canonical_sha256(
        {
            key: value
            for key, value in result.items()
            if key != "canonical_payload_sha256"
        }
    )
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {
        "figure.pdf",
        visualization.REPLAY_FILE,
        visualization.REPLAY_MANIFEST_FILE,
        visualization.RENDER_MANIFEST_FILE,
    }
    assert not list(tmp_path.glob(".published.render-staging-*"))

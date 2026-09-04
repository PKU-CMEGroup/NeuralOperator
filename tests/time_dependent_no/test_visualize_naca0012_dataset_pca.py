import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import scripts.time_dependent_no.visualize_naca0012_dataset_pca as pca_script
from scripts.time_dependent_no.visualize_naca0012_dataset_pca import (
    ARRAYS_SCHEMA,
    FINAL_HASH_SCHEMA,
    SnapshotPCA,
    _canonical_sha256,
    compute_snapshot_pca,
    pca_from_gram,
    render_pca_figure,
    select_train_bdf2_views,
    snapshot_gram,
    write_pca_packet,
)
from utility.time_dependent_no.pcno_naca0012 import NACANormalization


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.ones(5, dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )


def _synthetic_pca(num_samples: int = 12) -> SnapshotPCA:
    explained = np.asarray(
        [0.55, 0.41, 0.018, 0.017, 0.0025, 0.0015, 0.0005, 0.0005],
        dtype=np.float64,
    )
    explained /= np.sum(explained)
    scores = np.zeros((num_samples, len(explained)), dtype=np.float64)
    phase = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    scores[:, 0] = np.cos(phase)
    scores[:, 1] = np.sin(phase)
    return SnapshotPCA(
        eigenvalues=explained.copy(),
        explained_variance_ratio=explained,
        cumulative_explained_variance=np.cumsum(explained),
        scores=scores,
        total_sum_squared_deviation=1.0,
        participation_ratio=float(1.0 / np.sum(np.square(explained))),
        entropy_effective_rank=float(np.exp(-np.sum(explained * np.log(explained)))),
    )


def test_select_train_bdf2_views_uses_only_interior_current_frames() -> None:
    states = np.arange(5 * 2 * 5, dtype=np.float64).reshape(5, 2, 5)
    frames = np.arange(100, 105, dtype=np.int64)

    previous, current, centers = select_train_bdf2_views(
        states, frames, expected_frame_count=5
    )

    np.testing.assert_array_equal(previous, states[:-2])
    np.testing.assert_array_equal(current, states[1:-1])
    np.testing.assert_array_equal(centers, [101, 102, 103])
    with pytest.raises(ValueError, match="contiguous"):
        select_train_bdf2_views(
            states,
            np.asarray([100, 101, 103, 104, 105], dtype=np.int64),
            expected_frame_count=5,
        )


def test_source_reverification_detects_in_run_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(pca_script.PCA_SOURCE_AT_IMPORT)
    changed["sha256"] = "0" * 64
    monkeypatch.setattr(pca_script, "_source_record", lambda: changed)

    with pytest.raises(ValueError, match="source changed during execution"):
        pca_script._reverify_source()


def test_snapshot_gram_matches_direct_model_and_area_metrics() -> None:
    rng = np.random.default_rng(11)
    states = rng.normal(size=(7, 3, 5)).astype(np.float64)
    normalization = NACANormalization(
        state_mean=np.arange(5, dtype=np.float64) / 10.0,
        state_scale=np.arange(1, 6, dtype=np.float64),
        residual_scale=np.ones(5, dtype=np.float64),
        state_rms=np.ones(5, dtype=np.float64),
    )
    normalized = normalization.normalize_state(states)
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    direct = centered.reshape(7, -1)

    observed = snapshot_gram((states,), normalization, block_nodes=2)
    np.testing.assert_allclose(observed, direct @ direct.T, rtol=1e-13, atol=1e-13)

    weights = np.asarray([0.1, 0.2, 0.7], dtype=np.float64)
    weighted = centered * np.sqrt(weights)[None, :, None]
    observed_weighted = snapshot_gram(
        (states,), normalization, node_weights=weights[:, None], block_nodes=1
    )
    expected_weighted = weighted.reshape(7, -1) @ weighted.reshape(7, -1).T
    np.testing.assert_allclose(
        observed_weighted, expected_weighted, rtol=1e-13, atol=1e-13
    )


def test_pca_recovers_known_rank_and_canonicalizes_score_signs() -> None:
    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=(10, 2))
    coefficients -= np.mean(coefficients, axis=0, keepdims=True)
    features = rng.normal(size=(2, 20))
    centered = coefficients @ features
    gram = np.asarray(centered @ centered.T, dtype=np.float64)

    result = pca_from_gram(gram, max_components=6)

    assert result.cumulative_explained_variance[1] == pytest.approx(1.0)
    assert np.all(result.explained_variance_ratio[2:] < 1.0e-14)
    for component in range(result.scores.shape[1]):
        pivot = int(np.argmax(np.abs(result.scores[:, component])))
        assert result.scores[pivot, component] >= 0.0


def test_complete_bdf2_gram_is_sum_of_previous_and_current_grams() -> None:
    rng = np.random.default_rng(5)
    previous = rng.normal(size=(8, 4, 5)).astype(np.float64)
    current = rng.normal(size=(8, 4, 5)).astype(np.float64)
    normalization = _normalization()

    pair = snapshot_gram((previous, current), normalization, block_nodes=3)
    expected = snapshot_gram((previous,), normalization, block_nodes=3)
    expected += snapshot_gram((current,), normalization, block_nodes=3)

    np.testing.assert_allclose(pair, expected, rtol=1e-13, atol=1e-13)


def test_render_pca_figure_writes_vector_pdf_and_png(tmp_path: Path) -> None:
    model = _synthetic_pca()
    area = _synthetic_pca()
    pdf = tmp_path / "figure.pdf"
    png = tmp_path / "figure.png"

    render_pca_figure(
        pdf,
        png,
        np.arange(200, 212, dtype=np.int64),
        model,
        area,
        plot_components=8,
    )

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_packet_seals_train_only_outputs_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(13)
    phase = np.linspace(0.0, 5.0 * np.pi, 240, dtype=np.float64)
    states = np.zeros((240, 2, 5), dtype=np.float64)
    for field in range(5):
        states[:, 0, field] = np.sin((field + 1) * phase)
        states[:, 1, field] = np.cos((field + 1) * phase)
    states += 1.0e-4 * rng.normal(size=states.shape)
    frames = np.arange(955, 1195, dtype=np.int64)
    dataset_manifest = {
        "canonical_payload_sha256": "a" * 64,
        "source_manifest": {"source_set_sha256": "b" * 64},
    }
    output = tmp_path / "pca"
    packet_explained = np.geomspace(1.0, 1.0e-8, 238, dtype=np.float64)
    packet_explained /= np.sum(packet_explained)
    packet_scores = np.zeros((238, 16), dtype=np.float64)
    packet_scores[:, 0] = np.cos(phase[1:-1])
    packet_scores[:, 1] = np.sin(phase[1:-1])
    packet_pca = SnapshotPCA(
        eigenvalues=packet_explained.copy(),
        explained_variance_ratio=packet_explained,
        cumulative_explained_variance=np.cumsum(packet_explained),
        scores=packet_scores,
        total_sum_squared_deviation=1.0,
        participation_ratio=float(1.0 / np.sum(np.square(packet_explained))),
        entropy_effective_rank=float(
            np.exp(-np.sum(packet_explained * np.log(packet_explained)))
        ),
    )
    monkeypatch.setattr(
        pca_script, "compute_snapshot_pca", lambda *args, **kwargs: packet_pca
    )

    summary = write_pca_packet(
        output,
        dataset_manifest=dataset_manifest,
        dataset_final_hash_manifest_sha256="c" * 64,
        num_nodes=2,
        node_weights=np.asarray([[0.4], [0.6]], dtype=np.float64),
        train_states=states,
        train_frame_indices=frames,
        normalization=_normalization(),
        block_nodes=1,
        max_components=16,
    )

    assert summary["fit_scope"]["primary_samples"] == 238
    assert summary["fit_scope"]["primary_state_definition"] == "train_states[1:-1]"
    assert summary["access"]["population_values_used_in_pca"] == ["train"]
    assert summary["access"]["development_values_used_in_pca"] is False
    assert summary["access"]["prospective_opened"] is False
    assert summary["access"]["sealed_opened"] is False
    assert summary["interpretation"]["not_supported"]
    payload = dict(summary)
    observed_hash = payload.pop("canonical_payload_sha256")
    assert observed_hash == _canonical_sha256(payload)

    with np.load(output / "pca_arrays.npz", allow_pickle=False) as archive:
        assert np.asarray(archive["schema"]).item() == ARRAYS_SCHEMA
        np.testing.assert_array_equal(
            archive["train_center_frame_indices"], np.arange(956, 1194)
        )
        assert archive["model_metric_scores"].shape == (238, 16)

    final = json.loads((output / "final_hash_manifest.json").read_text("utf-8"))
    assert final["schema"] == FINAL_HASH_SCHEMA
    assert final["self_hash_excluded"] is True
    assert final["prospective_opened"] is False
    assert final["sealed_opened"] is False
    assert set(final["files"]) == {
        "naca0012_train_pca.pdf",
        "naca0012_train_pca.png",
        "pca_arrays.npz",
        "pca_summary.json",
    }
    for name, record in final["files"].items():
        payload_bytes = (output / name).read_bytes()
        assert record["bytes"] == len(payload_bytes)
        assert record["sha256"] == sha256(payload_bytes).hexdigest()

    with pytest.raises(FileExistsError):
        write_pca_packet(
            output,
            dataset_manifest=dataset_manifest,
            dataset_final_hash_manifest_sha256="c" * 64,
            num_nodes=2,
            node_weights=np.asarray([[0.4], [0.6]], dtype=np.float64),
            train_states=states,
            train_frame_indices=frames,
            normalization=_normalization(),
        )


def test_compute_snapshot_pca_rejects_nonpositive_area_weights() -> None:
    states = np.ones((8, 2, 5), dtype=np.float64)
    states[:, 0, 0] = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="positive"):
        compute_snapshot_pca(
            (states,),
            _normalization(),
            node_weights=np.asarray([1.0, 0.0], dtype=np.float64),
        )

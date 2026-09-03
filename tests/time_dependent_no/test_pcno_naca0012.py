from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

import utility.time_dependent_no.pcno_naca0012 as naca_core
from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACABDF2Dataset,
    NACAGeometry,
    NACANormalization,
    NACAPCNOResidual,
    NACAVerifiedContext,
    VerifiedNACAContract,
    VerifiedNACAGeometry,
    _build_naca_geometry_from_mesh,
    _build_synthetic_bdf2_dataset,
    _extract_naca_state,
    _extract_native_state,
    _read_with_stable_file_identity,
    _resolve_real_directory,
    build_input,
    build_naca_bdf2_dataset,
    build_naca_geometry,
    build_naca_pcno,
    extract_verified_diagnostic_state,
    extract_verified_trajectory_state,
    fit_naca_normalization,
    load_naca_baseline_contract,
    load_verified_naca_geometry,
    predict_normalized_residual,
    recurrent_step,
    validate_naca_model_config,
    validate_parent_evidence,
    validate_population_contract,
    validate_role_access,
)
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_RESTART_FIELDS,
    SU2_BINARY_MAGIC,
    SU2_FIELD_NAME_BYTES,
    SU2Mesh,
    sha256_file,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPOSITORY / "docs" / "time_dependent_no" / "R0_NACA_PCNO_BASELINE_CONTRACT.json"
)
RESOURCE_ROOT = (
    REPOSITORY / "artifacts" / "time_dependent_no" / "su2_naca0012_resources"
)
RESOURCE_MANIFEST_PATH = (
    REPOSITORY / "docs" / "time_dependent_no" / "R0_SU2_NACA_RESOURCE_MANIFEST.json"
)


def _contract() -> VerifiedNACAContract:
    return load_naca_baseline_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def verified_context() -> NACAVerifiedContext:
    return validate_parent_evidence(
        _contract(),
        repository_root=REPOSITORY,
        resource_root=RESOURCE_ROOT,
    )


@pytest.fixture(scope="module")
def verified_geometry(
    verified_context: NACAVerifiedContext,
) -> VerifiedNACAGeometry:
    return build_naca_geometry(verified_context)


def _synthetic_mesh() -> SU2Mesh:
    points = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.5, -1.5],
            [1.5, -1.5],
            [1.5, 1.5],
            [-1.5, 1.5],
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
        ],
        dtype=np.float64,
    )
    elements = (
        (9, (4, 5, 1, 0)),
        (9, (5, 6, 2, 1)),
        (9, (6, 7, 3, 2)),
        (9, (7, 4, 0, 3)),
        (9, (8, 9, 5, 4)),
        (9, (9, 10, 6, 5)),
        (9, (10, 11, 7, 6)),
        (9, (11, 8, 4, 7)),
    )
    markers = {
        "airfoil": (
            (3, (0, 1)),
            (3, (1, 2)),
            (3, (2, 3)),
            (3, (3, 0)),
        ),
        "farfield": (
            (3, (8, 9)),
            (3, (9, 10)),
            (3, (10, 11)),
            (3, (11, 8)),
        ),
    }
    return SU2Mesh(
        dimension=2,
        num_elements=len(elements),
        num_points=len(points),
        points=points,
        elements=elements,
        element_type_counts={9: len(elements)},
        marker_elements=markers,
        marker_element_counts={key: len(value) for key, value in markers.items()},
    )


def _write_restart(path: Path, fields: tuple[str, ...]) -> np.ndarray:
    num_points = 4
    values = np.arange(num_points * len(fields), dtype=np.float64).reshape(
        num_points, len(fields)
    )
    values[:, fields.index("x")] = np.arange(num_points, dtype=np.float64)
    values[:, fields.index("y")] = -np.arange(num_points, dtype=np.float64)
    header = np.asarray([SU2_BINARY_MAGIC, len(fields), num_points, 0, 0], dtype="<i4")
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        for field in fields:
            encoded = field.encode("ascii") + b"\0"
            handle.write(encoded.ljust(SU2_FIELD_NAME_BYTES, b"\0"))
        handle.write(np.asarray(values, dtype="<f8").tobytes())
    return values


def test_frozen_contract_loads_and_population_is_leakage_safe() -> None:
    assert sha256_file(CONTRACT_PATH) == BASELINE_CONTRACT_SHA256
    contract = _contract()
    validate_population_contract(contract)

    for section, key, value in (
        ("phase_population", "primary_rollout_horizon_steps", 209),
        ("training", "epochs", 101),
        ("R0_decision_rule", "initially_accurate", "altered"),
    ):
        modified = contract.to_mapping()
        modified[section][key] = value
        with pytest.raises(TypeError, match="VerifiedNACAContract"):
            validate_population_contract(modified)  # type: ignore[arg-type]
    validate_population_contract(contract)


def test_contract_hash_and_json_use_one_exact_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = CONTRACT_PATH.read_bytes()
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(snapshot)
    snapshot_sha256 = sha256(snapshot).hexdigest()
    original_read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path == contract_path:
            path.write_bytes(b"{}")
        return content

    monkeypatch.setattr(naca_core, "BASELINE_CONTRACT_SHA256", snapshot_sha256)
    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    contract = load_naca_baseline_contract(contract_path)
    assert contract.file_sha256 == snapshot_sha256
    assert contract["schema"] == "time_dependent_no.su2_naca0012_pcno_baseline.v1"


def test_verified_context_retains_bound_stage0_manifest_sha256(
    verified_context: NACAVerifiedContext,
) -> None:
    expected = sha256_file(RESOURCE_MANIFEST_PATH)
    assert (
        expected == "7e20bb67ebcf09447f0badd4d10752a78f749f5c038bb444cedff81d39ff56f5"
    )
    assert verified_context.resource_manifest_sha256 == expected


def test_role_access_has_no_protected_population_bypass() -> None:
    contract = _contract()
    train_anchors = contract["phase_population"]["roles"]["train"][
        "anchor_input_indices"
    ]
    assert validate_role_access(
        contract, "train", train_anchors, horizon_steps=208
    ) == tuple(train_anchors)
    prospective = contract["phase_population"]["roles"]["prospective"][
        "anchor_input_indices"
    ]
    with pytest.raises(PermissionError, match="remains sealed"):
        validate_role_access(contract, "prospective", prospective)
    with pytest.raises(TypeError, match="unexpected keyword"):
        validate_role_access(
            contract,
            "prospective",
            prospective,
            horizon_steps=208,
            revealed_roles={"prospective"},  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="integer indices"):
        validate_role_access(contract, "train", [956.5])


def test_extract_native_state_requires_exact_19_field_schema(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_flow_00001.dat"
    values = _write_restart(path, NACA_NATIVE_RESTART_FIELDS)
    expected = values[
        :,
        [
            NACA_NATIVE_RESTART_FIELDS.index(name)
            for name in (
                "Density",
                "Momentum_x",
                "Momentum_y",
                "Energy",
                "Nu_Tilde",
            )
        ],
    ]
    state = _extract_native_state(
        path,
        expected_sha256=sha256_file(path),
        expected_num_points=4,
        expected_coordinates=values[:, :2],
    )
    assert state.dtype == np.float64
    np.testing.assert_array_equal(state, expected)

    wrong = tmp_path / "wrong.dat"
    _write_restart(wrong, NACA_NATIVE_RESTART_FIELDS[:-1])
    with pytest.raises(ValueError, match="exact native 19-field"):
        _extract_native_state(wrong)


def test_extract_naca_state_accepts_only_exact_canonical_or_native_schema(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical_00497.dat"
    values = _write_restart(canonical, NACA_RESTART_FIELDS)
    expected = values[
        :,
        [
            NACA_RESTART_FIELDS.index(name)
            for name in (
                "Density",
                "Momentum_x",
                "Momentum_y",
                "Energy",
                "Nu_Tilde",
            )
        ],
    ]
    np.testing.assert_array_equal(
        _extract_naca_state(
            canonical,
            expected_sha256=sha256_file(canonical),
            expected_num_points=4,
            expected_coordinates=values[:, :2],
        ),
        expected,
    )
    malformed = tmp_path / "malformed.dat"
    _write_restart(malformed, NACA_RESTART_FIELDS[:-1])
    with pytest.raises(ValueError, match="canonical-17 nor native-19"):
        _extract_naca_state(malformed)


def test_quad_geometry_has_fixed_graph_measure_and_round_trips() -> None:
    geometry = _build_naca_geometry_from_mesh(_synthetic_mesh())
    assert geometry.elements.shape == (8, 5)
    assert np.all(geometry.elements[:, 0] == 2)
    assert geometry.static_features.shape == (12, 6)
    assert geometry.boundary_one_hot[:, 0].sum() == 4
    assert geometry.boundary_one_hot[:, 1].sum() == 4
    assert geometry.boundary_one_hot[:, 2].sum() == 4
    assert geometry.node_weights.sum() == pytest.approx(1.0)
    assert geometry.directed_edges.shape == geometry.edge_gradient_weights.shape
    assert geometry.directed_edges.shape[0] > 0

    restored = NACAGeometry.from_mapping(geometry.to_mapping())
    for key, value in geometry.to_mapping().items():
        np.testing.assert_array_equal(restored.to_mapping()[key], value)
    expanded = restored.expand(3, "cpu")
    assert expanded["nodes"].shape == (3, 12, 2)
    assert expanded["nodes"].stride(0) == 0
    assert expanded["node_weights"].stride(0) == 0
    assert expanded["directed_edges"].stride(0) == 0


def test_quad_geometry_rejects_non_vtk9_or_overlapping_markers() -> None:
    mesh = _synthetic_mesh()
    nonquad = SU2Mesh(
        **{
            **mesh.__dict__,
            "elements": ((5, (0, 1, 4)),),
            "num_elements": 1,
            "element_type_counts": {5: 1},
        }
    )
    with pytest.raises(ValueError, match="all-VTK9"):
        _build_naca_geometry_from_mesh(nonquad)

    overlapping = SU2Mesh(
        **{
            **mesh.__dict__,
            "marker_elements": {
                "airfoil": mesh.marker_elements["airfoil"],
                "farfield": ((3, (1, 8)), *mesh.marker_elements["farfield"]),
            },
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        _build_naca_geometry_from_mesh(overlapping)


def test_production_geometry_rejects_arbitrary_mesh_and_context_binds_live_mesh(
    verified_context: NACAVerifiedContext,
    verified_geometry: VerifiedNACAGeometry,
) -> None:
    with pytest.raises(TypeError, match="NACAVerifiedContext"):
        build_naca_geometry(_synthetic_mesh())  # type: ignore[arg-type]
    record = verified_context.mesh_record
    assert verified_context.mesh_path.name == record["file"]
    assert verified_context.mesh_path.stat().st_size == record["bytes"]
    assert sha256_file(verified_context.mesh_path) == record["sha256"]
    assert verified_context.mesh_path.parent == RESOURCE_ROOT.resolve()
    assert verified_geometry.num_nodes == 14576


def test_parent_evidence_rejects_trajectory_packet_as_stage0_resource_root() -> None:
    trajectory_root = (
        REPOSITORY
        / "artifacts"
        / "time_dependent_no"
        / "su2_naca0012_trajectory_phase_pilot_20260831a"
    )
    with pytest.raises(ValueError, match="diagnostic record 499"):
        validate_parent_evidence(
            _contract(),
            repository_root=REPOSITORY,
            resource_root=trajectory_root,
        )


def test_role_trajectory_and_diagnostic_guards_run_before_record_or_file_access(
    verified_context: NACAVerifiedContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_record_lookup(*args, **kwargs):
        raise AssertionError("record lookup must not occur")

    monkeypatch.setattr(naca_core, "_restart_record", fail_record_lookup)
    coordinates = np.empty((0, 2), dtype=np.float64)
    with pytest.raises(PermissionError, match="remains sealed"):
        extract_verified_trajectory_state(
            verified_context,
            "prospective",
            1511,
            expected_coordinates=coordinates,
        )
    with pytest.raises(ValueError, match="outside the train owned block"):
        extract_verified_trajectory_state(
            verified_context,
            "train",
            1233,
            expected_coordinates=coordinates,
        )

    def fail_filesystem_access(*args, **kwargs):
        raise AssertionError("filesystem access must not occur")

    monkeypatch.setattr(naca_core, "_safe_child", fail_filesystem_access)
    with pytest.raises(PermissionError, match="only diagnostic frames"):
        extract_verified_diagnostic_state(
            verified_context,
            500,
            expected_coordinates=coordinates,
        )


def test_train_only_normalization_reads_each_frozen_train_frame_once() -> None:
    contract = _contract()
    seen: list[int] = []

    def load(index: int) -> np.ndarray:
        seen.append(index)
        time = float(index - 955)
        nodes = np.arange(3, dtype=np.float64)[:, None]
        fields = np.arange(5, dtype=np.float64)[None, :]
        return time + 0.25 * nodes + 0.5 * fields

    normalization = fit_naca_normalization(load, contract)
    assert seen == list(range(955, 1195))
    stacked = np.stack([load(index) for index in range(955, 1195)])
    np.testing.assert_allclose(
        normalization.state_mean, stacked.mean(axis=(0, 1)), rtol=1e-14
    )
    np.testing.assert_allclose(
        normalization.state_scale, stacked.std(axis=(0, 1)), rtol=1e-13
    )
    np.testing.assert_allclose(normalization.residual_scale, np.ones(5))
    restored = NACANormalization.from_mapping(normalization.to_mapping())
    np.testing.assert_array_equal(restored.state_mean, normalization.state_mean)


def test_bdf2_dataset_keeps_storage_float64_and_casts_presentations() -> None:
    frame_indices = [10, 11, 12, 13]
    states = np.stack(
        [np.full((3, 5), float(index), dtype=np.float64) for index in frame_indices]
    )
    normalization = NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.full(5, 2.0),
        state_rms=np.ones(5),
    )
    with pytest.raises(TypeError, match="_token"):
        NACABDF2Dataset(
            states,
            frame_indices,
            [11, 12],
            normalization,
        )
    dataset = _build_synthetic_bdf2_dataset(
        states,
        frame_indices,
        [11, 12],
        normalization,
    )
    previous, current, target_residual, center = dataset[0]
    assert dataset.states.dtype == np.float64
    assert previous.dtype == current.dtype == target_residual.dtype == torch.float32
    assert center == 11
    torch.testing.assert_close(previous, torch.full((3, 5), 10.0))
    torch.testing.assert_close(current, torch.full((3, 5), 11.0))
    torch.testing.assert_close(target_residual, torch.full((3, 5), 0.5))


def test_production_dataset_factory_requires_exact_open_role_layout(
    verified_geometry: VerifiedNACAGeometry,
) -> None:
    normalization = NACANormalization(
        state_mean=np.zeros(5),
        state_scale=np.ones(5),
        residual_scale=np.ones(5),
        state_rms=np.ones(5),
    )
    train_indices = tuple(range(955, 1195))
    states = np.broadcast_to(np.zeros((1, 1, 5), dtype=np.float64), (240, 14576, 5))
    dataset = build_naca_bdf2_dataset(
        _contract(),
        verified_geometry,
        "train",
        states,
        train_indices,
        normalization,
    )
    assert len(dataset) == 238
    assert dataset.frame_indices == train_indices
    with pytest.raises(ValueError, match="exact 240 ordered frames"):
        build_naca_bdf2_dataset(
            _contract(),
            verified_geometry,
            "train",
            states,
            tuple(range(956, 1196)),
            normalization,
        )
    with pytest.raises(PermissionError, match="remains sealed"):
        build_naca_bdf2_dataset(
            _contract(),
            verified_geometry,
            "prospective",
            states,
            train_indices,
            normalization,
        )


def test_zero_head_is_exact_persistence_and_config_is_strict() -> None:
    contract = _contract()
    geometry = _build_naca_geometry_from_mesh(_synthetic_mesh())
    normalization = NACANormalization(
        state_mean=np.arange(5, dtype=np.float64),
        state_scale=np.arange(1, 6, dtype=np.float64),
        residual_scale=np.linspace(0.1, 0.5, 5),
        state_rms=np.arange(1, 6, dtype=np.float64),
    )
    torch.manual_seed(3)
    model = NACAPCNOResidual(
        fourier_lengths=geometry.fourier_lengths,
        zero_initialize=True,
    )
    batch = geometry.expand(2, "cpu")
    previous = torch.randn(2, geometry.num_nodes, 5)
    current = torch.randn(2, geometry.num_nodes, 5)
    features = build_input(previous, current, batch, normalization)
    assert features.shape == (2, geometry.num_nodes, 16)
    fourier = model.prepare_fourier_tensors(batch)
    residual = predict_normalized_residual(
        model,
        previous,
        current,
        batch,
        normalization,
        fourier_tensors=fourier,
    )
    assert torch.count_nonzero(residual).item() == 0
    shifted_previous, next_state = recurrent_step(
        model,
        previous,
        current,
        batch,
        normalization,
        fourier_tensors=fourier,
    )
    assert torch.equal(shifted_previous, current)
    assert torch.equal(next_state, current)

    assert model.model_config() == naca_core._expected_model_config(geometry)
    modified = model.model_config()
    modified["input_channels"] = 15
    with pytest.raises(TypeError, match="VerifiedNACAGeometry"):
        validate_naca_model_config(modified, contract, geometry)
    with pytest.raises(TypeError, match="VerifiedNACAGeometry"):
        build_naca_pcno(contract, geometry, zero_initialize=True)


def test_serialized_geometry_reloads_only_with_bound_hash_and_provenance(
    tmp_path: Path,
    verified_geometry: VerifiedNACAGeometry,
) -> None:
    path = tmp_path / "geometry.npz"
    np.savez(path, **verified_geometry.to_mapping())
    digest = sha256_file(path)
    restored = load_verified_naca_geometry(
        _contract(),
        path,
        expected_sha256=digest,
        expected_bytes=path.stat().st_size,
    )
    assert restored.num_nodes == verified_geometry.num_nodes
    with pytest.raises(ValueError, match="SHA256"):
        load_verified_naca_geometry(
            _contract(),
            path,
            expected_sha256="0" * 64,
            expected_bytes=path.stat().st_size,
        )


def test_stable_file_reader_rejects_mutation_during_parse(tmp_path: Path) -> None:
    path = tmp_path / "identity.bin"
    path.write_bytes(b"before")
    digest = sha256_file(path)

    def mutate(file_path: Path) -> bytes:
        content = file_path.read_bytes()
        file_path.write_bytes(b"after")
        return content

    with pytest.raises(ValueError, match="changed while it was being read"):
        _read_with_stable_file_identity(
            path,
            expected_bytes=6,
            expected_sha256=digest,
            label="test identity",
            reader=mutate,
        )


def test_directory_alias_is_rejected_before_resolution(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(ValueError, match="aliased"):
        _resolve_real_directory(alias, "test root")


def test_strict_builder_requires_zero_initialization(
    verified_geometry: VerifiedNACAGeometry,
) -> None:
    with pytest.raises(ValueError, match="zero output initialization"):
        build_naca_pcno(_contract(), verified_geometry, zero_initialize=False)
